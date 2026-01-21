"""
IOB Forcing Function Model

Predicts remaining IOB at a given horizon. The caller computes the BG pressure
by multiplying absorbed insulin by ISF.

This allows:
- ISF to be time-adjusted without retraining the model
- Clear interpretability: "1.5U absorbed" vs "-82 mg/dL"
- Independence from COB (they're separate forcing functions)

Key insight: IOB is always a NEGATIVE pressure on BG (insulin lowers BG).
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Default personalized half-life (learned from this child's BG data)
# This 7-year-old metabolizes insulin 30% faster than adult formula (81 min)
DEFAULT_HALF_LIFE_MIN = 54.0


class IOBForcingModel(nn.Module):
    """
    Neural network that predicts remaining IOB at a future horizon.

    Architecture:
    - Time branch: learns non-linear decay curve
    - Context branch: adjusts for circadian rhythm, meal state
    - Combined: outputs remaining fraction (0-1)

    Input features (7):
    - horizon_normalized: horizon_min / 360 (0-1)
    - current_iob_normalized: current_iob / 10 (0-1)
    - hour_sin, hour_cos: circadian rhythm
    - is_dawn_window: 4-8 AM flag
    - is_active_meal: recent carbs flag
    - personalized_half_life: from training

    Output:
    - remaining_fraction: (0-1) multiply by current_iob to get remaining IOB
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Time branch - learns the decay curve shape
        self.time_branch = nn.Sequential(
            nn.Linear(2, hidden_size),  # horizon + current_iob
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),  # Bounded output for stable decay
        )

        # Context branch - learns circadian and state adjustments
        self.context_branch = nn.Sequential(
            nn.Linear(5, hidden_size),  # hour_sin, hour_cos, dawn, meal, half_life
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined - merges decay with context
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Output: remaining fraction 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 7) with features

        Returns:
            remaining_fraction: Tensor of shape (batch, 1) in range [0, 1]
        """
        # Split features
        time_features = x[:, :2]  # horizon, current_iob
        context_features = x[:, 2:]  # hour_sin, hour_cos, dawn, meal, half_life

        # Process branches
        time_out = self.time_branch(time_features)
        context_out = self.context_branch(context_features)

        # Combine and predict remaining fraction
        combined = torch.cat([time_out, context_out], dim=1)
        remaining_fraction = self.combine(combined)

        return remaining_fraction


class IOBForcingService:
    """
    Service for predicting IOB-induced BG pressure.

    Uses the trained IOBForcingModel when available, falls back to
    exponential decay formula with personalized half-life.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    ):
        self.half_life_min = half_life_min
        self.model: Optional[IOBForcingModel] = None
        self._use_ml = False

        # Try to load trained model
        # Path: src/ml/models/iob_forcing.py -> ../../.. = src, ../../../.. = backend
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "iob_forcing.pth"

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                self.model = IOBForcingModel()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self._use_ml = True

                # Use trained half-life if available
                if 'half_life_min' in checkpoint:
                    self.half_life_min = checkpoint['half_life_min']

                logger.info(f"Loaded IOB forcing model (half_life={self.half_life_min:.1f} min)")
            except Exception as e:
                logger.warning(f"Failed to load IOB forcing model: {e}")
        else:
            logger.info(f"No IOB forcing model at {model_path}, using formula")

    def predict_remaining(
        self,
        current_iob: float,
        horizon_min: int,
        hour: Optional[int] = None,
        is_dawn_window: bool = False,
        is_active_meal: bool = False,
    ) -> float:
        """
        Predict remaining IOB at a future horizon.

        Args:
            current_iob: Current IOB in units
            horizon_min: Minutes into future (5-120)
            hour: Current hour (0-23), defaults to now
            is_dawn_window: True if 4-8 AM (dawn phenomenon)
            is_active_meal: True if recent carbs (may affect absorption)

        Returns:
            remaining_iob: IOB that will remain at horizon (units)
        """
        if current_iob <= 0:
            return 0.0

        if horizon_min <= 0:
            return current_iob

        if horizon_min > 360:
            return 0.0

        if hour is None:
            hour = datetime.utcnow().hour

        # Try ML model
        if self._use_ml and self.model is not None:
            try:
                features = self._prepare_features(
                    current_iob, horizon_min, hour,
                    is_dawn_window, is_active_meal
                )
                with torch.no_grad():
                    remaining_fraction = self.model(features).item()

                remaining_iob = current_iob * max(0.0, min(1.0, remaining_fraction))
                return round(remaining_iob, 3)
            except Exception as e:
                logger.warning(f"IOB ML prediction failed: {e}")

        # Fallback: exponential decay with personalized half-life
        remaining_fraction = 0.5 ** (horizon_min / self.half_life_min)
        return round(current_iob * remaining_fraction, 3)

    def predict_absorbed(
        self,
        current_iob: float,
        horizon_min: int,
        hour: Optional[int] = None,
        is_dawn_window: bool = False,
        is_active_meal: bool = False,
    ) -> float:
        """
        Predict how much insulin will be absorbed by horizon.

        Args:
            current_iob: Current IOB in units
            horizon_min: Minutes into future (5-120)
            hour: Current hour
            is_dawn_window: Dawn phenomenon flag
            is_active_meal: Active meal flag

        Returns:
            absorbed_insulin: Units of insulin absorbed (always positive)
        """
        remaining = self.predict_remaining(
            current_iob, horizon_min, hour, is_dawn_window, is_active_meal
        )
        return round(current_iob - remaining, 3)

    def get_bg_pressure(
        self,
        current_iob: float,
        horizon_min: int,
        isf: float,
        isf_adjustment: float = 1.0,
        hour: Optional[int] = None,
        is_dawn_window: bool = False,
        is_active_meal: bool = False,
    ) -> float:
        """
        Calculate the BG-lowering pressure from IOB at horizon.

        Args:
            current_iob: Current IOB in units
            horizon_min: Minutes into future
            isf: Insulin sensitivity factor (mg/dL per unit)
            isf_adjustment: Time-of-day multiplier (e.g., 1.2 for dawn)
            hour: Current hour
            is_dawn_window: Dawn phenomenon flag
            is_active_meal: Active meal flag

        Returns:
            bg_pressure: Expected BG change (NEGATIVE - insulin lowers BG)
        """
        absorbed = self.predict_absorbed(
            current_iob, horizon_min, hour, is_dawn_window, is_active_meal
        )

        # BG pressure = absorbed insulin * ISF * adjustment
        # Negative because insulin LOWERS BG
        bg_pressure = -absorbed * isf * isf_adjustment

        return round(bg_pressure, 1)

    def _prepare_features(
        self,
        current_iob: float,
        horizon_min: int,
        hour: int,
        is_dawn_window: bool,
        is_active_meal: bool,
    ) -> torch.Tensor:
        """Prepare feature tensor for model input."""
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        features = torch.tensor([[
            horizon_min / 360.0,  # Normalize horizon
            current_iob / 10.0,  # Normalize IOB
            hour_sin,
            hour_cos,
            1.0 if is_dawn_window else 0.0,
            1.0 if is_active_meal else 0.0,
            self.half_life_min / 100.0,  # Normalized half-life
        ]], dtype=torch.float32)

        return features

    def get_decay_curve(
        self,
        current_iob: float = 5.0,
        duration_min: int = 240,
        step_min: int = 15,
        hour: int = 12,
    ) -> list:
        """
        Generate IOB decay curve for visualization.

        Returns list of {minutes, remaining_iob, absorbed, remaining_fraction}
        """
        curve = []
        for t in range(0, duration_min + 1, step_min):
            remaining = self.predict_remaining(current_iob, t, hour)
            absorbed = current_iob - remaining

            curve.append({
                'minutes': t,
                'remaining_iob': round(remaining, 2),
                'absorbed_iob': round(absorbed, 2),
                'remaining_fraction': round(remaining / current_iob, 3) if current_iob > 0 else 0,
            })

        return curve


# Singleton instance
_iob_forcing_service: Optional[IOBForcingService] = None


def get_iob_forcing_service(model_path: Optional[Path] = None) -> IOBForcingService:
    """Get or create the IOB forcing service singleton."""
    global _iob_forcing_service
    if _iob_forcing_service is None:
        _iob_forcing_service = IOBForcingService(model_path)
    return _iob_forcing_service
