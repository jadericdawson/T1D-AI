"""
Personalized IOB Service

Uses ML model trained on actual BG drop sequences to predict
insulin absorption for THIS specific person.

Key finding: This 7-year-old has a ~51-57 minute half-life vs
the standard 81 minutes used in adult formulas (30% faster!).

This service provides:
- Personalized IOB decay predictions
- Fallback to learned half-life when model unavailable
- Integration with prediction service
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Learned parameters for this child
PERSONALIZED_HALF_LIFE_MIN = 54.0  # Learned from BG data (vs 81 for adults)
STANDARD_HALF_LIFE_MIN = 81.0


class IOBDecayNet(nn.Module):
    """The trained IOB model architecture."""

    def __init__(self):
        super().__init__()
        self.time_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        time = x[:, 0:1]
        context = x[:, 1:]
        time_features = self.time_branch(time)
        context_features = self.context_branch(context)
        combined = torch.cat([time_features, context_features], dim=1)
        return self.combine(combined)


class PersonalizedIOBService:
    """
    Service for calculating IOB using personalized ML model.

    Falls back to personalized half-life formula if model unavailable.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        default_isf: float = 55.0,
    ):
        self.default_isf = default_isf
        self.model: Optional[IOBDecayNet] = None
        self._use_ml = False

        # Try to load ML model
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "iob_from_bg_drops.pth"

        if model_path.exists():
            try:
                self.model = IOBDecayNet()
                self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                self.model.eval()
                self._use_ml = True
                logger.info(f"Loaded personalized IOB model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load IOB model: {e}, using formula fallback")
        else:
            logger.info(f"No IOB model at {model_path}, using personalized half-life formula")

    def get_remaining_fraction(
        self,
        minutes_since_bolus: float,
        bolus_units: float = 2.0,
        hour: int = 12,
        isf: float = 55.0,
        start_bg: float = 150.0,
    ) -> float:
        """
        Get the fraction of insulin remaining at a given time.

        Args:
            minutes_since_bolus: Time since injection
            bolus_units: Insulin dose
            hour: Hour of day
            isf: Insulin sensitivity factor
            start_bg: BG at time of bolus

        Returns:
            Fraction of insulin remaining (0-1)
        """
        if minutes_since_bolus <= 0:
            return 1.0
        if minutes_since_bolus > 360:
            return 0.0

        if self._use_ml and self.model is not None:
            try:
                features = torch.tensor([[
                    minutes_since_bolus / 360.0,
                    bolus_units / 10.0,
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    isf / 100.0,
                    start_bg / 200.0,
                ]], dtype=torch.float32)

                with torch.no_grad():
                    fraction = self.model(features).item()
                return max(0.0, min(1.0, fraction))
            except Exception as e:
                logger.warning(f"ML IOB prediction failed: {e}, using formula")

        # Fallback: use PERSONALIZED half-life (learned from data)
        return 0.5 ** (minutes_since_bolus / PERSONALIZED_HALF_LIFE_MIN)

    def calculate_iob(
        self,
        insulin_events: List[Dict],
        at_time: Optional[datetime] = None,
        isf: float = 55.0,
    ) -> float:
        """
        Calculate total IOB from multiple insulin events.

        Args:
            insulin_events: List of dicts with 'timestamp' and 'insulin' keys
            at_time: Time to calculate IOB for (default: now)
            isf: Insulin sensitivity factor

        Returns:
            Total IOB in units
        """
        if not insulin_events:
            return 0.0

        at_time = at_time or datetime.utcnow()
        hour = at_time.hour
        total_iob = 0.0

        for event in insulin_events:
            insulin = event.get('insulin', 0)
            if not insulin or insulin <= 0:
                continue

            timestamp = event.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            minutes_elapsed = (at_time - timestamp).total_seconds() / 60

            if 0 <= minutes_elapsed <= 360:
                fraction = self.get_remaining_fraction(
                    minutes_elapsed,
                    bolus_units=insulin,
                    hour=hour,
                    isf=isf,
                )
                total_iob += insulin * fraction

        return round(total_iob, 2)

    def get_iob_at_horizon(
        self,
        current_iob: float,
        horizon_min: int,
        hour: int = 12,
        isf: float = 55.0,
    ) -> float:
        """
        Project IOB remaining at a future horizon.

        Args:
            current_iob: Current IOB
            horizon_min: Minutes into future
            hour: Current hour
            isf: ISF

        Returns:
            Projected IOB at horizon
        """
        if current_iob <= 0:
            return 0.0

        # Use personalized decay
        fraction = self.get_remaining_fraction(
            horizon_min,
            bolus_units=current_iob,  # Treat as single bolus for projection
            hour=hour,
            isf=isf,
        )

        return current_iob * fraction

    def get_insulin_effect_at_horizon(
        self,
        current_iob: float,
        horizon_min: int,
        isf: float = 55.0,
        hour: int = 12,
    ) -> float:
        """
        Calculate the BG-lowering effect of insulin from now to horizon.

        This is the key integration point for BG prediction.

        Args:
            current_iob: Current IOB in units
            horizon_min: Prediction horizon in minutes
            isf: Insulin sensitivity factor
            hour: Current hour

        Returns:
            Expected BG drop in mg/dL (negative number)
        """
        if current_iob <= 0:
            return 0.0

        # IOB remaining at horizon
        iob_at_horizon = self.get_iob_at_horizon(current_iob, horizon_min, hour, isf)

        # Insulin absorbed = current_iob - iob_at_horizon
        insulin_absorbed = current_iob - iob_at_horizon

        # BG effect = absorbed insulin * ISF
        bg_effect = -insulin_absorbed * isf  # Negative = lowering BG

        return bg_effect

    def get_decay_curve(
        self,
        bolus_units: float = 2.0,
        duration_min: int = 360,
        step_min: int = 15,
        hour: int = 12,
    ) -> List[Dict]:
        """
        Generate IOB decay curve for visualization.

        Returns list of {minutes, iob, fraction, formula_fraction}
        """
        curve = []
        for t in range(0, duration_min + 1, step_min):
            fraction = self.get_remaining_fraction(t, bolus_units, hour)
            formula_fraction = 0.5 ** (t / STANDARD_HALF_LIFE_MIN)

            curve.append({
                'minutes': t,
                'iob': round(bolus_units * fraction, 3),
                'fraction': round(fraction, 3),
                'formula_fraction': round(formula_fraction, 3),
            })
        return curve


# Singleton instance
_personalized_iob_service: Optional[PersonalizedIOBService] = None


def get_personalized_iob_service() -> PersonalizedIOBService:
    """Get or create the personalized IOB service."""
    global _personalized_iob_service
    if _personalized_iob_service is None:
        _personalized_iob_service = PersonalizedIOBService()
    return _personalized_iob_service
