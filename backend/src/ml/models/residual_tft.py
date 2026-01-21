"""
Residual TFT Model

Learns adjustments to the physics baseline from secondary factors that
the physics layer doesn't capture:
- Time of day patterns beyond ISF (activity, meals at specific times)
- Day of week effects (weekday vs weekend schedules)
- Seasonal variations
- Lunar cycle (some individuals show sensitivity)
- Recent BG trend momentum
- Dawn phenomenon intensity

Output is clamped to +/- 25 mg/dL to ensure physics dominates.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum residual adjustment (mg/dL)
MAX_RESIDUAL = 25.0


def get_lunar_phase(dt: datetime) -> tuple:
    """
    Calculate lunar phase as sin/cos for cyclical feature.

    The lunar cycle affects fluid retention, hormones, and potentially
    insulin sensitivity in some individuals.

    Args:
        dt: datetime to calculate phase for

    Returns:
        (lunar_sin, lunar_cos): Cyclical encoding of lunar phase
    """
    lunar_cycle_days = 29.530588853  # Average synodic month

    # Reference: Jan 6, 2000 was a new moon
    reference = datetime(2000, 1, 6, 18, 14)

    # Handle timezone-aware datetimes
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to naive by removing timezone
        dt = dt.replace(tzinfo=None)

    days_since_ref = (dt - reference).total_seconds() / (24 * 3600)

    # Phase as fraction of cycle (0 = new moon, 0.5 = full moon)
    phase = (days_since_ref % lunar_cycle_days) / lunar_cycle_days

    # Convert to cyclical encoding
    lunar_sin = np.sin(2 * np.pi * phase)
    lunar_cos = np.cos(2 * np.pi * phase)

    return lunar_sin, lunar_cos


@dataclass
class SecondaryFeatures:
    """Container for secondary features used by residual model."""
    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    day_of_year: int  # 1-366
    lunar_sin: float
    lunar_cos: float
    is_weekend: bool
    recent_trend: float  # BG change per 5 min
    bg_volatility: float  # Std dev of recent BG
    dawn_intensity: float  # 0-1, how strong dawn phenomenon is today


class ResidualModel(nn.Module):
    """
    Neural network that learns adjustments to physics baseline.

    Input features (14):
    - hour_sin, hour_cos: Circadian pattern
    - dow_sin, dow_cos: Day of week pattern
    - doy_sin, doy_cos: Seasonal pattern
    - lunar_sin, lunar_cos: Lunar cycle
    - is_weekend: Weekend flag
    - recent_trend: BG momentum
    - bg_volatility: Recent variability
    - dawn_intensity: Dawn phenomenon strength
    - horizon_normalized: Prediction horizon
    - physics_pred_normalized: Physics baseline prediction

    Output:
    - residual: Adjustment in mg/dL (clamped to +/- 25)
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Temporal pattern branch
        self.temporal_branch = nn.Sequential(
            nn.Linear(8, hidden_size),  # hour, dow, doy, lunar
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Context branch
        self.context_branch = nn.Sequential(
            nn.Linear(6, hidden_size),  # weekend, trend, volatility, dawn, horizon, physics_pred
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),  # Output in [-1, 1], scaled to [-25, 25]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 14)

        Returns:
            residual: Tensor of shape (batch, 1) in [-MAX_RESIDUAL, MAX_RESIDUAL]
        """
        # Split features
        temporal = x[:, :8]  # hour, dow, doy, lunar
        context = x[:, 8:]   # weekend, trend, volatility, dawn, horizon, physics_pred

        # Process branches
        temporal_out = self.temporal_branch(temporal)
        context_out = self.context_branch(context)

        # Combine
        combined = torch.cat([temporal_out, context_out], dim=1)
        residual = self.combine(combined)

        # Scale from [-1, 1] to [-MAX_RESIDUAL, MAX_RESIDUAL]
        residual = residual * MAX_RESIDUAL

        return residual


class ResidualService:
    """
    Service for computing residual adjustments to physics predictions.

    Uses the trained ResidualModel when available, falls back to
    simple heuristics (dawn adjustment only).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[ResidualModel] = None
        self._use_ml = False

        # Try to load trained model
        # Path: src/ml/models/residual_tft.py -> ../../../.. = backend
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "residual_tft.pth"

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                self.model = ResidualModel()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self._use_ml = True
                logger.info("Loaded residual TFT model")
            except Exception as e:
                logger.warning(f"Failed to load residual model: {e}")
        else:
            logger.info(f"No residual model at {model_path}, using heuristics")

    def extract_features(
        self,
        dt: Optional[datetime] = None,
        recent_trend: float = 0,
        bg_volatility: float = 0,
        dawn_intensity: float = 0.5,
        horizon_min: int = 30,
        physics_pred: float = 120,
    ) -> SecondaryFeatures:
        """
        Extract secondary features from current context.

        Args:
            dt: Current datetime (defaults to now)
            recent_trend: BG change per 5 min
            bg_volatility: Std dev of recent BG readings
            dawn_intensity: How strong dawn phenomenon is (0-1)
            horizon_min: Prediction horizon
            physics_pred: Physics baseline prediction

        Returns:
            SecondaryFeatures dataclass
        """
        if dt is None:
            dt = datetime.utcnow()

        lunar_sin, lunar_cos = get_lunar_phase(dt)

        return SecondaryFeatures(
            hour=dt.hour,
            day_of_week=dt.weekday(),
            day_of_year=dt.timetuple().tm_yday,
            lunar_sin=lunar_sin,
            lunar_cos=lunar_cos,
            is_weekend=dt.weekday() >= 5,
            recent_trend=recent_trend,
            bg_volatility=bg_volatility,
            dawn_intensity=dawn_intensity,
        )

    def predict(
        self,
        features: SecondaryFeatures,
        horizon_min: int,
        physics_pred: float,
    ) -> float:
        """
        Predict residual adjustment.

        Args:
            features: Secondary features
            horizon_min: Prediction horizon
            physics_pred: Physics baseline prediction

        Returns:
            residual: Adjustment in mg/dL (clamped to +/- 25)
        """
        # Try ML model
        if self._use_ml and self.model is not None:
            try:
                feature_tensor = self._prepare_features(features, horizon_min, physics_pred)
                with torch.no_grad():
                    residual = self.model(feature_tensor).item()
                return round(max(-MAX_RESIDUAL, min(MAX_RESIDUAL, residual)), 1)
            except Exception as e:
                logger.warning(f"Residual ML prediction failed: {e}")

        # Fallback: simple heuristics
        return self._heuristic_residual(features, horizon_min, physics_pred)

    def _prepare_features(
        self,
        features: SecondaryFeatures,
        horizon_min: int,
        physics_pred: float,
    ) -> torch.Tensor:
        """Prepare feature tensor for model input."""
        hour_sin = np.sin(2 * np.pi * features.hour / 24)
        hour_cos = np.cos(2 * np.pi * features.hour / 24)
        dow_sin = np.sin(2 * np.pi * features.day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * features.day_of_week / 7)
        doy_sin = np.sin(2 * np.pi * features.day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * features.day_of_year / 365)

        tensor = torch.tensor([[
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            doy_sin,
            doy_cos,
            features.lunar_sin,
            features.lunar_cos,
            1.0 if features.is_weekend else 0.0,
            features.recent_trend / 10.0,  # Normalize trend
            features.bg_volatility / 50.0,  # Normalize volatility
            features.dawn_intensity,
            horizon_min / 120.0,  # Normalize horizon
            physics_pred / 200.0,  # Normalize physics pred
        ]], dtype=torch.float32)

        return tensor

    def _heuristic_residual(
        self,
        features: SecondaryFeatures,
        horizon_min: int,
        physics_pred: float,
    ) -> float:
        """
        Compute residual using simple heuristics when model unavailable.

        Primary heuristic: Dawn phenomenon adjustment.
        """
        residual = 0.0

        # Dawn phenomenon (4-8 AM)
        # If physics didn't fully capture it, add extra adjustment
        if 4 <= features.hour < 8:
            dawn_factor = features.dawn_intensity
            # Extra rise during dawn (max +15 at peak intensity)
            residual += 15.0 * dawn_factor * (1 - (features.hour - 6) ** 2 / 4)

        # Weekend vs weekday (different activity patterns)
        if features.is_weekend:
            # Weekends often have different meal timing, activity
            # Small adjustment for later meals
            if 10 <= features.hour < 14:
                residual -= 5.0  # Later breakfast, lower morning BG
            elif 18 <= features.hour < 22:
                residual += 5.0  # Bigger dinners on weekends

        # Trend momentum (if BG is changing fast, physics may lag)
        if abs(features.recent_trend) > 3:
            # Add small momentum adjustment
            residual += features.recent_trend * 0.5

        # Clamp to max residual
        return round(max(-MAX_RESIDUAL, min(MAX_RESIDUAL, residual)), 1)


# Singleton instance
_residual_service: Optional[ResidualService] = None


def get_residual_service(model_path: Optional[Path] = None) -> ResidualService:
    """Get or create the residual service singleton."""
    global _residual_service
    if _residual_service is None:
        _residual_service = ResidualService(model_path)
    return _residual_service
