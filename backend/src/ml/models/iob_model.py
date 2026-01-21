"""
Personalized IOB (Insulin On Board) ML Model

Learns individual insulin absorption curves from BG response data,
replacing fixed exponential decay formulas with personalized predictions.

Key insight: Insulin absorption varies by:
- Individual physiology
- Time of day (circadian rhythm)
- Activity level
- Injection site characteristics
- Current blood glucose (affects absorption rate)
- Season/lunar cycle (hormonal variations)
- Recent BG trend (metabolic state)

This model learns YOUR absorption curve from YOUR data,
considering all relevant physiological factors.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


def get_lunar_phase(dt: datetime) -> Tuple[float, float]:
    """Calculate lunar phase as sin/cos for cyclical feature."""
    ref_date = datetime(2000, 1, 6, 18, 14)  # New moon reference
    lunar_cycle_days = 29.530588853
    days_since_ref = (dt - ref_date).total_seconds() / 86400
    phase = (days_since_ref % lunar_cycle_days) / lunar_cycle_days
    theta = 2 * math.pi * phase
    return math.sin(theta), math.cos(theta)


class PersonalizedIOBModel(nn.Module):
    """
    MLP model that learns personalized insulin absorption curves.

    Architecture: Deeper MLP with residual connections
    - Input: 24 features capturing all relevant physiological factors
    - Output: Fraction of insulin remaining (0-1)

    Extended features (24 total):
    - Bolus features: units, type (correction vs meal)
    - Time features: minutes_since, hour sin/cos, dow sin/cos, month sin/cos
    - Lunar features: lunar_sin, lunar_cos (hormonal effects)
    - Metabolic state: current_bg, trend, rate_of_change
    - Activity: activity_level, is_sleeping
    - Context: is_fasting, recent_variability
    - Seasonal: day_of_year sin/cos

    The output is multiplied by the original bolus to get actual IOB.
    """

    def __init__(
        self,
        input_size: int = 24,  # Extended feature set
        hidden_sizes: List[int] = [64, 32, 16],  # Deeper network
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer: sigmoid for 0-1 fraction
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])

        self.network = nn.Sequential(*layers)

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 24) with extended features

        Returns:
            Remaining IOB fraction (0-1) of shape (batch, 1)
        """
        return self.network(x)

    def predict_iob(
        self,
        bolus_units: float,
        minutes_since_bolus: float,
        hour: int = 12,
        activity_level: float = 0.0,
        # Extended features
        current_bg: float = 120.0,
        trend: int = 0,
        rate_of_change: float = 0.0,
        day_of_week: int = 3,
        month: int = 6,
        day_of_year: int = 180,
        is_correction: bool = False,
        is_fasting: bool = False,
        is_sleeping: bool = False,
        recent_variability: float = 20.0,
    ) -> float:
        """
        Predict remaining IOB for a single bolus with extended features.

        Args:
            bolus_units: Original insulin dose
            minutes_since_bolus: Time elapsed since injection
            hour: Hour of day (0-23)
            activity_level: Activity level (0=rest, 1=light, 2=moderate, 3=intense)
            current_bg: Current blood glucose (affects absorption rate)
            trend: CGM trend (-3 to +3)
            rate_of_change: BG rate of change (mg/dL per 5 min)
            day_of_week: Day of week (0-6)
            month: Month (1-12)
            day_of_year: Day of year (1-365)
            is_correction: Whether this is a correction bolus
            is_fasting: Whether currently fasting
            is_sleeping: Whether currently sleeping
            recent_variability: Recent BG variability (std dev)

        Returns:
            Predicted remaining IOB in units
        """
        if bolus_units <= 0 or minutes_since_bolus < 0:
            return 0.0

        # Build extended feature vector (24 features)
        features = self._build_features(
            bolus_units, minutes_since_bolus, hour, activity_level,
            current_bg, trend, rate_of_change, day_of_week, month,
            day_of_year, is_correction, is_fasting, is_sleeping,
            recent_variability
        )

        x = torch.tensor([features], dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            remaining_fraction = self.forward(x).item()

        return bolus_units * remaining_fraction

    def _build_features(
        self,
        bolus_units: float,
        minutes_since_bolus: float,
        hour: int,
        activity_level: float,
        current_bg: float,
        trend: int,
        rate_of_change: float,
        day_of_week: int,
        month: int,
        day_of_year: int,
        is_correction: bool,
        is_fasting: bool,
        is_sleeping: bool,
        recent_variability: float,
    ) -> List[float]:
        """Build the 24-feature vector."""
        features = []

        # Bolus features (2)
        features.append(bolus_units / 10.0)  # Scaled
        features.append(1.0 if is_correction else 0.0)

        # Time since bolus (1)
        features.append(minutes_since_bolus / 360.0)

        # Hour cyclical (2)
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))

        # Day of week cyclical (2)
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))

        # Month cyclical (2)
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))

        # Day of year cyclical (2) - captures seasonal patterns
        features.append(np.sin(2 * np.pi * day_of_year / 365))
        features.append(np.cos(2 * np.pi * day_of_year / 365))

        # Lunar phase (2) - affects hormones/fluid retention
        dt = datetime(2024, month, max(1, min(28, int(day_of_year % 28 + 1))), hour)
        lunar_sin, lunar_cos = get_lunar_phase(dt)
        features.append(lunar_sin)
        features.append(lunar_cos)

        # Metabolic state (4)
        features.append(current_bg / 200.0)  # Scaled BG
        features.append(trend / 3.0)  # Normalized trend
        features.append(rate_of_change / 10.0)  # Scaled rate
        features.append(recent_variability / 100.0)  # Scaled variability

        # Activity and state (4)
        features.append(activity_level / 3.0)
        features.append(1.0 if is_sleeping else 0.0)
        features.append(1.0 if is_fasting else 0.0)

        # BG-dependent absorption rate adjustment
        # Higher BG = slightly slower absorption, lower BG = slightly faster
        bg_absorption_factor = 1.0 - (current_bg - 120) / 400.0
        features.append(np.clip(bg_absorption_factor, 0.5, 1.5))

        # Activity-dependent adjustment (exercise speeds absorption)
        activity_factor = 1.0 + activity_level * 0.1
        features.append(activity_factor)

        return features


class IOBModelService:
    """
    Service for calculating IOB using the ML model with exponential decay fallback.

    Provides:
    - Personalized IOB predictions when model is trained
    - Fallback to exponential decay when insufficient data
    - Training data collection from BG responses
    """

    def __init__(
        self,
        model: Optional[PersonalizedIOBModel] = None,
        fallback_half_life_min: float = 81.0,
        fallback_duration_min: int = 360
    ):
        """
        Initialize IOB service.

        Args:
            model: Trained PersonalizedIOBModel (None for fallback only)
            fallback_half_life_min: Half-life for exponential decay fallback
            fallback_duration_min: Maximum insulin action duration
        """
        self.model = model
        self.fallback_half_life_min = fallback_half_life_min
        self.fallback_duration_min = fallback_duration_min
        self._use_ml = model is not None

    def calculate_iob(
        self,
        bolus_units: float,
        minutes_since_bolus: float,
        hour: int = 12,
        activity_level: float = 0.0
    ) -> float:
        """
        Calculate IOB using ML model or fallback.

        Args:
            bolus_units: Original insulin dose
            minutes_since_bolus: Time elapsed since injection
            hour: Hour of day (0-23)
            activity_level: Activity level (0=rest, 1=light, 2=moderate, 3=intense)

        Returns:
            Remaining IOB in units
        """
        if bolus_units <= 0:
            return 0.0

        if minutes_since_bolus < 0:
            return bolus_units

        if minutes_since_bolus > self.fallback_duration_min:
            return 0.0

        if self._use_ml and self.model is not None:
            try:
                return self.model.predict_iob(
                    bolus_units,
                    minutes_since_bolus,
                    hour,
                    activity_level
                )
            except Exception as e:
                logger.warning(f"ML IOB prediction failed, using fallback: {e}")

        # Fallback: exponential decay
        decay_factor = 0.5 ** (minutes_since_bolus / self.fallback_half_life_min)
        return bolus_units * decay_factor

    def calculate_total_iob(
        self,
        insulin_events: List[Dict],
        at_time: Optional[datetime] = None,
        activity_level: float = 0.0
    ) -> float:
        """
        Calculate total IOB from multiple insulin events.

        Args:
            insulin_events: List of dicts with 'timestamp' and 'insulin' keys
            at_time: Time to calculate IOB for (default: now)
            activity_level: Current activity level

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

            if 0 <= minutes_elapsed <= self.fallback_duration_min:
                iob = self.calculate_iob(
                    insulin,
                    minutes_elapsed,
                    hour,
                    activity_level
                )
                total_iob += iob

        return round(total_iob, 2)

    def get_iob_curve(
        self,
        bolus_units: float,
        hour: int = 12,
        activity_level: float = 0.0,
        duration_min: int = 360,
        step_min: int = 5
    ) -> List[Dict]:
        """
        Generate the IOB decay curve for visualization.

        Args:
            bolus_units: Insulin dose to model
            hour: Hour of day
            activity_level: Activity level
            duration_min: Total curve duration
            step_min: Time step in minutes

        Returns:
            List of dicts with 'minutes' and 'iob' keys
        """
        curve = []
        for t in range(0, duration_min + 1, step_min):
            iob = self.calculate_iob(bolus_units, t, hour, activity_level)
            curve.append({
                'minutes': t,
                'iob': round(iob, 3),
                'fraction': round(iob / bolus_units, 3) if bolus_units > 0 else 0
            })
        return curve

    def calculate_bg_effect_curve(
        self,
        current_iob: float,
        isf: float = 50.0,
        duration_min: int = 60,
        step_min: int = 5
    ) -> List[Dict]:
        """
        Calculate projected IOB effect on BG over time.

        Args:
            current_iob: Current IOB in units
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            duration_min: Projection duration
            step_min: Time step

        Returns:
            List of dicts with time and BG effect
        """
        effects = []
        for t in range(0, duration_min + 1, step_min):
            # Estimate remaining IOB at time t using exponential decay
            remaining_iob = current_iob * (0.5 ** (t / self.fallback_half_life_min))

            # BG lowering effect (negative = lowering BG)
            iob_effect = -(remaining_iob * isf)

            effects.append({
                'minutesAhead': t,
                'remainingIOB': round(remaining_iob, 2),
                'bgEffect': round(iob_effect, 1)
            })

        return effects


# Model configuration constants
IOB_MODEL_CONFIG = {
    "input_size": 24,  # Extended feature set
    "hidden_sizes": [64, 32, 16],  # Deeper network
    "dropout": 0.1,
    "max_duration_min": 360,  # 6 hours
    "min_training_samples": 50,  # Minimum bolus events needed to train
    "fallback_half_life_min": 81.0,  # Novolog/Humalog half-life
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
}

# Extended input feature names (24 total)
IOB_INPUT_FEATURES = [
    # Bolus features (2)
    "bolus_units_scaled",       # Insulin amount (scaled by /10)
    "is_correction",            # 1 if correction bolus, 0 if meal bolus
    # Time features (1)
    "minutes_since_scaled",     # Time elapsed (scaled by /360)
    # Hour cyclical (2)
    "hour_sin",                 # Cyclical hour (sin)
    "hour_cos",                 # Cyclical hour (cos)
    # Day of week cyclical (2)
    "dow_sin",
    "dow_cos",
    # Month cyclical (2)
    "month_sin",
    "month_cos",
    # Day of year cyclical (2)
    "doy_sin",
    "doy_cos",
    # Lunar phase (2)
    "lunar_sin",
    "lunar_cos",
    # Metabolic state (4)
    "current_bg_scaled",        # BG / 200
    "trend_normalized",         # trend / 3
    "rate_of_change_scaled",    # rate / 10
    "variability_scaled",       # std_dev / 100
    # Activity and state (4)
    "activity_level_scaled",    # Activity (scaled by /3)
    "is_sleeping",
    "is_fasting",
    "bg_absorption_factor",     # BG-dependent absorption adjustment
    # Derived (1)
    "activity_absorption_factor",
]


def create_iob_training_sample(
    bolus_units: float,
    minutes_since_bolus: float,
    remaining_fraction: float,
    hour: int = 12,
    activity_level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single training sample for the IOB model.

    Args:
        bolus_units: Original insulin dose
        minutes_since_bolus: Time elapsed
        remaining_fraction: Observed remaining IOB fraction (0-1)
        hour: Hour of day
        activity_level: Activity level

    Returns:
        Tuple of (features, target) as numpy arrays
    """
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    features = np.array([
        bolus_units / 10.0,
        minutes_since_bolus / 360.0,
        hour_sin,
        hour_cos,
        activity_level / 3.0
    ], dtype=np.float32)

    target = np.array([remaining_fraction], dtype=np.float32)

    return features, target


def estimate_remaining_iob_from_bg(
    bg_before: float,
    bg_after: float,
    insulin_units: float,
    expected_isf: float
) -> float:
    """
    Estimate remaining IOB fraction from BG response.

    This is used to create training labels from actual BG data.

    Args:
        bg_before: BG at time of bolus
        bg_after: BG at measurement time
        insulin_units: Insulin given
        expected_isf: Expected insulin sensitivity factor

    Returns:
        Estimated fraction of insulin remaining (0-1)
    """
    if insulin_units <= 0 or expected_isf <= 0:
        return 1.0

    # Calculate expected total BG drop if all insulin acted
    expected_drop = insulin_units * expected_isf

    # Calculate actual BG change
    actual_drop = bg_before - bg_after

    # Estimate absorbed insulin
    if expected_drop > 0:
        absorbed_fraction = actual_drop / expected_drop
        absorbed_fraction = max(0, min(1, absorbed_fraction))
        remaining_fraction = 1.0 - absorbed_fraction
    else:
        remaining_fraction = 1.0

    return remaining_fraction
