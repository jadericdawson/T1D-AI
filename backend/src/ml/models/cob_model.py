"""
Personalized COB (Carbs On Board) ML Model

Learns individual carb absorption curves from BG response data,
accounting for food composition that affects absorption rate.

Key insight: Carb absorption varies dramatically by:
- Food type (glycemic index)
- Fat content (delays absorption significantly)
- Protein content (slows absorption)
- Fiber content (slows absorption)
- Individual digestion patterns
- Time of day
- Current blood glucose level
- Recent activity level
- Lunar cycle (affects fluid retention/digestion)
- Seasonal patterns

Pizza pressure is VERY different than apple juice pressure!
This model learns YOUR absorption patterns for different food types.
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


class PersonalizedCOBModel(nn.Module):
    """
    MLP model that learns personalized carb absorption curves.

    Architecture: Deeper MLP with extended features
    - Input: 32 features capturing food composition + metabolic state
    - Output: Fraction of carbs remaining (0-1)

    Extended features (32 total):
    - Food features: carbs, protein, fat, fiber, glycemic_index, glycemic_load
    - Absorption rate: fast/medium/slow one-hot, is_high_fat
    - Time features: minutes_since, hour sin/cos, dow sin/cos, month sin/cos
    - Lunar features: lunar_sin, lunar_cos
    - Day of year: doy sin/cos (seasonal patterns)
    - Metabolic state: current_bg, trend, rate_of_change
    - Activity: activity_level, is_sleeping
    - Context: is_fasting_before, recent_variability
    - Derived: absorption_delay_factor

    The output is multiplied by original carbs to get actual COB.
    """

    def __init__(
        self,
        input_size: int = 32,  # Extended feature set
        hidden_sizes: List[int] = [128, 64, 32],  # Deeper network
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

        # Initialize weights
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
            x: Input tensor of shape (batch, 32) with extended features

        Returns:
            Remaining COB fraction (0-1) of shape (batch, 1)
        """
        return self.network(x)

    def predict_cob(
        self,
        carbs: float,
        minutes_since_meal: float,
        protein: float = 0.0,
        fat: float = 0.0,
        glycemic_index: int = 55,
        hour: int = 12,
        # Extended features
        fiber: float = 0.0,
        current_bg: float = 120.0,
        trend: int = 0,
        rate_of_change: float = 0.0,
        day_of_week: int = 3,
        month: int = 6,
        day_of_year: int = 180,
        activity_level: float = 0.0,
        is_sleeping: bool = False,
        is_fasting_before: bool = False,
        recent_variability: float = 20.0,
        absorption_rate: str = 'medium',
    ) -> float:
        """
        Predict remaining COB for a single meal with extended features.

        Args:
            carbs: Original carb amount in grams
            minutes_since_meal: Time elapsed since eating
            protein: Protein in grams
            fat: Fat in grams
            glycemic_index: Food's GI (0-100)
            hour: Hour of day (0-23)
            fiber: Fiber in grams
            current_bg: Current blood glucose
            trend: CGM trend (-3 to +3)
            rate_of_change: BG rate of change
            day_of_week: Day of week (0-6)
            month: Month (1-12)
            day_of_year: Day of year (1-365)
            activity_level: Activity level (0-3)
            is_sleeping: Whether sleeping
            is_fasting_before: Whether was fasting before this meal
            recent_variability: Recent BG variability
            absorption_rate: 'fast', 'medium', or 'slow'

        Returns:
            Predicted remaining COB in grams
        """
        if carbs <= 0 or minutes_since_meal < 0:
            return 0.0

        # Build extended feature vector (32 features)
        features = self._build_features(
            carbs, minutes_since_meal, protein, fat, glycemic_index, hour,
            fiber, current_bg, trend, rate_of_change, day_of_week, month,
            day_of_year, activity_level, is_sleeping, is_fasting_before,
            recent_variability, absorption_rate
        )

        x = torch.tensor([features], dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            remaining_fraction = self.forward(x).item()

        return carbs * remaining_fraction

    def _build_features(
        self,
        carbs: float,
        minutes_since_meal: float,
        protein: float,
        fat: float,
        glycemic_index: int,
        hour: int,
        fiber: float,
        current_bg: float,
        trend: int,
        rate_of_change: float,
        day_of_week: int,
        month: int,
        day_of_year: int,
        activity_level: float,
        is_sleeping: bool,
        is_fasting_before: bool,
        recent_variability: float,
        absorption_rate: str,
    ) -> List[float]:
        """Build the 32-feature vector."""
        features = []

        # Food features (6)
        features.append(carbs / 100.0)
        features.append(protein / 50.0)
        features.append(fat / 50.0)
        features.append(fiber / 20.0)
        features.append(glycemic_index / 100.0)
        glycemic_load = (carbs * glycemic_index / 100) / 50.0  # Scaled GL
        features.append(glycemic_load)

        # Absorption rate one-hot (3)
        features.append(1.0 if absorption_rate == 'fast' else 0.0)
        features.append(1.0 if absorption_rate == 'medium' else 0.0)
        features.append(1.0 if absorption_rate == 'slow' else 0.0)

        # High fat indicator (1)
        features.append(1.0 if fat > 15 else 0.0)

        # Time since meal (1)
        features.append(minutes_since_meal / 480.0)  # Max 8 hours for high-fat

        # Hour cyclical (2)
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))

        # Day of week cyclical (2)
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))

        # Month cyclical (2)
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))

        # Day of year cyclical (2) - seasonal patterns
        features.append(np.sin(2 * np.pi * day_of_year / 365))
        features.append(np.cos(2 * np.pi * day_of_year / 365))

        # Lunar phase (2)
        dt = datetime(2024, month, max(1, min(28, int(day_of_year % 28 + 1))), hour)
        lunar_sin, lunar_cos = get_lunar_phase(dt)
        features.append(lunar_sin)
        features.append(lunar_cos)

        # Metabolic state (4)
        features.append(current_bg / 200.0)
        features.append(trend / 3.0)
        features.append(rate_of_change / 10.0)
        features.append(recent_variability / 100.0)

        # Activity and state (4)
        features.append(activity_level / 3.0)
        features.append(1.0 if is_sleeping else 0.0)
        features.append(1.0 if is_fasting_before else 0.0)

        # Absorption delay factor based on fat/protein/fiber
        # High fat/protein/fiber = delayed absorption
        fat_delay = min(1.0, fat / 30.0)  # Max at 30g fat
        protein_delay = min(1.0, protein / 40.0)  # Max at 40g protein
        fiber_delay = min(1.0, fiber / 15.0)  # Max at 15g fiber
        absorption_delay = (fat_delay * 0.5 + protein_delay * 0.3 + fiber_delay * 0.2)
        features.append(absorption_delay)

        return features


class COBModelService:
    """
    Service for calculating COB using the ML model with exponential decay fallback.

    Provides:
    - Personalized COB predictions when model is trained
    - Food-composition-aware absorption modeling
    - Fallback to exponential decay when insufficient data
    """

    def __init__(
        self,
        model: Optional[PersonalizedCOBModel] = None,
        fallback_half_life_min: float = 45.0,
        fallback_duration_min: int = 360
    ):
        """
        Initialize COB service.

        Args:
            model: Trained PersonalizedCOBModel (None for fallback only)
            fallback_half_life_min: Half-life for exponential decay fallback
            fallback_duration_min: Maximum carb absorption duration
        """
        self.model = model
        self.fallback_half_life_min = fallback_half_life_min
        self.fallback_duration_min = fallback_duration_min
        self._use_ml = model is not None

    def calculate_cob(
        self,
        carbs: float,
        minutes_since_meal: float,
        protein: float = 0.0,
        fat: float = 0.0,
        glycemic_index: int = 55,
        hour: int = 12
    ) -> float:
        """
        Calculate COB using ML model or fallback.

        Args:
            carbs: Original carb amount
            minutes_since_meal: Time elapsed since eating
            protein: Protein in grams
            fat: Fat in grams
            glycemic_index: Food's GI (0-100)
            hour: Hour of day

        Returns:
            Remaining COB in grams
        """
        if carbs <= 0:
            return 0.0

        if minutes_since_meal < 0:
            return carbs

        # Calculate adjusted duration based on fat content
        # High fat meals take much longer to absorb
        adjusted_duration = self._get_adjusted_duration(fat)

        if minutes_since_meal > adjusted_duration:
            return 0.0

        if self._use_ml and self.model is not None:
            try:
                return self.model.predict_cob(
                    carbs,
                    minutes_since_meal,
                    protein,
                    fat,
                    glycemic_index,
                    hour
                )
            except Exception as e:
                logger.warning(f"ML COB prediction failed, using fallback: {e}")

        # Fallback: exponential decay with food-adjusted half-life
        adjusted_half_life = self._get_adjusted_half_life(fat, glycemic_index)
        decay_factor = 0.5 ** (minutes_since_meal / adjusted_half_life)
        return carbs * decay_factor

    def _get_adjusted_half_life(self, fat: float, glycemic_index: int) -> float:
        """
        Adjust half-life based on food composition.

        Fat significantly slows absorption.
        Low GI foods absorb slower than high GI.
        """
        base_half_life = self.fallback_half_life_min

        # Fat adjustment: each 5g of fat adds ~10 min to half-life
        fat_adjustment = (fat / 5.0) * 10.0

        # GI adjustment: low GI (< 55) slower, high GI (> 70) faster
        gi_adjustment = 0.0
        if glycemic_index < 55:
            gi_adjustment = (55 - glycemic_index) * 0.3  # Slower
        elif glycemic_index > 70:
            gi_adjustment = -(glycemic_index - 70) * 0.4  # Faster

        adjusted = base_half_life + fat_adjustment + gi_adjustment
        return max(15.0, min(120.0, adjusted))  # Clamp to reasonable range

    def _get_adjusted_duration(self, fat: float) -> float:
        """
        Adjust maximum absorption duration based on fat content.

        Pizza with high fat can take 6+ hours to fully absorb.
        """
        base_duration = self.fallback_duration_min

        # High fat meals extend duration significantly
        if fat > 20:
            return min(480, base_duration + 120)  # Up to 8 hours
        elif fat > 10:
            return min(420, base_duration + 60)   # Up to 7 hours

        return base_duration

    def calculate_total_cob(
        self,
        carb_events: List[Dict],
        at_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate total COB from multiple carb events.

        Args:
            carb_events: List of dicts with 'timestamp', 'carbs', and optional
                        'protein', 'fat', 'glycemicIndex' keys
            at_time: Time to calculate COB for (default: now)

        Returns:
            Total COB in grams
        """
        if not carb_events:
            return 0.0

        at_time = at_time or datetime.utcnow()
        hour = at_time.hour
        total_cob = 0.0

        for event in carb_events:
            carbs = event.get('carbs', 0)
            if not carbs or carbs <= 0:
                continue

            timestamp = event.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            minutes_elapsed = (at_time - timestamp).total_seconds() / 60

            # Get food composition
            protein = event.get('protein', 0) or 0
            fat = event.get('fat', 0) or 0
            gi = event.get('glycemicIndex', 55) or 55

            # Calculate adjusted duration for this meal
            adjusted_duration = self._get_adjusted_duration(fat)

            if 0 <= minutes_elapsed <= adjusted_duration:
                cob = self.calculate_cob(
                    carbs,
                    minutes_elapsed,
                    protein,
                    fat,
                    gi,
                    hour
                )
                total_cob += cob

        return round(total_cob, 1)

    def get_cob_curve(
        self,
        carbs: float,
        protein: float = 0.0,
        fat: float = 0.0,
        glycemic_index: int = 55,
        hour: int = 12,
        duration_min: int = 360,
        step_min: int = 5
    ) -> List[Dict]:
        """
        Generate the COB decay curve for visualization.

        Args:
            carbs: Carb amount to model
            protein: Protein in grams
            fat: Fat in grams
            glycemic_index: Food's GI
            hour: Hour of day
            duration_min: Total curve duration
            step_min: Time step in minutes

        Returns:
            List of dicts with 'minutes' and 'cob' keys
        """
        curve = []
        for t in range(0, duration_min + 1, step_min):
            cob = self.calculate_cob(carbs, t, protein, fat, glycemic_index, hour)
            curve.append({
                'minutes': t,
                'cob': round(cob, 1),
                'fraction': round(cob / carbs, 3) if carbs > 0 else 0
            })
        return curve

    def calculate_bg_effect_curve(
        self,
        current_cob: float,
        icr: float = 10.0,
        isf: float = 50.0,
        duration_min: int = 60,
        step_min: int = 5,
        fat: float = 0.0,
        glycemic_index: int = 55
    ) -> List[Dict]:
        """
        Calculate projected COB effect on BG over time.

        Args:
            current_cob: Current COB in grams
            icr: Insulin to Carb Ratio (carbs per unit)
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            duration_min: Projection duration
            step_min: Time step
            fat: Fat content (affects absorption rate)
            glycemic_index: Food's GI

        Returns:
            List of dicts with time and BG effect
        """
        # BG rise factor: (carbs / ICR) * ISF = expected BG rise per gram
        bg_per_gram = isf / icr

        adjusted_half_life = self._get_adjusted_half_life(fat, glycemic_index)

        effects = []
        for t in range(0, duration_min + 1, step_min):
            # Estimate remaining COB at time t
            remaining_cob = current_cob * (0.5 ** (t / adjusted_half_life))

            # BG raising effect (positive = raising BG)
            cob_effect = remaining_cob * bg_per_gram

            effects.append({
                'minutesAhead': t,
                'remainingCOB': round(remaining_cob, 1),
                'bgEffect': round(cob_effect, 1)
            })

        return effects


# Absorption rate constants based on food type
ABSORPTION_PRESETS = {
    "fast": {
        "half_life_min": 25,
        "duration_min": 120,
        "gi_range": (70, 100),
        "examples": ["juice", "candy", "white bread", "glucose tabs"]
    },
    "medium": {
        "half_life_min": 45,
        "duration_min": 180,
        "gi_range": (55, 70),
        "examples": ["rice", "pasta", "banana", "cereal"]
    },
    "slow": {
        "half_life_min": 75,
        "duration_min": 360,
        "gi_range": (0, 55),
        "examples": ["pizza", "beans", "whole grain", "high-fat meals"]
    }
}

# Model configuration constants
COB_MODEL_CONFIG = {
    "input_size": 7,
    "hidden_sizes": [64, 32],
    "dropout": 0.1,
    "max_duration_min": 480,  # 8 hours for high-fat meals
    "min_training_samples": 30,  # Minimum meal events needed to train
    "fallback_half_life_min": 45.0,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
}

# Input feature names
COB_INPUT_FEATURES = [
    "carbs_scaled",             # Carbs (scaled by /100)
    "protein_scaled",           # Protein (scaled by /50)
    "fat_scaled",               # Fat (scaled by /50)
    "glycemic_index_scaled",    # GI (scaled by /100)
    "minutes_since_scaled",     # Time elapsed (scaled by /360)
    "hour_sin",                 # Cyclical hour (sin)
    "hour_cos",                 # Cyclical hour (cos)
]


def create_cob_training_sample(
    carbs: float,
    minutes_since_meal: float,
    remaining_fraction: float,
    protein: float = 0.0,
    fat: float = 0.0,
    glycemic_index: int = 55,
    hour: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single training sample for the COB model.

    Args:
        carbs: Original carb amount
        minutes_since_meal: Time elapsed
        remaining_fraction: Observed remaining COB fraction (0-1)
        protein: Protein in grams
        fat: Fat in grams
        glycemic_index: Food's GI
        hour: Hour of day

    Returns:
        Tuple of (features, target) as numpy arrays
    """
    # Handle NaN values - replace with defaults
    carbs = 0.0 if np.isnan(carbs) else float(carbs)
    minutes_since_meal = 0.0 if np.isnan(minutes_since_meal) else float(minutes_since_meal)
    remaining_fraction = 0.5 if np.isnan(remaining_fraction) else float(remaining_fraction)
    protein = 0.0 if np.isnan(protein) else float(protein)
    fat = 0.0 if np.isnan(fat) else float(fat)
    glycemic_index = 55 if np.isnan(glycemic_index) else int(glycemic_index)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    features = np.array([
        carbs / 100.0,
        protein / 50.0,
        fat / 50.0,
        glycemic_index / 100.0,
        minutes_since_meal / 360.0,
        hour_sin,
        hour_cos
    ], dtype=np.float32)

    target = np.array([remaining_fraction], dtype=np.float32)

    return features, target


def estimate_remaining_cob_from_bg(
    bg_before: float,
    bg_after: float,
    carbs: float,
    insulin_effect: float,
    expected_bg_per_gram: float = 4.0
) -> float:
    """
    Estimate remaining COB fraction from BG response.

    This is used to create training labels from actual BG data.

    Args:
        bg_before: BG at time of meal
        bg_after: BG at measurement time
        carbs: Original carbs eaten
        insulin_effect: BG drop from insulin given (already factored in)
        expected_bg_per_gram: Expected BG rise per gram of carbs

    Returns:
        Estimated fraction of carbs remaining (0-1)
    """
    if carbs <= 0 or expected_bg_per_gram <= 0:
        return 1.0

    # Calculate expected total BG rise if all carbs absorbed
    expected_rise = carbs * expected_bg_per_gram

    # Calculate actual BG change (accounting for insulin)
    actual_rise = (bg_after - bg_before) + insulin_effect

    # Estimate absorbed carbs
    if expected_rise > 0:
        absorbed_fraction = actual_rise / expected_rise
        absorbed_fraction = max(0, min(1, absorbed_fraction))
        remaining_fraction = 1.0 - absorbed_fraction
    else:
        remaining_fraction = 1.0

    return remaining_fraction


def get_absorption_rate_from_gi(glycemic_index: int) -> str:
    """
    Map glycemic index to absorption rate category.

    Args:
        glycemic_index: Food's GI (0-100)

    Returns:
        Absorption rate: "fast", "medium", or "slow"
    """
    if glycemic_index >= 70:
        return "fast"
    elif glycemic_index >= 55:
        return "medium"
    else:
        return "slow"
