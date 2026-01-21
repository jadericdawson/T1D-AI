"""
Absorption Curve Learner

Learns personalized insulin and carb absorption curves from actual BG response data.
Unlike hardcoded onset times, this learns:
- When absorption actually starts (onset delay)
- How quickly it ramps up (ramp shape)
- Time to peak effect
- Decay rate

Key insight: We observe BG changes after insulin/carb events.
The BG response tells us when and how fast absorption occurred.
"""
import torch
import torch.nn as nn
import numpy as np
import math
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionCurveParams:
    """Learned absorption curve parameters."""
    onset_min: float       # Time until absorption starts (minutes)
    ramp_duration: float   # Time from onset to peak (minutes)
    peak_fraction: float   # Fraction of dose at peak effect (0-1)
    half_life_min: float   # Decay half-life after peak (minutes)

    def to_dict(self) -> Dict:
        return {
            'onset_min': self.onset_min,
            'ramp_duration': self.ramp_duration,
            'peak_fraction': self.peak_fraction,
            'half_life_min': self.half_life_min,
        }


class AbsorptionCurveModel(nn.Module):
    """
    Neural network that learns absorption curve parameters.

    Input features:
    - Time since dose
    - Dose amount
    - Time of day (circadian)
    - Contextual features (BG, activity, etc.)

    Output:
    - Effective absorption fraction (0-1) at given time

    The model learns the full absorption curve shape including:
    - Delayed onset (ramp-up doesn't start immediately)
    - Ramp-up rate (how quickly effect builds)
    - Peak timing
    - Decay rate
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size

        # Network that learns curve shape
        self.curve_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Output 0-1 absorption fraction
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, input_size)

        Returns:
            Absorption fraction (batch, 1) in range [0, 1]
        """
        return self.curve_network(x)


class InsulinAbsorptionLearner:
    """
    Learns personalized insulin absorption curves from BG response data.

    Training data: Correction boluses (insulin without concurrent carbs)
    - We observe BG at bolus time and BG at later times
    - The BG drop tells us how much insulin has been absorbed

    This learns the ACTUAL onset, ramp-up, and decay for THIS person.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        default_onset_min: float = 20.0,
        default_half_life_min: float = 54.0,  # Personalized from training
    ):
        self.model: Optional[AbsorptionCurveModel] = None
        self.default_onset_min = default_onset_min
        self.default_half_life_min = default_half_life_min
        self._use_ml = False

        # Learned parameters (updated during training)
        self.learned_params: Optional[AbsorptionCurveParams] = None

        # Load model if exists
        if model_path and model_path.exists():
            try:
                self._load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load insulin absorption model: {e}")

    def _load_model(self, path: Path):
        """Load trained model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model = AbsorptionCurveModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self._use_ml = True

        if 'learned_params' in checkpoint:
            params = checkpoint['learned_params']
            self.learned_params = AbsorptionCurveParams(**params)
            logger.info(f"Loaded insulin absorption model with onset={self.learned_params.onset_min:.1f}min")

    def get_effective_iob(
        self,
        dose_units: float,
        minutes_since_dose: float,
        hour: int = 12,
        current_bg: float = 120.0,
    ) -> float:
        """
        Get effective IOB (accounting for absorption delay).

        Unlike simple exponential decay, this accounts for:
        - Onset delay: IOB ramps up from 0, not instant
        - Learned curve shape from actual data

        Args:
            dose_units: Insulin dose
            minutes_since_dose: Time elapsed
            hour: Hour of day
            current_bg: Current BG level

        Returns:
            Effective IOB in units (with absorption delay applied)
        """
        if dose_units <= 0 or minutes_since_dose < 0:
            return 0.0

        # Use learned curve if available
        if self._use_ml and self.model is not None:
            try:
                absorption_fraction = self._predict_absorption(
                    minutes_since_dose, dose_units, hour, current_bg
                )
                return dose_units * (1.0 - absorption_fraction)  # Remaining = 1 - absorbed
            except Exception as e:
                logger.warning(f"ML absorption prediction failed: {e}")

        # Fallback: parametric curve with learned or default parameters
        return self._parametric_iob(dose_units, minutes_since_dose)

    def _predict_absorption(
        self,
        minutes: float,
        dose: float,
        hour: int,
        current_bg: float,
    ) -> float:
        """Predict absorption fraction using ML model."""
        features = self._build_features(minutes, dose, hour, current_bg)
        x = torch.tensor([features], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            absorption = self.model(x).item()

        return absorption

    def _build_features(
        self,
        minutes: float,
        dose: float,
        hour: int,
        current_bg: float,
    ) -> List[float]:
        """Build feature vector for model."""
        return [
            minutes / 360.0,                          # Time scaled
            dose / 10.0,                              # Dose scaled
            math.sin(2 * math.pi * hour / 24),        # Hour sin
            math.cos(2 * math.pi * hour / 24),        # Hour cos
            current_bg / 200.0,                       # BG scaled
            1.0 if minutes < 30 else 0.0,             # Early phase flag
            1.0 if 30 <= minutes < 90 else 0.0,       # Peak phase flag
            1.0 if minutes >= 90 else 0.0,            # Decay phase flag
            min(minutes / 30.0, 1.0),                 # Onset progress
            max(0, (minutes - 30) / 60.0),            # Post-onset progress
            dose * minutes / 3600.0,                  # Interaction term
            (current_bg - 120) / 100.0,               # BG deviation
        ]

    def _parametric_iob(self, dose: float, minutes: float) -> float:
        """
        Calculate IOB using parametric curve with absorption delay.

        Curve shape:
        - Phase 1 (0 to onset): IOB ramps up from 0
        - Phase 2 (onset to peak): IOB at maximum
        - Phase 3 (after peak): Exponential decay
        """
        onset = self.learned_params.onset_min if self.learned_params else self.default_onset_min
        half_life = self.learned_params.half_life_min if self.learned_params else self.default_half_life_min

        if minutes < 0:
            return dose

        # Phase 1: Ramp-up (quadratic)
        if minutes < onset:
            ramp_factor = (minutes / onset) ** 2
            return dose * ramp_factor

        # Phase 2+3: Exponential decay from peak
        time_since_onset = minutes - onset
        decay_factor = 0.5 ** (time_since_onset / half_life)
        return dose * decay_factor

    def get_onset_time(self) -> float:
        """Get the learned onset time in minutes."""
        if self.learned_params:
            return self.learned_params.onset_min
        return self.default_onset_min

    def get_half_life(self) -> float:
        """Get the learned half-life in minutes."""
        if self.learned_params:
            return self.learned_params.half_life_min
        return self.default_half_life_min


class CarbAbsorptionLearner:
    """
    Learns personalized carb absorption curves from BG response data.

    Training data: Meal events with known carbs
    - We observe BG rise after eating
    - The BG rise pattern tells us absorption timing

    Key factors learned:
    - Onset delay (time until BG starts rising)
    - Peak timing (when maximum rise rate occurs)
    - Duration (how long carbs affect BG)
    - GI effect (faster/slower based on food type)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        default_onset_min: float = 15.0,
        default_half_life_min: float = 45.0,
    ):
        self.model: Optional[AbsorptionCurveModel] = None
        self.default_onset_min = default_onset_min
        self.default_half_life_min = default_half_life_min
        self._use_ml = False

        self.learned_params: Optional[AbsorptionCurveParams] = None

        if model_path and model_path.exists():
            try:
                self._load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load carb absorption model: {e}")

    def _load_model(self, path: Path):
        """Load trained model."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model = AbsorptionCurveModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self._use_ml = True

        if 'learned_params' in checkpoint:
            params = checkpoint['learned_params']
            self.learned_params = AbsorptionCurveParams(**params)
            logger.info(f"Loaded carb absorption model with onset={self.learned_params.onset_min:.1f}min")

    def get_effective_cob(
        self,
        carbs_grams: float,
        minutes_since_meal: float,
        glycemic_index: float = 55.0,
        hour: int = 12,
    ) -> float:
        """
        Get effective COB (accounting for absorption delay).

        Args:
            carbs_grams: Carbs consumed
            minutes_since_meal: Time elapsed
            glycemic_index: Food GI (affects timing)
            hour: Hour of day

        Returns:
            Effective COB in grams (with absorption delay applied)
        """
        if carbs_grams <= 0 or minutes_since_meal < 0:
            return 0.0

        # Adjust timing based on GI
        gi_factor = glycemic_index / 55.0

        # Use learned curve if available
        if self._use_ml and self.model is not None:
            try:
                absorption_fraction = self._predict_absorption(
                    minutes_since_meal, carbs_grams, gi_factor, hour
                )
                return carbs_grams * (1.0 - absorption_fraction)
            except Exception as e:
                logger.warning(f"ML carb absorption prediction failed: {e}")

        # Fallback: parametric curve
        return self._parametric_cob(carbs_grams, minutes_since_meal, gi_factor)

    def _predict_absorption(
        self,
        minutes: float,
        carbs: float,
        gi_factor: float,
        hour: int,
    ) -> float:
        """Predict carb absorption using ML model."""
        features = self._build_features(minutes, carbs, gi_factor, hour)
        x = torch.tensor([features], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            absorption = self.model(x).item()

        return absorption

    def _build_features(
        self,
        minutes: float,
        carbs: float,
        gi_factor: float,
        hour: int,
    ) -> List[float]:
        """Build feature vector."""
        return [
            minutes / 180.0,                          # Time scaled
            carbs / 100.0,                            # Carbs scaled
            gi_factor,                                # GI factor
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 if minutes < 20 else 0.0,             # Early phase
            1.0 if 20 <= minutes < 60 else 0.0,       # Peak phase
            1.0 if minutes >= 60 else 0.0,            # Decay phase
            min(minutes / 20.0, 1.0),                 # Onset progress
            max(0, (minutes - 20) / 40.0),            # Post-onset
            carbs * gi_factor / 100.0,                # Interaction
            gi_factor - 1.0,                          # GI deviation
        ]

    def _parametric_cob(self, carbs: float, minutes: float, gi_factor: float) -> float:
        """Calculate COB using parametric curve."""
        onset = self.learned_params.onset_min if self.learned_params else self.default_onset_min
        half_life = self.learned_params.half_life_min if self.learned_params else self.default_half_life_min

        # Adjust for GI
        onset = onset / gi_factor
        half_life = half_life / gi_factor

        if minutes < 0:
            return carbs

        # Phase 1: Ramp-up
        if minutes < onset:
            ramp_factor = (minutes / onset) ** 2
            return carbs * ramp_factor

        # Phase 2+3: Decay
        time_since_onset = minutes - onset
        decay_factor = 0.5 ** (time_since_onset / half_life)
        return carbs * decay_factor

    def get_onset_time(self, glycemic_index: float = 55.0) -> float:
        """Get onset time adjusted for GI."""
        base_onset = self.learned_params.onset_min if self.learned_params else self.default_onset_min
        return base_onset / (glycemic_index / 55.0)


def extract_absorption_training_data(
    glucose_readings: List[Dict],
    treatments: List[Dict],
    isf: float = 50.0,
    icr: float = 10.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract training data for absorption models from historical data.

    Finds:
    1. Clean correction boluses (no carbs nearby) for insulin absorption
    2. Meals with clear BG response for carb absorption

    Args:
        glucose_readings: BG data with timestamps
        treatments: Insulin and carb events
        isf: Insulin sensitivity factor
        icr: Insulin to carb ratio

    Returns:
        (insulin_samples, carb_samples) - training data for each model
    """
    insulin_samples = []
    carb_samples = []

    # Sort data by timestamp
    readings = sorted(glucose_readings, key=lambda x: x['timestamp'])
    doses = sorted(treatments, key=lambda x: x['timestamp'])

    for dose in doses:
        dose_time = datetime.fromisoformat(dose['timestamp'].replace('Z', '+00:00'))
        if dose_time.tzinfo:
            dose_time = dose_time.replace(tzinfo=None)

        # Get BG at dose time
        bg_at_dose = _find_nearest_bg(readings, dose_time)
        if bg_at_dose is None:
            continue

        insulin = dose.get('insulin', 0) or 0
        carbs = dose.get('carbs', 0) or 0

        # Check for clean correction bolus (insulin only, no carbs within 2 hours)
        if insulin > 0 and carbs == 0:
            is_clean = _check_no_nearby_carbs(doses, dose_time, window_hours=2)
            if is_clean:
                # Track BG response over next 3 hours
                for minutes in [15, 30, 45, 60, 90, 120, 180]:
                    future_time = dose_time + timedelta(minutes=minutes)
                    future_bg = _find_nearest_bg(readings, future_time, tolerance_min=10)

                    if future_bg is not None:
                        bg_drop = bg_at_dose - future_bg
                        expected_drop = insulin * isf

                        # Estimate absorbed fraction from BG response
                        if expected_drop > 0:
                            absorbed_fraction = min(1.0, max(0.0, bg_drop / expected_drop))
                        else:
                            absorbed_fraction = 0.0

                        insulin_samples.append({
                            'minutes': minutes,
                            'dose': insulin,
                            'hour': dose_time.hour,
                            'bg_at_dose': bg_at_dose,
                            'bg_later': future_bg,
                            'absorbed_fraction': absorbed_fraction,
                        })

        # Check for meal with clear response (carbs, maybe with insulin)
        if carbs > 0:
            # Track BG response over next 3 hours
            for minutes in [15, 30, 45, 60, 90, 120, 180]:
                future_time = dose_time + timedelta(minutes=minutes)
                future_bg = _find_nearest_bg(readings, future_time, tolerance_min=10)

                if future_bg is not None:
                    # Adjust for any insulin effect
                    insulin_effect = 0
                    if insulin > 0:
                        # Simple estimate of insulin absorbed
                        insulin_absorbed = insulin * min(1.0, minutes / 120.0)
                        insulin_effect = insulin_absorbed * isf

                    # BG rise from carbs = (actual rise) + (insulin drop prevented)
                    bg_rise = (future_bg - bg_at_dose) + insulin_effect
                    expected_rise = carbs * (isf / icr)

                    if expected_rise > 0:
                        absorbed_fraction = min(1.0, max(0.0, bg_rise / expected_rise))
                    else:
                        absorbed_fraction = 0.0

                    carb_samples.append({
                        'minutes': minutes,
                        'carbs': carbs,
                        'glycemic_index': dose.get('glycemicIndex', 55) or 55,
                        'hour': dose_time.hour,
                        'bg_at_meal': bg_at_dose,
                        'bg_later': future_bg,
                        'absorbed_fraction': absorbed_fraction,
                    })

    logger.info(f"Extracted {len(insulin_samples)} insulin and {len(carb_samples)} carb absorption samples")
    return insulin_samples, carb_samples


def _find_nearest_bg(
    readings: List[Dict],
    target_time: datetime,
    tolerance_min: int = 5,
) -> Optional[float]:
    """Find BG reading nearest to target time."""
    best_reading = None
    best_diff = float('inf')

    for r in readings:
        r_time = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00'))
        if r_time.tzinfo:
            r_time = r_time.replace(tzinfo=None)

        diff = abs((r_time - target_time).total_seconds() / 60)
        if diff < best_diff and diff <= tolerance_min:
            best_diff = diff
            best_reading = r.get('value')

    return best_reading


def _check_no_nearby_carbs(
    treatments: List[Dict],
    dose_time: datetime,
    window_hours: float = 2.0,
) -> bool:
    """Check if there are no carbs within window of dose time."""
    window_sec = window_hours * 3600

    for t in treatments:
        t_time = datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
        if t_time.tzinfo:
            t_time = t_time.replace(tzinfo=None)

        time_diff = abs((t_time - dose_time).total_seconds())
        if time_diff <= window_sec and (t.get('carbs', 0) or 0) > 0:
            return False

    return True


def fit_absorption_curve(
    samples: List[Dict],
    model_type: str = 'insulin',
) -> AbsorptionCurveParams:
    """
    Fit absorption curve parameters from training samples.

    Uses simple curve fitting to find onset, peak, and decay parameters
    that best explain the observed BG responses.

    Args:
        samples: Training samples with 'minutes' and 'absorbed_fraction'
        model_type: 'insulin' or 'carb'

    Returns:
        Fitted curve parameters
    """
    if not samples:
        # Return defaults
        if model_type == 'insulin':
            return AbsorptionCurveParams(
                onset_min=20.0,
                ramp_duration=30.0,
                peak_fraction=0.9,
                half_life_min=54.0,
            )
        else:
            return AbsorptionCurveParams(
                onset_min=15.0,
                ramp_duration=25.0,
                peak_fraction=0.85,
                half_life_min=45.0,
            )

    # Group by time bucket and average
    time_buckets = {}
    for s in samples:
        bucket = round(s['minutes'] / 15) * 15  # 15-min buckets
        if bucket not in time_buckets:
            time_buckets[bucket] = []
        time_buckets[bucket].append(s['absorbed_fraction'])

    avg_absorption = {t: np.mean(fracs) for t, fracs in time_buckets.items()}

    # Find onset (when absorption first exceeds 10%)
    onset_min = 20.0  # default
    for t in sorted(avg_absorption.keys()):
        if avg_absorption[t] > 0.1:
            onset_min = max(5, t - 15)  # Onset is ~15 min before 10% absorption
            break

    # Find peak (when absorption rate starts slowing)
    peak_time = onset_min + 30  # default
    max_rate = 0
    prev_absorption = 0
    for t in sorted(avg_absorption.keys()):
        if t > onset_min:
            rate = (avg_absorption[t] - prev_absorption) / 15
            if rate > max_rate:
                max_rate = rate
                peak_time = t
            prev_absorption = avg_absorption[t]

    # Estimate half-life from decay portion
    half_life = 54.0 if model_type == 'insulin' else 45.0  # defaults
    decay_samples = [(t, f) for t, f in avg_absorption.items() if t > peak_time and f > 0.1]
    if len(decay_samples) >= 2:
        # Fit exponential decay: f(t) = e^(-t/tau)
        # ln(f) = -t/tau
        # Linear regression to find tau
        times = np.array([s[0] - peak_time for s in decay_samples])
        log_fracs = np.log(np.array([1.0 - s[1] for s in decay_samples]) + 0.01)  # remaining fraction

        if len(times) >= 2:
            slope = np.polyfit(times, log_fracs, 1)[0]
            if slope < 0:
                tau = -1.0 / slope
                half_life = tau * np.log(2)
                half_life = max(30, min(120, half_life))  # Clamp to reasonable range

    return AbsorptionCurveParams(
        onset_min=onset_min,
        ramp_duration=peak_time - onset_min,
        peak_fraction=max(avg_absorption.values()) if avg_absorption else 0.9,
        half_life_min=half_life,
    )


# Singleton instances
_insulin_learner: Optional[InsulinAbsorptionLearner] = None
_carb_learner: Optional[CarbAbsorptionLearner] = None


def get_insulin_absorption_learner() -> InsulinAbsorptionLearner:
    """Get or create insulin absorption learner singleton."""
    global _insulin_learner
    if _insulin_learner is None:
        # Path: src/ml/models/absorption_learner.py -> ../../../.. = backend
        model_path = Path(__file__).parent.parent.parent.parent / "models" / "insulin_absorption.pth"
        _insulin_learner = InsulinAbsorptionLearner(model_path=model_path)
    return _insulin_learner


def get_carb_absorption_learner() -> CarbAbsorptionLearner:
    """Get or create carb absorption learner singleton."""
    global _carb_learner
    if _carb_learner is None:
        # Path: src/ml/models/absorption_learner.py -> ../../../.. = backend
        model_path = Path(__file__).parent.parent.parent.parent / "models" / "carb_absorption.pth"
        _carb_learner = CarbAbsorptionLearner(model_path=model_path)
    return _carb_learner
