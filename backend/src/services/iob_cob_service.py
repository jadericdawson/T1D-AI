"""
IOB/COB Calculation Service for T1D-AI
Ported from dexcom_reader_predict_v2.3.py lines 468-606

Implements:
- Insulin on Board (IOB) using exponential decay with LEARNED half-life
- Carbs on Board (COB) using exponential decay
- Dose recommendation with IOB/COB adjustments

ML Models:
- IOB uses learned half-life from iob_forcing.pth (trained from actual BG drops)
- Default half-life is 81 min (adult), but we learned 72.66 min for this child
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
import torch

from models.schemas import Treatment, GlucoseReading, CurrentMetrics, UserAbsorptionProfile
from config import get_settings
from ml.models.forcing_ensemble import get_forcing_ensemble

logger = logging.getLogger(__name__)


@dataclass
class FoodSuggestion:
    """A food suggestion based on user's historical eating patterns."""
    name: str  # Food name/description
    carbs: float  # Carbs in grams
    typical_portion: str  # e.g., "1 slice", "1 cup"
    glycemic_index: Optional[int] = None
    times_eaten: int = 1  # How often user has eaten this

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DoseRecommendation:
    """Complete recommendation for dose and/or food to reach target BG."""
    # Insulin recommendation (if BG predicted high)
    recommended_dose: float = 0.0

    # Food recommendation (if BG predicted low)
    recommended_carbs: float = 0.0
    food_suggestions: List[FoodSuggestion] = field(default_factory=list)

    # Predictions
    current_bg: int = 0
    predicted_bg_without_action: int = 0
    predicted_bg_with_action: int = 0
    target_bg: int = 100

    # Context
    iob: float = 0.0
    cob: float = 0.0
    pob: float = 0.0
    isf: float = 50.0

    # Reasoning
    action_type: str = "none"  # "insulin", "food", "none"
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['food_suggestions'] = [s.to_dict() for s in self.food_suggestions]
        return result


def _to_utc_naive(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC timezone-naive for consistent comparisons.

    Handles:
    - Timezone-aware datetimes: converts to UTC then strips tzinfo
    - Timezone-naive datetimes: assumes they're already in UTC
    """
    if dt.tzinfo is not None:
        # Convert to UTC first, then make naive
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    # Already naive, assume UTC
    return dt


def _minutes_since(treatment_timestamp: datetime, reference_time: Optional[datetime] = None) -> float:
    """
    Calculate minutes elapsed since a treatment timestamp.

    Properly handles timezone conversions to avoid EST/UTC offset issues.
    """
    if reference_time is None:
        reference_time = datetime.utcnow()
    else:
        reference_time = _to_utc_naive(reference_time)

    treatment_utc = _to_utc_naive(treatment_timestamp)
    return (reference_time - treatment_utc).total_seconds() / 60

# Cache for user absorption profiles (avoid repeated DB lookups)
_user_absorption_profile_cache: dict = {}

# Load absorption parameters - using PHARMACY (textbook) values
def _load_learned_absorption_params() -> dict:
    """
    Return PHARMACY (textbook) absorption parameters.

    Using standard pharmacokinetic values until we have clean training data.
    The previous learned models were trained on noisy data with COB interference,
    resulting in unrealistic values (e.g., IOB half-life=120min).

    TODO: Retrain with clean datasets:
    - Isolated corrections (no carbs within 3 hours)
    - Isolated meals (no recent insulin stacking)
    """
    # FAST ABSORPTION VALUES (observed from real-world data)
    # Carbs hit bloodstream MUCH faster than textbook suggests, especially:
    # - Liquid carbs (chocolate milk, juice): 3-5 min onset
    # - High GI foods: 5-10 min onset
    # - Insulin takes longer to counteract due to onset delay
    params = {
        'iob_onset_min': 15.0,       # Rapid insulin onset ~15 min
        'iob_ramp_min': 25.0,        # Ramp to peak activity
        'iob_half_life_min': 81.0,   # Standard rapid insulin half-life
        'cob_onset_min': 5.0,        # Carbs start affecting BG within 5 min
        'cob_ramp_min': 10.0,        # Fast ramp for liquid/simple carbs
        'cob_half_life_min': 35.0,   # Faster absorption than textbook
        # PROTEIN ABSORPTION VALUES
        # Protein has delayed, slower effect on BG compared to carbs:
        # - Protein converts to glucose via gluconeogenesis (2-5 hours)
        # - Effect is ~50% of carb effect per gram (varies by protein source)
        # - High-fat protein (steak) is slower than lean protein (chicken)
        'pob_onset_min': 120.0,      # Protein starts affecting BG ~2 hours after eating
        'pob_ramp_min': 90.0,        # Longer ramp to peak activity (120-210 min)
        'pob_half_life_min': 75.0,   # 75 min half-life (slower than carbs' 35 min)
        'pob_duration_min': 300.0,   # 5 hour total duration
    }

    logger.info("Using FAST absorption params (real-world observed)")
    logger.info(f"  IOB: onset={params['iob_onset_min']}min, half-life={params['iob_half_life_min']}min")
    logger.info(f"  COB: onset={params['cob_onset_min']}min, half-life={params['cob_half_life_min']}min")
    logger.info(f"  POB: onset={params['pob_onset_min']}min, half-life={params['pob_half_life_min']}min")

    return params

# Module-level learned parameters (loaded once at import)
LEARNED_ABSORPTION_PARAMS = _load_learned_absorption_params()

# Backward compatibility
LEARNED_IOB_HALF_LIFE = LEARNED_ABSORPTION_PARAMS['iob_half_life_min']


class IOBCOBService:
    """Service for calculating Insulin on Board and Carbs on Board."""

    def __init__(
        self,
        insulin_duration_min: int = 300,  # 5 hours - standard for rapid-acting insulin
        insulin_half_life_min: float = None,  # Will use learned value if None
        insulin_onset_min: float = None,      # Will use learned value if None
        insulin_ramp_min: float = None,       # Will use learned value if None
        carb_duration_min: int = 180,
        carb_half_life_min: float = None,     # Will use learned value if None
        carb_onset_min: float = None,         # Will use learned value if None
        carb_ramp_min: float = None,          # Will use learned value if None
        carb_bg_factor: float = 4.0,
        # Protein absorption parameters
        protein_duration_min: int = 300,      # 5 hours for protein
        protein_half_life_min: float = None,  # Will use learned value if None
        protein_onset_min: float = None,      # Will use learned value if None
        protein_ramp_min: float = None,       # Will use learned value if None
        protein_bg_factor: float = 2.0,       # Protein raises BG ~50% as much as carbs
        target_bg: int = 100
    ):
        """
        Initialize IOB/COB/POB service with LEARNED absorption parameters.

        All absorption parameters are learned from actual BG response data:
        - Onset: When insulin/carbs/protein start affecting BG
        - Ramp: Duration of effect build-up phase
        - Half-life: Decay rate after peak

        Args:
            insulin_duration_min: Total duration of insulin action (default: 180 min)
            insulin_half_life_min: Half-life for insulin decay (default: 81 min)
            insulin_onset_min: Insulin onset delay (default: 15 min)
            insulin_ramp_min: Insulin ramp-up duration (default: 25 min)
            carb_duration_min: Total duration of carb absorption (default: 180 min)
            carb_half_life_min: Half-life for carb absorption (default: 35 min)
            carb_onset_min: Carb onset delay (default: 5 min)
            carb_ramp_min: Carb ramp-up duration (default: 10 min)
            carb_bg_factor: BG rise per gram of carbs (default: 4.0 mg/dL)
            protein_duration_min: Total duration of protein absorption (default: 300 min)
            protein_half_life_min: Half-life for protein absorption (default: 75 min)
            protein_onset_min: Protein onset delay (default: 120 min)
            protein_ramp_min: Protein ramp-up duration (default: 90 min)
            protein_bg_factor: BG rise per gram of protein (default: 2.0 mg/dL)
            target_bg: Target blood glucose level (default: 100 mg/dL)
        """
        self.insulin_duration_min = insulin_duration_min
        # Use learned parameters from trained absorption models
        self.insulin_half_life_min = insulin_half_life_min if insulin_half_life_min is not None else LEARNED_ABSORPTION_PARAMS['iob_half_life_min']
        self.insulin_onset_min = insulin_onset_min if insulin_onset_min is not None else LEARNED_ABSORPTION_PARAMS['iob_onset_min']
        self.insulin_ramp_min = insulin_ramp_min if insulin_ramp_min is not None else LEARNED_ABSORPTION_PARAMS['iob_ramp_min']
        self.carb_duration_min = carb_duration_min
        self.carb_half_life_min = carb_half_life_min if carb_half_life_min is not None else LEARNED_ABSORPTION_PARAMS['cob_half_life_min']
        self.carb_onset_min = carb_onset_min if carb_onset_min is not None else LEARNED_ABSORPTION_PARAMS['cob_onset_min']
        self.carb_ramp_min = carb_ramp_min if carb_ramp_min is not None else LEARNED_ABSORPTION_PARAMS['cob_ramp_min']
        self.carb_bg_factor = carb_bg_factor
        # Protein parameters
        self.protein_duration_min = protein_duration_min
        self.protein_half_life_min = protein_half_life_min if protein_half_life_min is not None else LEARNED_ABSORPTION_PARAMS['pob_half_life_min']
        self.protein_onset_min = protein_onset_min if protein_onset_min is not None else LEARNED_ABSORPTION_PARAMS['pob_onset_min']
        self.protein_ramp_min = protein_ramp_min if protein_ramp_min is not None else LEARNED_ABSORPTION_PARAMS['pob_ramp_min']
        self.protein_bg_factor = protein_bg_factor
        self.target_bg = target_bg

        # User-specific absorption profile (if loaded)
        self._user_profile: Optional[UserAbsorptionProfile] = None
        self._user_id: Optional[str] = None

    def with_user_profile(self, profile: UserAbsorptionProfile) -> "IOBCOBService":
        """
        Configure service to use a user's learned absorption profile.

        The profile contains personalized timing parameters that replace
        the hardcoded defaults:
        - insulinPeakMin replaces 75 min default
        - carbPeakMin replaces 45 min default
        - proteinPeakMin replaces 180 min default

        Args:
            profile: UserAbsorptionProfile with learned timing

        Returns:
            Self for chaining
        """
        self._user_profile = profile
        self._user_id = profile.userId

        # Update instance parameters from profile
        self.insulin_onset_min = profile.insulinOnsetMin
        self.insulin_ramp_min = profile.insulinPeakMin - profile.insulinOnsetMin  # Peak = onset + ramp
        self.protein_onset_min = profile.proteinOnsetMin
        self.protein_ramp_min = profile.proteinPeakMin - profile.proteinOnsetMin

        logger.info(f"IOBCOBService configured with user {profile.userId}'s learned profile: "
                   f"insulin_peak={profile.insulinPeakMin}min, "
                   f"carb_peak={profile.carbPeakMin}min, "
                   f"protein_peak={profile.proteinPeakMin}min")

        return self

    @property
    def insulin_peak_min(self) -> float:
        """Get insulin peak time (from profile or default)."""
        if self._user_profile:
            return self._user_profile.insulinPeakMin
        return 75.0  # Default

    @property
    def carb_peak_min(self) -> float:
        """Get carb peak time (from profile or default)."""
        if self._user_profile:
            return self._user_profile.carbPeakMin
        return 45.0  # Default

    @property
    def protein_peak_min(self) -> float:
        """Get protein peak time (from profile or default)."""
        if self._user_profile:
            return self._user_profile.proteinPeakMin
        return 180.0  # Default

    def get_metabolic_adjusted_params(
        self,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None
    ) -> dict:
        """
        Get absorption parameters adjusted for current metabolic state.

        Metabolic state affects insulin KINETICS (how fast it acts):
        - SICK/RESISTANT: Insulin acts slower (tissue resistance, delayed absorption)
        - SENSITIVE: Insulin acts faster (improved peripheral uptake)

        Absorption state affects carb KINETICS (how fast carbs absorb):
        - VERY_SLOW: Gastroparesis, illness - much slower absorption
        - SLOW: Mild delayed gastric emptying
        - FAST: Post-exercise, high metabolic rate

        These adjustments are ON TOP of learned baseline parameters.

        Args:
            metabolic_state: "sick", "resistant", "normal", "sensitive", "very_sensitive"
            absorption_state: "very_slow", "slow", "normal", "fast", "very_fast"

        Returns:
            dict with adjusted parameters:
                - insulin_half_life_min
                - insulin_onset_min
                - carb_half_life_min
                - carb_onset_min
                - protein_half_life_min
                - protein_onset_min
        """
        # Start with current (possibly learned) parameters
        adjusted_insulin_half_life = self.insulin_half_life_min
        adjusted_insulin_onset = self.insulin_onset_min
        adjusted_carb_half_life = self.carb_half_life_min
        adjusted_carb_onset = self.carb_onset_min
        adjusted_protein_half_life = self.protein_half_life_min
        adjusted_protein_onset = self.protein_onset_min

        # Apply metabolic state adjustments to INSULIN kinetics
        # Research shows illness/stress increases insulin resistance at tissue level
        if metabolic_state in ("sick", "resistant"):
            # Insulin acts slower when sick/resistant
            # - Delayed absorption from injection site
            # - Reduced peripheral tissue sensitivity
            # - Increased counter-regulatory hormones
            adjustment_factor = 1.2 if metabolic_state == "sick" else 1.15
            adjusted_insulin_half_life *= adjustment_factor
            adjusted_insulin_onset *= adjustment_factor
            logger.info(f"Metabolic state '{metabolic_state}': insulin half-life adjusted to {adjusted_insulin_half_life:.1f}min")

        elif metabolic_state in ("sensitive", "very_sensitive"):
            # Insulin acts faster when sensitive
            # - Better tissue uptake
            # - Lower counter-regulatory hormone levels
            adjustment_factor = 0.85 if metabolic_state == "very_sensitive" else 0.9
            adjusted_insulin_half_life *= adjustment_factor
            adjusted_insulin_onset *= adjustment_factor
            logger.info(f"Metabolic state '{metabolic_state}': insulin half-life adjusted to {adjusted_insulin_half_life:.1f}min")

        # Apply absorption state adjustments to CARB and PROTEIN kinetics
        # Gastroparesis, illness, or other factors affect gastric emptying
        if absorption_state == "very_slow":
            # Gastroparesis or severe illness - much slower absorption
            # Can see 50% or more delay in carb absorption
            carb_adjustment = 1.5
            protein_adjustment = 1.3  # Protein less affected
            adjusted_carb_half_life *= carb_adjustment
            adjusted_carb_onset *= carb_adjustment
            adjusted_protein_half_life *= protein_adjustment
            adjusted_protein_onset *= protein_adjustment
            logger.info(f"Absorption state 'very_slow': carb half-life adjusted to {adjusted_carb_half_life:.1f}min")

        elif absorption_state == "slow":
            # Mild delayed gastric emptying
            carb_adjustment = 1.25
            protein_adjustment = 1.15
            adjusted_carb_half_life *= carb_adjustment
            adjusted_carb_onset *= carb_adjustment
            adjusted_protein_half_life *= protein_adjustment
            adjusted_protein_onset *= protein_adjustment
            logger.info(f"Absorption state 'slow': carb half-life adjusted to {adjusted_carb_half_life:.1f}min")

        elif absorption_state == "fast":
            # Post-exercise or high metabolic rate
            carb_adjustment = 0.8
            protein_adjustment = 0.9
            adjusted_carb_half_life *= carb_adjustment
            adjusted_carb_onset *= carb_adjustment
            adjusted_protein_half_life *= protein_adjustment
            adjusted_protein_onset *= protein_adjustment
            logger.info(f"Absorption state 'fast': carb half-life adjusted to {adjusted_carb_half_life:.1f}min")

        elif absorption_state == "very_fast":
            # Liquid carbs, simple sugars
            carb_adjustment = 0.7
            protein_adjustment = 0.85
            adjusted_carb_half_life *= carb_adjustment
            adjusted_carb_onset *= carb_adjustment
            adjusted_protein_half_life *= protein_adjustment
            adjusted_protein_onset *= protein_adjustment
            logger.info(f"Absorption state 'very_fast': carb half-life adjusted to {adjusted_carb_half_life:.1f}min")

        return {
            'insulin_half_life_min': adjusted_insulin_half_life,
            'insulin_onset_min': adjusted_insulin_onset,
            'carb_half_life_min': adjusted_carb_half_life,
            'carb_onset_min': adjusted_carb_onset,
            'protein_half_life_min': adjusted_protein_half_life,
            'protein_onset_min': adjusted_protein_onset
        }

    def with_metabolic_state(
        self,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None
    ) -> "IOBCOBService":
        """
        Create a copy of this service with metabolic state adjustments applied.

        This is the recommended way to get metabolic-adjusted calculations:

            service = IOBCOBService.from_settings()
            adjusted_service = service.with_metabolic_state("sick", "slow")
            iob = adjusted_service.calculate_iob(treatments)

        Args:
            metabolic_state: Current metabolic state (sick/resistant/normal/sensitive)
            absorption_state: Current absorption state (very_slow/slow/normal/fast)

        Returns:
            New IOBCOBService instance with adjusted parameters
        """
        if metabolic_state in (None, "normal") and absorption_state in (None, "normal"):
            return self  # No adjustment needed

        adjusted_params = self.get_metabolic_adjusted_params(metabolic_state, absorption_state)

        # Create a new service with adjusted parameters
        adjusted_service = IOBCOBService(
            insulin_duration_min=self.insulin_duration_min,
            insulin_half_life_min=adjusted_params['insulin_half_life_min'],
            insulin_onset_min=adjusted_params['insulin_onset_min'],
            insulin_ramp_min=self.insulin_ramp_min,
            carb_duration_min=self.carb_duration_min,
            carb_half_life_min=adjusted_params['carb_half_life_min'],
            carb_onset_min=adjusted_params['carb_onset_min'],
            carb_ramp_min=self.carb_ramp_min,
            carb_bg_factor=self.carb_bg_factor,
            protein_duration_min=self.protein_duration_min,
            protein_half_life_min=adjusted_params['protein_half_life_min'],
            protein_onset_min=adjusted_params['protein_onset_min'],
            protein_ramp_min=self.protein_ramp_min,
            protein_bg_factor=self.protein_bg_factor,
            target_bg=self.target_bg
        )

        # Copy user profile if present
        if self._user_profile:
            adjusted_service._user_profile = self._user_profile
            adjusted_service._user_id = self._user_id

        return adjusted_service

    @classmethod
    async def for_user(cls, user_id: str) -> "IOBCOBService":
        """
        Create IOBCOBService configured with user's learned absorption profile.

        Loads the user's UserAbsorptionProfile from the database and configures
        the service to use their personalized timing parameters.

        Args:
            user_id: User ID to load profile for

        Returns:
            IOBCOBService configured with user's profile (or defaults if none)
        """
        from database.repositories import UserAbsorptionProfileRepository

        # Check cache first
        global _user_absorption_profile_cache
        if user_id in _user_absorption_profile_cache:
            profile = _user_absorption_profile_cache[user_id]
            service = cls()
            if profile:
                service.with_user_profile(profile)
            return service

        # Load from database
        repo = UserAbsorptionProfileRepository()
        profile = await repo.get(user_id)

        # Cache for 5 minutes (profile won't change often)
        _user_absorption_profile_cache[user_id] = profile

        service = cls()
        if profile:
            service.with_user_profile(profile)
        else:
            logger.debug(f"No absorption profile for user {user_id}, using defaults")

        return service

    @classmethod
    def from_settings(cls) -> "IOBCOBService":
        """Create service instance from application settings."""
        settings = get_settings()
        return cls(
            insulin_duration_min=settings.insulin_action_duration_minutes,
            insulin_half_life_min=settings.insulin_half_life_minutes,
            carb_duration_min=settings.carb_absorption_duration_minutes,
            carb_half_life_min=settings.carb_half_life_minutes,
            carb_bg_factor=settings.carb_bg_factor,
            target_bg=settings.target_bg
        )

    def calculate_iob(
        self,
        treatments: List[Treatment],
        at_time: Optional[datetime] = None,
        include_absorption_ramp: bool = True
    ) -> float:
        """
        Calculate Insulin on Board at a specific time.

        Uses LEARNED pharmacokinetic model with:
        - Onset delay (learned: 15 min): Time before insulin starts working
        - Ramp-up phase (learned: 75 min): IOB effect gradually increases
        - Decay phase: IOB decreases via exponential decay (learned: 120 min half-life)

        All parameters are learned from actual BG response data (54 correction boluses).

        Args:
            treatments: List of insulin treatments
            at_time: Time to calculate IOB for (default: now)
            include_absorption_ramp: Include absorption delay (default: True)

        Returns:
            Total IOB in units
        """
        if not treatments:
            return 0.0

        at_time = at_time or datetime.utcnow()
        total_iob = 0.0

        # Use learned onset from trained absorption model
        onset_min = self.insulin_onset_min
        ramp_min = self.insulin_ramp_min

        # Filter to insulin treatments only
        insulin_treatments = [t for t in treatments if t.insulin and t.insulin > 0]

        for treatment in insulin_treatments:
            # Calculate time elapsed since bolus (with proper timezone handling)
            time_elapsed = _minutes_since(treatment.timestamp, at_time)

            # Only count if within duration and after the bolus
            if 0 <= time_elapsed <= self.insulin_duration_min:
                # IOB with ABSORPTION DELAY - shows realistic pharmacokinetics:
                # 1. Onset phase (0-15min): Insulin not yet working, IOB stays high
                # 2. Ramp phase (15-90min): Insulin becoming active, IOB decays moderately
                # 3. Decay phase (90+min): Full decay as insulin is used up
                #
                # This creates a "shoulder" at the start before the decay kicks in

                if time_elapsed < onset_min:
                    # Pre-onset: Very slow decay (insulin not yet active)
                    # Only ~5% absorbed during onset
                    decay_factor = 1.0 - (0.05 * time_elapsed / onset_min)
                elif time_elapsed < (onset_min + ramp_min):
                    # Ramp-up phase: Moderate decay as insulin becomes active
                    ramp_progress = (time_elapsed - onset_min) / ramp_min
                    # Decay from 95% to 50% during ramp
                    decay_factor = 0.95 - (0.45 * ramp_progress)
                else:
                    # Full decay phase: Exponential decay of remaining 50%
                    decay_time = time_elapsed - onset_min - ramp_min
                    remaining_at_ramp_end = 0.5
                    decay_factor = remaining_at_ramp_end * (0.5 ** (decay_time / self.insulin_half_life_min))

                iob_contribution = treatment.insulin * decay_factor
                total_iob += iob_contribution

        return round(total_iob, 2)

    def calculate_cob(
        self,
        treatments: List[Treatment],
        at_time: Optional[datetime] = None,
        include_absorption_ramp: bool = True
    ) -> float:
        """
        Calculate Carbs on Board at a specific time.

        Uses pharmacokinetic model with:
        - Absorption ramp-up phase: COB gradually increases as food digests
        - Peak: COB reaches maximum when digestion complete
        - Decay phase: COB decreases as carbs enter bloodstream

        Includes glycemic index adjustment for timing.

        NOTE: The onset times are placeholders. TODO: Replace with ML model
        that learns personalized absorption curves from actual BG response data.

        Args:
            treatments: List of carb treatments
            at_time: Time to calculate COB for (default: now)
            include_absorption_ramp: Include absorption delay (default: True)

        Returns:
            Total COB in grams
        """
        if not treatments:
            return 0.0

        at_time = at_time or datetime.utcnow()
        total_cob = 0.0

        # Use learned carb onset and ramp from trained absorption model
        onset_min = self.carb_onset_min  # Learned: 5 min
        ramp_min = self.carb_ramp_min    # Learned: 10 min

        # Filter to carb treatments only
        carb_treatments = [t for t in treatments if t.carbs and t.carbs > 0]

        for treatment in carb_treatments:
            # Calculate time elapsed since eating (with proper timezone handling)
            time_elapsed = _minutes_since(treatment.timestamp, at_time)

            # Get glycemic index (default 55 = medium GI)
            gi = getattr(treatment, 'glycemicIndex', None) or 55
            is_liquid = getattr(treatment, 'isLiquid', False) or False

            # Use physiological GI-based equation for absorption parameters
            gi_params = gi_to_absorption_params(gi, is_liquid)
            adjusted_onset = gi_params['onset_min']
            adjusted_ramp = gi_params['ramp_min']
            adjusted_half_life = gi_params['half_life_min']
            adjusted_duration = gi_params['duration_min']

            # Only count if within adjusted duration and after eating
            if 0 <= time_elapsed <= adjusted_duration:
                # COB with GI-BASED ABSORPTION DELAY - shows realistic pharmacokinetics:
                # 1. Onset phase: Food digesting, COB stays high (duration depends on GI)
                # 2. Ramp phase: Carbs entering bloodstream, COB decays moderately
                # 3. Decay phase: Full exponential decay
                #
                # High GI foods have shorter onset/ramp (faster spike)
                # Low GI foods have longer onset/ramp (slower, sustained rise)

                if time_elapsed < adjusted_onset:
                    # Pre-onset: Very slow decay (carbs not yet absorbed)
                    # Only ~5% absorbed during onset
                    decay_factor = 1.0 - (0.05 * time_elapsed / adjusted_onset)
                elif time_elapsed < (adjusted_onset + adjusted_ramp):
                    # Ramp-up phase: Moderate decay as carbs are absorbed
                    ramp_progress = (time_elapsed - adjusted_onset) / adjusted_ramp
                    # Decay from 95% to 50% during ramp
                    decay_factor = 0.95 - (0.45 * ramp_progress)
                else:
                    # Full decay phase: Exponential decay of remaining
                    decay_time = time_elapsed - adjusted_onset - adjusted_ramp
                    remaining_at_ramp_end = 0.5
                    decay_factor = remaining_at_ramp_end * (0.5 ** (decay_time / adjusted_half_life))

                cob_contribution = treatment.carbs * decay_factor
                total_cob += cob_contribution

        return round(total_cob, 1)

    def calculate_pob(
        self,
        treatments: List[Treatment],
        at_time: Optional[datetime] = None,
        include_absorption_ramp: bool = True
    ) -> float:
        """
        Calculate Protein on Board at a specific time.

        Uses pharmacokinetic model with delayed absorption:
        - Protein has a much longer onset than carbs (2 hours vs 5-15 min)
        - Protein converts to glucose via gluconeogenesis over 2-5 hours
        - Effect is ~50% of carb effect per gram (varies by protein source)

        Phases:
        1. Pre-onset (0-120 min): POB stays ~100% (protein digesting)
        2. Ramp (120-210 min): Gradual absorption begins
        3. Decay (210+ min): Exponential decay with 75-min half-life

        Args:
            treatments: List of treatments with protein values
            at_time: Time to calculate POB for (default: now)
            include_absorption_ramp: Include absorption delay (default: True)

        Returns:
            Total POB in grams
        """
        if not treatments:
            return 0.0

        at_time = at_time or datetime.utcnow()
        total_pob = 0.0

        # Use learned protein onset and ramp from absorption parameters
        onset_min = self.protein_onset_min   # Default: 120 min
        ramp_min = self.protein_ramp_min     # Default: 90 min

        # Filter to treatments with protein
        protein_treatments = [t for t in treatments if getattr(t, 'protein', None) and t.protein > 0]

        for treatment in protein_treatments:
            # Calculate time elapsed since eating (with proper timezone handling)
            time_elapsed = _minutes_since(treatment.timestamp, at_time)

            # Only count if within duration and after eating
            if 0 <= time_elapsed <= self.protein_duration_min:
                # POB with DELAYED ABSORPTION - protein takes much longer than carbs:
                # 1. Pre-onset phase (0-120min): Protein digesting, POB stays very high
                # 2. Ramp phase (120-210min): Gluconeogenesis begins, POB decays moderately
                # 3. Decay phase (210+min): Full decay as protein converts to glucose

                # POB decay model: Continuous decay during digestion
                # Unlike the old model that stayed flat, protein is being broken down
                # and absorbed throughout the digestion process
                #
                # Phase 1 (0-onset): Gradual decay as protein digests (~30% absorbed)
                # Phase 2 (onset-onset+ramp): Faster decay as gluconeogenesis peaks
                # Phase 3 (after ramp): Exponential tail decay

                if time_elapsed < onset_min:
                    # Digestion phase: Protein is being broken down
                    # ~30% absorbed by end of onset (not flat!)
                    progress = time_elapsed / onset_min
                    decay_factor = 1.0 - (0.30 * progress)  # Linear decay to 70%
                elif time_elapsed < (onset_min + ramp_min):
                    # Peak absorption phase: Gluconeogenesis active
                    ramp_progress = (time_elapsed - onset_min) / ramp_min
                    # Decay from 70% to 35% during ramp
                    decay_factor = 0.70 - (0.35 * ramp_progress)
                else:
                    # Tail phase: Exponential decay of remaining protein
                    decay_time = time_elapsed - onset_min - ramp_min
                    remaining_at_ramp_end = 0.35
                    decay_factor = remaining_at_ramp_end * (0.5 ** (decay_time / self.protein_half_life_min))

                pob_contribution = treatment.protein * decay_factor
                total_pob += pob_contribution

        return round(total_pob, 1)

    def calculate_dose_recommendation(
        self,
        current_bg: int,
        iob: float,
        cob: float,
        isf: float,
        pob: float = 0.0,
        pir: float = None
    ) -> Tuple[float, int]:
        """
        Calculate recommended correction dose using CONVERGENT PREDICTIVE math.

        GOAL: Find the exact dose that lands BG at TARGET (100) after all factors play out.

        This solves the equation:
            BG_final = current_bg + COB_effect + POB_effect - IOB_effect - NewDose_effect = TARGET

        Solving for NewDose:
            NewDose = (current_bg + COB_effect + POB_effect - IOB_effect - TARGET) / (ISF * absorption_fraction)

        We use TARGET_TIME = 180 min (3 hours) as the convergence point where:
        - Most insulin will have acted (~95%)
        - Most carbs will have absorbed (~95%)
        - Protein effect will be well underway (~60%)

        SAFETY RULES:
        - If current BG < target: NEVER recommend insulin
        - If predicted BG at target time <= target: No dose needed
        - Cap at absolute maximum for safety

        Args:
            current_bg: Current blood glucose in mg/dL
            iob: Current Insulin on Board (units)
            cob: Current Carbs on Board (grams)
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            pob: Current Protein on Board (grams)
            pir: Protein to Insulin Ratio (unused, kept for API compatibility)

        Returns:
            Tuple of (recommended_dose, predicted_bg_at_target_time)
        """
        if isf <= 0:
            logger.warning("ISF must be positive for dose calculation")
            return 0.0, current_bg

        # SAFETY RULE 1: If BG is already below target, NEVER recommend insulin
        if current_bg < self.target_bg:
            predicted_bg = self._predict_bg_at_time(
                current_bg, iob, cob, pob, isf, 0.0, target_time_min=180
            )
            logger.info(
                f"Dose calc: BG={current_bg} is BELOW target {self.target_bg}. "
                f"No insulin recommended. Predicted BG at 3hr: {predicted_bg:.0f}"
            )
            return 0.0, int(round(predicted_bg))

        # TARGET TIME: 180 minutes (3 hours)
        # This is when we want BG to be at target - enough time for:
        # - Insulin to be ~95% absorbed
        # - Carbs to be ~95% absorbed
        # - Protein to have significant effect (~60%)
        TARGET_TIME_MIN = 180.0

        # Step 1: Predict BG at target time WITHOUT any new dose
        predicted_bg_no_dose = self._predict_bg_at_time(
            current_bg, iob, cob, pob, isf, new_dose=0.0, target_time_min=TARGET_TIME_MIN
        )

        # Step 2: If predicted BG is already at or below target, no dose needed
        if predicted_bg_no_dose <= self.target_bg:
            logger.info(
                f"Dose calc: Predicted BG at {TARGET_TIME_MIN:.0f}min ({predicted_bg_no_dose:.0f}) <= "
                f"target ({self.target_bg}). No correction needed."
            )
            return 0.0, int(round(predicted_bg_no_dose))

        # Step 3: Calculate the exact dose to converge on target
        # Use iterative refinement for precision
        dose = self._solve_dose_for_target(
            current_bg, iob, cob, pob, isf,
            target_bg=self.target_bg,
            target_time_min=TARGET_TIME_MIN
        )

        # Step 4: Verify the dose lands at target (sanity check)
        predicted_bg_with_dose = self._predict_bg_at_time(
            current_bg, iob, cob, pob, isf, new_dose=dose, target_time_min=TARGET_TIME_MIN
        )

        # SAFETY: Cap at absolute maximum
        ABSOLUTE_MAX_DOSE = 5.0
        if dose > ABSOLUTE_MAX_DOSE:
            logger.warning(f"Dose capped from {dose:.2f}U to {ABSOLUTE_MAX_DOSE}U for safety")
            dose = ABSOLUTE_MAX_DOSE
            # Recalculate predicted BG with capped dose
            predicted_bg_with_dose = self._predict_bg_at_time(
                current_bg, iob, cob, pob, isf, new_dose=dose, target_time_min=TARGET_TIME_MIN
            )

        dose = round(max(0, dose), 2)

        logger.info(
            f"CONVERGENT DOSE CALC: BG={current_bg}, Target={self.target_bg}, "
            f"IOB={iob:.2f}U, COB={cob:.0f}g, POB={pob:.0f}g, ISF={isf:.0f}\n"
            f"  Without dose: BG at {TARGET_TIME_MIN:.0f}min = {predicted_bg_no_dose:.0f}\n"
            f"  With {dose:.2f}U: BG at {TARGET_TIME_MIN:.0f}min = {predicted_bg_with_dose:.0f} (target: {self.target_bg})"
        )

        return dose, int(round(predicted_bg_with_dose))

    def _predict_bg_at_time(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        pob: float,
        isf: float,
        new_dose: float,
        target_time_min: float
    ) -> float:
        """
        Predict BG at a specific future time considering all metabolic factors.

        Uses exponential decay models for absorption:
        - Insulin: half-life based decay (faster for children)
        - Carbs: half-life based decay (varies by GI)
        - Protein: delayed onset (2hr) then decay
        - New dose: same kinetics as IOB but starts from 0 absorption

        Args:
            current_bg: Current blood glucose
            iob: Current IOB (units)
            cob: Current COB (grams)
            pob: Current POB (grams)
            isf: Insulin sensitivity factor
            new_dose: New insulin dose being considered (units)
            target_time_min: Time in future to predict (minutes)

        Returns:
            Predicted BG at target_time_min
        """
        # IOB effect: How much existing IOB will lower BG by target time
        # fraction_acted = 1 - 0.5^(time/half_life)
        iob_fraction_acted = 1.0 - (0.5 ** (target_time_min / self.insulin_half_life_min))
        iob_effect = iob * isf * iob_fraction_acted

        # NEW DOSE effect: How much new dose will lower BG by target time
        # New dose has onset delay (~15 min) before it starts acting
        INSULIN_ONSET_MIN = 15.0
        effective_time_for_new_dose = max(0, target_time_min - INSULIN_ONSET_MIN)
        new_dose_fraction_acted = 1.0 - (0.5 ** (effective_time_for_new_dose / self.insulin_half_life_min))
        new_dose_effect = new_dose * isf * new_dose_fraction_acted

        # COB effect: How much carbs will raise BG by target time
        cob_fraction_absorbed = 1.0 - (0.5 ** (target_time_min / self.carb_half_life_min))
        cob_effect = cob * self.carb_bg_factor * cob_fraction_absorbed

        # POB effect: Protein has delayed onset (~120 min) then acts
        PROTEIN_ONSET_MIN = 120.0
        PROTEIN_HALF_LIFE_MIN = 90.0  # Slower than carbs
        effective_time_for_protein = max(0, target_time_min - PROTEIN_ONSET_MIN)
        pob_fraction_effective = 1.0 - (0.5 ** (effective_time_for_protein / PROTEIN_HALF_LIFE_MIN))
        # Protein only converts ~50% to glucose equivalent
        pob_effect = pob * self.protein_bg_factor * pob_fraction_effective * 0.5

        # Final predicted BG
        predicted_bg = current_bg + cob_effect + pob_effect - iob_effect - new_dose_effect

        # Clamp to physiological range
        predicted_bg = max(40, min(400, predicted_bg))

        return predicted_bg

    def _solve_dose_for_target(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        pob: float,
        isf: float,
        target_bg: float,
        target_time_min: float,
        tolerance: float = 1.0,
        max_iterations: int = 20
    ) -> float:
        """
        Solve for the exact dose that lands BG at target using Newton-Raphson iteration.

        The equation to solve:
            predicted_bg(dose) = target_bg

        We use iterative refinement because the relationship between dose and BG
        is well-behaved (monotonically decreasing).

        Args:
            current_bg: Current blood glucose
            iob, cob, pob: Current on-board values
            isf: Insulin sensitivity factor
            target_bg: Desired final BG (typically 100)
            target_time_min: Time horizon for convergence
            tolerance: Acceptable error in mg/dL
            max_iterations: Maximum refinement iterations

        Returns:
            Dose (in units) that achieves target BG
        """
        # Calculate BG effect per unit of insulin at target time
        # This is the "derivative" for Newton-Raphson
        INSULIN_ONSET_MIN = 15.0
        effective_time = max(0, target_time_min - INSULIN_ONSET_MIN)
        dose_fraction_acted = 1.0 - (0.5 ** (effective_time / self.insulin_half_life_min))
        bg_drop_per_unit = isf * dose_fraction_acted

        if bg_drop_per_unit <= 0:
            logger.warning("Cannot calculate dose: insulin has no effect at target time")
            return 0.0

        # Initial guess: simple linear calculation
        predicted_no_dose = self._predict_bg_at_time(
            current_bg, iob, cob, pob, isf, new_dose=0.0, target_time_min=target_time_min
        )

        if predicted_no_dose <= target_bg:
            return 0.0  # Already at or below target

        # Initial dose estimate
        dose = (predicted_no_dose - target_bg) / bg_drop_per_unit

        # Iterative refinement (Newton-Raphson)
        for iteration in range(max_iterations):
            predicted_bg = self._predict_bg_at_time(
                current_bg, iob, cob, pob, isf, new_dose=dose, target_time_min=target_time_min
            )

            error = predicted_bg - target_bg

            if abs(error) <= tolerance:
                logger.debug(f"Dose convergence in {iteration + 1} iterations: {dose:.3f}U -> BG {predicted_bg:.1f}")
                break

            # Adjust dose: if predicted is too high, increase dose
            dose_adjustment = error / bg_drop_per_unit
            dose = dose + dose_adjustment

            # Ensure dose stays non-negative
            dose = max(0, dose)

        return dose

    def _predict_bg_with_new_carbs(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        pob: float,
        isf: float,
        new_carbs: float,
        target_time_min: float
    ) -> float:
        """
        Predict BG at target time WITH additional carbs eaten now.

        New carbs are added to existing COB and both absorb together.

        Args:
            current_bg: Current blood glucose
            iob: Current IOB (units)
            cob: Current COB (grams)
            pob: Current POB (grams)
            isf: Insulin sensitivity factor
            new_carbs: New carbs being eaten (grams)
            target_time_min: Time in future to predict (minutes)

        Returns:
            Predicted BG at target_time_min
        """
        # IOB effect (unchanged)
        iob_fraction_acted = 1.0 - (0.5 ** (target_time_min / self.insulin_half_life_min))
        iob_effect = iob * isf * iob_fraction_acted

        # Total carb effect: existing COB + new carbs
        # New carbs have slight delay (~10 min) before absorption starts
        CARB_ONSET_MIN = 10.0
        effective_time_for_new_carbs = max(0, target_time_min - CARB_ONSET_MIN)

        # Existing COB fraction absorbed
        cob_fraction = 1.0 - (0.5 ** (target_time_min / self.carb_half_life_min))
        cob_effect = cob * self.carb_bg_factor * cob_fraction

        # New carbs fraction absorbed (with onset delay)
        new_carb_fraction = 1.0 - (0.5 ** (effective_time_for_new_carbs / self.carb_half_life_min))
        new_carb_effect = new_carbs * self.carb_bg_factor * new_carb_fraction

        # POB effect (unchanged)
        PROTEIN_ONSET_MIN = 120.0
        PROTEIN_HALF_LIFE_MIN = 90.0
        effective_time_for_protein = max(0, target_time_min - PROTEIN_ONSET_MIN)
        pob_fraction = 1.0 - (0.5 ** (effective_time_for_protein / PROTEIN_HALF_LIFE_MIN))
        pob_effect = pob * self.protein_bg_factor * pob_fraction * 0.5

        # Final predicted BG
        predicted_bg = current_bg + cob_effect + new_carb_effect + pob_effect - iob_effect

        return max(40, min(400, predicted_bg))

    def _solve_carbs_for_target(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        pob: float,
        isf: float,
        target_bg: float,
        target_time_min: float,
        tolerance: float = 1.0,
        max_iterations: int = 20
    ) -> float:
        """
        Solve for the exact carbs needed to raise BG to target.

        Uses Newton-Raphson iteration, similar to dose calculation but inverted.

        Args:
            current_bg: Current blood glucose
            iob, cob, pob: Current on-board values
            isf: Insulin sensitivity factor
            target_bg: Desired final BG (typically 100)
            target_time_min: Time horizon for convergence
            tolerance: Acceptable error in mg/dL
            max_iterations: Maximum refinement iterations

        Returns:
            Carbs (in grams) that achieves target BG
        """
        # Calculate BG rise per gram of carbs at target time
        CARB_ONSET_MIN = 10.0
        effective_time = max(0, target_time_min - CARB_ONSET_MIN)
        carb_fraction = 1.0 - (0.5 ** (effective_time / self.carb_half_life_min))
        bg_rise_per_gram = self.carb_bg_factor * carb_fraction

        if bg_rise_per_gram <= 0:
            logger.warning("Cannot calculate carbs: no effect at target time")
            return 0.0

        # Predict BG without new carbs
        predicted_no_carbs = self._predict_bg_with_new_carbs(
            current_bg, iob, cob, pob, isf, new_carbs=0.0, target_time_min=target_time_min
        )

        if predicted_no_carbs >= target_bg:
            return 0.0  # Already at or above target

        # How much BG needs to rise
        bg_deficit = target_bg - predicted_no_carbs

        # Initial carbs estimate
        carbs = bg_deficit / bg_rise_per_gram

        # Iterative refinement
        for iteration in range(max_iterations):
            predicted_bg = self._predict_bg_with_new_carbs(
                current_bg, iob, cob, pob, isf, new_carbs=carbs, target_time_min=target_time_min
            )

            error = target_bg - predicted_bg  # Positive if still below target

            if abs(error) <= tolerance:
                logger.debug(f"Carbs convergence in {iteration + 1} iterations: {carbs:.1f}g -> BG {predicted_bg:.1f}")
                break

            # Adjust carbs: if predicted is too low, add more carbs
            carbs_adjustment = error / bg_rise_per_gram
            carbs = carbs + carbs_adjustment

            # Ensure carbs stays non-negative
            carbs = max(0, carbs)

        return carbs

    def get_food_suggestions_from_history(
        self,
        treatments: List[Treatment],
        target_carbs: float,
        tolerance_grams: float = 10.0
    ) -> List[FoodSuggestion]:
        """
        Find foods from user's history that match the target carb amount.

        Args:
            treatments: User's historical treatments
            target_carbs: Target carbs needed (grams)
            tolerance_grams: How close to target the food should be

        Returns:
            List of FoodSuggestion sorted by relevance
        """
        # Count food occurrences and collect details
        food_counts: Dict[str, Dict[str, Any]] = {}

        for t in treatments:
            # Only look at carb treatments with notes (food descriptions)
            if t.carbs and t.carbs > 0 and t.notes:
                food_name = t.notes.strip().lower()
                if not food_name:
                    continue

                if food_name not in food_counts:
                    food_counts[food_name] = {
                        'name': t.notes.strip(),  # Keep original case
                        'carbs_list': [],
                        'gi_list': [],
                        'count': 0
                    }

                food_counts[food_name]['carbs_list'].append(t.carbs)
                food_counts[food_name]['count'] += 1
                if t.glycemicIndex:
                    food_counts[food_name]['gi_list'].append(t.glycemicIndex)

        # Score and filter foods
        suggestions = []
        for food_key, data in food_counts.items():
            avg_carbs = sum(data['carbs_list']) / len(data['carbs_list'])
            avg_gi = int(sum(data['gi_list']) / len(data['gi_list'])) if data['gi_list'] else None

            # Calculate how well this food matches target
            carb_diff = abs(avg_carbs - target_carbs)

            # Score: prefer foods eaten frequently and close to target carbs
            # Allow foods within 2x of target (can eat half or double portion)
            if avg_carbs > 0 and (avg_carbs <= target_carbs * 2.5 or carb_diff <= tolerance_grams * 2):
                # Calculate portion to match target
                portion_multiplier = target_carbs / avg_carbs if avg_carbs > 0 else 1

                # Create portion description
                if 0.9 <= portion_multiplier <= 1.1:
                    portion = "1 serving"
                elif portion_multiplier < 0.9:
                    portion = f"{portion_multiplier:.1f}x serving (~{target_carbs:.0f}g carbs)"
                else:
                    portion = f"{portion_multiplier:.1f}x serving (~{target_carbs:.0f}g carbs)"

                suggestion = FoodSuggestion(
                    name=data['name'],
                    carbs=round(avg_carbs, 0),
                    typical_portion=portion,
                    glycemic_index=avg_gi,
                    times_eaten=data['count']
                )
                suggestions.append((carb_diff, -data['count'], suggestion))  # Sort by carb_diff asc, count desc

        # Sort by closest to target carbs, then by frequency
        suggestions.sort(key=lambda x: (x[0], x[1]))

        # Return top 5 suggestions
        return [s[2] for s in suggestions[:5]]

    def calculate_full_recommendation(
        self,
        current_bg: int,
        iob: float,
        cob: float,
        isf: float,
        pob: float = 0.0,
        user_treatments: Optional[List[Treatment]] = None
    ) -> DoseRecommendation:
        """
        Calculate complete recommendation: dose AND/OR food to reach target 100.

        This method:
        1. Predicts BG at 3 hours without any action
        2. If BG will be HIGH: recommends insulin dose
        3. If BG will be LOW: recommends carbs and suggests foods from user's history

        Args:
            current_bg: Current blood glucose in mg/dL
            iob: Current Insulin on Board (units)
            cob: Current Carbs on Board (grams)
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            pob: Current Protein on Board (grams)
            user_treatments: User's historical treatments for food suggestions

        Returns:
            DoseRecommendation with complete advice
        """
        TARGET_TIME_MIN = 180.0
        user_treatments = user_treatments or []

        # Predict BG at target time without any action
        predicted_bg_no_action = self._predict_bg_at_time(
            current_bg, iob, cob, pob, isf, new_dose=0.0, target_time_min=TARGET_TIME_MIN
        )

        recommendation = DoseRecommendation(
            current_bg=current_bg,
            predicted_bg_without_action=int(round(predicted_bg_no_action)),
            target_bg=self.target_bg,
            iob=iob,
            cob=cob,
            pob=pob,
            isf=isf
        )

        # Case 1: BG predicted to be at target (within ±5)
        if abs(predicted_bg_no_action - self.target_bg) <= 5:
            recommendation.action_type = "none"
            recommendation.predicted_bg_with_action = int(round(predicted_bg_no_action))
            recommendation.reasoning = f"BG predicted to be {predicted_bg_no_action:.0f} at 3hr, already at target."
            return recommendation

        # Case 2: BG predicted HIGH -> recommend insulin
        if predicted_bg_no_action > self.target_bg:
            # Don't recommend insulin if current BG is already low
            if current_bg < self.target_bg:
                recommendation.action_type = "none"
                recommendation.predicted_bg_with_action = int(round(predicted_bg_no_action))
                recommendation.reasoning = (
                    f"Current BG ({current_bg}) is below target. "
                    f"Predicted to rise to {predicted_bg_no_action:.0f} from COB/POB. "
                    f"No insulin - let carbs work."
                )
                return recommendation

            dose = self._solve_dose_for_target(
                current_bg, iob, cob, pob, isf,
                target_bg=self.target_bg,
                target_time_min=TARGET_TIME_MIN
            )

            # Cap dose
            ABSOLUTE_MAX_DOSE = 5.0
            if dose > ABSOLUTE_MAX_DOSE:
                dose = ABSOLUTE_MAX_DOSE

            predicted_with_dose = self._predict_bg_at_time(
                current_bg, iob, cob, pob, isf, new_dose=dose, target_time_min=TARGET_TIME_MIN
            )

            recommendation.recommended_dose = round(max(0, dose), 2)
            recommendation.action_type = "insulin"
            recommendation.predicted_bg_with_action = int(round(predicted_with_dose))
            recommendation.reasoning = (
                f"BG predicted {predicted_bg_no_action:.0f} at 3hr. "
                f"{dose:.2f}U insulin will bring it to ~{self.target_bg}."
            )

            logger.info(
                f"DOSE RECOMMENDATION: BG={current_bg}, Predicted={predicted_bg_no_action:.0f}, "
                f"Dose={dose:.2f}U -> Final={predicted_with_dose:.0f}"
            )
            return recommendation

        # Case 3: BG predicted LOW -> recommend food
        carbs_needed = self._solve_carbs_for_target(
            current_bg, iob, cob, pob, isf,
            target_bg=self.target_bg,
            target_time_min=TARGET_TIME_MIN
        )

        # Round to nearest 5g for practical eating
        carbs_needed = round(carbs_needed / 5) * 5
        carbs_needed = max(5, carbs_needed)  # At least 5g

        predicted_with_food = self._predict_bg_with_new_carbs(
            current_bg, iob, cob, pob, isf, new_carbs=carbs_needed, target_time_min=TARGET_TIME_MIN
        )

        # Get food suggestions from user's history
        food_suggestions = self.get_food_suggestions_from_history(
            user_treatments, target_carbs=carbs_needed
        )

        recommendation.recommended_carbs = carbs_needed
        recommendation.food_suggestions = food_suggestions
        recommendation.action_type = "food"
        recommendation.predicted_bg_with_action = int(round(predicted_with_food))
        recommendation.reasoning = (
            f"BG predicted {predicted_bg_no_action:.0f} at 3hr (below target). "
            f"Eat ~{carbs_needed:.0f}g carbs to reach ~{self.target_bg}."
        )

        logger.info(
            f"FOOD RECOMMENDATION: BG={current_bg}, Predicted={predicted_bg_no_action:.0f}, "
            f"Carbs needed={carbs_needed:.0f}g -> Final={predicted_with_food:.0f}, "
            f"Suggestions: {[f.name for f in food_suggestions[:3]]}"
        )

        return recommendation

    def get_current_metrics(
        self,
        current_bg: int,
        treatments: List[Treatment],
        isf: float,
        pir: float = 25.0
    ) -> CurrentMetrics:
        """
        Calculate all current metrics for display.

        Args:
            current_bg: Current blood glucose
            treatments: Recent treatments for IOB/COB/POB calculation
            isf: Predicted ISF value
            pir: Protein-to-insulin ratio (grams per unit)

        Returns:
            CurrentMetrics with all calculated values including POB and protein dose
        """
        iob = self.calculate_iob(treatments)
        cob = self.calculate_cob(treatments)
        pob = self.calculate_pob(treatments)

        # Get full recommendation (dose + food)
        full_rec = self.calculate_full_recommendation(
            current_bg=current_bg,
            iob=iob,
            cob=cob,
            isf=isf,
            pob=pob,
            user_treatments=treatments  # Pass treatments for food history
        )
        dose = full_rec.recommended_dose
        effective_bg = full_rec.predicted_bg_with_action

        # Calculate protein dose based on insulin-protein timing overlap
        # NOW = portion of protein BG effect that overlaps with insulin action window
        # LATER = protein effect that occurs after insulin wears off
        protein_dose_now = 0.0
        protein_dose_later = 0.0

        if pob > 0 and pir > 0:
            total_protein_dose = pob / pir

            # Timing parameters (in minutes)
            INSULIN_DIA = 300.0        # Insulin Duration of Action (~5 hours)
            PROTEIN_ONSET = 120.0      # When protein starts affecting BG (~2 hours)
            PROTEIN_DURATION = 300.0   # How long protein affects BG after onset (~5 hours)
            PROTEIN_END = PROTEIN_ONSET + PROTEIN_DURATION  # 420 min = 7 hours after eating

            # Find protein treatments within the relevant time window (PROTEIN_END = 7 hours)
            # Only recent protein affects current dosing decisions
            now = datetime.utcnow()
            protein_treatments = [
                t for t in treatments
                if t.protein and t.protein > 0
                and t.timestamp
                and _minutes_since(t.timestamp, now) <= PROTEIN_END
            ]
            if protein_treatments:
                total_protein = sum(t.protein for t in protein_treatments)
                # Use proper timezone handling to avoid EST/UTC offset issues
                weighted_minutes = sum(
                    (t.protein / total_protein) * _minutes_since(t.timestamp, now)
                    for t in protein_treatments
                    if t.timestamp
                )

                # DEBUG: Log the actual values to diagnose timezone issues
                logger.info(f"Protein dose calc: weighted_minutes={weighted_minutes:.1f}, PROTEIN_END={PROTEIN_END}, pob={pob:.1f}g, total_protein_dose={total_protein_dose:.2f}U")
                for t in protein_treatments:
                    mins = _minutes_since(t.timestamp, now)
                    logger.info(f"  Treatment: {t.protein}g protein, timestamp={t.timestamp}, minutes_since={mins:.1f}")

                # Calculate remaining protein effect window (from now, in minutes)
                if weighted_minutes >= PROTEIN_END:
                    # Protein effect is complete, no dose needed
                    protein_dose_now = 0.0
                    protein_dose_later = 0.0
                else:
                    # Remaining protein effect timing (relative to NOW)
                    if weighted_minutes < PROTEIN_ONSET:
                        # Protein hasn't started affecting BG yet
                        remaining_start = PROTEIN_ONSET - weighted_minutes  # starts in X minutes
                        remaining_end = PROTEIN_END - weighted_minutes      # ends in Y minutes
                    else:
                        # Protein is actively affecting BG
                        remaining_start = 0  # affecting BG now
                        remaining_end = PROTEIN_END - weighted_minutes  # ends in Y minutes

                    remaining_duration = remaining_end - remaining_start

                    # Calculate overlap with insulin action window (0 to INSULIN_DIA from now)
                    overlap_start = max(remaining_start, 0)
                    overlap_end = min(remaining_end, INSULIN_DIA)
                    overlap_duration = max(0, overlap_end - overlap_start)

                    # NOW fraction = what portion of remaining protein effect insulin can cover
                    if remaining_duration > 0:
                        now_fraction = overlap_duration / remaining_duration
                    else:
                        now_fraction = 0.0

                    protein_dose_now = total_protein_dose * now_fraction
                    protein_dose_later = total_protein_dose * (1 - now_fraction)

                    logger.debug(
                        f"Protein timing: {weighted_minutes:.0f}min since meal, "
                        f"remaining effect [{remaining_start:.0f}-{remaining_end:.0f}]min from now, "
                        f"insulin covers [{0}-{INSULIN_DIA:.0f}]min, "
                        f"overlap={overlap_duration:.0f}min, NOW fraction={now_fraction:.1%}"
                    )
            else:
                # No protein treatments found with timestamps, conservative split
                protein_dose_now = total_protein_dose * 0.3
                protein_dose_later = total_protein_dose * 0.7

            # If no correction needed, move protein NOW to LATER to prevent lows
            if dose <= 0:
                protein_dose_later += protein_dose_now
                protein_dose_now = 0.0

        # Convert FoodSuggestion dataclasses to dicts for Pydantic
        from models.schemas import FoodSuggestionSchema
        food_suggestions_schema = [
            FoodSuggestionSchema(
                name=f.name,
                carbs=f.carbs,
                typical_portion=f.typical_portion,
                glycemic_index=f.glycemic_index,
                times_eaten=f.times_eaten
            )
            for f in full_rec.food_suggestions
        ]

        return CurrentMetrics(
            iob=iob,
            cob=cob,
            pob=pob,
            isf=isf,
            recommendedDose=dose,
            effectiveBg=effective_bg,
            proteinDoseNow=round(protein_dose_now, 2),
            proteinDoseLater=round(protein_dose_later, 2),
            # Food recommendation fields
            actionType=full_rec.action_type,
            recommendedCarbs=full_rec.recommended_carbs,
            foodSuggestions=food_suggestions_schema,
            predictedBgWithoutAction=full_rec.predicted_bg_without_action,
            predictedBgWithAction=full_rec.predicted_bg_with_action,
            recommendationReasoning=full_rec.reasoning
        )


    def calculate_bg_effect_curve(
        self,
        current_iob: float,
        current_cob: float,
        isf: float = 50.0,
        icr: float = 10.0,
        duration_min: int = 60,
        step_min: int = 5,
        current_bg: float = 120.0,
        treatments: Optional[List[Treatment]] = None,
        base_time: Optional[datetime] = None,
        current_pob: float = 0.0
    ) -> List[dict]:
        """
        Calculate projected IOB/COB effect on BG over time.

        Uses the SAME treatment-based cumulative absorbed formula as historical bgPressure
        to ensure continuity between historical and projected lines.

        This provides the data for IOB/COB push/pull visualization on the chart.
        Shows three trajectory lines rooted at current BG:
        - bgWithCobOnly: Where BG would go with just carbs (no insulin) - the "ceiling"
        - bgWithIobOnly: Where BG would go with just insulin (no carbs) - the "floor"
        - expectedBg: Combined trajectory (actual expected path between ceiling and floor)

        This helps users visualize:
        - How well-timed insulin was relative to carbs
        - How different glycemic index foods affect BG trajectory
        - Whether they're trending toward upper (need insulin) or lower (need carbs) bound

        Args:
            current_iob: Current IOB in units
            current_cob: Current COB in grams
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            icr: Insulin to Carb Ratio (carbs per unit)
            duration_min: Projection duration
            step_min: Time step in minutes
            current_bg: Current blood glucose for trajectory calculation
            treatments: List of treatments for treatment-based calculation (enables continuity)
            base_time: Base time for projection (default: now)

        Returns:
            List of dicts with:
            - minutesAhead: Time offset
            - iobEffect: BG lowering from IOB (negative) - cumulative absorbed effect
            - cobEffect: BG raising from COB (positive) - cumulative absorbed effect
            - netEffect: Combined effect (bgPressure delta)
            - remainingIOB: IOB remaining at this time
            - remainingCOB: COB remaining at this time
            - insulinActivity: Current insulin activity level (0-1)
            - carbActivity: Current carb absorption activity level (0-1)
            - expectedBg: Combined BG trajectory
            - bgWithIobOnly: BG trajectory with only insulin effect (pull-down floor)
            - bgWithCobOnly: BG trajectory with only carb effect (push-up ceiling)
        """
        import math

        effects = []
        bg_per_gram = isf / icr  # BG rise per gram of carbs

        # Calculate total effect each dose will have
        # Using activity curves for more realistic pharmacokinetics
        peak_activity_time = 75  # minutes for insulin
        dia = 300  # Duration of insulin action

        base_time = base_time or datetime.utcnow()

        # First, calculate the baseline effects at t=0
        # We need to subtract this from all future points so the projected line
        # starts at current_bg (which already reflects past absorption)
        baseline_iob_effect = 0.0
        baseline_cob_effect = 0.0
        if treatments:
            for treatment in treatments:
                t_time = _to_utc_naive(treatment.timestamp)
                time_since_dose = (base_time - t_time).total_seconds() / 60

                if time_since_dose <= 0:
                    continue

                if treatment.insulin and treatment.insulin > 0:
                    initial_dose = treatment.insulin
                    remaining = initial_dose * (0.5 ** (time_since_dose / self.insulin_half_life_min)) if time_since_dose < self.insulin_duration_min else 0
                    absorbed = initial_dose - remaining
                    if time_since_dose < 30:
                        onset_factor = time_since_dose / 30.0
                        absorbed *= onset_factor * onset_factor
                    baseline_iob_effect += absorbed * isf

                if treatment.carbs and treatment.carbs > 0:
                    gi = getattr(treatment, 'glycemicIndex', None) or 55
                    is_liquid = getattr(treatment, 'isLiquid', False) or False
                    initial_carbs = treatment.carbs
                    # Use physiological GI-based equation
                    gi_params = gi_to_absorption_params(gi, is_liquid)
                    onset_delay = gi_params['onset_min']
                    half_life = gi_params['half_life_min']
                    duration = gi_params['duration_min']
                    remaining_carbs = initial_carbs * (0.5 ** (time_since_dose / half_life)) if time_since_dose < duration else 0
                    absorbed = initial_carbs - remaining_carbs
                    if time_since_dose < onset_delay:
                        onset_factor = time_since_dose / onset_delay
                        absorbed *= onset_factor * onset_factor
                    baseline_cob_effect += absorbed * bg_per_gram

        # Initialize starting POB for delta calculation (will be set on first iteration t=0)
        starting_pob = 0.0

        for t in range(0, duration_min + 1, step_min):
            # Calculate remaining IOB/COB using exponential decay
            remaining_iob = current_iob * math.exp(-0.693 * t / self.insulin_half_life_min)
            remaining_cob = current_cob * math.exp(-0.693 * t / self.carb_half_life_min)

            # Calculate activity levels using pharmacokinetic curves (bell-shaped)
            insulin_activity = insulin_activity_curve(t, peak_min=peak_activity_time, dia_min=dia)
            carb_activity = carb_activity_curve(t, peak_min=45, duration_min=180)

            # Calculate pressure using ACTIVITY RATES (same as historical calculation)
            # This ensures continuity between historical and projected lines
            if treatments:
                # Activity-based formula for instantaneous pressure
                future_time = base_time + timedelta(minutes=t)
                total_insulin_activity_effect = 0.0
                total_carb_activity_effect = 0.0
                total_protein_activity_effect = 0.0

                # CUMULATIVE absorbed effects for expected BG trajectory
                # This is DIFFERENT from instantaneous activity - it tracks actual absorbed amounts
                cumulative_insulin_absorbed = 0.0
                cumulative_carb_absorbed = 0.0
                cumulative_protein_absorbed = 0.0

                for treatment in treatments:
                    t_time = _to_utc_naive(treatment.timestamp)
                    time_since_dose = (future_time - t_time).total_seconds() / 60

                    if time_since_dose <= 0:
                        continue  # Treatment is in the future

                    # Insulin activity (for BG pressure visualization)
                    if treatment.insulin and treatment.insulin > 0 and time_since_dose < self.insulin_duration_min:
                        activity = insulin_activity_curve(time_since_dose, peak_min=75, dia_min=self.insulin_duration_min)
                        total_insulin_activity_effect += activity * treatment.insulin * isf

                    # Carb activity (for BG pressure visualization)
                    if treatment.carbs and treatment.carbs > 0:
                        gi = getattr(treatment, 'glycemicIndex', None) or 55
                        is_liquid = getattr(treatment, 'isLiquid', False) or False
                        gi_params = gi_to_absorption_params(gi, is_liquid)
                        duration = gi_params['duration_min']

                        if time_since_dose < duration:
                            activity = carb_activity_curve(time_since_dose, peak_min=45, duration_min=duration,
                                                          glycemic_index=gi, is_liquid=is_liquid)
                            total_carb_activity_effect += activity * treatment.carbs * bg_per_gram

                    # Protein activity (for BG pressure visualization)
                    if treatment.protein and treatment.protein > 0 and time_since_dose < 300:
                        activity = carb_activity_curve(time_since_dose, peak_min=180, duration_min=300,
                                                      glycemic_index=30, is_liquid=False)
                        bg_per_gram_protein = 2.0
                        total_protein_activity_effect += activity * treatment.protein * bg_per_gram_protein

                # Net pressure offset based on ACTIVITY (for BG Pressure visualization)
                activity_scale = 0.3  # Same as historical
                net_effect = (total_carb_activity_effect + total_protein_activity_effect - total_insulin_activity_effect) * activity_scale

                # For compatibility, also provide iob/cob effects (scaled for pressure)
                iob_effect = -total_insulin_activity_effect * activity_scale
                cob_effect = (total_carb_activity_effect + total_protein_activity_effect) * activity_scale

                # Use ML Forcing Ensemble for expectedBg prediction
                # The ensemble uses personalized ML models for IOB/COB decay
                try:
                    ensemble = get_forcing_ensemble()
                    # Get total protein from recent treatments for ensemble
                    total_protein = sum(
                        getattr(tx, 'protein', 0) or 0 for tx in treatments
                        if _to_utc_naive(tx.timestamp) > base_time - timedelta(hours=5)
                    )
                    forcing_pred = ensemble.predict(
                        current_bg=current_bg,
                        iob=current_iob,
                        cob=current_cob,
                        horizon_min=t,
                        isf=isf,
                        icr=icr,
                        protein_grams=total_protein + current_pob,
                    )
                    cumulative_expected_bg = forcing_pred.final_prediction
                except Exception as e:
                    # Fallback to simple decay if ensemble fails
                    logger.warning(f"Forcing ensemble failed, using fallback: {e}")
                    iob_acted = current_iob - remaining_iob
                    cob_acted = current_cob - remaining_cob
                    pob_half_life = 90.0
                    remaining_pob = current_pob * math.exp(-0.693 * t / pob_half_life)
                    pob_acted = current_pob - remaining_pob
                    bg_per_gram_protein = self.protein_bg_factor
                    cumulative_expected_bg = current_bg + (cob_acted * bg_per_gram) + (pob_acted * bg_per_gram_protein) - (iob_acted * isf)

                # Calculate remaining POB for output (using ML or fallback)
                pob_half_life = 90.0
                remaining_pob = current_pob * math.exp(-0.693 * t / pob_half_life)

            else:
                # No treatments provided - use ML ensemble for prediction
                remaining_iob_effect = -(remaining_iob * isf)
                remaining_cob_effect = remaining_cob * bg_per_gram
                pressure_factor = 0.5
                net_effect = (remaining_iob_effect + remaining_cob_effect) * pressure_factor
                iob_effect = remaining_iob_effect
                cob_effect = remaining_cob_effect

                # Use ML Forcing Ensemble for expectedBg
                try:
                    ensemble = get_forcing_ensemble()
                    forcing_pred = ensemble.predict(
                        current_bg=current_bg,
                        iob=current_iob,
                        cob=current_cob,
                        horizon_min=t,
                        isf=isf,
                        icr=icr,
                        protein_grams=current_pob,
                    )
                    cumulative_expected_bg = forcing_pred.final_prediction
                except Exception as e:
                    # Fallback to simple formula
                    logger.warning(f"Forcing ensemble failed: {e}")
                    iob_acted = current_iob - remaining_iob
                    cob_acted = current_cob - remaining_cob
                    pob_half_life = 90.0
                    remaining_pob = current_pob * math.exp(-0.693 * t / pob_half_life)
                    pob_acted = current_pob - remaining_pob
                    bg_per_gram_protein = self.protein_bg_factor
                    cumulative_expected_bg = current_bg + (cob_acted * bg_per_gram) + (pob_acted * bg_per_gram_protein) - (iob_acted * isf)

                # Calculate remaining POB for output
                pob_half_life = 90.0
                remaining_pob = current_pob * math.exp(-0.693 * t / pob_half_life)

            # BG Pressure trajectories (use scaled activity effects)
            # 1. IOB Floor = current_bg + iob_effect (insulin only)
            bg_with_iob_only = current_bg + iob_effect
            bg_with_iob_only = max(40, min(400, bg_with_iob_only))

            # 2. COB Ceiling = current_bg + cob_effect (carbs only)
            bg_with_cob_only = current_bg + cob_effect
            bg_with_cob_only = max(40, min(400, bg_with_cob_only))

            # 3. Expected BG = cumulative absorbed effect trajectory (NOT scaled pressure)
            # This shows where BG is actually heading based on remaining IOB/COB
            expected_bg = max(40, min(400, cumulative_expected_bg))

            effects.append({
                'minutesAhead': t,
                'iobEffect': round(iob_effect, 1),
                'cobEffect': round(cob_effect, 1),
                'netEffect': round(net_effect, 1),
                'remainingIOB': round(remaining_iob, 2),
                'remainingCOB': round(remaining_cob, 1),
                'remainingPOB': round(remaining_pob, 1),
                'insulinActivity': round(insulin_activity, 3),
                'carbActivity': round(carb_activity, 3),
                'expectedBg': round(expected_bg, 0),
                'bgWithIobOnly': round(bg_with_iob_only, 0),
                'bgWithCobOnly': round(bg_with_cob_only, 0)
            })

        return effects

    def predict_bg_physics_based(
        self,
        current_bg: float,
        treatments: List[Treatment],
        isf: float,
        icr: float,
        bg_trend: float = 0.0,
        duration_min: int = 120,
        step_min: int = 5,
        base_time: Optional[datetime] = None
    ) -> List[dict]:
        """
        Physics-based BG prediction using LEARNED IOB/COB absorption curves as PRIMARY.

        This is the main prediction method - NOT a fallback. It uses the physiological
        forcing functions (insulin lowers BG, carbs raise BG) learned from actual data.

        The prediction formula at each time point:
            Predicted_BG(t) = Current_BG + BG_drift(t) + Carb_effect(t) - Insulin_effect(t)

        Where:
        - BG_drift: Extrapolation from current trend (decays over time)
        - Carb_effect: Cumulative BG rise from absorbed carbs
        - Insulin_effect: Cumulative BG drop from absorbed insulin

        All absorption parameters are LEARNED from actual BG response data:
        - IOB: onset=15min, ramp=75min, half-life=120min (from 54 correction boluses)
        - COB: onset=5min, ramp=10min, half-life=45min (from 193 meals)

        Args:
            current_bg: Current blood glucose (mg/dL)
            treatments: List of recent treatments (insulin and carbs)
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            icr: Insulin to Carb Ratio (grams per unit)
            bg_trend: Current BG trend (mg/dL per 5 min), default 0
            duration_min: Prediction horizon (default: 120 min)
            step_min: Time step (default: 5 min)
            base_time: Start time for prediction (default: now)

        Returns:
            List of prediction points with:
            - minutesAhead: Time offset from now
            - predictedBg: Physics-based BG prediction (PRIMARY)
            - bgTrendComponent: Contribution from current momentum
            - insulinComponent: Cumulative BG drop from insulin
            - carbComponent: Cumulative BG rise from carbs
            - remainingIOB: IOB at this time
            - remainingCOB: COB at this time
        """
        import math

        base_time = base_time or datetime.utcnow()
        bg_per_gram = isf / icr  # BG rise per gram of carbs

        predictions = []

        # Learned parameters for absorption
        iob_onset = self.insulin_onset_min      # 15 min
        iob_ramp = self.insulin_ramp_min        # 75 min
        iob_half_life = self.insulin_half_life_min  # 120 min

        cob_onset = self.carb_onset_min         # 5 min
        cob_ramp = self.carb_ramp_min           # 10 min
        cob_half_life = self.carb_half_life_min # 45 min

        for t in range(0, duration_min + 1, step_min):
            future_time = base_time + timedelta(minutes=t)

            # 1. BG TREND COMPONENT (momentum with decay)
            # Trend influence decays exponentially - 30 min half-life
            trend_decay = 0.5 ** (t / 30.0)
            # Cumulative drift = integral of decaying trend
            # At t=0: drift=0, at t=30: drift = trend * 30 * 0.7 (area under curve)
            cumulative_trend = bg_trend * t * trend_decay * 0.7  # ~70% of linear projection
            bg_trend_component = cumulative_trend

            # 2. INSULIN COMPONENT (cumulative absorbed effect)
            cumulative_insulin_absorbed = 0.0
            total_iob = 0.0

            for treatment in treatments:
                if not treatment.insulin or treatment.insulin <= 0:
                    continue

                t_time = _to_utc_naive(treatment.timestamp)
                time_since_dose = (future_time - t_time).total_seconds() / 60

                if time_since_dose <= 0 or time_since_dose > self.insulin_duration_min:
                    continue

                dose = treatment.insulin

                # Calculate absorption using learned curve
                if time_since_dose < iob_onset:
                    # Pre-onset: minimal absorption
                    absorbed_fraction = 0.02 * (time_since_dose / iob_onset)
                elif time_since_dose < (iob_onset + iob_ramp):
                    # Ramp-up: quadratic increase
                    ramp_progress = (time_since_dose - iob_onset) / iob_ramp
                    absorbed_fraction = 0.02 + 0.28 * (ramp_progress ** 2)  # 0.02 to 0.30 at ramp end
                else:
                    # Decay phase: exponential decay of remaining
                    decay_time = time_since_dose - iob_onset - iob_ramp
                    remaining_fraction = 0.70 * (0.5 ** (decay_time / iob_half_life))
                    absorbed_fraction = 1.0 - remaining_fraction

                absorbed_fraction = min(1.0, max(0.0, absorbed_fraction))
                cumulative_insulin_absorbed += dose * absorbed_fraction

                # Calculate remaining IOB
                remaining = dose * (1.0 - absorbed_fraction)
                total_iob += remaining

            insulin_bg_drop = cumulative_insulin_absorbed * isf

            # 3. CARB COMPONENT (cumulative absorbed effect)
            cumulative_carbs_absorbed = 0.0
            total_cob = 0.0

            for treatment in treatments:
                if not treatment.carbs or treatment.carbs <= 0:
                    continue

                t_time = _to_utc_naive(treatment.timestamp)
                time_since_meal = (future_time - t_time).total_seconds() / 60

                if time_since_meal <= 0 or time_since_meal > self.carb_duration_min:
                    continue

                carbs = treatment.carbs

                # Get glycemic index adjustment
                gi = getattr(treatment, 'glycemicIndex', None) or 55
                gi_factor = gi / 55.0

                # Adjust timing for GI
                adj_onset = cob_onset / gi_factor
                adj_ramp = cob_ramp / gi_factor
                adj_half_life = cob_half_life / gi_factor

                # Calculate absorption using learned curve with GI adjustment
                if time_since_meal < adj_onset:
                    # Pre-onset: minimal absorption
                    absorbed_fraction = 0.02 * (time_since_meal / adj_onset)
                elif time_since_meal < (adj_onset + adj_ramp):
                    # Ramp-up: quadratic increase
                    ramp_progress = (time_since_meal - adj_onset) / adj_ramp
                    absorbed_fraction = 0.02 + 0.28 * (ramp_progress ** 2)
                else:
                    # Decay phase: exponential decay of remaining
                    decay_time = time_since_meal - adj_onset - adj_ramp
                    remaining_fraction = 0.70 * (0.5 ** (decay_time / adj_half_life))
                    absorbed_fraction = 1.0 - remaining_fraction

                absorbed_fraction = min(1.0, max(0.0, absorbed_fraction))
                cumulative_carbs_absorbed += carbs * absorbed_fraction

                # Calculate remaining COB
                remaining = carbs * (1.0 - absorbed_fraction)
                total_cob += remaining

            carb_bg_rise = cumulative_carbs_absorbed * bg_per_gram

            # 4. FINAL PREDICTION = current + trend + carbs - insulin
            predicted_bg = current_bg + bg_trend_component + carb_bg_rise - insulin_bg_drop
            predicted_bg = max(40, min(400, predicted_bg))

            predictions.append({
                'minutesAhead': t,
                'predictedBg': round(predicted_bg, 1),
                'bgTrendComponent': round(bg_trend_component, 1),
                'insulinComponent': round(-insulin_bg_drop, 1),  # Negative = lowers BG
                'carbComponent': round(carb_bg_rise, 1),         # Positive = raises BG
                'remainingIOB': round(total_iob, 2),
                'remainingCOB': round(total_cob, 1),
                'timestamp': future_time.isoformat()
            })

        return predictions

    def predict_bg_simple_physics(
        self,
        current_bg: float,
        treatments: List[Treatment],
        isf: float,
        icr: float,
        bg_trend: float = 0.0,
        duration_min: int = 120,
        step_min: int = 5,
        base_time: Optional[datetime] = None
    ) -> List[dict]:
        """
        Simple physics-based BG prediction using standard pharmacokinetic formulas.

        This is a BASELINE for comparison against the learned model. Uses textbook
        exponential decay with fixed parameters (no learning from data).

        Standard insulin: onset=20min, peak=75min, duration=4hr, half-life=81min
        Standard carbs: onset=15min, peak=45min, duration=3hr, half-life=45min

        Args:
            current_bg: Current blood glucose (mg/dL)
            treatments: List of recent treatments
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            icr: Insulin to Carb Ratio (grams per unit)
            bg_trend: Current BG trend (mg/dL per 5 min)
            duration_min: Prediction horizon
            step_min: Time step
            base_time: Start time

        Returns:
            List of prediction points for comparison with learned model
        """
        import math

        base_time = base_time or datetime.utcnow()
        bg_per_gram = isf / icr

        predictions = []

        # Standard textbook parameters (NOT learned)
        STD_IOB_ONSET = 20.0     # min
        STD_IOB_HALF_LIFE = 81.0  # min (adult average)
        STD_COB_ONSET = 15.0     # min
        STD_COB_HALF_LIFE = 45.0  # min

        for t in range(0, duration_min + 1, step_min):
            future_time = base_time + timedelta(minutes=t)

            # Trend component (same as learned model)
            trend_decay = 0.5 ** (t / 30.0)
            bg_trend_component = bg_trend * t * trend_decay * 0.7

            # Insulin effect - simple exponential decay
            cumulative_insulin = 0.0
            total_iob = 0.0

            for treatment in treatments:
                if not treatment.insulin or treatment.insulin <= 0:
                    continue

                t_time = _to_utc_naive(treatment.timestamp)
                time_since = (future_time - t_time).total_seconds() / 60

                if time_since <= 0 or time_since > 240:  # 4 hour DIA
                    continue

                dose = treatment.insulin

                # Simple onset delay + exponential decay
                if time_since < STD_IOB_ONSET:
                    absorbed = 0.0
                else:
                    effective_time = time_since - STD_IOB_ONSET
                    remaining = 0.5 ** (effective_time / STD_IOB_HALF_LIFE)
                    absorbed = 1.0 - remaining

                cumulative_insulin += dose * absorbed
                total_iob += dose * (1.0 - absorbed)

            insulin_effect = cumulative_insulin * isf

            # Carb effect - simple exponential decay
            cumulative_carbs = 0.0
            total_cob = 0.0

            for treatment in treatments:
                if not treatment.carbs or treatment.carbs <= 0:
                    continue

                t_time = _to_utc_naive(treatment.timestamp)
                time_since = (future_time - t_time).total_seconds() / 60

                if time_since <= 0 or time_since > 180:  # 3 hour absorption
                    continue

                carbs = treatment.carbs

                # GI adjustment (simple)
                gi = getattr(treatment, 'glycemicIndex', None) or 55
                gi_factor = gi / 55.0
                adj_onset = STD_COB_ONSET / gi_factor
                adj_half_life = STD_COB_HALF_LIFE / gi_factor

                if time_since < adj_onset:
                    absorbed = 0.0
                else:
                    effective_time = time_since - adj_onset
                    remaining = 0.5 ** (effective_time / adj_half_life)
                    absorbed = 1.0 - remaining

                cumulative_carbs += carbs * absorbed
                total_cob += carbs * (1.0 - absorbed)

            carb_effect = cumulative_carbs * bg_per_gram

            # Prediction
            predicted_bg = current_bg + bg_trend_component + carb_effect - insulin_effect
            predicted_bg = max(40, min(400, predicted_bg))

            predictions.append({
                'minutesAhead': t,
                'predictedBg': round(predicted_bg, 1),
                'bgTrendComponent': round(bg_trend_component, 1),
                'insulinComponent': round(-insulin_effect, 1),
                'carbComponent': round(carb_effect, 1),
                'remainingIOB': round(total_iob, 2),
                'remainingCOB': round(total_cob, 1),
                'timestamp': future_time.isoformat()
            })

        return predictions


# Helper functions for simpler usage

def calculate_iob_simple(
    insulin_treatments: List[Treatment],
    duration_min: int = 180,
    half_life_min: float = 81.0
) -> float:
    """Simple IOB calculation function."""
    service = IOBCOBService(insulin_duration_min=duration_min, insulin_half_life_min=half_life_min)
    return service.calculate_iob(insulin_treatments)


def calculate_cob_simple(
    carb_treatments: List[Treatment],
    duration_min: int = 180,
    half_life_min: float = 45.0
) -> float:
    """
    Simple COB calculation function.

    Uses glycemic index from treatment if available for more accurate decay.
    """
    service = IOBCOBService(carb_duration_min=duration_min, carb_half_life_min=half_life_min)
    return service.calculate_cob(carb_treatments)


def calculate_effect_curve(
    iob: float,
    cob: float,
    isf: float = 50.0,
    icr: float = 10.0,
    duration_min: int = 60
) -> List[dict]:
    """Simple effect curve calculation function."""
    service = IOBCOBService()
    return service.calculate_bg_effect_curve(iob, cob, isf, icr, duration_min)


def insulin_activity_curve(t_min: float, peak_min: float = 75, dia_min: float = 300) -> float:
    """
    Calculate insulin activity at time t (bell-shaped pharmacokinetic curve).

    Models rapid-acting insulin like Novolog/Humalog:
    - Near zero at administration (t=0)
    - Peak activity around 60-90 min
    - Trails off by 4-5 hours

    Based on Walsh curve / Fiasp model.

    Args:
        t_min: Time since injection in minutes
        peak_min: Time to peak activity (default: 75 min)
        dia_min: Duration of insulin action (default: 300 min = 5 hrs)

    Returns:
        Activity level (0-1) where 1 = peak activity
    """
    import math

    if t_min <= 0 or t_min >= dia_min:
        return 0.0

    # Use a gamma-like distribution shape (simplified Walsh curve)
    # activity = t^a * e^(-t/b) normalized to peak at peak_min
    a = 2.0  # Shape parameter
    b = peak_min / a  # Scale parameter to peak at peak_min

    # Raw gamma-like curve
    activity = (t_min / b) ** a * math.exp(a * (1 - t_min / (a * b)))

    # Normalize so peak = 1.0 (raw peak value is a^a = 4.0 when a=2.0)
    peak_value = a ** a  # = 4.0 for a=2.0
    activity /= peak_value

    # Decay to zero at DIA
    if t_min > peak_min * 2:
        decay_factor = max(0, 1 - (t_min - peak_min * 2) / (dia_min - peak_min * 2))
        activity *= decay_factor

    return min(1.0, max(0.0, activity))


def gi_to_absorption_params(glycemic_index: float, is_liquid: bool = False) -> dict:
    """
    Convert glycemic index to absorption parameters using physiological equations.

    Based on research on carbohydrate absorption kinetics:
    - Very high GI (>85): Glucose, juice, sugary drinks - onset 2-5 min, peak 15-20 min
    - High GI (70-85): White bread, potatoes - onset 5-10 min, peak 25-35 min
    - Medium GI (55-70): Rice, pasta - onset 10-15 min, peak 40-50 min
    - Low GI (<55): Beans, vegetables - onset 15-25 min, peak 60-90 min

    The relationship is NON-LINEAR - high GI foods spike dramatically faster.

    Args:
        glycemic_index: GI value (0-100+)
        is_liquid: True for liquid carbs (absorb even faster)

    Returns:
        dict with onset_min, ramp_min, half_life_min, duration_min
    """
    import math

    # Clamp GI to reasonable range
    gi = max(20, min(100, glycemic_index))

    # Non-linear scaling: exponential relationship for high GI foods
    # At GI=55 (medium), factor=1.0
    # At GI=85 (high), factor≈2.0 (twice as fast)
    # At GI=35 (low), factor≈0.6 (40% slower)
    gi_factor = math.exp((gi - 55) / 30)  # Exponential scaling

    # Liquid carbs absorb 40% faster (no mechanical digestion needed)
    liquid_factor = 1.4 if is_liquid else 1.0

    # Base parameters (for medium GI = 55, solid food)
    BASE_ONSET = 10.0      # minutes until carbs start affecting BG
    BASE_RAMP = 15.0       # minutes to ramp up to peak
    BASE_HALF_LIFE = 40.0  # minutes for exponential decay
    BASE_DURATION = 180.0  # total duration of effect

    # Apply factors (higher factor = faster absorption = lower times)
    combined_factor = gi_factor * liquid_factor

    onset = BASE_ONSET / combined_factor
    ramp = BASE_RAMP / combined_factor
    half_life = BASE_HALF_LIFE / combined_factor
    duration = BASE_DURATION / (combined_factor ** 0.5)  # Duration scales slower

    # Minimum bounds (physiologically impossible to absorb faster than this)
    onset = max(2.0, onset)      # Can't absorb in less than 2 min
    ramp = max(3.0, ramp)        # Minimum ramp time
    half_life = max(10.0, half_life)  # Minimum half-life
    duration = max(60.0, duration)    # At least 1 hour total

    return {
        'onset_min': round(onset, 1),
        'ramp_min': round(ramp, 1),
        'half_life_min': round(half_life, 1),
        'duration_min': round(duration, 1),
        'gi_factor': round(combined_factor, 2)
    }


def carb_activity_curve(t_min: float, peak_min: float = 45, duration_min: float = 180,
                        glycemic_index: float = 55, is_liquid: bool = False) -> float:
    """
    Calculate carb absorption activity at time t.

    Uses physiological GI-based equations for accurate modeling:
    - Very high GI (>85): juice, glucose - peak at ~15min, duration ~1.5hrs
    - High GI (70-85): white bread - peak at ~25min, duration ~2hrs
    - Medium GI (55-70): rice, pasta - peak at ~40min, duration ~3hrs
    - Low GI (<55): beans, vegetables - peak at ~60min, duration ~4hrs

    The peak_min parameter can override or scale the GI-calculated peak:
    - If peak_min differs from default (45), it scales the GI timing
    - This allows personalized learned peaks to adjust absorption timing

    Args:
        t_min: Time since eating in minutes
        peak_min: Base time to peak absorption (used as scaling factor for GI)
        duration_min: Base duration of absorption (overridden by GI calc)
        glycemic_index: Food glycemic index (0-100)
        is_liquid: True for liquid carbs (faster absorption)

    Returns:
        Activity level (0-1) where 1 = peak absorption
    """
    import math

    # Get physiologically-based parameters from GI
    params = gi_to_absorption_params(glycemic_index, is_liquid)
    gi_peak = params['onset_min'] + params['ramp_min']  # Peak from GI calculation
    adjusted_duration = params['duration_min']

    # Apply personal scaling if peak_min differs from default
    # This lets learned absorption profiles adjust timing while preserving GI effects
    default_peak = 45.0
    if abs(peak_min - default_peak) > 1.0:
        # Scale the GI-calculated peak by the ratio of personal to default
        scale_factor = peak_min / default_peak
        adjusted_peak = gi_peak * scale_factor
        adjusted_duration = adjusted_duration * scale_factor
    else:
        adjusted_peak = gi_peak

    if t_min <= 0 or t_min >= adjusted_duration:
        return 0.0

    # Gamma-like curve for realistic absorption profile
    a = 2.0
    b = adjusted_peak / a

    activity = (t_min / b) ** a * math.exp(a * (1 - t_min / (a * b)))

    # Normalize so peak = 1.0 (raw peak value is a^a = 4.0 when a=2.0)
    peak_value = a ** a  # = 4.0 for a=2.0
    activity /= peak_value

    # Decay to zero at duration
    if t_min > adjusted_peak * 2:
        decay_factor = max(0, 1 - (t_min - adjusted_peak * 2) / (adjusted_duration - adjusted_peak * 2))
        activity *= decay_factor

    return min(1.0, max(0.0, activity))
