"""
Metabolic Parameters Service for T1D-AI

Provides effective ISF, ICR, and PIR values by combining:
- Long-term learned baselines from CosmosDB
- Short-term deviations (illness/sensitivity detection)
- User settings defaults as fallback

Key concept: ISF measures insulin EFFECT, not KINETICS:
- When sick (insulin resistant): ISF drops (e.g., 50 -> 35), meaning 1U drops less BG
- This affects dose calculations but NOT the IOB decay curve
- IOB curve shape stays the same (half-life is kinetic), only the EFFECT changes

Usage in predictions:
- Use effective_isf for all BG predictions and dose calculations
- Use baseline_isf for model training (to detect deviation)
- Deviation percentage indicates metabolic state (sick/normal/sensitive)
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from database.repositories import (
    LearnedISFRepository,
    LearnedICRRepository,
    LearnedPIRRepository,
)

logger = logging.getLogger(__name__)


class MetabolicState(str, Enum):
    """Overall metabolic state based on parameter deviations."""
    SICK = "sick"               # ISF < -15% AND (slow absorption OR other indicators)
    RESISTANT = "resistant"     # ISF < -10%
    NORMAL = "normal"           # ISF within ±10%
    SENSITIVE = "sensitive"     # ISF > +10%
    VERY_SENSITIVE = "very_sensitive"  # ISF > +20%


@dataclass
class EffectiveISF:
    """Effective ISF with deviation tracking."""
    value: float  # The effective ISF to use in calculations
    baseline: float  # Long-term learned baseline
    deviation_percent: float  # How much current differs from baseline (-50 to +50)
    confidence: float  # Confidence in this value (0-1)
    source: str  # "learned", "short_term", "default"
    is_resistant: bool  # True if < -10% deviation
    is_sick: bool  # True if < -15% deviation


@dataclass
class EffectiveICR:
    """Effective ICR with deviation tracking."""
    value: float  # The effective ICR to use in calculations
    baseline: float  # Long-term learned baseline
    deviation_percent: float  # How much current differs from baseline
    confidence: float
    source: str  # "learned", "short_term", "default", "meal_specific"
    meal_type: Optional[str] = None  # "breakfast", "lunch", "dinner" if meal-specific


@dataclass
class EffectivePIR:
    """Effective PIR with timing info."""
    value: float  # The effective PIR to use in calculations
    baseline: float  # Long-term learned baseline
    deviation_percent: float
    confidence: float
    source: str
    onset_minutes: int = 120  # When protein starts affecting BG
    peak_minutes: int = 210  # When protein effect peaks


@dataclass
class EffectiveAbsorption:
    """Effective carb absorption rate with deviation tracking."""
    time_to_peak: float  # Current time-to-peak in minutes
    baseline_time_to_peak: float  # Long-term baseline
    deviation_percent: float  # How much current differs from baseline
    confidence: float
    state: str  # "very_slow", "slow", "normal", "fast", "very_fast"
    state_description: str  # Human-readable description
    is_slow: bool = False  # True if >20% slower (potential gastroparesis)
    is_very_slow: bool = False  # True if >50% slower (likely illness)


@dataclass
class MetabolicParams:
    """Complete set of metabolic parameters."""
    isf: EffectiveISF
    icr: EffectiveICR
    pir: EffectivePIR
    absorption: Optional[EffectiveAbsorption]  # May be None if insufficient data
    metabolic_state: MetabolicState
    state_description: str  # Human-readable state description


class MetabolicParamsService:
    """
    Service for getting effective metabolic parameters.

    Combines learned baselines with short-term deviations to provide
    the most accurate parameters for predictions and dose calculations.
    """

    # Default values when no learned data is available
    DEFAULT_ISF = 50.0  # mg/dL per unit
    DEFAULT_ICR = 10.0  # grams per unit
    DEFAULT_PIR = 25.0  # grams per unit

    def __init__(self):
        self._isf_repo: Optional[LearnedISFRepository] = None
        self._icr_repo: Optional[LearnedICRRepository] = None
        self._pir_repo: Optional[LearnedPIRRepository] = None

    @property
    def isf_repo(self) -> LearnedISFRepository:
        if self._isf_repo is None:
            self._isf_repo = LearnedISFRepository()
        return self._isf_repo

    @property
    def icr_repo(self) -> LearnedICRRepository:
        if self._icr_repo is None:
            self._icr_repo = LearnedICRRepository()
        return self._icr_repo

    @property
    def pir_repo(self) -> LearnedPIRRepository:
        if self._pir_repo is None:
            self._pir_repo = LearnedPIRRepository()
        return self._pir_repo

    async def get_effective_isf(
        self,
        user_id: str,
        is_fasting: bool = True,
        include_short_term: bool = True,
        time_of_day: Optional[str] = None
    ) -> EffectiveISF:
        """
        Get effective ISF considering both long-term baseline and short-term deviation.

        The effective ISF is what should be used for ALL BG predictions and dose calculations.
        When a user is sick/resistant, the effective ISF will be LOWER than baseline,
        meaning insulin has LESS effect, so they need MORE insulin.

        Args:
            user_id: User ID to get ISF for
            is_fasting: Whether to prefer fasting or meal ISF
            include_short_term: Whether to include short-term illness detection
            time_of_day: Optional time period for time-specific ISF

        Returns:
            EffectiveISF with value, baseline, deviation, and source info
        """
        isf_type = "fasting" if is_fasting else "meal"

        # Get long-term learned baseline
        learned = await self.isf_repo.get(user_id, isf_type)
        if not learned:
            # Try the other type
            other_type = "meal" if is_fasting else "fasting"
            learned = await self.isf_repo.get(user_id, other_type)

        baseline = learned.value if learned else self.DEFAULT_ISF
        confidence = learned.confidence if learned else 0.0
        source = "learned" if learned else "default"

        # Check for time-of-day specific value
        if learned and time_of_day and learned.timeOfDayPattern.get(time_of_day):
            tod_value = learned.timeOfDayPattern[time_of_day]
            if tod_value:
                baseline = tod_value
                source = f"learned_tod_{time_of_day}"

        effective_value = baseline
        deviation_percent = 0.0

        # Calculate short-term deviation if requested
        if include_short_term:
            try:
                from ml.training.isf_learner import ISFLearner
                learner = ISFLearner()
                short_term = await learner.calculate_short_term_isf(user_id, days=3)

                if short_term.get("current_isf") and short_term.get("confidence", 0) > 0.3:
                    current_isf = short_term["current_isf"]
                    deviation_percent = ((current_isf - baseline) / baseline) * 100

                    # Use short-term if deviation is significant (>10%)
                    if abs(deviation_percent) > 10:
                        effective_value = current_isf
                        source = "short_term"
                        # Blend confidence
                        confidence = (confidence + short_term["confidence"]) / 2

                        logger.info(
                            f"ISF deviation for user {user_id}: {deviation_percent:.1f}% "
                            f"(baseline={baseline:.1f}, current={current_isf:.1f})"
                        )
            except Exception as e:
                logger.warning(f"Failed to calculate short-term ISF: {e}")

        return EffectiveISF(
            value=effective_value,
            baseline=baseline,
            deviation_percent=round(deviation_percent, 1),
            confidence=confidence,
            source=source,
            is_resistant=deviation_percent < -10,
            is_sick=deviation_percent < -15
        )

    async def get_effective_icr(
        self,
        user_id: str,
        meal_type: Optional[str] = None,
        include_short_term: bool = True
    ) -> EffectiveICR:
        """
        Get effective ICR (Insulin-to-Carb Ratio) for a user.

        ICR = grams of carbs covered by 1 unit of insulin.
        Lower ICR = more insulin needed per gram.

        Args:
            user_id: User ID
            meal_type: Optional specific meal type (breakfast, lunch, dinner)
            include_short_term: Whether to include short-term deviation

        Returns:
            EffectiveICR with value and source info
        """
        # Determine meal type from time if not provided
        if not meal_type:
            hour = datetime.now(timezone.utc).hour
            if 5 <= hour < 10:
                meal_type = "breakfast"
            elif 10 <= hour < 15:
                meal_type = "lunch"
            else:
                meal_type = "dinner"

        # Try meal-specific first, then overall
        learned = await self.icr_repo.get(user_id, meal_type)
        if not learned:
            learned = await self.icr_repo.get(user_id, "overall")

        baseline = learned.value if learned else self.DEFAULT_ICR
        confidence = learned.confidence if learned else 0.0
        source = f"learned_{meal_type}" if learned else "default"

        effective_value = baseline
        deviation_percent = 0.0

        # Calculate short-term ICR deviation if requested
        if include_short_term:
            try:
                from ml.training.icr_learner import ICRLearner
                learner = ICRLearner()
                short_term = await learner.calculate_short_term_icr(user_id, days=3)

                if short_term.get("current_icr") and short_term.get("confidence", 0) > 0.3:
                    current_icr = short_term["current_icr"]
                    deviation_percent = short_term.get("deviation_percent", 0.0)

                    # Use short-term if deviation is significant (>10%)
                    if abs(deviation_percent) > 10:
                        effective_value = current_icr
                        source = "short_term"
                        # Blend confidence
                        confidence = (confidence + short_term["confidence"]) / 2

                        logger.info(
                            f"ICR deviation for user {user_id}: {deviation_percent:.1f}% "
                            f"(baseline={baseline:.1f}, current={current_icr:.1f})"
                        )
            except Exception as e:
                logger.warning(f"Failed to calculate short-term ICR: {e}")

        return EffectiveICR(
            value=effective_value,
            baseline=baseline,
            deviation_percent=round(deviation_percent, 1),
            confidence=confidence,
            source=source,
            meal_type=meal_type if learned else None
        )

    async def get_effective_pir(
        self,
        user_id: str,
        meal_type: Optional[str] = None
    ) -> EffectivePIR:
        """
        Get effective PIR (Protein-to-Insulin Ratio) for a user.

        PIR = grams of protein covered by 1 unit of insulin.
        Typically ~2x ICR (protein has ~50% the BG impact of carbs).

        Args:
            user_id: User ID
            meal_type: Optional specific meal type

        Returns:
            EffectivePIR with value and timing info
        """
        # Try meal-specific first, then overall
        learned = None
        if meal_type:
            learned = await self.pir_repo.get(user_id, meal_type)
        if not learned:
            learned = await self.pir_repo.get(user_id, "overall")

        baseline = learned.value if learned else self.DEFAULT_PIR
        confidence = learned.confidence if learned else 0.0
        source = "learned" if learned else "default"

        # Get timing info from learned data or use defaults
        onset_minutes = 120
        peak_minutes = 210

        if learned:
            if learned.proteinOnsetMin:
                onset_minutes = int(learned.proteinOnsetMin)
            if learned.proteinPeakMin:
                peak_minutes = int(learned.proteinPeakMin)

        return EffectivePIR(
            value=baseline,
            baseline=baseline,
            deviation_percent=0.0,
            confidence=confidence,
            source=source,
            onset_minutes=onset_minutes,
            peak_minutes=peak_minutes
        )

    async def get_effective_absorption(
        self,
        user_id: str,
        days: int = 3
    ) -> Optional[EffectiveAbsorption]:
        """
        Get effective carb absorption rate for a user.

        Analyzes recent meals to detect changes in absorption speed.
        Returns None if insufficient data.

        Args:
            user_id: User ID
            days: Number of days to analyze (default 3)

        Returns:
            EffectiveAbsorption with current state and deviation
        """
        try:
            from ml.training.absorption_learner import AbsorptionLearner, AbsorptionState
            learner = AbsorptionLearner()
            profile = await learner.analyze_short_term_absorption(user_id, days)

            if profile.sample_count == 0:
                return None

            return EffectiveAbsorption(
                time_to_peak=profile.current_time_to_peak,
                baseline_time_to_peak=profile.baseline_time_to_peak,
                deviation_percent=profile.deviation_percent,
                confidence=profile.confidence,
                state=profile.state.value,
                state_description=profile.state_description,
                is_slow=profile.state in [AbsorptionState.SLOW, AbsorptionState.VERY_SLOW],
                is_very_slow=profile.state == AbsorptionState.VERY_SLOW
            )
        except Exception as e:
            logger.warning(f"Failed to get absorption state: {e}")
            return None

    async def get_all_params(
        self,
        user_id: str,
        is_fasting: bool = False,
        meal_type: Optional[str] = None,
        include_short_term: bool = True
    ) -> MetabolicParams:
        """
        Get all metabolic parameters and overall metabolic state.

        This is the main method for getting complete metabolic context.

        Args:
            user_id: User ID
            is_fasting: Whether this is a fasting context
            meal_type: Optional meal type for ICR/PIR
            include_short_term: Whether to include illness detection

        Returns:
            MetabolicParams with all parameters and metabolic state
        """
        isf = await self.get_effective_isf(user_id, is_fasting, include_short_term)
        icr = await self.get_effective_icr(user_id, meal_type, include_short_term)
        pir = await self.get_effective_pir(user_id, meal_type)

        # Get absorption state if we have recent meal data
        absorption = None
        if include_short_term:
            absorption = await self.get_effective_absorption(user_id)

        # Determine overall metabolic state (considers ISF + absorption)
        state = self._determine_metabolic_state(isf, absorption)
        description = self._get_state_description(state, isf, absorption)

        return MetabolicParams(
            isf=isf,
            icr=icr,
            pir=pir,
            absorption=absorption,
            metabolic_state=state,
            state_description=description
        )

    def _determine_metabolic_state(
        self,
        isf: EffectiveISF,
        absorption: Optional[EffectiveAbsorption] = None
    ) -> MetabolicState:
        """
        Determine overall metabolic state from ISF deviation and absorption.

        SICK is indicated by:
        - ISF < -15% deviation (strong insulin resistance), OR
        - ISF < -10% AND slow absorption (combined indicators)

        Args:
            isf: Effective ISF with deviation info
            absorption: Optional absorption state

        Returns:
            MetabolicState classification
        """
        deviation = isf.deviation_percent
        has_slow_absorption = absorption and absorption.is_slow

        # SICK: Strong ISF resistance, OR moderate resistance + slow absorption
        if deviation < -15:
            return MetabolicState.SICK
        elif deviation < -10 and has_slow_absorption:
            # Moderate resistance combined with slow absorption suggests illness
            return MetabolicState.SICK
        elif deviation < -10:
            return MetabolicState.RESISTANT
        elif deviation > 20:
            return MetabolicState.VERY_SENSITIVE
        elif deviation > 10:
            return MetabolicState.SENSITIVE
        else:
            return MetabolicState.NORMAL

    def _get_state_description(
        self,
        state: MetabolicState,
        isf: EffectiveISF,
        absorption: Optional[EffectiveAbsorption] = None
    ) -> str:
        """Get human-readable description of metabolic state."""
        deviation = isf.deviation_percent
        parts = []

        if state == MetabolicState.SICK:
            parts.append(f"Possibly sick or stressed: ISF is {abs(deviation):.0f}% lower than baseline")
            if absorption and absorption.is_slow:
                parts.append(f"and carb absorption is {abs(absorption.deviation_percent):.0f}% slower")
            parts.append("May need significantly more insulin.")
            return " ".join(parts)
        elif state == MetabolicState.RESISTANT:
            base = f"Mild insulin resistance: ISF is {abs(deviation):.0f}% lower"
            if absorption and absorption.is_slow:
                return f"{base}, with slower carb absorption. May need more insulin than usual."
            return f"{base}. May need more insulin than usual."
        elif state == MetabolicState.VERY_SENSITIVE:
            return f"Very sensitive to insulin: ISF is {deviation:.0f}% higher. Be cautious with insulin doses."
        elif state == MetabolicState.SENSITIVE:
            return f"More sensitive than usual: ISF is {deviation:.0f}% higher. May need less insulin."
        else:
            if absorption and absorption.is_slow:
                return f"Metabolic state is normal, but carb absorption is slower than usual ({absorption.deviation_percent:.0f}%)."
            return "Metabolic state is normal."


# Singleton instance
_metabolic_params_service: Optional[MetabolicParamsService] = None


def get_metabolic_params_service() -> MetabolicParamsService:
    """Get or create the metabolic params service singleton."""
    global _metabolic_params_service
    if _metabolic_params_service is None:
        _metabolic_params_service = MetabolicParamsService()
    return _metabolic_params_service


# Convenience functions for common use cases
async def get_effective_isf_for_user(
    user_id: str,
    is_fasting: bool = True,
    include_short_term: bool = True
) -> Tuple[float, float]:
    """
    Convenience function to get effective ISF and deviation for a user.

    Returns:
        Tuple of (effective_isf, deviation_percent)
    """
    service = get_metabolic_params_service()
    isf = await service.get_effective_isf(user_id, is_fasting, include_short_term)
    return isf.value, isf.deviation_percent


async def get_effective_icr_for_user(
    user_id: str,
    meal_type: Optional[str] = None
) -> float:
    """Convenience function to get effective ICR for a user."""
    service = get_metabolic_params_service()
    icr = await service.get_effective_icr(user_id, meal_type)
    return icr.value


async def get_effective_pir_for_user(
    user_id: str,
    meal_type: Optional[str] = None
) -> Tuple[float, int, int]:
    """
    Convenience function to get effective PIR for a user.

    Returns:
        Tuple of (pir_value, onset_minutes, peak_minutes)
    """
    service = get_metabolic_params_service()
    pir = await service.get_effective_pir(user_id, meal_type)
    return pir.value, pir.onset_minutes, pir.peak_minutes
