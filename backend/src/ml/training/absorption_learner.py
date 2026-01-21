"""
Absorption Rate Learner for T1D-AI

Detects short-term changes in carbohydrate absorption rate by analyzing:
1. Time-to-peak: How long after eating until BG peaks
2. Rise rate: How fast BG rises per minute
3. Absorption profile shape: Steep vs gradual rise

This helps detect:
- Gastroparesis (slow gastric emptying during illness)
- Fast absorption (liquids, high-GI foods)
- Normal absorption patterns

Key insight: Absorption RATE is different from absorption AMOUNT (ICR):
- ICR tells us how much insulin to give
- Absorption rate tells us WHEN the carbs will hit
- Illness can slow absorption without changing total carb impact
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from database.repositories import GlucoseRepository, TreatmentRepository, LearnedICRRepository
from models.schemas import GlucoseReading, Treatment
from ml.training.treatment_anchored_selector import (
    TreatmentAnchoredSelector, TreatmentType, AnchoredTreatmentMoment
)

logger = logging.getLogger(__name__)


class AbsorptionState(str, Enum):
    """Absorption rate state classification."""
    VERY_SLOW = "very_slow"  # >50% slower than baseline (gastroparesis)
    SLOW = "slow"            # 20-50% slower
    NORMAL = "normal"        # Within ±20% of baseline
    FAST = "fast"            # 20-50% faster
    VERY_FAST = "very_fast"  # >50% faster (liquids, high-GI)


@dataclass
class AbsorptionMetrics:
    """Metrics from a single meal absorption event."""
    timestamp: datetime
    time_to_peak_minutes: float  # Minutes from meal to BG peak
    peak_bg: float              # BG value at peak
    bg_at_meal: float           # BG when meal was eaten
    rise_rate: float            # Average BG rise rate (mg/dL per minute)
    max_rise_rate: float        # Maximum 5-min rise rate
    carbs_grams: float
    meal_type: str
    confidence: float


@dataclass
class AbsorptionProfile:
    """Short-term absorption profile analysis."""
    current_time_to_peak: float  # Current average time-to-peak
    baseline_time_to_peak: float  # Long-term baseline
    deviation_percent: float     # How much current differs from baseline
    state: AbsorptionState
    rise_rate_deviation: float   # Deviation in rise rate
    sample_count: int
    confidence: float
    recent_meals: List[AbsorptionMetrics]
    state_description: str


class AbsorptionLearner:
    """
    Service to analyze and detect changes in carbohydrate absorption rate.

    Uses meal moments to track:
    - Time-to-peak: When BG reaches maximum after eating
    - Rise rate: How steep the BG rise is
    - Pattern changes: Detecting slower/faster absorption vs baseline
    """

    # Default baseline values (can be personalized)
    DEFAULT_TIME_TO_PEAK = 60.0  # Minutes - typical for mixed meals
    DEFAULT_RISE_RATE = 1.5     # mg/dL per minute

    # Configuration
    MIN_CARBS = 15              # Minimum carbs to analyze
    MIN_BG_RISE = 20            # Minimum BG rise to count
    MAX_LOOKBACK_MINUTES = 180  # How long to look for peak after meal

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.icr_repo = LearnedICRRepository()
        self.selector = TreatmentAnchoredSelector()

    async def analyze_short_term_absorption(
        self,
        user_id: str,
        days: int = 3
    ) -> AbsorptionProfile:
        """
        Analyze short-term absorption patterns from recent meals.

        This helps detect:
        - Gastroparesis (illness-induced slow gastric emptying)
        - Hormonal changes affecting absorption
        - Unusually fast absorption (stress, liquids)

        Args:
            user_id: User ID to analyze
            days: Number of days to analyze (default 3)

        Returns:
            AbsorptionProfile with current state and deviation from baseline
        """
        # Get long-term baseline
        baseline = await self._get_baseline_absorption(user_id)

        # Get recent meal moments - don't require strict isolation for absorption analysis
        # since we're just measuring BG rise patterns, not insulin sensitivity
        all_moments = await self.selector.get_anchored_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS, TreatmentType.CARBS_ONLY],
            require_isolated=False  # Absorption analysis doesn't need strict isolation
        )
        # Filter for explainable moments with reasonable confidence
        moments = [m for m in all_moments if m.is_explainable and m.confidence >= 0.3]

        logger.info(f"Absorption analysis for user {user_id}: found {len(moments) if moments else 0} meal moments in last {days} days")

        if not moments:
            logger.info(f"No recent meal data for absorption analysis (user {user_id}) - check if meals have carbs logged")
            return AbsorptionProfile(
                current_time_to_peak=baseline["time_to_peak"],
                baseline_time_to_peak=baseline["time_to_peak"],
                deviation_percent=0.0,
                state=AbsorptionState.NORMAL,
                rise_rate_deviation=0.0,
                sample_count=0,
                confidence=0.0,
                recent_meals=[],
                state_description="Insufficient data for absorption analysis."
            )

        # Analyze each meal
        meal_metrics = []
        skipped_low_carbs = 0
        skipped_analysis_failed = 0
        for moment in moments:
            if moment.carbs_grams < self.MIN_CARBS:
                skipped_low_carbs += 1
                continue

            metrics = await self._analyze_meal_absorption(user_id, moment)
            if metrics:
                meal_metrics.append(metrics)
            else:
                skipped_analysis_failed += 1

        logger.info(
            f"Absorption analysis for user {user_id}: "
            f"{len(meal_metrics)} valid meals, "
            f"{skipped_low_carbs} skipped (low carbs), "
            f"{skipped_analysis_failed} skipped (analysis failed)"
        )

        if not meal_metrics:
            return AbsorptionProfile(
                current_time_to_peak=baseline["time_to_peak"],
                baseline_time_to_peak=baseline["time_to_peak"],
                deviation_percent=0.0,
                state=AbsorptionState.NORMAL,
                rise_rate_deviation=0.0,
                sample_count=0,
                confidence=0.0,
                recent_meals=[],
                state_description="No valid meal absorption data in recent period."
            )

        # Calculate weighted averages (weight by recency and confidence)
        weights = []
        time_to_peaks = []
        rise_rates = []

        for m in meal_metrics:
            hours_ago = (datetime.now(timezone.utc) - m.timestamp).total_seconds() / 3600
            recency_weight = max(0.3, 1.0 - (hours_ago / (days * 24)))
            weight = m.confidence * recency_weight

            weights.append(weight)
            time_to_peaks.append(m.time_to_peak_minutes)
            rise_rates.append(m.rise_rate)

        weights = np.array(weights)
        weights = weights / weights.sum()

        current_ttp = float(np.average(time_to_peaks, weights=weights))
        current_rise_rate = float(np.average(rise_rates, weights=weights))

        # Calculate deviations
        ttp_deviation = ((current_ttp - baseline["time_to_peak"]) / baseline["time_to_peak"]) * 100
        rise_deviation = ((current_rise_rate - baseline["rise_rate"]) / baseline["rise_rate"]) * 100 if baseline["rise_rate"] > 0 else 0

        # Determine state
        state = self._classify_absorption_state(ttp_deviation)
        description = self._get_state_description(state, ttp_deviation, current_ttp)

        confidence = min(1.0, len(meal_metrics) / 4)  # Full confidence at 4+ meals

        logger.info(
            f"Short-term absorption for user {user_id}: "
            f"time-to-peak={current_ttp:.0f}min (baseline={baseline['time_to_peak']:.0f}min, "
            f"deviation={ttp_deviation:.1f}%), state={state.value}"
        )

        return AbsorptionProfile(
            current_time_to_peak=round(current_ttp, 1),
            baseline_time_to_peak=baseline["time_to_peak"],
            deviation_percent=round(ttp_deviation, 1),
            state=state,
            rise_rate_deviation=round(rise_deviation, 1),
            sample_count=len(meal_metrics),
            confidence=round(confidence, 2),
            recent_meals=meal_metrics[-10:],  # Last 10 meals
            state_description=description
        )

    async def _analyze_meal_absorption(
        self,
        user_id: str,
        moment: AnchoredTreatmentMoment
    ) -> Optional[AbsorptionMetrics]:
        """
        Analyze absorption pattern for a single meal.

        Finds:
        - Time to peak BG
        - Rise rate
        - Peak BG value
        """
        meal_time = moment.timestamp

        # Get glucose readings from meal time to +3 hours
        end_time = meal_time + timedelta(minutes=self.MAX_LOOKBACK_MINUTES)
        readings = await self.glucose_repo.get_history(
            user_id,
            start_time=meal_time - timedelta(minutes=15),  # Slightly before for baseline
            end_time=end_time
        )

        if len(readings) < 6:  # Need enough data points
            logger.debug(f"Meal at {meal_time}: only {len(readings)} readings (need 6)")
            return None

        # Sort by timestamp
        readings.sort(key=lambda r: r.timestamp)

        # Find BG at meal time (or closest reading before)
        bg_at_meal = None
        for r in readings:
            if r.timestamp <= meal_time:
                bg_at_meal = r.value
            else:
                break

        if bg_at_meal is None:
            bg_at_meal = readings[0].value

        # Find peak BG and time to peak
        peak_bg = bg_at_meal
        peak_time = meal_time
        max_rise_rate = 0.0

        # Calculate rolling rise rates and find peak
        for i, r in enumerate(readings):
            if r.timestamp <= meal_time:
                continue

            if r.value > peak_bg:
                peak_bg = r.value
                peak_time = r.timestamp

            # Calculate 5-minute rise rate
            if i > 0:
                prev = readings[i - 1]
                time_diff = (r.timestamp - prev.timestamp).total_seconds() / 60
                if 2 < time_diff < 10:  # Valid interval
                    rate = (r.value - prev.value) / time_diff
                    if rate > max_rise_rate:
                        max_rise_rate = rate

        # Calculate metrics
        time_to_peak = (peak_time - meal_time).total_seconds() / 60
        bg_rise = peak_bg - bg_at_meal

        # Validate
        if bg_rise < self.MIN_BG_RISE:
            logger.debug(f"Meal at {meal_time}: BG rise {bg_rise:.0f} < {self.MIN_BG_RISE} threshold")
            return None

        if time_to_peak < 10 or time_to_peak > 180:
            logger.debug(f"Meal at {meal_time}: time to peak {time_to_peak:.0f}m out of range (10-180)")
            return None

        # Average rise rate
        avg_rise_rate = bg_rise / time_to_peak if time_to_peak > 0 else 0

        # Determine meal type for tracking
        from zoneinfo import ZoneInfo
        est = ZoneInfo("America/New_York")
        if meal_time.tzinfo is None:
            meal_time = meal_time.replace(tzinfo=timezone.utc)
        local_time = meal_time.astimezone(est)
        hour = local_time.hour

        if 5 <= hour < 10:
            meal_type = "breakfast"
        elif 10 <= hour < 15:
            meal_type = "lunch"
        else:
            meal_type = "dinner"

        return AbsorptionMetrics(
            timestamp=moment.timestamp,
            time_to_peak_minutes=round(time_to_peak, 1),
            peak_bg=peak_bg,
            bg_at_meal=bg_at_meal,
            rise_rate=round(avg_rise_rate, 2),
            max_rise_rate=round(max_rise_rate, 2),
            carbs_grams=moment.carbs_grams,
            meal_type=meal_type,
            confidence=moment.confidence
        )

    async def _get_baseline_absorption(self, user_id: str) -> Dict[str, float]:
        """
        Get long-term baseline absorption profile for user.

        Uses historical data to establish normal absorption patterns.
        Falls back to defaults if insufficient data.
        """
        # Try to get from longer history (30 days)
        try:
            moments = await self.selector.get_clean_moments(
                user_id=user_id,
                days=30,
                treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
                min_confidence=0.4
            )

            if len(moments) >= 10:
                time_to_peaks = []
                rise_rates = []

                for moment in moments:
                    if moment.carbs_grams < self.MIN_CARBS:
                        continue

                    metrics = await self._analyze_meal_absorption(user_id, moment)
                    if metrics:
                        time_to_peaks.append(metrics.time_to_peak_minutes)
                        rise_rates.append(metrics.rise_rate)

                if len(time_to_peaks) >= 5:
                    return {
                        "time_to_peak": float(np.median(time_to_peaks)),
                        "rise_rate": float(np.median(rise_rates))
                    }

        except Exception as e:
            logger.warning(f"Failed to get baseline absorption: {e}")

        # Return defaults
        return {
            "time_to_peak": self.DEFAULT_TIME_TO_PEAK,
            "rise_rate": self.DEFAULT_RISE_RATE
        }

    def _classify_absorption_state(self, ttp_deviation: float) -> AbsorptionState:
        """Classify absorption state based on time-to-peak deviation."""
        if ttp_deviation > 50:
            return AbsorptionState.VERY_SLOW
        elif ttp_deviation > 20:
            return AbsorptionState.SLOW
        elif ttp_deviation < -50:
            return AbsorptionState.VERY_FAST
        elif ttp_deviation < -20:
            return AbsorptionState.FAST
        else:
            return AbsorptionState.NORMAL

    def _get_state_description(
        self,
        state: AbsorptionState,
        deviation: float,
        current_ttp: float
    ) -> str:
        """Get human-readable description of absorption state."""
        if state == AbsorptionState.VERY_SLOW:
            return (
                f"Very slow absorption detected: carbs are peaking {abs(deviation):.0f}% "
                f"slower than usual (~{current_ttp:.0f} min). This may indicate gastroparesis "
                "or illness. Consider timing insulin closer to meals."
            )
        elif state == AbsorptionState.SLOW:
            return (
                f"Slower absorption: carbs are peaking {abs(deviation):.0f}% slower "
                f"(~{current_ttp:.0f} min). May need to adjust meal-insulin timing."
            )
        elif state == AbsorptionState.VERY_FAST:
            return (
                f"Very fast absorption: carbs are peaking {abs(deviation):.0f}% faster "
                f"than usual (~{current_ttp:.0f} min). Consider pre-bolusing earlier."
            )
        elif state == AbsorptionState.FAST:
            return (
                f"Faster absorption: carbs are hitting {abs(deviation):.0f}% quicker "
                f"(~{current_ttp:.0f} min). May benefit from earlier pre-bolus."
            )
        else:
            return "Carb absorption rate is normal."


# Convenience function
async def get_absorption_state(user_id: str, days: int = 3) -> AbsorptionProfile:
    """Get current absorption state for a user."""
    learner = AbsorptionLearner()
    return await learner.analyze_short_term_absorption(user_id, days)
