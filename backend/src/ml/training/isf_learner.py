"""
ISF (Insulin Sensitivity Factor) Learner for T1D-AI

Uses treatment-anchored approach:
1. Find documented insulin treatments
2. Get BG window around each treatment
3. Only use data where BG behavior is explainable by logged treatments
4. Learn ISF from clean, validated data

ISF = how much 1 unit of insulin drops blood glucose (mg/dL per unit)
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import numpy as np

from database.repositories import (
    GlucoseRepository, TreatmentRepository, LearnedISFRepository
)
from models.schemas import LearnedISF, ISFDataPoint, Treatment, GlucoseReading
from ml.training.treatment_anchored_selector import (
    TreatmentAnchoredSelector, TreatmentType, AnchoredTreatmentMoment
)

logger = logging.getLogger(__name__)


class ISFLearner:
    """
    Service to learn ISF from insulin bolus and glucose data.

    Uses treatment-anchored approach to ensure clean data:
    - Only uses documented treatment moments
    - Validates that BG behavior is explainable
    - Rejects data where undocumented treatments likely occurred
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.isf_repo = LearnedISFRepository()
        self.selector = TreatmentAnchoredSelector()

        # Configuration
        self.min_insulin_units = 0.5       # Minimum bolus to consider
        self.min_bg_change = 15            # Minimum BG change to count (mg/dL)
        self.min_isf = 15                  # Minimum valid ISF
        self.max_isf = 150                 # Maximum valid ISF
        self.min_confidence = 0.4          # Minimum confidence to include

    async def learn_fasting_isf(self, user_id: str, days: int = 30) -> Optional[LearnedISF]:
        """
        Learn fasting ISF from clean correction boluses.

        A clean correction bolus:
        - Insulin given without carbs
        - BG behavior is explainable (drops, doesn't unexpectedly rise)
        - No undocumented carbs detected

        Args:
            user_id: User ID to learn ISF for
            days: Number of days of history to analyze

        Returns:
            Updated LearnedISF or None if insufficient data
        """
        # Get clean insulin-only moments using anchored selector
        moments = await self.selector.get_clean_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_ONLY],
            min_confidence=self.min_confidence
        )

        if not moments:
            logger.info(f"No clean correction boluses found for user {user_id}")
            return None

        # Extract ISF from each clean moment
        isf_events = []
        for moment in moments:
            if moment.insulin_units < self.min_insulin_units:
                continue

            # Calculate ISF from the BG drop
            bg_drop = moment.bg_window.bg_before - moment.bg_window.bg_after

            # For corrections, we expect a drop. Skip if BG rose.
            if bg_drop < self.min_bg_change:
                continue

            isf = bg_drop / moment.insulin_units

            # Sanity check
            if not (self.min_isf <= isf <= self.max_isf):
                logger.debug(f"ISF {isf:.1f} outside valid range, skipping")
                continue

            time_of_day = self._get_time_of_day(moment.timestamp)

            isf_events.append({
                "timestamp": moment.timestamp,
                "value": isf,
                "bgBefore": moment.bg_window.bg_before,
                "bgAfter": moment.bg_window.bg_after,
                "insulinUnits": moment.insulin_units,
                "hoursAfterMeal": None,  # Fasting
                "confidence": moment.confidence,
                "timeOfDay": time_of_day,
                "reason": moment.explainability_reason
            })

        if not isf_events:
            logger.info(f"No valid fasting ISF events found for user {user_id}")
            return None

        # Calculate weighted average ISF
        final_isf = await self._compute_weighted_isf(user_id, "fasting", isf_events)
        logger.info(
            f"Learned fasting ISF for user {user_id}: {final_isf.value:.1f} "
            f"(n={len(isf_events)}, confidence={final_isf.confidence:.2f})"
        )

        return final_isf

    async def learn_meal_isf(self, user_id: str, days: int = 30) -> Optional[LearnedISF]:
        """
        Learn meal ISF from clean meal boluses.

        For meal boluses, ISF is harder to isolate because carbs and insulin
        act together. We estimate ISF by looking at how much "correction"
        component was needed.

        A clean meal bolus:
        - Insulin + carbs given together
        - BG behavior is explainable (can go up, down, or flat - all valid!)
        - No unexplained extra rises/drops

        Args:
            user_id: User ID to learn ISF for
            days: Number of days of history to analyze

        Returns:
            Updated LearnedISF or None if insufficient data
        """
        # Get clean meal moments
        moments = await self.selector.get_clean_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
            min_confidence=self.min_confidence
        )

        if not moments:
            logger.info(f"No clean meal boluses found for user {user_id}")
            return None

        # For meal ISF, we need to estimate carb effect to isolate insulin effect
        # Use a conservative estimate: ~4-5 mg/dL per gram of carbs
        isf_events = []
        for moment in moments:
            if moment.insulin_units < self.min_insulin_units:
                continue
            if moment.carbs_grams < 10:
                continue

            # Estimate the carb effect (how much BG would rise from carbs alone)
            # Conservative estimate: 4 mg/dL per gram
            estimated_carb_rise = moment.carbs_grams * 4

            # The total BG change we observed
            observed_change = moment.bg_window.bg_after - moment.bg_window.bg_before

            # If carbs raised BG by X, and insulin dropped it by Y,
            # then observed_change = X - Y
            # So Y = X - observed_change = estimated_carb_rise - observed_change
            estimated_insulin_drop = estimated_carb_rise - observed_change

            if estimated_insulin_drop < self.min_bg_change:
                # Insulin didn't seem to do much - skip
                continue

            isf = estimated_insulin_drop / moment.insulin_units

            # Sanity check
            if not (self.min_isf <= isf <= self.max_isf):
                continue

            # Lower confidence for meal ISF (more estimation involved)
            adjusted_confidence = moment.confidence * 0.7

            time_of_day = self._get_time_of_day(moment.timestamp)

            isf_events.append({
                "timestamp": moment.timestamp,
                "value": isf,
                "bgBefore": moment.bg_window.bg_before,
                "bgAfter": moment.bg_window.bg_after,
                "insulinUnits": moment.insulin_units,
                "hoursAfterMeal": 0.0,
                "confidence": adjusted_confidence,
                "timeOfDay": time_of_day,
                "reason": moment.explainability_reason
            })

        if not isf_events:
            logger.info(f"No valid meal ISF events found for user {user_id}")
            return None

        final_isf = await self._compute_weighted_isf(user_id, "meal", isf_events)
        logger.info(
            f"Learned meal ISF for user {user_id}: {final_isf.value:.1f} "
            f"(n={len(isf_events)}, confidence={final_isf.confidence:.2f})"
        )

        return final_isf

    async def learn_all_isf(self, user_id: str, days: int = 30) -> dict:
        """Learn both fasting and meal ISF for a user."""
        fasting = await self.learn_fasting_isf(user_id, days)
        meal = await self.learn_meal_isf(user_id, days)

        return {
            "fasting": fasting,
            "meal": meal,
            "default": fasting.value if fasting else (meal.value if meal else 50.0)
        }

    async def get_current_isf(
        self,
        user_id: str,
        is_fasting: bool = True,
        time_of_day: Optional[str] = None
    ) -> float:
        """
        Get the best ISF estimate for a user.

        Args:
            user_id: User ID
            is_fasting: Whether to use fasting or meal ISF
            time_of_day: Optional time period (morning, afternoon, evening, night)

        Returns:
            Best ISF estimate (defaults to 50 if no learned data)
        """
        isf_type = "fasting" if is_fasting else "meal"
        learned = await self.isf_repo.get(user_id, isf_type)

        if not learned:
            # Fallback to default
            return 50.0

        # Check for time-of-day specific value
        if time_of_day and learned.timeOfDayPattern.get(time_of_day):
            return learned.timeOfDayPattern[time_of_day]

        return learned.value

    async def calculate_short_term_isf(
        self,
        user_id: str,
        days: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate short-term ISF from recent data (last 2-3 days).

        This captures temporary changes in insulin sensitivity due to:
        - Illness (lower ISF = more insulin resistant)
        - Exercise (higher ISF = more sensitive)
        - Hormonal changes
        - Stress

        Returns:
            Dict with:
            - current_isf: The short-term calculated ISF
            - baseline_isf: The long-term learned ISF baseline
            - sample_count: Number of valid data points
            - confidence: Confidence in the calculation (0-1)
            - data_points: List of recent ISF observations
        """
        # Get baseline ISF from database first (needed for deviation calculations)
        baseline_isf = 55.0  # Default
        try:
            fasting_isf = await self.isf_repo.get(user_id, "fasting")
            meal_isf = await self.isf_repo.get(user_id, "meal")
            if meal_isf and meal_isf.value:
                baseline_isf = meal_isf.value
            elif fasting_isf and fasting_isf.value:
                baseline_isf = fasting_isf.value
        except Exception as e:
            logger.warning(f"Failed to get baseline ISF: {e}")

        # Get recent correction data (insulin only, no carbs)
        # Don't require strict isolation for short-term analysis - we're detecting
        # illness/resistance patterns, not learning precise ISF values
        all_moments = await self.selector.get_anchored_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_ONLY],
            require_isolated=False  # More lenient for illness detection
        )
        # Filter for explainable moments with reasonable confidence
        moments = [m for m in all_moments if m.is_explainable and m.confidence >= 0.3]

        if not moments:
            logger.info(f"No recent correction data found for short-term ISF (user {user_id})")
            return {
                "current_isf": None,
                "baseline_isf": baseline_isf,  # Always include baseline
                "sample_count": 0,
                "confidence": 0,
                "data_points": []
            }

        # Calculate ISF from each moment
        isf_observations = []
        for moment in moments:
            if moment.insulin_units < 0.3:  # Lower threshold for recent
                continue

            bg_drop = moment.bg_window.bg_before - moment.bg_window.bg_after

            # For corrections, we expect a drop
            if bg_drop < 10:  # Lower threshold - even small drops count
                continue

            isf = bg_drop / moment.insulin_units

            # Sanity check - wider range for detecting resistance
            if not (10 <= isf <= 200):
                continue

            # Weight by recency (hours ago)
            hours_ago = (datetime.now(timezone.utc) - moment.timestamp).total_seconds() / 3600
            recency_weight = max(0.3, 1.0 - (hours_ago / (days * 24)))

            isf_observations.append({
                "timestamp": moment.timestamp.isoformat(),
                "isf": isf,
                "bg_before": moment.bg_window.bg_before,
                "bg_after": moment.bg_window.bg_after,
                "insulin": moment.insulin_units,
                "confidence": moment.confidence * recency_weight,
                "hours_ago": round(hours_ago, 1)
            })

        if not isf_observations:
            return {
                "current_isf": None,
                "baseline_isf": baseline_isf,  # Always include baseline
                "sample_count": 0,
                "confidence": 0,
                "data_points": []
            }

        # Calculate weighted average ISF
        weights = np.array([obs["confidence"] for obs in isf_observations])
        values = np.array([obs["isf"] for obs in isf_observations])

        # Normalize weights
        weights = weights / weights.sum()

        current_isf = float(np.average(values, weights=weights))
        confidence = min(1.0, len(isf_observations) / 5)  # Full confidence at 5+ points

        logger.info(
            f"Short-term ISF for user {user_id}: {current_isf:.1f} "
            f"(n={len(isf_observations)}, confidence={confidence:.2f})"
        )

        return {
            "current_isf": round(current_isf, 1),
            "baseline_isf": baseline_isf,  # Always include baseline
            "sample_count": len(isf_observations),
            "confidence": round(confidence, 2),
            "data_points": isf_observations[-10:]  # Last 10 for display
        }

    def _get_time_of_day(self, timestamp: datetime) -> str:
        """Categorize timestamp into time of day period based on EST local time."""
        # Convert UTC timestamp to EST for time-of-day classification
        from zoneinfo import ZoneInfo
        est = ZoneInfo("America/New_York")

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Convert to EST and get local hour
        local_time = timestamp.astimezone(est)
        hour = local_time.hour

        if 6 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    async def _compute_weighted_isf(
        self,
        user_id: str,
        isf_type: str,
        events: List[dict]
    ) -> LearnedISF:
        """
        Compute weighted average ISF and store in database.

        Uses exponential weighting to prioritize recent observations.
        """
        # Sort by timestamp (most recent last)
        events.sort(key=lambda e: e["timestamp"])

        # Calculate weights (exponential decay - recent = higher)
        n = len(events)
        weights = np.exp(np.linspace(-1, 0, n))

        # Weight by confidence too
        confidence_weights = np.array([e["confidence"] for e in events])
        final_weights = weights * confidence_weights
        final_weights = final_weights / final_weights.sum()

        # Calculate weighted average
        values = np.array([e["value"] for e in events])
        weighted_isf = float(np.average(values, weights=final_weights))

        # Calculate time-of-day patterns
        tod_pattern = {"morning": None, "afternoon": None, "evening": None, "night": None}
        for tod in tod_pattern.keys():
            tod_events = [e for e in events if e.get("timeOfDay") == tod]
            if len(tod_events) >= 2:  # Need at least 2 observations
                tod_values = [e["value"] for e in tod_events]
                tod_pattern[tod] = float(np.median(tod_values))

        # Convert events to ISFDataPoint format
        history = []
        for e in events[-30:]:  # Keep last 30
            history.append(ISFDataPoint(
                timestamp=e["timestamp"],
                value=e["value"],
                bgBefore=int(round(e["bgBefore"])),
                bgAfter=int(round(e["bgAfter"])),
                insulinUnits=e["insulinUnits"],
                hoursAfterMeal=e.get("hoursAfterMeal"),
                confidence=e["confidence"]
            ))

        # Create or update learned ISF
        learned_isf = LearnedISF(
            id=f"{user_id}_{isf_type}",
            userId=user_id,
            isfType=isf_type,
            value=weighted_isf,
            confidence=min(1.0, len(events) / 10),
            sampleCount=len(events),
            lastUpdated=datetime.now(timezone.utc),
            history=history,
            timeOfDayPattern=tod_pattern,
            meanISF=float(np.mean(values)),
            stdISF=float(np.std(values)) if n > 1 else 0.0,
            minISF=float(np.min(values)),
            maxISF=float(np.max(values))
        )

        # Save to database
        return await self.isf_repo.upsert(learned_isf)


# Convenience function
async def learn_isf_for_user(user_id: str, days: int = 30) -> dict:
    """Learn ISF for a user and return both fasting and meal ISF."""
    learner = ISFLearner()
    return await learner.learn_all_isf(user_id, days)
