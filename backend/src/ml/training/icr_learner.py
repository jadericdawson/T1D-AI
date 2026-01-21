"""
ICR (Insulin-to-Carb Ratio) Learner for T1D-AI

Uses treatment-anchored approach:
1. Find documented meal moments (carbs + insulin together)
2. Get BG window around each meal
3. Only use data where BG behavior is explainable
4. Learn ICR from clean, validated data

ICR = how many grams of carbs are covered by 1 unit of insulin
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import numpy as np

from database.repositories import (
    GlucoseRepository, TreatmentRepository, LearnedICRRepository, LearnedISFRepository
)
from models.schemas import LearnedICR, ICRDataPoint
from ml.training.treatment_anchored_selector import (
    TreatmentAnchoredSelector, TreatmentType, AnchoredTreatmentMoment
)

logger = logging.getLogger(__name__)


class ICRLearner:
    """
    Service to learn ICR from meal bolus and glucose data.

    Uses treatment-anchored approach to ensure clean data:
    - Only uses documented meal moments (carbs + insulin)
    - Validates that BG behavior is explainable
    - Rejects data where undocumented treatments likely occurred

    ICR is learned by observing how many carbs are covered by insulin:
    - If BG returns to ~starting point: bolus was correct, ICR = carbs / carb_insulin
    - If BG ends higher: under-bolused, effective ICR > logged carbs / insulin
    - If BG ends lower: over-bolused, effective ICR < logged carbs / insulin
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.icr_repo = LearnedICRRepository()
        self.isf_repo = LearnedISFRepository()
        self.selector = TreatmentAnchoredSelector()

        # Configuration
        self.min_carbs = 15               # Minimum carbs to consider (g)
        self.min_insulin = 0.5            # Minimum insulin to consider (U)
        self.target_bg = 100              # Target BG for correction calculation
        self.return_to_target_tolerance = 60  # BG should be within ±60 of start
        self.min_icr = 3                  # Minimum valid ICR (g/U)
        self.max_icr = 25                 # Maximum valid ICR (g/U)
        self.default_isf = 50.0           # Default ISF if not learned
        self.min_confidence = 0.4         # Minimum confidence to include

    async def learn_icr(
        self,
        user_id: str,
        days: int = 30,
        meal_type: Optional[str] = None
    ) -> Optional[LearnedICR]:
        """
        Learn ICR from clean meal bolus events.

        A clean meal event:
        - Has logged carbs (≥15g)
        - Has logged insulin (≥0.5U)
        - BG behavior is explainable by the logged treatments

        Args:
            user_id: User ID to learn ICR for
            days: Number of days of history to analyze
            meal_type: Optional meal type filter (breakfast, lunch, dinner)

        Returns:
            Updated LearnedICR or None if insufficient data
        """
        # Get user's learned ISF for correction calculation
        isf = await self._get_user_isf(user_id)
        logger.info(f"Using ISF {isf:.1f} for ICR correction calculation")

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

        # Filter by meal type if specified
        if meal_type:
            moments = [m for m in moments if self._get_meal_type(m.timestamp) == meal_type]

        if not moments:
            logger.info(f"No clean {meal_type} boluses found for user {user_id}")
            return None

        # Extract ICR from each clean moment
        icr_events = []
        for moment in moments:
            if moment.carbs_grams < self.min_carbs:
                continue
            if moment.insulin_units < self.min_insulin:
                continue

            # Calculate how much insulin was used for carbs vs correction
            bg_before = moment.bg_window.bg_before
            bg_after = moment.bg_window.bg_after

            # Correction component: how much insulin was needed to bring BG to target
            correction_needed = (bg_before - self.target_bg) / isf
            correction_component = max(0, correction_needed)

            # Carb insulin: total insulin minus correction
            carb_insulin = moment.insulin_units - correction_component

            if carb_insulin < 0.3:
                # Most of the insulin was correction, not useful for ICR
                continue

            # Calculate ICR from the carb portion
            # Base ICR: carbs / carb_insulin
            base_icr = moment.carbs_grams / carb_insulin

            # Adjust based on outcome
            # If BG ended higher than target: we under-dosed, actual ICR is lower
            # If BG ended lower than target: we over-dosed, actual ICR is higher
            bg_outcome_error = bg_after - self.target_bg
            icr_adjustment = bg_outcome_error / isf / carb_insulin * moment.carbs_grams

            # The effective ICR that would have been correct
            effective_icr = base_icr - icr_adjustment * 0.3  # Partial adjustment

            # Sanity check
            if not (self.min_icr <= effective_icr <= self.max_icr):
                logger.debug(f"ICR {effective_icr:.1f} outside valid range, skipping")
                continue

            # Higher confidence if BG returned close to starting point
            bg_return_error = abs(bg_after - bg_before)
            outcome_confidence = max(0.3, 1.0 - bg_return_error / 100)
            final_confidence = moment.confidence * outcome_confidence

            meal_type_detected = self._get_meal_type(moment.timestamp)

            icr_events.append({
                "timestamp": moment.timestamp,
                "value": effective_icr,
                "bgBefore": bg_before,
                "bgAfter": bg_after,
                "insulinUnits": moment.insulin_units,
                "carbsGrams": moment.carbs_grams,
                "proteinGrams": moment.protein_grams,
                "fatGrams": moment.fat_grams,
                "glycemicIndex": moment.glycemic_index,
                "mealType": meal_type_detected,
                "correctionComponent": correction_component,
                "confidence": final_confidence,
                "reason": moment.explainability_reason
            })

        if not icr_events:
            logger.info(f"No valid ICR events found for user {user_id}")
            return None

        # Determine the meal type for storage
        storage_meal_type = meal_type or "overall"

        # Calculate weighted average ICR
        final_icr = await self._compute_weighted_icr(user_id, storage_meal_type, icr_events)
        logger.info(
            f"Learned {storage_meal_type} ICR for user {user_id}: {final_icr.value:.1f} "
            f"(n={len(icr_events)}, confidence={final_icr.confidence:.2f})"
        )

        return final_icr

    async def learn_all_icr(self, user_id: str, days: int = 30) -> Dict[str, Optional[LearnedICR]]:
        """
        Learn ICR for all meal types.

        Returns:
            Dict with overall, breakfast, lunch, dinner ICR values
        """
        results = {}

        # Learn overall ICR
        results["overall"] = await self.learn_icr(user_id, days, meal_type=None)

        # Learn meal-specific ICR
        for meal_type in ["breakfast", "lunch", "dinner"]:
            results[meal_type] = await self.learn_icr(user_id, days, meal_type=meal_type)

        # Determine default value
        default_value = 10.0  # Fallback default
        if results.get("overall"):
            default_value = results["overall"].value
        elif results.get("lunch"):
            default_value = results["lunch"].value

        results["default"] = default_value
        return results

    async def get_current_icr(
        self,
        user_id: str,
        meal_type: Optional[str] = None
    ) -> float:
        """
        Get the best ICR estimate for a user.

        Args:
            user_id: User ID
            meal_type: Optional meal type (breakfast, lunch, dinner)

        Returns:
            Best ICR estimate (defaults to 10 if no learned data)
        """
        # Try meal-specific first
        if meal_type:
            learned = await self.icr_repo.get(user_id, meal_type)
            if learned:
                return learned.value

        # Fallback to overall
        learned = await self.icr_repo.get(user_id, "overall")
        if learned:
            return learned.value

        return 10.0

    async def calculate_short_term_icr(
        self,
        user_id: str,
        days: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate short-term ICR from recent meal data (last 2-3 days).

        PROPER CALCULATION:
        Instead of looking at insulin doses (which user adjusts), we measure
        the carb effect on BG directly:

        1. carb_sensitivity = BG_rise_from_carbs / carbs (mg/dL per gram)
           where BG_rise_from_carbs = (BG_after - BG_before) + (insulin × ISF)

        2. effective_ICR = current_ISF / carb_sensitivity

        This way ICR properly reflects insulin requirements:
        - During illness: ISF drops → ICR drops → need more insulin
        - During exercise: ISF rises → ICR rises → need less insulin

        Returns:
            Dict with:
            - current_icr: The short-term calculated ICR
            - baseline_icr: The long-term learned ICR
            - deviation_percent: How much current differs from baseline
            - sample_count: Number of valid data points
            - confidence: Confidence in the calculation (0-1)
            - data_points: List of recent ICR observations
        """
        # Get long-term baseline ICR and ISF
        baseline_icr = 10.0
        learned = await self.icr_repo.get(user_id, "overall")
        if learned and learned.value:
            baseline_icr = learned.value

        baseline_isf = await self._get_user_isf(user_id)

        # Get current (short-term) ISF - this is critical for proper ICR calculation
        from ml.training.isf_learner import ISFLearner
        isf_learner = ISFLearner()
        short_term_isf_data = await isf_learner.calculate_short_term_isf(user_id, days=days)

        current_isf = baseline_isf
        if short_term_isf_data.get("current_isf") and short_term_isf_data.get("confidence", 0) > 0.3:
            current_isf = short_term_isf_data["current_isf"]
            logger.info(f"Using short-term ISF {current_isf:.1f} for ICR (baseline: {baseline_isf:.1f})")

        # Get recent meal moments (carbs + insulin together)
        moments = await self.selector.get_clean_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
            min_confidence=0.3  # Lower threshold for recent data
        )

        if not moments:
            logger.info(f"No recent meal data found for short-term ICR (user {user_id})")
            return {
                "current_icr": None,
                "baseline_icr": baseline_icr,
                "deviation_percent": 0.0,
                "sample_count": 0,
                "confidence": 0,
                "data_points": []
            }

        # PROPER CALCULATION: Measure carb sensitivity directly
        # carb_sensitivity = how much BG rises per gram of carb (mg/dL per gram)
        # This is INDEPENDENT of what insulin dose was given
        carb_sensitivity_observations = []

        for moment in moments:
            if moment.carbs_grams < 10:
                continue
            if moment.insulin_units < 0.3:
                continue

            bg_before = moment.bg_window.bg_before
            bg_after = moment.bg_window.bg_after

            # Calculate the BG rise from carbs alone:
            # BG_rise_from_carbs = (BG_after - BG_before) + (insulin_effect)
            # insulin_effect = insulin_given × current_ISF (how much insulin dropped BG)
            insulin_effect = moment.insulin_units * current_isf
            bg_rise_from_carbs = (bg_after - bg_before) + insulin_effect

            # Carb sensitivity = BG rise per gram
            carb_sensitivity = bg_rise_from_carbs / moment.carbs_grams

            # Sanity check: carb sensitivity should be positive and reasonable (2-15 mg/dL per gram)
            if not (1 <= carb_sensitivity <= 20):
                logger.debug(f"Carb sensitivity {carb_sensitivity:.1f} outside range, skipping")
                continue

            # Weight by recency
            hours_ago = (datetime.now(timezone.utc) - moment.timestamp).total_seconds() / 3600
            recency_weight = max(0.3, 1.0 - (hours_ago / (days * 24)))

            carb_sensitivity_observations.append({
                "timestamp": moment.timestamp.isoformat(),
                "carb_sensitivity": carb_sensitivity,
                "bg_before": bg_before,
                "bg_after": bg_after,
                "bg_rise_from_carbs": round(bg_rise_from_carbs, 1),
                "carbs": moment.carbs_grams,
                "insulin": moment.insulin_units,
                "insulin_effect": round(insulin_effect, 1),
                "meal_type": self._get_meal_type(moment.timestamp),
                "weight": moment.confidence * recency_weight,
                "hours_ago": round(hours_ago, 1)
            })

        if not carb_sensitivity_observations:
            return {
                "current_icr": None,
                "baseline_icr": baseline_icr,
                "deviation_percent": 0.0,
                "sample_count": 0,
                "confidence": 1.0,
                "data_points": []
            }

        # Calculate weighted average carb sensitivity
        weights = np.array([obs["weight"] for obs in carb_sensitivity_observations])
        sensitivities = np.array([obs["carb_sensitivity"] for obs in carb_sensitivity_observations])
        weights = weights / weights.sum()
        avg_carb_sensitivity = float(np.average(sensitivities, weights=weights))

        # NOW calculate effective ICR using current ISF
        # ICR = ISF / carb_sensitivity
        # This naturally makes ICR track ISF during illness!
        current_icr = current_isf / avg_carb_sensitivity

        # Sanity check
        if not (2 <= current_icr <= 30):
            current_icr = max(2, min(30, current_icr))

        # Calculate deviation from baseline
        deviation_percent = ((current_icr - baseline_icr) / baseline_icr) * 100 if baseline_icr > 0 else 0.0

        logger.info(
            f"Short-term ICR for user {user_id}: {current_icr:.1f} g/U "
            f"(baseline: {baseline_icr:.1f}, deviation: {deviation_percent:+.1f}%) "
            f"[ISF={current_isf:.1f}, carb_sens={avg_carb_sensitivity:.1f} mg/dL/g, n={len(carb_sensitivity_observations)}]"
        )

        # Convert observations to ICR for display
        icr_observations = []
        for obs in carb_sensitivity_observations:
            obs_icr = current_isf / obs["carb_sensitivity"] if obs["carb_sensitivity"] > 0 else baseline_icr
            icr_observations.append({
                "timestamp": obs["timestamp"],
                "icr": round(obs_icr, 1),
                "bg_before": obs["bg_before"],
                "bg_after": obs["bg_after"],
                "carbs": obs["carbs"],
                "insulin": obs["insulin"],
                "carb_sensitivity": round(obs["carb_sensitivity"], 2),
                "meal_type": obs["meal_type"],
                "confidence": obs["weight"],
                "hours_ago": obs["hours_ago"]
            })

        return {
            "current_icr": round(current_icr, 1),
            "baseline_icr": round(baseline_icr, 1),
            "deviation_percent": round(deviation_percent, 1),
            "sample_count": len(carb_sensitivity_observations),
            "confidence": 1.0,  # Proper calculation, not low confidence
            "data_points": icr_observations[-10:]
        }

    async def _get_user_isf(self, user_id: str) -> float:
        """Get user's learned ISF or default."""
        try:
            learned = await self.isf_repo.get(user_id, "fasting")
            if learned and learned.value:
                return learned.value
        except Exception:
            pass
        return self.default_isf

    def _get_meal_type(self, timestamp: datetime) -> str:
        """Categorize timestamp into meal type based on EST local time."""
        # Convert UTC timestamp to EST for meal type classification
        # Timestamps in DB are UTC, but meal types should be based on local time
        from zoneinfo import ZoneInfo
        est = ZoneInfo("America/New_York")

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Convert to EST and get local hour
        local_time = timestamp.astimezone(est)
        hour = local_time.hour

        if 5 <= hour < 10:
            return "breakfast"
        elif 10 <= hour < 15:
            return "lunch"
        elif 15 <= hour < 21:
            return "dinner"
        else:
            return "snack"

    async def _compute_weighted_icr(
        self,
        user_id: str,
        meal_type: str,
        events: List[dict]
    ) -> LearnedICR:
        """
        Compute weighted average ICR and store in database.

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
        weighted_icr = float(np.average(values, weights=final_weights))

        # Calculate meal-type patterns
        meal_pattern = {"breakfast": None, "lunch": None, "dinner": None, "snack": None}
        for mt in meal_pattern.keys():
            mt_events = [e for e in events if e.get("mealType") == mt]
            if len(mt_events) >= 2:
                mt_values = [e["value"] for e in mt_events]
                meal_pattern[mt] = float(np.median(mt_values))

        # Convert events to ICRDataPoint format
        history = []
        for e in events[-30:]:  # Keep last 30
            history.append(ICRDataPoint(
                timestamp=e["timestamp"],
                value=e["value"],
                bgBefore=int(round(e["bgBefore"])),
                bgAfter=int(round(e["bgAfter"])),
                insulinUnits=e["insulinUnits"],
                carbsGrams=e["carbsGrams"],
                proteinGrams=e.get("proteinGrams"),
                fatGrams=e.get("fatGrams"),
                glycemicIndex=e.get("glycemicIndex"),
                mealType=e.get("mealType"),
                correctionComponent=e.get("correctionComponent", 0),
                confidence=e["confidence"]
            ))

        # Create or update learned ICR
        learned_icr = LearnedICR(
            id=f"{user_id}_{meal_type}",
            userId=user_id,
            value=weighted_icr,
            confidence=min(1.0, len(events) / 10),
            sampleCount=len(events),
            lastUpdated=datetime.now(timezone.utc),
            history=history,
            mealTypePattern=meal_pattern,
            meanICR=float(np.mean(values)),
            stdICR=float(np.std(values)) if n > 1 else 0.0,
            minICR=float(np.min(values)),
            maxICR=float(np.max(values))
        )

        # Save to database
        return await self.icr_repo.upsert(learned_icr)


# Convenience function
async def learn_icr_for_user(user_id: str, days: int = 30) -> dict:
    """Learn ICR for a user and return all meal types."""
    learner = ICRLearner()
    return await learner.learn_all_icr(user_id, days)
