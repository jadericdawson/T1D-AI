"""
Treatment-Anchored Data Selector for T1D-AI

Core principle: BG doesn't change by itself. Every BG movement has a cause.
We only use data where we can EXPLAIN the BG behavior with logged treatments.

Approach:
1. Find documented treatment moments (insulin and/or carbs)
2. Get BG window around each moment
3. Validate that BG behavior is explainable by ONLY those logged treatments
4. If explainable → clean data for training
5. If not explainable → something undocumented happened, skip

This replaces time-based filtering with intelligence-based filtering.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from database.repositories import GlucoseRepository, TreatmentRepository
from models.schemas import Treatment, GlucoseReading

logger = logging.getLogger(__name__)


class TreatmentType(Enum):
    """Types of treatment moments we can anchor on."""
    INSULIN_ONLY = "insulin_only"       # Correction bolus, no carbs
    CARBS_ONLY = "carbs_only"           # Carbs without insulin (rare but valid)
    INSULIN_WITH_CARBS = "meal_bolus"   # Meal with insulin coverage


@dataclass
class BGWindow:
    """BG readings around a treatment moment."""
    readings: List[Tuple[int, float]]  # [(minutes_from_treatment, bg_value), ...]
    bg_before: float                    # BG at treatment time
    bg_after: float                     # BG at end of window
    min_bg: float                       # Lowest BG in window
    max_bg: float                       # Highest BG in window
    trend_before: Optional[str]         # BG trend at treatment time

    @property
    def bg_change(self) -> float:
        """Total BG change from start to end."""
        return self.bg_after - self.bg_before

    @property
    def max_rise(self) -> float:
        """Maximum rise above starting BG."""
        return self.max_bg - self.bg_before

    @property
    def max_drop(self) -> float:
        """Maximum drop below starting BG."""
        return self.bg_before - self.min_bg


@dataclass
class AnchoredTreatmentMoment:
    """A treatment moment with surrounding BG data and validation."""
    timestamp: datetime
    treatment_type: TreatmentType
    insulin_units: float
    carbs_grams: float
    protein_grams: float
    fat_grams: float
    glycemic_index: Optional[int]
    bg_window: BGWindow
    is_explainable: bool
    explainability_reason: str
    confidence: float

    # For learning
    treatment_id: str
    meal_description: Optional[str] = None


class TreatmentAnchoredSelector:
    """
    Selects clean training data by anchoring on documented treatment moments
    and validating that BG behavior is explainable.
    """

    def __init__(
        self,
        # Window parameters
        bg_window_before_min: int = 30,      # Get BG starting 30 min before treatment
        bg_window_after_min: int = 240,      # Get BG up to 4 hours after

        # Validation thresholds
        max_unexplained_rise: float = 30.0,  # Max BG rise with no carbs logged
        max_unexplained_drop: float = 30.0,  # Max BG drop with no insulin logged

        # Treatment isolation (to avoid overlapping effects)
        min_treatment_gap_min: int = 120,    # Minimum gap between treatments to consider isolated

        # Data quality
        min_bg_readings_in_window: int = 8,  # Need at least 8 readings (~40 min of CGM)
        max_bg_gap_min: int = 30,            # Max gap between BG readings
    ):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()

        self.bg_window_before_min = bg_window_before_min
        self.bg_window_after_min = bg_window_after_min
        self.max_unexplained_rise = max_unexplained_rise
        self.max_unexplained_drop = max_unexplained_drop
        self.min_treatment_gap_min = min_treatment_gap_min
        self.min_bg_readings_in_window = min_bg_readings_in_window
        self.max_bg_gap_min = max_bg_gap_min

    async def get_anchored_moments(
        self,
        user_id: str,
        days: int = 30,
        treatment_types: Optional[List[TreatmentType]] = None,
        require_isolated: bool = True,
    ) -> List[AnchoredTreatmentMoment]:
        """
        Find documented treatment moments and validate their BG windows.

        Args:
            user_id: User to get data for
            days: Number of days of history
            treatment_types: Filter for specific types (default: all)
            require_isolated: Only return treatments without nearby other treatments

        Returns:
            List of validated, explainable treatment moments
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Get all treatments
        all_treatments = await self.treatment_repo.get_by_user(
            user_id=user_id,
            start_time=start_time,
            limit=5000
        )

        if not all_treatments:
            logger.info(f"No treatments found for user {user_id}")
            return []

        # Group treatments by moment (insulin + carbs logged together)
        treatment_moments = self._group_treatments_by_moment(all_treatments)
        logger.info(f"Found {len(treatment_moments)} treatment moments")

        # Filter by isolation if required
        if require_isolated:
            treatment_moments = self._filter_isolated_moments(treatment_moments)
            logger.info(f"After isolation filter: {len(treatment_moments)} moments")

        # Process each moment
        anchored_moments = []
        for moment in treatment_moments:
            # Classify treatment type
            t_type = self._classify_treatment_type(moment)

            # Filter by requested types
            if treatment_types and t_type not in treatment_types:
                continue

            # Get BG window
            bg_window = await self._get_bg_window(user_id, moment["timestamp"])

            if bg_window is None:
                continue

            # Validate explainability
            is_explainable, reason, confidence = self._validate_explainability(
                treatment_type=t_type,
                insulin=moment["insulin"],
                carbs=moment["carbs"],
                bg_window=bg_window
            )

            anchored_moment = AnchoredTreatmentMoment(
                timestamp=moment["timestamp"],
                treatment_type=t_type,
                insulin_units=moment["insulin"],
                carbs_grams=moment["carbs"],
                protein_grams=moment["protein"],
                fat_grams=moment["fat"],
                glycemic_index=moment.get("gi"),
                bg_window=bg_window,
                is_explainable=is_explainable,
                explainability_reason=reason,
                confidence=confidence,
                treatment_id=moment["id"],
                meal_description=moment.get("description")
            )

            anchored_moments.append(anchored_moment)

        # Log summary
        explainable = [m for m in anchored_moments if m.is_explainable]
        logger.info(
            f"Processed {len(anchored_moments)} moments: "
            f"{len(explainable)} explainable, "
            f"{len(anchored_moments) - len(explainable)} contaminated"
        )

        return anchored_moments

    async def get_clean_moments(
        self,
        user_id: str,
        days: int = 30,
        treatment_types: Optional[List[TreatmentType]] = None,
        min_confidence: float = 0.5,
    ) -> List[AnchoredTreatmentMoment]:
        """Get only the clean, explainable moments for training."""
        all_moments = await self.get_anchored_moments(
            user_id=user_id,
            days=days,
            treatment_types=treatment_types,
            require_isolated=True
        )

        return [
            m for m in all_moments
            if m.is_explainable and m.confidence >= min_confidence
        ]

    def _group_treatments_by_moment(
        self,
        treatments: List[Treatment]
    ) -> List[Dict[str, Any]]:
        """
        Group treatments that happen close together into single moments.
        E.g., carbs logged at 12:00 and insulin at 12:02 = same moment.
        """
        if not treatments:
            return []

        # Helper to normalize timestamps for comparison
        def normalize_ts(ts):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        # Sort by timestamp (normalize to handle mixed tz-aware/naive)
        sorted_treatments = sorted(treatments, key=lambda t: normalize_ts(t.timestamp))

        moments = []
        current_moment = None

        for treatment in sorted_treatments:
            ts = treatment.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if current_moment is None:
                current_moment = {
                    "timestamp": ts,
                    "insulin": treatment.insulin or 0,
                    "carbs": treatment.carbs or 0,
                    "protein": treatment.protein or 0,
                    "fat": treatment.fat or 0,
                    "gi": getattr(treatment, 'glycemicIndex', None),
                    "description": treatment.notes or treatment.foodDescription if hasattr(treatment, 'foodDescription') else None,
                    "id": treatment.id,
                    "treatments": [treatment]
                }
            else:
                # Check if this treatment is within 15 minutes of current moment
                time_diff = abs((ts - current_moment["timestamp"]).total_seconds() / 60)

                if time_diff <= 15:
                    # Same moment - aggregate
                    current_moment["insulin"] += treatment.insulin or 0
                    current_moment["carbs"] += treatment.carbs or 0
                    current_moment["protein"] += treatment.protein or 0
                    current_moment["fat"] += treatment.fat or 0
                    current_moment["treatments"].append(treatment)
                    if treatment.notes:
                        current_moment["description"] = treatment.notes
                else:
                    # New moment
                    if current_moment["insulin"] > 0 or current_moment["carbs"] > 0:
                        moments.append(current_moment)

                    current_moment = {
                        "timestamp": ts,
                        "insulin": treatment.insulin or 0,
                        "carbs": treatment.carbs or 0,
                        "protein": treatment.protein or 0,
                        "fat": treatment.fat or 0,
                        "gi": getattr(treatment, 'glycemicIndex', None),
                        "description": treatment.notes or getattr(treatment, 'foodDescription', None),
                        "id": treatment.id,
                        "treatments": [treatment]
                    }

        # Don't forget the last moment
        if current_moment and (current_moment["insulin"] > 0 or current_moment["carbs"] > 0):
            moments.append(current_moment)

        return moments

    def _filter_isolated_moments(
        self,
        moments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter to only moments that are isolated (no nearby treatments)."""
        if len(moments) <= 1:
            return moments

        isolated = []
        gap_minutes = self.min_treatment_gap_min

        for i, moment in enumerate(moments):
            # Check gap to previous moment
            if i > 0:
                prev_moment = moments[i - 1]
                gap_to_prev = (moment["timestamp"] - prev_moment["timestamp"]).total_seconds() / 60
                if gap_to_prev < gap_minutes:
                    continue

            # Check gap to next moment
            if i < len(moments) - 1:
                next_moment = moments[i + 1]
                gap_to_next = (next_moment["timestamp"] - moment["timestamp"]).total_seconds() / 60
                if gap_to_next < gap_minutes:
                    continue

            isolated.append(moment)

        return isolated

    def _classify_treatment_type(self, moment: Dict[str, Any]) -> TreatmentType:
        """Classify a treatment moment by what was logged."""
        has_insulin = moment["insulin"] > 0.3  # At least 0.3U
        has_carbs = moment["carbs"] > 5        # At least 5g

        if has_insulin and has_carbs:
            return TreatmentType.INSULIN_WITH_CARBS
        elif has_insulin:
            return TreatmentType.INSULIN_ONLY
        else:
            return TreatmentType.CARBS_ONLY

    async def _get_bg_window(
        self,
        user_id: str,
        treatment_time: datetime
    ) -> Optional[BGWindow]:
        """Get BG readings around a treatment moment."""
        start_time = treatment_time - timedelta(minutes=self.bg_window_before_min)
        end_time = treatment_time + timedelta(minutes=self.bg_window_after_min)

        readings = await self.glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )

        if not readings or len(readings) < self.min_bg_readings_in_window:
            return None

        # Convert to (minutes_from_treatment, bg_value) tuples
        bg_sequence = []
        for reading in readings:
            ts = reading.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            treatment_ts = treatment_time if treatment_time.tzinfo else treatment_time.replace(tzinfo=timezone.utc)

            minutes_from_treatment = (ts - treatment_ts).total_seconds() / 60
            bg_sequence.append((minutes_from_treatment, reading.value))

        bg_sequence.sort(key=lambda x: x[0])

        # Check for gaps
        for i in range(1, len(bg_sequence)):
            gap = bg_sequence[i][0] - bg_sequence[i-1][0]
            if gap > self.max_bg_gap_min:
                # Too big a gap in the data
                return None

        # Find BG at key times
        bg_before = self._interpolate_bg(bg_sequence, 0)
        bg_after = self._interpolate_bg(bg_sequence, self.bg_window_after_min - 30)  # Use 3.5hr mark

        if bg_before is None or bg_after is None:
            return None

        # Get min/max
        bg_values = [bg for _, bg in bg_sequence if _ >= 0]  # Only after treatment
        if not bg_values:
            return None

        min_bg = min(bg_values)
        max_bg = max(bg_values)

        # Get trend at treatment time
        trend_before = None
        readings_at_treatment = [r for r in readings if abs((r.timestamp.replace(tzinfo=timezone.utc) - treatment_time.replace(tzinfo=timezone.utc)).total_seconds()) < 600]
        if readings_at_treatment and hasattr(readings_at_treatment[0], 'trend'):
            trend_before = readings_at_treatment[0].trend

        return BGWindow(
            readings=bg_sequence,
            bg_before=bg_before,
            bg_after=bg_after,
            min_bg=min_bg,
            max_bg=max_bg,
            trend_before=trend_before
        )

    def _interpolate_bg(
        self,
        bg_sequence: List[Tuple[float, float]],
        target_minutes: float
    ) -> Optional[float]:
        """Interpolate BG value at a specific time."""
        if not bg_sequence:
            return None

        # Find readings around target time
        before = [(t, bg) for t, bg in bg_sequence if t <= target_minutes]
        after = [(t, bg) for t, bg in bg_sequence if t >= target_minutes]

        if not before and not after:
            return None

        if not before:
            return after[0][1]
        if not after:
            return before[-1][1]

        t1, bg1 = before[-1]
        t2, bg2 = after[0]

        if t1 == t2:
            return bg1

        # Linear interpolation
        ratio = (target_minutes - t1) / (t2 - t1)
        return bg1 + ratio * (bg2 - bg1)

    def _validate_explainability(
        self,
        treatment_type: TreatmentType,
        insulin: float,
        carbs: float,
        bg_window: BGWindow
    ) -> Tuple[bool, str, float]:
        """
        Validate if the BG behavior is explainable by the logged treatments.

        Core principle: BG doesn't change by itself.

        Returns:
            (is_explainable, reason, confidence)
        """

        if treatment_type == TreatmentType.INSULIN_ONLY:
            # Insulin without carbs: BG should not rise significantly
            # A small rise is OK (dawn phenomenon, stress), but big rise = undocumented carbs

            if bg_window.max_rise > self.max_unexplained_rise:
                return (
                    False,
                    f"BG rose {bg_window.max_rise:.0f} with insulin only - likely undocumented carbs",
                    0.2
                )

            # BG should drop (or at least not rise much)
            if bg_window.bg_change > 20:  # BG ended higher than it started
                return (
                    False,
                    f"BG ended {bg_window.bg_change:.0f} higher despite insulin - unexplained",
                    0.3
                )

            # Looks clean!
            confidence = min(1.0, bg_window.max_drop / 30)  # More drop = more confident
            return (True, "clean_insulin_correction", confidence)

        elif treatment_type == TreatmentType.CARBS_ONLY:
            # Carbs without insulin: BG should not drop significantly
            # A small drop is OK, but big drop = undocumented insulin

            if bg_window.max_drop > self.max_unexplained_drop:
                return (
                    False,
                    f"BG dropped {bg_window.max_drop:.0f} with carbs only - likely undocumented insulin",
                    0.2
                )

            # Looks clean!
            confidence = min(1.0, bg_window.max_rise / 30)  # Expected rise = more confident
            return (True, "clean_carbs_only", confidence)

        else:  # INSULIN_WITH_CARBS (meal bolus)
            # Meal bolus: BG can go up, down, or flat - all valid!
            # The key is: does the PATTERN make sense?

            # Check for unexplained behavior during the meal window
            # A well-timed bolus might show: flat, small rise then flat, or small rise then drop
            # An undertreated meal: rise and stay high
            # An overtreated meal: drop too much

            # What we're looking for as "unexplainable":
            # 1. BG drops dramatically in first 30 min (insulin shouldn't work that fast)
            # 2. BG drops way more than expected for the insulin given
            # 3. BG rises way more than expected for the carbs given

            early_readings = [(t, bg) for t, bg in bg_window.readings if 0 <= t <= 30]
            if early_readings:
                early_min = min(bg for _, bg in early_readings)
                early_drop = bg_window.bg_before - early_min

                if early_drop > 40:
                    return (
                        False,
                        f"BG dropped {early_drop:.0f} in first 30 min - too fast for insulin",
                        0.3
                    )

            # Check for unexplained additional drop (more than insulin could cause)
            # Rough estimate: 1U drops BG by 30-70 mg/dL
            max_expected_drop = insulin * 100  # Conservative upper bound
            if bg_window.max_drop > max_expected_drop + 50:
                return (
                    False,
                    f"BG dropped {bg_window.max_drop:.0f} - more than {insulin:.1f}U could explain",
                    0.3
                )

            # Check for unexplained additional rise (more than carbs could cause)
            # Rough estimate: 1g carbs raises BG by 3-5 mg/dL
            max_expected_rise = carbs * 6  # Conservative upper bound
            if bg_window.max_rise > max_expected_rise + 50:
                return (
                    False,
                    f"BG rose {bg_window.max_rise:.0f} - more than {carbs:.0f}g could explain",
                    0.3
                )

            # Looks explainable!
            # Higher confidence if BG returns close to starting point (good bolus timing)
            return_to_start = abs(bg_window.bg_change)
            confidence = max(0.5, 1.0 - return_to_start / 100)

            return (True, "clean_meal_bolus", confidence)


# Convenience function
async def get_clean_training_data(
    user_id: str,
    days: int = 30,
    treatment_types: Optional[List[TreatmentType]] = None
) -> List[AnchoredTreatmentMoment]:
    """Get clean, explainable treatment moments for training."""
    selector = TreatmentAnchoredSelector()
    return await selector.get_clean_moments(
        user_id=user_id,
        days=days,
        treatment_types=treatment_types
    )
