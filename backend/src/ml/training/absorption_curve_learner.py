"""
Absorption Curve Timing Learner

Uses treatment-anchored approach:
1. Find documented treatment moments (insulin, carbs, protein)
2. Get BG window around each treatment
3. Only use data where BG behavior is explainable
4. Learn absorption timing from clean, validated data

Learns personalized IOB/COB/POB activity curve timing parameters:
- Insulin: when it starts (onset), peaks, and ends (duration)
- Carbs: when they start raising BG, peak, and finish
- Protein: late rise timing parameters

These replace hardcoded defaults in iob_cob_service.py:
- Insulin: peak_min=75 -> learned insulinPeakMin
- Carbs: peak_min=45 -> learned carbPeakMin
- Protein: peak_min=180 -> learned proteinPeakMin
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from models.schemas import (
    UserAbsorptionProfile,
    AbsorptionCurveDataPoint,
)
from database.repositories import (
    TreatmentRepository, GlucoseRepository, UserAbsorptionProfileRepository
)
from ml.training.treatment_anchored_selector import (
    TreatmentAnchoredSelector, TreatmentType, AnchoredTreatmentMoment, BGWindow
)

logger = logging.getLogger(__name__)


class AbsorptionCurveLearner:
    """
    Learns personalized absorption curve timing from user data.

    Uses treatment-anchored approach to ensure clean data:
    - Only uses documented treatment moments
    - Validates that BG behavior is explainable
    - Detects actual onset, peak, and duration from BG response curve shape
    - Aggregates across multiple treatments to find average timing
    """

    def __init__(
        self,
        user_id: str,
        min_samples_per_type: int = 5,
    ):
        """
        Initialize the learner.

        Args:
            user_id: User ID to learn for
            min_samples_per_type: Minimum observations per treatment type for learning
        """
        self.user_id = user_id
        self.min_samples_per_type = min_samples_per_type
        self.treatment_repo = TreatmentRepository()
        self.glucose_repo = GlucoseRepository()

        # Use different window sizes for different treatment types
        self.insulin_selector = TreatmentAnchoredSelector(
            bg_window_after_min=300,  # 5 hours for insulin
            min_treatment_gap_min=180  # Need 3 hour gap for clean insulin data
        )
        self.meal_selector = TreatmentAnchoredSelector(
            bg_window_after_min=360,  # 6 hours for meals (need to see protein effect)
            min_treatment_gap_min=180
        )

    async def learn_from_recent_treatments(
        self,
        days: int = 30,
        isf: float = 50.0,
        icr: float = 10.0
    ) -> Optional[UserAbsorptionProfile]:
        """
        Learn absorption timing from recent treatments.

        This method:
        1. Gets clean treatment moments using anchored selector
        2. Analyzes BG curve shape to find onset/peak/duration
        3. Aggregates timing data to find user's typical values
        4. Creates UserAbsorptionProfile with learned values

        Args:
            days: Days of history to analyze
            isf: Insulin sensitivity factor (for reference, not used in timing)
            icr: Insulin-to-carb ratio (for reference, not used in timing)

        Returns:
            UserAbsorptionProfile with learned timing, or None if insufficient data
        """
        logger.info(f"Learning absorption curves for user {self.user_id} from {days} days")

        # Learn insulin timing from correction boluses
        insulin_timing = await self._learn_insulin_timing(days)

        # Learn carb timing from meal boluses
        carb_timing = await self._learn_carb_timing(days)

        # Learn protein timing from high-protein meals
        protein_timing = await self._learn_protein_timing(days)

        # Check if we have enough data
        total_samples = (
            len(insulin_timing.get("samples", [])) +
            len(carb_timing.get("samples", [])) +
            len(protein_timing.get("samples", []))
        )

        if total_samples < 5:
            logger.warning(f"Insufficient data for absorption learning: {total_samples} total samples")
            return None

        # Calculate confidence based on sample counts
        confidence = min(1.0, total_samples / 30)

        # Build the profile
        profile = UserAbsorptionProfile(
            id=f"{self.user_id}_absorption_profile",
            userId=self.user_id,

            # Insulin timing
            insulinOnsetMin=insulin_timing.get("onset", 15.0),
            insulinPeakMin=insulin_timing.get("peak", 75.0),
            insulinDurationMin=insulin_timing.get("duration", 240.0),
            insulinSampleCount=len(insulin_timing.get("samples", [])),

            # Carb timing
            carbOnsetMin=carb_timing.get("onset", 10.0),
            carbPeakMin=carb_timing.get("peak", 45.0),
            carbDurationMin=carb_timing.get("duration", 180.0),
            carbSampleCount=len(carb_timing.get("samples", [])),

            # Protein timing
            proteinOnsetMin=protein_timing.get("onset", 90.0),
            proteinPeakMin=protein_timing.get("peak", 180.0),
            proteinDurationMin=protein_timing.get("duration", 300.0),
            proteinSampleCount=len(protein_timing.get("samples", [])),

            confidence=confidence,
            lastUpdated=datetime.now(timezone.utc),

            # History for each type
            insulinHistory=insulin_timing.get("history", []),
            carbHistory=carb_timing.get("history", []),
            proteinHistory=protein_timing.get("history", []),

            # Time of day adjustments (can be learned separately)
            timeOfDayAdjustments={
                "morning": {"insulinPeakMultiplier": 0.9, "carbPeakMultiplier": 1.0},
                "afternoon": {"insulinPeakMultiplier": 1.0, "carbPeakMultiplier": 1.0},
                "evening": {"insulinPeakMultiplier": 1.1, "carbPeakMultiplier": 1.1},
                "night": {"insulinPeakMultiplier": 1.2, "carbPeakMultiplier": 1.0},
            }
        )

        logger.info(
            f"Learned absorption profile: insulin_peak={profile.insulinPeakMin:.0f}min, "
            f"carb_peak={profile.carbPeakMin:.0f}min, protein_peak={profile.proteinPeakMin:.0f}min "
            f"(confidence={confidence:.0%})"
        )

        return profile

    async def _learn_insulin_timing(self, days: int) -> Dict[str, Any]:
        """Learn insulin absorption timing from correction boluses."""

        # Get clean insulin-only moments
        moments = await self.insulin_selector.get_clean_moments(
            user_id=self.user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_ONLY],
            min_confidence=0.5
        )

        if len(moments) < self.min_samples_per_type:
            logger.info(f"Insufficient insulin samples: {len(moments)}")
            return {"onset": 15.0, "peak": 75.0, "duration": 240.0, "samples": [], "history": []}

        # Analyze each moment for timing
        onset_times = []
        peak_times = []
        duration_times = []
        history = []

        for moment in moments:
            timing = self._analyze_insulin_curve(moment.bg_window, moment.insulin_units)
            if timing:
                onset_times.append(timing["onset"])
                peak_times.append(timing["peak"])
                duration_times.append(timing["duration"])

                history.append(AbsorptionCurveDataPoint(
                    timestamp=moment.timestamp,
                    treatmentId=moment.treatment_id,
                    treatmentType="insulin",
                    detectedOnsetMin=timing["onset"],
                    detectedPeakMin=timing["peak"],
                    detectedHalfLifeMin=timing["duration"] / 2,
                    dataQuality=moment.confidence
                ))

        if not onset_times:
            return {"onset": 15.0, "peak": 75.0, "duration": 240.0, "samples": [], "history": []}

        return {
            "onset": float(np.median(onset_times)),
            "peak": float(np.median(peak_times)),
            "duration": float(np.median(duration_times)),
            "samples": onset_times,
            "history": history
        }

    async def _learn_carb_timing(self, days: int) -> Dict[str, Any]:
        """Learn carb absorption timing from meal boluses."""

        # Get clean meal moments
        moments = await self.meal_selector.get_clean_moments(
            user_id=self.user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
            min_confidence=0.5
        )

        # Filter to meals with significant carbs
        carb_moments = [m for m in moments if m.carbs_grams >= 20]

        if len(carb_moments) < self.min_samples_per_type:
            logger.info(f"Insufficient carb samples: {len(carb_moments)}")
            return {"onset": 10.0, "peak": 45.0, "duration": 180.0, "samples": [], "history": []}

        # Analyze each moment for carb timing
        onset_times = []
        peak_times = []
        duration_times = []
        history = []

        for moment in carb_moments:
            timing = self._analyze_carb_curve(moment.bg_window, moment.carbs_grams)
            if timing:
                onset_times.append(timing["onset"])
                peak_times.append(timing["peak"])
                duration_times.append(timing["duration"])

                history.append(AbsorptionCurveDataPoint(
                    timestamp=moment.timestamp,
                    treatmentId=moment.treatment_id,
                    treatmentType="carbs",
                    detectedOnsetMin=timing["onset"],
                    detectedPeakMin=timing["peak"],
                    detectedHalfLifeMin=timing["duration"] / 2,
                    dataQuality=moment.confidence
                ))

        if not onset_times:
            return {"onset": 10.0, "peak": 45.0, "duration": 180.0, "samples": [], "history": []}

        return {
            "onset": float(np.median(onset_times)),
            "peak": float(np.median(peak_times)),
            "duration": float(np.median(duration_times)),
            "samples": onset_times,
            "history": history
        }

    async def _learn_protein_timing(self, days: int) -> Dict[str, Any]:
        """Learn protein absorption timing from high-protein meals."""

        # Get clean meal moments
        moments = await self.meal_selector.get_clean_moments(
            user_id=self.user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
            min_confidence=0.5
        )

        # Filter to meals with significant protein
        protein_moments = [m for m in moments if m.protein_grams >= 15]

        if len(protein_moments) < self.min_samples_per_type:
            logger.info(f"Insufficient protein samples: {len(protein_moments)}")
            return {"onset": 90.0, "peak": 180.0, "duration": 300.0, "samples": [], "history": []}

        # Analyze each moment for protein timing (late rise)
        onset_times = []
        peak_times = []
        duration_times = []
        history = []

        for moment in protein_moments:
            timing = self._analyze_protein_curve(moment.bg_window)
            if timing:
                onset_times.append(timing["onset"])
                peak_times.append(timing["peak"])
                duration_times.append(timing["duration"])

                history.append(AbsorptionCurveDataPoint(
                    timestamp=moment.timestamp,
                    treatmentId=moment.treatment_id,
                    treatmentType="protein",
                    detectedOnsetMin=timing["onset"],
                    detectedPeakMin=timing["peak"],
                    detectedHalfLifeMin=timing["duration"] / 2,
                    dataQuality=moment.confidence
                ))

        if not onset_times:
            return {"onset": 90.0, "peak": 180.0, "duration": 300.0, "samples": [], "history": []}

        return {
            "onset": float(np.median(onset_times)),
            "peak": float(np.median(peak_times)),
            "duration": float(np.median(duration_times)),
            "samples": onset_times,
            "history": history
        }

    def _analyze_insulin_curve(
        self,
        bg_window: BGWindow,
        insulin_units: float
    ) -> Optional[Dict[str, float]]:
        """
        Analyze BG curve to find insulin timing.

        For insulin, we look for:
        - Onset: when BG starts dropping
        - Peak: when BG drops fastest
        - Duration: when effect ends
        """
        readings = [(t, bg) for t, bg in bg_window.readings if t >= 0]
        if len(readings) < 10:
            return None

        readings.sort(key=lambda x: x[0])
        times = [t for t, _ in readings]
        bgs = [bg for _, bg in readings]

        # Find onset (when BG starts dropping from initial value)
        initial_bg = bg_window.bg_before
        onset = 15  # default
        for t, bg in readings:
            if bg < initial_bg - 10:
                onset = max(5, t - 10)  # A bit before the detected drop
                break

        # Find peak effect (maximum rate of drop)
        # Calculate BG change rate at each point
        if len(bgs) >= 3:
            rates = []
            for i in range(1, len(bgs) - 1):
                if times[i+1] - times[i-1] > 0:
                    rate = (bgs[i-1] - bgs[i+1]) / (times[i+1] - times[i-1])  # positive = dropping
                    rates.append((times[i], rate))

            if rates:
                # Peak is where rate is highest (fastest drop)
                peak_time, _ = max(rates, key=lambda x: x[1])
                peak = peak_time
            else:
                peak = 75
        else:
            peak = 75

        # Find duration (when BG stabilizes or starts rising again)
        min_bg = min(bgs)
        min_idx = bgs.index(min_bg)
        if min_idx < len(times):
            duration = times[min_idx] + 60  # Duration extends past the minimum
        else:
            duration = 240

        # Sanity checks
        onset = max(5, min(60, onset))
        peak = max(30, min(150, peak))
        duration = max(180, min(360, duration))

        return {"onset": onset, "peak": peak, "duration": duration}

    def _analyze_carb_curve(
        self,
        bg_window: BGWindow,
        carbs_grams: float
    ) -> Optional[Dict[str, float]]:
        """
        Analyze BG curve to find carb timing.

        For carbs, we look for:
        - Onset: when BG starts rising
        - Peak: when BG rises fastest (or reaches maximum)
        - Duration: when carb effect ends
        """
        readings = [(t, bg) for t, bg in bg_window.readings if t >= 0]
        if len(readings) < 10:
            return None

        readings.sort(key=lambda x: x[0])
        times = [t for t, _ in readings]
        bgs = [bg for _, bg in readings]

        initial_bg = bg_window.bg_before

        # Find onset (when BG starts rising)
        onset = 10  # default
        for t, bg in readings:
            if bg > initial_bg + 10:
                onset = max(5, t - 5)
                break

        # Find peak (maximum BG or fastest rise)
        max_bg = max(bgs[:min(len(bgs), 24)])  # Look in first 2 hours
        max_idx = bgs.index(max_bg)
        if max_idx < len(times):
            peak = times[max_idx]
        else:
            peak = 45

        # Find duration (when BG returns toward baseline)
        duration = 180  # default
        for i, (t, bg) in enumerate(readings):
            if t > peak and bg < max_bg - 30:
                duration = t
                break

        # Sanity checks
        onset = max(5, min(30, onset))
        peak = max(20, min(120, peak))
        duration = max(90, min(300, duration))

        return {"onset": onset, "peak": peak, "duration": duration}

    def _analyze_protein_curve(
        self,
        bg_window: BGWindow
    ) -> Optional[Dict[str, float]]:
        """
        Analyze BG curve to find protein timing (late rise).

        For protein, we look for:
        - Onset: when late BG rise starts (typically 90+ min after meal)
        - Peak: when late rise reaches maximum
        - Duration: total protein effect window
        """
        # Look for late readings (90+ min after meal)
        late_readings = [(t, bg) for t, bg in bg_window.readings if t >= 90]
        early_readings = [(t, bg) for t, bg in bg_window.readings if 30 <= t <= 90]

        if len(late_readings) < 6 or len(early_readings) < 3:
            return None

        # Baseline is the minimum in the early window (after carb effect settles)
        baseline = min(bg for _, bg in early_readings)

        # Find late rise
        late_max = max(bg for _, bg in late_readings)
        late_rise = late_max - baseline

        if late_rise < 20:
            # No significant late rise detected
            return None

        # Find onset of late rise
        onset = 90
        for t, bg in sorted(late_readings):
            if bg > baseline + 10:
                onset = t
                break

        # Find peak of late rise
        peak_time = 180
        for t, bg in late_readings:
            if bg == late_max:
                peak_time = t
                break

        # Duration extends past the peak
        duration = peak_time + 120

        # Sanity checks
        onset = max(60, min(180, onset))
        peak_time = max(90, min(300, peak_time))
        duration = max(180, min(480, duration))

        return {"onset": onset, "peak": peak_time, "duration": duration}


async def learn_absorption_curves(
    user_id: str,
    days: int = 30,
    isf: float = 50.0,
    icr: float = 10.0
) -> Optional[UserAbsorptionProfile]:
    """Convenience function to learn absorption curves for a user."""
    learner = AbsorptionCurveLearner(user_id=user_id)
    return await learner.learn_from_recent_treatments(days=days, isf=isf, icr=icr)
