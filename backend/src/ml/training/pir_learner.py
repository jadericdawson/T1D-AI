"""
PIR (Protein-to-Insulin Ratio) Learner for T1D-AI

Uses treatment-anchored approach:
1. Find documented high-protein meal moments
2. Track the late BG rise that occurs 2-4 hours after meals (protein effect)
3. Only use data where BG behavior is explainable
4. Learn PIR from clean, validated data

PIR = how many grams of protein are covered by 1 unit of insulin
Protein typically raises BG 2-4 hours after consumption as it converts to glucose.

Pump-aware mode:
- Detects pump users automatically from treatment data
- Dual detection: BG late rise + excess auto-correction activity
- Control-IQ auto-corrects protein rise, masking it in BG data
- Uses auto-correction signal when BG stays flat but pump works harder
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from database.repositories import (
    GlucoseRepository, TreatmentRepository, LearnedPIRRepository, LearnedISFRepository
)
from models.schemas import LearnedPIR, PIRDataPoint
from ml.training.treatment_anchored_selector import (
    TreatmentAnchoredSelector, TreatmentType, AnchoredTreatmentMoment, BGWindow, is_pump_user
)

logger = logging.getLogger(__name__)


class PIRLearner:
    """
    Service to learn PIR from high-protein meal events and BG response.

    Uses treatment-anchored approach to ensure clean data:
    - Only uses documented meal moments with protein logged
    - Looks for late BG rise pattern (2-4h after meal) that indicates protein effect
    - Validates that BG behavior is explainable
    - Rejects data where undocumented treatments likely occurred

    Pump-aware:
    - Auto-detects pump users from treatment deliveryMethod
    - Dual detection: BG rise signal + auto-correction excess signal
    - Control-IQ fights protein rises with auto-corrections, masking BG effect
    - Compares auto-correction activity to historical baseline
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.pir_repo = LearnedPIRRepository()
        self.isf_repo = LearnedISFRepository()
        self.selector = TreatmentAnchoredSelector(
            bg_window_after_min=300  # Longer window for protein (5 hours)
        )

        # Configuration
        self.min_protein = 10             # Minimum protein to consider (g)
        self.late_rise_window_start = 90  # Start looking for late rise (min after meal)
        self.late_rise_window_end = 300   # End looking for late rise (5h after meal)
        self.min_late_rise = 15           # Minimum late rise to count (mg/dL)
        self.min_pir = 5                  # Minimum valid PIR (g/U)
        self.max_pir = 20                 # Maximum valid PIR (g/U) - typical for children ~10
        self.default_isf = 50.0           # Default ISF if not learned
        self.min_confidence = 0.4         # Minimum confidence to include

        # Pump-specific state
        self._pump_aware = False
        self._baseline_auto_correction_rate = None  # U per 3.5h window, learned from data

    async def _auto_detect_pump(self, user_id: str) -> None:
        """Auto-detect pump user and configure selector accordingly."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        treatments = await self.treatment_repo.get_by_user(
            user_id=user_id, start_time=start_time, limit=500
        )
        self._pump_aware = is_pump_user(treatments)
        if self._pump_aware:
            logger.info(f"Pump user detected for PIR learning (user {user_id})")
            self.selector = TreatmentAnchoredSelector(
                bg_window_after_min=300,
                pump_aware=True
            )
            # Learn baseline auto-correction rate from all treatments
            self._baseline_auto_correction_rate = self._learn_baseline_auto_correction(treatments)

    def _learn_baseline_auto_correction(self, treatments) -> float:
        """
        Calculate baseline auto-correction rate (insulin per 3.5-hour window).
        This is the average auto-correction activity when NOT dealing with protein.
        """
        auto_corrections = [
            t for t in treatments
            if getattr(t, 'deliveryMethod', None) == 'pump_auto_correction'
        ]

        if not auto_corrections or len(auto_corrections) < 3:
            return 0.0

        # Sum all auto-correction insulin and divide by total time span
        total_insulin = sum(t.insulin or 0 for t in auto_corrections)
        timestamps = sorted(t.timestamp for t in auto_corrections)

        # Ensure timezone-aware
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        if first_ts.tzinfo is None:
            first_ts = first_ts.replace(tzinfo=timezone.utc)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        total_hours = (last_ts - first_ts).total_seconds() / 3600.0
        if total_hours < 1:
            return 0.0

        # Return average per 3.5-hour window (late protein window duration)
        rate_per_hour = total_insulin / total_hours
        return rate_per_hour * 3.5

    async def learn_pir(
        self,
        user_id: str,
        days: int = 30,
        meal_type: Optional[str] = None
    ) -> Optional[LearnedPIR]:
        """
        Learn PIR from high-protein meal events.

        A valid protein event:
        - Has significant protein logged (≥10g)
        - Shows late BG rise pattern (2-5h after meal)
        - BG behavior is explainable by logged treatments

        For pump users, uses dual detection:
        - Signal A: BG late rise (existing)
        - Signal B: Excess auto-correction activity vs baseline

        Args:
            user_id: User ID to learn PIR for
            days: Number of days of history to analyze
            meal_type: Optional meal type filter (breakfast, lunch, dinner)

        Returns:
            Updated LearnedPIR or None if insufficient data
        """
        await self._auto_detect_pump(user_id)

        # Get user's learned ISF for calculations
        isf = await self._get_user_isf(user_id)
        logger.info(f"Using ISF {isf:.1f} for PIR calculation")

        # Get clean meal moments (we need protein data)
        moments = await self.selector.get_clean_moments(
            user_id=user_id,
            days=days,
            treatment_types=[TreatmentType.INSULIN_WITH_CARBS],
            min_confidence=self.min_confidence
        )

        if not moments:
            logger.info(f"No clean meal moments found for user {user_id}")
            return None

        # Filter to high-protein meals
        protein_moments = [m for m in moments if m.protein_grams >= self.min_protein]

        if not protein_moments:
            logger.info(f"No high-protein meals (≥{self.min_protein}g) found for user {user_id}")
            return None

        # Filter by meal type if specified
        if meal_type:
            protein_moments = [m for m in protein_moments if self._get_meal_type(m.timestamp) == meal_type]

        if not protein_moments:
            logger.info(f"No high-protein {meal_type} meals found for user {user_id}")
            return None

        # For pump users, fetch all treatments for auto-correction analysis
        all_treatments = []
        if self._pump_aware:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)
            all_treatments = await self.treatment_repo.get_by_user(
                user_id=user_id, start_time=start_time, limit=5000
            )

        # Extract PIR from each protein moment by looking at late rise
        pir_events = []
        onset_times = []
        peak_times = []

        for moment in protein_moments:
            # Signal A: Raw BG late rise (existing detection)
            late_rise_info = self._detect_late_rise(moment.bg_window)

            # Signal B (pump only): Excess auto-correction activity
            auto_correction_info = None
            if self._pump_aware and all_treatments:
                auto_correction_info = self._detect_protein_via_auto_corrections(
                    moment.timestamp, all_treatments, isf
                )

            # Combine signals: use whichever gives a stronger protein signal
            rise_insulin = 0.0
            auto_insulin = 0.0
            onset_min = 120
            peak_min = 180

            if late_rise_info is not None:
                rise_insulin = late_rise_info["rise_amount"] / isf
                onset_min = late_rise_info["onset_min"]
                peak_min = late_rise_info["peak_min"]

            if auto_correction_info is not None:
                auto_insulin = auto_correction_info["excess_auto_insulin"]

            # Use the stronger signal
            protein_effect_insulin = max(rise_insulin, auto_insulin)

            if protein_effect_insulin < 0.3:
                # Neither signal detected meaningful protein effect
                continue

            # PIR = protein_grams / insulin_needed
            pir = moment.protein_grams / protein_effect_insulin

            # Sanity check
            if not (self.min_pir <= pir <= self.max_pir):
                logger.debug(f"PIR {pir:.1f} outside valid range, skipping")
                continue

            onset_times.append(onset_min)
            peak_times.append(peak_min)

            meal_type_detected = self._get_meal_type(moment.timestamp)

            # Determine which signal we used
            signal_source = "bg_rise" if rise_insulin >= auto_insulin else "auto_correction"

            pir_events.append({
                "timestamp": moment.timestamp,
                "value": pir,
                "bgBefore": moment.bg_window.bg_before,
                "bgPeak": moment.bg_window.bg_before + (late_rise_info["rise_amount"] if late_rise_info else 0),
                "bgAfter": moment.bg_window.bg_after,
                "insulinForProtein": protein_effect_insulin,
                "proteinGrams": moment.protein_grams,
                "fatGrams": moment.fat_grams,
                "carbsGrams": moment.carbs_grams,
                "mealDescription": moment.meal_description,
                "proteinOnsetMin": onset_min,
                "proteinPeakMin": peak_min,
                "mealType": meal_type_detected,
                "confidence": moment.confidence,
                "reason": f"{moment.explainability_reason} (signal: {signal_source})"
            })

        if not pir_events:
            logger.info(f"No valid PIR events found for user {user_id}")
            return None

        # Calculate timing statistics
        avg_onset = int(np.mean(onset_times)) if onset_times else 120
        avg_peak = int(np.mean(peak_times)) if peak_times else 180

        # Determine the meal type for storage
        storage_meal_type = meal_type or "overall"

        # Calculate weighted average PIR
        final_pir = await self._compute_weighted_pir(
            user_id, storage_meal_type, pir_events, avg_onset, avg_peak
        )
        logger.info(
            f"Learned {storage_meal_type} PIR for user {user_id}: {final_pir.value:.1f} "
            f"(n={len(pir_events)}, onset={avg_onset}min, peak={avg_peak}min, pump_aware={self._pump_aware})"
        )

        return final_pir

    def _detect_protein_via_auto_corrections(
        self,
        meal_time: datetime,
        all_treatments: list,
        isf: float
    ) -> Optional[Dict[str, Any]]:
        """
        Detect protein effect by looking at excess auto-correction activity
        in the late window (90-300 min after meal) compared to baseline.

        Control-IQ auto-corrects the late protein BG rise, masking it.
        BG stays flat while pump works harder. The excess auto-correction
        insulin tells us the protein effect the pump is fighting.

        Returns:
            Dict with excess_auto_insulin and equivalent_bg_rise, or None
        """
        if self._baseline_auto_correction_rate is None:
            return None

        meal_ts = meal_time
        if meal_ts.tzinfo is None:
            meal_ts = meal_ts.replace(tzinfo=timezone.utc)

        window_start = meal_ts + timedelta(minutes=self.late_rise_window_start)
        window_end = meal_ts + timedelta(minutes=self.late_rise_window_end)

        # Sum auto-correction insulin in the late window
        window_auto_insulin = 0.0
        for t in all_treatments:
            if getattr(t, 'deliveryMethod', None) != 'pump_auto_correction':
                continue
            ts = t.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if window_start <= ts <= window_end:
                window_auto_insulin += (t.insulin or 0)

        # Compare to baseline
        excess = window_auto_insulin - self._baseline_auto_correction_rate

        if excess < 0.2:
            # No meaningful excess — protein effect not detected via auto-corrections
            return None

        # Equivalent BG rise = excess_auto_insulin * ISF
        equivalent_bg_rise = excess * isf

        return {
            "excess_auto_insulin": excess,
            "equivalent_bg_rise": equivalent_bg_rise,
            "window_auto_insulin": window_auto_insulin,
            "baseline_rate": self._baseline_auto_correction_rate
        }

    async def learn_all_pir(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Learn PIR for all meal types.

        Returns:
            Dict with overall, breakfast, lunch, dinner PIR values plus timing
        """
        results = {}

        # Learn overall PIR
        results["overall"] = await self.learn_pir(user_id, days, meal_type=None)

        # Learn meal-specific PIR
        for meal_type in ["breakfast", "lunch", "dinner"]:
            results[meal_type] = await self.learn_pir(user_id, days, meal_type=meal_type)

        # Determine default value
        default_value = 25.0  # Fallback default
        if results.get("overall"):
            default_value = results["overall"].value

        results["default"] = default_value

        # Include timing info
        if results.get("overall"):
            results["timing"] = {
                "onset_minutes": int(results["overall"].proteinOnsetMin),
                "peak_minutes": int(results["overall"].proteinPeakMin)
            }

        return results

    async def get_current_pir(
        self,
        user_id: str,
        meal_type: Optional[str] = None
    ) -> Tuple[float, Optional[int], Optional[int]]:
        """
        Get the best PIR estimate for a user.

        Args:
            user_id: User ID
            meal_type: Optional meal type (breakfast, lunch, dinner)

        Returns:
            Tuple of (PIR value, onset_minutes, peak_minutes)
        """
        # Try meal-specific first
        if meal_type:
            learned = await self.pir_repo.get(user_id, meal_type)
            if learned:
                return learned.value, int(learned.proteinOnsetMin), int(learned.proteinPeakMin)

        # Fallback to overall
        learned = await self.pir_repo.get(user_id, "overall")
        if learned:
            return learned.value, int(learned.proteinOnsetMin), int(learned.proteinPeakMin)

        return 25.0, None, None

    async def _get_user_isf(self, user_id: str) -> float:
        """Get user's learned ISF or default."""
        try:
            learned = await self.isf_repo.get(user_id, "fasting")
            if learned and learned.value:
                return learned.value
        except Exception:
            pass
        return self.default_isf

    def _detect_late_rise(self, bg_window: BGWindow) -> Optional[Dict[str, Any]]:
        """
        Detect late BG rise pattern in the BG window.

        Returns info about the late rise if found, None otherwise.
        """
        # Get readings in the late window (90-300 min after meal)
        late_readings = [
            (t, bg) for t, bg in bg_window.readings
            if self.late_rise_window_start <= t <= self.late_rise_window_end
        ]

        if len(late_readings) < 4:
            return None

        # Find the minimum BG in the post-meal window (0-90 min)
        # This is the "baseline" after initial carb effect
        early_readings = [
            (t, bg) for t, bg in bg_window.readings
            if 30 <= t <= 90
        ]

        if not early_readings:
            baseline = bg_window.bg_before
        else:
            baseline = min(bg for _, bg in early_readings)

        # Find the peak in the late window
        peak_time, peak_bg = max(late_readings, key=lambda x: x[1])

        # The late rise is peak - baseline
        late_rise = peak_bg - baseline

        if late_rise < self.min_late_rise:
            return None

        # Find onset (when BG started rising above baseline)
        onset_time = self.late_rise_window_start
        for t, bg in sorted(late_readings):
            if bg > baseline + 10:
                onset_time = t
                break

        return {
            "rise_amount": late_rise,
            "onset_min": int(onset_time),
            "peak_min": int(peak_time),
            "baseline": baseline,
            "peak_bg": peak_bg
        }

    def _get_meal_type(self, timestamp: datetime) -> str:
        """Categorize timestamp into meal type based on EST local time."""
        # Convert UTC timestamp to EST for meal type classification
        from zoneinfo import ZoneInfo
        from datetime import timezone as tz
        est = ZoneInfo("America/New_York")

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=tz.utc)

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

    async def _compute_weighted_pir(
        self,
        user_id: str,
        meal_type: str,
        events: List[dict],
        avg_onset: int,
        avg_peak: int
    ) -> LearnedPIR:
        """
        Compute weighted average PIR and store in database.

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
        weighted_pir = float(np.average(values, weights=final_weights))

        # Convert events to PIRDataPoint format
        history = []
        for e in events[-30:]:  # Keep last 30
            history.append(PIRDataPoint(
                timestamp=e["timestamp"],
                value=e["value"],
                bgBefore=int(round(e["bgBefore"])),
                bgPeak=int(round(e["bgPeak"])),
                bgAfter=int(round(e["bgAfter"])),
                insulinForProtein=e["insulinForProtein"],
                proteinGrams=e["proteinGrams"],
                fatGrams=e.get("fatGrams"),
                carbsGrams=e.get("carbsGrams"),
                mealDescription=e.get("mealDescription"),
                proteinOnsetMin=e["proteinOnsetMin"],
                proteinPeakMin=e["proteinPeakMin"],
                confidence=e["confidence"]
            ))

        # Create or update learned PIR
        learned_pir = LearnedPIR(
            id=f"{user_id}_{meal_type}",
            userId=user_id,
            value=weighted_pir,
            proteinOnsetMin=float(avg_onset),
            proteinPeakMin=float(avg_peak),
            proteinDurationMin=300.0,  # 5 hours typical
            confidence=min(1.0, len(events) / 8),
            sampleCount=len(events),
            lastUpdated=datetime.now(timezone.utc),
            history=history,
            meanPIR=float(np.mean(values)),
            stdPIR=float(np.std(values)) if n > 1 else 0.0,
            minPIR=float(np.min(values)),
            maxPIR=float(np.max(values))
        )

        # Save to database
        return await self.pir_repo.upsert(learned_pir)


# Convenience function
async def learn_pir_for_user(user_id: str, days: int = 30) -> dict:
    """Learn PIR for a user and return all meal types."""
    learner = PIRLearner()
    return await learner.learn_all_pir(user_id, days)
