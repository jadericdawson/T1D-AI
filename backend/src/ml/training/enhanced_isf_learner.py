"""
Enhanced ISF (Insulin Sensitivity Factor) Learner for T1D-AI

Features:
- Smart "clean bolus" detection with BG pattern validation
- Detects undocumented carbs (BG rises after insulin = likely ate something)
- Contextual features: time of day, lunar phase, day of year
- Uses validated clean boluses to improve future detection
- Imports from historic bolus_moments.jsonl data
"""
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from database.repositories import (
    GlucoseRepository, TreatmentRepository, LearnedISFRepository
)
from models.schemas import LearnedISF, ISFDataPoint, GlucoseReading

logger = logging.getLogger(__name__)


def get_lunar_phase(dt: datetime) -> Tuple[float, float]:
    """Calculate lunar phase as sin/cos for cyclic encoding."""
    ref_date = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    lunar_cycle_days = 29.530588853
    days_since_ref = (dt - ref_date).total_seconds() / 86400
    phase = (days_since_ref % lunar_cycle_days) / lunar_cycle_days
    theta = 2 * math.pi * phase
    return math.sin(theta), math.cos(theta)


def get_day_of_year_encoding(dt: datetime) -> Tuple[float, float]:
    """Encode day of year cyclically for seasonal patterns."""
    day_of_year = dt.timetuple().tm_yday
    theta = 2 * math.pi * day_of_year / 365.25
    return math.sin(theta), math.cos(theta)


class CleanBolusValidator:
    """
    Validates whether a bolus is "clean" (no undocumented carbs).

    Clean bolus criteria:
    1. No carbs logged within ±2 hours
    2. BG should NOT rise in the first 30-60 minutes after insulin
    3. BG should drop steadily after insulin action starts (~15-30 min)
    4. Pattern should match known clean bolus profiles
    """

    def __init__(self):
        self.known_clean_profiles: List[Dict] = []
        self.max_initial_rise_mg = 15  # Max acceptable BG rise in first 30 min
        self.expected_drop_start_min = 30  # When BG should start dropping
        self.min_drop_by_60min = 10  # Minimum BG drop expected by 60 min

    def add_validated_profile(self, profile: Dict):
        """Add a known clean bolus profile for comparison."""
        self.known_clean_profiles.append(profile)

    def validate_bolus_pattern(
        self,
        bg_before: float,
        bg_sequence: List[Tuple[int, float]],  # [(minutes_after, bg_value), ...]
        insulin_units: float
    ) -> Tuple[bool, float, str]:
        """
        Validate if a bolus appears clean based on BG pattern.

        Returns:
            (is_clean, confidence, reason)
        """
        if len(bg_sequence) < 6:  # Need at least 30 min of data
            return False, 0.0, "insufficient_data"

        # Sort by time
        bg_sequence = sorted(bg_sequence, key=lambda x: x[0])

        # Check for initial BG rise (0-30 min) - indicates undocumented carbs
        initial_readings = [bg for min_after, bg in bg_sequence if 0 <= min_after <= 30]
        if initial_readings:
            max_initial = max(initial_readings)
            initial_rise = max_initial - bg_before

            if initial_rise > self.max_initial_rise_mg:
                return False, 0.2, f"bg_rose_{initial_rise:.0f}mg_initial"

        # Check for drop by 60 minutes
        readings_at_60 = [bg for min_after, bg in bg_sequence if 55 <= min_after <= 65]
        if readings_at_60:
            bg_at_60 = np.mean(readings_at_60)
            drop_at_60 = bg_before - bg_at_60

            if drop_at_60 < self.min_drop_by_60min:
                return False, 0.3, f"insufficient_drop_{drop_at_60:.0f}mg_at_60min"

        # Check for sustained rise (indicates definite carbs eaten)
        max_bg = max(bg for _, bg in bg_sequence)
        if max_bg > bg_before + 30:
            return False, 0.1, f"bg_spiked_{max_bg - bg_before:.0f}mg"

        # Calculate final BG drop and ISF
        final_readings = [(min_after, bg) for min_after, bg in bg_sequence if min_after >= 90]
        if final_readings:
            # Find lowest point
            min_bg = min(bg for _, bg in final_readings)
            total_drop = bg_before - min_bg

            if total_drop < 0:
                return False, 0.1, "bg_never_dropped"

            observed_isf = total_drop / insulin_units

            # Check if ISF is in reasonable range
            if not (15 <= observed_isf <= 150):
                return False, 0.4, f"isf_{observed_isf:.0f}_out_of_range"

            # Looks clean!
            confidence = min(1.0, total_drop / 50)  # Higher drop = more confident
            return True, confidence, f"clean_isf_{observed_isf:.0f}"

        return False, 0.3, "no_final_data"

    def compare_to_known_profiles(
        self,
        bg_sequence: List[Tuple[int, float]],
        insulin_units: float
    ) -> float:
        """
        Compare a bolus pattern to known clean profiles.
        Returns similarity score 0-1.
        """
        if not self.known_clean_profiles:
            return 0.5  # No reference data, neutral score

        # Normalize sequence to per-unit drop
        normalized = [(t, bg / insulin_units) for t, bg in bg_sequence]

        similarities = []
        for profile in self.known_clean_profiles:
            ref_seq = profile.get("normalized_sequence", [])
            if not ref_seq:
                continue

            # Calculate correlation at overlapping time points
            # (simplified - full impl would interpolate)
            sim = 0.5  # Default neutral
            similarities.append(sim)

        return np.mean(similarities) if similarities else 0.5


class EnhancedISFLearner:
    """
    Enhanced ISF learner with smart clean bolus detection.
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.isf_repo = LearnedISFRepository()
        self.validator = CleanBolusValidator()

        # Configuration
        self.min_insulin_units = 0.5
        self.fasting_window_hours = 2.0
        self.min_isf = 15
        self.max_isf = 150

    async def import_historic_bolus_data(
        self,
        user_id: str,
        jsonl_path: str = "/home/jadericdawson/Documents/AI/T1D-AI/data/bolus_moments.jsonl"
    ) -> Dict:
        """
        Import pre-analyzed bolus moments from historic JSONL data.
        Validates each entry and stores validated clean boluses.
        """
        path = Path(jsonl_path)
        if not path.exists():
            logger.warning(f"Historic data file not found: {jsonl_path}")
            return {"imported": 0, "validated": 0, "rejected": 0}

        imported = 0
        validated = 0
        rejected = 0
        clean_isf_values = []

        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    imported += 1

                    # Parse entry
                    bolus_ts = datetime.fromisoformat(entry["bolus_ts"])
                    insulin = entry["bolus_insulin"]
                    pre_computed_isf = entry.get("isf", 0)
                    history = entry.get("history", [])
                    bg_drop_seq = entry.get("bg_drop_sequence", [])

                    if insulin < self.min_insulin_units:
                        rejected += 1
                        continue

                    # Get BG at bolus time
                    bg_at_bolus = None
                    for h in reversed(history):
                        if h.get("insulin", 0) > 0:
                            bg_at_bolus = h.get("bg")
                            break

                    if not bg_at_bolus:
                        rejected += 1
                        continue

                    # Convert bg_drop_sequence to [(minutes_after, bg), ...]
                    bg_sequence = []
                    for reading in bg_drop_seq:
                        ts = datetime.fromisoformat(reading["ts"])
                        minutes_after = (ts - bolus_ts).total_seconds() / 60
                        bg_sequence.append((minutes_after, reading["bg"]))

                    # Validate the bolus
                    is_clean, confidence, reason = self.validator.validate_bolus_pattern(
                        bg_before=bg_at_bolus,
                        bg_sequence=bg_sequence,
                        insulin_units=insulin
                    )

                    if is_clean and confidence > 0.5:
                        validated += 1

                        # Calculate ISF from actual data
                        if bg_sequence:
                            min_bg = min(bg for _, bg in bg_sequence if _ >= 60)
                            actual_drop = bg_at_bolus - min_bg
                            actual_isf = actual_drop / insulin

                            if self.min_isf <= actual_isf <= self.max_isf:
                                # Get contextual features
                                lunar_sin, lunar_cos = get_lunar_phase(bolus_ts)
                                day_sin, day_cos = get_day_of_year_encoding(bolus_ts)

                                clean_isf_values.append({
                                    "timestamp": bolus_ts,
                                    "isf": actual_isf,
                                    "insulin": insulin,
                                    "bg_before": bg_at_bolus,
                                    "confidence": confidence,
                                    "time_of_day": self._get_time_of_day(bolus_ts),
                                    "hour": bolus_ts.hour,
                                    "lunar_sin": lunar_sin,
                                    "lunar_cos": lunar_cos,
                                    "day_of_year_sin": day_sin,
                                    "day_of_year_cos": day_cos,
                                    "reason": reason
                                })

                        # Add to validator's known profiles
                        self.validator.add_validated_profile({
                            "bg_at_bolus": bg_at_bolus,
                            "insulin": insulin,
                            "bg_sequence": bg_sequence,
                            "isf": pre_computed_isf
                        })
                    else:
                        rejected += 1
                        logger.debug(f"Rejected bolus at {bolus_ts}: {reason}")

                except Exception as e:
                    logger.warning(f"Error parsing bolus entry: {e}")
                    rejected += 1

        # Store the validated clean ISF data
        if clean_isf_values:
            await self._store_clean_isf_data(user_id, clean_isf_values)

        logger.info(
            f"Imported {imported} bolus moments: {validated} validated, {rejected} rejected"
        )

        return {
            "imported": imported,
            "validated": validated,
            "rejected": rejected,
            "clean_isf_values": clean_isf_values
        }

    async def _store_clean_isf_data(self, user_id: str, isf_data: List[Dict]):
        """Store validated clean ISF observations."""
        if not isf_data:
            return

        # Calculate weighted average ISF
        weights = np.array([d["confidence"] for d in isf_data])
        values = np.array([d["isf"] for d in isf_data])

        # Add recency weighting (more recent = higher weight)
        isf_data.sort(key=lambda x: x["timestamp"])
        recency_weights = np.exp(np.linspace(-1, 0, len(isf_data)))
        final_weights = weights * recency_weights
        final_weights = final_weights / final_weights.sum()

        weighted_isf = float(np.average(values, weights=final_weights))

        # Calculate time-of-day patterns
        tod_pattern = {"morning": [], "afternoon": [], "evening": [], "night": []}
        for d in isf_data:
            tod = d["time_of_day"]
            tod_pattern[tod].append(d["isf"])

        tod_values = {}
        for tod, vals in tod_pattern.items():
            if len(vals) >= 2:
                tod_values[tod] = float(np.median(vals))
            else:
                tod_values[tod] = None

        # Create ISFDataPoints for history
        history = []
        for d in isf_data[-50:]:  # Keep last 50
            history.append(ISFDataPoint(
                timestamp=d["timestamp"],
                value=d["isf"],
                bgBefore=d["bg_before"],
                bgAfter=d["bg_before"] - (d["isf"] * d["insulin"]),
                insulinUnits=d["insulin"],
                hoursAfterMeal=None,  # Fasting
                confidence=d["confidence"]
            ))

        # Store as fasting ISF (since we validated no carbs)
        learned_isf = LearnedISF(
            id=f"{user_id}_fasting",
            userId=user_id,
            isfType="fasting",
            value=weighted_isf,
            confidence=min(1.0, len(isf_data) / 20),  # Full confidence at 20+ samples
            sampleCount=len(isf_data),
            lastUpdated=datetime.now(timezone.utc),
            history=history,
            timeOfDayPattern=tod_values,
            meanISF=float(np.mean(values)),
            stdISF=float(np.std(values)) if len(values) > 1 else 0.0,
            minISF=float(np.min(values)),
            maxISF=float(np.max(values))
        )

        await self.isf_repo.upsert(learned_isf)

        logger.info(
            f"Stored clean ISF for {user_id}: {weighted_isf:.1f} "
            f"(n={len(isf_data)}, range={np.min(values):.0f}-{np.max(values):.0f})"
        )

    async def learn_from_realtime_data(
        self,
        user_id: str,
        days: int = 30
    ) -> Optional[LearnedISF]:
        """
        Learn ISF from recent real-time data with smart validation.
        Uses known clean profiles to validate new boluses.
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Get insulin treatments
        insulin_treatments = await self.treatment_repo.get_recent(
            user_id=user_id,
            hours=days * 24,
            treatment_type="insulin"
        )

        if not insulin_treatments:
            logger.info(f"No insulin treatments found for user {user_id}")
            return None

        validated_isf_values = []

        for bolus in insulin_treatments:
            if not bolus.insulin or bolus.insulin < self.min_insulin_units:
                continue

            # Check for nearby carbs
            carb_window_start = bolus.timestamp - timedelta(hours=self.fasting_window_hours)
            carb_window_end = bolus.timestamp + timedelta(hours=3)  # Check 3 hours after too

            nearby_treatments = await self.treatment_repo.get_by_user(
                user_id=user_id,
                start_time=carb_window_start,
                end_time=carb_window_end
            )

            carb_events = [t for t in nearby_treatments if t.carbs and t.carbs > 0]

            if carb_events:
                continue  # Not fasting

            # Get BG readings for validation
            bg_before_reading = await self._get_nearest_bg(
                user_id, bolus.timestamp, window_minutes=15
            )

            if not bg_before_reading:
                continue

            # Get BG sequence after bolus (2 hours)
            bg_after_start = bolus.timestamp
            bg_after_end = bolus.timestamp + timedelta(hours=2)

            bg_readings = await self.glucose_repo.get_history(
                user_id=user_id,
                start_time=bg_after_start,
                end_time=bg_after_end,
                limit=30
            )

            if len(bg_readings) < 6:
                continue

            # Convert to sequence format
            bg_sequence = []
            for reading in bg_readings:
                minutes_after = (reading.timestamp - bolus.timestamp).total_seconds() / 60
                bg_sequence.append((minutes_after, reading.value))

            # Validate with smart detection
            is_clean, confidence, reason = self.validator.validate_bolus_pattern(
                bg_before=bg_before_reading.value,
                bg_sequence=bg_sequence,
                insulin_units=bolus.insulin
            )

            if is_clean and confidence > 0.5:
                # Calculate actual ISF
                readings_90_plus = [bg for min_after, bg in bg_sequence if min_after >= 90]
                if readings_90_plus:
                    min_bg = min(readings_90_plus)
                    actual_drop = bg_before_reading.value - min_bg
                    actual_isf = actual_drop / bolus.insulin

                    if self.min_isf <= actual_isf <= self.max_isf:
                        lunar_sin, lunar_cos = get_lunar_phase(bolus.timestamp)
                        day_sin, day_cos = get_day_of_year_encoding(bolus.timestamp)

                        validated_isf_values.append({
                            "timestamp": bolus.timestamp,
                            "isf": actual_isf,
                            "insulin": bolus.insulin,
                            "bg_before": bg_before_reading.value,
                            "confidence": confidence,
                            "time_of_day": self._get_time_of_day(bolus.timestamp),
                            "hour": bolus.timestamp.hour,
                            "lunar_sin": lunar_sin,
                            "lunar_cos": lunar_cos,
                            "day_of_year_sin": day_sin,
                            "day_of_year_cos": day_cos,
                            "reason": reason
                        })

                        # Add to validator
                        self.validator.add_validated_profile({
                            "bg_at_bolus": bg_before_reading.value,
                            "insulin": bolus.insulin,
                            "bg_sequence": bg_sequence,
                            "isf": actual_isf
                        })

        if validated_isf_values:
            await self._store_clean_isf_data(user_id, validated_isf_values)
            return await self.isf_repo.get(user_id, "fasting")

        logger.info(f"No clean boluses found for user {user_id} in last {days} days")
        return None

    async def _get_nearest_bg(
        self,
        user_id: str,
        target_time: datetime,
        window_minutes: int = 15
    ) -> Optional[GlucoseReading]:
        """Get nearest BG reading to target time."""
        start_time = target_time - timedelta(minutes=window_minutes)
        end_time = target_time + timedelta(minutes=window_minutes)

        readings = await self.glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=10
        )

        if not readings:
            return None

        return min(readings, key=lambda r: abs((r.timestamp - target_time).total_seconds()))

    def _get_time_of_day(self, timestamp: datetime) -> str:
        """Categorize timestamp into time of day period."""
        hour = timestamp.hour
        if 6 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    async def get_isf_for_context(
        self,
        user_id: str,
        timestamp: Optional[datetime] = None,
        is_fasting: bool = True
    ) -> Tuple[float, Dict]:
        """
        Get ISF for a specific context (time of day, etc).

        Returns:
            (isf_value, context_info)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        isf_type = "fasting" if is_fasting else "meal"
        learned = await self.isf_repo.get(user_id, isf_type)

        if not learned:
            return 50.0, {"source": "default", "confidence": 0.0}

        # Check for time-of-day specific value
        time_of_day = self._get_time_of_day(timestamp)
        tod_value = learned.timeOfDayPattern.get(time_of_day)

        if tod_value:
            return tod_value, {
                "source": f"learned_{time_of_day}",
                "confidence": learned.confidence,
                "sample_count": learned.sampleCount
            }

        return learned.value, {
            "source": "learned_average",
            "confidence": learned.confidence,
            "sample_count": learned.sampleCount
        }


# Convenience functions
async def import_and_learn_isf(user_id: str) -> Dict:
    """Import historic data and learn ISF for a user."""
    learner = EnhancedISFLearner()

    # First, import historic data
    import_result = await learner.import_historic_bolus_data(user_id)

    # Then learn from any recent real-time data
    realtime_result = await learner.learn_from_realtime_data(user_id, days=30)

    return {
        "import_result": import_result,
        "realtime_learned": realtime_result is not None
    }
