"""
Food Inference Service for T1D-AI
Detects unlogged food consumption by monitoring BG deviations from predictions.
Creates inferred treatment entries for suspected snacks.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
from uuid import uuid4

from models.schemas import (
    Treatment, TreatmentType, GlucoseReading, InferenceStatus
)
from database.repositories import (
    TreatmentRepository, GlucoseRepository, DataSourceRepository
)
from services.gluroo_service import GlurooService
from utils.encryption import decrypt_secret
from config import get_settings

logger = logging.getLogger(__name__)


class FoodInferenceService:
    """
    Detects unlogged food consumption by analyzing BG patterns.

    When actual BG rises significantly above predictions without logged carbs,
    creates an inferred treatment entry that can be confirmed/edited by user.
    """

    # Detection thresholds
    MIN_DEVIATION_MGDL = 25  # Minimum deviation to trigger detection
    MIN_RISE_MGDL = 30  # Minimum total BG rise to consider
    DETECTION_WINDOW_MINUTES = 45  # Time window to detect rising pattern
    COOLDOWN_MINUTES = 60  # Don't create another inference within this time

    # Carb estimation constants
    DEFAULT_ICR = 10.0  # grams per unit
    DEFAULT_ISF = 50.0  # mg/dL per unit

    def __init__(self):
        self.settings = get_settings()
        self.treatment_repo = TreatmentRepository()
        self.glucose_repo = GlucoseRepository()
        self.datasource_repo = DataSourceRepository()

    async def check_for_unlogged_food(
        self,
        user_id: str,
        recent_glucose: List[GlucoseReading],
        predicted_values: Optional[List[float]] = None
    ) -> Optional[Treatment]:
        """
        Analyze recent glucose data to detect potential unlogged food.

        Args:
            user_id: User ID
            recent_glucose: Recent glucose readings (newest first)
            predicted_values: Predicted glucose values for comparison

        Returns:
            Created inferred treatment if food detected, None otherwise
        """
        if len(recent_glucose) < 6:  # Need at least 30 min of data
            return None

        # Sort by timestamp (oldest first for analysis)
        readings = sorted(recent_glucose, key=lambda r: r.timestamp)

        # Check if there's a recent inferred treatment (cooldown)
        recent_inferred = await self._get_recent_inferred_treatment(user_id)
        if recent_inferred:
            time_since = datetime.now(timezone.utc) - recent_inferred.timestamp
            if time_since.total_seconds() < self.COOLDOWN_MINUTES * 60:
                logger.debug(f"Cooldown active, skipping inference check")
                return None

        # Check for recent logged carbs (don't infer if carbs were logged)
        recent_carbs = await self._get_recent_carbs(user_id, minutes=90)
        if recent_carbs:
            logger.debug(f"Recent carbs logged, skipping inference")
            return None

        # Detect rising pattern
        rise_detected, start_bg, peak_bg, rise_start_time = self._detect_rising_pattern(readings)

        if not rise_detected:
            return None

        # Calculate deviation from prediction if available
        deviation = 0
        if predicted_values and len(predicted_values) > 0:
            current_bg = readings[-1].value
            predicted_bg = predicted_values[-1] if predicted_values else current_bg
            deviation = current_bg - predicted_bg

            if deviation < self.MIN_DEVIATION_MGDL:
                logger.debug(f"Deviation {deviation} below threshold")
                return None

        # Calculate estimated carbs from BG rise
        total_rise = peak_bg - start_bg
        estimated_carbs = self._estimate_carbs_from_rise(total_rise, user_id)

        # Create inferred treatment
        treatment = await self._create_inferred_treatment(
            user_id=user_id,
            timestamp=rise_start_time,
            estimated_carbs=estimated_carbs,
            bg_rise=total_rise,
            deviation=deviation
        )

        # Push to Gluroo
        await self._push_to_gluroo(user_id, treatment)

        logger.info(
            f"Detected unlogged food for user {user_id}: "
            f"BG rise {total_rise} mg/dL, estimated {estimated_carbs}g carbs"
        )

        return treatment

    def _detect_rising_pattern(
        self,
        readings: List[GlucoseReading]
    ) -> Tuple[bool, int, int, datetime]:
        """
        Detect a sustained rising pattern in glucose readings.

        Returns:
            (detected, start_bg, peak_bg, rise_start_time)
        """
        if len(readings) < 6:
            return False, 0, 0, datetime.now(timezone.utc)

        # Look at last ~45 minutes of data
        window_readings = []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.DETECTION_WINDOW_MINUTES)
        for r in readings:
            if r.timestamp >= cutoff:
                window_readings.append(r)

        if len(window_readings) < 4:
            return False, 0, 0, datetime.now(timezone.utc)

        # Find the minimum (potential start of rise) and maximum (current/peak)
        min_reading = min(window_readings, key=lambda r: r.value)
        max_reading = max(window_readings, key=lambda r: r.value)

        # Check if max came after min (rising pattern)
        if max_reading.timestamp <= min_reading.timestamp:
            return False, 0, 0, datetime.now(timezone.utc)

        # Check if rise is significant
        rise = max_reading.value - min_reading.value
        if rise < self.MIN_RISE_MGDL:
            return False, 0, 0, datetime.now(timezone.utc)

        # Check trend direction (should be rising or flat-ish at peak)
        recent_readings = window_readings[-3:]
        trend_values = [r.value for r in recent_readings]

        # Calculate simple trend (positive = rising)
        if len(trend_values) >= 2:
            trend = (trend_values[-1] - trend_values[0]) / len(trend_values)
            if trend < -5:  # Dropping significantly, might be coming down already
                # Still report if we had a significant rise
                pass

        return True, min_reading.value, max_reading.value, min_reading.timestamp

    def _estimate_carbs_from_rise(
        self,
        bg_rise: float,
        user_id: str
    ) -> float:
        """
        Estimate carbs consumed based on BG rise.

        Uses formula: carbs = (BG_rise / ISF) * ICR
        This is the inverse of the bolus calculation.
        """
        # TODO: Get personalized ICR/ISF from user settings
        isf = self.DEFAULT_ISF
        icr = self.DEFAULT_ICR

        # Calculate: how many units would have caused this rise?
        # units = bg_rise / isf
        # Then: carbs = units * icr
        estimated_units = bg_rise / isf
        estimated_carbs = estimated_units * icr

        # Round to reasonable precision
        return round(estimated_carbs, 0)

    async def _get_recent_inferred_treatment(
        self,
        user_id: str
    ) -> Optional[Treatment]:
        """Get the most recent inferred treatment within cooldown window."""
        try:
            treatments = await self.treatment_repo.get_recent(
                user_id=user_id,
                hours=2,
                treatment_type="carbs"
            )

            for t in treatments:
                if t.isInferred:
                    return t
            return None
        except Exception as e:
            logger.error(f"Error getting recent inferred treatments: {e}")
            return None

    async def _get_recent_carbs(
        self,
        user_id: str,
        minutes: int = 90
    ) -> Optional[Treatment]:
        """Check if carbs were logged recently."""
        try:
            treatments = await self.treatment_repo.get_recent(
                user_id=user_id,
                hours=max(1, minutes // 60 + 1),
                treatment_type="carbs"
            )

            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            for t in treatments:
                if t.timestamp >= cutoff and not t.isInferred:
                    return t
            return None
        except Exception as e:
            logger.error(f"Error checking recent carbs: {e}")
            return None

    async def _create_inferred_treatment(
        self,
        user_id: str,
        timestamp: datetime,
        estimated_carbs: float,
        bg_rise: float,
        deviation: float
    ) -> Treatment:
        """Create an inferred treatment for suspected unlogged food."""
        treatment = Treatment(
            id=f"{user_id}_{uuid4().hex[:12]}",
            userId=user_id,
            timestamp=timestamp,
            type=TreatmentType.CARBS,
            carbs=estimated_carbs,
            notes=f"🔍 Suspected unlogged food (estimated from {bg_rise:.0f} mg/dL BG rise)",
            source="t1d-ai-inference",
            isInferred=True,
            inferenceConfidence=min(0.9, 0.5 + (bg_rise / 100)),  # Higher rise = higher confidence
            inferenceReason=f"BG rose {bg_rise:.0f} mg/dL without logged carbs. Deviation from prediction: {deviation:.0f} mg/dL.",
            confirmationStatus=InferenceStatus.PENDING
        )

        created = await self.treatment_repo.create(treatment)
        return created

    async def _push_to_gluroo(self, user_id: str, treatment: Treatment):
        """Push inferred treatment to Gluroo."""
        try:
            datasource = await self.datasource_repo.get(user_id, "gluroo")
            if not datasource:
                return

            creds = datasource.credentials
            if not creds or not creds.syncEnabled:
                return

            url = creds.url
            api_secret_encrypted = creds.apiSecretEncrypted

            if not url or not api_secret_encrypted:
                return

            api_secret = decrypt_secret(api_secret_encrypted)
            service = GlurooService(base_url=url, api_secret=api_secret)

            success, message, _ = await service.push_treatment(
                treatment_type="carbs",
                value=treatment.carbs,
                timestamp=treatment.timestamp,
                notes=treatment.notes
            )

            if success:
                logger.info(f"Pushed inferred treatment to Gluroo for user {user_id}")
            else:
                logger.warning(f"Failed to push inferred treatment to Gluroo: {message}")

        except Exception as e:
            logger.error(f"Error pushing inferred treatment to Gluroo: {e}")

    async def update_inferred_carbs_from_peak(
        self,
        user_id: str,
        treatment_id: str,
        final_bg_rise: float
    ) -> Optional[Treatment]:
        """
        Update an inferred treatment with refined carb estimate after BG peak.

        Called when we detect the BG has peaked and started to come down,
        giving us the full picture of the food impact.
        """
        try:
            treatment = await self.treatment_repo.get_by_id(treatment_id, user_id)
            if not treatment or not treatment.isInferred:
                return None

            # Recalculate with final rise
            new_carbs = self._estimate_carbs_from_rise(final_bg_rise, user_id)

            treatment.carbs = new_carbs
            treatment.inferenceReason = (
                f"BG peaked at +{final_bg_rise:.0f} mg/dL. "
                f"Updated estimate: {new_carbs:.0f}g carbs."
            )

            updated = await self.treatment_repo.upsert(treatment)

            # Update in Gluroo too
            await self._push_to_gluroo(user_id, updated)

            return updated

        except Exception as e:
            logger.error(f"Error updating inferred treatment: {e}")
            return None


# Singleton instance
_food_inference_service: Optional[FoodInferenceService] = None


def get_food_inference_service() -> FoodInferenceService:
    """Get or create the food inference service singleton."""
    global _food_inference_service
    if _food_inference_service is None:
        _food_inference_service = FoodInferenceService()
    return _food_inference_service
