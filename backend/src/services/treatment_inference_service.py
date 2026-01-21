"""
Enhanced Treatment Inference Service for T1D-AI

Uses curve fitting to detect unlogged treatments (both carbs AND insulin)
by finding the combination that best explains observed BG patterns.

Algorithm:
1. Get actual BG curve over observation window
2. Calculate "expected" BG from known IOB/COB
3. Compute residual (actual - expected)
4. Grid search to find (carbs, insulin, timing) that minimizes residual
5. Create inferred treatments if confidence is high enough
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from uuid import uuid4
import math

from models.schemas import (
    Treatment, TreatmentType, GlucoseReading, InferenceStatus
)
from database.repositories import (
    TreatmentRepository, GlucoseRepository, DataSourceRepository
)
from services.gluroo_service import GlurooService
from services.iob_cob_service import IOBCOBService
from utils.encryption import decrypt_secret
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceCandidate:
    """A candidate solution for what might explain the BG pattern."""
    carbs: float  # Estimated carbs (grams)
    insulin: float  # Estimated insulin (units)
    carb_time: datetime  # When carbs were likely consumed
    insulin_time: datetime  # When insulin was likely given
    residual_error: float  # How well this explains the data (lower = better)
    confidence: float  # 0-1 confidence score


@dataclass
class BGSimulation:
    """Result of simulating BG with given treatments."""
    timestamps: List[datetime]
    predicted_values: List[float]
    residual: float  # Sum of squared errors vs actual


class TreatmentInferenceService:
    """
    Detects unlogged treatments by curve fitting.

    Uses physics-based BG simulation to find the carb/insulin combination
    that best explains observed glucose patterns.
    """

    # Simulation parameters
    OBSERVATION_WINDOW_HOURS = 3  # How far back to look
    SIMULATION_STEP_MINUTES = 5  # Granularity of simulation
    MIN_BG_DEVIATION = 20  # Minimum deviation to trigger inference
    MIN_CONFIDENCE = 0.6  # Minimum confidence to create inference
    COOLDOWN_MINUTES = 90  # Don't infer again within this time

    # Grid search ranges
    CARB_SEARCH_RANGE = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]  # grams
    INSULIN_SEARCH_RANGE = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # units
    TIME_OFFSET_RANGE = [-30, -15, 0, 15, 30]  # minutes from detection point

    # Physiological constants (can be personalized)
    DEFAULT_ISF = 50.0  # mg/dL per unit
    DEFAULT_ICR = 10.0  # grams per unit
    CARB_FACTOR = 4.0  # mg/dL rise per gram carb
    INSULIN_HALF_LIFE = 54  # minutes (child-specific, faster than adult 81)
    CARB_HALF_LIFE = 45  # minutes

    def __init__(self):
        self.settings = get_settings()
        self.treatment_repo = TreatmentRepository()
        self.glucose_repo = GlucoseRepository()
        self.datasource_repo = DataSourceRepository()
        self.iob_cob_service = IOBCOBService.from_settings()

    async def detect_unlogged_treatments(
        self,
        user_id: str,
        recent_glucose: List[GlucoseReading],
        known_treatments: Optional[List[Treatment]] = None
    ) -> List[Treatment]:
        """
        Analyze BG patterns to detect unlogged carbs and/or insulin.

        Args:
            user_id: User ID
            recent_glucose: Recent glucose readings (newest first)
            known_treatments: Already logged treatments in this window

        Returns:
            List of inferred treatments (may be empty, or contain carbs and/or insulin)
        """
        if len(recent_glucose) < 12:  # Need ~1 hour of data
            return []

        # Sort by timestamp (oldest first for simulation)
        readings = sorted(recent_glucose, key=lambda r: r.timestamp)

        # Check cooldown
        recent_inferred = await self._get_recent_inferred(user_id)
        if recent_inferred:
            time_since = datetime.now(timezone.utc) - recent_inferred.timestamp
            if time_since.total_seconds() < self.COOLDOWN_MINUTES * 60:
                logger.debug("Cooldown active, skipping inference")
                return []

        # Get known treatments if not provided
        if known_treatments is None:
            known_treatments = await self._get_known_treatments(user_id)

        # Calculate expected BG from known treatments
        expected_bg = self._simulate_bg_from_treatments(
            readings[0].value,  # Starting BG
            readings[0].timestamp,
            known_treatments,
            [r.timestamp for r in readings]
        )

        # Calculate residuals (actual - expected)
        actual_values = [r.value for r in readings]
        residuals = [a - e for a, e in zip(actual_values, expected_bg)]

        # Check if residuals are significant enough
        max_residual = max(abs(r) for r in residuals)
        if max_residual < self.MIN_BG_DEVIATION:
            logger.debug(f"Max residual {max_residual:.1f} below threshold, no inference needed")
            return []

        # Determine pattern type
        pattern = self._analyze_residual_pattern(residuals, readings)
        logger.info(f"Detected pattern: {pattern['type']} (max residual: {max_residual:.1f})")

        # Grid search for best explanation
        best_candidate = self._grid_search_best_fit(
            readings,
            known_treatments,
            pattern
        )

        if not best_candidate or best_candidate.confidence < self.MIN_CONFIDENCE:
            logger.debug(f"No confident inference found (best confidence: {best_candidate.confidence if best_candidate else 0:.2f})")
            return []

        # Create inferred treatments
        inferred = []

        if best_candidate.carbs > 0:
            carb_treatment = await self._create_inferred_treatment(
                user_id=user_id,
                treatment_type=TreatmentType.CARBS,
                carbs=best_candidate.carbs,
                timestamp=best_candidate.carb_time,
                confidence=best_candidate.confidence,
                reason=self._build_carb_reason(best_candidate, pattern)
            )
            inferred.append(carb_treatment)
            await self._push_to_gluroo(user_id, carb_treatment)

        if best_candidate.insulin > 0:
            insulin_treatment = await self._create_inferred_treatment(
                user_id=user_id,
                treatment_type=TreatmentType.INSULIN,
                insulin=best_candidate.insulin,
                timestamp=best_candidate.insulin_time,
                confidence=best_candidate.confidence,
                reason=self._build_insulin_reason(best_candidate, pattern)
            )
            inferred.append(insulin_treatment)
            await self._push_to_gluroo(user_id, insulin_treatment)

        return inferred

    def _analyze_residual_pattern(
        self,
        residuals: List[float],
        readings: List[GlucoseReading]
    ) -> dict:
        """
        Analyze the residual pattern to determine what type of unlogged treatment.

        Returns dict with:
            - type: 'rise', 'drop', 'rise_then_drop', 'drop_then_rise', 'complex'
            - peak_time: When the max deviation occurred
            - magnitude: Size of deviation
        """
        if not residuals:
            return {'type': 'none', 'magnitude': 0}

        # Find peak positive and negative residuals
        max_positive = max(residuals)
        max_negative = min(residuals)
        max_positive_idx = residuals.index(max_positive)
        max_negative_idx = residuals.index(max_negative)

        # Calculate trends in first and second half
        mid = len(residuals) // 2
        first_half_trend = sum(residuals[:mid]) / max(1, mid)
        second_half_trend = sum(residuals[mid:]) / max(1, len(residuals) - mid)

        pattern = {
            'max_positive': max_positive,
            'max_negative': max_negative,
            'peak_positive_time': readings[max_positive_idx].timestamp if max_positive_idx < len(readings) else None,
            'peak_negative_time': readings[max_negative_idx].timestamp if max_negative_idx < len(readings) else None,
        }

        # Determine pattern type
        if max_positive > 25 and abs(max_negative) < 15:
            pattern['type'] = 'rise'  # Unlogged carbs
            pattern['magnitude'] = max_positive
        elif abs(max_negative) > 25 and max_positive < 15:
            pattern['type'] = 'drop'  # Unlogged insulin
            pattern['magnitude'] = abs(max_negative)
        elif max_positive > 20 and abs(max_negative) > 20:
            if max_positive_idx < max_negative_idx:
                pattern['type'] = 'rise_then_drop'  # Carbs then insulin
            else:
                pattern['type'] = 'drop_then_rise'  # Insulin then carbs (less common)
            pattern['magnitude'] = max(max_positive, abs(max_negative))
        else:
            pattern['type'] = 'minor'
            pattern['magnitude'] = max(max_positive, abs(max_negative))

        return pattern

    def _grid_search_best_fit(
        self,
        readings: List[GlucoseReading],
        known_treatments: List[Treatment],
        pattern: dict
    ) -> Optional[InferenceCandidate]:
        """
        Grid search to find the best carb/insulin combination.
        """
        best_candidate = None
        best_error = float('inf')

        # Determine search strategy based on pattern
        if pattern['type'] == 'rise':
            # Only search carbs
            carb_range = self.CARB_SEARCH_RANGE
            insulin_range = [0]
        elif pattern['type'] == 'drop':
            # Only search insulin
            carb_range = [0]
            insulin_range = self.INSULIN_SEARCH_RANGE
        else:
            # Search both (more expensive)
            carb_range = self.CARB_SEARCH_RANGE
            insulin_range = self.INSULIN_SEARCH_RANGE

        # Reference point for timing (when we first noticed deviation)
        ref_time = pattern.get('peak_positive_time') or pattern.get('peak_negative_time') or readings[-1].timestamp

        actual_values = [r.value for r in readings]
        timestamps = [r.timestamp for r in readings]
        start_bg = readings[0].value

        for carbs in carb_range:
            for insulin in insulin_range:
                for time_offset in self.TIME_OFFSET_RANGE:
                    # Skip if both are zero
                    if carbs == 0 and insulin == 0:
                        continue

                    # Calculate treatment times
                    carb_time = ref_time - timedelta(minutes=60 + time_offset) if carbs > 0 else ref_time
                    insulin_time = ref_time - timedelta(minutes=30 + time_offset) if insulin > 0 else ref_time

                    # Build hypothetical treatments
                    hypothetical = list(known_treatments)
                    if carbs > 0:
                        hypothetical.append(self._make_temp_treatment(
                            TreatmentType.CARBS, carbs=carbs, timestamp=carb_time
                        ))
                    if insulin > 0:
                        hypothetical.append(self._make_temp_treatment(
                            TreatmentType.INSULIN, insulin=insulin, timestamp=insulin_time
                        ))

                    # Simulate BG with these hypothetical treatments
                    predicted = self._simulate_bg_from_treatments(
                        start_bg, timestamps[0], hypothetical, timestamps
                    )

                    # Calculate error
                    error = sum((a - p) ** 2 for a, p in zip(actual_values, predicted))

                    if error < best_error:
                        best_error = error
                        best_candidate = InferenceCandidate(
                            carbs=carbs,
                            insulin=insulin,
                            carb_time=carb_time,
                            insulin_time=insulin_time,
                            residual_error=error,
                            confidence=self._calculate_confidence(error, actual_values, pattern)
                        )

        return best_candidate

    def _simulate_bg_from_treatments(
        self,
        start_bg: float,
        start_time: datetime,
        treatments: List[Treatment],
        timestamps: List[datetime]
    ) -> List[float]:
        """
        Simulate BG values based on treatments using physiological model.
        """
        predictions = []

        for ts in timestamps:
            bg = start_bg

            for t in treatments:
                if t.timestamp > ts:
                    continue  # Treatment hasn't happened yet

                minutes_elapsed = (ts - t.timestamp).total_seconds() / 60

                if minutes_elapsed < 0:
                    continue

                # Insulin effect (drops BG)
                if t.insulin and t.insulin > 0:
                    # Exponential decay model
                    insulin_remaining = t.insulin * (0.5 ** (minutes_elapsed / self.INSULIN_HALF_LIFE))
                    insulin_absorbed = t.insulin - insulin_remaining
                    bg -= insulin_absorbed * self.DEFAULT_ISF

                # Carb effect (raises BG)
                if t.carbs and t.carbs > 0:
                    # Exponential absorption model
                    carbs_remaining = t.carbs * (0.5 ** (minutes_elapsed / self.CARB_HALF_LIFE))
                    carbs_absorbed = t.carbs - carbs_remaining
                    bg += carbs_absorbed * self.CARB_FACTOR

            predictions.append(max(40, min(400, bg)))  # Clamp to reasonable range

        return predictions

    def _calculate_confidence(
        self,
        error: float,
        actual_values: List[float],
        pattern: dict
    ) -> float:
        """
        Calculate confidence score for an inference.

        Based on:
        - How well the fit explains the data (low error = high confidence)
        - Pattern clarity (clear rise/drop = higher confidence)
        - Magnitude of deviation
        """
        if not actual_values:
            return 0.0

        # Normalize error by variance in actual data
        variance = sum((v - sum(actual_values)/len(actual_values))**2 for v in actual_values) / len(actual_values)
        if variance < 1:
            variance = 1

        normalized_error = error / (len(actual_values) * variance)

        # Error-based confidence (lower error = higher confidence)
        error_confidence = max(0, 1 - normalized_error / 10)

        # Pattern clarity bonus
        pattern_bonus = 0
        if pattern['type'] in ['rise', 'drop']:
            pattern_bonus = 0.2  # Clear single-direction pattern
        elif pattern['type'] == 'rise_then_drop':
            pattern_bonus = 0.1  # Recognizable pattern

        # Magnitude bonus (bigger deviation = more confident it's real)
        magnitude = pattern.get('magnitude', 0)
        magnitude_bonus = min(0.2, magnitude / 200)

        confidence = error_confidence + pattern_bonus + magnitude_bonus
        return min(0.95, max(0.0, confidence))

    def _make_temp_treatment(
        self,
        treatment_type: TreatmentType,
        carbs: float = None,
        insulin: float = None,
        timestamp: datetime = None
    ) -> Treatment:
        """Create a temporary treatment object for simulation."""
        return Treatment(
            id="temp",
            userId="temp",
            timestamp=timestamp or datetime.now(timezone.utc),
            type=treatment_type,
            carbs=carbs,
            insulin=insulin,
            source="simulation"
        )

    async def _get_recent_inferred(self, user_id: str) -> Optional[Treatment]:
        """Get most recent inferred treatment."""
        try:
            treatments = await self.treatment_repo.get_recent(
                user_id=user_id, hours=3
            )
            for t in treatments:
                if t.isInferred:
                    return t
            return None
        except Exception:
            return None

    async def _get_known_treatments(self, user_id: str) -> List[Treatment]:
        """Get logged treatments in observation window."""
        try:
            return await self.treatment_repo.get_recent(
                user_id=user_id,
                hours=self.OBSERVATION_WINDOW_HOURS
            )
        except Exception:
            return []

    async def _create_inferred_treatment(
        self,
        user_id: str,
        treatment_type: TreatmentType,
        timestamp: datetime,
        confidence: float,
        reason: str,
        carbs: float = None,
        insulin: float = None
    ) -> Treatment:
        """Create and store an inferred treatment."""
        treatment = Treatment(
            id=f"{user_id}_{uuid4().hex[:12]}",
            userId=user_id,
            timestamp=timestamp,
            type=treatment_type,
            carbs=carbs,
            insulin=insulin,
            notes=f"🔍 {'Suspected unlogged carbs' if carbs else 'Suspected unlogged insulin'} (AI detected)",
            source="t1d-ai-inference",
            isInferred=True,
            inferenceConfidence=confidence,
            inferenceReason=reason,
            confirmationStatus=InferenceStatus.PENDING
        )

        created = await self.treatment_repo.create(treatment)
        logger.info(
            f"Created inferred treatment: {treatment_type.value} "
            f"({'carbs=' + str(carbs) + 'g' if carbs else 'insulin=' + str(insulin) + 'U'}) "
            f"confidence={confidence:.2f}"
        )
        return created

    def _build_carb_reason(self, candidate: InferenceCandidate, pattern: dict) -> str:
        """Build human-readable reason for carb inference."""
        return (
            f"BG rose {pattern.get('max_positive', 0):.0f} mg/dL without logged carbs. "
            f"Pattern best explained by ~{candidate.carbs:.0f}g carbs consumed around "
            f"{candidate.carb_time.strftime('%H:%M')}."
        )

    def _build_insulin_reason(self, candidate: InferenceCandidate, pattern: dict) -> str:
        """Build human-readable reason for insulin inference."""
        return (
            f"BG dropped {abs(pattern.get('max_negative', 0)):.0f} mg/dL without logged insulin. "
            f"Pattern best explained by ~{candidate.insulin:.1f}U given around "
            f"{candidate.insulin_time.strftime('%H:%M')}."
        )

    async def _push_to_gluroo(self, user_id: str, treatment: Treatment):
        """Push inferred treatment to Gluroo."""
        try:
            datasource = await self.datasource_repo.get(user_id, "gluroo")
            if not datasource or not datasource.credentials:
                return

            creds = datasource.credentials
            if not creds.syncEnabled or not creds.url or not creds.apiSecretEncrypted:
                return

            api_secret = decrypt_secret(creds.apiSecretEncrypted)
            service = GlurooService(base_url=creds.url, api_secret=api_secret)

            if treatment.carbs:
                await service.push_treatment(
                    treatment_type="carbs",
                    value=treatment.carbs,
                    timestamp=treatment.timestamp,
                    notes=treatment.notes
                )
            elif treatment.insulin:
                await service.push_treatment(
                    treatment_type="insulin",
                    value=treatment.insulin,
                    timestamp=treatment.timestamp,
                    notes=treatment.notes
                )

            logger.info(f"Pushed inferred treatment to Gluroo")

        except Exception as e:
            logger.warning(f"Failed to push inferred treatment to Gluroo: {e}")


# Singleton instance
_treatment_inference_service: Optional[TreatmentInferenceService] = None


def get_treatment_inference_service() -> TreatmentInferenceService:
    """Get or create the treatment inference service singleton."""
    global _treatment_inference_service
    if _treatment_inference_service is None:
        _treatment_inference_service = TreatmentInferenceService()
    return _treatment_inference_service
