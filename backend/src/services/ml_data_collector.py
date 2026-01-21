"""
ML Data Collector Service for T1D-AI

Collects prediction vs actual BG data for ML training.
This enables the system to learn from real outcomes and improve over time.

Flow:
1. On treatment logged → store prediction context
2. Schedule checks at +30, +60, +90 min
3. Capture actual BG at each checkpoint
4. Calculate prediction error
5. Store complete training point
"""
import math
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from models.schemas import (
    Treatment, MLTrainingDataPoint, GlucoseReading
)
from database.repositories import (
    MLTrainingDataRepository, GlucoseRepository, TreatmentRepository
)
from ml.models.hybrid_absorption_model import HybridAbsorptionModel
from services.iob_cob_service import IOBCOBService

logger = logging.getLogger(__name__)


def normalize_food_description(description: str) -> str:
    """
    Normalize food description for consistent matching.

    Removes case differences, extra whitespace, and common variations.
    """
    if not description:
        return ""

    # Lowercase and strip
    normalized = description.lower().strip()

    # Remove common filler words
    filler_words = ['a', 'an', 'the', 'some', 'few', 'lot', 'of', 'with', 'and']
    words = normalized.split()
    words = [w for w in words if w not in filler_words]

    # Rejoin and remove extra whitespace
    normalized = ' '.join(words)

    return normalized


def generate_food_id(description: str) -> str:
    """
    Generate a stable food ID from the normalized description.

    Uses SHA256 hash truncated to 12 characters for readability.
    """
    normalized = normalize_food_description(description)
    if not normalized:
        return "unknown"

    hash_obj = hashlib.sha256(normalized.encode())
    return hash_obj.hexdigest()[:12]


def get_lunar_phase(dt: datetime) -> Tuple[float, float]:
    """
    Calculate lunar phase as sine and cosine components.

    Used as a cyclical feature that may correlate with hormonal changes.
    Returns (sin, cos) where both range from -1 to 1.
    """
    # Reference date: known new moon
    ref_date = datetime(2000, 1, 6, 18, 14)
    lunar_cycle_days = 29.530588853

    days_since_ref = (dt - ref_date).total_seconds() / 86400
    phase = (days_since_ref % lunar_cycle_days) / lunar_cycle_days
    theta = 2 * math.pi * phase

    return math.sin(theta), math.cos(theta)


class MLDataCollector:
    """
    Collects clean training data for ML model improvement.

    This service:
    1. Captures prediction context when meals are logged
    2. Collects actual BG at +30, +60, +90 minutes
    3. Computes prediction errors for model training
    4. Only stores "clean" meals (no overlapping treatments)
    """

    # Checkpoints at which to collect actual BG (minutes after meal)
    CHECKPOINTS = [30, 60, 90]

    # Window before/after checkpoint to look for BG reading (minutes)
    BG_WINDOW = 5

    # Minimum time between meals to be considered "clean" (minutes)
    MIN_MEAL_GAP = 90

    def __init__(
        self,
        training_repo: Optional[MLTrainingDataRepository] = None,
        glucose_repo: Optional[GlucoseRepository] = None,
        treatment_repo: Optional[TreatmentRepository] = None,
        iob_cob_service: Optional[IOBCOBService] = None,
        absorption_model: Optional[HybridAbsorptionModel] = None
    ):
        self.training_repo = training_repo or MLTrainingDataRepository()
        self.glucose_repo = glucose_repo or GlucoseRepository()
        self.treatment_repo = treatment_repo or TreatmentRepository()
        self.iob_cob_service = iob_cob_service or IOBCOBService.from_settings()
        self.absorption_model = absorption_model or HybridAbsorptionModel()

    async def on_treatment_logged(
        self,
        treatment: Treatment,
        current_bg: float,
        bg_trend: int = 0,
        isf: float = 50.0,
        icr: float = 10.0,
        weather: Optional[dict] = None
    ) -> Optional[MLTrainingDataPoint]:
        """
        Called when a carb treatment is logged.

        Creates a training data point with prediction context.
        The actual BG values will be filled in later via checkpoint collection.

        Args:
            treatment: The logged carb treatment
            current_bg: Current BG at time of meal
            bg_trend: Current trend direction (-3 to +3)
            isf: User's ISF
            icr: User's ICR
            weather: Optional weather data dict with temp_c, humidity, pressure

        Returns:
            Created MLTrainingDataPoint or None if not eligible
        """
        # Only process carb treatments
        if not treatment.carbs or treatment.carbs <= 0:
            logger.debug(f"Skipping non-carb treatment: {treatment.id}")
            return None

        user_id = treatment.userId
        meal_time = treatment.timestamp.replace(tzinfo=None)

        # Check if this is a "clean" meal (no overlapping treatments)
        is_clean, quality_score = await self._check_meal_cleanliness(
            user_id, meal_time, treatment.id
        )

        if not is_clean:
            logger.info(f"Skipping non-clean meal: {treatment.id} (quality={quality_score:.2f})")
            # Still create the data point but mark it as not clean
            # It can still be useful for analysis, just not for training

        # Generate food ID from description
        food_description = treatment.notes or f"{treatment.carbs}g carbs"
        food_id = generate_food_id(food_description)

        # Get absorption parameters for prediction
        gi = treatment.glycemicIndex or 55
        is_liquid = treatment.isLiquid or False
        fat_g = treatment.fat or 0
        protein_g = treatment.protein or 0
        fiber_g = treatment.fiber or 0

        params = await self.absorption_model.calculate_absorption_params(
            glycemic_index=gi,
            is_liquid=is_liquid,
            fat_g=fat_g,
            protein_g=protein_g,
            fiber_g=fiber_g,
            food_id=food_id,
            user_id=user_id
        )

        # Calculate predicted BG at each checkpoint
        bg_per_gram = isf / icr
        predicted_bgs = self._calculate_predicted_bgs(
            current_bg=current_bg,
            carbs=treatment.carbs,
            params=params,
            bg_per_gram=bg_per_gram,
            bg_trend=bg_trend
        )

        # Get previous meal timing
        minutes_since_last = await self._get_minutes_since_last_meal(user_id, meal_time)

        # Get IOB/COB at meal time
        treatments_for_calc = await self.treatment_repo.get_recent(user_id, hours=6)
        iob_at_meal = self.iob_cob_service.calculate_iob(treatments_for_calc, at_time=meal_time)
        cob_at_meal = self.iob_cob_service.calculate_cob(treatments_for_calc, at_time=meal_time)

        # Get lunar phase
        lunar_sin, lunar_cos = get_lunar_phase(meal_time)

        # Create training data point
        data_point = MLTrainingDataPoint(
            userId=user_id,
            treatmentId=treatment.id,
            timestamp=meal_time,

            # Food data
            foodId=food_id,
            foodDescription=food_description,
            carbs=treatment.carbs,
            protein=protein_g,
            fat=fat_g,
            fiber=fiber_g,
            glycemicIndex=gi,
            isLiquid=is_liquid,

            # Predictions
            predictedOnsetMin=params.onset_min,
            predictedHalfLifeMin=params.half_life_min,
            predictedBg30=predicted_bgs[30],
            predictedBg60=predicted_bgs[60],
            predictedBg90=predicted_bgs[90],

            # Context features
            hourOfDay=meal_time.hour,
            dayOfWeek=meal_time.weekday(),
            dayOfYear=meal_time.timetuple().tm_yday,
            lunarPhaseSin=lunar_sin,
            lunarPhaseCos=lunar_cos,

            # Weather (if available)
            weatherTempC=weather.get('temp_c') if weather else None,
            weatherHumidity=weather.get('humidity') if weather else None,
            weatherPressure=weather.get('pressure') if weather else None,

            # Metabolic context
            bgAtMeal=current_bg,
            bgTrend=bg_trend,
            iobAtMeal=iob_at_meal,
            cobAtMeal=cob_at_meal,
            minutesSinceLastMeal=minutes_since_last,

            # Quality
            isCleanMeal=is_clean,
            dataQualityScore=quality_score,
            isComplete=False
        )

        # Store the data point
        created = await self.training_repo.create(data_point)
        logger.info(f"Created ML training data point: {created.id} for food '{food_description}'")

        return created

    async def collect_checkpoint(
        self,
        data_point_id: str,
        user_id: str,
        checkpoint_min: int
    ) -> Optional[MLTrainingDataPoint]:
        """
        Collect actual BG at a checkpoint (+30, +60, or +90 min).

        Looks for the nearest BG reading within the collection window.

        Args:
            data_point_id: ID of the training data point
            user_id: User ID
            checkpoint_min: Checkpoint in minutes (30, 60, or 90)

        Returns:
            Updated MLTrainingDataPoint or None if no BG found
        """
        if checkpoint_min not in self.CHECKPOINTS:
            logger.warning(f"Invalid checkpoint: {checkpoint_min}")
            return None

        # Get the data point
        data_point = await self.training_repo.get_by_treatment(user_id, data_point_id)
        if not data_point:
            # Try by ID directly
            try:
                from azure.cosmos import exceptions
                data_point = None  # Will need to query differently
            except:
                pass

        # Find BG reading near the checkpoint time
        target_time = data_point.timestamp + timedelta(minutes=checkpoint_min)
        bg_reading = await self._find_nearest_bg(user_id, target_time)

        if not bg_reading:
            logger.warning(f"No BG reading found for checkpoint {checkpoint_min}min")
            return None

        actual_bg = float(bg_reading.value)

        # Update the data point with actual BG
        updated = await self.training_repo.update_checkpoint(
            data_point_id=data_point.id,
            user_id=user_id,
            checkpoint_min=checkpoint_min,
            actual_bg=actual_bg
        )

        if updated:
            logger.info(
                f"Collected checkpoint {checkpoint_min}min for {data_point_id}: "
                f"actual={actual_bg:.0f}, predicted={getattr(data_point, f'predictedBg{checkpoint_min}'):.0f}"
            )

        return updated

    async def collect_all_pending_checkpoints(self, user_id: str) -> int:
        """
        Process all incomplete training data points and collect any due checkpoints.

        This should be called periodically (e.g., every 5 minutes) to fill in
        actual BG readings for pending data points.

        Args:
            user_id: User ID

        Returns:
            Number of checkpoints collected
        """
        incomplete = await self.training_repo.get_incomplete(user_id, limit=50)
        now = datetime.utcnow()
        collected = 0

        for dp in incomplete:
            meal_time = dp.timestamp.replace(tzinfo=None)

            for checkpoint_min in self.CHECKPOINTS:
                # Check if this checkpoint is due
                checkpoint_time = meal_time + timedelta(minutes=checkpoint_min)
                if checkpoint_time > now:
                    continue  # Not due yet

                # Check if already collected
                if checkpoint_min == 30 and dp.actualBg30 is not None:
                    continue
                if checkpoint_min == 60 and dp.actualBg60 is not None:
                    continue
                if checkpoint_min == 90 and dp.actualBg90 is not None:
                    continue

                # Try to collect
                result = await self.collect_checkpoint(dp.id, user_id, checkpoint_min)
                if result:
                    collected += 1

        logger.info(f"Collected {collected} checkpoints for user {user_id}")
        return collected

    def _calculate_predicted_bgs(
        self,
        current_bg: float,
        carbs: float,
        params,  # AbsorptionParams
        bg_per_gram: float,
        bg_trend: int = 0
    ) -> dict:
        """
        Calculate predicted BG at each checkpoint.

        Uses the absorption model to estimate how much carbs will be absorbed
        at each time point, then calculates expected BG rise.
        """
        predictions = {}

        for checkpoint in self.CHECKPOINTS:
            # Calculate absorbed carbs at this time
            absorbed = self.absorption_model.calculate_absorbed_carbs(
                initial_carbs=carbs,
                time_elapsed_min=checkpoint,
                params=params
            )

            # BG rise from absorbed carbs
            bg_rise = absorbed * bg_per_gram

            # Add trend component (decaying over time)
            trend_rate = bg_trend * 3  # Approximate mg/dL per 5 min
            trend_decay = 0.5 ** (checkpoint / 30.0)
            trend_effect = trend_rate * (checkpoint / 5) * trend_decay * 0.5

            predicted_bg = current_bg + bg_rise + trend_effect
            predictions[checkpoint] = round(max(40, min(400, predicted_bg)), 0)

        return predictions

    async def _check_meal_cleanliness(
        self,
        user_id: str,
        meal_time: datetime,
        treatment_id: str
    ) -> Tuple[bool, float]:
        """
        Check if a meal is "clean" for training purposes.

        A clean meal has no overlapping treatments in the 90-minute window
        after eating. This ensures we can attribute BG changes to this meal.

        Returns:
            Tuple of (is_clean: bool, quality_score: float)
        """
        # Look for other treatments in the window
        window_end = meal_time + timedelta(minutes=self.MIN_MEAL_GAP)

        # Get all treatments in window
        treatments = await self.treatment_repo.get_recent(user_id, hours=3)

        overlapping = []
        for t in treatments:
            if t.id == treatment_id:
                continue
            t_time = t.timestamp.replace(tzinfo=None)
            if meal_time < t_time < window_end:
                overlapping.append(t)

        is_clean = len(overlapping) == 0

        # Calculate quality score based on overlap severity
        if is_clean:
            quality_score = 1.0
        else:
            # Reduce quality based on number and timing of overlapping treatments
            quality_score = max(0.2, 1.0 - (len(overlapping) * 0.3))

        return is_clean, quality_score

    async def _get_minutes_since_last_meal(
        self,
        user_id: str,
        current_time: datetime
    ) -> Optional[int]:
        """Get minutes since the last carb treatment before current_time."""
        treatments = await self.treatment_repo.get_recent(user_id, hours=12)

        last_meal_time = None
        for t in treatments:
            if t.carbs and t.carbs > 0:
                t_time = t.timestamp.replace(tzinfo=None)
                if t_time < current_time:
                    if last_meal_time is None or t_time > last_meal_time:
                        last_meal_time = t_time

        if last_meal_time:
            return int((current_time - last_meal_time).total_seconds() / 60)
        return None

    async def _find_nearest_bg(
        self,
        user_id: str,
        target_time: datetime
    ) -> Optional[GlucoseReading]:
        """Find the nearest BG reading to the target time."""
        # Search within window
        start_time = target_time - timedelta(minutes=self.BG_WINDOW)
        end_time = target_time + timedelta(minutes=self.BG_WINDOW)

        readings = await self.glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=20
        )

        if not readings:
            return None

        # Find closest to target time
        closest = None
        min_diff = float('inf')

        for r in readings:
            r_time = r.timestamp.replace(tzinfo=None)
            diff = abs((r_time - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest = r

        return closest


# Singleton instance
_collector: Optional[MLDataCollector] = None


def get_ml_data_collector() -> MLDataCollector:
    """Get the singleton ML data collector instance."""
    global _collector
    if _collector is None:
        _collector = MLDataCollector()
    return _collector
