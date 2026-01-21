"""
Model Drift Detection Service for T1D-AI

Detects when model performance degrades and triggers retraining.

Triggers retraining when:
1. Rolling 30-day MAE increases >20%
2. Consistent bias detected (over/under predicting)
3. 100+ new samples since last training
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from models.schemas import MLTrainingDataPoint
from database.repositories import MLTrainingDataRepository

logger = logging.getLogger(__name__)


# Drift detection thresholds
MAE_INCREASE_THRESHOLD = 0.20  # 20% increase triggers drift
BIAS_THRESHOLD = 15.0  # mg/dL systematic bias
MIN_SAMPLES_FOR_DRIFT_CHECK = 20  # Need at least 20 samples
SAMPLES_TRIGGER_RETRAIN = 100  # Auto-retrain at 100+ new samples


class DriftType(str, Enum):
    """Types of detected drift."""
    NONE = "none"
    MAE_INCREASE = "mae_increase"
    BIAS_DETECTED = "bias_detected"
    SAMPLE_COUNT = "sample_count"
    USER_REQUESTED = "user_requested"


@dataclass
class DriftStatus:
    """Status of model drift detection."""
    has_drift: bool
    drift_type: DriftType
    current_mae: float
    baseline_mae: float
    mae_change_percent: float
    bias_direction: Optional[str]  # "over" or "under" predicting
    bias_magnitude: float
    samples_since_training: int
    confidence: float
    recommendation: str


class ModelDriftDetector:
    """
    Detects when model performance degrades and needs retraining.

    Monitors:
    - Rolling prediction accuracy (MAE)
    - Systematic bias (over/under predicting)
    - Sample count since last training
    """

    def __init__(
        self,
        training_repo: Optional[MLTrainingDataRepository] = None
    ):
        self.training_repo = training_repo or MLTrainingDataRepository()

    async def check_drift(
        self,
        user_id: str,
        baseline_mae: Optional[float] = None,
        last_training_date: Optional[datetime] = None
    ) -> DriftStatus:
        """
        Check for model drift based on recent prediction performance.

        Args:
            user_id: User ID to check
            baseline_mae: MAE from last training (if known)
            last_training_date: Date of last model training

        Returns:
            DriftStatus with drift detection results
        """
        # Get recent completed training data
        recent_data = await self.training_repo.get_recent_complete(
            user_id, days=30, limit=500
        )

        if len(recent_data) < MIN_SAMPLES_FOR_DRIFT_CHECK:
            return DriftStatus(
                has_drift=False,
                drift_type=DriftType.NONE,
                current_mae=0,
                baseline_mae=baseline_mae or 0,
                mae_change_percent=0,
                bias_direction=None,
                bias_magnitude=0,
                samples_since_training=len(recent_data),
                confidence=0.0,
                recommendation="Not enough samples for drift detection"
            )

        # Calculate current metrics
        current_mae, bias_direction, bias_magnitude = self._calculate_metrics(recent_data)

        # Estimate baseline if not provided
        if baseline_mae is None:
            # Use first half of data as baseline estimate
            baseline_data = recent_data[len(recent_data)//2:]
            baseline_mae = self._calculate_mae(baseline_data)

        # Calculate change
        mae_change = (current_mae - baseline_mae) / max(baseline_mae, 1.0)

        # Count samples since training
        samples_since_training = len(recent_data)
        if last_training_date:
            samples_since_training = sum(
                1 for dp in recent_data
                if dp.timestamp.replace(tzinfo=None) > last_training_date
            )

        # Determine drift type
        drift_type = DriftType.NONE
        has_drift = False
        recommendation = "Model performance is stable"

        # Check MAE increase
        if mae_change > MAE_INCREASE_THRESHOLD:
            drift_type = DriftType.MAE_INCREASE
            has_drift = True
            recommendation = (
                f"Prediction accuracy has degraded by {mae_change*100:.0f}%. "
                f"Current MAE: {current_mae:.1f} mg/dL (was {baseline_mae:.1f}). "
                "Consider retraining the model."
            )

        # Check for systematic bias
        elif bias_magnitude > BIAS_THRESHOLD:
            drift_type = DriftType.BIAS_DETECTED
            has_drift = True
            recommendation = (
                f"Model is consistently {bias_direction}-predicting by "
                f"{bias_magnitude:.1f} mg/dL. Consider retraining to correct bias."
            )

        # Check sample count
        elif samples_since_training >= SAMPLES_TRIGGER_RETRAIN:
            drift_type = DriftType.SAMPLE_COUNT
            has_drift = True
            recommendation = (
                f"Accumulated {samples_since_training} new samples since last training. "
                "Recommend retraining to incorporate new data."
            )

        # Calculate confidence in drift detection
        confidence = min(1.0, len(recent_data) / 50)

        return DriftStatus(
            has_drift=has_drift,
            drift_type=drift_type,
            current_mae=current_mae,
            baseline_mae=baseline_mae,
            mae_change_percent=mae_change * 100,
            bias_direction=bias_direction,
            bias_magnitude=bias_magnitude,
            samples_since_training=samples_since_training,
            confidence=confidence,
            recommendation=recommendation
        )

    async def should_retrain(
        self,
        user_id: str,
        baseline_mae: Optional[float] = None,
        last_training_date: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.

        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        status = await self.check_drift(user_id, baseline_mae, last_training_date)

        if status.has_drift:
            return True, status.recommendation

        return False, "No retraining needed"

    def _calculate_metrics(
        self,
        data_points: List[MLTrainingDataPoint]
    ) -> Tuple[float, Optional[str], float]:
        """
        Calculate MAE and bias from training data points.

        Returns:
            Tuple of (mae, bias_direction, bias_magnitude)
        """
        errors_30 = []
        errors_60 = []
        errors_90 = []

        for dp in data_points:
            if dp.error30 is not None:
                errors_30.append(dp.error30)
            if dp.error60 is not None:
                errors_60.append(dp.error60)
            if dp.error90 is not None:
                errors_90.append(dp.error90)

        # Calculate MAE (primarily using 60-min as main metric)
        all_errors = errors_30 + errors_60 + errors_90
        if not all_errors:
            return 0.0, None, 0.0

        mae = sum(abs(e) for e in all_errors) / len(all_errors)

        # Calculate bias (mean error, not absolute)
        mean_error = sum(all_errors) / len(all_errors)

        if mean_error > 0:
            bias_direction = "under"  # Actual > predicted = under-predicting
        elif mean_error < 0:
            bias_direction = "over"   # Actual < predicted = over-predicting
        else:
            bias_direction = None

        bias_magnitude = abs(mean_error)

        return mae, bias_direction, bias_magnitude

    def _calculate_mae(self, data_points: List[MLTrainingDataPoint]) -> float:
        """Calculate MAE from a list of data points."""
        errors = []
        for dp in data_points:
            if dp.error30 is not None:
                errors.append(abs(dp.error30))
            if dp.error60 is not None:
                errors.append(abs(dp.error60))
            if dp.error90 is not None:
                errors.append(abs(dp.error90))

        return sum(errors) / len(errors) if errors else 0.0


class RetrainingScheduler:
    """
    Manages trigger-based retraining of ML models.

    Triggers:
    1. Drift detected (prediction error increasing)
    2. 100+ new training samples
    3. User requests manual retrain
    """

    def __init__(
        self,
        drift_detector: Optional[ModelDriftDetector] = None,
        training_repo: Optional[MLTrainingDataRepository] = None
    ):
        self.drift_detector = drift_detector or ModelDriftDetector()
        self.training_repo = training_repo or MLTrainingDataRepository()
        self._last_check: dict = {}  # user_id -> last check time

    async def check_and_queue_retrain(
        self,
        user_id: str,
        baseline_mae: Optional[float] = None,
        last_training_date: Optional[datetime] = None,
        force: bool = False
    ) -> Optional[dict]:
        """
        Check if retraining is needed and queue if so.

        Args:
            user_id: User ID
            baseline_mae: MAE from last training
            last_training_date: Date of last training
            force: Force retraining regardless of drift status

        Returns:
            Training job info if queued, None otherwise
        """
        # Rate limit checks (at most once per hour per user)
        now = datetime.utcnow()
        if not force and user_id in self._last_check:
            if (now - self._last_check[user_id]) < timedelta(hours=1):
                return None

        self._last_check[user_id] = now

        # Check for drift
        if force:
            should_retrain = True
            reason = "User requested manual retrain"
        else:
            should_retrain, reason = await self.drift_detector.should_retrain(
                user_id, baseline_mae, last_training_date
            )

        if not should_retrain:
            logger.debug(f"No retraining needed for user {user_id}: {reason}")
            return None

        # Queue training job
        job_info = await self._queue_training_job(user_id, reason)
        logger.info(f"Queued retraining for user {user_id}: {reason}")

        return job_info

    async def _queue_training_job(
        self,
        user_id: str,
        reason: str
    ) -> dict:
        """
        Queue a training job for the user.

        In a full implementation, this would:
        1. Create a TrainingJob in the database
        2. Trigger Azure Container Instance or Azure ML
        3. Return job tracking info

        For now, returns placeholder info.
        """
        import uuid

        job_id = str(uuid.uuid4())
        job_info = {
            "jobId": job_id,
            "userId": user_id,
            "status": "queued",
            "reason": reason,
            "queuedAt": datetime.utcnow().isoformat(),
            "estimatedDuration": "5-10 minutes"
        }

        # TODO: Actually queue to Azure
        # - Export training data to blob storage
        # - Create Azure Container Instance
        # - Start training container with data path
        # - Update TrainingJob status

        logger.info(f"Training job queued: {job_id} for user {user_id}")
        return job_info

    async def get_training_status(
        self,
        user_id: str
    ) -> dict:
        """
        Get the status of any active training jobs for a user.

        Returns:
            Dict with training status information
        """
        # TODO: Query TrainingJob table for active jobs
        return {
            "userId": user_id,
            "activeJobs": [],
            "lastTraining": None,
            "nextScheduled": None
        }


# Singleton instances
_drift_detector: Optional[ModelDriftDetector] = None
_retraining_scheduler: Optional[RetrainingScheduler] = None


def get_drift_detector() -> ModelDriftDetector:
    """Get the singleton drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = ModelDriftDetector()
    return _drift_detector


def get_retraining_scheduler() -> RetrainingScheduler:
    """Get the singleton retraining scheduler instance."""
    global _retraining_scheduler
    if _retraining_scheduler is None:
        _retraining_scheduler = RetrainingScheduler()
    return _retraining_scheduler
