"""
Personalized Model Manager

Manages per-user TFT model training, versioning, and storage.
Each user gets their own personalized glucose prediction model.
"""
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import torch

from database.repositories import UserModelRepository, TrainingJobRepository
from models.schemas import UserModel, TrainingJob, TrainingStatus
from services.cosmos_training_loader import CosmosTrainingDataLoader

logger = logging.getLogger(__name__)


class PersonalizedModelManager:
    """
    Manage personalized TFT models for each user.

    Handles:
    - Checking training eligibility
    - Initiating training jobs
    - Storing/loading per-user models
    - Model versioning
    """

    def __init__(self, models_dir: str = "/app/models/users"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = CosmosTrainingDataLoader()
        self.model_repo = UserModelRepository()
        self.job_repo = TrainingJobRepository()

        # Training configuration
        self.min_readings_for_training = 500
        self.min_treatments_for_training = 20
        self.min_days_for_training = 7

    async def check_eligibility(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user is eligible for personalized model training.

        Returns:
            Dict with eligibility status and details
        """
        eligible, reason = await self.data_loader.check_training_eligibility(
            user_id=user_id,
            min_days=self.min_days_for_training,
            min_readings=self.min_readings_for_training,
            min_treatments=self.min_treatments_for_training
        )

        stats = await self.data_loader.get_training_stats(user_id)

        return {
            'user_id': user_id,
            'eligible': eligible,
            'reason': reason,
            'stats': stats,
            'requirements': {
                'min_readings': self.min_readings_for_training,
                'min_treatments': self.min_treatments_for_training,
                'min_days': self.min_days_for_training,
            }
        }

    async def start_training(
        self,
        user_id: str,
        force: bool = False
    ) -> TrainingJob:
        """
        Start a training job for a user.

        Args:
            user_id: User ID to train model for
            force: If True, train even if existing model is recent

        Returns:
            TrainingJob with job details
        """
        # Check eligibility
        if not force:
            eligibility = await self.check_eligibility(user_id)
            if not eligibility['eligible']:
                raise ValueError(f"User not eligible for training: {eligibility['reason']}")

        # Check for existing running job
        existing_job = await self.job_repo.get_active_job(user_id)
        if existing_job:
            raise ValueError(f"Training job already in progress: {existing_job.id}")

        # Create training job
        job = await self.job_repo.create(
            user_id=user_id,
            model_type="tft",
            status=TrainingStatus.PENDING
        )

        # Start training in background
        asyncio.create_task(self._run_training(job))

        logger.info(f"Started training job {job.id} for user {user_id}")
        return job

    async def _run_training(self, job: TrainingJob):
        """Execute training job in background."""
        try:
            # Update status to running
            job = await self.job_repo.update_status(
                job.id,
                TrainingStatus.RUNNING,
                progress=0.0
            )

            # Load training data
            glucose_df, treatments_df, metadata = await self.data_loader.get_user_training_data(
                user_id=job.userId,
                days=90,
                min_readings=self.min_readings_for_training
            )

            await self.job_repo.update_status(job.id, TrainingStatus.RUNNING, progress=0.1)

            # Import training pipeline
            from ml.training.tft_trainer import TFTTrainingPipeline, TFTTrainingConfig

            # Configure for personalized training (smaller model, fewer epochs)
            config = TFTTrainingConfig(
                n_features=69,
                hidden_size=32,  # Smaller for per-user models
                n_heads=2,
                n_lstm_layers=1,
                dropout=0.1,
                encoder_length=24,
                prediction_length=12,
                horizons_minutes=[15, 30, 45, 60],
                quantiles=[0.1, 0.5, 0.9],
                quantile_weights=[1.5, 1.0, 0.8],
                learning_rate=0.001,
                batch_size=16,
                epochs=50,  # Fewer epochs for per-user
                patience=10,
                grad_clip=1.0,
                weight_decay=1e-5,
                min_completeness_score=0.3,
                max_glucose_gap_minutes=30,
                min_treatments_per_day=0.3,
                time_exclusion_patterns=[],
                val_split=0.15,
                test_split=0.15,
            )

            # Initialize pipeline
            pipeline = TFTTrainingPipeline(config=config, device="auto")

            await self.job_repo.update_status(job.id, TrainingStatus.RUNNING, progress=0.2)

            # Prepare data
            data_metrics = pipeline.prepare_data(glucose_df, treatments_df)

            if data_metrics.valid_windows < 50:
                raise ValueError(
                    f"Not enough valid training windows: {data_metrics.valid_windows}"
                )

            await self.job_repo.update_status(job.id, TrainingStatus.RUNNING, progress=0.3)

            # Train model
            test_metrics = pipeline.train(
                epochs=config.epochs,
                patience=config.patience,
                mlflow_tracking=False
            )

            await self.job_repo.update_status(job.id, TrainingStatus.RUNNING, progress=0.9)

            # Save model
            model_path = self._get_model_path(job.userId)
            pipeline.save_checkpoint(str(model_path))

            # Record model in database
            model_record = await self.model_repo.create(
                user_id=job.userId,
                model_type="tft",
                version=await self._get_next_version(job.userId),
                metrics=test_metrics,
                config=config.to_dict(),
                path=str(model_path)
            )

            # Update job as completed
            await self.job_repo.update_status(
                job.id,
                TrainingStatus.COMPLETED,
                progress=1.0,
                metrics=test_metrics,
                model_id=model_record.id
            )

            logger.info(
                f"Training completed for user {job.userId}. "
                f"MAE@30min: {test_metrics.get('mae_30min', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Training failed for job {job.id}: {e}")
            import traceback
            traceback.print_exc()

            await self.job_repo.update_status(
                job.id,
                TrainingStatus.FAILED,
                error=str(e)
            )

    def _get_model_path(self, user_id: str) -> Path:
        """Get path for user's model file."""
        user_dir = self.models_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "tft_model.pth"

    async def _get_next_version(self, user_id: str) -> int:
        """Get next version number for user's model."""
        models = await self.model_repo.get_user_models(user_id)
        if not models:
            return 1
        return max(m.version for m in models) + 1

    async def get_user_model(self, user_id: str) -> Optional[UserModel]:
        """Get the latest model for a user."""
        return await self.model_repo.get_latest(user_id)

    async def load_user_model(self, user_id: str):
        """Load a user's trained model for inference."""
        model_record = await self.get_user_model(user_id)

        if not model_record:
            logger.info(f"No personalized model for user {user_id}, using global model")
            return None

        model_path = Path(model_record.path)
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            from ml.training.tft_trainer import TFTTrainingPipeline
            pipeline = TFTTrainingPipeline.load_checkpoint(str(model_path))
            logger.info(f"Loaded personalized model for user {user_id} (v{model_record.version})")
            return pipeline.model
        except Exception as e:
            logger.error(f"Failed to load model for {user_id}: {e}")
            return None

    async def get_training_status(self, user_id: str) -> Dict[str, Any]:
        """Get training status and model info for a user."""
        # Check for active job
        active_job = await self.job_repo.get_active_job(user_id)

        # Get latest model
        latest_model = await self.get_user_model(user_id)

        # Get recent jobs
        recent_jobs = await self.job_repo.get_user_jobs(user_id, limit=5)

        return {
            'user_id': user_id,
            'has_personalized_model': latest_model is not None,
            'latest_model': latest_model.model_dump() if latest_model else None,
            'active_job': active_job.model_dump() if active_job else None,
            'recent_jobs': [j.model_dump() for j in recent_jobs],
        }

    async def should_retrain(
        self,
        user_id: str,
        new_readings_threshold: int = 500
    ) -> Tuple[bool, str]:
        """
        Check if user's model should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        model = await self.get_user_model(user_id)

        if not model:
            return True, "No existing model"

        # Check how much new data since last training
        stats = await self.data_loader.get_training_stats(user_id)
        readings_since_training = stats.get('glucose_count', 0)

        # Simple heuristic: retrain if significant new data
        if readings_since_training > new_readings_threshold:
            return True, f"Have {readings_since_training} new readings"

        # Check model age
        model_age_days = (datetime.utcnow() - model.createdAt).days
        if model_age_days > 30:
            return True, f"Model is {model_age_days} days old"

        return False, "Model is up to date"


# Type alias for Tuple
from typing import Tuple
