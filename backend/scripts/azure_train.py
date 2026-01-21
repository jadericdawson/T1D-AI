#!/usr/bin/env python3
"""
Azure Container Instance Training Script

This script runs inside an Azure Container Instance to train a personalized
TFT model for a user. It:
1. Pulls glucose/treatment data from CosmosDB
2. Trains the TFT model
3. Uploads the trained model to Azure Blob Storage
4. Updates the user's model status in CosmosDB

Environment variables required:
- COSMOS_ENDPOINT: CosmosDB endpoint
- COSMOS_KEY: CosmosDB key
- COSMOS_DATABASE: Database name
- STORAGE_ACCOUNT_URL: Blob storage URL
- STORAGE_ACCOUNT_KEY: Blob storage key (or use managed identity)
- USER_ID: User ID to train model for
- MODEL_TYPE: Model type (default: tft)
"""
import os
import sys
import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from services.cosmos_training_loader import CosmosTrainingDataLoader
from services.blob_model_store import get_model_store
from database.repositories import UserModelRepository, TrainingJobRepository
from models.schemas import UserModel, UserModelStatus
from ml.training.tft_trainer import TFTTrainingPipeline, TFTTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def train_user_model(user_id: str, model_type: str = "tft", job_id: str = None):
    """Train a personalized model for a user."""

    logger.info(f"Starting training for user {user_id}, model type: {model_type}")

    # Initialize repositories
    data_loader = CosmosTrainingDataLoader()
    model_repo = UserModelRepository()
    job_repo = TrainingJobRepository()
    model_store = get_model_store()

    try:
        # Update job status if job_id provided
        if job_id:
            await job_repo.update_status(job_id, user_id, "running")

        # Update model status to training
        model = await model_repo.get(user_id, model_type)
        if not model:
            model = UserModel(
                id=f"{user_id}_{model_type}",
                userId=user_id,
                modelType=model_type,
                status=UserModelStatus.TRAINING
            )
        else:
            model.status = UserModelStatus.TRAINING
        await model_repo.upsert(model)

        # Load training data from CosmosDB
        logger.info("Loading training data from CosmosDB...")
        glucose_df, treatments_df, metadata = await data_loader.get_user_training_data(
            user_id=user_id,
            days=90,
            min_readings=100
        )

        logger.info(f"Loaded {len(glucose_df)} glucose readings, {len(treatments_df)} treatments")

        # Configure training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        config = TFTTrainingConfig(
            n_features=69,
            hidden_size=32,
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
            epochs=50,
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
        pipeline = TFTTrainingPipeline(config=config, device=device)

        # Prepare data
        logger.info("Preparing training data...")
        data_metrics = pipeline.prepare_data(glucose_df, treatments_df)
        logger.info(f"Valid training windows: {data_metrics.valid_windows}")

        if data_metrics.valid_windows < 50:
            raise ValueError(f"Not enough valid training windows: {data_metrics.valid_windows}")

        # Train
        logger.info("Starting training...")
        start_time = datetime.now()

        test_metrics = pipeline.train(
            epochs=config.epochs,
            patience=config.patience,
            mlflow_tracking=False
        )

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.1f} seconds")

        # Save checkpoint to temp file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = Path(f.name)

        pipeline.save_checkpoint(str(temp_path))
        model_data = temp_path.read_bytes()
        temp_path.unlink()

        # Upload to Blob Storage
        logger.info("Uploading model to Blob Storage...")
        blob_metadata = await model_store.upload_user_model(
            user_id=user_id,
            model_type=model_type,
            model_data=model_data,
            metrics=test_metrics,
            config={}
        )

        # Update model record
        model.status = UserModelStatus.ACTIVE
        model.version = blob_metadata.get("version", 1)
        model.blobPath = blob_metadata.get("blob_path")
        model.metrics = test_metrics
        model.dataPoints = len(glucose_df)
        model.trainedAt = datetime.now(timezone.utc)
        await model_repo.upsert(model)

        # Update job as completed
        if job_id:
            await job_repo.update_status(job_id, user_id, "completed", metrics=test_metrics)

        logger.info(f"Training completed successfully!")
        logger.info(f"MAE @ 30min: {test_metrics.get('mae_30min', 'N/A')}")
        logger.info(f"MAE @ 60min: {test_metrics.get('mae_60min', 'N/A')}")

        return test_metrics

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Update status to failed
        if job_id:
            await job_repo.update_status(job_id, user_id, "failed", error=str(e))

        model = await model_repo.get(user_id, model_type)
        if model:
            model.status = UserModelStatus.FAILED
            model.lastError = str(e)
            await model_repo.upsert(model)

        raise


def main():
    """Main entry point for ACI training."""
    user_id = os.environ.get("USER_ID")
    model_type = os.environ.get("MODEL_TYPE", "tft")
    job_id = os.environ.get("JOB_ID")

    if not user_id:
        logger.error("USER_ID environment variable required")
        sys.exit(1)

    logger.info(f"Azure Container Instance Training")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Job ID: {job_id or 'N/A'}")

    try:
        asyncio.run(train_user_model(user_id, model_type, job_id))
        logger.info("Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
