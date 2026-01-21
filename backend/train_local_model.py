#!/usr/bin/env python3
"""
Local TFT Model Training Script

Train a personalized glucose prediction model using local JSONL data.
This uses the EXACT same code as Azure training for 100% compatibility.

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python train_local_model.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml.training.tft_trainer import TFTTrainingPipeline, TFTTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_local_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load local JSONL data and convert to glucose/treatments DataFrames.

    The gluroo_readings.jsonl has combined data:
    {"timestamp": "...", "value": 137, "trend": 3, "carbs": 0.0, "insulin": 4.0, ...}

    We split this into:
    - glucose_df: timestamp, value, trend (all rows)
    - treatments_df: timestamp, insulin, carbs, protein, fat (rows with insulin OR carbs > 0)
    """
    jsonl_path = data_dir / "gluroo_readings.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Data file not found: {jsonl_path}")

    logger.info(f"Loading data from {jsonl_path}")

    # Load JSONL
    df = pd.read_json(jsonl_path, lines=True)
    logger.info(f"Loaded {len(df)} records")

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create glucose DataFrame (all rows)
    glucose_df = df[['timestamp', 'value', 'trend']].copy()
    glucose_df['trend'] = glucose_df['trend'].fillna(0).astype(int)

    # Create treatments DataFrame (only rows with insulin OR carbs)
    # First, create treatments from rows where insulin > 0
    insulin_mask = df['insulin'].fillna(0) > 0
    carb_mask = df['carbs'].fillna(0) > 0
    treatment_mask = insulin_mask | carb_mask

    treatments_df = df.loc[treatment_mask, ['timestamp', 'insulin', 'carbs', 'protein', 'fat']].copy()
    treatments_df = treatments_df.fillna(0)

    # Add type column for compatibility
    treatments_df['type'] = treatments_df.apply(
        lambda row: 'combo' if row['insulin'] > 0 and row['carbs'] > 0
        else 'insulin' if row['insulin'] > 0
        else 'carbs',
        axis=1
    )

    logger.info(f"Glucose readings: {len(glucose_df)}")
    logger.info(f"Treatment events: {len(treatments_df)}")
    logger.info(f"  - Insulin only: {len(treatments_df[treatments_df['type'] == 'insulin'])}")
    logger.info(f"  - Carbs only: {len(treatments_df[treatments_df['type'] == 'carbs'])}")
    logger.info(f"  - Combo: {len(treatments_df[treatments_df['type'] == 'combo'])}")

    # Show date range
    start_date = glucose_df['timestamp'].min()
    end_date = glucose_df['timestamp'].max()
    days = (end_date - start_date).days
    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({days} days)")

    return glucose_df, treatments_df


def train_model():
    """Train the TFT model using local data."""

    # Paths - data is in repo root /data/, not backend/data/
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        logger.info("Using CPU (training will be slower)")

    # Load data
    glucose_df, treatments_df = load_local_data(data_dir)

    # Configure training - IDENTICAL to Azure config (api/v1/training.py:349-372)
    config = TFTTrainingConfig(
        n_features=69,
        hidden_size=32,  # Same as Azure per-user models
        n_heads=2,
        n_lstm_layers=1,
        dropout=0.1,
        encoder_length=24,  # 120 min history
        prediction_length=12,  # 60 min prediction
        horizons_minutes=[15, 30, 45, 60],  # Same as Azure
        quantiles=[0.1, 0.5, 0.9],
        quantile_weights=[1.5, 1.0, 0.8],  # Prioritize hypoglycemia
        learning_rate=0.001,
        batch_size=16,  # Same as Azure
        epochs=50,  # Same as Azure
        patience=10,  # Same as Azure
        grad_clip=1.0,
        weight_decay=1e-5,
        min_completeness_score=0.3,  # More lenient for local data
        max_glucose_gap_minutes=30,
        min_treatments_per_day=0.3,
        time_exclusion_patterns=[],
        val_split=0.15,
        test_split=0.15,
    )

    logger.info("Configuration:")
    logger.info(f"  - Hidden size: {config.hidden_size}")
    logger.info(f"  - Attention heads: {config.n_heads}")
    logger.info(f"  - LSTM layers: {config.n_lstm_layers}")
    logger.info(f"  - Encoder length: {config.encoder_length} (= {config.encoder_length * 5} min)")
    logger.info(f"  - Prediction horizons: {config.horizons_minutes} min")
    logger.info(f"  - Epochs: {config.epochs}")
    logger.info(f"  - Patience: {config.patience}")

    # Initialize pipeline
    pipeline = TFTTrainingPipeline(config=config, device=device)

    # Prepare data
    logger.info("\n" + "="*60)
    logger.info("PREPARING DATA")
    logger.info("="*60)

    try:
        data_metrics = pipeline.prepare_data(glucose_df, treatments_df)
        logger.info(f"Valid training windows: {data_metrics.valid_windows}")
        logger.info(f"Excluded windows: {data_metrics.excluded_windows}")

        if data_metrics.valid_windows < 50:
            raise ValueError(f"Not enough valid training windows: {data_metrics.valid_windows}")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

    # Train
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)

    start_time = datetime.now()

    test_metrics = pipeline.train(
        epochs=config.epochs,
        patience=config.patience,
        mlflow_tracking=False  # No MLflow for local training
    )

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nTraining completed in {training_time:.1f} seconds")

    # Results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)

    if test_metrics:
        logger.info("Test Set Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")

    # Save checkpoint
    checkpoint_path = models_dir / "tft_glucose_predictor.pth"
    pipeline.save_checkpoint(str(checkpoint_path))
    logger.info(f"\nModel saved to: {checkpoint_path}")

    # Verify checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    logger.info(f"Checkpoint contents: {list(checkpoint.keys())}")

    if 'scaler_state' in checkpoint:
        logger.info("✓ Scaler saved (required for correct predictions)")
    else:
        logger.warning("⚠ Scaler NOT saved - predictions may be inaccurate!")

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)
    logger.info(f"Model ready at: {checkpoint_path}")
    logger.info("Deploy with: follow DEPLOYMENT.md steps")

    return test_metrics


if __name__ == "__main__":
    try:
        metrics = train_model()
        print("\n✅ Training successful!")
        if metrics:
            print(f"MAE @ 30min: {metrics.get('mae_30min', 'N/A')}")
            print(f"MAE @ 60min: {metrics.get('mae_60min', 'N/A')}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
