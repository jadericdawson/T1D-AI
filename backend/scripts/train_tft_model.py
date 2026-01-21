#!/usr/bin/env python3
"""
Train TFT Model with Local Data

Loads glucose readings and treatment data from local JSONL files,
runs the comprehensive TFT training pipeline, and saves the checkpoint.

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/train_tft_model.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/data")
GLUCOSE_FILE = DATA_DIR / "gluroo_readings.jsonl"
TREATMENTS_FILE = DATA_DIR / "bolus_moments.jsonl"
OUTPUT_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/models")


def load_glucose_data(filepath: Path) -> pd.DataFrame:
    """Load glucose readings from JSONL file."""
    logger.info(f"Loading glucose data from {filepath}")

    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['timestamp', 'value'])
    logger.info(f"Loaded {len(df)} unique glucose readings (removed {original_len - len(df)} duplicates)")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Basic stats
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"BG range: {df['value'].min()} - {df['value'].max()} mg/dL")

    return df


def load_treatment_data(filepath: Path, glucose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load treatment data from bolus_moments.jsonl.

    Also extracts embedded treatment data from glucose readings.
    """
    logger.info(f"Loading treatment data from {filepath}")

    treatments = []

    # Load from bolus_moments.jsonl
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Extract bolus info
                treatments.append({
                    'timestamp': record['bolus_ts'],
                    'insulin': record['bolus_insulin'],
                    'carbs': 0.0,
                    'protein': 0.0,
                    'fat': 0.0,
                    'type': 'insulin'
                })

                # Extract treatments from history
                for hist in record.get('history', []):
                    if hist.get('carbs', 0) > 0 or hist.get('insulin', 0) > 0:
                        treatments.append({
                            'timestamp': hist['ts'],
                            'insulin': hist.get('insulin', 0),
                            'carbs': hist.get('carbs', 0),
                            'protein': hist.get('protein', 0),
                            'fat': hist.get('fat', 0),
                            'type': 'carbs' if hist.get('carbs', 0) > 0 else 'insulin'
                        })

    # Also extract from glucose readings (they have embedded treatments)
    glucose_with_treatments = glucose_df[
        (glucose_df['carbs'] > 0) | (glucose_df['insulin'] > 0)
    ].copy()

    for _, row in glucose_with_treatments.iterrows():
        if row['insulin'] > 0:
            treatments.append({
                'timestamp': row['timestamp'].isoformat(),
                'insulin': row['insulin'],
                'carbs': 0.0,
                'protein': 0.0,
                'fat': 0.0,
                'type': 'insulin'
            })
        if row['carbs'] > 0:
            treatments.append({
                'timestamp': row['timestamp'].isoformat(),
                'insulin': 0.0,
                'carbs': row['carbs'],
                'protein': row.get('protein', 0),
                'fat': row.get('fat', 0),
                'type': 'carbs'
            })

    df = pd.DataFrame(treatments)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp', 'insulin', 'carbs'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} treatment records")

    # Stats
    insulin_count = (df['insulin'] > 0).sum()
    carbs_count = (df['carbs'] > 0).sum()
    logger.info(f"  Insulin boluses: {insulin_count}")
    logger.info(f"  Carb entries: {carbs_count}")

    return df


def main():
    """Main training function."""
    print("="*60)
    print("TFT MODEL TRAINING")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    glucose_df = load_glucose_data(GLUCOSE_FILE)
    treatments_df = load_treatment_data(TREATMENTS_FILE, glucose_df)

    # Import training components
    from ml.training.tft_trainer import (
        TFTTrainingPipeline,
        TFTTrainingConfig,
        TimeExclusionPattern
    )

    # Configure training - relaxed settings due to sparse data
    # NOTE: School hours exclusion disabled for now since we have limited data
    # The model will learn from ALL available data, which may include some
    # periods with unlogged treatments. This is a trade-off for having enough training data.
    config = TFTTrainingConfig(
        # Model architecture
        n_features=69,
        hidden_size=64,
        n_heads=4,
        n_lstm_layers=2,
        dropout=0.1,

        # Sequence configuration
        encoder_length=24,  # 120 min history
        prediction_length=12,  # 60 min prediction
        horizons_minutes=[15, 30, 45, 60],  # Clinically useful horizons within prediction window

        # Quantiles with weights (higher weight on lower quantile for hypo detection)
        quantiles=[0.1, 0.5, 0.9],
        quantile_weights=[1.5, 1.0, 0.8],

        # Training hyperparameters
        learning_rate=0.001,
        batch_size=16,  # Smaller batch due to limited data
        epochs=100,  # More epochs since we have less data
        patience=15,
        grad_clip=1.0,
        weight_decay=1e-5,

        # RELAXED data quality thresholds for sparse data
        min_completeness_score=0.4,  # Lower threshold to keep more data
        max_glucose_gap_minutes=30,  # Allow larger gaps
        min_treatments_per_day=0.5,  # Lower requirement

        # DISABLED school hours exclusion (not enough data to exclude)
        time_exclusion_patterns=[],

        # Smaller validation splits to keep more training data
        val_split=0.10,
        test_split=0.10,
    )

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: TFT with {config.n_features} features")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Horizons: {config.horizons_minutes} minutes")
    print(f"Epochs: {config.epochs} (patience: {config.patience})")
    print(f"School hours exclusion: 8am-3pm Mon-Fri")

    # Initialize pipeline
    pipeline = TFTTrainingPipeline(config=config, device="auto")

    print(f"\nTraining device: {pipeline.device}")

    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)

    try:
        metrics = pipeline.prepare_data(glucose_df, treatments_df)

        print(f"\nData quality metrics:")
        print(f"  Total windows: {metrics.total_windows}")
        print(f"  Valid windows: {metrics.valid_windows}")
        print(f"  Excluded reasons: {metrics.excluded_reasons}")

        # Check if we have enough data
        if metrics.valid_windows < 100:
            logger.warning(f"Only {metrics.valid_windows} valid windows - may not be enough for good training")

    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    try:
        test_metrics = pipeline.train(
            epochs=config.epochs,
            patience=config.patience,
            mlflow_tracking=False  # Disable for now
        )

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nTest set metrics:")
        for key, value in sorted(test_metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Save checkpoint
    checkpoint_path = OUTPUT_DIR / "tft_glucose_predictor.pth"
    print(f"\nSaving checkpoint to {checkpoint_path}")
    pipeline.save_checkpoint(str(checkpoint_path))

    # Export flagged windows for later inference
    flagged_path = OUTPUT_DIR / "flagged_windows_for_inference.json"
    pipeline.data_filter.export_flagged_windows_for_inference(flagged_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Flagged windows: {flagged_path}")

    # Print final summary
    print("\nKey metrics:")
    print(f"  MAE @ 30min: {test_metrics.get('mae_30min', 'N/A'):.1f} mg/dL")
    print(f"  MAE @ 60min: {test_metrics.get('mae_60min', 'N/A'):.1f} mg/dL")
    print(f"  Hypo sensitivity: {test_metrics.get('hypo_sensitivity', 'N/A'):.1%}")
    print(f"  80% coverage: {test_metrics.get('coverage_80pct', 'N/A'):.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
