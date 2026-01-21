#!/usr/bin/env python3
"""
Run TFT (Temporal Fusion Transformer) model training for T1D-AI.

Trains a state-of-the-art TFT model for long-horizon glucose prediction
with uncertainty quantification (30, 45, 60 minute horizons).
"""
import asyncio
import logging
import sys
import gc
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.repositories import GlucoseRepository, TreatmentRepository
from ml.feature_engineering import engineer_extended_features, TFT_FEATURE_COLUMNS
from ml.training.train_absorption_models import TFTTrainer
from ml.inference.isf_inference import create_isf_service
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_training_data(user_id: str, days: int = 90):
    """Fetch glucose and treatment data for training."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()

    # Fetch glucose readings
    glucose_readings = await glucose_repo.get_history(
        user_id=user_id,
        start_time=since,
        limit=50000
    )

    # Fetch treatments
    treatments = await treatment_repo.get_recent(
        user_id=user_id,
        hours=days * 24
    )

    logger.info(f"Fetched {len(glucose_readings)} glucose readings")
    logger.info(f"Fetched {len(treatments)} treatments")

    # Convert to DataFrames
    glucose_df = pd.DataFrame([r.model_dump() for r in glucose_readings])
    treatments_df = pd.DataFrame([t.model_dump() for t in treatments])

    return glucose_df, treatments_df


def convert_trend_to_numeric(trend_val):
    """Convert trend string to numeric value."""
    trend_map = {
        "DoubleUp": 3, "SingleUp": 2, "FortyFiveUp": 1,
        "Flat": 0,
        "FortyFiveDown": -1, "SingleDown": -2, "DoubleDown": -3,
        "NotComputable": 0, "RateOutOfRange": 0, None: 0
    }
    if isinstance(trend_val, str):
        return trend_map.get(trend_val, 0)
    return int(trend_val) if pd.notna(trend_val) else 0


def prepare_tft_sequences(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
    isf: float = 55.0,
    icr: float = 10.0,
    encoder_length: int = 24,
    prediction_length: int = 12
):
    """
    Prepare sequences for TFT training.

    Args:
        glucose_df: Glucose readings DataFrame
        treatments_df: Treatments DataFrame
        isf: Insulin sensitivity factor
        icr: Insulin to carb ratio
        encoder_length: Input sequence length (24 = 120 min)
        prediction_length: Output horizon length (12 = 60 min)

    Returns:
        Tuple of (X, y) arrays for training
    """
    # Merge treatments with glucose at nearest timestamp
    glucose_df = glucose_df.copy()
    glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)
    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    # Convert trend to numeric
    glucose_df['trend'] = glucose_df['trend'].apply(convert_trend_to_numeric)

    # Add treatment columns with defaults
    for col in ['insulin', 'carbs', 'protein', 'fat']:
        if col not in glucose_df.columns:
            glucose_df[col] = 0.0

    # Merge treatments into glucose readings
    if not treatments_df.empty:
        treatments_df = treatments_df.copy()
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)

        for _, treat in treatments_df.iterrows():
            treat_time = treat['timestamp']

            # Find closest glucose reading
            time_diffs = abs(glucose_df['timestamp'] - treat_time)
            closest_idx = time_diffs.idxmin()

            # Add treatment data
            if treat.get('type') == 'insulin' or (treat.get('insulin') or 0) > 0:
                insulin_val = treat.get('insulin', 0) or 0
                glucose_df.loc[closest_idx, 'insulin'] += insulin_val
            elif treat.get('type') == 'carbs' or (treat.get('carbs') or 0) > 0:
                glucose_df.loc[closest_idx, 'carbs'] += treat.get('carbs', 0) or 0
                protein_val = treat.get('protein', 0)
                fat_val = treat.get('fat', 0)
                glucose_df.loc[closest_idx, 'protein'] += 0 if pd.isna(protein_val) else protein_val
                glucose_df.loc[closest_idx, 'fat'] += 0 if pd.isna(fat_val) else fat_val

    logger.info("Engineering extended features...")

    # Engineer extended features
    df = engineer_extended_features(
        glucose_df,
        treatments_df=treatments_df,
        ml_iob=0.0,  # Will be computed per-row in full implementation
        ml_cob=0.0,
        isf=isf,
        icr=icr
    )

    if df.empty:
        logger.error("Feature engineering failed - empty DataFrame")
        return None, None

    logger.info(f"Engineered {len(df)} rows with {len(TFT_FEATURE_COLUMNS)} features")

    # Get available features
    available_features = [col for col in TFT_FEATURE_COLUMNS if col in df.columns]
    missing_features = set(TFT_FEATURE_COLUMNS) - set(available_features)

    if missing_features:
        logger.warning(f"Missing features (will use 0): {missing_features}")
        for col in missing_features:
            df[col] = 0.0

    # Fill NaN values
    df[TFT_FEATURE_COLUMNS] = df[TFT_FEATURE_COLUMNS].fillna(0)

    # Create sequences
    feature_data = df[TFT_FEATURE_COLUMNS].values.astype(np.float32)
    target_data = df['value'].values.astype(np.float32)

    total_length = encoder_length + prediction_length
    horizon_steps = [5, 8, 11]  # 30, 45, 60 min predictions (0-indexed)

    sequences = []
    targets = []

    valid_count = 0
    nan_count = 0

    for i in range(len(feature_data) - total_length + 1):
        seq = feature_data[i:i + encoder_length]

        # Get targets at prediction horizons
        target_30 = target_data[i + encoder_length + horizon_steps[0]]
        target_45 = target_data[i + encoder_length + horizon_steps[1]]
        target_60 = target_data[i + encoder_length + horizon_steps[2]]

        # Check for NaN
        if np.isnan(seq).any() or np.isnan([target_30, target_45, target_60]).any():
            nan_count += 1
            continue

        sequences.append(seq)
        targets.append([target_30, target_45, target_60])
        valid_count += 1

    logger.info(f"Created {valid_count} valid sequences ({nan_count} skipped due to NaN)")

    if not sequences:
        return None, None

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    return X, y


async def main():
    """Main training function."""
    user_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    logger.info(f"Starting TFT training for user {user_id}")

    # Get ISF from model
    models_dir = Path("/home/jadericdawson/Documents/AI/dexcom_reader_ML_complete")
    isf_service = create_isf_service(models_dir, device="cpu")
    isf = isf_service.get_default_isf() if not isf_service.is_loaded else 55.0
    icr = 10.0

    logger.info(f"Using ISF={isf:.1f}, ICR={icr}")

    # Fetch data (90 days for better model)
    logger.info("Fetching training data...")
    glucose_df, treatments_df = await fetch_training_data(user_id, days=90)

    if glucose_df.empty:
        logger.error("No glucose data found")
        return

    # Prepare sequences
    logger.info("Preparing TFT sequences...")
    X, y = prepare_tft_sequences(
        glucose_df, treatments_df,
        isf=isf, icr=icr,
        encoder_length=24,
        prediction_length=12
    )

    if X is None or len(X) < 100:
        logger.error(f"Insufficient training data: {len(X) if X is not None else 0} sequences")
        return

    logger.info(f"Training data: X shape={X.shape}, y shape={y.shape}")

    # Clear memory before training
    del glucose_df, treatments_df
    gc.collect()

    # Train TFT using GPU for speed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    trainer = TFTTrainer(
        device=device,
        n_features=X.shape[2],
        epochs=30,
        batch_size=128,  # Maximize GPU usage
        learning_rate=0.001
    )

    logger.info("Training TFT model...")
    model, metrics = trainer.train_tft(X, y)

    # Log results
    logger.info("=" * 60)
    logger.info("TFT TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Train samples: {metrics['train_samples']}")
    logger.info(f"  Val samples: {metrics['val_samples']}")
    logger.info(f"  Overall MAE: {metrics['mae']:.2f} mg/dL")
    logger.info(f"  Overall RMSE: {metrics['rmse']:.2f} mg/dL")
    logger.info(f"  30-min MAE: {metrics['mae_30min']:.2f} mg/dL")
    logger.info(f"  45-min MAE: {metrics['mae_45min']:.2f} mg/dL")
    logger.info(f"  60-min MAE: {metrics['mae_60min']:.2f} mg/dL")
    logger.info("=" * 60)

    # Save model
    model_path = Path("models/tft_glucose_predictor.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    model, metrics = asyncio.run(main())
    if metrics:
        print(f"\nFinal status: SUCCESS - MAE: {metrics['mae']:.2f} mg/dL")
    else:
        print("\nFinal status: FAILED")
