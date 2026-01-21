#!/usr/bin/env python3
"""
Train All T1D-AI ML Models on Gluroo Data

This script trains all personalized ML models using historical Gluroo data:
1. IOB Model - Personalized insulin absorption curves
2. COB Model - Personalized carb absorption curves
3. BG Pressure Model - Combined IOB+COB effect prediction

All training runs are tracked with MLflow on port 5002.

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --model iob  # Train only IOB
    python scripts/train_all_models.py --model cob  # Train only COB
    python scripts/train_all_models.py --model bg_pressure  # Train only BG Pressure
    python scripts/train_all_models.py --epochs 200  # Custom epochs

Requirements:
    - MLflow server running on port 5002: ./scripts/start_mlflow.sh
    - Gluroo data in ../data/gluroo_readings.jsonl and ../data/bolus_moments.jsonl
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add backend src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.mlflow_tracking import ModelTracker, setup_mlflow
from ml.models.iob_model import PersonalizedIOBModel, IOB_MODEL_CONFIG
from ml.models.cob_model import PersonalizedCOBModel, COB_MODEL_CONFIG
from ml.models.bg_pressure_model import BGPressureModel, BG_PRESSURE_MODEL_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
READINGS_FILE = DATA_DIR / "gluroo_readings.jsonl"
TREATMENTS_FILE = DATA_DIR / "bolus_moments.jsonl"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_gluroo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load glucose readings and treatment data from Gluroo exports."""
    logger.info(f"Loading data from {DATA_DIR}")

    # Load glucose readings
    readings = []
    with open(READINGS_FILE, 'r') as f:
        for line in f:
            readings.append(json.loads(line))

    readings_df = pd.DataFrame(readings)
    readings_df['timestamp'] = pd.to_datetime(readings_df.get('date', readings_df.get('timestamp', readings_df.get('dateString'))))
    readings_df = readings_df.sort_values('timestamp').reset_index(drop=True)

    # Normalize value column
    if 'sgv' in readings_df.columns and 'value' not in readings_df.columns:
        readings_df['value'] = readings_df['sgv']

    logger.info(f"Loaded {len(readings_df)} glucose readings")

    # Load treatments
    treatments = []
    with open(TREATMENTS_FILE, 'r') as f:
        for line in f:
            treatments.append(json.loads(line))

    treatments_df = pd.DataFrame(treatments)
    treatments_df['timestamp'] = pd.to_datetime(treatments_df.get('date', treatments_df.get('timestamp')))
    treatments_df = treatments_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(treatments_df)} treatments")

    return readings_df, treatments_df


def create_iob_training_data(readings_df: pd.DataFrame, treatments_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create IOB training data by analyzing BG response to insulin.

    For each insulin bolus:
    1. Find BG at time of bolus
    2. Track BG over next 6 hours
    3. Estimate remaining IOB based on BG drop vs expected ISF effect
    4. Create training samples at each time point
    """
    logger.info("Creating IOB training data...")

    X_list = []
    y_list = []

    # Get insulin treatments
    insulin_treatments = treatments_df[treatments_df['insulin'].notna() & (treatments_df['insulin'] > 0)]

    # Estimated ISF (will be learned, start with typical value)
    isf_estimate = 50.0

    for _, treatment in insulin_treatments.iterrows():
        bolus_time = treatment['timestamp']
        bolus_units = float(treatment['insulin'])

        if bolus_units <= 0:
            continue

        # Get BG at bolus time
        bg_at_bolus = readings_df[
            (readings_df['timestamp'] >= bolus_time - timedelta(minutes=5)) &
            (readings_df['timestamp'] <= bolus_time + timedelta(minutes=5))
        ]

        if bg_at_bolus.empty:
            continue

        bg_initial = float(bg_at_bolus.iloc[0]['value'])

        # Track BG over next 6 hours
        for minutes_after in range(5, 361, 5):
            sample_time = bolus_time + timedelta(minutes=minutes_after)

            bg_at_time = readings_df[
                (readings_df['timestamp'] >= sample_time - timedelta(minutes=3)) &
                (readings_df['timestamp'] <= sample_time + timedelta(minutes=3))
            ]

            if bg_at_time.empty:
                continue

            bg_current = float(bg_at_time.iloc[0]['value'])

            # Calculate BG drop from initial
            bg_drop = bg_initial - bg_current

            # Estimate absorbed insulin based on BG drop
            # absorbed = bg_drop / ISF
            absorbed_estimate = max(0, bg_drop / isf_estimate)
            absorbed_estimate = min(absorbed_estimate, bolus_units)

            # Remaining IOB fraction
            remaining_fraction = max(0, min(1, 1 - (absorbed_estimate / bolus_units)))

            # Extract features (extended 24 features)
            hour = sample_time.hour
            dow = sample_time.weekday()
            month = sample_time.month
            doy = sample_time.timetuple().tm_yday

            # Get trend from nearby readings
            recent = readings_df[
                (readings_df['timestamp'] >= sample_time - timedelta(minutes=30)) &
                (readings_df['timestamp'] <= sample_time)
            ]
            if len(recent) >= 2:
                rate = (float(recent.iloc[-1]['value']) - float(recent.iloc[0]['value'])) / max(1, len(recent))
            else:
                rate = 0

            features = [
                bolus_units / 10.0,                          # bolus_scaled
                0.0 if treatment.get('carbs', 0) else 1.0,   # is_correction
                minutes_after / 360.0,                       # time_scaled
                np.sin(2 * np.pi * hour / 24),              # hour_sin
                np.cos(2 * np.pi * hour / 24),              # hour_cos
                np.sin(2 * np.pi * dow / 7),                # dow_sin
                np.cos(2 * np.pi * dow / 7),                # dow_cos
                np.sin(2 * np.pi * month / 12),             # month_sin
                np.cos(2 * np.pi * month / 12),             # month_cos
                np.sin(2 * np.pi * doy / 365),              # doy_sin
                np.cos(2 * np.pi * doy / 365),              # doy_cos
                0.0,                                         # lunar_sin (simplified)
                0.0,                                         # lunar_cos (simplified)
                bg_current / 200.0,                          # bg_scaled
                0.0,                                         # trend_normalized (would need CGM trend)
                rate / 10.0,                                 # rate_of_change
                0.0,                                         # variability (simplified)
                0.0,                                         # activity_level
                1.0 if 23 <= hour or hour < 6 else 0.0,     # is_sleeping
                0.0,                                         # is_fasting
                1.0,                                         # bg_absorption_factor
                1.0,                                         # activity_absorption_factor
                0.0,                                         # placeholder
                0.0,                                         # placeholder
            ]

            X_list.append(features)
            y_list.append([remaining_fraction])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Created {len(X)} IOB training samples")
    return X, y


def create_cob_training_data(readings_df: pd.DataFrame, treatments_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create COB training data by analyzing BG response to carbs.

    For each carb treatment:
    1. Find BG at time of meal
    2. Track BG over next 6 hours
    3. Account for any insulin given
    4. Estimate remaining COB based on BG rise pattern
    """
    logger.info("Creating COB training data...")

    X_list = []
    y_list = []

    # Get carb treatments
    carb_treatments = treatments_df[treatments_df['carbs'].notna() & (treatments_df['carbs'] > 0)]

    isf_estimate = 50.0
    icr_estimate = 10.0
    bg_per_gram = isf_estimate / icr_estimate

    for _, treatment in carb_treatments.iterrows():
        meal_time = treatment['timestamp']
        carbs = float(treatment['carbs'])
        protein = float(treatment.get('protein', 0) or 0)
        fat = float(treatment.get('fat', 0) or 0)
        gi = int(treatment.get('glycemicIndex', 55) or 55)

        if carbs <= 0:
            continue

        # Get BG at meal time
        bg_at_meal = readings_df[
            (readings_df['timestamp'] >= meal_time - timedelta(minutes=5)) &
            (readings_df['timestamp'] <= meal_time + timedelta(minutes=5))
        ]

        if bg_at_meal.empty:
            continue

        bg_initial = float(bg_at_meal.iloc[0]['value'])

        # Get any insulin given with meal
        insulin_with_meal = treatments_df[
            (treatments_df['timestamp'] >= meal_time - timedelta(minutes=10)) &
            (treatments_df['timestamp'] <= meal_time + timedelta(minutes=10)) &
            (treatments_df['insulin'].notna())
        ]['insulin'].sum()

        # Track BG over next 6 hours
        for minutes_after in range(5, 361, 5):
            sample_time = meal_time + timedelta(minutes=minutes_after)

            bg_at_time = readings_df[
                (readings_df['timestamp'] >= sample_time - timedelta(minutes=3)) &
                (readings_df['timestamp'] <= sample_time + timedelta(minutes=3))
            ]

            if bg_at_time.empty:
                continue

            bg_current = float(bg_at_time.iloc[0]['value'])

            # Calculate BG rise from initial (accounting for insulin)
            insulin_effect = insulin_with_meal * isf_estimate * (1 - 0.5 ** (minutes_after / 90))
            net_bg_rise = (bg_current - bg_initial) + insulin_effect

            # Estimate absorbed carbs based on BG rise
            absorbed_estimate = max(0, net_bg_rise / bg_per_gram)
            absorbed_estimate = min(absorbed_estimate, carbs)

            # Remaining COB fraction
            remaining_fraction = max(0, min(1, 1 - (absorbed_estimate / carbs)))

            # Determine absorption rate
            if gi >= 70:
                absorption_rate = 'fast'
            elif gi >= 55:
                absorption_rate = 'medium'
            else:
                absorption_rate = 'slow'

            # Extract features (32 features)
            hour = sample_time.hour
            dow = sample_time.weekday()
            month = sample_time.month
            doy = sample_time.timetuple().tm_yday

            features = [
                carbs / 100.0,                              # carbs_scaled
                protein / 50.0,                             # protein_scaled
                fat / 50.0,                                 # fat_scaled
                0.0,                                        # fiber_scaled (not available)
                gi / 100.0,                                 # gi_scaled
                (carbs * gi / 100) / 50.0,                  # glycemic_load
                1.0 if absorption_rate == 'fast' else 0.0,  # absorption_fast
                1.0 if absorption_rate == 'medium' else 0.0,# absorption_medium
                1.0 if absorption_rate == 'slow' else 0.0,  # absorption_slow
                1.0 if fat > 15 else 0.0,                   # is_high_fat
                minutes_after / 480.0,                      # time_scaled
                np.sin(2 * np.pi * hour / 24),             # hour_sin
                np.cos(2 * np.pi * hour / 24),             # hour_cos
                np.sin(2 * np.pi * dow / 7),               # dow_sin
                np.cos(2 * np.pi * dow / 7),               # dow_cos
                np.sin(2 * np.pi * month / 12),            # month_sin
                np.cos(2 * np.pi * month / 12),            # month_cos
                np.sin(2 * np.pi * doy / 365),             # doy_sin
                np.cos(2 * np.pi * doy / 365),             # doy_cos
                0.0,                                        # lunar_sin
                0.0,                                        # lunar_cos
                bg_current / 200.0,                         # bg_scaled
                0.0,                                        # trend
                0.0,                                        # rate_of_change
                0.0,                                        # variability
                0.0,                                        # activity_level
                1.0 if 23 <= hour or hour < 6 else 0.0,    # is_sleeping
                0.0,                                        # is_fasting_before
                min(1.0, fat / 30.0) * 0.5 + min(1.0, protein / 40.0) * 0.3,  # absorption_delay
                0.0,                                        # placeholder
                0.0,                                        # placeholder
                0.0,                                        # placeholder
            ]

            X_list.append(features)
            y_list.append([remaining_fraction])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Created {len(X)} COB training samples")
    return X, y


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    tracker: ModelTracker,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
) -> Dict:
    """Train a PyTorch model with MLflow tracking."""
    logger.info(f"Training model with {len(X)} samples, {epochs} epochs")

    # Create dataset
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Log parameters
    tracker.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": train_size,
        "val_samples": val_size,
        "input_size": X.shape[1],
    })

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log metrics
        tracker.log_training_curve(epoch, train_loss, val_loss, learning_rate)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Log final metrics
    tracker.log_metrics({
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
    })

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }


def train_iob_model(readings_df: pd.DataFrame, treatments_df: pd.DataFrame, epochs: int = 100) -> Optional[Path]:
    """Train IOB model."""
    logger.info("=" * 50)
    logger.info("Training IOB Model")
    logger.info("=" * 50)

    tracker = ModelTracker(model_type="iob", user_id="gluroo_user")
    tracker.start_run(run_name=f"iob_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    try:
        # Create training data
        X, y = create_iob_training_data(readings_df, treatments_df)

        if len(X) < 100:
            logger.warning(f"Insufficient IOB training data: {len(X)} samples")
            tracker.end_run("FAILED")
            return None

        # Create model
        model = PersonalizedIOBModel(
            input_size=24,
            hidden_sizes=[64, 32, 16],
            dropout=0.1
        )

        # Train
        metrics = train_model(model, X, y, tracker, epochs=epochs)

        # Save model
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "iob_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': IOB_MODEL_CONFIG,
            'metrics': metrics,
        }, model_path)

        # Log to MLflow
        tracker.log_model(model, artifact_path="iob_model")
        tracker.log_artifact(str(model_path))

        logger.info(f"IOB model saved to {model_path}")
        tracker.end_run()
        return model_path

    except Exception as e:
        logger.error(f"IOB training failed: {e}")
        tracker.end_run("FAILED")
        return None


def train_cob_model(readings_df: pd.DataFrame, treatments_df: pd.DataFrame, epochs: int = 100) -> Optional[Path]:
    """Train COB model."""
    logger.info("=" * 50)
    logger.info("Training COB Model")
    logger.info("=" * 50)

    tracker = ModelTracker(model_type="cob", user_id="gluroo_user")
    tracker.start_run(run_name=f"cob_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    try:
        # Create training data
        X, y = create_cob_training_data(readings_df, treatments_df)

        if len(X) < 100:
            logger.warning(f"Insufficient COB training data: {len(X)} samples")
            tracker.end_run("FAILED")
            return None

        # Create model
        model = PersonalizedCOBModel(
            input_size=32,
            hidden_sizes=[128, 64, 32],
            dropout=0.1
        )

        # Train
        metrics = train_model(model, X, y, tracker, epochs=epochs)

        # Save model
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "cob_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': COB_MODEL_CONFIG,
            'metrics': metrics,
        }, model_path)

        # Log to MLflow
        tracker.log_model(model, artifact_path="cob_model")
        tracker.log_artifact(str(model_path))

        logger.info(f"COB model saved to {model_path}")
        tracker.end_run()
        return model_path

    except Exception as e:
        logger.error(f"COB training failed: {e}")
        tracker.end_run("FAILED")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train T1D-AI ML models on Gluroo data")
    parser.add_argument("--model", choices=["all", "iob", "cob", "bg_pressure"], default="all",
                       help="Which model to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("T1D-AI Model Training")
    logger.info(f"Model: {args.model}, Epochs: {args.epochs}")
    logger.info("=" * 60)

    # Setup MLflow
    setup_mlflow("T1D-AI/Training")

    # Check for data files
    if not READINGS_FILE.exists():
        logger.error(f"Glucose readings file not found: {READINGS_FILE}")
        logger.info("Please export Gluroo data to the data/ directory")
        return 1

    if not TREATMENTS_FILE.exists():
        logger.error(f"Treatments file not found: {TREATMENTS_FILE}")
        logger.info("Please export Gluroo bolus data to the data/ directory")
        return 1

    # Load data
    readings_df, treatments_df = load_gluroo_data()

    # Train models
    results = {}

    if args.model in ["all", "iob"]:
        iob_path = train_iob_model(readings_df, treatments_df, epochs=args.epochs)
        results["iob"] = iob_path

    if args.model in ["all", "cob"]:
        cob_path = train_cob_model(readings_df, treatments_df, epochs=args.epochs)
        results["cob"] = cob_path

    # Summary
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    for model_name, path in results.items():
        status = "SUCCESS" if path else "FAILED"
        logger.info(f"  {model_name}: {status} - {path}")

    logger.info("\nView training runs at: http://localhost:5002")

    return 0


if __name__ == "__main__":
    sys.exit(main())
