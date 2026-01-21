#!/usr/bin/env python3
"""
Train Core Metabolism Models: IOB, COB, ISF

These are the PRIMARY drivers of BG:
1. IOB - How insulin decays over time (lowers BG)
2. COB - How carbs absorb over time (raises BG)
3. ISF - How sensitive you are to insulin (mg/dL per unit)

Train these models first, then use them as inputs to BG prediction.

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/train_core_models.py

Author: T1D-AI
Date: 2026-01-07
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import mlflow
import mlflow.pytorch

# Local imports
from ml.models.iob_model import PersonalizedIOBModel, IOB_MODEL_CONFIG
from ml.models.cob_model import PersonalizedCOBModel, COB_MODEL_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/data")
GLUCOSE_FILE = DATA_DIR / "gluroo_readings.jsonl"
TREATMENTS_FILE = DATA_DIR / "bolus_moments.jsonl"
OUTPUT_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/models")

# MLflow
MLFLOW_URI = "http://localhost:5002"
EXPERIMENT_NAME = "T1D-AI/Core-Models"

# Default parameters (these are what we're trying to learn!)
DEFAULT_ISF = 55.0  # Will be learned
DEFAULT_ICR = 10.0  # Will be learned
INSULIN_HALF_LIFE = 81.0  # Known for Humalog/Novolog


def load_data():
    """Load glucose and treatment data."""
    # Load glucose
    glucose_records = []
    with open(GLUCOSE_FILE, 'r') as f:
        for line in f:
            if line.strip():
                glucose_records.append(json.loads(line))

    glucose_df = pd.DataFrame(glucose_records)
    glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)
    glucose_df['timestamp'] = glucose_df['timestamp'].dt.tz_localize(None)
    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    # Load treatments
    treatment_records = []
    with open(TREATMENTS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Extract bolus
                treatment_records.append({
                    'timestamp': record['bolus_ts'],
                    'insulin': record['bolus_insulin'],
                    'carbs': 0.0,
                    'notes': '',
                })
                # Extract history
                for hist in record.get('history', []):
                    if hist.get('carbs', 0) > 0 or hist.get('insulin', 0) > 0:
                        treatment_records.append({
                            'timestamp': hist['ts'],
                            'insulin': hist.get('insulin', 0),
                            'carbs': hist.get('carbs', 0),
                            'notes': hist.get('notes', ''),
                        })

    treatments_df = pd.DataFrame(treatment_records)
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
    treatments_df['timestamp'] = treatments_df['timestamp'].dt.tz_localize(None)
    treatments_df = treatments_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(glucose_df)} glucose, {len(treatments_df)} treatments")
    return glucose_df, treatments_df


def create_iob_training_data(glucose_df: pd.DataFrame, treatments_df: pd.DataFrame) -> List[Dict]:
    """
    Create IOB training samples by finding insulin boluses and tracking BG response.

    For each insulin bolus, we track:
    - BG at time of bolus
    - BG at various times after (15, 30, 60, 90, 120, 180 min)
    - This lets us estimate how much insulin was absorbed at each point
    """
    samples = []

    # Get insulin-only treatments
    insulin_df = treatments_df[treatments_df['insulin'] > 0].copy()

    for _, bolus in insulin_df.iterrows():
        bolus_time = bolus['timestamp']
        bolus_units = bolus['insulin']

        # Get BG at bolus time
        bg_at_bolus = get_bg_at_time(glucose_df, bolus_time)
        if bg_at_bolus is None:
            continue

        # Check if there are carbs within 30 min (meal bolus vs correction)
        carbs_nearby = treatments_df[
            (treatments_df['carbs'] > 0) &
            (abs((treatments_df['timestamp'] - bolus_time).dt.total_seconds()) < 30*60)
        ]
        is_correction = len(carbs_nearby) == 0

        # Track BG at various times after bolus
        for minutes_after in [30, 60, 90, 120, 180, 240]:
            check_time = bolus_time + timedelta(minutes=minutes_after)
            bg_after = get_bg_at_time(glucose_df, check_time)

            if bg_after is None:
                continue

            # Check for interfering carbs (would invalidate the IOB observation)
            carbs_between = treatments_df[
                (treatments_df['carbs'] > 0) &
                (treatments_df['timestamp'] > bolus_time) &
                (treatments_df['timestamp'] < check_time)
            ]['carbs'].sum()

            if carbs_between > 10:  # Skip if significant carbs eaten
                continue

            # Estimate remaining IOB from BG change
            # If BG dropped by X, and ISF is ~55, then absorbed_insulin = X / 55
            bg_drop = bg_at_bolus - bg_after
            expected_drop = bolus_units * DEFAULT_ISF

            if expected_drop > 0:
                absorbed_fraction = min(1.0, max(0.0, bg_drop / expected_drop))
                remaining_fraction = 1.0 - absorbed_fraction
            else:
                remaining_fraction = 1.0

            samples.append({
                'bolus_units': bolus_units,
                'minutes_since': minutes_after,
                'remaining_fraction': remaining_fraction,
                'hour': bolus_time.hour,
                'is_correction': is_correction,
                'bg_at_bolus': bg_at_bolus,
                'bg_after': bg_after,
            })

    logger.info(f"Created {len(samples)} IOB training samples")
    return samples


def create_cob_training_data(glucose_df: pd.DataFrame, treatments_df: pd.DataFrame) -> List[Dict]:
    """
    Create COB training samples by finding carb events and tracking BG response.
    """
    samples = []

    # Get carb treatments
    carb_df = treatments_df[treatments_df['carbs'] > 0].copy()

    for _, meal in carb_df.iterrows():
        meal_time = meal['timestamp']
        carbs = meal['carbs']
        notes = meal.get('notes', '')

        # Get BG at meal time
        bg_at_meal = get_bg_at_time(glucose_df, meal_time)
        if bg_at_meal is None:
            continue

        # Check for insulin with this meal
        insulin_with_meal = treatments_df[
            (treatments_df['insulin'] > 0) &
            (abs((treatments_df['timestamp'] - meal_time).dt.total_seconds()) < 15*60)
        ]['insulin'].sum()

        # Track BG at various times after meal
        for minutes_after in [30, 60, 90, 120, 180]:
            check_time = meal_time + timedelta(minutes=minutes_after)
            bg_after = get_bg_at_time(glucose_df, check_time)

            if bg_after is None:
                continue

            # Account for insulin effect
            insulin_effect = 0
            if insulin_with_meal > 0:
                # Estimate how much insulin has acted by this point
                absorbed_insulin = insulin_with_meal * (1 - 0.5 ** (minutes_after / INSULIN_HALF_LIFE))
                insulin_effect = absorbed_insulin * DEFAULT_ISF

            # Estimate remaining COB from BG change
            bg_rise = bg_after - bg_at_meal + insulin_effect  # Add back insulin effect
            expected_rise = carbs * (DEFAULT_ISF / DEFAULT_ICR)  # ~5.5 mg/dL per gram

            if expected_rise > 0:
                absorbed_fraction = min(1.0, max(0.0, bg_rise / expected_rise))
                remaining_fraction = 1.0 - absorbed_fraction
            else:
                remaining_fraction = 1.0

            samples.append({
                'carbs': carbs,
                'minutes_since': minutes_after,
                'remaining_fraction': remaining_fraction,
                'hour': meal_time.hour,
                'notes': notes,
                'bg_at_meal': bg_at_meal,
                'bg_after': bg_after,
                'insulin_with_meal': insulin_with_meal,
            })

    logger.info(f"Created {len(samples)} COB training samples")
    return samples


def get_bg_at_time(glucose_df: pd.DataFrame, target_time: datetime, tolerance_min: int = 10) -> float:
    """Get BG reading closest to target time within tolerance."""
    time_diffs = abs((glucose_df['timestamp'] - target_time).dt.total_seconds() / 60)
    min_idx = time_diffs.idxmin()

    if time_diffs[min_idx] <= tolerance_min:
        return float(glucose_df.loc[min_idx, 'value'])
    return None


class IOBDataset(Dataset):
    """Dataset for IOB model training."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Build feature vector (simplified 8 features)
        hour = s['hour']
        features = torch.tensor([
            s['bolus_units'] / 10.0,
            s['minutes_since'] / 360.0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            1.0 if s['is_correction'] else 0.0,
            s['bg_at_bolus'] / 200.0,
            (s['bg_at_bolus'] - s['bg_after']) / 100.0,  # BG change
            s['bolus_units'] * (s['minutes_since'] / 60),  # Interaction term
        ], dtype=torch.float32)

        target = torch.tensor([s['remaining_fraction']], dtype=torch.float32)

        return features, target


class COBDataset(Dataset):
    """Dataset for COB model training."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        hour = s['hour']
        features = torch.tensor([
            s['carbs'] / 100.0,
            s['minutes_since'] / 360.0,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            s['bg_at_meal'] / 200.0,
            (s['bg_after'] - s['bg_at_meal']) / 100.0,  # BG change
            s['insulin_with_meal'] / 10.0,
            s['carbs'] * (s['minutes_since'] / 60) / 100.0,  # Interaction
        ], dtype=torch.float32)

        target = torch.tensor([s['remaining_fraction']], dtype=torch.float32)

        return features, target


class SimpleIOBNet(nn.Module):
    """Simpler IOB network for limited data."""

    def __init__(self, input_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCOBNet(nn.Module):
    """Simpler COB network for limited data."""

    def __init__(self, input_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, config: Dict, model_name: str):
    """Train a model with MLflow logging."""

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_params(config)
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('train_samples', len(train_loader.dataset))
        mlflow.log_param('val_samples', len(val_loader.dataset))

        for epoch in range(config['epochs']):
            # Train
            model.train()
            train_loss = 0
            for features, targets in train_loader:
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            predictions = []
            actuals = []
            with torch.no_grad():
                for features, targets in val_loader:
                    output = model(features)
                    val_loss += criterion(output, targets).item()
                    predictions.extend(output.numpy().flatten())
                    actuals.extend(targets.numpy().flatten())
            val_loss /= len(val_loader)

            # Calculate MAE
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_mae', mae, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, mae={mae:.4f}")

        # Load best
        if best_state:
            model.load_state_dict(best_state)

        # Final metrics
        mlflow.log_metric('best_val_loss', best_val_loss)
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"{model_name} best val loss: {best_val_loss:.4f}")

    return model


def main():
    logger.info("=" * 60)
    logger.info("Training Core Metabolism Models: IOB, COB")
    logger.info("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    glucose_df, treatments_df = load_data()

    # Create training data
    iob_samples = create_iob_training_data(glucose_df, treatments_df)
    cob_samples = create_cob_training_data(glucose_df, treatments_df)

    if len(iob_samples) < 20:
        logger.warning(f"Only {len(iob_samples)} IOB samples - need more data!")
    if len(cob_samples) < 20:
        logger.warning(f"Only {len(cob_samples)} COB samples - need more data!")

    # Train IOB model
    if len(iob_samples) >= 10:
        logger.info("\n" + "=" * 40)
        logger.info("Training IOB Model")
        logger.info("=" * 40)

        # Split
        np.random.shuffle(iob_samples)
        split = int(len(iob_samples) * 0.8)
        train_samples = iob_samples[:split]
        val_samples = iob_samples[split:]

        train_ds = IOBDataset(train_samples)
        val_ds = IOBDataset(val_samples)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        iob_model = SimpleIOBNet(input_size=8)
        iob_model = train_model(
            iob_model, train_loader, val_loader,
            {'lr': 0.001, 'epochs': 100},
            'IOB_Model'
        )

        # Save
        torch.save(iob_model.state_dict(), OUTPUT_DIR / 'iob_model_trained.pth')
        logger.info(f"Saved IOB model to {OUTPUT_DIR / 'iob_model_trained.pth'}")

    # Train COB model
    if len(cob_samples) >= 10:
        logger.info("\n" + "=" * 40)
        logger.info("Training COB Model")
        logger.info("=" * 40)

        np.random.shuffle(cob_samples)
        split = int(len(cob_samples) * 0.8)
        train_samples = cob_samples[:split]
        val_samples = cob_samples[split:]

        train_ds = COBDataset(train_samples)
        val_ds = COBDataset(val_samples)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        cob_model = SimpleCOBNet(input_size=8)
        cob_model = train_model(
            cob_model, train_loader, val_loader,
            {'lr': 0.001, 'epochs': 100},
            'COB_Model'
        )

        # Save
        torch.save(cob_model.state_dict(), OUTPUT_DIR / 'cob_model_trained.pth')
        logger.info(f"Saved COB model to {OUTPUT_DIR / 'cob_model_trained.pth'}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"View results: {MLFLOW_URI}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
