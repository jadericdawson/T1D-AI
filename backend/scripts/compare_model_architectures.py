#!/usr/bin/env python3
"""
Compare Model Architectures with MLflow Tracking

This script trains multiple model architectures and compares them in MLflow:
1. Physics-TFT (baseline from train_physics_tft.py)
2. Simple MLP (baseline - no temporal)
3. LSTM-only (no attention)
4. Physics-TFT with more hidden units
5. Physics-TFT with attention (full TFT style)

All results are logged to MLflow for comparison.

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/compare_model_architectures.py

MLflow UI: http://localhost:5002

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
from typing import List, Tuple, Dict, Optional, Type
import mlflow
import mlflow.pytorch

# Local imports
from ml.models.physics_tft import PREDICTION_HORIZONS, DEFAULT_ISF
from ml.training.data_quality import TrainingDataQualityFilter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/data")
GLUCOSE_FILE = DATA_DIR / "gluroo_readings.jsonl"
TREATMENTS_FILE = DATA_DIR / "bolus_moments.jsonl"
OUTPUT_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/models")

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5002"
EXPERIMENT_NAME = "T1D-AI/Model-Architecture-Comparison"

# Horizons to predict
HORIZONS = [15, 30, 60, 90, 120]


# =============================================================================
# DATASET
# =============================================================================

class GlucoseDataset(Dataset):
    """Dataset for glucose prediction training."""

    def __init__(self, glucose_df: pd.DataFrame, treatments_df: pd.DataFrame, seq_length: int = 24):
        self.seq_length = seq_length
        self.samples = []
        self._create_samples(glucose_df, treatments_df)

    def _create_samples(self, glucose_df: pd.DataFrame, treatments_df: pd.DataFrame):
        glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)
        max_horizon = max(HORIZONS) // 5

        for i in range(len(glucose_df) - self.seq_length - max_horizon):
            try:
                sample = self._create_single_sample(glucose_df, treatments_df, i)
                if sample:
                    self.samples.append(sample)
            except Exception:
                continue

    def _create_single_sample(self, glucose_df: pd.DataFrame, treatments_df: pd.DataFrame, start_idx: int):
        end_idx = start_idx + self.seq_length
        sequence = glucose_df.iloc[start_idx:end_idx]

        # Check for gaps
        time_diffs = sequence['timestamp'].diff().dt.total_seconds() / 60
        if time_diffs.max() > 10:
            return None

        current_ts = sequence.iloc[-1]['timestamp']
        current_bg = float(sequence.iloc[-1]['value'])

        # Get treatments window
        treatments_window_start = current_ts - timedelta(hours=4)
        recent_treatments = treatments_df[
            (treatments_df['timestamp'] >= treatments_window_start) &
            (treatments_df['timestamp'] <= current_ts)
        ]

        # Calculate IOB
        iob = 0.0
        for _, t in recent_treatments.iterrows():
            if t.get('insulin', 0) > 0:
                time_since = (current_ts - t['timestamp']).total_seconds() / 3600
                remaining = max(0, 1 - time_since / 4) * t['insulin']
                iob += remaining

        # Calculate COB
        cob = 0.0
        for _, t in recent_treatments.iterrows():
            if t.get('carbs', 0) > 0:
                time_since = (current_ts - t['timestamp']).total_seconds() / 3600
                remaining = max(0, 1 - time_since / 3) * t['carbs']
                cob += remaining

        # Trend (rate per 5 min)
        recent_values = sequence['value'].values[-6:]
        trend = (recent_values[-1] - recent_values[0]) / 5 if len(recent_values) >= 2 else 0

        # Time features
        hour = current_ts.hour + current_ts.minute / 60

        # Get targets
        targets = {}
        for horizon in HORIZONS:
            target_idx = end_idx + horizon // 5
            if target_idx < len(glucose_df):
                targets[horizon] = float(glucose_df.iloc[target_idx]['value'])
            else:
                return None

        # Build features
        values = sequence['value'].values
        features = np.column_stack([
            values / 200,  # Normalized glucose
            np.gradient(values),  # Rate of change
            np.full(len(values), iob),
            np.full(len(values), cob),
            np.sin(2 * np.pi * hour / 24) * np.ones(len(values)),
            np.cos(2 * np.pi * hour / 24) * np.ones(len(values)),
        ])

        return {
            'features': features,
            'current_bg': current_bg,
            'iob': iob,
            'cob': cob,
            'trend': trend,
            'hour': hour,
            'targets': targets,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        targets = torch.tensor([sample['targets'][h] for h in HORIZONS], dtype=torch.float32)

        return {
            'features': features,
            'current_bg': torch.tensor(sample['current_bg'], dtype=torch.float32),
            'iob': torch.tensor(sample['iob'], dtype=torch.float32),
            'cob': torch.tensor(sample['cob'], dtype=torch.float32),
            'trend': torch.tensor(sample['trend'], dtype=torch.float32),
            'targets': targets,
        }


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP baseline - no temporal modeling."""

    def __init__(self, n_features: int = 6, hidden_size: int = 64):
        super().__init__()
        self.name = "SimpleMLP"

        # Flatten sequence into single vector
        self.flatten = nn.Flatten()

        # MLP layers
        self.fc = nn.Sequential(
            nn.Linear(24 * n_features + 4, hidden_size * 2),  # +4 for current_bg, iob, cob, trend
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(HORIZONS)),
        )

    def forward(self, features, current_bg, iob, cob, trend):
        batch_size = features.shape[0]

        # Flatten sequence
        flat = self.flatten(features)

        # Concat with scalars
        scalars = torch.stack([current_bg, iob, cob, trend], dim=1)
        x = torch.cat([flat, scalars], dim=1)

        # MLP
        return self.fc(x)


class LSTMOnly(nn.Module):
    """LSTM without attention."""

    def __init__(self, n_features: int = 6, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.name = "LSTMOnly"

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),  # +4 for scalars
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(HORIZONS)),
        )

    def forward(self, features, current_bg, iob, cob, trend):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]  # Last timestep

        # Concat with scalars
        scalars = torch.stack([current_bg, iob, cob, trend], dim=1)
        x = torch.cat([last_hidden, scalars], dim=1)

        return self.fc(x)


class PhysicsLSTM(nn.Module):
    """LSTM with physics-informed baseline."""

    def __init__(self, n_features: int = 6, hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.name = "PhysicsLSTM"
        self.isf = DEFAULT_ISF

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )

        # Output: correction factor per horizon
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(HORIZONS)),
        )

    def forward(self, features, current_bg, iob, cob, trend):
        batch_size = features.shape[0]

        # Compute physics baseline
        baselines = []
        for horizon in HORIZONS:
            t_hours = horizon / 60

            # IOB effect (linear decay)
            iob_remaining = iob * max(0, 1 - t_hours / 4)
            iob_effect = -iob_remaining * self.isf

            # COB effect (linear absorption)
            cob_absorbed = cob * min(1.0, t_hours / 3)
            icr = 10.0  # Insulin to carb ratio
            cob_effect = cob_absorbed * (self.isf / icr)

            # Trend effect (damped)
            trend_effect = trend * (horizon / 5) * np.exp(-horizon / 60)

            baseline = current_bg + iob_effect + cob_effect + trend_effect
            baselines.append(baseline)

        physics_baseline = torch.stack(baselines, dim=1)

        # LSTM correction
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]

        scalars = torch.stack([current_bg, iob, cob, trend], dim=1)
        x = torch.cat([last_hidden, scalars], dim=1)

        correction = self.fc(x)

        return physics_baseline + correction


class PhysicsLSTMAttention(nn.Module):
    """LSTM with attention and physics baseline."""

    def __init__(self, n_features: int = 6, hidden_size: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.name = "PhysicsLSTMAttention"
        self.isf = DEFAULT_ISF

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, len(HORIZONS)),
        )

    def forward(self, features, current_bg, iob, cob, trend):
        # Physics baseline (same as PhysicsLSTM)
        baselines = []
        for horizon in HORIZONS:
            t_hours = horizon / 60
            iob_remaining = iob * max(0, 1 - t_hours / 4)
            iob_effect = -iob_remaining * self.isf
            cob_absorbed = cob * min(1.0, t_hours / 3)
            cob_effect = cob_absorbed * (self.isf / 10.0)
            trend_effect = trend * (horizon / 5) * np.exp(-horizon / 60)
            baseline = current_bg + iob_effect + cob_effect + trend_effect
            baselines.append(baseline)
        physics_baseline = torch.stack(baselines, dim=1)

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_hidden = attn_out[:, -1, :]

        scalars = torch.stack([current_bg, iob, cob, trend], dim=1)
        x = torch.cat([last_hidden, scalars], dim=1)

        correction = self.fc(x)

        return physics_baseline + correction


# =============================================================================
# TRAINING
# =============================================================================

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
    glucose_df = glucose_df.drop_duplicates(subset=['timestamp', 'value'])
    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    # Load treatments
    treatment_records = []
    with open(TREATMENTS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                treatment_records.append({
                    'timestamp': record['bolus_ts'],
                    'insulin': record['bolus_insulin'],
                    'carbs': 0.0,
                })
                for hist in record.get('history', []):
                    if hist.get('carbs', 0) > 0 or hist.get('insulin', 0) > 0:
                        treatment_records.append({
                            'timestamp': hist['ts'],
                            'insulin': hist.get('insulin', 0),
                            'carbs': hist.get('carbs', 0),
                        })

    treatments_df = pd.DataFrame(treatment_records)
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
    treatments_df['timestamp'] = treatments_df['timestamp'].dt.tz_localize(None)
    treatments_df = treatments_df.drop_duplicates(subset=['timestamp'])
    treatments_df = treatments_df.sort_values('timestamp').reset_index(drop=True)

    return glucose_df, treatments_df


def create_splits(glucose_df: pd.DataFrame, treatments_df: pd.DataFrame, min_score: float = 0.3):
    """Create train/val/test splits with quality filtering."""
    # Apply quality filter
    quality_filter = TrainingDataQualityFilter(
        exclude_school_weekdays=True,
        min_completeness_score=min_score,
    )

    filtered_glucose, filtered_treatments, _ = quality_filter.create_filtered_dataset(
        glucose_df, treatments_df
    )

    if len(filtered_glucose) == 0:
        raise ValueError("No data passed quality filter!")

    # Time-based split (70/15/15)
    timestamps = np.sort(filtered_glucose['timestamp'].unique())
    n_total = len(timestamps)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    train_ts = timestamps[train_end]
    val_ts = timestamps[val_end]

    train_glucose = filtered_glucose[filtered_glucose['timestamp'] < train_ts]
    train_treatments = filtered_treatments[filtered_treatments['timestamp'] < train_ts]

    val_glucose = filtered_glucose[
        (filtered_glucose['timestamp'] >= train_ts) &
        (filtered_glucose['timestamp'] < val_ts)
    ]
    val_treatments = filtered_treatments[
        (filtered_treatments['timestamp'] >= train_ts) &
        (filtered_treatments['timestamp'] < val_ts)
    ]

    test_glucose = filtered_glucose[filtered_glucose['timestamp'] >= val_ts]
    test_treatments = filtered_treatments[filtered_treatments['timestamp'] >= val_ts]

    return (
        GlucoseDataset(train_glucose, train_treatments),
        GlucoseDataset(val_glucose, val_treatments),
        GlucoseDataset(test_glucose, test_treatments),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """Train a model and return best state."""

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            output = model(
                features=features,
                current_bg=batch['current_bg'].to(device),
                iob=batch['iob'].to(device),
                cob=batch['cob'].to(device),
                trend=batch['trend'].to(device),
            )

            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                output = model(
                    features=features,
                    current_bg=batch['current_bg'].to(device),
                    iob=batch['iob'].to(device),
                    cob=batch['cob'].to(device),
                    trend=batch['trend'].to(device),
                )

                val_loss += criterion(output, targets).item()

        val_loss /= len(val_loader)

        # Log to MLflow
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('val_loss', val_loss, step=epoch)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_val_loss': best_val_loss}


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """Evaluate model and return per-horizon metrics."""

    model.eval()
    horizon_errors = {h: [] for h in HORIZONS}

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            output = model(
                features=features,
                current_bg=batch['current_bg'].to(device),
                iob=batch['iob'].to(device),
                cob=batch['cob'].to(device),
                trend=batch['trend'].to(device),
            )

            for i, horizon in enumerate(HORIZONS):
                errors = torch.abs(output[:, i] - targets[:, i])
                horizon_errors[horizon].extend(errors.cpu().numpy().tolist())

    # Calculate metrics
    metrics = {}
    for horizon in HORIZONS:
        mae = np.mean(horizon_errors[horizon])
        rmse = np.sqrt(np.mean(np.array(horizon_errors[horizon]) ** 2))
        metrics[f'mae_{horizon}min'] = mae
        metrics[f'rmse_{horizon}min'] = rmse

    # Overall
    all_errors = sum(horizon_errors.values(), [])
    metrics['mae_overall'] = np.mean(all_errors)
    metrics['rmse_overall'] = np.sqrt(np.mean(np.array(all_errors) ** 2))

    return metrics


def run_experiment(
    model_class: Type[nn.Module],
    model_kwargs: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    device: str = 'cpu',
):
    """Run a single experiment and log to MLflow."""

    model = model_class(**model_kwargs).to(device)
    model_name = model.name

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_name}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"{'='*60}")

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M%S')}"):
        # Log parameters
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('model_params', sum(p.numel() for p in model.parameters()))
        mlflow.log_params(config)
        mlflow.log_params(model_kwargs)

        # Train
        model, train_metrics = train_model(model, train_loader, val_loader, config, device)

        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device)

        # Log test metrics
        for name, value in test_metrics.items():
            mlflow.log_metric(f'test_{name}', value)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"\n{model_name} Results:")
        for horizon in HORIZONS:
            mae = test_metrics.get(f'mae_{horizon}min', 0)
            logger.info(f"  +{horizon}min MAE: {mae:.1f} mg/dL")
        logger.info(f"  Overall MAE: {test_metrics['mae_overall']:.1f} mg/dL")

        return test_metrics


def main():
    """Main comparison pipeline."""

    logger.info("=" * 60)
    logger.info("T1D-AI Model Architecture Comparison")
    logger.info("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    logger.info("Loading data...")
    glucose_df, treatments_df = load_data()
    logger.info(f"Loaded {len(glucose_df)} glucose readings, {len(treatments_df)} treatments")

    # Create splits
    logger.info("Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_splits(glucose_df, treatments_df)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Training config
    config = {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 15,
    }

    # Models to compare
    models = [
        (SimpleMLP, {'n_features': 6, 'hidden_size': 64}),
        (LSTMOnly, {'n_features': 6, 'hidden_size': 64, 'n_layers': 2}),
        (PhysicsLSTM, {'n_features': 6, 'hidden_size': 64, 'n_layers': 2}),
        (PhysicsLSTMAttention, {'n_features': 6, 'hidden_size': 64, 'n_layers': 2, 'n_heads': 4}),
        # Larger models
        (PhysicsLSTM, {'n_features': 6, 'hidden_size': 128, 'n_layers': 3}),
        (PhysicsLSTMAttention, {'n_features': 6, 'hidden_size': 128, 'n_layers': 3, 'n_heads': 8}),
    ]

    # Run experiments
    results = {}
    for model_class, model_kwargs in models:
        metrics = run_experiment(
            model_class, model_kwargs,
            train_loader, val_loader, test_loader,
            config, device
        )
        model_name = model_class.__name__
        if model_kwargs.get('hidden_size', 64) > 64:
            model_name += "_Large"
        results[model_name] = metrics

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<25} {'MAE@30':<10} {'MAE@60':<10} {'MAE@120':<10} {'Overall':<10}")
    logger.info("-" * 60)

    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['mae_overall']):
        logger.info(
            f"{model_name:<25} "
            f"{metrics.get('mae_30min', 0):<10.1f} "
            f"{metrics.get('mae_60min', 0):<10.1f} "
            f"{metrics.get('mae_120min', 0):<10.1f} "
            f"{metrics['mae_overall']:<10.1f}"
        )

    logger.info("\n" + "=" * 60)
    logger.info(f"View results in MLflow: {MLFLOW_TRACKING_URI}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
