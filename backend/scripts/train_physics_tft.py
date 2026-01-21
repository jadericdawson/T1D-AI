#!/usr/bin/env python3
"""
Train Physics-Informed TFT Model with MLflow Tracking

This script:
1. Loads historic glucose and treatment data
2. Applies quality filtering (skips school hours, incomplete periods)
3. Creates train/val/test splits (70/15/15)
4. Trains the physics-informed TFT model
5. Logs all metrics, params, and artifacts to MLflow
6. Evaluates on held-out test set

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/train_physics_tft.py

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
from typing import List, Tuple, Dict, Optional
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split

# Local imports
from ml.models.physics_tft import (
    PhysicsInformedTFT, create_physics_tft,
    PREDICTION_HORIZONS, DEFAULT_ISF, DEFAULT_ICR
)
from ml.training.data_quality import TrainingDataQualityFilter, filter_for_training

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
EXPERIMENT_NAME = "T1D-AI/Physics-TFT-v2"

# Training config
TRAIN_CONFIG = {
    'seq_length': 24,           # 2 hours of history (5-min intervals)
    'hidden_size': 64,
    'n_features': 32,           # Reduced from 69 - focus on key features
    'n_heads': 4,
    'n_lstm_layers': 2,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,
    'weight_decay': 1e-5,
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'min_completeness_score': 0.3,  # Lower threshold to get more training data
    'exclude_school_hours': True,
}


class GlucosePredictionDataset(Dataset):
    """Dataset for glucose prediction training."""

    def __init__(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        seq_length: int = 24,
        horizons: List[int] = PREDICTION_HORIZONS,
    ):
        self.seq_length = seq_length
        self.horizons = horizons
        self.samples = []

        # Sort by timestamp
        glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

        # Create samples
        self._create_samples(glucose_df, treatments_df)

    def _create_samples(self, glucose_df: pd.DataFrame, treatments_df: pd.DataFrame):
        """Create training samples from data."""
        logger.info(f"Creating samples from {len(glucose_df)} glucose readings...")

        # For each possible starting point
        for i in range(len(glucose_df) - self.seq_length - max(self.horizons) // 5):
            try:
                sample = self._create_single_sample(glucose_df, treatments_df, i)
                if sample is not None:
                    self.samples.append(sample)
            except Exception as e:
                continue

        logger.info(f"Created {len(self.samples)} training samples")

    def _create_single_sample(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        start_idx: int,
    ) -> Optional[Dict]:
        """Create a single training sample."""
        # Get sequence of glucose readings
        end_idx = start_idx + self.seq_length
        sequence = glucose_df.iloc[start_idx:end_idx]

        # Check for gaps (require continuous data)
        time_diffs = sequence['timestamp'].diff().dt.total_seconds() / 60
        if time_diffs.max() > 10:  # Max 10 min gap allowed
            return None

        # Current values
        current_ts = sequence.iloc[-1]['timestamp']
        current_bg = float(sequence.iloc[-1]['value'])

        # Calculate trend (rate per 5 min from last 6 readings)
        recent_values = sequence['value'].values[-6:]
        trend = (recent_values[-1] - recent_values[0]) / 5 if len(recent_values) >= 2 else 0

        # Get treatments in the past 4 hours
        treatments_window_start = current_ts - timedelta(hours=4)
        recent_treatments = treatments_df[
            (treatments_df['timestamp'] >= treatments_window_start) &
            (treatments_df['timestamp'] <= current_ts)
        ]

        # Calculate IOB (simplified linear decay over 4 hours)
        iob = 0.0
        for _, t in recent_treatments.iterrows():
            if t.get('insulin', 0) > 0:
                time_since = (current_ts - t['timestamp']).total_seconds() / 3600
                remaining = max(0, 1 - time_since / 4) * t['insulin']
                iob += remaining

        # Calculate COB (simplified linear decay over 3 hours)
        cob = 0.0
        for _, t in recent_treatments.iterrows():
            if t.get('carbs', 0) > 0:
                time_since = (current_ts - t['timestamp']).total_seconds() / 3600
                remaining = max(0, 1 - time_since / 3) * t['carbs']
                cob += remaining

        # Time features
        hour = current_ts.hour + current_ts.minute / 60
        day_of_week = current_ts.weekday()
        month = current_ts.month

        # Get targets (future glucose values)
        targets = {}
        for horizon in self.horizons:
            target_idx = end_idx + horizon // 5
            if target_idx < len(glucose_df):
                targets[horizon] = float(glucose_df.iloc[target_idx]['value'])
            else:
                return None  # Skip if we don't have target

        # Build feature sequence
        features = self._build_features(sequence, recent_treatments, iob, cob)

        return {
            'features': features,
            'current_bg': current_bg,
            'iob': iob,
            'cob': cob,
            'trend': trend,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'targets': targets,
            'timestamp': current_ts,
        }

    def _build_features(
        self,
        glucose_seq: pd.DataFrame,
        treatments: pd.DataFrame,
        iob: float,
        cob: float,
    ) -> np.ndarray:
        """Build feature array for the sequence."""
        n_features = 32  # Match TRAIN_CONFIG

        # Initialize feature array
        features = np.zeros((len(glucose_seq), n_features))

        values = glucose_seq['value'].values
        trends = glucose_seq['trend'].values if 'trend' in glucose_seq.columns else np.zeros(len(glucose_seq))

        # Core features
        features[:, 0] = values / 200  # Normalized glucose
        features[:, 1] = np.gradient(values)  # Rate of change
        features[:, 2] = np.gradient(np.gradient(values))  # Acceleration

        # Convert trend codes to numeric
        trend_map = {
            'DoubleDown': -3, 'SingleDown': -2, 'FortyFiveDown': -1,
            'Flat': 0, 'FortyFiveUp': 1, 'SingleUp': 2, 'DoubleUp': 3,
            -3: -3, -2: -2, -1: -1, 0: 0, 1: 1, 2: 2, 3: 3,
        }
        numeric_trends = [trend_map.get(t, 0) for t in trends]
        features[:, 3] = numeric_trends

        # IOB/COB (constant for sequence, model will learn time decay)
        features[:, 4] = iob
        features[:, 5] = cob

        # Rolling statistics (windows: 15, 30, 60 min = 3, 6, 12 readings)
        for i, window in enumerate([3, 6, 12]):
            if len(values) >= window:
                rolling_mean = pd.Series(values).rolling(window, min_periods=1).mean().values
                rolling_std = pd.Series(values).rolling(window, min_periods=1).std().fillna(0).values
                features[:, 6 + i*2] = rolling_mean / 200
                features[:, 7 + i*2] = rolling_std / 50

        # Time features (cyclical)
        timestamps = pd.to_datetime(glucose_seq['timestamp'])
        hours = timestamps.dt.hour + timestamps.dt.minute / 60

        features[:, 12] = np.sin(2 * np.pi * hours / 24)
        features[:, 13] = np.cos(2 * np.pi * hours / 24)
        features[:, 14] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        features[:, 15] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)

        # Variability metrics
        if len(values) >= 6:
            cv = np.std(values[-6:]) / (np.mean(values[-6:]) + 1e-6)
            features[:, 16] = cv

        # Range indicators (binary)
        features[:, 17] = (values < 70).astype(float)  # Low
        features[:, 18] = ((values >= 70) & (values <= 180)).astype(float)  # In range
        features[:, 19] = (values > 180).astype(float)  # High

        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors
        features = torch.tensor(sample['features'], dtype=torch.float32)
        targets = torch.tensor([sample['targets'][h] for h in self.horizons], dtype=torch.float32)

        return {
            'features': features,
            'current_bg': torch.tensor(sample['current_bg'], dtype=torch.float32),
            'iob': torch.tensor(sample['iob'], dtype=torch.float32),
            'cob': torch.tensor(sample['cob'], dtype=torch.float32),
            'trend': torch.tensor(sample['trend'], dtype=torch.float32),
            'hour': torch.tensor(sample['hour'], dtype=torch.float32),
            'day_of_week': torch.tensor(sample['day_of_week'], dtype=torch.float32),
            'month': torch.tensor(sample['month'], dtype=torch.float32),
            'targets': targets,
        }


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load glucose and treatment data from JSONL files."""
    logger.info("Loading data...")

    # Load glucose
    glucose_records = []
    with open(GLUCOSE_FILE, 'r') as f:
        for line in f:
            if line.strip():
                glucose_records.append(json.loads(line))

    glucose_df = pd.DataFrame(glucose_records)
    glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)
    glucose_df['timestamp'] = glucose_df['timestamp'].dt.tz_localize(None)  # Make tz-naive
    glucose_df = glucose_df.drop_duplicates(subset=['timestamp', 'value'])
    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(glucose_df)} glucose readings")
    logger.info(f"Date range: {glucose_df['timestamp'].min()} to {glucose_df['timestamp'].max()}")

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
                })
                # Extract history treatments
                for hist in record.get('history', []):
                    if hist.get('carbs', 0) > 0 or hist.get('insulin', 0) > 0:
                        treatment_records.append({
                            'timestamp': hist['ts'],
                            'insulin': hist.get('insulin', 0),
                            'carbs': hist.get('carbs', 0),
                        })

    treatments_df = pd.DataFrame(treatment_records)
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
    treatments_df['timestamp'] = treatments_df['timestamp'].dt.tz_localize(None)  # Make tz-naive
    treatments_df = treatments_df.drop_duplicates(subset=['timestamp'])
    treatments_df = treatments_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(treatments_df)} treatments")

    return glucose_df, treatments_df


def create_data_splits(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
    config: Dict,
) -> Tuple[GlucosePredictionDataset, GlucosePredictionDataset, GlucosePredictionDataset]:
    """Create train/val/test splits with quality filtering."""
    logger.info("Applying quality filter...")

    # Apply quality filter
    quality_filter = TrainingDataQualityFilter(
        exclude_school_weekdays=config['exclude_school_hours'],
        min_completeness_score=config['min_completeness_score'],
    )

    # Generate quality report
    report = quality_filter.generate_quality_report(glucose_df, treatments_df)
    logger.info(f"Data quality report: {json.dumps(report, indent=2, default=str)}")

    # Filter data
    filtered_glucose, filtered_treatments, windows = quality_filter.create_filtered_dataset(
        glucose_df, treatments_df
    )

    if len(filtered_glucose) == 0:
        raise ValueError("No data passed quality filter!")

    logger.info(f"After filtering: {len(filtered_glucose)} glucose, {len(filtered_treatments)} treatments")

    # Split by time (not random) to prevent data leakage
    timestamps = filtered_glucose['timestamp'].unique()
    timestamps = np.sort(timestamps)

    n_total = len(timestamps)
    train_end = int(n_total * config['train_split'])
    val_end = int(n_total * (config['train_split'] + config['val_split']))

    train_end_ts = timestamps[train_end]
    val_end_ts = timestamps[val_end]

    train_glucose = filtered_glucose[filtered_glucose['timestamp'] < train_end_ts]
    train_treatments = filtered_treatments[filtered_treatments['timestamp'] < train_end_ts]

    val_glucose = filtered_glucose[
        (filtered_glucose['timestamp'] >= train_end_ts) &
        (filtered_glucose['timestamp'] < val_end_ts)
    ]
    val_treatments = filtered_treatments[
        (filtered_treatments['timestamp'] >= train_end_ts) &
        (filtered_treatments['timestamp'] < val_end_ts)
    ]

    test_glucose = filtered_glucose[filtered_glucose['timestamp'] >= val_end_ts]
    test_treatments = filtered_treatments[filtered_treatments['timestamp'] >= val_end_ts]

    logger.info(f"Train: {len(train_glucose)} readings, Val: {len(val_glucose)}, Test: {len(test_glucose)}")

    # Create datasets
    train_dataset = GlucosePredictionDataset(
        train_glucose, train_treatments, config['seq_length']
    )
    val_dataset = GlucosePredictionDataset(
        val_glucose, val_treatments, config['seq_length']
    )
    test_dataset = GlucosePredictionDataset(
        test_glucose, test_treatments, config['seq_length']
    )

    return train_dataset, val_dataset, test_dataset, report


def train_epoch(
    model: PhysicsInformedTFT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move to device
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)

        # Forward pass
        output = model(
            features=features,
            current_bg=batch['current_bg'].to(device),
            iob=batch['iob'].to(device),
            cob=batch['cob'].to(device),
            trend=batch['trend'].to(device),
            hour=batch['hour'].to(device),
            day_of_week=batch['day_of_week'].to(device),
            month=batch['month'].to(device),
        )

        # Calculate loss for all horizons
        loss = 0.0
        for i, pred_dict in enumerate(output.predictions):
            pred = pred_dict['value']
            target = targets[:, i]
            loss += criterion(pred, target)

        loss = loss / len(PREDICTION_HORIZONS)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: PhysicsInformedTFT,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0

    # Per-horizon metrics
    horizon_mae = {h: [] for h in PREDICTION_HORIZONS}
    horizon_rmse = {h: [] for h in PREDICTION_HORIZONS}

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            output = model(
                features=features,
                current_bg=batch['current_bg'].to(device),
                iob=batch['iob'].to(device),
                cob=batch['cob'].to(device),
                trend=batch['trend'].to(device),
                hour=batch['hour'].to(device),
                day_of_week=batch['day_of_week'].to(device),
                month=batch['month'].to(device),
            )

            # Loss and metrics for each horizon
            for i, (pred_dict, horizon) in enumerate(zip(output.predictions, PREDICTION_HORIZONS)):
                pred = pred_dict['value']
                target = targets[:, i]

                # MAE and RMSE
                errors = torch.abs(pred - target)
                horizon_mae[horizon].extend(errors.cpu().numpy().tolist())
                horizon_rmse[horizon].extend((errors ** 2).cpu().numpy().tolist())

                total_loss += criterion(pred, target).item()

    # Aggregate metrics
    metrics = {
        'loss': total_loss / len(data_loader) / len(PREDICTION_HORIZONS),
    }

    for horizon in PREDICTION_HORIZONS:
        mae = np.mean(horizon_mae[horizon])
        rmse = np.sqrt(np.mean(horizon_rmse[horizon]))
        metrics[f'mae_{horizon}min'] = mae
        metrics[f'rmse_{horizon}min'] = rmse

    # Overall MAE and RMSE
    all_mae = sum(horizon_mae.values(), [])
    all_rmse = sum(horizon_rmse.values(), [])
    metrics['mae_overall'] = np.mean(all_mae)
    metrics['rmse_overall'] = np.sqrt(np.mean(all_rmse))

    return metrics


def train_model(config: Dict) -> Tuple[PhysicsInformedTFT, Dict]:
    """Main training function."""
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    glucose_df, treatments_df = load_data()

    # Create splits
    train_dataset, val_dataset, test_dataset, quality_report = create_data_splits(
        glucose_df, treatments_df, config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create model
    model = PhysicsInformedTFT(
        n_features=config['n_features'],
        hidden_size=config['hidden_size'],
        n_heads=config['n_heads'],
        n_lstm_layers=config['n_lstm_layers'],
        dropout=config['dropout'],
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Start MLflow run
    with mlflow.start_run(run_name=f"physics_tft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param('model_params', sum(p.numel() for p in model.parameters()))
        mlflow.log_param('train_samples', len(train_dataset))
        mlflow.log_param('val_samples', len(val_dataset))
        mlflow.log_param('test_samples', len(test_dataset))

        # Log data quality report
        mlflow.log_dict(quality_report, 'data_quality_report.json')

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(config['epochs']):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, config['grad_clip']
            )

            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            for name, value in val_metrics.items():
                mlflow.log_metric(f'val_{name}', value, step=epoch)

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', current_lr, step=epoch)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae_overall']:.1f} mg/dL"
            )

            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info(f"New best model! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final test evaluation
        test_metrics = evaluate(model, test_loader, criterion, device)

        logger.info("=" * 50)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 50)
        for name, value in test_metrics.items():
            mlflow.log_metric(f'test_{name}', value)
            if 'mae' in name or 'rmse' in name:
                logger.info(f"Test {name}: {value:.1f} mg/dL")

        # Save model
        output_path = OUTPUT_DIR / 'physics_tft_model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'test_metrics': test_metrics,
            'quality_report': quality_report,
            'created_at': datetime.now().isoformat(),
            'horizons': PREDICTION_HORIZONS,
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Model saved to {output_path}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        return model, test_metrics


if __name__ == '__main__':
    logger.info("Starting Physics-Informed TFT Training")
    logger.info(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")

    model, metrics = train_model(TRAIN_CONFIG)

    logger.info("\nTraining complete!")
    logger.info(f"Overall Test MAE: {metrics['mae_overall']:.1f} mg/dL")
    logger.info(f"Overall Test RMSE: {metrics['rmse_overall']:.1f} mg/dL")

    # Per-horizon summary
    logger.info("\nPer-horizon MAE:")
    for horizon in PREDICTION_HORIZONS:
        mae = metrics.get(f'mae_{horizon}min', 0)
        logger.info(f"  +{horizon}min: {mae:.1f} mg/dL")
