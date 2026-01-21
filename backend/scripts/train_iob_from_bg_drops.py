#!/usr/bin/env python3
"""
Train IOB Model from BG Drop Sequences

The bolus_moments.jsonl file contains gold data for IOB training:
- bolus_ts: when insulin was given
- bolus_insulin: how much insulin
- isf: the ISF at that moment
- bg_drop_sequence: BG every 5 min after bolus for 2 hours

This lets us learn EXACTLY how insulin absorbs for this person!

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/train_iob_from_bg_drops.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = Path("/home/jadericdawson/Documents/AI/T1D-AI/data/bolus_moments.jsonl")
OUTPUT_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/models")
MLFLOW_URI = "http://localhost:5002"


def load_bolus_moments():
    """Load bolus moments with BG drop sequences."""
    moments = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            if line.strip():
                moments.append(json.loads(line))
    logger.info(f"Loaded {len(moments)} bolus moments")
    return moments


def create_iob_samples(moments):
    """
    Create IOB training samples from BG drop sequences.

    For each bolus, we know:
    - Starting BG (from history)
    - BG at each 5-min interval after (from bg_drop_sequence)
    - ISF for this person/time

    We can calculate: absorbed_insulin = bg_drop / isf
    Then: remaining_fraction = 1 - (absorbed_insulin / bolus_units)
    """
    samples = []

    for moment in moments:
        bolus_units = moment['bolus_insulin']
        isf = moment.get('isf', 55)  # Use recorded ISF!
        bolus_ts = datetime.fromisoformat(moment['bolus_ts'].replace('Z', '+00:00'))

        # Get starting BG (last entry in history)
        if not moment.get('history'):
            continue
        start_bg = moment['history'][-1]['bg']

        # Process BG drop sequence
        bg_drops = moment.get('bg_drop_sequence', [])
        if not bg_drops:
            continue

        for drop_point in bg_drops:
            drop_ts = datetime.fromisoformat(drop_point['ts'].replace('Z', '+00:00'))
            drop_bg = drop_point['bg']

            # Calculate minutes since bolus
            minutes_since = (drop_ts - bolus_ts).total_seconds() / 60

            if minutes_since <= 0 or minutes_since > 360:
                continue

            # Calculate BG drop from start
            bg_drop = start_bg - drop_bg

            # Estimate absorbed insulin
            # absorbed = bg_drop / isf (but capped to bolus amount)
            if isf > 0 and bolus_units > 0:
                absorbed = bg_drop / isf
                absorbed = max(0, min(bolus_units, absorbed))  # Clamp to [0, bolus]
                remaining_fraction = 1 - (absorbed / bolus_units)
                remaining_fraction = max(0, min(1, remaining_fraction))
            else:
                remaining_fraction = 0.5

            samples.append({
                'bolus_units': bolus_units,
                'minutes_since': minutes_since,
                'remaining_fraction': remaining_fraction,
                'isf': isf,
                'start_bg': start_bg,
                'current_bg': drop_bg,
                'bg_drop': bg_drop,
                'hour': bolus_ts.hour,
            })

    logger.info(f"Created {len(samples)} IOB training samples")
    return samples


class IOBFromBGDataset(Dataset):
    """Dataset for IOB model using BG drop data."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Key features: time is PRIMARY driver
        features = torch.tensor([
            s['minutes_since'] / 360.0,  # Time - most important!
            s['bolus_units'] / 10.0,
            np.sin(2 * np.pi * s['hour'] / 24),
            np.cos(2 * np.pi * s['hour'] / 24),
            s['isf'] / 100.0,
            s['start_bg'] / 200.0,
        ], dtype=torch.float32)

        target = torch.tensor([s['remaining_fraction']], dtype=torch.float32)

        return features, target


class IOBDecayNet(nn.Module):
    """
    IOB model that explicitly learns decay curve.

    Architecture ensures time is the primary driver:
    - First layer processes time separately
    - Then combines with other features
    """

    def __init__(self):
        super().__init__()

        # Time processing branch (learns decay shape)
        self.time_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Tanh(),
        )

        # Context branch (modulates decay based on context)
        self.context_branch = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )

        # Combine and output
        self.combine = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Initialize time branch to approximate exponential decay
        self._init_time_branch()

    def _init_time_branch(self):
        """Initialize time branch to start near exponential decay."""
        # The decay should be ~0.5 at t=81 min (half-life)
        # Initialize with small weights so output starts reasonable
        for m in self.time_branch:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Split features
        time = x[:, 0:1]  # Just time feature
        context = x[:, 1:]  # Other features

        # Process separately
        time_features = self.time_branch(time)
        context_features = self.context_branch(context)

        # Combine
        combined = torch.cat([time_features, context_features], dim=1)
        return self.combine(combined)


def train_model(model, train_loader, val_loader, epochs=200):
    """Train IOB model."""
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                output = model(features)
                val_loss += criterion(output, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


def test_model(model):
    """Test the trained model."""
    model.eval()

    print("\n" + "=" * 70)
    print("IOB MODEL TEST: 2U bolus, ISF=55, start BG=150")
    print("=" * 70)
    print(f"{'Minutes':<10} {'Formula':<12} {'Model':<12} {'Actual IOB':<12}")
    print("-" * 70)

    for mins in [0, 15, 30, 45, 60, 90, 120, 150, 180, 240, 300]:
        # Standard formula
        formula = 0.5 ** (mins / 81.0)

        # Model prediction
        features = torch.tensor([[
            mins / 360.0,
            2.0 / 10.0,
            np.sin(2 * np.pi * 14 / 24),
            np.cos(2 * np.pi * 14 / 24),
            55 / 100.0,
            150 / 200.0,
        ]], dtype=torch.float32)

        with torch.no_grad():
            model_pred = model(features).item()

        # Actual IOB = fraction * bolus_units
        actual_iob = model_pred * 2.0

        print(f"{mins:<10} {formula*100:>8.1f}%    {model_pred*100:>8.1f}%    {actual_iob:>8.2f}U")


def main():
    logger.info("=" * 60)
    logger.info("Training IOB Model from BG Drop Sequences")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("T1D-AI/IOB-From-BG-Drops")

    # Load and prepare data
    moments = load_bolus_moments()
    samples = create_iob_samples(moments)

    if len(samples) < 50:
        logger.warning(f"Only {len(samples)} samples - may not train well!")

    # Analyze the data
    fractions = [s['remaining_fraction'] for s in samples]
    times = [s['minutes_since'] for s in samples]
    logger.info(f"Remaining fraction range: {min(fractions):.2f} - {max(fractions):.2f}")
    logger.info(f"Time range: {min(times):.0f} - {max(times):.0f} min")

    # Split data
    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = IOBFromBGDataset(train_samples)
    val_ds = IOBFromBGDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Train
    with mlflow.start_run(run_name=f"IOB_BG_Drop_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param('train_samples', len(train_samples))
        mlflow.log_param('val_samples', len(val_samples))

        model = IOBDecayNet()
        model, best_loss = train_model(model, train_loader, val_loader, epochs=200)

        mlflow.log_metric('best_val_loss', best_loss)
        mlflow.pytorch.log_model(model, "model")

        # Save locally
        torch.save(model.state_dict(), OUTPUT_DIR / 'iob_from_bg_drops.pth')
        logger.info(f"Saved model to {OUTPUT_DIR / 'iob_from_bg_drops.pth'}")

    # Test
    test_model(model)

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
