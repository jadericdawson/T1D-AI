#!/usr/bin/env python3
"""
Forcing Function Model Training Pipeline

Trains the physics-informed BG prediction system with IOB/COB as forcing functions.

Architecture:
1. IOB Forcing Model - Predicts remaining IOB fraction (negative BG pressure)
2. COB Forcing Model - Predicts remaining COB fraction (positive BG pressure)
3. Physics Baseline - Combines IOB/COB with ISF for deterministic prediction
4. Residual Model - Neural adjustments for secondary factors (+/- 25 mg/dL max)

Key Insight from User:
"IOB and COB can influence BG with upward and downward pressure that are
independently learned as their own models. When both are present their
'pressures' negate one another. They are the primary forcing functions
applied to current BG."

Expected Improvements (per arXiv:2502.00065v1 benchmark):
- Pure ML on CGM-only: RMSE ~22.5 mg/dL at 30 min
- Our approach with IOB/COB: Expected 15-25% improvement

Usage:
    cd /home/jadericdawson/Documents/AI/T1D-AI/backend
    PYTHONPATH=./src python3 scripts/train_forcing_models.py

MLflow Server:
    Must be running on port 5002 before training:
    mlflow server --host 0.0.0.0 --port 5002 --backend-store-uri file://./mlruns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mlflow
import mlflow.pytorch
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/data")
OUTPUT_DIR = Path("/home/jadericdawson/Documents/AI/T1D-AI/backend/models")
MLFLOW_URI = "http://localhost:5002"

# Training configuration
USER_ID = "emrys"
DEFAULT_ISF = 55.0
DEFAULT_ICR = 10.0


# =============================================================================
# Data Loading
# =============================================================================

def load_bolus_moments() -> List[Dict]:
    """Load bolus moments with BG drop sequences for IOB training."""
    data_file = DATA_DIR / "bolus_moments.jsonl"
    moments = []

    if not data_file.exists():
        logger.warning(f"No bolus moments file at {data_file}")
        return moments

    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                moments.append(json.loads(line))

    logger.info(f"Loaded {len(moments)} bolus moments")
    return moments


def load_glucose_readings() -> pd.DataFrame:
    """Load glucose readings from gluroo_readings.jsonl."""
    data_file = DATA_DIR / "gluroo_readings.jsonl"
    readings = []

    if not data_file.exists():
        logger.warning(f"No glucose readings file at {data_file}")
        return pd.DataFrame()

    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                readings.append(json.loads(line))

    df = pd.DataFrame(readings)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} glucose readings")
    return df


def estimate_gi_from_carbs(carbs: float) -> dict:
    """Estimate GI, protein, fat from carb count when no food notes available."""
    if carbs <= 15:
        return {'glycemicIndex': 65, 'protein': 2.0, 'fat': 2.0}  # Snack
    elif carbs <= 40:
        return {'glycemicIndex': 55, 'protein': 15.0, 'fat': 10.0}  # Light meal
    else:
        return {'glycemicIndex': 50, 'protein': 25.0, 'fat': 20.0}  # Full meal


def load_enriched_treatments() -> pd.DataFrame:
    """Load enriched treatments with GPT-4.1 food composition data."""
    # Look for enriched treatments in data folder
    enriched_file = DATA_DIR / "enriched_treatments.jsonl"

    if not enriched_file.exists():
        # Fall back to extracting from gluroo_readings (has carbs/insulin)
        logger.info("No enriched treatments file, extracting from gluroo readings")

        gluroo_file = DATA_DIR / "gluroo_readings.jsonl"
        treatments = []
        seen_ts = set()  # Avoid duplicates

        if gluroo_file.exists():
            with open(gluroo_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        ts = entry.get('timestamp')
                        carbs = entry.get('carbs', 0)
                        insulin = entry.get('insulin', 0)

                        # Skip if no treatment or already seen
                        if (carbs == 0 and insulin == 0) or ts in seen_ts:
                            continue
                        seen_ts.add(ts)

                        # Estimate macros from carb count if not provided
                        protein = entry.get('protein', 0)
                        fat = entry.get('fat', 0)
                        gi = 55  # Default

                        if carbs > 0 and protein == 0 and fat == 0:
                            est = estimate_gi_from_carbs(carbs)
                            gi = est['glycemicIndex']
                            protein = est['protein']
                            fat = est['fat']

                        if carbs > 0:
                            treatments.append({
                                'timestamp': ts,
                                'carbs': carbs,
                                'insulin': 0,
                                'type': 'carbs',
                                'glycemicIndex': gi,
                                'fat': fat,
                                'protein': protein,
                            })

                        if insulin > 0:
                            treatments.append({
                                'timestamp': ts,
                                'insulin': insulin,
                                'carbs': 0,
                                'type': 'insulin',
                                'glycemicIndex': 55,
                                'fat': 0,
                                'protein': 0,
                            })

        # Also include bolus_moments insulin entries
        moments = load_bolus_moments()
        for m in moments:
            ts = m['bolus_ts']
            if ts not in seen_ts and m.get('bolus_insulin', 0) > 0:
                seen_ts.add(ts)
                treatments.append({
                    'timestamp': ts,
                    'insulin': m['bolus_insulin'],
                    'carbs': 0,
                    'type': 'insulin',
                    'glycemicIndex': 55,
                    'fat': 0,
                    'protein': 0,
                })

        if not treatments:
            return pd.DataFrame()

        df = pd.DataFrame(treatments)
    else:
        treatments = []
        with open(enriched_file, 'r') as f:
            for line in f:
                if line.strip():
                    treatments.append(json.loads(line))
        df = pd.DataFrame(treatments)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} treatments ({len(df[df['carbs']>0])} with carbs, {len(df[df['insulin']>0])} with insulin)")
    return df


# =============================================================================
# Stage 1: IOB Forcing Model Training
# =============================================================================

class IOBTrainingDataset(Dataset):
    """Dataset for IOB forcing model training from BG drop sequences."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Features: time is PRIMARY driver
        features = torch.tensor([
            s['minutes_since'] / 360.0,  # Normalized time (most important)
            s['bolus_units'] / 10.0,      # Normalized bolus
            np.sin(2 * np.pi * s['hour'] / 24),  # Circadian
            np.cos(2 * np.pi * s['hour'] / 24),
            1.0 if s.get('is_dawn_window', False) else 0.0,  # Dawn flag
            1.0 if s.get('is_active_meal', False) else 0.0,  # Meal flag
            s.get('half_life_min', 54.0) / 100.0,  # Personalized half-life
        ], dtype=torch.float32)

        target = torch.tensor([s['remaining_fraction']], dtype=torch.float32)

        return features, target


def create_iob_training_samples(moments: List[Dict]) -> List[Dict]:
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
        bolus_units = moment.get('bolus_insulin', 0)
        if bolus_units <= 0:
            continue

        isf = moment.get('isf', DEFAULT_ISF)
        bolus_ts = datetime.fromisoformat(moment['bolus_ts'].replace('Z', '+00:00'))
        hour = bolus_ts.hour

        # Check if dawn window (4-8 AM)
        is_dawn_window = 4 <= hour < 8

        # Check if active meal (carbs within 30 min)
        is_active_meal = moment.get('carbs', 0) > 5

        # Get starting BG (last entry in history)
        history = moment.get('history', [])
        if not history:
            continue
        start_bg = history[-1].get('bg', 120)

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
            if isf > 0 and bolus_units > 0:
                absorbed = bg_drop / isf
                absorbed = max(0, min(bolus_units, absorbed))  # Clamp
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
                'hour': hour,
                'is_dawn_window': is_dawn_window,
                'is_active_meal': is_active_meal,
                'half_life_min': 54.0,  # Personalized for this child
            })

    logger.info(f"Created {len(samples)} IOB training samples")
    return samples


def train_iob_forcing_model(
    samples: List[Dict],
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.003,
) -> Tuple[nn.Module, float, float]:
    """
    Train the IOB forcing model.

    Returns:
        model: Trained model
        best_val_loss: Best validation loss
        learned_half_life: Estimated personalized half-life
    """
    from ml.models.iob_forcing import IOBForcingModel

    # Split data
    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = IOBTrainingDataset(train_samples)
    val_ds = IOBTrainingDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Create model
    model = IOBForcingModel()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        if epoch % 50 == 0:
            logger.info(f"IOB Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Log to MLflow
        mlflow.log_metrics({
            'iob_train_loss': train_loss,
            'iob_val_loss': val_loss,
        }, step=epoch)

    if best_state:
        model.load_state_dict(best_state)

    # Estimate personalized half-life by finding where model predicts 0.5
    learned_half_life = estimate_half_life_from_model(model)

    return model, best_loss, learned_half_life


def estimate_half_life_from_model(model: nn.Module) -> float:
    """Estimate personalized half-life by finding where model predicts 0.5 remaining."""
    model.eval()

    # Binary search for half-life
    low, high = 30, 150
    while high - low > 1:
        mid = (low + high) / 2

        features = torch.tensor([[
            mid / 360.0,      # Time
            0.5,              # Typical bolus (5U)
            0.0, 1.0,         # Noon
            0.0,              # Not dawn
            0.0,              # Not meal
            0.54,             # Default half-life hint
        ]], dtype=torch.float32)

        with torch.no_grad():
            pred = model(features).item()

        if pred > 0.5:
            low = mid
        else:
            high = mid

    half_life = (low + high) / 2
    logger.info(f"Estimated personalized half-life: {half_life:.1f} min")
    return half_life


# =============================================================================
# Stage 2: COB Forcing Model Training
# =============================================================================

class COBTrainingDataset(Dataset):
    """Dataset for COB forcing model training."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        half_life = s.get('half_life_min', 45.0)

        features = torch.tensor([
            s['minutes_since'] / 360.0,       # Time
            s['carbs'] / 100.0,               # Carb amount
            s.get('glycemic_index', 55) / 100.0,  # GI
            s.get('fat', 0) / 50.0,           # Fat content
            s.get('protein', 0) / 50.0,       # Protein content
            half_life / 100.0,                # Estimated half-life
            np.sin(2 * np.pi * s['hour'] / 24),
            np.cos(2 * np.pi * s['hour'] / 24),
            1.0 if s.get('is_recent_meal', False) else 0.0,
            1.0 if s.get('is_stacking', False) else 0.0,
        ], dtype=torch.float32)

        target = torch.tensor([s['remaining_fraction']], dtype=torch.float32)

        return features, target


def create_cob_training_samples(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
) -> List[Dict]:
    """
    Create COB training samples by tracking BG rise after carbs.

    For meals without significant insulin overlap, we can estimate
    carb absorption from BG rise patterns.
    """
    samples = []

    if glucose_df.empty or treatments_df.empty:
        return samples

    # Filter carb treatments
    carb_treats = treatments_df[treatments_df['carbs'] > 5].copy()

    for _, treat in carb_treats.iterrows():
        treat_time = treat['timestamp']
        carbs = treat['carbs']
        gi = treat.get('glycemicIndex', 55) or 55
        fat = treat.get('fat', 0) or 0
        protein = treat.get('protein', 0) or 0
        hour = treat_time.hour

        # Estimate half-life based on food composition
        if gi >= 70:
            base_half_life = 30.0  # Fast
        elif gi >= 55:
            base_half_life = 45.0  # Medium
        else:
            base_half_life = 60.0  # Slow

        # Fat extends duration
        fat_extension = (fat / 10.0) * 15.0
        half_life = base_half_life + fat_extension

        # Generate samples at each horizon using exponential decay formula
        for horizon in range(5, 361, 5):
            # Theoretical remaining fraction (exponential decay)
            remaining_fraction = 0.5 ** (horizon / half_life)
            remaining_fraction = max(0, min(1, remaining_fraction))

            samples.append({
                'carbs': carbs,
                'minutes_since': horizon,
                'remaining_fraction': remaining_fraction,
                'glycemic_index': gi,
                'fat': fat,
                'protein': protein,
                'hour': hour,
                'half_life_min': half_life,
                'is_recent_meal': horizon <= 30,
                'is_stacking': False,  # Could detect from overlapping meals
            })

    logger.info(f"Created {len(samples)} COB training samples")
    return samples


def train_cob_forcing_model(
    samples: List[Dict],
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 0.003,
) -> Tuple[nn.Module, float]:
    """Train the COB forcing model."""
    from ml.models.cob_forcing import COBForcingModel

    # Split data
    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = COBTrainingDataset(train_samples)
    val_ds = COBTrainingDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = COBForcingModel()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
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

        if epoch % 50 == 0:
            logger.info(f"COB Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        mlflow.log_metrics({
            'cob_train_loss': train_loss,
            'cob_val_loss': val_loss,
        }, step=epoch)

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


# =============================================================================
# Stage 3 & 4: Physics Baseline + Residual Training
# =============================================================================

class ResidualTrainingDataset(Dataset):
    """Dataset for residual model training."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        features = torch.tensor([
            np.sin(2 * np.pi * s['hour'] / 24),  # Hour sin
            np.cos(2 * np.pi * s['hour'] / 24),  # Hour cos
            np.sin(2 * np.pi * s['day_of_week'] / 7),  # DOW sin
            np.cos(2 * np.pi * s['day_of_week'] / 7),  # DOW cos
            np.sin(2 * np.pi * s['day_of_year'] / 365),  # DOY sin
            np.cos(2 * np.pi * s['day_of_year'] / 365),  # DOY cos
            s.get('lunar_sin', 0),  # Lunar cycle
            s.get('lunar_cos', 1),
            1.0 if s.get('is_weekend', False) else 0.0,
            s.get('recent_trend', 0) / 10.0,
            s.get('bg_volatility', 0) / 50.0,
            s.get('dawn_intensity', 0.5),
            s['horizon_min'] / 120.0,
            s['physics_pred'] / 200.0,
        ], dtype=torch.float32)

        # Target: actual - physics (clamped to +/- 25)
        residual = s['actual_bg'] - s['physics_pred']
        residual = max(-25, min(25, residual))
        target = torch.tensor([residual / 25.0], dtype=torch.float32)  # Normalized

        return features, target


def compute_physics_baselines_and_residuals(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
    iob_model: nn.Module,
    cob_model: nn.Module,
    horizons: List[int] = [5, 15, 30, 60],
) -> List[Dict]:
    """
    Compute physics baseline predictions and residuals for training.

    For each glucose reading, we:
    1. Calculate IOB/COB at that moment from historical treatments
    2. Apply physics baseline to get prediction
    3. Compare with actual BG at horizon to get residual
    """
    from ml.models.physics_baseline import PhysicsBaseline
    from ml.models.residual_tft import get_lunar_phase

    physics = PhysicsBaseline()
    samples = []

    # Ensure glucose sorted by time
    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    # For each potential starting point
    for i in range(len(glucose_df) - max(horizons) // 5 - 1):
        row = glucose_df.iloc[i]
        current_bg = row['value']
        current_time = row['timestamp']
        hour = current_time.hour
        day_of_week = current_time.weekday()
        day_of_year = current_time.timetuple().tm_yday

        # Calculate IOB from recent insulin
        iob = calculate_iob_at_time(current_time, treatments_df)

        # Calculate COB from recent carbs
        cob = calculate_cob_at_time(current_time, treatments_df)

        # Get lunar phase
        lunar_sin, lunar_cos = get_lunar_phase(current_time)

        # Calculate recent trend (last 15 min)
        recent_trend = calculate_recent_trend(glucose_df, i)
        bg_volatility = calculate_bg_volatility(glucose_df, i)

        for horizon in horizons:
            # Find actual BG at horizon
            steps_ahead = horizon // 5
            if i + steps_ahead >= len(glucose_df):
                continue

            actual_bg = glucose_df.iloc[i + steps_ahead]['value']

            # Compute physics baseline
            pred = physics.predict(
                current_bg=current_bg,
                iob=iob,
                cob=cob,
                horizon_min=horizon,
                isf=DEFAULT_ISF,
                icr=DEFAULT_ICR,
                hour=hour,
            )

            samples.append({
                'hour': hour,
                'day_of_week': day_of_week,
                'day_of_year': day_of_year,
                'lunar_sin': lunar_sin,
                'lunar_cos': lunar_cos,
                'is_weekend': day_of_week >= 5,
                'recent_trend': recent_trend,
                'bg_volatility': bg_volatility,
                'dawn_intensity': 0.5 if 4 <= hour < 8 else 0.0,
                'horizon_min': horizon,
                'physics_pred': pred.predicted_bg,
                'actual_bg': actual_bg,
                'current_bg': current_bg,
                'iob': iob,
                'cob': cob,
            })

    logger.info(f"Created {len(samples)} residual training samples")
    return samples


def calculate_iob_at_time(time: datetime, treatments_df: pd.DataFrame) -> float:
    """Calculate IOB at a specific time using exponential decay."""
    if treatments_df.empty:
        return 0.0

    # Convert time to comparable format
    time = pd.Timestamp(time)
    if time.tz is None:
        time = time.tz_localize('UTC')

    half_life_min = 54.0  # Personalized
    insulin_duration_min = 240.0

    total_iob = 0.0

    insulin_treats = treatments_df[treatments_df['insulin'] > 0]
    for _, treat in insulin_treats.iterrows():
        treat_time = pd.Timestamp(treat['timestamp'])
        if treat_time.tz is None:
            treat_time = treat_time.tz_localize('UTC')

        minutes_since = (time - treat_time).total_seconds() / 60

        if 0 < minutes_since < insulin_duration_min:
            # Exponential decay
            remaining = 0.5 ** (minutes_since / half_life_min)
            total_iob += treat['insulin'] * remaining

    return total_iob


def calculate_cob_at_time(time: datetime, treatments_df: pd.DataFrame) -> float:
    """Calculate COB at a specific time."""
    if treatments_df.empty:
        return 0.0

    time = pd.Timestamp(time)
    if time.tz is None:
        time = time.tz_localize('UTC')

    default_half_life = 45.0
    carb_duration_min = 240.0

    total_cob = 0.0

    carb_treats = treatments_df[treatments_df['carbs'] > 0]
    for _, treat in carb_treats.iterrows():
        treat_time = pd.Timestamp(treat['timestamp'])
        if treat_time.tz is None:
            treat_time = treat_time.tz_localize('UTC')

        minutes_since = (time - treat_time).total_seconds() / 60

        if 0 < minutes_since < carb_duration_min:
            # Get GI-adjusted half-life
            gi = treat.get('glycemicIndex', 55) or 55
            fat = treat.get('fat', 0) or 0

            if gi >= 70:
                half_life = 30.0
            elif gi >= 55:
                half_life = 45.0
            else:
                half_life = 60.0

            half_life += (fat / 10.0) * 15.0  # Fat extends

            remaining = 0.5 ** (minutes_since / half_life)
            total_cob += treat['carbs'] * remaining

    return total_cob


def calculate_recent_trend(glucose_df: pd.DataFrame, idx: int, lookback: int = 3) -> float:
    """Calculate recent BG trend (mg/dL per 5 min)."""
    if idx < lookback:
        return 0.0

    recent = glucose_df.iloc[idx - lookback:idx + 1]['value']
    if len(recent) < 2:
        return 0.0

    return (recent.iloc[-1] - recent.iloc[0]) / lookback


def calculate_bg_volatility(glucose_df: pd.DataFrame, idx: int, lookback: int = 12) -> float:
    """Calculate BG volatility (std dev of last hour)."""
    start = max(0, idx - lookback)
    window = glucose_df.iloc[start:idx + 1]['value']
    return window.std() if len(window) > 1 else 0.0


def train_residual_model(
    samples: List[Dict],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.002,
) -> Tuple[nn.Module, float]:
    """Train the residual adjustment model."""
    from ml.models.residual_tft import ResidualModel

    # Split data
    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = ResidualTrainingDataset(train_samples)
    val_ds = ResidualTrainingDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ResidualModel()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            output = model(features) / 25.0  # Normalize output
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                output = model(features) / 25.0
                val_loss += criterion(output, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 25 == 0:
            logger.info(f"Residual Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        mlflow.log_metrics({
            'residual_train_loss': train_loss,
            'residual_val_loss': val_loss,
        }, step=epoch)

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


# =============================================================================
# Stage 5: Ensemble Validation
# =============================================================================

def validate_ensemble(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
    iob_model: nn.Module,
    cob_model: nn.Module,
    residual_model: nn.Module,
    horizons: List[int] = [5, 15, 30, 60, 120],
) -> Dict[int, Dict[str, float]]:
    """
    Validate the complete forcing function ensemble.

    Returns metrics by horizon: MAE, RMSE, MAPE
    """
    from ml.models.forcing_ensemble import ForcingFunctionEnsemble
    from ml.models.iob_forcing import IOBForcingService
    from ml.models.cob_forcing import COBForcingService
    from ml.models.residual_tft import ResidualService

    # Create services with trained models (mock for now, will load from file)
    ensemble = ForcingFunctionEnsemble()

    results = {h: {'errors': [], 'actuals': [], 'predictions': []} for h in horizons}

    glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    for i in range(len(glucose_df) - max(horizons) // 5 - 1):
        row = glucose_df.iloc[i]
        current_bg = row['value']
        current_time = row['timestamp']

        iob = calculate_iob_at_time(current_time, treatments_df)
        cob = calculate_cob_at_time(current_time, treatments_df)

        for horizon in horizons:
            steps_ahead = horizon // 5
            if i + steps_ahead >= len(glucose_df):
                continue

            actual_bg = glucose_df.iloc[i + steps_ahead]['value']

            # Get ensemble prediction
            pred = ensemble.predict(
                current_bg=current_bg,
                iob=iob,
                cob=cob,
                horizon_min=horizon,
                isf=DEFAULT_ISF,
                icr=DEFAULT_ICR,
                hour=current_time.hour,
            )

            error = abs(pred.final_prediction - actual_bg)
            results[horizon]['errors'].append(error)
            results[horizon]['actuals'].append(actual_bg)
            results[horizon]['predictions'].append(pred.final_prediction)

    # Calculate metrics
    metrics = {}
    for h in horizons:
        errors = np.array(results[h]['errors'])
        actuals = np.array(results[h]['actuals'])

        if len(errors) > 0:
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(errors ** 2))
            mape = np.mean(errors / (actuals + 1e-8)) * 100

            metrics[h] = {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2),
                'n_samples': len(errors),
            }

            # Log to MLflow
            mlflow.log_metrics({
                f'mae_{h}min': mae,
                f'rmse_{h}min': rmse,
                f'mape_{h}min': mape,
            })

            logger.info(f"Horizon {h} min: MAE={mae:.1f}, RMSE={rmse:.1f}, n={len(errors)}")

    return metrics


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Run the complete forcing function training pipeline."""
    logger.info("=" * 70)
    logger.info("FORCING FUNCTION MODEL TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"MLflow URI: {MLFLOW_URI}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("T1D-AI/Forcing-Ensemble")

    # Load data
    logger.info("\n" + "=" * 50)
    logger.info("Loading data...")
    moments = load_bolus_moments()
    glucose_df = load_glucose_readings()
    treatments_df = load_enriched_treatments()

    if not moments:
        logger.error("No bolus moments data available. Cannot train IOB model.")
        return

    with mlflow.start_run(run_name=f"forcing_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            'user_id': USER_ID,
            'default_isf': DEFAULT_ISF,
            'default_icr': DEFAULT_ICR,
            'n_bolus_moments': len(moments),
            'n_glucose_readings': len(glucose_df),
            'n_treatments': len(treatments_df),
        })

        # =====================================================================
        # Stage 1: Train IOB Forcing Model
        # =====================================================================
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1: Training IOB Forcing Model")
        logger.info("=" * 50)

        iob_samples = create_iob_training_samples(moments)

        if len(iob_samples) < 50:
            logger.warning(f"Only {len(iob_samples)} IOB samples - may not train well!")

        iob_model, iob_loss, half_life = train_iob_forcing_model(iob_samples)

        mlflow.log_metrics({
            'iob_final_val_loss': iob_loss,
            'learned_half_life_min': half_life,
        })

        # Save IOB model
        torch.save({
            'model_state_dict': iob_model.state_dict(),
            'half_life_min': half_life,
        }, OUTPUT_DIR / 'iob_forcing.pth')

        logger.info(f"IOB model saved. Learned half-life: {half_life:.1f} min")

        # =====================================================================
        # Stage 2: Train COB Forcing Model
        # =====================================================================
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2: Training COB Forcing Model")
        logger.info("=" * 50)

        cob_samples = create_cob_training_samples(glucose_df, treatments_df)

        if len(cob_samples) < 50:
            logger.warning(f"Only {len(cob_samples)} COB samples - using formula-based training")

        if len(cob_samples) > 0:
            cob_model, cob_loss = train_cob_forcing_model(cob_samples)

            mlflow.log_metric('cob_final_val_loss', cob_loss)

            # Save COB model
            torch.save({
                'model_state_dict': cob_model.state_dict(),
            }, OUTPUT_DIR / 'cob_forcing.pth')

            logger.info(f"COB model saved. Final loss: {cob_loss:.4f}")
        else:
            cob_model = None
            logger.info("No COB training samples, will use formula fallback")

        # =====================================================================
        # Stage 3 & 4: Physics Baseline + Residual Training
        # =====================================================================
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 3 & 4: Physics Baseline + Residual Training")
        logger.info("=" * 50)

        if not glucose_df.empty:
            residual_samples = compute_physics_baselines_and_residuals(
                glucose_df, treatments_df, iob_model, cob_model
            )

            if len(residual_samples) > 100:
                residual_model, residual_loss = train_residual_model(residual_samples)

                mlflow.log_metric('residual_final_val_loss', residual_loss)

                # Save residual model
                torch.save({
                    'model_state_dict': residual_model.state_dict(),
                }, OUTPUT_DIR / 'residual_tft.pth')

                logger.info(f"Residual model saved. Final loss: {residual_loss:.4f}")
            else:
                residual_model = None
                logger.info("Insufficient residual samples, will use heuristics")
        else:
            residual_model = None
            logger.info("No glucose data for residual training")

        # =====================================================================
        # Stage 5: Validation
        # =====================================================================
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 5: Ensemble Validation")
        logger.info("=" * 50)

        if not glucose_df.empty:
            metrics = validate_ensemble(
                glucose_df, treatments_df,
                iob_model, cob_model, residual_model
            )

            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("VALIDATION RESULTS")
            logger.info("=" * 50)

            for h, m in metrics.items():
                logger.info(f"  {h:3d} min: MAE={m['mae']:5.1f}, RMSE={m['rmse']:5.1f}, n={m['n_samples']}")

        # Log models to MLflow
        mlflow.pytorch.log_model(iob_model, "iob_forcing_model")
        if cob_model:
            mlflow.pytorch.log_model(cob_model, "cob_forcing_model")
        if residual_model:
            mlflow.pytorch.log_model(residual_model, "residual_model")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Models saved to: {OUTPUT_DIR}")
    logger.info("Next steps:")
    logger.info("  1. Update prediction_service.py to use forcing functions")
    logger.info("  2. Test predictions with live data")
    logger.info("  3. Compare with benchmarks (target: RMSE < 22 at 30 min)")


if __name__ == '__main__':
    main()
