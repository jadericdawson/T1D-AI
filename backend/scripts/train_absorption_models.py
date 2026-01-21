#!/usr/bin/env python3
"""
Train Absorption Curve Models from Actual BG Response Data

This script learns personalized insulin and carb absorption curves by:
1. Finding clean correction boluses (insulin without carbs)
2. Finding meals with clear BG responses
3. Analyzing BG changes to determine actual absorption timing
4. Fitting models to the observed data

Output:
- insulin_absorption.pth - Learned insulin absorption curve
- carb_absorption.pth - Learned carb absorption curve
- Absorption parameters (onset, ramp, half-life) derived from YOUR data
"""
import asyncio
import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import torch
import numpy as np

# Load .env file first
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.cosmos_client import CosmosDBManager
from ml.models.absorption_learner import (
    InsulinAbsorptionLearner,
    CarbAbsorptionLearner,
    AbsorptionCurveModel,
    AbsorptionCurveParams,
    extract_absorption_training_data,
    fit_absorption_curve,
)
from ml.mlflow_tracking import ModelTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
USER_ID = "b93870c7-4528-4924-add7-5765fec3a228"  # Emrys
DAYS_OF_DATA = 90
OUTPUT_DIR = Path(__file__).parent.parent / "models"


def fetch_training_data_local(days: int = 90):
    """Fetch glucose and treatment data from local JSONL files for training."""
    logger.info(f"Loading training data from local files")

    DATA_DIR = Path(__file__).parent.parent.parent / "data"

    # Load glucose readings from gluroo_readings.jsonl
    glucose_file = DATA_DIR / "gluroo_readings.jsonl"
    glucose_readings = []
    if glucose_file.exists():
        with open(glucose_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    glucose_readings.append({
                        'timestamp': entry['timestamp'],
                        'value': entry['value'],
                        'trend': entry.get('trend', 0),
                    })
    logger.info(f"Loaded {len(glucose_readings)} glucose readings from {glucose_file}")

    # Load treatments from gluroo_readings.jsonl (has carbs/insulin)
    treatments = []
    seen_ts = set()
    if glucose_file.exists():
        with open(glucose_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    ts = entry.get('timestamp')
                    carbs = entry.get('carbs', 0) or 0
                    insulin = entry.get('insulin', 0) or 0

                    if (carbs > 0 or insulin > 0) and ts not in seen_ts:
                        seen_ts.add(ts)
                        treatments.append({
                            'timestamp': ts,
                            'carbs': carbs,
                            'insulin': insulin,
                            'glycemicIndex': 55,  # Default, estimate from carbs
                        })

    # Also load from bolus_moments.jsonl for correction boluses
    bolus_file = DATA_DIR / "bolus_moments.jsonl"
    if bolus_file.exists():
        with open(bolus_file, 'r') as f:
            for line in f:
                if line.strip():
                    moment = json.loads(line)
                    ts = moment.get('bolus_ts')
                    insulin = moment.get('bolus_insulin', 0) or 0
                    if ts and ts not in seen_ts and insulin > 0:
                        seen_ts.add(ts)
                        treatments.append({
                            'timestamp': ts,
                            'carbs': 0,
                            'insulin': insulin,
                            'glycemicIndex': 55,
                        })

    logger.info(f"Loaded {len(treatments)} treatments")

    # Default ISF/ICR
    isf = 55
    icr = 10

    return glucose_readings, treatments, isf, icr


def train_absorption_model(
    samples: list,
    model_type: str,
    output_path: Path,
    tracker: ModelTracker,
) -> AbsorptionCurveParams:
    """
    Train absorption curve model from samples.

    Args:
        samples: Training data
        model_type: 'insulin' or 'carb'
        output_path: Where to save model
        tracker: MLflow tracker

    Returns:
        Learned curve parameters
    """
    logger.info(f"Training {model_type} absorption model with {len(samples)} samples")

    # Fit curve parameters from data
    params = fit_absorption_curve(samples, model_type)
    logger.info(f"Fitted {model_type} parameters: {params.to_dict()}")

    # Train neural network if enough data
    if len(samples) >= 100:
        model = AbsorptionCurveModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Prepare training data
        X = []
        y = []
        for s in samples:
            if model_type == 'insulin':
                features = [
                    s['minutes'] / 360.0,
                    s['dose'] / 10.0,
                    np.sin(2 * np.pi * s['hour'] / 24),
                    np.cos(2 * np.pi * s['hour'] / 24),
                    s['bg_at_dose'] / 200.0,
                    1.0 if s['minutes'] < 30 else 0.0,
                    1.0 if 30 <= s['minutes'] < 90 else 0.0,
                    1.0 if s['minutes'] >= 90 else 0.0,
                    min(s['minutes'] / 30.0, 1.0),
                    max(0, (s['minutes'] - 30) / 60.0),
                    s['dose'] * s['minutes'] / 3600.0,
                    (s['bg_at_dose'] - 120) / 100.0,
                ]
            else:
                gi = s.get('glycemic_index', 55) / 55.0
                features = [
                    s['minutes'] / 180.0,
                    s['carbs'] / 100.0,
                    gi,
                    np.sin(2 * np.pi * s['hour'] / 24),
                    np.cos(2 * np.pi * s['hour'] / 24),
                    1.0 if s['minutes'] < 20 else 0.0,
                    1.0 if 20 <= s['minutes'] < 60 else 0.0,
                    1.0 if s['minutes'] >= 60 else 0.0,
                    min(s['minutes'] / 20.0, 1.0),
                    max(0, (s['minutes'] - 20) / 40.0),
                    s['carbs'] * gi / 100.0,
                    gi - 1.0,
                ]
            X.append(features)
            y.append([s['absorbed_fraction']])

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Training loop
        model.train()
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(200):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
                tracker.log_metrics({f'{model_type}_loss': loss.item()}, step=epoch)

        # Save model
        model.eval()
        torch.save({
            'model_state_dict': model.state_dict(),
            'learned_params': params.to_dict(),
            'n_samples': len(samples),
            'trained_at': datetime.utcnow().isoformat(),
        }, output_path)

        logger.info(f"Saved {model_type} model to {output_path}")
    else:
        # Save just the parameters
        torch.save({
            'learned_params': params.to_dict(),
            'n_samples': len(samples),
            'trained_at': datetime.utcnow().isoformat(),
        }, output_path)

    # Log parameters to MLflow
    tracker.log_params({
        f'{model_type}_onset_min': params.onset_min,
        f'{model_type}_ramp_duration': params.ramp_duration,
        f'{model_type}_half_life_min': params.half_life_min,
        f'{model_type}_n_samples': len(samples),
    })

    return params


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("ABSORPTION CURVE LEARNING")
    logger.info("Learning personalized IOB/COB absorption from BG response data")
    logger.info("=" * 60)

    # Initialize MLflow
    tracker = ModelTracker(model_type="forcing_ensemble", user_id=USER_ID)
    tracker.start_run(run_name=f"absorption_learning_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    try:
        # Fetch data from local JSONL files (for local training)
        glucose, treatments, isf, icr = fetch_training_data_local(DAYS_OF_DATA)

        if len(glucose) < 1000 or len(treatments) < 50:
            logger.error("Insufficient data for training")
            return

        # Extract training samples
        insulin_samples, carb_samples = extract_absorption_training_data(
            glucose, treatments, isf, icr
        )

        logger.info(f"Extracted {len(insulin_samples)} insulin absorption samples")
        logger.info(f"Extracted {len(carb_samples)} carb absorption samples")

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Train insulin absorption model
        if len(insulin_samples) >= 20:
            insulin_params = train_absorption_model(
                insulin_samples,
                'insulin',
                OUTPUT_DIR / 'insulin_absorption.pth',
                tracker,
            )
            logger.info(f"Learned insulin onset: {insulin_params.onset_min:.1f} min")
            logger.info(f"Learned insulin half-life: {insulin_params.half_life_min:.1f} min")
        else:
            logger.warning("Not enough clean correction boluses to learn insulin absorption")

        # Train carb absorption model
        if len(carb_samples) >= 20:
            carb_params = train_absorption_model(
                carb_samples,
                'carb',
                OUTPUT_DIR / 'carb_absorption.pth',
                tracker,
            )
            logger.info(f"Learned carb onset: {carb_params.onset_min:.1f} min")
            logger.info(f"Learned carb half-life: {carb_params.half_life_min:.1f} min")
        else:
            logger.warning("Not enough meal data to learn carb absorption")

        # Summary
        logger.info("=" * 60)
        logger.info("ABSORPTION LEARNING COMPLETE")
        logger.info(f"Models saved to: {OUTPUT_DIR}")
        logger.info("=" * 60)

        tracker.end_run()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
