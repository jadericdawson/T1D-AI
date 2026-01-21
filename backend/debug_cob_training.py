#!/usr/bin/env python3
"""Debug COB model training NaN issue."""
import asyncio
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml.training.train_absorption_models import AbsorptionModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    user_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"
    isf = 50.0
    icr = 10.0

    trainer = AbsorptionModelTrainer(
        min_iob_samples=20,
        min_cob_samples=15,
    )

    # Fetch data
    logger.info("Fetching data...")
    glucose_df, treatments_df = await trainer.fetch_data(user_id, days=180)

    logger.info(f"Glucose: {len(glucose_df)} rows")
    logger.info(f"Treatments: {len(treatments_df)} rows")

    # Find meal events
    meals = trainer.find_meal_events(treatments_df, glucose_df)
    logger.info(f"Found {len(meals)} meal events")

    if not meals:
        logger.error("No meal events found!")
        return

    # Show sample meal
    logger.info(f"Sample meal: {meals[0]}")

    # Create training data
    logger.info("Creating COB training data...")
    X_cob, y_cob = trainer.create_cob_training_data(
        meals, glucose_df, treatments_df, isf, icr
    )

    logger.info(f"X_cob shape: {X_cob.shape}")
    logger.info(f"y_cob shape: {y_cob.shape}")

    # Check for NaN/Inf in features
    logger.info("\n=== CHECKING FOR NaN/Inf ===")

    nan_in_X = np.isnan(X_cob).sum()
    inf_in_X = np.isinf(X_cob).sum()
    nan_in_y = np.isnan(y_cob).sum()
    inf_in_y = np.isinf(y_cob).sum()

    logger.info(f"NaN in X: {nan_in_X}")
    logger.info(f"Inf in X: {inf_in_X}")
    logger.info(f"NaN in y: {nan_in_y}")
    logger.info(f"Inf in y: {inf_in_y}")

    # Check column-wise for NaN
    feature_names = ['carbs', 'minutes_since', 'protein', 'fat', 'glycemic_index', 'hour_sin', 'hour_cos']

    logger.info("\n=== NaN per feature ===")
    for i, name in enumerate(feature_names):
        nan_count = np.isnan(X_cob[:, i]).sum()
        inf_count = np.isinf(X_cob[:, i]).sum()
        min_val = np.nanmin(X_cob[:, i])
        max_val = np.nanmax(X_cob[:, i])
        logger.info(f"  {name}: NaN={nan_count}, Inf={inf_count}, range=[{min_val:.2f}, {max_val:.2f}]")

    # Check y values
    logger.info("\n=== Target (y) stats ===")
    logger.info(f"  y min: {np.nanmin(y_cob):.4f}")
    logger.info(f"  y max: {np.nanmax(y_cob):.4f}")
    logger.info(f"  y mean: {np.nanmean(y_cob):.4f}")

    # Check for extreme values
    logger.info("\n=== Checking for extreme values ===")
    for i, name in enumerate(feature_names):
        col = X_cob[:, i]
        outliers = np.abs(col) > 1e6
        if outliers.sum() > 0:
            logger.warning(f"  {name}: {outliers.sum()} extreme values!")


if __name__ == "__main__":
    asyncio.run(main())
