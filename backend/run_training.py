#!/usr/bin/env python3
"""
Run ML training pipelines for T1D-AI.
Trains IOB, COB, and TFT models for a user.
Uses ISF model for dynamic ISF prediction instead of hardcoded values.
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml.training.train_absorption_models import train_all_models, AbsorptionModelTrainer
from ml.inference.isf_inference import create_isf_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_predicted_isf(user_id: str, days: int = 30) -> float:
    """
    Get predicted ISF using the trained ISF model.

    Loads the ISF model and uses recent glucose data to predict ISF.
    Falls back to default if model is unavailable.

    Args:
        user_id: User ID to get ISF for
        days: Days of recent data to sample for ISF prediction

    Returns:
        Predicted ISF in mg/dL per unit
    """
    # Try to load ISF model from dexcom_reader_ML_complete
    models_dir = Path("/home/jadericdawson/Documents/AI/dexcom_reader_ML_complete")
    isf_service = create_isf_service(models_dir, device="cpu")

    if not isf_service.is_loaded:
        logger.warning("ISF model not loaded, using default ISF=55.0")
        return 55.0

    # Get recent glucose data to build feature sequence for ISF prediction
    from database.repositories import GlucoseRepository

    try:
        glucose_repo = GlucoseRepository()
        since = datetime.now(timezone.utc) - timedelta(days=days)

        readings = await glucose_repo.get_history(
            user_id=user_id,
            start_time=since,
            limit=500  # Get sample of recent readings
        )

        if len(readings) < 24:  # Need at least 2 hours of data
            logger.warning(f"Insufficient glucose data ({len(readings)} readings), using default ISF")
            return isf_service.get_default_isf()

        # Build feature sequence for ISF prediction
        # Get most recent 24 readings (2 hours at 5-min intervals)
        recent = readings[:24] if len(readings) >= 24 else readings

        # Map trend strings to numeric values (Dexcom trend arrows)
        trend_map = {
            "DoubleUp": 3, "SingleUp": 2, "FortyFiveUp": 1,
            "Flat": 0,
            "FortyFiveDown": -1, "SingleDown": -2, "DoubleDown": -3,
            "NotComputable": 0, "RateOutOfRange": 0, None: 0
        }

        # Create feature array - ISF model expects n_feat features
        # Based on isf_feature_list.pkl, build features from glucose data
        feature_sequence = []
        for r in recent:
            # Convert trend string to numeric
            trend_val = r.trend if r.trend else 0
            if isinstance(trend_val, str):
                trend_val = trend_map.get(trend_val, 0)

            # Basic features from glucose reading
            features = [
                float(r.value),  # Glucose value
                float(trend_val),  # Trend as numeric
            ]
            # Pad with zeros if needed (the model will handle scaling)
            while len(features) < 26:  # ISF_MODEL_CONFIG["n_feat"]
                features.append(0.0)
            feature_sequence.append(features)

        # Shape: (1, seq_len, n_feat)
        feature_array = np.array([feature_sequence], dtype=np.float32)

        # Predict ISF
        predicted_isf = isf_service.predict(feature_array)

        if predicted_isf is not None:
            logger.info(f"ISF Model prediction: {predicted_isf:.1f} mg/dL per unit")
            return predicted_isf
        else:
            return isf_service.get_default_isf()

    except Exception as e:
        logger.error(f"Error predicting ISF: {e}")
        return 55.0  # Default fallback


async def main():
    # Emrys's user ID (from CosmosDB)
    user_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    logger.info(f"Starting ML training for user {user_id}")

    # Get predicted ISF from ISF model instead of hardcoded value
    isf = await get_predicted_isf(user_id, days=30)

    # ICR typically correlates with ISF - common ratio is ISF/500 rule
    # Or use a known ICR if available
    icr = 10.0  # This should also come from a model or user profile

    logger.info(f"Using ISF={isf:.1f} (from ISF model), ICR={icr}")

    try:
        # Train absorption models (IOB/COB)
        trainer = AbsorptionModelTrainer(
            device="cpu",
            default_isf=isf,
            default_icr=icr,
            min_iob_samples=20,  # Lower threshold for testing
            min_cob_samples=15,  # Lower threshold for testing
            epochs=100,
            batch_size=32
        )

        results = await trainer.train_user_absorption_models(
            user_id=user_id,
            days=180,  # 6 months of data
            isf=isf,
            icr=icr
        )

        logger.info("=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)

        if results.get("iob_model_trained"):
            logger.info("IOB Model: TRAINED")
            metrics = results.get("iob_metrics", {})
            logger.info(f"  - Samples: {metrics.get('train_samples', 0)}")
            logger.info(f"  - MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"  - RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"  - Model saved: {results.get('iob_model_path', 'N/A')}")
        else:
            logger.info(f"IOB Model: NOT TRAINED - {results.get('iob_error', 'Unknown')}")

        if results.get("cob_model_trained"):
            logger.info("COB Model: TRAINED")
            metrics = results.get("cob_metrics", {})
            logger.info(f"  - Samples: {metrics.get('train_samples', 0)}")
            logger.info(f"  - MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"  - RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"  - Model saved: {results.get('cob_model_path', 'N/A')}")
        else:
            logger.info(f"COB Model: NOT TRAINED - {results.get('cob_error', 'Unknown')}")

        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"\nFinal status: {'SUCCESS' if results.get('success') else 'FAILED'}")
