"""
Glucose API Endpoints for T1D-AI
Provides glucose data, predictions, and metrics.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends

from models.schemas import (
    GlucoseReading, GlucoseWithPredictions, GlucoseCurrentResponse,
    GlucoseHistoryResponse, CurrentMetrics, PredictionAccuracy,
    GlucosePrediction
)
from database.repositories import GlucoseRepository, TreatmentRepository
from services.iob_cob_service import IOBCOBService
from services.prediction_service import get_prediction_service, PredictionService
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Repository instances (would use dependency injection in production)
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
iob_cob_service = IOBCOBService.from_settings()


# Temporary: hardcoded user ID until auth is implemented
TEMP_USER_ID = "demo_user"


def get_pred_service() -> PredictionService:
    """Get prediction service dependency."""
    settings = get_settings()
    models_dir = None
    for path in [Path("./models"), Path("./data/models"), Path("/app/models")]:
        if path.exists():
            models_dir = path
            break
    return get_prediction_service(models_dir, settings.model_device)


@router.get("/current", response_model=GlucoseCurrentResponse)
async def get_current_glucose(user_id: str = Query(default=TEMP_USER_ID)):
    """
    Get current glucose reading with predictions and metrics.

    Returns the latest glucose value along with:
    - ML predictions (Linear and LSTM) for 5, 10, 15 minutes ahead
    - IOB (Insulin on Board)
    - COB (Carbs on Board)
    - ISF (Insulin Sensitivity Factor)
    - Recommended correction dose
    """
    try:
        # Get latest glucose reading
        latest = await glucose_repo.get_latest(user_id)
        if not latest:
            raise HTTPException(status_code=404, detail="No glucose data found")

        # Get recent treatments for IOB/COB calculation
        treatments = await treatment_repo.get_recent(user_id, hours=6)

        # Calculate IOB and COB
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        # Get predictions from ML service
        pred_service = get_pred_service()

        # Get glucose history for LSTM predictions
        start_time = datetime.utcnow() - timedelta(minutes=120)
        glucose_history = await glucose_repo.get_history(user_id, start_time)

        # Convert trend to int for prediction
        trend_val = 0
        if latest.trend:
            trend_map = {
                "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
            }
            trend_val = trend_map.get(str(latest.trend), 0)

        # Generate predictions
        prediction_result = pred_service.predict(
            current_bg=float(latest.value),
            trend=trend_val,
            iob=iob,
            glucose_history=[r.model_dump() for r in glucose_history],
            treatments=[t.model_dump() for t in treatments]
        )

        # Get ISF from prediction service
        isf = prediction_result.isf

        # Calculate metrics with predicted ISF
        metrics = iob_cob_service.get_current_metrics(
            current_bg=latest.value,
            treatments=treatments,
            isf=isf
        )

        # Build predictions response
        predictions = GlucosePrediction(
            timestamp=prediction_result.timestamp,
            linear=prediction_result.linear,
            lstm=prediction_result.lstm or []
        )

        glucose_with_predictions = GlucoseWithPredictions(
            **latest.model_dump(),
            predictions=predictions
        )

        # Get accuracy stats
        accuracy_stats = pred_service.get_accuracy_stats()
        accuracy = PredictionAccuracy(
            linearWins=accuracy_stats.linear_count,
            lstmWins=accuracy_stats.lstm_count,
            totalComparisons=accuracy_stats.linear_count + accuracy_stats.lstm_count
        )

        return GlucoseCurrentResponse(
            glucose=glucose_with_predictions,
            metrics=metrics,
            accuracy=accuracy
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current glucose: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=GlucoseHistoryResponse)
async def get_glucose_history(
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history (1-168)"),
    limit: int = Query(default=1000, ge=1, le=5000)
):
    """
    Get historical glucose readings.

    Returns glucose readings for the specified time period.
    """
    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()

        readings = await glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return GlucoseHistoryResponse(
            readings=readings,
            totalCount=len(readings),
            startTime=start_time,
            endTime=end_time
        )

    except Exception as e:
        logger.error(f"Error getting glucose history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/range-stats")
async def get_range_stats(
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=24, ge=1, le=168)
):
    """
    Get time-in-range statistics.

    Returns percentage of time in each glucose range:
    - Critical Low (<54)
    - Low (54-70)
    - In Range (70-180)
    - High (180-250)
    - Critical High (>250)
    """
    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        readings = await glucose_repo.get_history(user_id, start_time)

        if not readings:
            return {
                "totalReadings": 0,
                "criticalLow": 0,
                "low": 0,
                "inRange": 0,
                "high": 0,
                "criticalHigh": 0,
                "averageBg": None,
                "estimatedA1c": None
            }

        # Count readings in each range
        critical_low = sum(1 for r in readings if r.value < 54)
        low = sum(1 for r in readings if 54 <= r.value < 70)
        in_range = sum(1 for r in readings if 70 <= r.value <= 180)
        high = sum(1 for r in readings if 180 < r.value <= 250)
        critical_high = sum(1 for r in readings if r.value > 250)

        total = len(readings)
        avg_bg = sum(r.value for r in readings) / total

        # Estimate A1C from average BG
        # Formula: A1C = (average_bg + 46.7) / 28.7
        estimated_a1c = (avg_bg + 46.7) / 28.7

        return {
            "totalReadings": total,
            "criticalLow": round(critical_low / total * 100, 1),
            "low": round(low / total * 100, 1),
            "inRange": round(in_range / total * 100, 1),
            "high": round(high / total * 100, 1),
            "criticalHigh": round(critical_high / total * 100, 1),
            "averageBg": round(avg_bg, 0),
            "estimatedA1c": round(estimated_a1c, 1)
        }

    except Exception as e:
        logger.error(f"Error getting range stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
