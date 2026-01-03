"""
Predictions API Endpoints
Blood glucose and ISF predictions with accuracy tracking.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pathlib import Path

from ...config import get_settings
from ...services.prediction_service import (
    PredictionService,
    PredictionResult,
    AccuracyStats,
    get_prediction_service,
)
from ...database.repositories import GlucoseRepository, TreatmentRepository
from ...services.iob_cob_service import IOBCOBService

router = APIRouter(prefix="/predictions", tags=["predictions"])


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request for glucose prediction."""
    current_bg: float = Field(..., ge=40, le=500, description="Current BG in mg/dL")
    trend: int = Field(0, ge=-3, le=3, description="CGM trend arrow")
    include_history: bool = Field(True, description="Include recent glucose history")
    history_minutes: int = Field(120, ge=30, le=360, description="Minutes of history")


class PredictionResponse(BaseModel):
    """Response with glucose predictions."""
    linear: List[float] = Field(..., description="Linear predictions [+5, +10, +15]")
    lstm: Optional[List[float]] = Field(None, description="LSTM predictions [+5, +10, +15]")
    horizons_min: List[int] = Field([5, 10, 15], description="Prediction horizons")
    timestamp: datetime
    current_bg: float
    trend: int
    trend_arrow: str
    isf: float = Field(..., description="ISF in mg/dL per unit")
    method: str = Field(..., description="Method used: linear, lstm, or hybrid")
    model_available: bool


class DoseRecommendationRequest(BaseModel):
    """Request for dose recommendation."""
    current_bg: float = Field(..., ge=40, le=500)
    target_bg: float = Field(100, ge=70, le=150)
    isf: Optional[float] = Field(None, ge=10, le=150, description="Override ISF")


class DoseRecommendationResponse(BaseModel):
    """Response with dose recommendation."""
    current_bg: float
    target_bg: float
    effective_bg: float
    iob: float
    cob: float
    isf: float
    iob_effect_mgdl: float
    cob_effect_mgdl: float
    raw_correction_units: float
    recommended_dose_units: float
    formula: str
    timestamp: datetime


class AccuracyResponse(BaseModel):
    """Response with prediction accuracy stats."""
    linear_mae: float
    lstm_mae: Optional[float]
    linear_count: int
    lstm_count: int
    winner: str
    lstm_available: bool


# Trend arrow mapping
TREND_ARROWS = {
    -3: "⇊",   # DoubleDown
    -2: "↓",   # SingleDown
    -1: "↘",   # FortyFiveDown
    0: "→",    # Flat
    1: "↗",    # FortyFiveUp
    2: "↑",    # SingleUp
    3: "⇈",    # DoubleUp
}


def get_trend_arrow(trend: int) -> str:
    """Convert trend int to arrow symbol."""
    return TREND_ARROWS.get(trend, "→")


# Dependencies
async def get_pred_service() -> PredictionService:
    """Get the prediction service dependency."""
    settings = get_settings()

    # Models directory - check multiple locations
    models_dir = None
    possible_paths = [
        Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else None,
        Path("/app/models"),
        Path("./models"),
        Path("./data/models"),
    ]

    for path in possible_paths:
        if path and path.exists():
            models_dir = path
            break

    return get_prediction_service(models_dir)


# Endpoints
@router.post("/bg", response_model=PredictionResponse)
async def predict_glucose(
    request: PredictionRequest,
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Get blood glucose predictions for +5, +10, +15 minutes.

    Uses LSTM model if available, otherwise falls back to linear extrapolation.
    """
    # Get recent glucose history
    glucose_history = None
    treatments = None

    if request.include_history:
        try:
            glucose_repo = GlucoseRepository()
            treatment_repo = TreatmentRepository()
            iob_cob_service = IOBCOBService()

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=request.history_minutes)

            glucose_readings = await glucose_repo.get_history(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time
            )
            glucose_history = [r.model_dump() for r in glucose_readings]

            treatment_records = await treatment_repo.get_by_user(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time
            )
            treatments = [t.model_dump() for t in treatment_records]

            # Get current IOB
            iob = iob_cob_service.calculate_iob(treatment_records)
        except Exception as e:
            # Log but continue without history
            iob = 0.0
    else:
        iob = 0.0

    # Generate prediction
    result = pred_service.predict(
        current_bg=request.current_bg,
        trend=request.trend,
        iob=iob,
        glucose_history=glucose_history,
        treatments=treatments
    )

    return PredictionResponse(
        linear=result.linear,
        lstm=result.lstm,
        horizons_min=result.horizons_min,
        timestamp=result.timestamp,
        current_bg=result.current_bg,
        trend=result.trend,
        trend_arrow=get_trend_arrow(result.trend),
        isf=result.isf,
        method=result.method,
        model_available=pred_service.lstm_available
    )


@router.get("/bg/current", response_model=PredictionResponse)
async def get_current_prediction(
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Get prediction based on current glucose reading.

    Fetches the latest glucose reading and generates predictions.
    """
    try:
        glucose_repo = GlucoseRepository()
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get latest reading
        current = await glucose_repo.get_current(user_id)
        if not current:
            raise HTTPException(status_code=404, detail="No current glucose reading")

        # Get history for LSTM
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=120)

        glucose_readings = await glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        glucose_history = [r.model_dump() for r in glucose_readings]

        treatments = await treatment_repo.get_by_user(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate IOB
        iob = iob_cob_service.calculate_iob(treatments)

        # Generate prediction
        result = pred_service.predict(
            current_bg=current.value,
            trend=current.trend or 0,
            iob=iob,
            glucose_history=glucose_history,
            treatments=[t.model_dump() for t in treatments]
        )

        return PredictionResponse(
            linear=result.linear,
            lstm=result.lstm,
            horizons_min=result.horizons_min,
            timestamp=result.timestamp,
            current_bg=result.current_bg,
            trend=result.trend,
            trend_arrow=get_trend_arrow(result.trend),
            isf=result.isf,
            method=result.method,
            model_available=pred_service.lstm_available
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )


@router.post("/dose", response_model=DoseRecommendationResponse)
async def calculate_dose(
    request: DoseRecommendationRequest,
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Calculate recommended correction dose.

    Accounts for current IOB (insulin on board) and COB (carbs on board).
    Uses the formula: (effective_bg - target_bg) / ISF

    Where effective_bg = current_bg + (COB * 4.0) - (IOB * ISF)
    """
    try:
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get recent treatments
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=180)

        treatments = await treatment_repo.get_by_user(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate IOB and COB
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        # Get ISF (from request or prediction service)
        isf = request.isf or pred_service._get_isf(iob)

        # Calculate dose
        result = pred_service.calculate_dose_correction(
            current_bg=request.current_bg,
            target_bg=request.target_bg,
            isf=isf,
            iob=iob,
            cob=cob
        )

        return DoseRecommendationResponse(
            **result,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate dose: {str(e)}"
        )


@router.get("/accuracy", response_model=AccuracyResponse)
async def get_accuracy(
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Get prediction accuracy statistics.

    Returns MAE (Mean Absolute Error) for both linear and LSTM predictions,
    along with which method is currently more accurate.
    """
    stats = pred_service.get_accuracy_stats()

    return AccuracyResponse(
        linear_mae=stats.linear_mae,
        lstm_mae=stats.lstm_mae,
        linear_count=stats.linear_count,
        lstm_count=stats.lstm_count,
        winner=stats.winner,
        lstm_available=pred_service.lstm_available
    )


@router.get("/isf")
async def get_isf(
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Get current Insulin Sensitivity Factor (ISF) prediction.

    ISF represents how much 1 unit of insulin will lower blood glucose.
    """
    try:
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get recent treatments to calculate current IOB
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=180)

        treatments = await treatment_repo.get_by_user(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )

        iob = iob_cob_service.calculate_iob(treatments)
        isf = pred_service._get_isf(iob)

        return {
            "isf": round(isf, 1),
            "unit": "mg/dL per unit",
            "current_iob": round(iob, 2),
            "model_available": pred_service.isf_available,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ISF: {str(e)}"
        )


@router.get("/status")
async def get_prediction_status(
    pred_service: PredictionService = Depends(get_pred_service),
):
    """
    Get prediction service status.

    Returns whether LSTM and ISF models are loaded and available.
    """
    return {
        "initialized": pred_service._initialized,
        "lstm_available": pred_service.lstm_available,
        "isf_available": pred_service.isf_available,
        "models_dir": str(pred_service.models_dir) if pred_service.models_dir else None,
        "device": str(pred_service.device)
    }
