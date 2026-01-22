"""
Predictions API Endpoints
Blood glucose and ISF predictions with accuracy tracking.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pathlib import Path

from config import get_settings
from services.prediction_service import (
    PredictionService,
    PredictionResult,
    AccuracyStats,
    TFTPredictionItem,
    get_prediction_service,
)
from database.repositories import GlucoseRepository, TreatmentRepository, SharingRepository, UserRepository
from services.iob_cob_service import IOBCOBService
from models.schemas import User
from auth import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])


def get_data_user_id(profile_id: str) -> str:
    """
    Convert a profile ID to the actual data user ID.

    For 'self' profiles, the profile ID is 'profile_{user_id}' but data
    is stored with just the raw user_id. This function strips the prefix.
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix (8 chars)
    return profile_id


# Repository instances for access validation
sharing_repo = SharingRepository()
user_repo = UserRepository()


async def validate_user_access(requester_id: str, target_user_id: str) -> bool:
    """
    Check if requester has access to target user's data.
    Access is granted if:
    - requester_id == target_user_id (viewing own data)
    - requester owns the target profile (managed profile)
    - requester is a parent with target_user_id in their linkedChildIds
    - target_user has shared their data with requester via the sharing system

    Returns False on any error to fail safely.
    """
    try:
        # Normalize both IDs for comparison
        # Profile IDs may have 'profile_' prefix but user IDs don't
        normalized_requester = get_data_user_id(requester_id)
        normalized_target = get_data_user_id(target_user_id)

        if normalized_requester == normalized_target:
            return True

        # Also check raw IDs in case one has prefix and one doesn't
        if requester_id == target_user_id:
            return True

        # Check if requester OWNS this profile (managed profile system)
        try:
            from database.repositories import ProfileRepository
            profile_repo = ProfileRepository()
            profile = await profile_repo.get_by_id(target_user_id, normalized_requester)
            if profile:
                logger.info(f"Access granted via profile ownership: {normalized_requester} owns profile {target_user_id}")
                return True
        except Exception as e:
            logger.warning(f"Error checking profile ownership: {e}")

        # Check if requester is a parent of the target
        try:
            requester = await user_repo.get_by_id(requester_id)
            if requester and requester.linkedChildIds:
                # Check both normalized and raw target IDs
                if target_user_id in requester.linkedChildIds or normalized_target in requester.linkedChildIds:
                    return True
        except Exception as e:
            logger.warning(f"Error checking parent-child access: {e}")

        # Check if target user has shared data with requester
        # This handles two cases:
        # 1. Direct share: ownerId = target_user_id (user shares their own data)
        # 2. Profile share: profileId = target_user_id (parent shares child's data)
        try:
            # Try with original target_user_id first (for profile shares)
            share = await sharing_repo.get_share_for_profile(target_user_id, requester_id)
            if share and share.isActive:
                role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                logger.info(f"Access granted via share: {target_user_id} shared with {requester_id} (role: {role_str})")
                return True
            # Also try with normalized target (for direct user shares)
            if normalized_target != target_user_id:
                share = await sharing_repo.get_share_for_profile(normalized_target, requester_id)
                if share and share.isActive:
                    role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                    logger.info(f"Access granted via share: {normalized_target} shared with {requester_id} (role: {role_str})")
                    return True
        except Exception as e:
            logger.warning(f"Error checking share access: {e}")

        return False
    except Exception as e:
        logger.error(f"Unexpected error in validate_user_access: {e}")
        return False


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request for glucose prediction."""
    current_bg: float = Field(..., ge=40, le=500, description="Current BG in mg/dL")
    trend: int = Field(0, ge=-3, le=3, description="CGM trend arrow")
    include_history: bool = Field(True, description="Include recent glucose history")
    history_minutes: int = Field(120, ge=30, le=360, description="Minutes of history")


class TFTPredictionResponse(BaseModel):
    """TFT prediction with uncertainty bounds."""
    horizon_min: int = Field(..., description="Prediction horizon in minutes")
    value: float = Field(..., description="Median prediction (50th percentile)")
    lower: float = Field(..., description="Lower bound (10th percentile)")
    upper: float = Field(..., description="Upper bound (90th percentile)")
    confidence: float = Field(0.8, description="Confidence level")


class PredictionResponse(BaseModel):
    """Response with glucose predictions."""
    linear: List[float] = Field(..., description="Linear predictions [+5, +10, +15]")
    lstm: Optional[List[float]] = Field(None, description="LSTM predictions [+5, +10, +15]")
    tft: Optional[List[TFTPredictionResponse]] = Field(None, description="TFT predictions [+30, +45, +60]")
    horizons_min: List[int] = Field([5, 10, 15], description="Prediction horizons")
    timestamp: datetime
    current_bg: float
    trend: int
    trend_arrow: str
    isf: float = Field(..., description="ISF in mg/dL per unit")
    method: str = Field(..., description="Method used: linear, lstm, tft, or hybrid")
    model_available: bool
    tft_available: bool = Field(False, description="Whether TFT model is loaded")


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
    current_user: User = Depends(get_current_user),
):
    """
    Get blood glucose predictions for +5, +10, +15 minutes.

    Uses LSTM model if available, otherwise falls back to linear extrapolation.
    Requires JWT authentication. Users can view their own data or shared data.
    """
    # SECURITY: Validate access to the requested user's data
    try:
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user access: {e}")
        raise HTTPException(status_code=500, detail="Error validating access")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(user_id)

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
                user_id=data_user_id,
                start_time=start_time,
                end_time=end_time
            )
            glucose_history = [r.model_dump() for r in glucose_readings]

            treatment_records = await treatment_repo.get_by_user(
                user_id=data_user_id,
                start_time=start_time,
                end_time=end_time
            )
            treatments = [t.model_dump() for t in treatment_records]

            # Get current IOB and COB
            iob = iob_cob_service.calculate_iob(treatment_records)
            cob = iob_cob_service.calculate_cob(treatment_records)
        except Exception as e:
            # Log but continue without history
            iob = 0.0
            cob = 0.0
    else:
        iob = 0.0
        cob = 0.0

    # Generate prediction
    result = pred_service.predict(
        current_bg=request.current_bg,
        trend=request.trend,
        iob=iob,
        cob=cob,
        glucose_history=glucose_history,
        treatments=treatments
    )

    # Log predictions for accuracy tracking
    try:
        from services.prediction_tracker import get_prediction_tracker
        tracker = get_prediction_tracker()

        # Log linear predictions
        for i, horizon in enumerate([5, 10, 15]):
            if i < len(result.linear):
                tracker.log_prediction(
                    user_id=user_id,
                    prediction_time=result.timestamp,
                    model_type="linear",
                    horizon_min=horizon,
                    predicted_value=result.linear[i],
                )

        # Log TFT predictions with bounds
        if result.tft:
            for tft_pred in result.tft:
                tracker.log_prediction(
                    user_id=user_id,
                    prediction_time=result.timestamp,
                    model_type="tft",
                    horizon_min=tft_pred.horizon_min,
                    predicted_value=tft_pred.value,
                    lower_bound=tft_pred.lower,
                    upper_bound=tft_pred.upper,
                )
    except Exception as e:
        # Don't fail the prediction if tracking fails
        import logging
        logging.getLogger(__name__).warning(f"Prediction tracking failed: {e}")

    # Convert TFT predictions to response format
    tft_response = None
    if result.tft:
        tft_response = [
            TFTPredictionResponse(
                horizon_min=p.horizon_min,
                value=p.value,
                lower=p.lower,
                upper=p.upper,
                confidence=p.confidence
            )
            for p in result.tft
        ]

    return PredictionResponse(
        linear=result.linear,
        lstm=result.lstm,
        tft=tft_response,
        horizons_min=result.horizons_min,
        timestamp=result.timestamp,
        current_bg=result.current_bg,
        trend=result.trend,
        trend_arrow=get_trend_arrow(result.trend),
        isf=result.isf,
        method=result.method,
        model_available=pred_service.lstm_available,
        tft_available=pred_service.tft_available
    )


@router.get("/bg/current", response_model=PredictionResponse)
async def get_current_prediction(
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
    current_user: User = Depends(get_current_user),
):
    """
    Get prediction based on current glucose reading.

    Fetches the latest glucose reading and generates predictions.
    Requires JWT authentication. Users can view their own data or shared data.
    """
    # SECURITY: Validate access to the requested user's data
    try:
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user access: {e}")
        raise HTTPException(status_code=500, detail="Error validating access")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(user_id)

    try:
        glucose_repo = GlucoseRepository()
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get latest reading
        current = await glucose_repo.get_current(data_user_id)
        if not current:
            raise HTTPException(status_code=404, detail="No current glucose reading")

        # Get history for LSTM
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=120)

        glucose_readings = await glucose_repo.get_history(
            user_id=data_user_id,
            start_time=start_time,
            end_time=end_time
        )
        glucose_history = [r.model_dump() for r in glucose_readings]

        treatments = await treatment_repo.get_by_user(
            user_id=data_user_id,
            start_time=start_time,
            end_time=end_time
        )

        # Calculate IOB and COB
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        # Generate prediction
        result = pred_service.predict(
            current_bg=current.value,
            trend=current.trend or 0,
            iob=iob,
            cob=cob,
            glucose_history=glucose_history,
            treatments=[t.model_dump() for t in treatments]
        )

        # Convert TFT predictions to response format
        tft_response = None
        if result.tft:
            tft_response = [
                TFTPredictionResponse(
                    horizon_min=p.horizon_min,
                    value=p.value,
                    lower=p.lower,
                    upper=p.upper,
                    confidence=p.confidence
                )
                for p in result.tft
            ]

        return PredictionResponse(
            linear=result.linear,
            lstm=result.lstm,
            tft=tft_response,
            horizons_min=result.horizons_min,
            timestamp=result.timestamp,
            current_bg=result.current_bg,
            trend=result.trend,
            trend_arrow=get_trend_arrow(result.trend),
            isf=result.isf,
            method=result.method,
            model_available=pred_service.lstm_available,
            tft_available=pred_service.tft_available
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
    current_user: User = Depends(get_current_user),
):
    """
    Calculate recommended correction dose.

    Accounts for current IOB (insulin on board) and COB (carbs on board).
    Uses the formula: (effective_bg - target_bg) / ISF

    Where effective_bg = current_bg + (COB * 4.0) - (IOB * ISF)
    Requires JWT authentication. Users can view their own data or shared data.
    """
    # SECURITY: Validate access to the requested user's data
    try:
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user access: {e}")
        raise HTTPException(status_code=500, detail="Error validating access")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(user_id)

    try:
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get recent treatments
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=180)

        treatments = await treatment_repo.get_by_user(
            user_id=data_user_id,
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


@router.get("/accuracy/detailed")
async def get_detailed_accuracy():
    """
    Get detailed prediction accuracy by model type and horizon.

    Returns MAE, RMSE, MAPE, and 80% prediction interval coverage
    for each model type (linear, lstm, tft) and prediction horizon.
    """
    from services.prediction_tracker import get_prediction_tracker

    tracker = get_prediction_tracker()

    # Get overall stats by model type
    model_comparison = tracker.get_model_comparison()

    # Get stats by horizon for TFT
    tft_by_horizon = {}
    for horizon in [5, 10, 15, 30, 45, 60, 90, 120]:
        stats = tracker.get_accuracy_stats(model_type="tft", horizon_min=horizon)
        if stats["count"] > 0:
            tft_by_horizon[f"+{horizon}min"] = stats

    # Get stats by horizon for linear
    linear_by_horizon = {}
    for horizon in [5, 10, 15]:
        stats = tracker.get_accuracy_stats(model_type="linear", horizon_min=horizon)
        if stats["count"] > 0:
            linear_by_horizon[f"+{horizon}min"] = stats

    return {
        "model_comparison": model_comparison,
        "tft_by_horizon": tft_by_horizon,
        "linear_by_horizon": linear_by_horizon,
        "pending_predictions": len(tracker._pending),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/isf")
async def get_isf(
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
    current_user: User = Depends(get_current_user),
):
    """
    Get current Insulin Sensitivity Factor (ISF) prediction.

    ISF represents how much 1 unit of insulin will lower blood glucose.
    Requires JWT authentication. Users can view their own data or shared data.
    """
    # SECURITY: Validate access to the requested user's data
    try:
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user access: {e}")
        raise HTTPException(status_code=500, detail="Error validating access")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(user_id)

    try:
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get recent treatments to calculate current IOB
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=180)

        treatments = await treatment_repo.get_by_user(
            user_id=data_user_id,
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

    Returns whether LSTM, TFT, and ISF models are loaded and available.
    """
    return {
        "initialized": pred_service._initialized,
        "lstm_available": pred_service.lstm_available,
        "tft_available": pred_service.tft_available,
        "isf_available": pred_service.isf_available,
        "models_dir": str(pred_service.models_dir) if pred_service.models_dir else None,
        "device": str(pred_service.device)
    }


@router.get("/dual")
async def get_dual_predictions(
    user_id: str = Query(..., description="User ID"),
    pred_service: PredictionService = Depends(get_pred_service),
    current_user: User = Depends(get_current_user),
):
    """
    Get BOTH model-based and hardcoded BG predictions for comparison.

    Returns two prediction lines:
    1. Model-based: Uses LEARNED absorption curves (from this person's BG data)
       - IOB: onset=15min, ramp=75min, half-life=120min
       - COB: onset=5min, ramp=10min, half-life=45min
    2. Hardcoded: Uses standard textbook parameters (adult average)
       - IOB: onset=20min, half-life=81min
       - COB: onset=15min, half-life=45min

    Over time you can compare which prediction is more accurate!
    Requires JWT authentication. Users can view their own data or shared data.
    """
    # SECURITY: Validate access to the requested user's data
    try:
        has_access = await validate_user_access(current_user.id, user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating user access: {e}")
        raise HTTPException(status_code=500, detail="Error validating access")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(user_id)

    try:
        glucose_repo = GlucoseRepository()
        treatment_repo = TreatmentRepository()
        iob_cob_service = IOBCOBService()

        # Get latest reading
        current = await glucose_repo.get_current(data_user_id)
        if not current:
            raise HTTPException(status_code=404, detail="No current glucose reading")

        # Get recent treatments
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=180)

        treatments = await treatment_repo.get_by_user(
            user_id=data_user_id,
            start_time=start_time,
            end_time=end_time
        )
        treatment_dicts = [t.model_dump() for t in treatments]

        # Get ISF and ICR
        isf = pred_service._get_isf(iob_cob_service.calculate_iob(treatments))
        icr = 10.0  # Default ICR

        # Calculate current BG trend from recent readings
        glucose_readings = await glucose_repo.get_history(
            user_id=data_user_id,
            start_time=end_time - timedelta(minutes=30),
            end_time=end_time
        )
        if len(glucose_readings) >= 2:
            sorted_readings = sorted(glucose_readings, key=lambda r: r.timestamp, reverse=True)
            bg_trend = (sorted_readings[0].value - sorted_readings[-1].value) / max(1, len(sorted_readings) - 1)
        else:
            bg_trend = 0.0

        # Get dual predictions
        result = pred_service.get_dual_bg_predictions(
            current_bg=current.value,
            treatments=treatment_dicts,
            isf=isf,
            icr=icr,
            bg_trend=bg_trend,
            duration_min=120,
            step_min=5
        )

        return {
            **result,
            'current_bg': current.value,
            'current_trend': bg_trend,
            'isf': isf,
            'icr': icr,
            'timestamp': datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dual predictions: {str(e)}"
        )
