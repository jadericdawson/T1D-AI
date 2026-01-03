"""
Calculations API Endpoints
IOB, COB, and dose calculation endpoints.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.repositories import TreatmentRepository
from services.iob_cob_service import IOBCOBService
from services.prediction_service import get_prediction_service
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calculations", tags=["calculations"])

# Service instances
treatment_repo = TreatmentRepository()
iob_cob_service = IOBCOBService.from_settings()

TEMP_USER_ID = "demo_user"


# Request/Response Models
class IOBResponse(BaseModel):
    """Insulin on Board response."""
    iob: float = Field(..., description="Current IOB in units")
    peak_iob: float = Field(0.0, description="Peak IOB in last hour")
    total_insulin_24h: float = Field(0.0, description="Total insulin in last 24h")
    active_insulin_duration_min: int = Field(180, description="Insulin action duration")
    half_life_min: int = Field(81, description="Insulin half-life")
    timestamp: datetime


class COBResponse(BaseModel):
    """Carbs on Board response."""
    cob: float = Field(..., description="Current COB in grams")
    total_carbs_24h: float = Field(0.0, description="Total carbs in last 24h")
    absorption_duration_min: int = Field(180, description="Carb absorption duration")
    half_life_min: int = Field(45, description="Carb absorption half-life")
    bg_impact: float = Field(0.0, description="Expected BG rise from COB")
    timestamp: datetime


class DoseCalculationRequest(BaseModel):
    """Request for dose calculation."""
    current_bg: float = Field(..., ge=40, le=500, description="Current BG in mg/dL")
    target_bg: float = Field(100, ge=70, le=150, description="Target BG")
    isf_override: Optional[float] = Field(None, ge=10, le=200, description="Override ISF")
    include_cob: bool = Field(True, description="Include COB in calculation")


class DoseCalculationResponse(BaseModel):
    """Response with dose calculation."""
    current_bg: float
    target_bg: float
    effective_bg: float = Field(..., description="BG adjusted for IOB/COB")
    iob: float
    cob: float
    isf: float
    iob_effect_mgdl: float = Field(..., description="How much IOB will lower BG")
    cob_effect_mgdl: float = Field(..., description="How much COB will raise BG")
    raw_correction_units: float = Field(..., description="Raw correction (can be negative)")
    recommended_dose_units: float = Field(..., description="Recommended dose (floored at 0)")
    formula: str
    warning: Optional[str] = None
    timestamp: datetime


class ActiveInsulinDetail(BaseModel):
    """Detail of an active insulin dose."""
    timestamp: datetime
    original_dose: float
    remaining: float
    minutes_ago: int
    percent_remaining: float


class ActiveInsulinResponse(BaseModel):
    """Response with active insulin breakdown."""
    total_iob: float
    doses: List[ActiveInsulinDetail]
    timestamp: datetime


# Endpoints
@router.get("/iob", response_model=IOBResponse)
async def get_iob(
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=6, ge=1, le=24, description="Hours of treatment history")
):
    """
    Get current Insulin on Board (IOB).

    Calculates remaining active insulin using exponential decay model.
    Uses Novolog insulin curve with 81-minute half-life.
    """
    try:
        # Get recent treatments
        treatments = await treatment_repo.get_recent(user_id, hours=hours)

        # Calculate IOB
        iob = iob_cob_service.calculate_iob(treatments)

        # Calculate 24h total insulin
        treatments_24h = await treatment_repo.get_recent(user_id, hours=24)
        total_insulin = sum(t.insulin or 0 for t in treatments_24h)

        # Calculate peak IOB in last hour
        # This would require more detailed tracking
        peak_iob = iob  # Placeholder

        return IOBResponse(
            iob=round(iob, 2),
            peak_iob=round(peak_iob, 2),
            total_insulin_24h=round(total_insulin, 1),
            active_insulin_duration_min=iob_cob_service.insulin_duration_min,
            half_life_min=81,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error calculating IOB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cob", response_model=COBResponse)
async def get_cob(
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=6, ge=1, le=24)
):
    """
    Get current Carbs on Board (COB).

    Calculates remaining carbs using exponential decay model.
    Uses 45-minute half-life and 4.0 mg/dL per gram BG impact.
    """
    try:
        treatments = await treatment_repo.get_recent(user_id, hours=hours)
        cob = iob_cob_service.calculate_cob(treatments)

        # Calculate 24h total carbs
        treatments_24h = await treatment_repo.get_recent(user_id, hours=24)
        total_carbs = sum(t.carbs or 0 for t in treatments_24h)

        # Calculate expected BG impact
        bg_impact = cob * 4.0  # 4 mg/dL per gram

        return COBResponse(
            cob=round(cob, 1),
            total_carbs_24h=round(total_carbs, 0),
            absorption_duration_min=iob_cob_service.carb_duration_min,
            half_life_min=45,
            bg_impact=round(bg_impact, 0),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error calculating COB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dose", response_model=DoseCalculationResponse)
async def calculate_dose(
    request: DoseCalculationRequest,
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Calculate recommended correction dose.

    Uses the formula: dose = (effective_bg - target_bg) / ISF

    Where effective_bg = current_bg + (COB * 4.0) - (IOB * ISF)

    This accounts for:
    - Pending insulin that will lower BG
    - Pending carbs that will raise BG
    """
    try:
        # Get current IOB and COB
        treatments = await treatment_repo.get_recent(user_id, hours=6)
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments) if request.include_cob else 0.0

        # Get ISF from prediction service or use override
        if request.isf_override:
            isf = request.isf_override
        else:
            settings = get_settings()
            pred_service = get_prediction_service(None, settings.model_device)
            isf = pred_service._get_isf(iob)

        # Calculate effects
        cob_effect = cob * 4.0  # mg/dL rise from carbs
        iob_effect = iob * isf  # mg/dL drop from insulin

        # Effective BG considering pending changes
        effective_bg = request.current_bg + cob_effect - iob_effect

        # Raw correction (can be negative if BG will drop below target)
        raw_correction = (effective_bg - request.target_bg) / isf

        # Recommended dose (floor at 0)
        recommended = max(0.0, raw_correction)

        # Generate warning if applicable
        warning = None
        if raw_correction < -1.0:
            warning = f"BG expected to drop {abs(raw_correction * isf):.0f} mg/dL below target. Consider carbs."
        elif request.current_bg < 70:
            warning = "Current BG is low. Treat hypoglycemia first."
        elif iob > 5.0:
            warning = "High IOB detected. Be cautious with additional insulin."

        return DoseCalculationResponse(
            current_bg=request.current_bg,
            target_bg=request.target_bg,
            effective_bg=round(effective_bg, 0),
            iob=round(iob, 2),
            cob=round(cob, 1),
            isf=round(isf, 1),
            iob_effect_mgdl=round(iob_effect, 0),
            cob_effect_mgdl=round(cob_effect, 0),
            raw_correction_units=round(raw_correction, 2),
            recommended_dose_units=round(recommended, 2),
            formula=f"({effective_bg:.0f} - {request.target_bg}) / {isf:.0f} = {raw_correction:.2f}U",
            warning=warning,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error calculating dose: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-insulin", response_model=ActiveInsulinResponse)
async def get_active_insulin(
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Get breakdown of active insulin doses.

    Shows each insulin dose with remaining active insulin.
    """
    try:
        # Get insulin treatments from last 4 hours
        treatments = await treatment_repo.get_recent(user_id, hours=4)
        insulin_treatments = [t for t in treatments if (t.insulin or 0) > 0]

        now = datetime.utcnow()
        doses = []
        total_iob = 0.0

        for t in insulin_treatments:
            # Calculate time since dose
            if t.timestamp.tzinfo:
                # Convert to naive UTC for comparison
                t_naive = t.timestamp.replace(tzinfo=None)
            else:
                t_naive = t.timestamp

            minutes_ago = (now - t_naive).total_seconds() / 60

            # Calculate remaining using exponential decay (81 min half-life)
            half_life = 81.0
            remaining = (t.insulin or 0) * (0.5 ** (minutes_ago / half_life))

            # Skip if essentially gone
            if remaining < 0.01:
                continue

            percent_remaining = (remaining / (t.insulin or 1)) * 100

            doses.append(ActiveInsulinDetail(
                timestamp=t.timestamp,
                original_dose=t.insulin or 0,
                remaining=round(remaining, 2),
                minutes_ago=int(minutes_ago),
                percent_remaining=round(percent_remaining, 1)
            ))

            total_iob += remaining

        # Sort by timestamp (most recent first)
        doses.sort(key=lambda x: x.timestamp, reverse=True)

        return ActiveInsulinResponse(
            total_iob=round(total_iob, 2),
            doses=doses,
            timestamp=now
        )

    except Exception as e:
        logger.error(f"Error getting active insulin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_calculations_summary(
    user_id: str = Query(default=TEMP_USER_ID),
    current_bg: float = Query(default=120, ge=40, le=500)
):
    """
    Get a summary of all current calculations.

    Returns IOB, COB, effective BG, and recommended dose in one call.
    """
    try:
        treatments = await treatment_repo.get_recent(user_id, hours=6)

        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)

        settings = get_settings()
        pred_service = get_prediction_service(None, settings.model_device)
        isf = pred_service._get_isf(iob)

        # Calculate effective BG and dose
        cob_effect = cob * 4.0
        iob_effect = iob * isf
        effective_bg = current_bg + cob_effect - iob_effect
        target_bg = 100  # Default target
        raw_correction = (effective_bg - target_bg) / isf
        recommended = max(0.0, raw_correction)

        return {
            "current_bg": current_bg,
            "effective_bg": round(effective_bg, 0),
            "iob": {
                "value": round(iob, 2),
                "unit": "units",
                "effect_mgdl": round(iob_effect, 0)
            },
            "cob": {
                "value": round(cob, 1),
                "unit": "grams",
                "effect_mgdl": round(cob_effect, 0)
            },
            "isf": {
                "value": round(isf, 1),
                "unit": "mg/dL per unit"
            },
            "recommended_dose": {
                "value": round(recommended, 2),
                "unit": "units",
                "to_reach": target_bg
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting calculations summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
