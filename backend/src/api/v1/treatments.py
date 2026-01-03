"""
Treatments API Endpoints for T1D-AI
Provides insulin and carb treatment data.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from uuid import uuid4

from models.schemas import Treatment, TreatmentType
from database.repositories import TreatmentRepository

logger = logging.getLogger(__name__)
router = APIRouter()

treatment_repo = TreatmentRepository()

# Temporary: hardcoded user ID until auth is implemented
TEMP_USER_ID = "demo_user"


class LogTreatmentRequest(BaseModel):
    """Request to log a new treatment."""
    type: TreatmentType
    insulin: Optional[float] = None
    carbs: Optional[float] = None
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None


@router.get("/recent", response_model=List[Treatment])
async def get_recent_treatments(
    user_id: str = Query(default=TEMP_USER_ID),
    hours: int = Query(default=24, ge=1, le=168),
    treatment_type: Optional[str] = Query(default=None)
):
    """
    Get recent treatments.

    Returns insulin and/or carb treatments for the specified time period.
    """
    try:
        treatments = await treatment_repo.get_recent(
            user_id=user_id,
            hours=hours,
            treatment_type=treatment_type
        )
        return treatments

    except Exception as e:
        logger.error(f"Error getting treatments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/log", response_model=Treatment)
async def log_treatment(
    request: LogTreatmentRequest,
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Log a new treatment (insulin or carbs).
    """
    try:
        # Validate request
        if request.type == TreatmentType.INSULIN and not request.insulin:
            raise HTTPException(status_code=400, detail="Insulin amount required for insulin treatment")
        if request.type == TreatmentType.CARBS and not request.carbs:
            raise HTTPException(status_code=400, detail="Carbs amount required for carb treatment")

        treatment = Treatment(
            id=f"{user_id}_{uuid4().hex[:12]}",
            userId=user_id,
            timestamp=request.timestamp or datetime.utcnow(),
            type=request.type,
            insulin=request.insulin,
            carbs=request.carbs,
            notes=request.notes,
            source="manual"
        )

        created = await treatment_repo.create(treatment)
        logger.info(f"Logged treatment: {request.type} for user {user_id}")

        return created

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging treatment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/iob")
async def get_iob(user_id: str = Query(default=TEMP_USER_ID)):
    """
    Get current Insulin on Board (IOB).
    """
    from services.iob_cob_service import IOBCOBService

    try:
        service = IOBCOBService.from_settings()
        treatments = await treatment_repo.get_for_iob_calculation(user_id)
        iob = service.calculate_iob(treatments)

        return {
            "iob": iob,
            "unit": "U",
            "calculatedAt": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating IOB: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cob")
async def get_cob(user_id: str = Query(default=TEMP_USER_ID)):
    """
    Get current Carbs on Board (COB).
    """
    from services.iob_cob_service import IOBCOBService

    try:
        service = IOBCOBService.from_settings()
        treatments = await treatment_repo.get_for_cob_calculation(user_id)
        cob = service.calculate_cob(treatments)

        return {
            "cob": cob,
            "unit": "g",
            "calculatedAt": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating COB: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
