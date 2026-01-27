"""
Calculations API Endpoints
IOB, COB, and dose calculation endpoints.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from database.repositories import TreatmentRepository, LearnedISFRepository, LearnedICRRepository, LearnedPIRRepository, UserRepository, SharingRepository
from services.iob_cob_service import IOBCOBService
from services.prediction_service import get_prediction_service
from services.metabolic_params_service import get_metabolic_params_service, MetabolicState
from config import get_settings
from models.schemas import User
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calculations", tags=["calculations"])


def get_data_user_id(profile_id: str) -> str:
    """
    Strip profile_ prefix for data access.
    
    Data is stored under base user ID without prefix.
    Profile IDs like 'profile_05bf...' map to data under '05bf...'
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix
    return profile_id


# Service instances
treatment_repo = TreatmentRepository()
iob_cob_service = IOBCOBService.from_settings()
isf_repo = LearnedISFRepository()
icr_repo = LearnedICRRepository()
pir_repo = LearnedPIRRepository()
user_repo = UserRepository()
sharing_repo = SharingRepository()
metabolic_params_service = get_metabolic_params_service()


async def validate_user_access(requester_id: str, target_user_id: str) -> bool:
    """
    Check if requester has access to target user's data.
    Access is granted if:
    - requester_id == target_user_id (viewing own data)
    - requester owns the target profile (managed profile)
    - requester is a parent with target_user_id in their linkedChildIds
    - target_user has shared their data with requester via the sharing system
    """
    try:
        # Normalize both IDs for comparison
        # Profile IDs may have 'profile_' prefix but user IDs don't
        normalized_requester = get_data_user_id(requester_id)
        normalized_target = get_data_user_id(target_user_id)

        if normalized_requester == normalized_target:
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

        # Check if requester is a parent of the target (try both normalized and raw IDs)
        try:
            requester = await user_repo.get_by_id(normalized_requester)
            if requester and requester.linkedChildIds:
                # Check both normalized and raw target IDs
                if normalized_target in requester.linkedChildIds or target_user_id in requester.linkedChildIds:
                    return True
        except Exception as e:
            logger.warning(f"Error checking parent-child access: {e}")

        # Check if target user has shared data with requester (try both ID forms)
        try:
            share = await sharing_repo.get_share_for_profile(normalized_target, normalized_requester)
            if not share:
                # Try with profile_ prefix
                share = await sharing_repo.get_share_for_profile(target_user_id, requester_id)
            if share and share.isActive:
                role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                logger.info(f"Access granted via share: {target_user_id} shared with {requester_id} (role: {role_str})")
                return True
        except Exception as e:
            logger.warning(f"Error checking share access: {e}")

        return False
    except Exception as e:
        logger.error(f"Unexpected error in validate_user_access: {e}")
        return False


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


class POBResponse(BaseModel):
    """Protein on Board response."""
    pob: float = Field(..., description="Current POB in grams")
    total_protein_24h: float = Field(0.0, description="Total protein in last 24h")
    absorption_duration_min: int = Field(300, description="Protein absorption duration")
    half_life_min: int = Field(75, description="Protein absorption half-life")
    onset_min: int = Field(120, description="Protein onset delay")
    bg_impact: float = Field(0.0, description="Expected BG rise from POB")
    timestamp: datetime


class ProteinDoseDecayResponse(BaseModel):
    """Response for protein dose decay calculation."""
    time_since_meal_min: float = Field(..., description="Minutes since meal was logged")
    original_dose_now: float = Field(..., description="Original immediate protein dose")
    original_dose_later: float = Field(..., description="Original delayed protein dose")
    current_dose_now: float = Field(..., description="Current dose NOW (includes decayed)")
    current_dose_later: float = Field(..., description="Current dose LATER (after decay)")
    decayed_amount: float = Field(..., description="Amount that has decayed from LATER to NOW")
    decay_percent: float = Field(..., description="Percent of LATER that has decayed to NOW")
    all_now: bool = Field(..., description="True if all protein is NOW (past onset)")
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
    hours: int = Query(default=6, ge=1, le=24, description="Hours of treatment history"),
    user_id: Optional[str] = Query(default=None, description="User ID to get IOB for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current Insulin on Board (IOB).

    Calculates remaining active insulin using exponential decay model.
    Uses Novolog insulin curve with 81-minute half-life.
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's IOB"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        # Get recent treatments
        treatments = await treatment_repo.get_recent(data_user_id, hours=hours)

        # Calculate IOB
        iob = iob_cob_service.calculate_iob(treatments)

        # Calculate 24h total insulin
        treatments_24h = await treatment_repo.get_recent(data_user_id, hours=24)
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
    hours: int = Query(default=6, ge=1, le=24),
    user_id: Optional[str] = Query(default=None, description="User ID to get COB for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current Carbs on Board (COB).

    Calculates remaining carbs using exponential decay model.
    Uses 45-minute half-life and 4.0 mg/dL per gram BG impact.
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's COB"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        treatments = await treatment_repo.get_recent(data_user_id, hours=hours)
        cob = iob_cob_service.calculate_cob(treatments)

        # Calculate 24h total carbs
        treatments_24h = await treatment_repo.get_recent(data_user_id, hours=24)
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


@router.get("/pob", response_model=POBResponse)
async def get_pob(
    hours: int = Query(default=6, ge=1, le=24),
    user_id: Optional[str] = Query(default=None, description="User ID to get POB for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current Protein on Board (POB).

    Calculates remaining protein using exponential decay model with delayed onset.
    Protein has a 2-hour onset delay (vs 5-15 min for carbs) and 5-hour total duration.
    Uses 75-minute half-life and 2.0 mg/dL per gram BG impact.
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's POB"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        treatments = await treatment_repo.get_recent(data_user_id, hours=hours)
        pob = iob_cob_service.calculate_pob(treatments)

        # Calculate 24h total protein
        treatments_24h = await treatment_repo.get_recent(data_user_id, hours=24)
        total_protein = sum(getattr(t, 'protein', 0) or 0 for t in treatments_24h)

        # Calculate expected BG impact (protein raises BG ~50% as much as carbs)
        bg_impact = pob * iob_cob_service.protein_bg_factor  # 2.0 mg/dL per gram

        return POBResponse(
            pob=round(pob, 1),
            total_protein_24h=round(total_protein, 0),
            absorption_duration_min=iob_cob_service.protein_duration_min,
            half_life_min=int(iob_cob_service.protein_half_life_min),
            onset_min=int(iob_cob_service.protein_onset_min),
            bg_impact=round(bg_impact, 0),
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error calculating POB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_protein_dose_with_decay(
    total_protein_dose: float,
    upfront_percent: float,
    time_since_meal_min: float,
    protein_onset_min: float = 120.0,
    decay_half_life_min: float = 60.0
) -> tuple:
    """
    Calculate protein dose split with time-based continuous exponential decay.

    As time passes, the LATER portion decays into NOW:
    - At t=0: split per upfront_percent (e.g., 40% NOW, 60% LATER)
    - As time passes: LATER decays into NOW continuously
    - At t=onset: all becomes NOW (protein is hitting BG)

    Args:
        total_protein_dose: Total insulin dose for protein
        upfront_percent: Percent to give immediately (0-100)
        time_since_meal_min: Minutes since meal was logged
        protein_onset_min: When protein starts affecting BG (default: 120 min)
        decay_half_life_min: Half-life for the decay from LATER to NOW (default: 60 min)

    Returns:
        Tuple of (dose_now, dose_later, decayed_amount, decay_percent)
    """
    # If past onset, all protein is NOW
    if time_since_meal_min >= protein_onset_min:
        return total_protein_dose, 0.0, total_protein_dose * (1 - upfront_percent / 100), 100.0

    # Original split
    base_now = total_protein_dose * (upfront_percent / 100)
    base_later = total_protein_dose * (1 - upfront_percent / 100)

    if base_later <= 0:
        return total_protein_dose, 0.0, 0.0, 0.0

    # Calculate decay factor (0 to 1 as time progresses)
    # Use exponential decay: decay_factor approaches 1 as time approaches onset
    decay_factor = 1 - (0.5 ** (time_since_meal_min / decay_half_life_min))

    # Cap at 100% when reaching onset
    progress_to_onset = time_since_meal_min / protein_onset_min
    if progress_to_onset >= 1.0:
        decay_factor = 1.0

    # Calculate decayed amounts
    decayed_amount = base_later * decay_factor
    dose_now = base_now + decayed_amount
    dose_later = base_later - decayed_amount
    decay_percent = decay_factor * 100

    return dose_now, dose_later, decayed_amount, decay_percent


@router.get("/protein-dose-decay", response_model=ProteinDoseDecayResponse)
async def get_protein_dose_decay(
    treatment_id: str = Query(..., description="Treatment ID to calculate decay for"),
    upfront_percent: float = Query(default=40, ge=0, le=100, description="Original upfront percentage"),
    current_user: User = Depends(get_current_user)
):
    """
    Calculate current protein dose split accounting for time decay.

    As time passes since a meal, the delayed protein dose "decays" into the
    immediate dose, reflecting that protein is getting closer to affecting BG.

    Use this to update dose recommendations after a meal was logged.
    """
    user_id = current_user.id
    try:
        # Get the treatment
        treatment = await treatment_repo.get_by_id(treatment_id)
        if not treatment or treatment.userId != user_id:
            raise HTTPException(status_code=404, detail="Treatment not found")

        protein = getattr(treatment, 'protein', 0) or 0
        if protein <= 0:
            return ProteinDoseDecayResponse(
                time_since_meal_min=0,
                original_dose_now=0,
                original_dose_later=0,
                current_dose_now=0,
                current_dose_later=0,
                decayed_amount=0,
                decay_percent=0,
                all_now=True,
                timestamp=datetime.utcnow()
            )

        # Get PIR for this user
        pir = 25.0  # Default
        learned_pir = await pir_repo.get(user_id, "overall")
        if learned_pir and learned_pir.value:
            pir = learned_pir.value
            protein_onset = learned_pir.proteinOnsetMin
        else:
            protein_onset = 120.0

        # Calculate total protein dose
        total_protein_dose = protein / pir

        # Calculate time since meal
        now = datetime.utcnow()
        meal_time = treatment.timestamp.replace(tzinfo=None) if treatment.timestamp.tzinfo else treatment.timestamp
        time_since_meal_min = (now - meal_time).total_seconds() / 60

        # Calculate decay
        dose_now, dose_later, decayed_amount, decay_percent = calculate_protein_dose_with_decay(
            total_protein_dose=total_protein_dose,
            upfront_percent=upfront_percent,
            time_since_meal_min=time_since_meal_min,
            protein_onset_min=protein_onset
        )

        # Original split for reference
        original_now = total_protein_dose * (upfront_percent / 100)
        original_later = total_protein_dose * (1 - upfront_percent / 100)

        return ProteinDoseDecayResponse(
            time_since_meal_min=round(time_since_meal_min, 1),
            original_dose_now=round(original_now, 2),
            original_dose_later=round(original_later, 2),
            current_dose_now=round(dose_now, 2),
            current_dose_later=round(dose_later, 2),
            decayed_amount=round(decayed_amount, 2),
            decay_percent=round(decay_percent, 1),
            all_now=time_since_meal_min >= protein_onset,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating protein dose decay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dose", response_model=DoseCalculationResponse)
async def calculate_dose(
    request: DoseCalculationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate recommended correction dose using LEARNED metabolic parameters.

    Uses the formula: dose = (effective_bg - target_bg) / ISF

    Where effective_bg = current_bg + (COB * 4.0) - (IOB * ISF)

    IMPORTANT: This endpoint uses EFFECTIVE ISF which accounts for:
    - Long-term learned baseline (from historical data)
    - Short-term deviation (illness/sensitivity detection from last 2-3 days)

    If a user is sick (insulin resistant), the effective ISF will be LOWER,
    meaning insulin has LESS effect, so they need MORE insulin.
    """
    user_id = current_user.id
    try:
        # Get current IOB and COB
        treatments = await treatment_repo.get_recent(user_id, hours=6)
        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments) if request.include_cob else 0.0

        # Get EFFECTIVE ISF with illness detection, or use override
        isf_deviation = 0.0
        metabolic_state = MetabolicState.NORMAL
        if request.isf_override:
            isf = request.isf_override
        else:
            # Use learned ISF with short-term deviation (illness detection)
            effective_isf = await metabolic_params_service.get_effective_isf(
                user_id, is_fasting=(cob == 0), include_short_term=True
            )
            isf = effective_isf.value
            isf_deviation = effective_isf.deviation_percent
            if effective_isf.is_sick:
                metabolic_state = MetabolicState.SICK
            elif effective_isf.is_resistant:
                metabolic_state = MetabolicState.RESISTANT

        # Calculate effects using EFFECTIVE ISF
        cob_effect = cob * 4.0  # mg/dL rise from carbs
        iob_effect = iob * isf  # mg/dL drop from insulin (uses effective ISF)

        # Effective BG considering pending changes
        effective_bg = request.current_bg + cob_effect - iob_effect

        # Raw correction (can be negative if BG will drop below target)
        raw_correction = (effective_bg - request.target_bg) / isf

        # Recommended dose (floor at 0)
        recommended = max(0.0, raw_correction)

        # Generate warning if applicable
        warning = None
        if metabolic_state == MetabolicState.SICK:
            warning = f"ILLNESS DETECTED: ISF is {abs(isf_deviation):.0f}% lower than baseline. May need significantly more insulin."
        elif metabolic_state == MetabolicState.RESISTANT:
            warning = f"Mild resistance detected: ISF is {abs(isf_deviation):.0f}% lower. May need more insulin."
        elif raw_correction < -1.0:
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
    current_user: User = Depends(get_current_user)
):
    """
    Get breakdown of active insulin doses.

    Shows each insulin dose with remaining active insulin.
    """
    user_id = current_user.id
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
    current_bg: float = Query(default=120, ge=40, le=500),
    user_id: Optional[str] = Query(default=None, description="User ID to get summary for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get a summary of all current calculations using LEARNED metabolic parameters.

    Returns IOB, COB, effective BG, recommended dose, and metabolic state.
    Supports viewing shared accounts when user_id is provided.

    IMPORTANT: Uses EFFECTIVE ISF with illness detection - if a user is sick,
    the ISF will be lower and the metabolic state will indicate this.
    """
    # Use provided user_id or default to current user
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's calculations"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        treatments = await treatment_repo.get_recent(data_user_id, hours=6)

        iob = iob_cob_service.calculate_iob(treatments)
        cob = iob_cob_service.calculate_cob(treatments)
        pob = iob_cob_service.calculate_pob(treatments)

        # Get all effective metabolic parameters with illness detection
        params = await metabolic_params_service.get_all_params(
            user_id=data_user_id,
            is_fasting=(cob == 0 and pob == 0),
            include_short_term=True
        )
        isf = params.isf.value
        icr = params.icr.value

        # Calculate effective BG and dose using EFFECTIVE ISF
        cob_effect = cob * (isf / icr)  # Use learned ICR instead of hardcoded 4.0
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
            "pob": {
                "value": round(pob, 1),
                "unit": "grams"
            },
            "isf": {
                "value": round(isf, 1),
                "unit": "mg/dL per unit",
                "baseline": round(params.isf.baseline, 1),
                "deviation_percent": round(params.isf.deviation_percent, 1),
                "source": params.isf.source
            },
            "icr": {
                "value": round(icr, 1),
                "unit": "grams per unit",
                "source": params.icr.source
            },
            "metabolic_state": {
                "state": params.metabolic_state.value,
                "description": params.state_description,
                "is_sick": params.isf.is_sick,
                "is_resistant": params.isf.is_resistant
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


# ==================== Meal Dose Calculation with ICR/PIR ====================

class MealDoseRequest(BaseModel):
    """Request for meal dose calculation."""
    current_bg: float = Field(..., ge=40, le=500, description="Current BG in mg/dL")
    target_bg: float = Field(100, ge=70, le=150, description="Target BG")
    carbs: float = Field(0, ge=0, le=500, description="Carbs in grams")
    protein: float = Field(0, ge=0, le=300, description="Protein in grams")
    fat: float = Field(0, ge=0, le=200, description="Fat in grams (optional, for context)")
    use_learned_icr: bool = Field(True, description="Use AI-learned ICR if available")
    use_learned_pir: bool = Field(True, description="Use AI-learned PIR if available")
    include_protein: bool = Field(True, description="Include protein in calculation")
    protein_upfront_percent: float = Field(40, ge=0, le=100, description="Percent of protein insulin to give upfront")
    icr_override: Optional[float] = Field(None, ge=3, le=50, description="Override ICR")
    pir_override: Optional[float] = Field(None, ge=10, le=100, description="Override PIR")
    isf_override: Optional[float] = Field(None, ge=10, le=200, description="Override ISF")


class MealDoseResponse(BaseModel):
    """Response with detailed meal dose breakdown."""
    # Input summary
    current_bg: float
    target_bg: float
    carbs: float
    protein: float
    fat: float

    # Ratios used
    isf: float
    isf_source: str  # "learned" or "manual"
    icr: float
    icr_source: str  # "learned" or "manual"
    pir: Optional[float] = None
    pir_source: Optional[str] = None  # "learned" or "manual"

    # IOB adjustment
    iob: float
    iob_effect_mgdl: float

    # Dose breakdown
    correction_dose: float = Field(..., description="Insulin for BG correction")
    carb_dose: float = Field(..., description="Insulin for carbs")
    protein_dose_immediate: float = Field(0, description="Protein insulin to give NOW")
    protein_dose_delayed: float = Field(0, description="Protein insulin to give LATER")
    protein_dose_total: float = Field(0, description="Total protein insulin")

    # Totals
    immediate_total: float = Field(..., description="Total to give NOW (correction + carb + immediate protein)")
    delayed_total: float = Field(..., description="Total to give LATER (delayed protein)")
    grand_total: float = Field(..., description="All insulin for this meal")

    # Timing info
    protein_onset_minutes: Optional[int] = None
    protein_peak_minutes: Optional[int] = None
    delayed_timing_advice: Optional[str] = None

    # Formulas for transparency
    correction_formula: str
    carb_formula: str
    protein_formula: Optional[str] = None

    # Warnings
    warnings: List[str] = []
    timestamp: datetime


@router.post("/meal-dose", response_model=MealDoseResponse)
async def calculate_meal_dose(
    request: MealDoseRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Calculate comprehensive meal dose with carb and protein breakdown.

    Returns:
    - Correction dose: For current BG above target
    - Carb dose: carbs / ICR (using learned or manual ICR)
    - Protein dose: Split into immediate + delayed portions
      - Immediate: Give at meal time (faster protein effect)
      - Delayed: Give later via extended bolus (for late rise)

    The protein_upfront_percent determines how much protein insulin
    is given immediately vs delayed.
    """
    user_id = current_user.id
    try:
        warnings = []

        # Get current IOB
        treatments = await treatment_repo.get_recent(user_id, hours=6)
        iob = iob_cob_service.calculate_iob(treatments)

        # ========== Get ISF ==========
        isf = request.isf_override
        isf_source = "override"

        if not isf:
            # Try learned ISF first
            learned_isf = await isf_repo.get(user_id, "meal")
            if not learned_isf:
                learned_isf = await isf_repo.get(user_id, "fasting")

            if learned_isf and learned_isf.value:
                isf = learned_isf.value
                isf_source = "learned"
            else:
                # Fall back to default
                settings = get_settings()
                pred_service = get_prediction_service(None, settings.model_device)
                isf = pred_service._get_isf(iob)
                isf_source = "default"

        # ========== Get ICR ==========
        icr = request.icr_override
        icr_source = "override"

        if not icr:
            if request.use_learned_icr:
                # Determine meal type from time
                hour = datetime.utcnow().hour
                if 5 <= hour < 10:
                    meal_type = "breakfast"
                elif 10 <= hour < 15:
                    meal_type = "lunch"
                else:
                    meal_type = "dinner"

                # Try meal-specific, then overall
                learned_icr = await icr_repo.get(user_id, meal_type)
                if not learned_icr:
                    learned_icr = await icr_repo.get(user_id, "overall")

                if learned_icr and learned_icr.value:
                    icr = learned_icr.value
                    icr_source = f"learned ({meal_type})"
                else:
                    # Fall back to user settings default
                    icr = 10.0  # Default
                    icr_source = "default"
            else:
                icr = 10.0  # Manual/default
                icr_source = "manual"

        # ========== Get PIR ==========
        pir = None
        pir_source = None
        protein_onset = None
        protein_peak = None

        if request.include_protein and request.protein > 0:
            pir = request.pir_override
            pir_source = "override" if pir else None

            if not pir:
                if request.use_learned_pir:
                    learned_pir = await pir_repo.get(user_id, "overall")
                    if learned_pir and learned_pir.value:
                        pir = learned_pir.value
                        pir_source = "learned"
                        protein_onset = int(learned_pir.proteinOnsetMin)
                        protein_peak = int(learned_pir.proteinPeakMin)
                    else:
                        pir = 25.0  # Default: 25g protein per unit
                        pir_source = "default"
                else:
                    pir = 25.0  # Manual/default
                    pir_source = "manual"

        # ========== Calculate Doses ==========

        # IOB effect on BG
        iob_effect = iob * isf

        # Effective BG (accounting for IOB that will lower it)
        effective_bg = request.current_bg - iob_effect

        # Correction dose
        correction_dose = 0.0
        if effective_bg > request.target_bg:
            correction_dose = (effective_bg - request.target_bg) / isf
        correction_formula = f"({effective_bg:.0f} - {request.target_bg}) / {isf:.0f} = {correction_dose:.2f}U"

        # Carb dose
        carb_dose = 0.0
        if request.carbs > 0:
            carb_dose = request.carbs / icr
        carb_formula = f"{request.carbs}g / {icr:.1f} = {carb_dose:.2f}U"

        # Protein dose with split timing
        protein_dose_total = 0.0
        protein_dose_immediate = 0.0
        protein_dose_delayed = 0.0
        protein_formula = None

        if request.include_protein and request.protein > 0 and pir:
            protein_dose_total = request.protein / pir

            # Split based on upfront percentage
            upfront_ratio = request.protein_upfront_percent / 100.0
            protein_dose_immediate = protein_dose_total * upfront_ratio
            protein_dose_delayed = protein_dose_total * (1 - upfront_ratio)

            protein_formula = (
                f"{request.protein}g / {pir:.0f} = {protein_dose_total:.2f}U "
                f"({request.protein_upfront_percent:.0f}% now = {protein_dose_immediate:.2f}U, "
                f"{100 - request.protein_upfront_percent:.0f}% delayed = {protein_dose_delayed:.2f}U)"
            )

        # ========== Totals ==========
        immediate_total = correction_dose + carb_dose + protein_dose_immediate
        delayed_total = protein_dose_delayed
        grand_total = immediate_total + delayed_total

        # ========== Timing Advice ==========
        delayed_timing_advice = None
        if delayed_total > 0:
            if protein_onset and protein_peak:
                delayed_timing_advice = (
                    f"Give {delayed_total:.1f}U extended bolus over {protein_onset}-{protein_peak} minutes, "
                    f"or give manually around {protein_onset//60}h after meal"
                )
            else:
                delayed_timing_advice = (
                    f"Give {delayed_total:.1f}U as extended bolus over 2-3 hours, "
                    f"or give manually 2h after meal"
                )

        # ========== Warnings ==========
        if request.current_bg < 70:
            warnings.append("Current BG is low. Treat hypoglycemia first before bolusing.")

        if iob > 3.0:
            warnings.append(f"High IOB ({iob:.1f}U) detected. Already accounts for -{iob_effect:.0f} mg/dL.")

        if correction_dose < 0:
            warnings.append("BG is expected to drop. No correction needed.")
            correction_dose = 0.0

        if grand_total > 10:
            warnings.append(f"Large total dose ({grand_total:.1f}U). Double-check carb/protein counts.")

        if protein_dose_total > 0 and pir_source == "default":
            warnings.append("Using default PIR (25g/U). Consider learning your PIR for more accuracy.")

        return MealDoseResponse(
            current_bg=request.current_bg,
            target_bg=request.target_bg,
            carbs=request.carbs,
            protein=request.protein,
            fat=request.fat,
            isf=round(isf, 1),
            isf_source=isf_source,
            icr=round(icr, 1),
            icr_source=icr_source,
            pir=round(pir, 1) if pir else None,
            pir_source=pir_source,
            iob=round(iob, 2),
            iob_effect_mgdl=round(iob_effect, 0),
            correction_dose=round(max(0, correction_dose), 2),
            carb_dose=round(carb_dose, 2),
            protein_dose_immediate=round(protein_dose_immediate, 2),
            protein_dose_delayed=round(protein_dose_delayed, 2),
            protein_dose_total=round(protein_dose_total, 2),
            immediate_total=round(immediate_total, 2),
            delayed_total=round(delayed_total, 2),
            grand_total=round(grand_total, 2),
            protein_onset_minutes=protein_onset,
            protein_peak_minutes=protein_peak,
            delayed_timing_advice=delayed_timing_advice,
            correction_formula=correction_formula,
            carb_formula=carb_formula,
            protein_formula=protein_formula,
            warnings=warnings,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error calculating meal dose: {e}")
        raise HTTPException(status_code=500, detail=str(e))
