"""
Home Assistant Integration API

Provides:
- API key generation/management for long-lived machine auth
- /ha/status endpoint returning glucose, IOB, COB, predictions for HA sensors
- Simple X-API-Key header auth (no JWT refresh needed)
"""
import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Header, Depends, Query
from pydantic import BaseModel, Field

from auth.routes import get_current_user
from database.cosmos_client import get_cosmos_manager
from database.repositories import (
    GlucoseRepository, TreatmentRepository, UserRepository,
    UserAbsorptionProfileRepository,
)
from services.iob_cob_service import IOBCOBService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ha", tags=["Home Assistant"])

# Repositories
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
user_repo = UserRepository()
absorption_profile_repo = UserAbsorptionProfileRepository()
iob_cob_service = IOBCOBService.from_settings()


# ===================== Schemas =====================

class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Friendly name for this key")
    profile_id: Optional[str] = Field(None, description="Profile ID to scope the key to (optional)")


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key: Optional[str] = Field(None, description="Only returned on creation")
    key_prefix: str = Field(..., description="First 8 chars for identification")
    profile_id: Optional[str] = None
    created_at: str
    last_used_at: Optional[str] = None


class ApiKeyListResponse(BaseModel):
    keys: List[ApiKeyResponse]


class HAStatusResponse(BaseModel):
    glucose: Optional[int] = Field(None, description="Current glucose mg/dL")
    trend: Optional[str] = Field(None, description="Trend direction")
    trend_arrow: Optional[str] = Field(None, description="Trend arrow symbol")
    glucose_timestamp: Optional[str] = Field(None, description="When glucose was read")
    minutes_ago: Optional[float] = Field(None, description="Minutes since last reading")
    iob: Optional[float] = Field(None, description="Insulin on board (units)")
    cob: Optional[float] = Field(None, description="Carbs on board (grams)")
    last_bolus_units: Optional[float] = None
    last_bolus_minutes_ago: Optional[float] = None
    last_carbs_grams: Optional[float] = None
    last_carbs_minutes_ago: Optional[float] = None
    high_threshold: int = 180
    low_threshold: int = 70
    critical_high_threshold: int = 250
    critical_low_threshold: int = 55
    status: str = Field("ok", description="ok, high, low, critical_high, critical_low, stale")
    profile_name: Optional[str] = None


# ===================== Helpers =====================

def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key for storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _get_api_keys_container():
    """Get or create the api_keys container."""
    manager = get_cosmos_manager()
    return manager.get_container("api_keys", "/userId")


TREND_ARROWS = {
    "DoubleDown": "⇊", "SingleDown": "↓", "FortyFiveDown": "↘",
    "Flat": "→", "FortyFiveUp": "↗", "SingleUp": "↑", "DoubleUp": "⇈",
    # Also handle string enum values
    "DOUBLE_DOWN": "⇊", "SINGLE_DOWN": "↓", "FORTY_FIVE_DOWN": "↘",
    "FLAT": "→", "FORTY_FIVE_UP": "↗", "SINGLE_UP": "↑", "DOUBLE_UP": "⇈",
}


def _get_data_user_id(profile_id: str) -> str:
    """Strip profile_ prefix for data access."""
    if profile_id.startswith("profile_"):
        return profile_id[8:]
    return profile_id


# ===================== API Key Auth =====================

async def get_user_from_api_key(x_api_key: str = Header(...)) -> dict:
    """Authenticate via X-API-Key header. Returns the API key document."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_hash = _hash_key(x_api_key)
    container = _get_api_keys_container()

    try:
        query = "SELECT * FROM c WHERE c.keyHash = @hash AND c.revoked != true"
        items = list(container.query_items(
            query=query,
            parameters=[{"name": "@hash", "value": key_hash}],
            enable_cross_partition_query=True,
        ))
        if not items:
            raise HTTPException(status_code=401, detail="Invalid API key")

        key_doc = items[0]

        # Update last_used_at (fire-and-forget, don't block response)
        try:
            key_doc["last_used_at"] = datetime.now(timezone.utc).isoformat()
            container.upsert_item(key_doc)
        except Exception:
            pass  # Non-critical

        return key_doc

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")


# ===================== Key Management Endpoints =====================

@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    request: ApiKeyCreateRequest,
    current_user=Depends(get_current_user),
):
    """Generate a new API key for Home Assistant integration.

    The raw key is returned ONCE in the response. Store it securely.
    """
    raw_key = f"t1d_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    key_id = str(uuid.uuid4())

    doc = {
        "id": key_id,
        "userId": current_user.id,
        "name": request.name,
        "keyHash": key_hash,
        "keyPrefix": raw_key[:12],
        "profileId": request.profile_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_used_at": None,
        "revoked": False,
    }

    container = _get_api_keys_container()
    container.upsert_item(doc)

    logger.info(f"API key created: {request.name} for user {current_user.id}")

    return ApiKeyResponse(
        id=key_id,
        name=request.name,
        key=raw_key,
        key_prefix=raw_key[:12],
        profile_id=request.profile_id,
        created_at=doc["created_at"],
    )


@router.get("/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(current_user=Depends(get_current_user)):
    """List all API keys for the current user (keys are NOT returned, only prefixes)."""
    container = _get_api_keys_container()
    query = "SELECT * FROM c WHERE c.userId = @userId AND c.revoked != true"
    items = list(container.query_items(
        query=query,
        parameters=[{"name": "@userId", "value": current_user.id}],
        partition_key=current_user.id,
    ))

    keys = [
        ApiKeyResponse(
            id=item["id"],
            name=item["name"],
            key_prefix=item.get("keyPrefix", "t1d_***"),
            profile_id=item.get("profileId"),
            created_at=item["created_at"],
            last_used_at=item.get("last_used_at"),
        )
        for item in items
    ]

    return ApiKeyListResponse(keys=keys)


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str, current_user=Depends(get_current_user)):
    """Revoke an API key."""
    container = _get_api_keys_container()

    try:
        doc = container.read_item(item=key_id, partition_key=current_user.id)
    except Exception:
        raise HTTPException(status_code=404, detail="API key not found")

    if doc["userId"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not your API key")

    doc["revoked"] = True
    doc["revoked_at"] = datetime.now(timezone.utc).isoformat()
    container.upsert_item(doc)

    logger.info(f"API key revoked: {doc['name']} for user {current_user.id}")
    return {"message": "API key revoked"}


# ===================== HA Status Endpoint =====================

@router.get("/status", response_model=HAStatusResponse)
async def get_ha_status(key_doc: dict = Depends(get_user_from_api_key)):
    """Get current diabetes status for Home Assistant.

    Authenticates via X-API-Key header (no JWT needed).
    Returns glucose, trend, IOB, COB, last bolus/carbs, and alert thresholds.
    """
    user_id = key_doc["userId"]
    profile_id = key_doc.get("profileId")

    # If key is scoped to a profile, use that; otherwise use user's own data
    data_user_id = _get_data_user_id(profile_id) if profile_id else user_id

    # Get user settings for thresholds
    user = await user_repo.get_by_id(user_id)
    settings = user.settings if user else None

    high_threshold = settings.highThreshold if settings and settings.highThreshold else 180
    low_threshold = settings.lowThreshold if settings and settings.lowThreshold else 70
    critical_high = settings.criticalHighThreshold if settings and settings.criticalHighThreshold else 250
    critical_low = settings.criticalLowThreshold if settings and settings.criticalLowThreshold else 55

    # Get current glucose
    latest = await glucose_repo.get_latest(data_user_id)

    # Try Dexcom if available
    try:
        from services.dexcom_service import DexcomShareService
        dexcom_svc = DexcomShareService()
        dexcom_reading = await dexcom_svc.get_latest_reading_async()
        if dexcom_reading:
            from api.v1.glucose import dexcom_to_glucose_reading
            dexcom_glucose = dexcom_to_glucose_reading(dexcom_reading, data_user_id)
            if not latest or dexcom_glucose.timestamp > latest.timestamp:
                latest = dexcom_glucose
    except Exception:
        pass  # Dexcom not configured or unavailable

    if not latest:
        return HAStatusResponse(
            status="no_data",
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            critical_high_threshold=critical_high,
            critical_low_threshold=critical_low,
        )

    # Calculate age of reading
    now = datetime.now(timezone.utc)
    reading_time = latest.timestamp
    if reading_time.tzinfo is None:
        reading_time = reading_time.replace(tzinfo=timezone.utc)
    minutes_ago = (now - reading_time).total_seconds() / 60

    # Determine trend arrow
    trend_str = str(latest.trend) if latest.trend else "Flat"
    trend_arrow = TREND_ARROWS.get(trend_str, "→")

    # Get treatments for IOB/COB
    treatments = await treatment_repo.get_recent(data_user_id, hours=8)

    # Load absorption profile for personalized calculations
    absorption_profile = None
    try:
        absorption_profile = await absorption_profile_repo.get(data_user_id)
        if absorption_profile and absorption_profile.confidence <= 0.3:
            absorption_profile = None
    except Exception:
        pass

    iob = iob_cob_service.calculate_iob(treatments, include_absorption_ramp=False)
    cob = iob_cob_service.calculate_cob(treatments, include_absorption_ramp=False)

    # Find last bolus and last carbs
    last_bolus_units = None
    last_bolus_minutes_ago = None
    last_carbs_grams = None
    last_carbs_minutes_ago = None

    for t in sorted(treatments, key=lambda x: x.timestamp, reverse=True):
        t_time = t.timestamp
        if t_time.tzinfo is None:
            t_time = t_time.replace(tzinfo=timezone.utc)
        t_mins = (now - t_time).total_seconds() / 60

        if last_bolus_units is None and t.insulin and t.insulin > 0:
            last_bolus_units = round(t.insulin, 2)
            last_bolus_minutes_ago = round(t_mins, 0)

        if last_carbs_grams is None and t.carbs and t.carbs > 0:
            last_carbs_grams = round(t.carbs, 0)
            last_carbs_minutes_ago = round(t_mins, 0)

        if last_bolus_units is not None and last_carbs_grams is not None:
            break

    # Determine status
    glucose_val = latest.value
    if minutes_ago > 15:
        status = "stale"
    elif glucose_val <= critical_low:
        status = "critical_low"
    elif glucose_val <= low_threshold:
        status = "low"
    elif glucose_val >= critical_high:
        status = "critical_high"
    elif glucose_val >= high_threshold:
        status = "high"
    else:
        status = "ok"

    # Get profile name if scoped
    profile_name = None
    if profile_id:
        try:
            from database.repositories import ProfileRepository
            profile_repo = ProfileRepository()
            profile = await profile_repo.get_by_id(profile_id, user_id)
            if profile:
                profile_name = profile.get("displayName") or profile.get("name")
        except Exception:
            pass

    return HAStatusResponse(
        glucose=glucose_val,
        trend=trend_str,
        trend_arrow=trend_arrow,
        glucose_timestamp=reading_time.isoformat(),
        minutes_ago=round(minutes_ago, 1),
        iob=round(iob, 2),
        cob=round(cob, 0),
        last_bolus_units=last_bolus_units,
        last_bolus_minutes_ago=last_bolus_minutes_ago,
        last_carbs_grams=last_carbs_grams,
        last_carbs_minutes_ago=last_carbs_minutes_ago,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        critical_high_threshold=critical_high,
        critical_low_threshold=critical_low,
        status=status,
        profile_name=profile_name,
    )
