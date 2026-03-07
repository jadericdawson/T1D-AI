"""
Home Assistant Integration API

Provides:
- API key generation/management for long-lived machine auth
- /ha/status endpoint returning comprehensive diabetes data for HA sensors
- Simple X-API-Key header auth (no JWT refresh needed)

Response sections (controlled via `include` query param):
- glucose_history: Recent glucose readings with trend arrows
- treatments: Recent insulin/carb treatments
- pump: Battery, mode, control mode, suspend, alerts, site change, cartridge
- predictions: Linear + TFT glucose predictions
- metabolic: ISF, ICR, PIR, metabolic state, absorption
- computed: IOB, COB, POB, dose recommendation
- daily_totals: Daily insulin/carb totals from pump
- thresholds: Alert thresholds (always included)
"""
import asyncio
import hashlib
import logging
import secrets
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Header, Depends, Query
from pydantic import BaseModel, Field

from auth.routes import get_current_user
from database.cosmos_client import get_cosmos_manager
from database.repositories import (
    GlucoseRepository, TreatmentRepository, UserRepository,
    UserAbsorptionProfileRepository, PumpStatusRepository,
)
from services.iob_cob_service import IOBCOBService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ha", tags=["Home Assistant"])

# Repositories
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
user_repo = UserRepository()
absorption_profile_repo = UserAbsorptionProfileRepository()
pump_status_repo = PumpStatusRepository()
iob_cob_service = IOBCOBService.from_settings()

# All sections that can be requested via ?include=
ALL_SECTIONS = {
    "glucose_history", "treatments", "pump", "predictions",
    "metabolic", "computed", "daily_totals",
}


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
    "DOUBLE_DOWN": "⇊", "SINGLE_DOWN": "↓", "FORTY_FIVE_DOWN": "↘",
    "FLAT": "→", "FORTY_FIVE_UP": "↗", "SINGLE_UP": "↑", "DOUBLE_UP": "⇈",
}

# Numeric trend to string mapping for predictions
TREND_NUM_MAP = {
    -3: "DoubleDown", -2: "SingleDown", -1: "FortyFiveDown",
    0: "Flat",
    1: "FortyFiveUp", 2: "SingleUp", 3: "DoubleUp",
}


def _get_data_user_id(profile_id: str) -> str:
    """Strip profile_ prefix for data access."""
    if profile_id.startswith("profile_"):
        return profile_id[8:]
    return profile_id


def _trend_to_numeric(trend_str: str) -> int:
    """Convert trend string to numeric value for prediction service."""
    str_to_num = {v: k for k, v in TREND_NUM_MAP.items()}
    return str_to_num.get(trend_str, 0)


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

@router.get("/status")
async def get_ha_status(
    key_doc: dict = Depends(get_user_from_api_key),
    hours: float = Query(default=3, ge=0.5, le=24, description="Hours of history to include"),
    include: Optional[str] = Query(
        default=None,
        description="Comma-separated sections to include (glucose_history,treatments,pump,"
                    "predictions,metabolic,computed,daily_totals) or 'all'. "
                    "Default: all sections included.",
    ),
):
    """Get comprehensive diabetes status for Home Assistant.

    Authenticates via X-API-Key header (no JWT needed).

    Top-level fields (glucose, trend, IOB, COB, status, thresholds) are always returned
    for backward compatibility. Additional sections are opt-in via `include` param.
    """
    user_id = key_doc["userId"]
    profile_id = key_doc.get("profileId")
    data_user_id = _get_data_user_id(profile_id) if profile_id else user_id

    # Parse requested sections
    if include is None or include.strip().lower() == "all":
        requested_sections = ALL_SECTIONS
    else:
        requested_sections = {s.strip().lower() for s in include.split(",") if s.strip()}

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    # --- Parallel data fetches ---
    # Always need: user settings, latest glucose, treatments (for IOB/COB)
    # Conditionally: pump status, glucose history, predictions, metabolic params
    coros: dict[str, Any] = {
        "user": user_repo.get_by_id(user_id),
        "latest_glucose": glucose_repo.get_latest(data_user_id),
        "treatments": treatment_repo.get_recent(data_user_id, hours=8),
    }

    if "glucose_history" in requested_sections:
        coros["glucose_history"] = glucose_repo.get_history(data_user_id, start_time=since)

    if "pump" in requested_sections or "daily_totals" in requested_sections:
        coros["pump_status"] = pump_status_repo.get(data_user_id)

    # Run all fetches in parallel
    keys = list(coros.keys())
    results = await asyncio.gather(*coros.values(), return_exceptions=True)
    data = {}
    for k, r in zip(keys, results):
        if isinstance(r, Exception):
            logger.warning(f"Failed to fetch {k}: {r}")
            data[k] = None
        else:
            data[k] = r

    user = data.get("user")
    latest = data.get("latest_glucose")
    treatments = data.get("treatments") or []
    settings = user.settings if user else None

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
        pass

    # --- Thresholds (always included) ---
    high_threshold = settings.highThreshold if settings and settings.highThreshold else 180
    low_threshold = settings.lowThreshold if settings and settings.lowThreshold else 70
    critical_high = settings.criticalHighThreshold if settings and settings.criticalHighThreshold else 250
    critical_low = settings.criticalLowThreshold if settings and settings.criticalLowThreshold else 55

    # --- Base response (backward compatible) ---
    response: dict[str, Any] = {
        "thresholds": {
            "high": high_threshold,
            "low": low_threshold,
            "critical_high": critical_high,
            "critical_low": critical_low,
        },
    }

    if not latest:
        response.update({
            "glucose": None,
            "trend": None,
            "trend_arrow": None,
            "glucose_timestamp": None,
            "minutes_ago": None,
            "status": "no_data",
            "profile_name": None,
        })
        return response

    # Glucose fields
    reading_time = latest.timestamp
    if reading_time.tzinfo is None:
        reading_time = reading_time.replace(tzinfo=timezone.utc)
    minutes_ago = (now - reading_time).total_seconds() / 60

    trend_str = str(latest.trend) if latest.trend else "Flat"
    trend_arrow = TREND_ARROWS.get(trend_str, "→")

    glucose_val = latest.value

    # Status
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

    # Profile name
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

    # Load absorption profile for IOB/COB calculations
    try:
        absorption_profile = await absorption_profile_repo.get(data_user_id)
        if absorption_profile and absorption_profile.confidence <= 0.3:
            absorption_profile = None
    except Exception:
        absorption_profile = None

    # IOB & COB (always computed for backward compat)
    iob = iob_cob_service.calculate_iob(treatments, include_absorption_ramp=False)
    cob = iob_cob_service.calculate_cob(treatments, include_absorption_ramp=False)

    # Last bolus/carbs
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

    response.update({
        "glucose": glucose_val,
        "trend": trend_str,
        "trend_arrow": trend_arrow,
        "glucose_timestamp": reading_time.isoformat(),
        "minutes_ago": round(minutes_ago, 1),
        "iob": round(iob, 2),
        "cob": round(cob, 0),
        "last_bolus_units": last_bolus_units,
        "last_bolus_minutes_ago": last_bolus_minutes_ago,
        "last_carbs_grams": last_carbs_grams,
        "last_carbs_minutes_ago": last_carbs_minutes_ago,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "critical_high_threshold": critical_high,
        "critical_low_threshold": critical_low,
        "status": status,
        "profile_name": profile_name,
    })

    # --- Optional Sections ---

    # Glucose history
    if "glucose_history" in requested_sections:
        glucose_history_data = data.get("glucose_history") or []
        response["glucose_history"] = [
            {
                "value": r.value,
                "trend": str(r.trend) if r.trend else "Flat",
                "trend_arrow": TREND_ARROWS.get(str(r.trend) if r.trend else "Flat", "→"),
                "timestamp": (r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc)).isoformat(),
                "minutes_ago": round((now - (r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc))).total_seconds() / 60, 1),
            }
            for r in sorted(glucose_history_data, key=lambda x: x.timestamp, reverse=True)
        ]

    # Treatments history
    if "treatments" in requested_sections:
        # Get treatments within the requested time range
        recent_treatments = [t for t in treatments if _get_treatment_ts(t) >= since]
        response["treatments"] = [
            _format_treatment(t, now) for t in sorted(recent_treatments, key=lambda x: x.timestamp, reverse=True)
        ]

    # Pump status
    if "pump" in requested_sections:
        pump_doc = data.get("pump_status")
        if pump_doc:
            response["pump"] = {
                "battery_percent": pump_doc.get("battery_percent"),
                "battery_millivolts": pump_doc.get("battery_millivolts"),
                "current_mode": pump_doc.get("current_mode"),
                "control_mode": pump_doc.get("control_mode"),
                "is_suspended": pump_doc.get("is_suspended", False),
                "last_suspend_reason": pump_doc.get("last_suspend_reason"),
                "last_alert": pump_doc.get("last_alert"),
                "last_alert_at": pump_doc.get("last_alert_at"),
                "last_alarm": pump_doc.get("last_alarm"),
                "last_alarm_at": pump_doc.get("last_alarm_at"),
                "last_site_change_at": pump_doc.get("last_site_change_at"),
                "site_age_hours": pump_doc.get("site_age_hours"),
                "last_cartridge_change_at": pump_doc.get("last_cartridge_change_at"),
                "last_cartridge_volume": pump_doc.get("last_cartridge_volume"),
                "pump_iob": pump_doc.get("pump_iob"),
                "last_updated": pump_doc.get("last_updated"),
                "recent_alerts": pump_doc.get("recent_alerts", []),
                "recent_alarms": pump_doc.get("recent_alarms", []),
                "recent_mode_changes": pump_doc.get("recent_mode_changes", []),
            }
        else:
            response["pump"] = None

    # Predictions
    if "predictions" in requested_sections:
        response["predictions"] = await _get_predictions(
            data_user_id, glucose_val, trend_str, iob, cob, treatments, data.get("glucose_history")
        )

    # Metabolic params
    if "metabolic" in requested_sections:
        response["metabolic"] = await _get_metabolic_params(data_user_id)

    # Computed (IOB, COB, POB, dose recommendation)
    if "computed" in requested_sections:
        response["computed"] = await _get_computed(
            data_user_id, glucose_val, iob, cob, treatments
        )

    # Daily totals from pump status
    if "daily_totals" in requested_sections:
        pump_doc = data.get("pump_status")
        if pump_doc:
            response["daily_totals"] = {
                "daily_basal_units": pump_doc.get("daily_basal_units"),
                "daily_bolus_units": pump_doc.get("daily_bolus_units"),
                "daily_total_insulin": pump_doc.get("daily_total_insulin"),
                "daily_carbs": pump_doc.get("daily_carbs"),
                "daily_auto_corrections": pump_doc.get("daily_auto_corrections"),
            }
        else:
            response["daily_totals"] = None

    return response


# ===================== Section Helpers =====================

def _get_treatment_ts(t) -> datetime:
    """Get timezone-aware timestamp from a treatment."""
    ts = t.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _format_treatment(t, now: datetime) -> dict:
    """Format a treatment for the API response."""
    ts = _get_treatment_ts(t)
    mins_ago = (now - ts).total_seconds() / 60

    entry: dict[str, Any] = {
        "timestamp": ts.isoformat(),
        "minutes_ago": round(mins_ago, 0),
        "type": t.type if hasattr(t, "type") else "unknown",
    }

    if hasattr(t, "insulin") and t.insulin and t.insulin > 0:
        entry["insulin"] = round(t.insulin, 2)
    if hasattr(t, "carbs") and t.carbs and t.carbs > 0:
        entry["carbs"] = round(t.carbs, 0)
    if hasattr(t, "bolusType") and t.bolusType:
        entry["bolus_type"] = t.bolusType
    if hasattr(t, "notes") and t.notes:
        entry["notes"] = t.notes

    return entry


async def _get_predictions(
    data_user_id: str,
    current_bg: float,
    trend_str: str,
    iob: float,
    cob: float,
    treatments: list,
    glucose_history_data,
) -> Optional[dict]:
    """Get glucose predictions."""
    try:
        from services.prediction_service import get_prediction_service
        pred_service = get_prediction_service()

        trend_num = _trend_to_numeric(trend_str)

        # Build glucose history dicts for prediction service
        gh_dicts = None
        if glucose_history_data:
            gh_dicts = [
                {
                    "value": r.value,
                    "timestamp": (r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc)).isoformat(),
                    "trend": str(r.trend) if r.trend else "Flat",
                }
                for r in glucose_history_data
            ]

        result = pred_service.predict(
            current_bg=current_bg,
            trend=trend_num,
            iob=iob,
            cob=cob,
            glucose_history=gh_dicts,
        )

        pred_response: dict[str, Any] = {
            "method": result.method,
            "generated_at": result.timestamp.isoformat() if result.timestamp else datetime.now(timezone.utc).isoformat(),
        }

        # Linear predictions
        if result.linear:
            pred_response["linear"] = [
                {"horizon_min": h, "value": round(v, 0)}
                for h, v in zip(result.horizons_min, result.linear)
            ]

        # TFT predictions
        if result.tft:
            pred_response["tft"] = [
                {
                    "horizon_min": item.horizon_min,
                    "value": round(item.value, 0),
                    "lower": round(item.lower, 0),
                    "upper": round(item.upper, 0),
                    "confidence": item.confidence,
                }
                for item in result.tft
            ]

        return pred_response

    except Exception as e:
        logger.warning(f"Prediction failed for HA: {e}")
        return None


async def _get_metabolic_params(data_user_id: str) -> Optional[dict]:
    """Get metabolic parameters (ISF, ICR, PIR, state)."""
    try:
        from services.metabolic_params_service import get_metabolic_params_service
        service = get_metabolic_params_service()
        params = await service.get_all_params(data_user_id, is_fasting=False)

        result: dict[str, Any] = {
            "metabolic_state": params.metabolic_state.value if hasattr(params.metabolic_state, "value") else str(params.metabolic_state),
            "state_description": params.state_description,
        }

        # ISF
        if params.isf:
            result.update({
                "isf": round(params.isf.value, 1),
                "isf_baseline": round(params.isf.baseline, 1),
                "isf_deviation_percent": round(params.isf.deviation_percent, 1),
                "isf_source": params.isf.source,
            })

        # ICR
        if params.icr:
            result.update({
                "icr": round(params.icr.value, 1),
                "icr_baseline": round(params.icr.baseline, 1),
                "icr_source": params.icr.source,
            })

        # PIR
        if params.pir:
            result.update({
                "pir": round(params.pir.value, 1),
                "pir_onset_minutes": params.pir.onset_minutes,
                "pir_peak_minutes": params.pir.peak_minutes,
            })

        # Absorption
        if params.absorption:
            result.update({
                "absorption_state": params.absorption.state,
                "absorption_time_to_peak": round(params.absorption.time_to_peak_minutes, 1) if hasattr(params.absorption, "time_to_peak_minutes") else None,
            })

        return result

    except Exception as e:
        logger.warning(f"Metabolic params failed for HA: {e}")
        return None


async def _get_computed(
    data_user_id: str,
    current_bg: float,
    iob: float,
    cob: float,
    treatments: list,
) -> dict:
    """Get computed values: IOB, COB, POB, dose recommendation."""
    pob = iob_cob_service.calculate_pob(treatments) if hasattr(iob_cob_service, "calculate_pob") else 0.0

    result: dict[str, Any] = {
        "iob": round(iob, 2),
        "cob": round(cob, 0),
        "pob": round(pob, 1),
    }

    # Dose recommendation
    try:
        from services.metabolic_params_service import get_metabolic_params_service
        service = get_metabolic_params_service()
        isf_data = await service.get_effective_isf(data_user_id)
        isf = isf_data.value

        recommended_dose, predicted_bg = iob_cob_service.calculate_dose_recommendation(
            current_bg=int(current_bg),
            iob=iob,
            cob=cob,
            isf=isf,
            pob=pob,
        )

        if recommended_dose > 0:
            action_type = "insulin"
            reasoning = f"BG {int(current_bg)} predicted to be {predicted_bg} without action. Recommending {recommended_dose:.1f}u correction."
        elif predicted_bg < 70:
            recommended_carbs = max(0, (70 - predicted_bg) / 3)  # ~3 mg/dL per gram carb
            action_type = "carbs"
            reasoning = f"BG predicted to drop to {predicted_bg}. Recommending {recommended_carbs:.0f}g carbs."
            result["dose_recommendation"] = {
                "recommended_dose": 0.0,
                "recommended_carbs": round(recommended_carbs, 0),
                "current_bg": int(current_bg),
                "predicted_bg_without_action": predicted_bg,
                "target_bg": 100,
                "action_type": action_type,
                "reasoning": reasoning,
            }
            return result
        else:
            action_type = "none"
            reasoning = f"BG {int(current_bg)} predicted at {predicted_bg}. In range, no action needed."

        result["dose_recommendation"] = {
            "recommended_dose": round(recommended_dose, 2),
            "recommended_carbs": 0.0,
            "current_bg": int(current_bg),
            "predicted_bg_without_action": predicted_bg,
            "target_bg": 100,
            "action_type": action_type,
            "reasoning": reasoning,
        }
    except Exception as e:
        logger.warning(f"Dose recommendation failed: {e}")
        result["dose_recommendation"] = None

    return result
