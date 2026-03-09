"""
Reports API Endpoints

Generates comprehensive pump/diabetes data reports with deduplication,
timezone-aware daily grouping, and pre-computed statistics for frontend rendering.
"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from database.repositories import GlucoseRepository, TreatmentRepository
from auth.routes import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])

glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()


def get_data_user_id(profile_id: str) -> str:
    if profile_id.startswith("profile_"):
        return profile_id[8:]
    return profile_id


# ── Response Models ──────────────────────────────────────────────────────

class DailySummary(BaseModel):
    date: str
    basal_units: float = 0
    bolus_units: float = 0
    bolus_count: int = 0
    auto_correction_units: float = 0
    auto_correction_count: int = 0
    total_insulin: float = 0
    carbs: float = 0
    meal_count: int = 0
    avg_bg: Optional[float] = None
    tir: Optional[float] = None
    time_low: Optional[float] = None
    time_high: Optional[float] = None
    readings_count: int = 0


class HourlyPattern(BaseModel):
    hour: int
    avg_basal_rate: float = 0
    avg_glucose: Optional[float] = None
    min_glucose: Optional[int] = None
    max_glucose: Optional[int] = None
    auto_correction_insulin: float = 0


class BolusEvent(BaseModel):
    time: str
    type: str
    units: float
    notes: str = ""


class MealEvent(BaseModel):
    time: str
    carbs: float
    notes: str = ""


class GlucosePoint(BaseModel):
    time: str
    value: int


class TirBreakdown(BaseModel):
    very_low: float = 0
    low: float = 0
    in_range: float = 0
    high: float = 0
    very_high: float = 0


class InsulinSplit(BaseModel):
    basal_total: float = 0
    basal_pct: float = 0
    bolus_total: float = 0
    bolus_pct: float = 0
    auto_total: float = 0
    auto_pct: float = 0


class PumpStatus(BaseModel):
    battery_percent: Optional[float] = None
    control_mode: Optional[str] = None
    pump_iob: Optional[float] = None
    daily_basal_units: Optional[float] = None
    daily_bolus_units: Optional[float] = None
    daily_total_insulin: Optional[float] = None


class ReportData(BaseModel):
    period_label: str
    start_date: str
    end_date: str
    total_days: int
    timezone: str

    # Overview stats
    avg_daily_insulin: float = 0
    avg_daily_carbs: float = 0
    avg_glucose: Optional[float] = None
    gmi: Optional[float] = None
    cv: Optional[float] = None
    total_readings: int = 0
    total_auto_corrections: int = 0
    avg_daily_auto_corrections: float = 0

    # Breakdowns
    tir: TirBreakdown = Field(default_factory=TirBreakdown)
    insulin_split: InsulinSplit = Field(default_factory=InsulinSplit)
    pump_status: Optional[PumpStatus] = None

    # Time series
    daily_summaries: List[DailySummary] = []
    hourly_patterns: List[HourlyPattern] = []
    glucose_trace: List[GlucosePoint] = []
    bolus_events: List[BolusEvent] = []
    meal_events: List[MealEvent] = []


# ── Deduplication ────────────────────────────────────────────────────────

def _dedup_key(t: dict) -> tuple:
    """Create a dedup key from treatment based on type, time bucket, and value."""
    ts_str = t.get('timestamp', '')
    try:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return (ts_str, '', 0)

    bucket = ts.replace(second=0, microsecond=0)
    bucket = bucket.replace(minute=(bucket.minute // 2) * 2)

    carbs = float(t.get('carbs') or 0)
    insulin = float(t.get('insulin') or 0)
    ttype = t.get('type', '')

    if carbs > 0:
        return (bucket.isoformat(), 'carbs', round(carbs))
    elif insulin > 0:
        return (bucket.isoformat(), ttype, round(insulin, 1))
    else:
        return (bucket.isoformat(), ttype, 0)


def _dedup_treatments(treatments: list) -> list:
    """Deduplicate treatments across sources. Prefer tandem > manual > enriched > first."""
    groups: Dict[tuple, list] = defaultdict(list)
    for t in treatments:
        groups[_dedup_key(t)].append(t)

    deduped = []
    for group in groups.values():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        tandem = [t for t in group if t.get('source') == 'tandem']
        manual = [t for t in group if t.get('source') == 'manual']
        enriched = [t for t in group if t.get('glycemicIndex')]

        if tandem:
            best = tandem[0]
        elif manual:
            best = manual[0]
        elif enriched:
            best = enriched[0]
        else:
            best = group[0]
        deduped.append(best)

    deduped.sort(key=lambda t: t.get('timestamp', ''))
    return deduped


# ── Report Generation ────────────────────────────────────────────────────

PERIOD_MAP = {
    "24h": 1,
    "3d": 3,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "1y": 365,
}


def _build_report(
    treatments_raw: list,
    glucose_raw: list,
    period: str,
    tz_offset_hours: float,
    pump_status_doc: Optional[dict],
) -> ReportData:
    """Build the full report from raw CosmosDB documents."""
    local_tz = timezone(timedelta(hours=tz_offset_hours))

    # Deduplicate treatments
    treatments = _dedup_treatments(treatments_raw)

    # Parse timestamps to local
    for t in treatments:
        try:
            t['_dt'] = datetime.fromisoformat(
                t['timestamp'].replace('Z', '+00:00')
            ).astimezone(local_tz)
        except (ValueError, TypeError):
            t['_dt'] = None

    for g in glucose_raw:
        try:
            g['_dt'] = datetime.fromisoformat(
                g['timestamp'].replace('Z', '+00:00')
            ).astimezone(local_tz)
        except (ValueError, TypeError):
            g['_dt'] = None

    treatments = [t for t in treatments if t.get('_dt')]
    glucose_raw = [g for g in glucose_raw if g.get('_dt')]

    # ── Daily summaries ──────────────────────────────────────────
    daily: Dict[str, dict] = defaultdict(lambda: {
        'basal_units': 0, 'bolus_units': 0, 'auto_correction_units': 0,
        'carbs': 0, 'meal_count': 0, 'auto_correction_count': 0,
        'bolus_count': 0, 'glucose_values': [], 'basal_rates': [],
    })

    total_basal = 0
    total_bolus = 0
    total_auto = 0
    total_carbs = 0
    total_auto_count = 0

    for t in treatments:
        day = t['_dt'].strftime('%Y-%m-%d')
        d = daily[day]
        ttype = t.get('type', '')
        insulin = float(t.get('insulin') or 0)
        carbs = float(t.get('carbs') or 0)

        if ttype == 'basal':
            d['basal_units'] += insulin
            total_basal += insulin
            if t.get('basalRate'):
                d['basal_rates'].append(float(t['basalRate']))
        elif ttype == 'auto_correction':
            d['auto_correction_units'] += insulin
            d['auto_correction_count'] += 1
            total_auto += insulin
            total_auto_count += 1
        elif ttype == 'insulin':
            d['bolus_units'] += insulin
            d['bolus_count'] += 1
            total_bolus += insulin
        elif ttype == 'carbs':
            d['carbs'] += carbs
            d['meal_count'] += 1
            total_carbs += carbs

    for g in glucose_raw:
        day = g['_dt'].strftime('%Y-%m-%d')
        daily[day]['glucose_values'].append(int(g.get('value', 0)))

    sorted_days = sorted(daily.keys())
    total_days = max(len(sorted_days), 1)
    total_insulin = total_basal + total_bolus + total_auto

    daily_summaries = []
    for day in sorted_days:
        d = daily[day]
        gv = d['glucose_values']
        avg_bg = sum(gv) / len(gv) if gv else None
        tir = sum(1 for v in gv if 70 <= v <= 180) / len(gv) * 100 if gv else None
        t_low = sum(1 for v in gv if v < 70) / len(gv) * 100 if gv else None
        t_high = sum(1 for v in gv if v > 180) / len(gv) * 100 if gv else None

        daily_summaries.append(DailySummary(
            date=day,
            basal_units=round(d['basal_units'], 2),
            bolus_units=round(d['bolus_units'], 2),
            bolus_count=d['bolus_count'],
            auto_correction_units=round(d['auto_correction_units'], 2),
            auto_correction_count=d['auto_correction_count'],
            total_insulin=round(d['basal_units'] + d['bolus_units'] + d['auto_correction_units'], 2),
            carbs=round(d['carbs']),
            meal_count=d['meal_count'],
            avg_bg=round(avg_bg) if avg_bg else None,
            tir=round(tir, 1) if tir is not None else None,
            time_low=round(t_low, 1) if t_low is not None else None,
            time_high=round(t_high, 1) if t_high is not None else None,
            readings_count=len(gv),
        ))

    # ── Glucose stats ────────────────────────────────────────────
    all_bg = [int(g.get('value', 0)) for g in glucose_raw if g.get('value')]
    avg_bg = sum(all_bg) / len(all_bg) if all_bg else None
    gmi_val = (3.31 + 0.02392 * avg_bg) if avg_bg else None
    cv_val = ((sum((v - avg_bg) ** 2 for v in all_bg) / len(all_bg)) ** 0.5 / avg_bg * 100) if avg_bg and avg_bg > 0 else None

    tir = TirBreakdown()
    if all_bg:
        n = len(all_bg)
        tir.very_low = round(sum(1 for v in all_bg if v < 54) / n * 100, 1)
        tir.low = round(sum(1 for v in all_bg if 54 <= v < 70) / n * 100, 1)
        tir.in_range = round(sum(1 for v in all_bg if 70 <= v <= 180) / n * 100, 1)
        tir.high = round(sum(1 for v in all_bg if 180 < v <= 250) / n * 100, 1)
        tir.very_high = round(sum(1 for v in all_bg if v > 250) / n * 100, 1)

    # ── Insulin split ────────────────────────────────────────────
    insulin_split = InsulinSplit()
    if total_insulin > 0:
        insulin_split.basal_total = round(total_basal / total_days, 1)
        insulin_split.basal_pct = round(total_basal / total_insulin * 100)
        insulin_split.bolus_total = round(total_bolus / total_days, 1)
        insulin_split.bolus_pct = round(total_bolus / total_insulin * 100)
        insulin_split.auto_total = round(total_auto / total_days, 1)
        insulin_split.auto_pct = round(total_auto / total_insulin * 100)

    # ── Hourly patterns ──────────────────────────────────────────
    hourly_basal: Dict[int, list] = defaultdict(list)
    hourly_auto: Dict[int, list] = defaultdict(list)
    hourly_glucose: Dict[int, list] = defaultdict(list)

    for t in treatments:
        hour = t['_dt'].hour
        if t.get('type') == 'basal' and t.get('basalRate'):
            hourly_basal[hour].append(float(t['basalRate']))
        elif t.get('type') == 'auto_correction':
            hourly_auto[hour].append(float(t.get('insulin') or 0))

    for g in glucose_raw:
        hourly_glucose[g['_dt'].hour].append(int(g.get('value', 0)))

    hourly_patterns = []
    for h in range(24):
        br = hourly_basal.get(h, [])
        gl = hourly_glucose.get(h, [])
        au = hourly_auto.get(h, [])
        hourly_patterns.append(HourlyPattern(
            hour=h,
            avg_basal_rate=round(sum(br) / len(br), 2) if br else 0,
            avg_glucose=round(sum(gl) / len(gl)) if gl else None,
            min_glucose=min(gl) if gl else None,
            max_glucose=max(gl) if gl else None,
            auto_correction_insulin=round(sum(au) / max(total_days, 1), 2),
        ))

    # ── Glucose trace (downsample for performance) ───────────────
    # For 24h: every point. 3d: every 2nd. 7d: every 3rd. 30d+: every 6th.
    days_count = PERIOD_MAP.get(period, 7)
    step = 1 if days_count <= 1 else (2 if days_count <= 3 else (3 if days_count <= 7 else 6))
    glucose_trace = [
        GlucosePoint(time=g['_dt'].strftime('%m/%d %I:%M %p'), value=int(g.get('value', 0)))
        for g in glucose_raw[::step]
    ]

    # ── Bolus events ─────────────────────────────────────────────
    bolus_events = [
        BolusEvent(
            time=t['_dt'].strftime('%m/%d %I:%M %p'),
            type=t.get('bolusType', t.get('type', '?')),
            units=round(float(t.get('insulin') or 0), 1),
            notes=(t.get('notes') or '')[:60],
        )
        for t in treatments
        if t.get('type') in ('insulin', 'auto_correction')
    ]

    # ── Meal events ──────────────────────────────────────────────
    meal_events = [
        MealEvent(
            time=t['_dt'].strftime('%m/%d %I:%M %p'),
            carbs=round(float(t.get('carbs') or 0)),
            notes=(t.get('notes') or '')[:60],
        )
        for t in treatments
        if t.get('type') == 'carbs' and float(t.get('carbs') or 0) > 0
    ]

    # ── Pump status ──────────────────────────────────────────────
    ps = None
    if pump_status_doc:
        ps = PumpStatus(
            battery_percent=pump_status_doc.get('battery_percent'),
            control_mode=pump_status_doc.get('control_mode'),
            pump_iob=pump_status_doc.get('pump_iob'),
            daily_basal_units=pump_status_doc.get('daily_basal_units'),
            daily_bolus_units=pump_status_doc.get('daily_bolus_units'),
            daily_total_insulin=pump_status_doc.get('daily_total_insulin'),
        )

    # ── Assemble ─────────────────────────────────────────────────
    tz_label = f"UTC{tz_offset_hours:+.0f}" if tz_offset_hours != int(tz_offset_hours) else f"UTC{int(tz_offset_hours):+d}"

    return ReportData(
        period_label=period,
        start_date=sorted_days[0] if sorted_days else "",
        end_date=sorted_days[-1] if sorted_days else "",
        total_days=total_days,
        timezone=tz_label,
        avg_daily_insulin=round(total_insulin / total_days, 1) if total_days else 0,
        avg_daily_carbs=round(total_carbs / total_days) if total_days else 0,
        avg_glucose=round(avg_bg) if avg_bg else None,
        gmi=round(gmi_val, 1) if gmi_val else None,
        cv=round(cv_val) if cv_val else None,
        total_readings=len(all_bg),
        total_auto_corrections=total_auto_count,
        avg_daily_auto_corrections=round(total_auto_count / total_days, 1) if total_days else 0,
        tir=tir,
        insulin_split=insulin_split,
        pump_status=ps,
        daily_summaries=daily_summaries,
        hourly_patterns=hourly_patterns,
        glucose_trace=glucose_trace,
        bolus_events=bolus_events,
        meal_events=meal_events,
    )


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.get("/pump-report", response_model=ReportData)
async def get_pump_report(
    period: str = Query(default="7d", pattern="^(24h|3d|7d|30d|90d|1y)$"),
    tz_offset: float = Query(default=-4, description="Timezone offset from UTC in hours (e.g. -4 for EDT)"),
    current_user=Depends(get_current_user),
):
    """
    Generate a comprehensive pump data report.

    Returns deduplicated, timezone-aware data including daily summaries,
    hourly patterns, glucose trace, bolus events, and meal log.
    """
    try:
        data_user_id = get_data_user_id(current_user.id)
        days = PERIOD_MAP.get(period, 7)

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)

        # Fetch larger limits for longer periods
        glucose_limit = min(days * 288, 50000)  # ~288 readings/day
        treatment_limit = min(days * 100, 20000)

        glucose_raw = await glucose_repo.get_history(
            data_user_id, start, now, limit=glucose_limit
        )
        treatments_raw = await treatment_repo.get_by_user(
            data_user_id, start, now, limit=treatment_limit
        )

        # Convert to dicts for dedup processing
        glucose_dicts = [g.model_dump(mode='json') for g in glucose_raw]
        treatment_dicts = [t.model_dump(mode='json') for t in treatments_raw]

        # Get pump status
        pump_status_doc = None
        try:
            from database.cosmos_manager import get_cosmos_manager
            manager = get_cosmos_manager()
            ps_container = manager.get_container("pump_status")
            items = list(ps_container.query_items(
                query="SELECT * FROM c WHERE c.userId = @uid",
                parameters=[{"name": "@uid", "value": data_user_id}],
                partition_key=data_user_id,
                max_item_count=1,
            ))
            if items:
                pump_status_doc = items[0]
        except Exception as e:
            logger.warning(f"Could not fetch pump status: {e}")

        report = _build_report(
            treatment_dicts, glucose_dicts, period, tz_offset, pump_status_doc
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating pump report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
