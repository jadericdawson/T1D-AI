"""
Basal Aggregator - Groups 5-min basal events into 60-min treatment windows.

Reduces ~288 events/day to ~24 records, each representing one hour of basal delivery.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from services.tandem.tandem_models import TandemBasalEvent
from services.tandem.t1dai_models import T1DAITreatment

logger = logging.getLogger(__name__)


def aggregate_basal_events(
    events: List[TandemBasalEvent],
    user_id: str,
    window_minutes: int = 60
) -> List[T1DAITreatment]:
    """
    Aggregate 5-min basal events into hourly treatment windows.

    Args:
        events: Raw basal events sorted by timestamp
        user_id: T1D-AI user ID for document creation
        window_minutes: Aggregation window size (default 60 min)

    Returns:
        List of T1DAITreatment objects representing aggregated basal windows
    """
    if not events:
        return []

    treatments = []
    window_start = _floor_to_window(events[0].timestamp, window_minutes)
    window_events: List[TandemBasalEvent] = []

    for event in sorted(events, key=lambda e: e.timestamp):
        event_window = _floor_to_window(event.timestamp, window_minutes)

        if event_window != window_start:
            if window_events:
                treatment = _create_basal_treatment(window_events, window_start, user_id, window_minutes)
                if treatment:
                    treatments.append(treatment)
            window_start = event_window
            window_events = []

        window_events.append(event)

    # Flush last window
    if window_events:
        treatment = _create_basal_treatment(window_events, window_start, user_id, window_minutes)
        if treatment:
            treatments.append(treatment)

    logger.info(
        f"Aggregated {len(events)} basal events into {len(treatments)} "
        f"{window_minutes}-min windows"
    )
    return treatments


def _floor_to_window(ts: datetime, window_minutes: int) -> datetime:
    """Floor timestamp to the start of its aggregation window."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    epoch = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_since_midnight = (ts - epoch).total_seconds() / 60
    window_index = int(minutes_since_midnight // window_minutes)
    return epoch + timedelta(minutes=window_index * window_minutes)


def _create_basal_treatment(
    events: List[TandemBasalEvent],
    window_start: datetime,
    user_id: str,
    window_minutes: int
) -> Optional[T1DAITreatment]:
    """Create a single basal treatment from aggregated events."""
    total_units = sum(e.delivered_units for e in events)
    if total_units <= 0:
        return None

    avg_rate = sum(e.rate for e in events) / len(events)
    has_ciq_adjustment = any(e.delivery_type == "controliq_adjustment" for e in events)

    window_id = f"basal_{int(window_start.timestamp())}"

    notes_parts = [f"Avg rate: {avg_rate:.2f} U/hr"]
    if has_ciq_adjustment:
        notes_parts.append("Control-IQ adjusted")
    notes = ", ".join(notes_parts)

    return T1DAITreatment(
        id=f"{user_id}_tandem_{window_id}",
        userId=user_id,
        timestamp=window_start.isoformat(),
        type="basal",
        insulin=round(total_units, 3),
        source="tandem",
        sourceId=f"tandem_{window_id}",
        basalRate=round(avg_rate, 2),
        deliveryMethod="pump_basal",
        pumpSource="tandem_mobi",
        durationMinutes=window_minutes,
        notes=notes,
    )
