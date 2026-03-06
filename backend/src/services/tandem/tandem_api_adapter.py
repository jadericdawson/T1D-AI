"""
Tandem API Adapter using tconnectsync library.

Uses the TandemSource API (pump_events) to fetch basal, bolus, and carb events
from the Tandem Source portal.

Data format notes (from real Tandem Mobi):
- Basal rates: stored in milliUnits/hr (350 = 0.350 U/hr)
- Bolus insulin: stored in milliUnits (530 = 0.53 U)
- commandedRateSourceRaw 3 = algorithm (Control-IQ)
- bolustypeRaw in request: 2 = auto-correction, 3 = user-initiated
- bolusSourceRaw: 7 = CIQ algorithm, 8 = user/app
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from tconnectsync.api import TConnectApi

from services.tandem.tandem_models import TandemBasalEvent, TandemBolusEvent

logger = logging.getLogger(__name__)

# Milliunit to unit conversion
MU_TO_U = 1 / 1000.0


class TandemApiAdapter:
    """Adapter that uses tconnectsync TandemSource API to pull pump data."""

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self._api: Optional[TConnectApi] = None
        self._device_id: Optional[int] = None

    def _get_api(self) -> TConnectApi:
        """Lazy-initialize and authenticate with Tandem API."""
        if self._api is None:
            self._api = TConnectApi(self.email, self.password)
            logger.info("Authenticated with Tandem Source API")
        return self._api

    def _get_device_id(self) -> int:
        """Get the tconnect device ID for the pump."""
        if self._device_id is not None:
            return self._device_id

        api = self._get_api()
        ts = api.tandemsource
        metadata = ts.pump_event_metadata()

        if not metadata:
            raise RuntimeError("No pump devices found in Tandem Source")

        # Use the first (most recent) device
        self._device_id = metadata[0]["tconnectDeviceId"]
        serial = metadata[0].get("serialNumber", "unknown")
        model = metadata[0].get("modelNumber", "unknown")
        logger.info(f"Using pump device: {serial} (model {model}, id {self._device_id})")
        return self._device_id

    def _fetch_raw_events(self, since: datetime, until: datetime) -> list:
        """Fetch all pump events in the time range."""
        api = self._get_api()
        ts = api.tandemsource
        device_id = self._get_device_id()

        raw_events = list(ts.pump_events(
            device_id,
            min_date=since,
            max_date=until,
            fetch_all_event_types=True
        ))
        logger.info(f"Fetched {len(raw_events)} raw pump events")
        return raw_events

    def fetch_all(self, since: datetime, until: Optional[datetime] = None
                  ) -> Tuple[List[TandemBasalEvent], List[TandemBolusEvent]]:
        """
        Fetch basal and bolus events in one API call (more efficient).

        Returns (basal_events, bolus_events).
        """
        until = until or datetime.now(timezone.utc)
        raw_events = self._fetch_raw_events(since, until)

        basal_events = self._extract_basal_events(raw_events, since, until)
        bolus_events = self._extract_bolus_events(raw_events, since, until)

        return basal_events, bolus_events

    def _extract_basal_events(self, raw_events: list, since: datetime, until: datetime) -> List[TandemBasalEvent]:
        """Extract basal delivery events from raw pump events."""
        events = []
        basal_raw = [e for e in raw_events if type(e).__name__ == "LidBasalDelivery"]

        for i, entry in enumerate(basal_raw):
            try:
                d = entry.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None or not (since <= ts <= until):
                    continue

                commanded_rate_mu = d.get("commandedRate", 0)
                profile_rate_mu = d.get("profileBasalRate", 0)

                rate_u_hr = commanded_rate_mu * MU_TO_U

                # Calculate duration: time until next basal event (typically 5 min)
                duration_sec = 300  # default 5 min
                if i + 1 < len(basal_raw):
                    next_d = basal_raw[i + 1].todict()
                    next_ts = _parse_event_timestamp(next_d.get("eventTimestamp"))
                    if next_ts:
                        diff = (next_ts - ts).total_seconds()
                        if 0 < diff <= 900:  # Cap at 15 min
                            duration_sec = int(diff)

                delivered_units = rate_u_hr * (duration_sec / 3600)

                # Classify delivery type
                is_ciq = (commanded_rate_mu != profile_rate_mu) or d.get("commandedRateSourceRaw") == 3
                delivery_type = "controliq_adjustment" if is_ciq and commanded_rate_mu != profile_rate_mu else "profile"

                events.append(TandemBasalEvent(
                    timestamp=ts,
                    duration_seconds=duration_sec,
                    rate=round(rate_u_hr, 3),
                    delivered_units=round(delivered_units, 4),
                    delivery_type=delivery_type,
                    event_id=f"basal_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as e:
                logger.warning(f"Failed to parse basal event: {e}")
                continue

        logger.info(f"Extracted {len(events)} basal events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_bolus_events(self, raw_events: list, since: datetime, until: datetime) -> List[TandemBolusEvent]:
        """Extract bolus events from raw pump events."""
        # Index request data by bolusid for correlation
        request_data = {}
        for e in raw_events:
            if type(e).__name__ == "LidBolusRequestedMsg1":
                d = e.todict()
                request_data[d.get("bolusid")] = d

        # Index carbs entered by timestamp proximity
        carb_events = {}
        for e in raw_events:
            if type(e).__name__ == "LidCarbsEntered":
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts:
                    carb_events[d.get("seqNum")] = {"timestamp": ts, "carbs": d.get("carbs", 0)}

        events = []
        completed = [e for e in raw_events if type(e).__name__ == "LidBolusCompleted"]

        for entry in completed:
            try:
                d = entry.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None or not (since <= ts <= until):
                    continue

                insulin_u = d.get("insulindelivered", 0)
                if not isinstance(insulin_u, (int, float)) or insulin_u <= 0:
                    continue

                bolus_id = d.get("bolusid")
                req = request_data.get(bolus_id, {})

                bolus_type_raw = req.get("bolustypeRaw", 0)
                correction_included = req.get("correctionbolusincludedRaw", 0)
                carbs_from_calc = req.get("carbamount", 0)

                if bolus_type_raw == 2:
                    bolus_type = "auto_correction"
                elif carbs_from_calc and carbs_from_calc > 0:
                    bolus_type = "standard"
                elif correction_included:
                    bolus_type = "standard"
                else:
                    bolus_type = "standard"

                requested_u = d.get("insulinrequested", 0)
                status_raw = d.get("completionstatusRaw", 0)
                completion = "completed" if status_raw == 3 else "incomplete"

                events.append(TandemBolusEvent(
                    timestamp=ts,
                    insulin=round(insulin_u, 2),
                    bolus_type=bolus_type,
                    duration_seconds=0,
                    carbs=float(carbs_from_calc) if carbs_from_calc and carbs_from_calc > 0 else None,
                    event_id=f"bolus_{bolus_id}_{d.get('seqNum', '')}",
                    completion_status=completion,
                    requested_insulin=round(requested_u, 2) if requested_u else None,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse bolus event: {e}")
                continue

        logger.info(f"Extracted {len(events)} bolus events")
        return sorted(events, key=lambda e: e.timestamp)


def _parse_event_timestamp(value) -> Optional[datetime]:
    """Parse eventTimestamp from pump events and normalize to UTC."""
    if value is None:
        return None
    dt = None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
    if dt is None:
        return None
    return dt.astimezone(timezone.utc)
