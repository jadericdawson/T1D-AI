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

from services.tandem.tandem_models import (
    TandemBasalEvent, TandemBolusEvent, TandemFetchResult,
    TandemDailyBasalEvent, TandemModeChangeEvent, TandemPcmChangeEvent,
    TandemSuspendEvent, TandemAlertEvent, TandemAlarmEvent,
    TandemCartridgeEvent, TandemSiteChangeEvent, TandemTubingEvent,
    TandemBgReadingEvent, TandemBolusDetailEvent, TandemDailyStatusEvent,
)

logger = logging.getLogger(__name__)

# Milliunit to unit conversion
MU_TO_U = 1 / 1000.0

# Mode maps
USER_MODE_MAP = {0: "Normal", 1: "Sleeping", 2: "Exercising", 3: "EatingSoon"}
PCM_MAP = {0: "NoControl", 1: "OpenLoop", 2: "Pining", 3: "ClosedLoop"}
SUSPEND_REASON_MAP = {0: "User", 1: "Alarm", 2: "Malfunction", 6: "PLGS"}


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

    def fetch_all_expanded(self, since: datetime, until: Optional[datetime] = None
                           ) -> TandemFetchResult:
        """
        Fetch ALL pump event types in one API call.

        Returns TandemFetchResult with all event type lists populated.
        """
        until = until or datetime.now(timezone.utc)
        raw_events = self._fetch_raw_events(since, until)

        # Log event type distribution for debugging
        type_counts = {}
        for e in raw_events:
            name = type(e).__name__
            type_counts[name] = type_counts.get(name, 0) + 1
        logger.info(f"Event type distribution: {type_counts}")

        return TandemFetchResult(
            basal_events=self._extract_basal_events(raw_events, since, until),
            bolus_events=self._extract_bolus_events(raw_events, since, until),
            daily_basal_events=self._extract_daily_basal(raw_events),
            mode_changes=self._extract_mode_changes(raw_events),
            pcm_changes=self._extract_pcm_changes(raw_events),
            suspend_events=self._extract_suspends(raw_events),
            alerts=self._extract_alerts(raw_events),
            alarms=self._extract_alarms(raw_events),
            cartridge_events=self._extract_cartridge(raw_events),
            site_changes=self._extract_site_changes(raw_events),
            tubing_events=self._extract_tubing(raw_events),
            bg_readings=self._extract_bg_readings(raw_events),
            bolus_details=self._extract_bolus_details(raw_events),
            daily_status=self._extract_daily_status(raw_events),
        )

    def _extract_basal_events(self, raw_events: list, since: datetime, until: datetime) -> List[TandemBasalEvent]:
        """Extract basal delivery events from raw pump events.

        Note: The Tandem API returns events with timestamps that lag hours behind
        the requested min_date/max_date window. We trust the API's date filtering
        and do NOT re-filter by since/until here — CosmosDB upserts handle dedup.
        """
        events = []
        basal_raw = [e for e in raw_events if type(e).__name__ == "LidBasalDelivery"]

        for i, entry in enumerate(basal_raw):
            try:
                d = entry.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
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
        """Extract bolus events from raw pump events.

        Note: The Tandem API returns events with timestamps that lag hours behind
        the requested min_date/max_date window. We trust the API's date filtering
        and do NOT re-filter by since/until here — CosmosDB upserts handle dedup.
        """
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
                if ts is None:
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

    # ===================== Expanded Extraction Methods =====================

    def _extract_daily_basal(self, raw_events: list) -> List[TandemDailyBasalEvent]:
        """Extract daily basal summaries with battery and IOB (LidDailyBasal)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidDailyBasal":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue

                # Battery: try direct properties first, then compute from MSB/LSB
                battery_pct = None
                battery_mv = None
                msb = d.get("batteryLevelMsb") or d.get("batterylevelmsb")
                lsb = d.get("batteryLevelLsb") or d.get("batterylevellsb")
                if msb is not None and lsb is not None:
                    try:
                        raw_mv = 256 * int(msb) + int(lsb)
                        battery_mv = raw_mv
                        # Tandem Mobi: 3600mV = ~0%, 4200mV = ~100%
                        battery_pct = max(0.0, min(100.0, (raw_mv - 3600) / 6.0))
                    except (ValueError, TypeError):
                        pass

                # Daily totals (in milliUnits)
                daily_basal_mu = d.get("dailyTotalBasal", d.get("dailytotalbasal", 0)) or 0
                daily_bolus_mu = d.get("dailyTotalBolus", d.get("dailytotalbolus", 0)) or 0
                pump_iob_mu = d.get("iob", 0) or 0

                events.append(TandemDailyBasalEvent(
                    timestamp=ts,
                    battery_percent=round(battery_pct, 1) if battery_pct is not None else None,
                    battery_millivolts=battery_mv,
                    daily_basal_units=round(daily_basal_mu * MU_TO_U, 2) if daily_basal_mu else None,
                    daily_bolus_units=round(daily_bolus_mu * MU_TO_U, 2) if daily_bolus_mu else None,
                    daily_total_insulin=round((daily_basal_mu + daily_bolus_mu) * MU_TO_U, 2) if (daily_basal_mu or daily_bolus_mu) else None,
                    pump_iob=round(pump_iob_mu * MU_TO_U, 2) if pump_iob_mu else None,
                    event_id=f"daily_basal_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse daily basal event: {ex}")
        logger.info(f"Extracted {len(events)} daily basal events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_mode_changes(self, raw_events: list) -> List[TandemModeChangeEvent]:
        """Extract Control-IQ mode changes (LidAaUserModeChange)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidAaUserModeChange":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                prev_raw = d.get("previoususermodeRaw", d.get("previoususermode", 0))
                curr_raw = d.get("currentusermodeRaw", d.get("currentusermode", 0))
                events.append(TandemModeChangeEvent(
                    timestamp=ts,
                    previous_mode=USER_MODE_MAP.get(int(prev_raw), f"Unknown({prev_raw})"),
                    current_mode=USER_MODE_MAP.get(int(curr_raw), f"Unknown({curr_raw})"),
                    event_id=f"mode_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse mode change: {ex}")
        logger.info(f"Extracted {len(events)} mode change events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_pcm_changes(self, raw_events: list) -> List[TandemPcmChangeEvent]:
        """Extract pump control mode changes (LidAaPcmChange)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidAaPcmChange":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                prev_raw = d.get("previouspcmRaw", d.get("previouspcm", 0))
                curr_raw = d.get("currentpcmRaw", d.get("currentpcm", 0))
                events.append(TandemPcmChangeEvent(
                    timestamp=ts,
                    previous_pcm=PCM_MAP.get(int(prev_raw), f"Unknown({prev_raw})"),
                    current_pcm=PCM_MAP.get(int(curr_raw), f"Unknown({curr_raw})"),
                    event_id=f"pcm_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse PCM change: {ex}")
        logger.info(f"Extracted {len(events)} PCM change events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_suspends(self, raw_events: list) -> List[TandemSuspendEvent]:
        """Extract pump suspend/resume events."""
        events = []
        for e in raw_events:
            name = type(e).__name__
            if name not in ("LidPumpingSuspended", "LidPumpingResumed"):
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                action = "suspended" if name == "LidPumpingSuspended" else "resumed"
                reason = None
                if action == "suspended":
                    reason_raw = d.get("suspendreasonRaw", d.get("suspendreason"))
                    if reason_raw is not None:
                        reason = SUSPEND_REASON_MAP.get(int(reason_raw), f"Unknown({reason_raw})")
                events.append(TandemSuspendEvent(
                    timestamp=ts,
                    action=action,
                    reason=reason,
                    event_id=f"suspend_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse suspend event: {ex}")
        logger.info(f"Extracted {len(events)} suspend/resume events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_alerts(self, raw_events: list) -> List[TandemAlertEvent]:
        """Extract pump alerts (LidAlertActivated)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidAlertActivated":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                alert_id = d.get("alertidRaw", d.get("alertid"))
                alert_type = d.get("alertid", d.get("alertidRaw", "unknown"))
                events.append(TandemAlertEvent(
                    timestamp=ts,
                    alert_type=str(alert_type),
                    alert_id=int(alert_id) if alert_id is not None else None,
                    event_id=f"alert_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse alert: {ex}")
        logger.info(f"Extracted {len(events)} alert events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_alarms(self, raw_events: list) -> List[TandemAlarmEvent]:
        """Extract pump alarms (LidAlarmActivated)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidAlarmActivated":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                alarm_id = d.get("alarmidRaw", d.get("alarmid"))
                alarm_type = d.get("alarmid", d.get("alarmidRaw", "unknown"))
                events.append(TandemAlarmEvent(
                    timestamp=ts,
                    alarm_type=str(alarm_type),
                    alarm_id=int(alarm_id) if alarm_id is not None else None,
                    event_id=f"alarm_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse alarm: {ex}")
        logger.info(f"Extracted {len(events)} alarm events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_cartridge(self, raw_events: list) -> List[TandemCartridgeEvent]:
        """Extract cartridge fill events (LidCartridgeFilled)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidCartridgeFilled":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                volume_mu = d.get("insulinVolume", d.get("insulinvolume", 0)) or 0
                events.append(TandemCartridgeEvent(
                    timestamp=ts,
                    volume=round(volume_mu * MU_TO_U, 1) if volume_mu else None,
                    event_id=f"cartridge_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse cartridge event: {ex}")
        logger.info(f"Extracted {len(events)} cartridge events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_site_changes(self, raw_events: list) -> List[TandemSiteChangeEvent]:
        """Extract cannula/site change events (LidCannulaFilled)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidCannulaFilled":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                prime_mu = d.get("primeSize", d.get("primesize", 0)) or 0
                events.append(TandemSiteChangeEvent(
                    timestamp=ts,
                    prime_volume=round(prime_mu * MU_TO_U, 3) if prime_mu else None,
                    event_id=f"site_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse site change: {ex}")
        logger.info(f"Extracted {len(events)} site change events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_tubing(self, raw_events: list) -> List[TandemTubingEvent]:
        """Extract tubing fill events (LidTubingFilled)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidTubingFilled":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                volume_mu = d.get("primeSize", d.get("primesize", 0)) or 0
                events.append(TandemTubingEvent(
                    timestamp=ts,
                    volume=round(volume_mu * MU_TO_U, 3) if volume_mu else None,
                    event_id=f"tubing_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse tubing event: {ex}")
        logger.info(f"Extracted {len(events)} tubing events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_bg_readings(self, raw_events: list) -> List[TandemBgReadingEvent]:
        """Extract manual BG readings from pump (LidBgReadingTaken)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidBgReadingTaken":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                bg = d.get("bgReading", d.get("bgreading", 0))
                if bg and int(bg) > 0:
                    events.append(TandemBgReadingEvent(
                        timestamp=ts,
                        bg_value=int(bg),
                        event_id=f"bg_{d.get('seqNum', int(ts.timestamp()))}",
                    ))
            except Exception as ex:
                logger.warning(f"Failed to parse BG reading: {ex}")
        logger.info(f"Extracted {len(events)} BG reading events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_bolus_details(self, raw_events: list) -> List[TandemBolusDetailEvent]:
        """Extract detailed bolus breakdown from Msg2+Msg3 by correlating on bolusid."""
        msg2_data = {}
        msg3_data = {}
        for e in raw_events:
            name = type(e).__name__
            if name == "LidBolusRequestedMsg2":
                d = e.todict()
                msg2_data[d.get("bolusid")] = d
            elif name == "LidBolusRequestedMsg3":
                d = e.todict()
                msg3_data[d.get("bolusid")] = d

        events = []
        all_ids = set(msg2_data.keys()) | set(msg3_data.keys())
        for bolus_id in all_ids:
            try:
                m2 = msg2_data.get(bolus_id, {})
                m3 = msg3_data.get(bolus_id, {})
                # Timestamp from whichever message we have
                ts_raw = m2.get("eventTimestamp") or m3.get("eventTimestamp")
                ts = _parse_event_timestamp(ts_raw)
                if ts is None:
                    continue

                food_mu = m2.get("foodBolus", m2.get("foodbolus", 0)) or 0
                correction_mu = m2.get("correctionBolus", m2.get("correctionbolus", 0)) or 0
                isf_val = m3.get("isf", m3.get("iSF", 0)) or 0
                target_bg = m3.get("targetBg", m3.get("targetbg", 0)) or 0
                current_bg_val = m3.get("bgReading", m3.get("bgreading", 0)) or 0
                icr_val = m3.get("icr", m3.get("iCR", 0)) or 0

                events.append(TandemBolusDetailEvent(
                    timestamp=ts,
                    bolus_id=int(bolus_id) if bolus_id is not None else 0,
                    food_insulin=round(food_mu * MU_TO_U, 3) if food_mu else None,
                    correction_insulin=round(correction_mu * MU_TO_U, 3) if correction_mu else None,
                    isf=round(isf_val, 1) if isf_val else None,
                    target_bg=round(target_bg, 0) if target_bg else None,
                    current_bg=round(current_bg_val, 0) if current_bg_val else None,
                    icr=round(icr_val, 1) if icr_val else None,
                    event_id=f"bolus_detail_{bolus_id}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse bolus detail for id {bolus_id}: {ex}")
        logger.info(f"Extracted {len(events)} bolus detail events")
        return sorted(events, key=lambda e: e.timestamp)

    def _extract_daily_status(self, raw_events: list) -> List[TandemDailyStatusEvent]:
        """Extract daily algorithm status (LidAaDailyStatus)."""
        events = []
        for e in raw_events:
            if type(e).__name__ != "LidAaDailyStatus":
                continue
            try:
                d = e.todict()
                ts = _parse_event_timestamp(d.get("eventTimestamp"))
                if ts is None:
                    continue
                auto_corrections = d.get("numAutoBolus", d.get("numAutobolus"))
                sensor = d.get("sensorType", d.get("sensortype"))
                events.append(TandemDailyStatusEvent(
                    timestamp=ts,
                    auto_corrections_today=int(auto_corrections) if auto_corrections is not None else None,
                    sensor_type=str(sensor) if sensor is not None else None,
                    event_id=f"daily_status_{d.get('seqNum', int(ts.timestamp()))}",
                ))
            except Exception as ex:
                logger.warning(f"Failed to parse daily status: {ex}")
        logger.info(f"Extracted {len(events)} daily status events")
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
