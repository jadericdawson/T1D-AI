"""
Tandem Sync Service - Fetches pump data from Tandem, writes to CosmosDB and Gluroo.

Data flow:
  All treatments:   Tandem API -> CosmosDB (direct write)
  Bolus + Carbs:    Tandem API -> Gluroo (Nightscout API) as Correction Bolus / Carb Correction
  Basal rates:      Tandem API -> Gluroo (Nightscout API) as Temp Basal (~24/day hourly)
  Pump status:      Tandem API -> CosmosDB pump_status container (battery, mode, alerts, etc.)

Gluroo sync skips entries with enteredBy='tandem-sync' to avoid duplicates,
so all Tandem data must be written directly to CosmosDB.
"""
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import httpx

from services.tandem import (
    TandemApiAdapter,
    aggregate_basal_events,
    map_bolus_events,
)
from services.tandem.t1dai_models import T1DAITreatment

logger = logging.getLogger(__name__)


class TandemSyncService:
    """Fetches Tandem pump data. Writes all treatments to CosmosDB, pushes bolus/carbs/basal to Gluroo."""

    def __init__(self):
        self._treatment_repo = None
        self._pump_status_repo = None

    @property
    def treatment_repo(self):
        if self._treatment_repo is None:
            from database.repositories import TreatmentRepository
            self._treatment_repo = TreatmentRepository()
        return self._treatment_repo

    @property
    def pump_status_repo(self):
        if self._pump_status_repo is None:
            from database.repositories import PumpStatusRepository
            self._pump_status_repo = PumpStatusRepository()
        return self._pump_status_repo

    async def sync_for_source(
        self,
        email: str,
        password: str,
        user_id: str,
        since: datetime,
        until: Optional[datetime] = None,
        gluroo_url: Optional[str] = None,
        gluroo_api_secret: Optional[str] = None,
    ) -> dict:
        """
        Run one sync cycle for a single Tandem data source.

        - All treatments (basal, bolus, carbs) -> CosmosDB directly
        - Bolus/carb/basal treatments -> also pushed to Gluroo for mobile app visibility
        - Pump status snapshot -> CosmosDB pump_status container
        """
        until = until or datetime.now(timezone.utc)
        logger.info(f"Tandem sync for {user_id}: {since.isoformat()} to {until.isoformat()}")

        # Fetch ALL event types from Tandem API
        adapter = TandemApiAdapter(email, password)
        fetch_result = adapter.fetch_all_expanded(since, until)

        # Transform treatments (existing logic)
        basal_treatments = aggregate_basal_events(fetch_result.basal_events, user_id)
        bolus_treatments = map_bolus_events(fetch_result.bolus_events, user_id)

        bolus_count = sum(1 for t in bolus_treatments if t.type != "carbs")
        carb_count = sum(1 for t in bolus_treatments if t.type == "carbs")

        # Write ALL treatments to CosmosDB directly
        basal_count = 0
        for t in basal_treatments:
            try:
                await self.treatment_repo.upsert_dict(t.to_cosmos_dict())
                basal_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert basal treatment {t.id}: {e}")

        bolus_cosmos_count = 0
        for t in bolus_treatments:
            try:
                await self.treatment_repo.upsert_dict(t.to_cosmos_dict())
                bolus_cosmos_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert treatment {t.id}: {e}")

        # Build and upsert pump_status document from expanded events
        try:
            pump_status = await self._build_pump_status(user_id, fetch_result)
            await self.pump_status_repo.upsert(pump_status)
            logger.info(f"Pump status upserted for {user_id}")
        except Exception as e:
            logger.warning(f"Failed to upsert pump status for {user_id}: {e}")

        # Also push treatments to Gluroo for mobile app visibility
        gluroo_bolus_count = 0
        gluroo_basal_count = 0
        if gluroo_url and gluroo_api_secret:
            if bolus_treatments:
                gluroo_bolus_count = self._push_to_gluroo(
                    bolus_treatments, gluroo_url, gluroo_api_secret
                )
            if basal_treatments:
                gluroo_basal_count = self._push_basal_to_gluroo(
                    basal_treatments, gluroo_url, gluroo_api_secret
                )

        total = basal_count + bolus_count + carb_count
        logger.info(
            f"Tandem sync complete for {user_id}: "
            f"{basal_count} basal + {bolus_cosmos_count} bolus/carbs -> CosmosDB, "
            f"{gluroo_bolus_count} bolus+carbs + {gluroo_basal_count} basal -> Gluroo"
        )

        return {
            "basal": basal_count,
            "bolus": bolus_count,
            "carbs": carb_count,
            "total": total,
            "gluroo": gluroo_bolus_count + gluroo_basal_count,
        }

    async def _build_pump_status(self, user_id: str, result) -> dict:
        """Build a pump_status document by merging new events into the existing document.

        We merge rather than replace because site changes, cartridge fills, etc.
        happen every 2-3 days but the sync window is only 24 hours. Without merging,
        those fields would disappear between events.
        """
        now = datetime.now(timezone.utc)

        # Load existing document to preserve fields not in this sync window
        try:
            existing = await self.pump_status_repo.get(user_id)
        except Exception:
            existing = None

        status = dict(existing) if existing else {}
        # Remove CosmosDB metadata so upsert works cleanly
        for key in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            status.pop(key, None)

        status["id"] = f"{user_id}_pump_status"
        status["userId"] = user_id
        status["last_updated"] = now.isoformat()

        # Get user timezone for daily totals (midnight-to-midnight in local time)
        user_tz = ZoneInfo("America/New_York")  # default
        try:
            from database.repositories import UserRepository
            user_repo = UserRepository()
            user = await user_repo.get_by_id(user_id)
            if user and user.settings and user.settings.timezone:
                user_tz = ZoneInfo(user.settings.timezone)
        except Exception:
            pass  # fall back to Eastern

        # Battery and daily totals from most recent daily basal event
        if result.daily_basal_events:
            latest = result.daily_basal_events[-1]
            status["battery_percent"] = latest.battery_percent
            status["battery_millivolts"] = latest.battery_millivolts
            status["daily_basal_units"] = latest.daily_basal_units
            status["pump_iob"] = latest.pump_iob
            status["battery_updated_at"] = latest.timestamp.isoformat()

        # Daily bolus total from bolus events (today in user's local timezone)
        today_local = now.astimezone(user_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = today_local.astimezone(timezone.utc)
        daily_bolus_u = sum(
            b.insulin for b in result.bolus_events
            if b.insulin and b.insulin > 0 and b.timestamp >= today_start
        )
        if daily_bolus_u > 0:
            status["daily_bolus_units"] = round(daily_bolus_u, 2)
            basal_u = status.get("daily_basal_units") or 0
            status["daily_total_insulin"] = round(basal_u + daily_bolus_u, 2)

        # Current Control-IQ mode
        if result.mode_changes:
            latest = result.mode_changes[-1]
            status["current_mode"] = latest.current_mode
            status["mode_changed_at"] = latest.timestamp.isoformat()
            # Recent mode changes (last 10)
            status["recent_mode_changes"] = [
                {
                    "from": mc.previous_mode,
                    "to": mc.current_mode,
                    "at": mc.timestamp.isoformat(),
                }
                for mc in result.mode_changes[-10:]
            ]

        # Current pump control mode (ClosedLoop, OpenLoop, etc.)
        if result.pcm_changes:
            latest = result.pcm_changes[-1]
            status["control_mode"] = latest.current_pcm
            status["control_mode_changed_at"] = latest.timestamp.isoformat()

        # Suspend state
        if result.suspend_events:
            latest = result.suspend_events[-1]
            status["is_suspended"] = latest.action == "suspended"
            status["last_suspend_action"] = latest.action
            status["last_suspend_reason"] = latest.reason
            status["last_suspend_at"] = latest.timestamp.isoformat()

        # Alerts (last 10)
        if result.alerts:
            latest = result.alerts[-1]
            status["last_alert"] = latest.alert_type
            status["last_alert_at"] = latest.timestamp.isoformat()
            status["recent_alerts"] = [
                {"alert": a.alert_type, "at": a.timestamp.isoformat()}
                for a in result.alerts[-10:]
            ]

        # Alarms (last 10)
        if result.alarms:
            latest = result.alarms[-1]
            status["last_alarm"] = latest.alarm_type
            status["last_alarm_at"] = latest.timestamp.isoformat()
            status["recent_alarms"] = [
                {"alarm": a.alarm_type, "at": a.timestamp.isoformat()}
                for a in result.alarms[-10:]
            ]

        # Site change
        if result.site_changes:
            latest = result.site_changes[-1]
            status["last_site_change_at"] = latest.timestamp.isoformat()

        # Always recompute site age from stored timestamp
        if status.get("last_site_change_at"):
            try:
                site_ts = datetime.fromisoformat(status["last_site_change_at"])
                hours = (now - site_ts).total_seconds() / 3600
                status["site_age_hours"] = round(hours, 1)
            except (ValueError, TypeError):
                pass

        # Cartridge
        if result.cartridge_events:
            latest = result.cartridge_events[-1]
            status["last_cartridge_change_at"] = latest.timestamp.isoformat()
            status["last_cartridge_volume"] = latest.volume

        # Compute insulin remaining: cartridge fill volume minus total delivered since fill
        if status.get("last_cartridge_change_at") and status.get("last_cartridge_volume"):
            try:
                fill_ts = datetime.fromisoformat(status["last_cartridge_change_at"])
                # Sum all insulin delivered since cartridge fill
                delivered = 0.0
                for b in result.basal_events:
                    if b.timestamp >= fill_ts:
                        delivered += b.delivered_units or 0
                for b in result.bolus_events:
                    if b.timestamp >= fill_ts and b.insulin:
                        delivered += b.insulin
                # Only update if we have delivery data
                if delivered > 0:
                    remaining = max(0, status["last_cartridge_volume"] - delivered)
                    status["insulin_remaining"] = round(remaining, 1)
            except (ValueError, TypeError):
                pass

        # Tubing
        if result.tubing_events:
            latest = result.tubing_events[-1]
            status["last_tubing_fill_at"] = latest.timestamp.isoformat()

        # Daily status (auto corrections)
        if result.daily_status:
            latest = result.daily_status[-1]
            status["daily_auto_corrections"] = latest.auto_corrections_today
            status["sensor_type"] = latest.sensor_type

        # Daily carbs from bolus events (today only)
        daily_carbs = sum(
            b.carbs for b in result.bolus_events
            if b.carbs and b.carbs > 0 and b.timestamp >= today_start
        )
        if daily_carbs > 0:
            status["daily_carbs"] = round(daily_carbs, 1)

        return status

    def _push_to_gluroo(
        self,
        treatments: List[T1DAITreatment],
        gluroo_url: str,
        api_secret: str,
    ) -> int:
        """Push bolus/carb treatments to Gluroo via Nightscout API, with dedup.

        Uses separate entries per treatment (Correction Bolus + Carb Correction)
        which is the proven format that displays correctly in the Gluroo app.
        """
        base = gluroo_url.rstrip('/')
        secret_hash = hashlib.sha1(api_secret.encode()).hexdigest()
        headers = {
            "API-SECRET": secret_hash,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Build set of existing timestamps in Gluroo to avoid pushing duplicates.
        # Note: Gluroo overwrites enteredBy to "System", so we can't identify
        # our entries — instead we dedup by matching timestamp + event type.
        existing_keys: set = set()
        try:
            with httpx.Client(timeout=15) as client:
                for event_type in ["Correction Bolus", "Carb Correction"]:
                    resp = client.get(
                        f"{base}/api/v1/treatments.json",
                        params={
                            "count": 500,
                            "find[eventType]": event_type,
                        },
                        headers=headers,
                    )
                    resp.raise_for_status()
                    for entry in resp.json():
                        ts = entry.get("created_at", "")
                        existing_keys.add((event_type, ts))
        except Exception as e:
            logger.warning(f"Could not fetch existing Gluroo entries for dedup: {e}")

        # Push each treatment as a separate entry, skipping duplicates
        pushed = 0
        skipped = 0
        for treatment in treatments:
            ns_entry = self._to_nightscout_entry(treatment)
            if ns_entry is None:
                continue

            key = (ns_entry["eventType"], ns_entry["created_at"])
            if key in existing_keys:
                skipped += 1
                continue

            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(
                        f"{base}/api/v1/treatments",
                        json=ns_entry,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    pushed += 1
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Gluroo push failed: "
                    f"{e.response.status_code} {e.response.text[:200]}"
                )
            except Exception as e:
                logger.warning(f"Gluroo push error: {e}")

        if pushed > 0 or skipped > 0:
            logger.info(f"Gluroo push: {pushed} new, {skipped} skipped (already exist)")
        return pushed

    def _push_basal_to_gluroo(
        self,
        treatments: List[T1DAITreatment],
        gluroo_url: str,
        api_secret: str,
    ) -> int:
        """Push aggregated basal treatments to Gluroo as Temp Basal entries."""
        base = gluroo_url.rstrip('/')
        secret_hash = hashlib.sha1(api_secret.encode()).hexdigest()
        headers = {
            "API-SECRET": secret_hash,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Get existing Temp Basal timestamps for dedup
        existing_ts: set = set()
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{base}/api/v1/treatments.json",
                    params={"count": 500, "find[eventType]": "Temp Basal"},
                    headers=headers,
                )
                resp.raise_for_status()
                for entry in resp.json():
                    existing_ts.add(entry.get("created_at", ""))
        except Exception as e:
            logger.warning(f"Could not fetch existing Gluroo basal entries: {e}")

        pushed = 0
        skipped = 0
        for treatment in treatments:
            if treatment.type != "basal":
                continue

            ts = treatment.timestamp
            if ts in existing_ts:
                skipped += 1
                continue

            entry = {
                "eventType": "Temp Basal",
                "created_at": ts,
                "duration": treatment.durationMinutes or 60,
                "rate": treatment.basalRate or 0,
                "absolute": treatment.basalRate or 0,
                "enteredBy": "tandem-sync",
            }

            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(
                        f"{base}/api/v1/treatments",
                        json=entry,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    pushed += 1
            except httpx.HTTPStatusError as e:
                logger.warning(f"Gluroo basal push failed: {e.response.status_code}")
            except Exception as e:
                logger.warning(f"Gluroo basal push error: {e}")

        if pushed > 0 or skipped > 0:
            logger.info(f"Gluroo basal push: {pushed} new, {skipped} skipped")
        return pushed

    # Nightscout event type mapping (matches proven standalone gluroo_writer.py)
    _EVENT_TYPE_MAP = {
        "insulin": "Correction Bolus",
        "auto_correction": "Correction Bolus",
        "carbs": "Carb Correction",
    }

    @classmethod
    def _to_nightscout_entry(cls, treatment: T1DAITreatment) -> dict | None:
        """Convert a single treatment to a Nightscout entry.

        Uses separate entries per treatment (not combined Meal Bolus),
        which is the proven format that displays correctly in Gluroo.

        Carb entries use a +1 second timestamp offset because Gluroo silently
        drops entries at timestamps that already have another entry (e.g. the
        bolus). This offset ensures both insulin and carbs are stored.
        """
        event_type = cls._EVENT_TYPE_MAP.get(treatment.type)
        if event_type is None:
            return None

        timestamp = treatment.timestamp

        entry = {
            "eventType": event_type,
            "created_at": timestamp,
            "enteredBy": "tandem-sync",
            "notes": treatment.notes or "",
        }

        if treatment.type in ("insulin", "auto_correction"):
            entry["insulin"] = treatment.insulin
            if treatment.bolusType:
                entry["notes"] = f"[{treatment.bolusType}] {entry['notes']}".strip()
        elif treatment.type == "carbs":
            entry["carbs"] = treatment.carbs
            # Offset by +1 second to avoid Gluroo's silent timestamp dedup
            entry["created_at"] = cls._offset_timestamp(timestamp, seconds=1)

        return entry

    @staticmethod
    def _offset_timestamp(iso_ts: str, seconds: int = 1) -> str:
        """Add seconds to an ISO timestamp string."""
        try:
            dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            dt += timedelta(seconds=seconds)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except (ValueError, AttributeError):
            return iso_ts

    async def test_connection(self, email: str, password: str) -> tuple[bool, str]:
        """Test Tandem API credentials."""
        try:
            adapter = TandemApiAdapter(email, password)
            adapter._get_api()
            device_id = adapter._get_device_id()
            return True, f"Connected to Tandem pump (device {device_id})"
        except RuntimeError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Tandem connection test failed: {e}")
            return False, f"Connection failed: {str(e)}"
