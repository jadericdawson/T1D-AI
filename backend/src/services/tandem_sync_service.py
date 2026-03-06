"""
Tandem Sync Service - Fetches pump data from Tandem and pushes to Gluroo + CosmosDB.

Data flow:
  Basal:            Tandem API -> CosmosDB (direct, Gluroo doesn't track basal)
  Bolus + Carbs:    Tandem API -> Gluroo (Nightscout API) -> CosmosDB (via Gluroo sync)

Bolus/carb data is pushed to Gluroo so it appears in the Gluroo mobile app
and gets picked up by the normal Gluroo sync into CosmosDB.
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from services.tandem import (
    TandemApiAdapter,
    aggregate_basal_events,
    map_bolus_events,
)
from services.tandem.t1dai_models import T1DAITreatment

logger = logging.getLogger(__name__)

# Nightscout event type mapping for Gluroo push
_EVENT_TYPE_MAP = {
    "insulin": "Correction Bolus",
    "auto_correction": "Correction Bolus",
    "basal": "Temp Basal",
    "carbs": "Carb Correction",
}


class TandemSyncService:
    """Fetches Tandem pump data. Writes basal to CosmosDB, bolus/carbs to Gluroo."""

    def __init__(self):
        self._treatment_repo = None

    @property
    def treatment_repo(self):
        if self._treatment_repo is None:
            from database.repositories import TreatmentRepository
            self._treatment_repo = TreatmentRepository()
        return self._treatment_repo

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

        - Basal treatments -> CosmosDB directly
        - Bolus/carb treatments -> Gluroo (if credentials provided)
        """
        until = until or datetime.now(timezone.utc)
        logger.info(f"Tandem sync for {user_id}: {since.isoformat()} to {until.isoformat()}")

        # Fetch from Tandem API
        adapter = TandemApiAdapter(email, password)
        basal_events, bolus_events = adapter.fetch_all(since, until)

        # Transform
        basal_treatments = aggregate_basal_events(basal_events, user_id)
        bolus_treatments = map_bolus_events(bolus_events, user_id)

        bolus_count = sum(1 for t in bolus_treatments if t.type != "carbs")
        carb_count = sum(1 for t in bolus_treatments if t.type == "carbs")

        # Write basal to CosmosDB (Gluroo doesn't track basal)
        basal_count = 0
        for t in basal_treatments:
            try:
                await self.treatment_repo.upsert_dict(t.to_cosmos_dict())
                basal_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert basal treatment {t.id}: {e}")

        # Push bolus/carb treatments to Gluroo so they appear in the mobile app
        gluroo_count = 0
        if gluroo_url and gluroo_api_secret and bolus_treatments:
            gluroo_count = self._push_to_gluroo(
                bolus_treatments, gluroo_url, gluroo_api_secret
            )

        total = basal_count + bolus_count + carb_count
        logger.info(
            f"Tandem sync complete for {user_id}: "
            f"{basal_count} basal -> CosmosDB, "
            f"{gluroo_count}/{len(bolus_treatments)} bolus+carbs -> Gluroo"
        )

        return {
            "basal": basal_count,
            "bolus": bolus_count,
            "carbs": carb_count,
            "total": total,
            "gluroo": gluroo_count,
        }

    def _push_to_gluroo(
        self,
        treatments: List[T1DAITreatment],
        gluroo_url: str,
        api_secret: str,
    ) -> int:
        """Push bolus/carb treatments to Gluroo via Nightscout API."""
        secret_hash = hashlib.sha1(api_secret.encode()).hexdigest()
        headers = {
            "API-SECRET": secret_hash,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        pushed = 0
        for treatment in treatments:
            ns_entry = self._to_nightscout_entry(treatment)
            if ns_entry is None:
                continue
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.post(
                        f"{gluroo_url.rstrip('/')}/api/v1/treatments",
                        json=ns_entry,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    pushed += 1
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Gluroo push failed for {treatment.sourceId}: "
                    f"{e.response.status_code} {e.response.text[:200]}"
                )
            except Exception as e:
                logger.warning(f"Gluroo push error for {treatment.sourceId}: {e}")

        if pushed > 0:
            logger.info(f"Pushed {pushed}/{len(treatments)} treatments to Gluroo")
        return pushed

    @staticmethod
    def _to_nightscout_entry(treatment: T1DAITreatment) -> Optional[dict]:
        """Convert T1DAITreatment to Nightscout treatment format."""
        event_type = _EVENT_TYPE_MAP.get(treatment.type)
        if event_type is None:
            return None

        entry = {
            "eventType": event_type,
            "created_at": treatment.timestamp,
            "enteredBy": "tandem-sync",
            "notes": treatment.notes or "",
        }

        if treatment.type == "basal":
            entry["duration"] = treatment.durationMinutes or 60
            entry["rate"] = treatment.basalRate or 0
            entry["absolute"] = treatment.basalRate or 0
            if treatment.insulin:
                entry["insulin"] = treatment.insulin
        elif treatment.type in ("insulin", "auto_correction"):
            entry["insulin"] = treatment.insulin
            if treatment.bolusType:
                entry["notes"] = f"[{treatment.bolusType}] {entry['notes']}".strip()
        elif treatment.type == "carbs":
            entry["carbs"] = treatment.carbs

        return entry

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
