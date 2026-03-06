"""
Tandem Sync Service - Fetches pump data from Tandem, writes to CosmosDB and Gluroo.

Data flow:
  All treatments:   Tandem API -> CosmosDB (direct write)
  Bolus + Carbs:    Tandem API -> Gluroo (Nightscout API) for mobile app visibility

Gluroo sync skips entries with enteredBy='tandem-sync' to avoid duplicates,
so all Tandem data must be written directly to CosmosDB.
"""
import hashlib
import logging
from collections import defaultdict
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


class TandemSyncService:
    """Fetches Tandem pump data. Writes all treatments to CosmosDB, pushes bolus/carbs to Gluroo."""

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

        - All treatments (basal, bolus, carbs) -> CosmosDB directly
        - Bolus/carb treatments -> also pushed to Gluroo for mobile app visibility
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

        # Also push bolus/carb treatments to Gluroo for mobile app visibility
        gluroo_count = 0
        if gluroo_url and gluroo_api_secret and bolus_treatments:
            gluroo_count = self._push_to_gluroo(
                bolus_treatments, gluroo_url, gluroo_api_secret
            )

        total = basal_count + bolus_count + carb_count
        logger.info(
            f"Tandem sync complete for {user_id}: "
            f"{basal_count} basal + {bolus_cosmos_count} bolus/carbs -> CosmosDB, "
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
        """Push bolus/carb treatments to Gluroo via Nightscout API, with dedup.

        Combines paired insulin+carb treatments at the same timestamp into a
        single "Meal Bolus" entry so Gluroo displays them together.
        """
        base = gluroo_url.rstrip('/')
        secret_hash = hashlib.sha1(api_secret.encode()).hexdigest()
        headers = {
            "API-SECRET": secret_hash,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Fetch existing tandem-sync entries from Gluroo to avoid duplicates
        existing_timestamps: set = set()
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{base}/api/v1/treatments.json",
                    params={
                        "count": 500,
                        "find[enteredBy]": "tandem-sync",
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                for entry in resp.json():
                    ts = entry.get("created_at", "")
                    existing_timestamps.add(ts)
            logger.info(f"Gluroo dedup: found {len(existing_timestamps)} existing tandem-sync entries")
        except Exception as e:
            logger.warning(f"Could not fetch existing Gluroo entries for dedup: {e}")

        # Build combined Nightscout entries (merge insulin+carbs at same timestamp)
        ns_entries = self._build_nightscout_entries(treatments)

        pushed = 0
        skipped = 0
        for ns_entry in ns_entries:
            # Skip if this timestamp already exists in Gluroo
            if ns_entry.get("created_at") in existing_timestamps:
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
            logger.info(f"Gluroo push: {pushed} new, {skipped} already existed")
        return pushed

    @staticmethod
    def _build_nightscout_entries(treatments: List[T1DAITreatment]) -> list:
        """Build Nightscout entries, combining insulin+carbs at the same timestamp.

        When a bolus has associated carbs (from pump calculator), Gluroo expects
        a single "Meal Bolus" entry with both insulin and carbs fields.
        """
        # Group by timestamp
        by_ts: dict = defaultdict(list)
        for t in treatments:
            by_ts[t.timestamp].append(t)

        entries = []
        for ts, group in by_ts.items():
            insulin_t = next((t for t in group if t.type in ("insulin", "auto_correction")), None)
            carb_t = next((t for t in group if t.type == "carbs"), None)

            if insulin_t and carb_t:
                # Combined meal bolus — Gluroo shows these together
                notes = insulin_t.notes or ""
                if insulin_t.bolusType:
                    notes = f"[{insulin_t.bolusType}] {notes}".strip()
                entries.append({
                    "eventType": "Meal Bolus",
                    "created_at": ts,
                    "enteredBy": "tandem-sync",
                    "insulin": insulin_t.insulin,
                    "carbs": carb_t.carbs,
                    "notes": notes,
                })
            elif insulin_t:
                # Correction or auto bolus (no carbs)
                notes = insulin_t.notes or ""
                if insulin_t.bolusType:
                    notes = f"[{insulin_t.bolusType}] {notes}".strip()
                entries.append({
                    "eventType": "Correction Bolus",
                    "created_at": ts,
                    "enteredBy": "tandem-sync",
                    "insulin": insulin_t.insulin,
                    "notes": notes,
                })
            elif carb_t:
                # Standalone carbs (rare from pump)
                entries.append({
                    "eventType": "Carb Correction",
                    "created_at": ts,
                    "enteredBy": "tandem-sync",
                    "carbs": carb_t.carbs,
                    "notes": carb_t.notes or "",
                })

        return entries

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
