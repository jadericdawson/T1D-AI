"""
Tandem Sync Service - Fetches basal data from Tandem pump.

Data flow:
  Boluses:  Pump -> Gluroo (native) -> CosmosDB (via Gluroo sync)
  Basal:    Pump -> Tandem API -> CosmosDB (direct, Gluroo doesn't track basal)

Tandem sync only writes BASAL to CosmosDB. Gluroo already captures all
bolus/auto-correction data natively from the pump — no push needed.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from services.tandem import (
    TandemApiAdapter,
    aggregate_basal_events,
    map_bolus_events,
)

logger = logging.getLogger(__name__)


class TandemSyncService:
    """Fetches Tandem pump data. Writes only basal to CosmosDB."""

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
    ) -> dict:
        """
        Run one sync cycle for a single Tandem data source.

        Only writes basal treatments to CosmosDB.
        Boluses are already in Gluroo and synced via Gluroo sync.
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

        if not basal_treatments:
            logger.info(f"Tandem sync for {user_id}: no new basal treatments")
            return {"basal": 0, "bolus": bolus_count, "carbs": carb_count,
                    "total": bolus_count + carb_count, "gluroo": 0}

        # Write only basal to CosmosDB (Gluroo handles boluses)
        basal_count = 0
        for t in basal_treatments:
            try:
                await self.treatment_repo.upsert_dict(t.to_cosmos_dict())
                basal_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert basal treatment {t.id}: {e}")

        total = basal_count + bolus_count + carb_count
        logger.info(
            f"Tandem sync complete for {user_id}: "
            f"{basal_count} basal -> CosmosDB, "
            f"{bolus_count} bolus + {carb_count} carbs (via Gluroo)"
        )

        return {
            "basal": basal_count,
            "bolus": bolus_count,
            "carbs": carb_count,
            "total": total,
            "gluroo": 0,
        }

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
