"""
Gluroo Sync Azure Function
Timer-triggered function that syncs data from Gluroo for all connected users.
Runs every 5 minutes.
"""
import asyncio
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import azure.functions as func
from azure.cosmos import CosmosClient, PartitionKey
from cryptography.fernet import Fernet
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Environment variables
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
COSMOS_DATABASE = os.environ.get("COSMOS_DATABASE", "T1D-AI-DB")
ENCRYPTION_MASTER_KEY = os.environ.get("ENCRYPTION_MASTER_KEY")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")


def get_fernet() -> Fernet:
    """Get Fernet instance for decryption."""
    import base64
    key = ENCRYPTION_MASTER_KEY
    if not key:
        # Derive from JWT secret as fallback
        derived = hashlib.sha256(JWT_SECRET_KEY.encode()).digest()
        key = base64.urlsafe_b64encode(derived).decode()
    return Fernet(key.encode())


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt an encrypted API secret."""
    fernet = get_fernet()
    decrypted = fernet.decrypt(ciphertext.encode('utf-8'))
    return decrypted.decode('utf-8')


class GlurooSyncService:
    """Service for syncing data from Gluroo."""

    def __init__(self, base_url: str, api_secret: str):
        self.base_url = base_url.rstrip('/')
        self.api_secret_hash = hashlib.sha1(api_secret.encode('utf-8')).hexdigest()
        self.headers = {"API-SECRET": self.api_secret_hash}

    async def fetch_glucose(
        self,
        user_id: str,
        count: int = 500,
        since: Optional[datetime] = None
    ) -> List[dict]:
        """Fetch glucose entries from Gluroo."""
        url = f"{self.base_url}/api/v1/entries.json?count={min(count, 1000)}"

        if since:
            since_ms = int(since.timestamp() * 1000)
            url += f"&find[date][$gte]={since_ms}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            entries = response.json() or []

        readings = []
        for entry in entries:
            sgv = entry.get('sgv')
            if sgv is None:
                continue

            # Parse timestamp
            date_ms = entry.get('date')
            if date_ms:
                timestamp = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
            elif entry.get('dateString'):
                timestamp = datetime.fromisoformat(entry['dateString'].replace('Z', '+00:00'))
            else:
                continue

            source_id = entry.get('_id', str(entry.get('mills', '')))
            reading_id = f"{user_id}_{source_id}"

            readings.append({
                "id": reading_id,
                "userId": user_id,
                "timestamp": timestamp.isoformat(),
                "value": int(sgv),
                "trend": entry.get('direction', 'Flat'),
                "source": "gluroo",
                "sourceId": source_id
            })

        return readings

    async def fetch_treatments(
        self,
        user_id: str,
        count: int = 500,
        since: Optional[datetime] = None
    ) -> List[dict]:
        """Fetch treatments from Gluroo."""
        url = f"{self.base_url}/api/v1/treatments.json?count={min(count, 1000)}"

        if since:
            since_ms = int(since.timestamp() * 1000)
            url += f"&find[date][$gte]={since_ms}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            entries = response.json() or []

        treatments = []
        for entry in entries:
            insulin = entry.get('insulin')
            carbs = entry.get('carbs')

            if not (insulin or carbs):
                continue

            # Determine type
            if insulin and float(insulin) > 0:
                treatment_type = "insulin"
            elif carbs and float(carbs) > 0:
                treatment_type = "carbs"
            else:
                continue

            # Parse timestamp
            created_at = entry.get('created_at')
            if created_at:
                try:
                    timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except ValueError:
                    continue
            else:
                mills = entry.get('mills')
                if mills:
                    timestamp = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)
                else:
                    continue

            source_id = entry.get('_id', str(entry.get('mills', '')))
            treatment_id = f"{user_id}_{source_id}"

            treatments.append({
                "id": treatment_id,
                "userId": user_id,
                "timestamp": timestamp.isoformat(),
                "type": treatment_type,
                "insulin": float(insulin) if insulin else None,
                "carbs": float(carbs) if carbs else None,
                "protein": float(entry.get('protein', 0)) if entry.get('protein') else None,
                "fat": float(entry.get('fat', 0)) if entry.get('fat') else None,
                "notes": entry.get('notes'),
                "source": "gluroo",
                "sourceId": source_id
            })

        # Dedupe treatments - same type+value within 5 min window is one event
        return self._dedupe_treatments(treatments)

    def _dedupe_treatments(self, treatments: List[dict], window_minutes: int = 5) -> List[dict]:
        """
        Deduplicate treatments that appear as multiple events within a short window.

        Gluroo sometimes reports the same insulin/carb event multiple times.
        This groups treatments by type and value within a time window.
        """
        if not treatments:
            return []

        # Sort by timestamp
        sorted_treatments = sorted(treatments, key=lambda t: t["timestamp"])
        deduped = []

        for treatment in sorted_treatments:
            is_dupe = False
            treatment_time = datetime.fromisoformat(treatment["timestamp"].replace('Z', '+00:00'))

            for existing in reversed(deduped):
                existing_time = datetime.fromisoformat(existing["timestamp"].replace('Z', '+00:00'))
                time_diff = (treatment_time - existing_time).total_seconds() / 60

                if time_diff > window_minutes:
                    break  # Outside window

                # Check same type and value
                if treatment["type"] == existing["type"]:
                    if treatment["type"] == "insulin" and treatment.get("insulin") == existing.get("insulin"):
                        is_dupe = True
                        break
                    elif treatment["type"] == "carbs" and treatment.get("carbs") == existing.get("carbs"):
                        is_dupe = True
                        break

            if not is_dupe:
                deduped.append(treatment)

        if len(deduped) < len(treatments):
            logger.info(f"Deduped treatments: {len(treatments)} -> {len(deduped)}")

        return deduped


async def sync_user(
    cosmos_client: CosmosClient,
    user_id: str,
    datasource: dict
) -> dict:
    """Sync data for a single user."""
    try:
        # Get database and containers
        database = cosmos_client.get_database_client(COSMOS_DATABASE)
        glucose_container = database.get_container_client("glucose_readings")
        treatment_container = database.get_container_client("treatments")
        datasource_container = database.get_container_client("datasources")

        # Get credentials
        credentials = datasource.get("credentials", {})
        url = credentials.get("url")
        encrypted_secret = credentials.get("apiSecretEncrypted")

        if not url or not encrypted_secret:
            return {"user_id": user_id, "error": "Missing credentials"}

        # Decrypt API secret
        api_secret = decrypt_secret(encrypted_secret)

        # Determine sync start time
        last_sync = datasource.get("lastSyncAt")
        if last_sync:
            since = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
        else:
            # First sync - get last 7 days
            since = datetime.now(timezone.utc) - timedelta(days=7)

        # Create service and fetch data
        service = GlurooSyncService(url, api_secret)

        glucose_readings = await service.fetch_glucose(user_id, count=500, since=since)
        treatments = await service.fetch_treatments(user_id, count=500, since=since)

        # Upsert glucose readings
        glucose_count = 0
        for reading in glucose_readings:
            try:
                glucose_container.upsert_item(reading)
                glucose_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert glucose reading: {e}")

        # Upsert treatments
        treatment_count = 0
        for treatment in treatments:
            try:
                treatment_container.upsert_item(treatment)
                treatment_count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert treatment: {e}")

        # Update datasource with last sync time
        now = datetime.now(timezone.utc).isoformat()
        datasource["lastSyncAt"] = now
        datasource["recordsSynced"] = datasource.get("recordsSynced", 0) + glucose_count + treatment_count
        datasource_container.upsert_item(datasource)

        logger.info(f"Synced user {user_id}: {glucose_count} glucose, {treatment_count} treatments")

        return {
            "user_id": user_id,
            "glucose_count": glucose_count,
            "treatment_count": treatment_count,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error syncing user {user_id}: {e}")
        return {"user_id": user_id, "error": str(e), "success": False}


async def run_sync():
    """Main sync function - syncs all users with connected Gluroo datasources."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        logger.error("Missing Cosmos DB configuration")
        return {"error": "Missing configuration"}

    # Initialize Cosmos client
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    database = cosmos_client.get_database_client(COSMOS_DATABASE)
    datasource_container = database.get_container_client("datasources")

    # Query all Gluroo datasources
    query = "SELECT * FROM c WHERE c.sourceType = 'gluroo' AND c.status = 'connected'"
    datasources = list(datasource_container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))

    logger.info(f"Found {len(datasources)} connected Gluroo datasources")

    # Sync each user
    results = []
    for datasource in datasources:
        user_id = datasource.get("userId")
        if user_id:
            result = await sync_user(cosmos_client, user_id, datasource)
            results.append(result)

    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    logger.info(f"Sync complete: {successful} successful, {failed} failed")

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }


def main(mytimer: func.TimerRequest) -> None:
    """Azure Function entry point - Timer triggered."""
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    if mytimer.past_due:
        logger.info('Timer is past due!')

    logger.info(f'Gluroo sync function started at {utc_timestamp}')

    # Run async sync
    result = asyncio.run(run_sync())

    logger.info(f'Gluroo sync function completed: {result}')
