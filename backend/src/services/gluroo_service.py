"""
Gluroo API Service for T1D-AI
Handles authentication and data fetching from Gluroo/Nightscout API.
Ported from gluroo_api.py
"""
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import uuid4

import httpx

from models.schemas import GlucoseReading, Treatment, TrendDirection, TreatmentType

logger = logging.getLogger(__name__)


class GlurooService:
    """Service for interacting with Gluroo Nightscout API."""

    def __init__(self, base_url: str, api_secret: str):
        """
        Initialize Gluroo service.

        Args:
            base_url: Gluroo Nightscout URL (e.g., https://share.gluroo.com)
            api_secret: Plain text API secret (will be SHA1 hashed)
        """
        self.base_url = base_url.rstrip('/')
        # Hash the API secret using SHA1 (Nightscout style)
        self.api_secret_hash = hashlib.sha1(api_secret.encode('utf-8')).hexdigest()
        self.headers = {"API-SECRET": self.api_secret_hash}

    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test the connection to Gluroo API.

        Returns:
            Tuple of (success, message)
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/entries.json?count=1",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return True, "Connection successful"
                elif response.status_code == 401:
                    return False, "Invalid API secret"
                else:
                    return False, f"Error: HTTP {response.status_code}"
        except httpx.TimeoutException:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    async def fetch_glucose_entries(
        self,
        user_id: str,
        count: int = 200,
        since: Optional[datetime] = None
    ) -> List[GlucoseReading]:
        """
        Fetch CGM glucose entries from Gluroo.

        Args:
            user_id: User ID to associate with readings
            count: Number of entries to fetch (max 1000)
            since: Only fetch entries after this time

        Returns:
            List of GlucoseReading objects
        """
        url = f"{self.base_url}/api/v1/entries.json?count={min(count, 1000)}"

        if since:
            # Gluroo uses 'find[date][$gte]' for date filtering
            since_ms = int(since.timestamp() * 1000)
            url += f"&find[date][$gte]={since_ms}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                entries = response.json() or []

            readings = []
            for entry in entries:
                try:
                    reading = self._parse_glucose_entry(entry, user_id)
                    if reading:
                        readings.append(reading)
                except Exception as e:
                    logger.warning(f"Failed to parse glucose entry: {e}")
                    continue

            logger.info(f"Fetched {len(readings)} glucose readings from Gluroo")
            return readings

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching glucose: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching glucose entries: {e}")
            raise

    async def fetch_treatments(
        self,
        user_id: str,
        count: int = 200,
        since: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> List[Treatment]:
        """
        Fetch treatments (insulin, carbs) from Gluroo.

        Args:
            user_id: User ID to associate with treatments
            count: Number of treatments to fetch
            since: Only fetch treatments after this time
            event_type: Filter by event type (e.g., "Correction Bolus", "Carb Correction")

        Returns:
            List of Treatment objects
        """
        url = f"{self.base_url}/api/v1/treatments.json?count={min(count, 1000)}"

        if since:
            since_ms = int(since.timestamp() * 1000)
            url += f"&find[date][$gte]={since_ms}"

        if event_type:
            url += f"&find[eventType]={event_type.replace(' ', '%20')}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                entries = response.json() or []

            treatments = []
            for entry in entries:
                try:
                    treatment = self._parse_treatment_entry(entry, user_id)
                    if treatment:
                        treatments.append(treatment)
                except Exception as e:
                    logger.warning(f"Failed to parse treatment entry: {e}")
                    continue

            logger.info(f"Fetched {len(treatments)} treatments from Gluroo")
            return treatments

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching treatments: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching treatments: {e}")
            raise

    async def fetch_all_treatments(
        self,
        user_id: str,
        count: int = 200,
        since: Optional[datetime] = None
    ) -> List[Treatment]:
        """
        Fetch all treatment types (insulin and carbs) from Gluroo.

        Args:
            user_id: User ID to associate with treatments
            count: Number of treatments to fetch per type
            since: Only fetch treatments after this time

        Returns:
            Combined list of Treatment objects sorted by timestamp
        """
        # Fetch both treatment types
        insulin = await self.fetch_treatments(
            user_id, count, since, event_type="Correction Bolus"
        )
        carbs = await self.fetch_treatments(
            user_id, count, since, event_type="Carb Correction"
        )

        # Combine and sort by timestamp
        all_treatments = insulin + carbs
        all_treatments.sort(key=lambda t: t.timestamp, reverse=True)

        return all_treatments

    def _parse_glucose_entry(self, entry: dict, user_id: str) -> Optional[GlucoseReading]:
        """Parse a raw Gluroo glucose entry into a GlucoseReading."""
        sgv = entry.get('sgv')
        if sgv is None:
            return None

        # Parse timestamp
        date_ms = entry.get('date')
        if date_ms:
            timestamp = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        elif entry.get('dateString'):
            timestamp = datetime.fromisoformat(entry['dateString'].replace('Z', '+00:00'))
        else:
            return None

        # Parse trend direction
        direction = entry.get('direction', 'Flat')
        try:
            trend = TrendDirection(direction)
        except ValueError:
            trend = TrendDirection.FLAT

        # Generate unique ID
        source_id = entry.get('_id', str(entry.get('mills', '')))
        reading_id = f"{user_id}_{source_id}" if source_id else f"{user_id}_{uuid4().hex[:12]}"

        return GlucoseReading(
            id=reading_id,
            userId=user_id,
            timestamp=timestamp,
            value=int(sgv),
            trend=trend,
            source="gluroo",
            sourceId=source_id
        )

    def _parse_treatment_entry(self, entry: dict, user_id: str) -> Optional[Treatment]:
        """Parse a raw Gluroo treatment entry into a Treatment."""
        event_type = entry.get('eventType', '')

        # Determine treatment type
        insulin = entry.get('insulin')
        carbs = entry.get('carbs')

        if insulin is not None and float(insulin) > 0:
            treatment_type = TreatmentType.INSULIN
        elif carbs is not None and float(carbs) > 0:
            treatment_type = TreatmentType.CARBS
        else:
            return None  # Skip entries without insulin or carbs

        # Parse timestamp
        created_at = entry.get('created_at')
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.utcnow()
        else:
            mills = entry.get('mills')
            if mills:
                timestamp = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)
            else:
                return None

        # Generate unique ID
        source_id = entry.get('_id', str(entry.get('mills', '')))
        treatment_id = f"{user_id}_{source_id}" if source_id else f"{user_id}_{uuid4().hex[:12]}"

        return Treatment(
            id=treatment_id,
            userId=user_id,
            timestamp=timestamp,
            type=treatment_type,
            insulin=float(insulin) if insulin else None,
            carbs=float(carbs) if carbs else None,
            protein=float(entry.get('protein', 0)) if entry.get('protein') else None,
            fat=float(entry.get('fat', 0)) if entry.get('fat') else None,
            notes=entry.get('notes'),
            source="gluroo",
            sourceId=source_id
        )


async def create_gluroo_service(base_url: str, api_secret: str) -> GlurooService:
    """
    Factory function to create and test a Gluroo service.

    Args:
        base_url: Gluroo Nightscout URL
        api_secret: Plain text API secret

    Returns:
        Configured GlurooService instance

    Raises:
        ValueError: If connection test fails
    """
    service = GlurooService(base_url, api_secret)
    success, message = await service.test_connection()

    if not success:
        raise ValueError(f"Failed to connect to Gluroo: {message}")

    logger.info(f"Gluroo service connected: {base_url}")
    return service
