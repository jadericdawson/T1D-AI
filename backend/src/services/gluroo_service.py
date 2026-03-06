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

    async def push_treatment(
        self,
        treatment_type: str,  # "insulin" or "carbs"
        value: float,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
        protein: Optional[float] = None,
        fat: Optional[float] = None,
        glycemic_index: Optional[int] = None,
        absorption_rate: Optional[str] = None,
        is_liquid: Optional[bool] = None
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Push a treatment (insulin or carbs) to Gluroo via Nightscout API.

        Args:
            treatment_type: "insulin" or "carbs"
            value: Amount (units for insulin, grams for carbs)
            timestamp: When the treatment occurred (defaults to now)
            notes: Optional notes about the treatment
            protein: Protein in grams (for carb treatments)
            fat: Fat in grams (for carb treatments)
            glycemic_index: GI value from AI enrichment
            absorption_rate: Absorption speed (very_slow, slow, medium, fast, very_fast)
            is_liquid: Whether the food is liquid

        Returns:
            Tuple of (success, message, response_data)
        """
        url = f"{self.base_url}/api/v1/treatments"

        # Build treatment payload
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Nightscout treatment format
        treatment_data = {
            "created_at": timestamp.isoformat(),
            "enteredBy": "T1D-AI",
        }

        if treatment_type == "insulin":
            treatment_data["eventType"] = "Correction Bolus"
            treatment_data["insulin"] = value
        elif treatment_type == "carbs":
            treatment_data["eventType"] = "Carb Correction"
            treatment_data["carbs"] = value
            # Add macros if available (Nightscout supports these)
            if protein is not None and protein > 0:
                treatment_data["protein"] = protein
            if fat is not None and fat > 0:
                treatment_data["fat"] = fat
        else:
            return False, f"Invalid treatment type: {treatment_type}", None

        # Build enriched notes with GI info
        note_parts = []
        if notes:
            note_parts.append(notes)
        if glycemic_index is not None:
            note_parts.append(f"GI:{glycemic_index}")
        if absorption_rate:
            note_parts.append(f"[{absorption_rate}]")
        if is_liquid:
            note_parts.append("[liquid]")

        if note_parts:
            treatment_data["notes"] = " ".join(note_parts)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={**self.headers, "Content-Type": "application/json"},
                    json=treatment_data
                )

                if response.status_code in [200, 201]:
                    result = response.json() if response.text else {}
                    logger.info(f"Pushed {treatment_type} treatment to Gluroo: {value}")
                    return True, "Treatment pushed successfully", result
                elif response.status_code == 401:
                    return False, "Unauthorized - check API secret", None
                elif response.status_code == 403:
                    return False, "Forbidden - API may be read-only", None
                else:
                    return False, f"HTTP {response.status_code}: {response.text}", None

        except httpx.TimeoutException:
            return False, "Connection timeout", None
        except Exception as e:
            logger.error(f"Error pushing treatment to Gluroo: {e}")
            return False, f"Error: {str(e)}", None

    async def delete_treatment(
        self,
        nightscout_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        treatment_type: Optional[str] = None,
        value: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Delete a treatment from Gluroo via Nightscout API.

        Can delete by:
        - nightscout_id: Direct deletion by Nightscout _id
        - timestamp + treatment_type + value: Find and delete matching treatment

        Args:
            nightscout_id: The Nightscout _id (if known from previous push)
            timestamp: When the treatment occurred
            treatment_type: "insulin" or "carbs"
            value: Amount (for matching)

        Returns:
            Tuple of (success, message)
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # If we have the Nightscout ID, delete directly
                if nightscout_id:
                    url = f"{self.base_url}/api/v1/treatments/{nightscout_id}"
                    response = await client.delete(url, headers=self.headers)

                    if response.status_code in [200, 204]:
                        logger.info(f"Deleted treatment {nightscout_id} from Gluroo")
                        return True, "Treatment deleted successfully"
                    elif response.status_code == 404:
                        # Treatment not found in Gluroo - could be already deleted there
                        # Treat this as success since the end goal (treatment not in Gluroo) is achieved
                        logger.info(f"Treatment {nightscout_id} not found in Gluroo (may be already deleted)")
                        return True, "Treatment not found in Gluroo (already deleted?)"
                    else:
                        return False, f"HTTP {response.status_code}: {response.text}"

                # Otherwise, search for matching treatment and delete
                elif timestamp and treatment_type:
                    # Search for treatments in a 10-minute window around the timestamp
                    from datetime import timedelta

                    # Ensure timestamp is timezone-aware (assume UTC if naive)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)

                    window_start = timestamp - timedelta(minutes=5)
                    window_end = timestamp + timedelta(minutes=5)

                    logger.info(f"Gluroo delete: looking for {treatment_type}={value} at {timestamp.isoformat()}")
                    logger.info(f"Gluroo delete: search window {window_start.isoformat()} to {window_end.isoformat()}")

                    # Format timestamps for Nightscout API - use mills (milliseconds) for reliable searching
                    # Use UTC timestamp to get correct milliseconds
                    window_start_ms = int(window_start.timestamp() * 1000)
                    window_end_ms = int(window_end.timestamp() * 1000)
                    logger.info(f"Gluroo delete: mills range {window_start_ms} to {window_end_ms}")

                    # Map treatment type to Nightscout eventType (same as sync uses)
                    event_type = "Correction%20Bolus" if treatment_type == "insulin" else "Carb%20Correction"

                    # Search by mills AND eventType (matching how sync pulls treatments)
                    search_url = (
                        f"{self.base_url}/api/v1/treatments.json?"
                        f"find[eventType]={event_type}&"
                        f"find[mills][$gte]={window_start_ms}&"
                        f"find[mills][$lte]={window_end_ms}&count=50"
                    )

                    logger.info(f"Gluroo delete search URL: {search_url}")
                    search_response = await client.get(search_url, headers=self.headers)

                    if search_response.status_code != 200:
                        logger.warning(f"Gluroo search failed: {search_response.status_code} - {search_response.text}")
                        return False, f"Search failed: HTTP {search_response.status_code}"

                    treatments = search_response.json() or []
                    logger.info(f"Gluroo delete: found {len(treatments)} treatments in search window")

                    # Log each treatment for debugging
                    for t in treatments:
                        t_type = "insulin" if t.get("insulin") else "carbs" if t.get("carbs") else "unknown"
                        t_value = t.get("insulin") or t.get("carbs") or 0
                        t_created = t.get("created_at", "unknown")
                        t_id = t.get("_id", "unknown")
                        logger.info(f"  - Gluroo treatment: _id={t_id}, type={t_type}, value={t_value}, created_at={t_created}")

                    # Find matching treatment by type and value
                    # Don't require enteredBy match - treatment may have been synced originally from Gluroo
                    logger.info(f"Gluroo delete: looking for type={treatment_type}, value={value}")
                    for t in treatments:
                        t_type = "insulin" if t.get("insulin") else "carbs" if t.get("carbs") else None
                        t_value = t.get("insulin") or t.get("carbs")
                        logger.info(f"Gluroo delete: checking t_type={t_type}, t_value={t_value}")

                        # Match by type and value (with small tolerance for floating point)
                        if (t_type == treatment_type and
                            abs(float(t_value or 0) - float(value or 0)) < 0.1):
                            logger.info(f"Gluroo delete: MATCH FOUND!")

                            # Found it, delete
                            t_id = t.get("_id")
                            if t_id:
                                logger.info(f"Found matching treatment {t_id} ({t_type}={t_value}), deleting...")
                                del_url = f"{self.base_url}/api/v1/treatments/{t_id}"
                                del_response = await client.delete(del_url, headers=self.headers)

                                if del_response.status_code in [200, 204]:
                                    logger.info(f"Successfully deleted treatment {t_id} from Gluroo")
                                    return True, "Treatment deleted successfully"
                                else:
                                    logger.warning(f"Delete failed: {del_response.status_code} - {del_response.text}")

                    # No matching treatment found - could be already deleted
                    logger.info(f"No matching {treatment_type}={value} found in Gluroo (may be already deleted)")
                    return True, "Treatment not found in Gluroo (already deleted?)"
                else:
                    return False, "Need either nightscout_id or timestamp+type+value to delete"

        except httpx.TimeoutException:
            return False, "Connection timeout"
        except Exception as e:
            logger.error(f"Error deleting treatment from Gluroo: {e}")
            return False, f"Error: {str(e)}"

    def _parse_treatment_entry(self, entry: dict, user_id: str) -> Optional[Treatment]:
        """Parse a raw Gluroo treatment entry into a Treatment."""
        # Skip treatments that originated from T1D-AI or tandem-sync to prevent duplicates
        # tandem-sync entries are already in CosmosDB via TandemSyncService;
        # note sync is handled separately by the legacy GlurooSyncService
        entered_by = entry.get('enteredBy', '')
        if entered_by in ('T1D-AI', 'tandem-sync'):
            logger.debug(f"Skipping {entered_by} originated treatment to prevent duplicate")
            return None

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


def dedupe_treatments(treatments: List[Treatment], window_minutes: int = 5) -> List[Treatment]:
    """
    Deduplicate treatments that appear as multiple events within a short window.

    Gluroo sometimes reports the same insulin/carb event multiple times within
    a short period. This groups treatments by type and value within a time window,
    keeping only the first occurrence.

    Args:
        treatments: List of Treatment objects
        window_minutes: Time window in minutes to consider as same event

    Returns:
        Deduplicated list of treatments
    """
    if not treatments:
        return []

    # Sort by timestamp
    sorted_treatments = sorted(treatments, key=lambda t: t.timestamp)
    deduped = []

    for treatment in sorted_treatments:
        # Check if this is a duplicate of a recent treatment
        is_dupe = False
        for existing in reversed(deduped):
            # Only compare within the time window
            time_diff = (treatment.timestamp - existing.timestamp).total_seconds() / 60
            if time_diff > window_minutes:
                break  # Outside window, stop checking

            # Check if same type and value
            if treatment.type == existing.type:
                if treatment.type == TreatmentType.INSULIN:
                    if treatment.insulin == existing.insulin:
                        is_dupe = True
                        break
                elif treatment.type == TreatmentType.CARBS:
                    if treatment.carbs == existing.carbs:
                        is_dupe = True
                        break

        if not is_dupe:
            deduped.append(treatment)

    if len(deduped) < len(treatments):
        logger.info(f"Deduped treatments: {len(treatments)} -> {len(deduped)}")

    return deduped


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
