#!/usr/bin/env python3
"""
Gluroo to CosmosDB Continuous Sync Service
Polls Gluroo API every 5 minutes and syncs to CosmosDB.
Enriches carb treatments with AI glycemic prediction.
Triggers ML learning after data sync (ISF, COB absorption, etc.)

Supports both:
- Legacy user-based sync (backward compatible)
- Profile-based sync via DataSourceManager
"""
import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import httpx
from azure.cosmos import CosmosClient

# Import food enrichment service for GI prediction
try:
    from services.food_enrichment_service import food_enrichment_service
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False
    food_enrichment_service = None

# Import enhanced ISF learner for auto-learning with smart clean bolus detection
try:
    from ml.training.enhanced_isf_learner import EnhancedISFLearner
    ISF_LEARNING_AVAILABLE = True
except ImportError:
    ISF_LEARNING_AVAILABLE = False
    EnhancedISFLearner = None

# Import treatment inference service for detecting unlogged treatments (carbs AND insulin)
try:
    from services.treatment_inference_service import get_treatment_inference_service
    TREATMENT_INFERENCE_AVAILABLE = True
except ImportError:
    TREATMENT_INFERENCE_AVAILABLE = False
    get_treatment_inference_service = None

# Import DataSourceManager for profile-based sync
try:
    from services.data_source_manager import get_data_source_manager, DataSourceManager
    PROFILE_SYNC_AVAILABLE = True
except ImportError:
    PROFILE_SYNC_AVAILABLE = False
    get_data_source_manager = None
    DataSourceManager = None

# Import ML data collector for prediction accuracy tracking
try:
    from services.ml_data_collector import MLDataCollector
    ML_DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    ML_DATA_COLLECTOR_AVAILABLE = False
    MLDataCollector = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gluroo_sync")

# Configuration - uses environment variables
GLUROO_URL = os.getenv("GLUROO_URL", "https://81ca.ns.gluroo.com")
GLUROO_API_SECRET = os.getenv("GLUROO_API_SECRET", "")
USER_ID = os.getenv("GLUROO_USER_ID", "")

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")
COSMOS_KEY = os.getenv("COSMOS_KEY", "")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "T1D-AI-DB")

POLL_INTERVAL = 300  # 5 minutes
FETCH_COUNT = 50  # Number of records to fetch per poll


class GlurooSyncService:
    """Service to sync data from Gluroo to CosmosDB."""

    def __init__(self):
        self.api_secret_hash = hashlib.sha1(GLUROO_API_SECRET.encode()).hexdigest()
        self.headers = {"API-SECRET": self.api_secret_hash}
        self.client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        self.db = self.client.get_database_client(COSMOS_DATABASE)
        self.glucose_container = self.db.get_container_client("glucose_readings")
        self.treatment_container = self.db.get_container_client("treatments")
        self.last_glucose_ms: Optional[int] = None
        self.last_treatment_ms: Optional[int] = None

        # Enhanced ISF learner for automatic learning with clean bolus detection
        self.isf_learner = EnhancedISFLearner() if ISF_LEARNING_AVAILABLE else None
        self.last_isf_learning: Optional[datetime] = None
        self.isf_learning_interval_hours = 6  # Re-learn ISF every 6 hours

        # Tandem pump awareness: when True, skip Gluroo insulin (pump is ground truth)
        self._tandem_active: Optional[bool] = None
        self._tandem_check_time: Optional[datetime] = None
        self.historic_data_imported = False  # Track if historic data has been imported

    def _is_tandem_active(self) -> bool:
        """Check if Tandem pump data source is active (has recent data).

        Caches the result for 30 minutes to avoid repeated queries.
        When active, Gluroo sync skips insulin treatments (pump is ground truth).
        """
        now = datetime.now(timezone.utc)
        if self._tandem_active is not None and self._tandem_check_time:
            if (now - self._tandem_check_time).total_seconds() < 1800:
                return self._tandem_active

        try:
            # Check for any Tandem-source treatment in last 24 hours
            since = (now - timedelta(hours=24)).isoformat()
            query = """
                SELECT TOP 1 c.id FROM c
                WHERE c.userId = @userId AND c.source = 'tandem'
                  AND c.timestamp >= @since
            """
            items = list(self.treatment_container.query_items(
                query=query,
                parameters=[
                    {"name": "@userId", "value": USER_ID},
                    {"name": "@since", "value": since},
                ],
                partition_key=USER_ID,
                max_item_count=1,
            ))
            self._tandem_active = len(items) > 0
            self._tandem_check_time = now
            if self._tandem_active:
                logger.info("Tandem pump active: Gluroo insulin treatments will be skipped")
            return self._tandem_active
        except Exception as e:
            logger.warning(f"Failed to check Tandem status: {e}")
            self._tandem_active = False
            self._tandem_check_time = now
            return False

    def _get_last_sync_time(self) -> int:
        """Get the timestamp of the most recent glucose reading in CosmosDB."""
        query = "SELECT TOP 1 c.timestamp FROM c ORDER BY c.timestamp DESC"
        items = list(self.glucose_container.query_items(
            query=query,
            partition_key=USER_ID,
            max_item_count=1
        ))
        if items:
            ts = items[0]['timestamp']
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
        return 0

    async def fetch_glucose(self, since_ms: int) -> list:
        """Fetch glucose entries from Gluroo API."""
        url = f"{GLUROO_URL}/api/v1/entries.json?count={FETCH_COUNT}&find[date][$gt]={since_ms}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json() or []
        return []

    async def fetch_treatments(self, since_ms: int) -> list:
        """Fetch treatments from Gluroo API."""
        all_treatments = []
        for event_type in ["Correction Bolus", "Carb Correction"]:
            url = (f"{GLUROO_URL}/api/v1/treatments.json?count={FETCH_COUNT}"
                   f"&find[eventType]={event_type.replace(' ', '%20')}"
                   f"&find[mills][$gt]={since_ms}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers)
                if response.status_code == 200:
                    all_treatments.extend(response.json() or [])
        return all_treatments

    def parse_glucose(self, entry: dict) -> Optional[dict]:
        """Parse Gluroo glucose entry to CosmosDB format."""
        sgv = entry.get('sgv')
        if sgv is None:
            return None

        date_ms = entry.get('date')
        if date_ms:
            ts = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
        elif entry.get('dateString'):
            ts = datetime.fromisoformat(entry['dateString'].replace('Z', '+00:00'))
        else:
            return None

        source_id = entry.get('_id', str(date_ms))
        return {
            "id": f"{USER_ID}_{source_id}",
            "userId": USER_ID,
            "timestamp": ts.isoformat(),
            "value": int(sgv),
            "trend": entry.get('direction', 'Flat'),
            "source": "gluroo",
            "sourceId": source_id
        }

    def parse_treatment(self, entry: dict) -> Optional[dict]:
        """Parse Gluroo treatment to CosmosDB format."""
        # Skip treatments that originated from T1D-AI to prevent duplicates
        # Note: tandem-sync entries are handled separately in sync_once() for note sync
        entered_by = entry.get('enteredBy', '')
        if entered_by == 'T1D-AI':
            logger.debug(f"Skipping T1D-AI originated treatment to prevent duplicate")
            return None

        insulin = entry.get('insulin')
        carbs = entry.get('carbs')

        # When Tandem pump is active, skip insulin from Gluroo (pump is ground truth)
        # Keep carb/food treatments - they often have richer descriptions from Gluroo
        if insulin and float(insulin) > 0 and not carbs and self._is_tandem_active():
            logger.debug(f"Skipping Gluroo insulin (Tandem pump is ground truth)")
            return None

        if not insulin and not carbs:
            return None

        created_at = entry.get('created_at')
        if created_at:
            ts = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            mills = entry.get('mills')
            if mills:
                ts = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)
            else:
                return None

        source_id = entry.get('_id', str(int(ts.timestamp() * 1000)))
        treatment_type = "insulin" if (insulin and float(insulin) > 0) else "carbs"

        # Try to get food name from various Gluroo fields
        notes = entry.get('notes') or entry.get('foodType') or entry.get('enteredBy') or ''

        # Log full entry for debugging unknown fields
        if carbs and not notes:
            logger.debug(f"Gluroo carb entry fields: {list(entry.keys())}")

        return {
            "id": f"{USER_ID}_{source_id}",
            "userId": USER_ID,
            "timestamp": ts.isoformat(),
            "type": treatment_type,
            "insulin": float(insulin) if insulin else None,
            "carbs": float(carbs) if carbs else None,
            "protein": float(entry.get('protein', 0)) if entry.get('protein') else None,
            "fat": float(entry.get('fat', 0)) if entry.get('fat') else None,
            "notes": notes,
            "source": "gluroo",
            "sourceId": source_id
        }

    def _get_existing_treatment(self, doc_id: str) -> Optional[dict]:
        """Check if a treatment already exists in CosmosDB."""
        try:
            existing = self.treatment_container.read_item(
                item=doc_id,
                partition_key=USER_ID
            )
            return existing
        except Exception:
            return None

    def _merge_treatment(self, existing: dict, new_doc: dict) -> dict:
        """
        Merge new Gluroo data with existing document, preserving user edits.

        Rules:
        - If document has 'userEdited' flag, preserve all user fields
        - If any editable field differs from original Gluroo sync, keep user's value
        - Preserve enrichment data (GI, absorption rate, etc.)
        """
        # Fields that users can edit and should be preserved
        editable_fields = ['notes', 'carbs', 'insulin', 'protein', 'fat',
                          'glycemicIndex', 'absorptionRate', 'isLiquid']

        # Enrichment fields to always preserve
        enrichment_fields = ['glycemicIndex', 'glycemicLoad', 'absorptionRate',
                            'fatContent', 'isLiquid', 'enrichedAt', 'fiber']

        # Check if document has been marked as user-edited
        user_edited = existing.get('userEdited', False)

        if user_edited:
            # User has explicitly edited - don't overwrite ANY user fields
            logger.info(f"Preserving user edits for treatment {existing['id']}")
            merged = existing.copy()
            merged['lastGlurooSync'] = datetime.now(timezone.utc).isoformat()
            return merged

        # Even without userEdited flag, preserve fields that differ from new Gluroo data
        # This handles cases where user edited but flag wasn't set
        merged = new_doc.copy()

        for field in editable_fields:
            existing_val = existing.get(field)
            new_val = new_doc.get(field)

            # If existing has a value and it's different, assume user edited it
            if existing_val is not None:
                # For notes, check if they're meaningfully different
                if field == 'notes':
                    if existing_val and existing_val != new_val:
                        logger.info(f"Preserving user-edited notes: '{existing_val}'")
                        merged[field] = existing_val
                # For numeric fields, check if different
                elif existing_val != new_val:
                    logger.info(f"Preserving user-edited {field}: {existing_val} (gluroo: {new_val})")
                    merged[field] = existing_val

        # Always preserve enrichment data
        for field in enrichment_fields:
            if field in existing and existing[field] is not None:
                merged[field] = existing[field]

        merged['lastGlurooSync'] = datetime.now(timezone.utc).isoformat()
        return merged

    async def _trigger_isf_learning(self):
        """
        Trigger enhanced ISF learning if enough time has passed.
        Uses smart clean bolus detection to learn accurate ISF.

        Features:
        - Imports historic bolus data on first run
        - Validates clean boluses (no undocumented carbs)
        - Learns time-of-day patterns
        - Tracks contextual features (lunar phase, day of year)
        """
        if not self.isf_learner:
            return

        now = datetime.now(timezone.utc)

        # Check if we should run learning (every N hours)
        if self.last_isf_learning:
            hours_since_learning = (now - self.last_isf_learning).total_seconds() / 3600
            if hours_since_learning < self.isf_learning_interval_hours:
                return

        try:
            # Import historic data on first run
            if not self.historic_data_imported:
                logger.info("Importing historic bolus data for ISF learning...")
                import_result = await self.isf_learner.import_historic_bolus_data(USER_ID)
                logger.info(
                    f"Historic import: {import_result.get('imported', 0)} total, "
                    f"{import_result.get('validated', 0)} clean, "
                    f"{import_result.get('rejected', 0)} rejected"
                )
                self.historic_data_imported = True

            # Learn from recent real-time data
            logger.info("Triggering enhanced ISF learning from real-time data...")
            learned = await self.isf_learner.learn_from_realtime_data(USER_ID, days=30)

            if learned:
                logger.info(
                    f"Learned ISF: {learned.value:.1f} "
                    f"(n={learned.sampleCount}, confidence={learned.confidence:.2f}, "
                    f"range={learned.minISF:.0f}-{learned.maxISF:.0f})"
                )

                # Log time-of-day patterns if available
                tod_pattern = learned.timeOfDayPattern
                if tod_pattern:
                    pattern_str = ", ".join(
                        f"{k}={v:.0f}" for k, v in tod_pattern.items() if v
                    )
                    if pattern_str:
                        logger.info(f"Time-of-day ISF pattern: {pattern_str}")
            else:
                logger.info("Not enough clean boluses for ISF learning yet")

            self.last_isf_learning = now

        except Exception as e:
            logger.warning(f"Enhanced ISF learning failed (non-critical): {e}")

    async def _collect_prediction_checkpoints(self):
        """
        Collect prediction checkpoints when new glucose data arrives.

        This fills in actualBg30/60/90 for pending MLTrainingDataPoints,
        enabling continuous comparison of predictions vs actual BG values.
        """
        if not ML_DATA_COLLECTOR_AVAILABLE:
            return

        try:
            collector = MLDataCollector()
            collected = await collector.collect_all_pending_checkpoints(USER_ID)

            if collected > 0:
                logger.info(f"Collected {collected} prediction checkpoints for accuracy tracking")

        except Exception as e:
            # Don't fail sync if checkpoint collection fails
            logger.debug(f"Prediction checkpoint collection failed (non-critical): {e}")

    async def _enrich_carb_treatment(self, doc: dict) -> dict:
        """
        Enrich carb treatment with GPT-4.1 macro estimation and GI prediction.

        If protein/fat are not provided from Gluroo but we have food notes,
        uses GPT-4.1 to estimate them from the food description.
        """
        try:
            carbs = doc.get('carbs', 0)
            protein = doc.get('protein', 0) or 0
            fat = doc.get('fat', 0) or 0
            notes = doc.get('notes', '')

            # Step 1: If protein/fat not provided and we have food notes, estimate them with GPT-4.1
            if notes and notes.strip() and (protein == 0 or fat == 0):
                try:
                    await food_enrichment_service.initialize()
                    macro_estimate = await food_enrichment_service.estimate_macros_from_description(
                        food_description=notes,
                        known_carbs=carbs
                    )
                    # Use estimated values if not already provided
                    if protein == 0:
                        protein = macro_estimate.protein_g
                        doc['protein'] = protein
                    if fat == 0:
                        fat = macro_estimate.fat_g
                        doc['fat'] = fat

                    logger.info(
                        f"AI macro estimation for '{notes[:40]}': "
                        f"protein={protein:.1f}g, fat={fat:.1f}g "
                        f"(confidence={macro_estimate.confidence:.2f})"
                    )
                except Exception as macro_err:
                    logger.warning(f"Macro estimation failed: {macro_err}")

            # Step 2: Get glycemic features (GI, absorption rate, etc.)
            features = await food_enrichment_service.extract_food_features(
                food_text=notes,
                carbs=carbs,
                protein=protein,
                fat=fat
            )

            # Add enrichment data to document
            doc['glycemicIndex'] = features.glycemic_index
            doc['glycemicLoad'] = features.glycemic_load
            doc['absorptionRate'] = features.absorption_rate
            doc['fatContent'] = features.fat_content
            doc['isLiquid'] = features.is_liquid
            doc['enrichedAt'] = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Enriched carb treatment: {notes or 'no description'} -> "
                f"GI={features.glycemic_index}, absorption={features.absorption_rate}, "
                f"protein={protein:.1f}g, fat={fat:.1f}g"
            )

        except Exception as e:
            logger.warning(f"Failed to enrich carb treatment: {e}")
            # Set default values on failure
            doc['glycemicIndex'] = 55  # Medium GI default
            doc['absorptionRate'] = 'medium'

        return doc

    def _sync_tandem_notes(self, gluroo_entry: dict):
        """
        Sync user-added notes from a tandem-sync entry in Gluroo back to the
        matching Tandem entry in CosmosDB. This lets users annotate pump boluses
        in Gluroo and have those notes appear in T1D-AI.
        """
        notes = gluroo_entry.get('notes', '') or gluroo_entry.get('foodType', '') or ''
        created_at = gluroo_entry.get('created_at', '')
        if not notes.strip() or not created_at:
            return

        try:
            query = """
                SELECT * FROM c
                WHERE c.userId = @userId AND c.source = 'tandem'
                  AND c.timestamp = @ts
            """
            items = list(self.treatment_container.query_items(
                query=query,
                parameters=[
                    {"name": "@userId", "value": USER_ID},
                    {"name": "@ts", "value": created_at},
                ],
                partition_key=USER_ID,
            ))
            for item in items:
                if notes != item.get('notes', ''):
                    item['notes'] = notes
                    self.treatment_container.upsert_item(item)
                    logger.info(f"Synced Gluroo notes to Tandem entry: '{notes[:50]}'")
        except Exception as e:
            logger.warning(f"Failed to sync tandem notes: {e}")

    async def sync_once(self) -> tuple[int, int]:
        """Perform one sync cycle. Returns (glucose_count, treatment_count)."""
        # Initialize last sync time if not set
        if self.last_glucose_ms is None:
            self.last_glucose_ms = self._get_last_sync_time()
            logger.info(f"Starting sync from timestamp: {self.last_glucose_ms}")

        if self.last_treatment_ms is None:
            self.last_treatment_ms = self.last_glucose_ms

        # Fetch new data
        glucose_entries = await self.fetch_glucose(self.last_glucose_ms)
        treatment_entries = await self.fetch_treatments(self.last_treatment_ms)

        glucose_count = 0
        treatment_count = 0
        max_glucose_ms = self.last_glucose_ms
        max_treatment_ms = self.last_treatment_ms

        # Process glucose entries
        for entry in glucose_entries:
            doc = self.parse_glucose(entry)
            if doc:
                try:
                    self.glucose_container.upsert_item(doc)
                    glucose_count += 1
                    date_ms = entry.get('date', 0)
                    if date_ms > max_glucose_ms:
                        max_glucose_ms = date_ms

                    # Compare this glucose reading against pending predictions
                    try:
                        from services.prediction_tracker import get_prediction_tracker
                        tracker = get_prediction_tracker()
                        ts = doc.get('timestamp')
                        if isinstance(ts, str):
                            actual_time = datetime.fromisoformat(ts.replace('Z', '+00:00')).replace(tzinfo=None)
                        else:
                            actual_time = ts
                        tracker.compare_with_actual(
                            user_id=doc.get('userId', USER_ID),
                            actual_timestamp=actual_time,
                            actual_bg=float(doc.get('value', 0)),
                        )
                    except Exception as track_err:
                        # Don't fail sync if tracking fails
                        logger.debug(f"Prediction tracking comparison failed: {track_err}")
                except Exception as e:
                    logger.error(f"Error upserting glucose: {e}")

        # Process treatments with GI enrichment and user edit preservation
        for entry in treatment_entries:
            # tandem-sync entries: sync user-added notes back to Tandem entries in CosmosDB
            if entry.get('enteredBy') == 'tandem-sync':
                self._sync_tandem_notes(entry)
                mills = entry.get('mills', 0)
                if mills > max_treatment_ms:
                    max_treatment_ms = mills
                continue

            doc = self.parse_treatment(entry)
            if doc:
                try:
                    # Check if treatment already exists in CosmosDB
                    existing = self._get_existing_treatment(doc['id'])

                    if existing:
                        # Merge new Gluroo data with existing, preserving user edits
                        doc = self._merge_treatment(existing, doc)
                    else:
                        # New treatment - enrich carb treatments with GI prediction
                        if doc.get('type') == 'carbs' and doc.get('carbs') and ENRICHMENT_AVAILABLE:
                            doc = await self._enrich_carb_treatment(doc)

                    self.treatment_container.upsert_item(doc)
                    treatment_count += 1
                    mills = entry.get('mills', 0)
                    if mills > max_treatment_ms:
                        max_treatment_ms = mills
                except Exception as e:
                    logger.error(f"Error upserting treatment: {e}")

        # Update last sync times
        self.last_glucose_ms = max_glucose_ms
        self.last_treatment_ms = max_treatment_ms

        # Trigger ML learning after sync (runs periodically, not every sync)
        if glucose_count > 0 or treatment_count > 0:
            await self._trigger_isf_learning()

        # Collect prediction checkpoints when new glucose data arrives
        # This fills in actualBg30/60/90 for pending training data points
        if glucose_count > 0:
            await self._collect_prediction_checkpoints()

        # DISABLED: Auto-inference was causing unwanted treatments to be created
        # The frontend toggles (autoSyncInsulin/autoSyncCarbs) do not control this backend logic
        # To re-enable, uncomment the lines below
        # if glucose_count > 0 and TREATMENT_INFERENCE_AVAILABLE:
        #     await self._check_for_unlogged_treatments()

        return glucose_count, treatment_count

    async def _check_for_unlogged_treatments(self):
        """
        Check recent glucose data for signs of unlogged treatments (carbs OR insulin).
        Uses curve fitting to find the best explanation for BG patterns.
        Creates inferred treatment entries that can be confirmed/edited by user.
        """
        try:
            from models.schemas import GlucoseReading, Treatment

            # Get recent glucose readings (last 3 hours for curve fitting)
            query = """
                SELECT * FROM c
                WHERE c.userId = @userId
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT 50
            """
            glucose_items = list(self.glucose_container.query_items(
                query=query,
                parameters=[{"name": "@userId", "value": USER_ID}],
                enable_cross_partition_query=True
            ))

            if len(glucose_items) < 15:
                return

            # Convert to GlucoseReading objects
            readings = []
            for item in glucose_items:
                try:
                    readings.append(GlucoseReading(
                        id=item['id'],
                        userId=item['userId'],
                        timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                        value=item['value'],
                        trend=item.get('trend'),
                        source=item.get('source', 'gluroo'),
                        sourceId=item.get('sourceId')
                    ))
                except Exception:
                    continue

            if len(readings) < 15:
                return

            # Get known treatments for context
            treatment_query = """
                SELECT * FROM c
                WHERE c.userId = @userId
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT 20
            """
            treatment_items = list(self.treatment_container.query_items(
                query=treatment_query,
                parameters=[{"name": "@userId", "value": USER_ID}],
                enable_cross_partition_query=True
            ))

            known_treatments = []
            for item in treatment_items:
                try:
                    from models.schemas import TreatmentType
                    t_type = item.get('type', 'carbs')
                    if t_type == 'insulin':
                        t_type = TreatmentType.INSULIN
                    else:
                        t_type = TreatmentType.CARBS

                    known_treatments.append(Treatment(
                        id=item['id'],
                        userId=item['userId'],
                        timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                        type=t_type,
                        insulin=item.get('insulin'),
                        carbs=item.get('carbs'),
                        source=item.get('source', 'gluroo'),
                        isInferred=item.get('isInferred', False)
                    ))
                except Exception:
                    continue

            # Use enhanced treatment inference
            inference_service = get_treatment_inference_service()
            inferred_treatments = await inference_service.detect_unlogged_treatments(
                user_id=USER_ID,
                recent_glucose=readings,
                known_treatments=[t for t in known_treatments if not t.isInferred]
            )

            for treatment in inferred_treatments:
                if treatment.carbs:
                    logger.info(
                        f"Detected unlogged carbs: estimated {treatment.carbs}g "
                        f"(confidence: {treatment.inferenceConfidence:.0%})"
                    )
                elif treatment.insulin:
                    logger.info(
                        f"Detected unlogged insulin: estimated {treatment.insulin}U "
                        f"(confidence: {treatment.inferenceConfidence:.0%})"
                    )

        except Exception as e:
            logger.warning(f"Treatment inference check failed (non-critical): {e}")

    async def run_forever(self):
        """Run continuous sync loop."""
        logger.info("Starting Gluroo sync service...")
        logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
        logger.info(f"User ID: {USER_ID}")

        while True:
            try:
                start = time.time()
                glucose_count, treatment_count = await self.sync_once()

                if glucose_count > 0 or treatment_count > 0:
                    logger.info(f"Synced {glucose_count} glucose, {treatment_count} treatments")
                else:
                    logger.debug("No new data")

                # Wait for next poll
                elapsed = time.time() - start
                sleep_time = max(0, POLL_INTERVAL - elapsed)
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Sync service stopped by user")
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(60)  # Wait before retry


class ProfileBasedSyncService:
    """
    Profile-based sync service using DataSourceManager.

    This service syncs data for all active profiles across all accounts,
    supporting multiple data sources per profile.
    """

    def __init__(self):
        self.data_source_manager = get_data_source_manager() if PROFILE_SYNC_AVAILABLE else None
        self.poll_interval = POLL_INTERVAL

    async def sync_once(self) -> Dict[str, Any]:
        """
        Perform one sync cycle for all profiles.

        Returns summary of sync results.
        """
        if not self.data_source_manager:
            logger.warning("DataSourceManager not available, skipping profile sync")
            return {"profiles": 0, "glucose": 0, "treatments": 0}

        try:
            results = await self.data_source_manager.sync_all_profiles()

            # Aggregate results
            total_profiles = len(results)
            total_glucose = sum(
                sum(r.glucose_count for r in profile_results)
                for profile_results in results.values()
            )
            total_treatments = sum(
                sum(r.treatment_count for r in profile_results)
                for profile_results in results.values()
            )

            return {
                "profiles": total_profiles,
                "glucose": total_glucose,
                "treatments": total_treatments,
                "details": results
            }

        except Exception as e:
            logger.error(f"Profile-based sync failed: {e}")
            return {"profiles": 0, "glucose": 0, "treatments": 0, "error": str(e)}

    async def run_forever(self):
        """Run continuous profile-based sync loop."""
        if not PROFILE_SYNC_AVAILABLE:
            logger.warning("Profile sync not available, DataSourceManager not imported")
            return

        logger.info("Starting profile-based sync service...")
        logger.info(f"Poll interval: {self.poll_interval} seconds")

        while True:
            try:
                start = time.time()
                result = await self.sync_once()

                if result.get("glucose", 0) > 0 or result.get("treatments", 0) > 0:
                    logger.info(
                        f"Profile sync: {result['profiles']} profiles, "
                        f"{result['glucose']} glucose, {result['treatments']} treatments"
                    )
                else:
                    logger.debug(f"Profile sync: {result['profiles']} profiles, no new data")

                # Wait for next poll
                elapsed = time.time() - start
                sleep_time = max(0, self.poll_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Profile sync service stopped by user")
                break
            except Exception as e:
                logger.error(f"Profile sync error: {e}")
                await asyncio.sleep(60)  # Wait before retry


async def run_combined_sync_loop():
    """
    Run combined sync loop for both legacy and profile-based syncing.

    This function handles both:
    1. Legacy user-based sync (for backward compatibility)
    2. Profile-based sync (new multi-profile support)
    """
    legacy_service = GlurooSyncService()
    profile_service = ProfileBasedSyncService()

    logger.info("Starting combined sync service (legacy + profile-based)...")
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")

    while True:
        try:
            start = time.time()

            # Run legacy sync (for users without profiles yet)
            try:
                legacy_glucose, legacy_treatments = await legacy_service.sync_once()
                if legacy_glucose > 0 or legacy_treatments > 0:
                    logger.info(f"Legacy sync: {legacy_glucose} glucose, {legacy_treatments} treatments")
            except Exception as e:
                logger.error(f"Legacy sync error: {e}")
                legacy_glucose, legacy_treatments = 0, 0

            # Run profile-based sync (for users with profiles)
            if PROFILE_SYNC_AVAILABLE:
                try:
                    profile_result = await profile_service.sync_once()
                    if profile_result.get("glucose", 0) > 0 or profile_result.get("treatments", 0) > 0:
                        logger.info(
                            f"Profile sync: {profile_result['profiles']} profiles, "
                            f"{profile_result['glucose']} glucose, {profile_result['treatments']} treatments"
                        )
                except Exception as e:
                    logger.error(f"Profile sync error: {e}")

            # Wait for next poll
            elapsed = time.time() - start
            sleep_time = max(0, POLL_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Combined sync service stopped by user")
            break
        except Exception as e:
            logger.error(f"Combined sync error: {e}")
            await asyncio.sleep(60)  # Wait before retry


async def main():
    """Main entry point."""
    service = GlurooSyncService()
    await service.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
