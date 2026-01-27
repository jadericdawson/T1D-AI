"""
Data Sources API Endpoints for T1D-AI
Manages connections to external data sources (Gluroo).
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel

from models.schemas import DataSource, GlurooCredentials, User, TreatmentType
from database.repositories import DataSourceRepository, GlucoseRepository, TreatmentRepository
from services.gluroo_service import GlurooService, create_gluroo_service
from services.food_enrichment_service import food_enrichment_service
from utils.encryption import encrypt_secret, decrypt_secret
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

datasource_repo = DataSourceRepository()
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()


def get_data_user_id(profile_id: str) -> str:
    """
    Get the userId to use for database queries.

    For 'self' profiles, the profile ID is 'profile_{user_id}' but data
    is stored with just the user_id.
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix (8 chars)
    return profile_id


class ConnectGlurooRequest(BaseModel):
    """Request to connect Gluroo account."""
    url: str
    apiSecret: str


class TestConnectionResponse(BaseModel):
    """Response for connection test."""
    success: bool
    message: str


class SyncResponse(BaseModel):
    """Response for data sync."""
    glucoseReadings: int
    treatments: int
    lastSyncAt: datetime


@router.post("/gluroo/test", response_model=TestConnectionResponse)
async def test_gluroo_connection(request: ConnectGlurooRequest):
    """
    Test connection to Gluroo API.

    Tests the provided URL and API secret without saving credentials.
    """
    try:
        service = GlurooService(request.url, request.apiSecret)
        success, message = await service.test_connection()

        return TestConnectionResponse(success=success, message=message)

    except Exception as e:
        logger.error(f"Error testing Gluroo connection: {e}")
        return TestConnectionResponse(success=False, message=str(e))


@router.post("/gluroo/connect", response_model=DataSource)
async def connect_gluroo(
    request: ConnectGlurooRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Connect Gluroo account.

    Saves encrypted credentials and performs initial data sync.
    """
    user_id = get_data_user_id(current_user.id)
    try:
        # Test connection first
        service = GlurooService(request.url, request.apiSecret)
        success, message = await service.test_connection()

        if not success:
            raise HTTPException(status_code=400, detail=f"Connection failed: {message}")

        # Encrypt the API secret for secure storage
        try:
            encrypted_secret = encrypt_secret(request.apiSecret)
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise HTTPException(status_code=500, detail="Failed to secure credentials")

        credentials = GlurooCredentials(
            url=request.url,
            apiSecretEncrypted=encrypted_secret,
            lastSyncAt=None,
            syncEnabled=True
        )

        datasource = DataSource(
            id=f"{user_id}_gluroo",
            userId=user_id,
            type="gluroo",
            credentials=credentials,
            createdAt=datetime.utcnow()
        )

        # Save to database
        saved = await datasource_repo.create(datasource)
        logger.info(f"Connected Gluroo for user {user_id}")

        return saved

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting Gluroo: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/gluroo")
async def disconnect_gluroo(current_user: User = Depends(get_current_user)):
    """
    Disconnect Gluroo account.

    Removes stored credentials but does not delete synced data.
    """
    user_id = get_data_user_id(current_user.id)
    try:
        deleted = await datasource_repo.delete(user_id, "gluroo")
        if not deleted:
            raise HTTPException(status_code=404, detail="Gluroo connection not found")

        logger.info(f"Disconnected Gluroo for user {user_id}")
        return {"message": "Gluroo disconnected successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting Gluroo: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/gluroo/sync", response_model=SyncResponse)
async def sync_gluroo(
    full_sync: bool = Query(default=False, description="Perform full sync instead of incremental"),
    current_user: User = Depends(get_current_user)
):
    """
    Sync data from Gluroo.

    Fetches new glucose readings and treatments from Gluroo API.
    Performs incremental sync by default (since last sync).
    Use full_sync=true to fetch all available data.
    """
    user_id = get_data_user_id(current_user.id)
    try:
        # Get stored credentials
        datasource = await datasource_repo.get(user_id, "gluroo")
        if not datasource:
            raise HTTPException(status_code=404, detail="Gluroo not connected")

        if not datasource.credentials or not datasource.credentials.apiSecretEncrypted:
            raise HTTPException(status_code=400, detail="Invalid stored credentials")

        # Decrypt the API secret
        try:
            api_secret = decrypt_secret(datasource.credentials.apiSecretEncrypted)
        except ValueError as e:
            logger.error(f"Failed to decrypt credentials for user {user_id}: {e}")
            raise HTTPException(
                status_code=400,
                detail="Cannot decrypt stored credentials. Please reconnect Gluroo."
            )

        # Create Gluroo service
        service = GlurooService(datasource.credentials.url, api_secret)

        # Determine sync window
        since = None
        if not full_sync and datasource.credentials.lastSyncAt:
            # Incremental sync - fetch since last sync minus 1 hour buffer
            since = datasource.credentials.lastSyncAt - timedelta(hours=1)
            logger.info(f"Incremental sync for user {user_id} since {since}")
        else:
            # Full sync - fetch last 30 days (or max Gluroo provides)
            since = datetime.now(timezone.utc) - timedelta(days=30)
            logger.info(f"Full sync for user {user_id} since {since}")

        # Fetch glucose readings
        glucose_readings = await service.fetch_glucose_entries(
            user_id=user_id,
            count=1000,
            since=since
        )

        # Fetch treatments (insulin and carbs)
        treatments = await service.fetch_all_treatments(
            user_id=user_id,
            count=500,
            since=since
        )

        # Store glucose readings in CosmosDB (upsert to avoid duplicates)
        glucose_count = 0
        for reading in glucose_readings:
            try:
                await glucose_repo.upsert(reading)
                glucose_count += 1
            except Exception as e:
                logger.warning(f"Failed to store glucose reading: {e}")

        # Store treatments in CosmosDB (upsert to avoid duplicates)
        # Enrich carb treatments with GPT-4.1 macro estimation
        treatment_count = 0
        for treatment in treatments:
            try:
                # Enrich carb treatments that have food notes but missing protein/fat
                if (treatment.type == TreatmentType.CARBS and
                    treatment.carbs and
                    treatment.notes and treatment.notes.strip() and
                    (not treatment.protein or not treatment.fat)):
                    try:
                        await food_enrichment_service.initialize()

                        # Step 1: Estimate protein/fat from food description
                        macro_estimate = await food_enrichment_service.estimate_macros_from_description(
                            food_description=treatment.notes,
                            known_carbs=treatment.carbs
                        )
                        if not treatment.protein:
                            treatment.protein = macro_estimate.protein_g
                        if not treatment.fat:
                            treatment.fat = macro_estimate.fat_g

                        logger.info(
                            f"AI macro estimation for '{treatment.notes[:40]}': "
                            f"protein={treatment.protein:.1f}g, fat={treatment.fat:.1f}g"
                        )

                        # Step 2: Get glycemic features
                        features = await food_enrichment_service.extract_food_features(
                            food_text=treatment.notes,
                            carbs=treatment.carbs,
                            protein=treatment.protein or 0,
                            fat=treatment.fat or 0
                        )
                        treatment.glycemicIndex = features.glycemic_index
                        treatment.glycemicLoad = features.glycemic_load
                        treatment.absorptionRate = features.absorption_rate
                        treatment.fatContent = features.fat_content
                        treatment.isLiquid = features.is_liquid
                        treatment.enrichedAt = datetime.now(timezone.utc)

                        logger.info(
                            f"Enriched Gluroo treatment: GI={features.glycemic_index}, "
                            f"absorption={features.absorption_rate}"
                        )
                    except Exception as enrich_err:
                        logger.warning(f"Failed to enrich treatment: {enrich_err}")

                await treatment_repo.upsert(treatment)
                treatment_count += 1
            except Exception as e:
                logger.warning(f"Failed to store treatment: {e}")

        # Update last sync timestamp
        now = datetime.now(timezone.utc)
        datasource.credentials.lastSyncAt = now
        await datasource_repo.update(datasource)

        logger.info(
            f"Sync completed for user {user_id}: "
            f"{glucose_count} glucose readings, {treatment_count} treatments"
        )

        return SyncResponse(
            glucoseReadings=glucose_count,
            treatments=treatment_count,
            lastSyncAt=now
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing Gluroo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/gluroo/status")
async def get_gluroo_status(current_user: User = Depends(get_current_user)):
    """
    Get Gluroo connection status.
    """
    user_id = get_data_user_id(current_user.id)
    try:
        datasource = await datasource_repo.get(user_id, "gluroo")

        if not datasource:
            return {
                "connected": False,
                "url": None,
                "lastSyncAt": None,
                "syncEnabled": False
            }

        return {
            "connected": True,
            "url": datasource.credentials.url,
            "lastSyncAt": datasource.credentials.lastSyncAt,
            "syncEnabled": datasource.credentials.syncEnabled
        }

    except Exception as e:
        logger.error(f"Error getting Gluroo status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/gluroo/defaults")
async def get_gluroo_defaults(current_user: User = Depends(get_current_user)):
    """
    Get default Gluroo configuration for auto-fill.

    Only returns credentials for the owner's account (hardcoded in gluroo_sync.py).
    Other users get empty defaults.
    """
    try:
        # Import the hardcoded config from gluroo_sync
        from services.gluroo_sync import GLUROO_URL, GLUROO_API_SECRET, USER_ID as OWNER_USER_ID

        # Only return credentials for the owner's account
        if current_user.id == OWNER_USER_ID:
            return {
                "url": GLUROO_URL,
                "apiSecret": GLUROO_API_SECRET,
                "syncInterval": 5,
                "isOwner": True
            }
        else:
            # Return empty defaults for other users
            return {
                "url": "",
                "apiSecret": "",
                "syncInterval": 5,
                "isOwner": False
            }

    except ImportError:
        # If gluroo_sync module is not available, return empty defaults
        return {
            "url": "",
            "apiSecret": "",
            "syncInterval": 5,
            "isOwner": False
        }
    except Exception as e:
        logger.error(f"Error getting Gluroo defaults: {e}")
        return {
            "url": "",
            "apiSecret": "",
            "syncInterval": 5,
            "isOwner": False
        }
