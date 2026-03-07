"""
Data Sources API Endpoints for T1D-AI
Manages connections to external data sources (Gluroo, Tandem).
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
    Strip profile_ prefix for data access.

    Data is stored under base user ID without prefix.
    Profile IDs like 'profile_05bf...' map to data under '05bf...'
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix
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


# ==================== Tandem Endpoints ====================


class ConnectTandemRequest(BaseModel):
    """Request to connect Tandem account."""
    email: str
    password: str


@router.post("/tandem/test", response_model=TestConnectionResponse)
async def test_tandem_connection(request: ConnectTandemRequest):
    """
    Test connection to Tandem API.

    Validates credentials and confirms pump device is found.
    """
    try:
        from services.tandem_sync_service import TandemSyncService
        service = TandemSyncService()
        success, message = await service.test_connection(request.email, request.password)
        return TestConnectionResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Error testing Tandem connection: {e}")
        return TestConnectionResponse(success=False, message=str(e))


@router.post("/tandem/connect")
async def connect_tandem(
    request: ConnectTandemRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Connect Tandem account.

    Tests credentials, encrypts password, and creates a ProfileDataSource.
    """
    from services.tandem_sync_service import TandemSyncService
    from database.repositories import ProfileDataSourceRepository
    from models.schemas import DataSourceType
    import json

    user_id = get_data_user_id(current_user.id)
    profile_id = current_user.id  # Full profile ID (with prefix) for data source lookup

    # Test connection first
    service = TandemSyncService()
    success, message = await service.test_connection(request.email, request.password)
    if not success:
        raise HTTPException(status_code=400, detail=f"Connection failed: {message}")

    # Encrypt password
    try:
        encrypted_password = encrypt_secret(request.password)
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(status_code=500, detail="Failed to secure credentials")

    # Store credentials as JSON with encrypted password
    credentials_json = json.dumps({
        "email": request.email,
        "password": encrypted_password,
    })

    # Create ProfileDataSource — use full profile ID to match get_by_profile queries
    source_repo = ProfileDataSourceRepository()
    source_id = f"{profile_id}_tandem"

    source_data = {
        "id": source_id,
        "profileId": profile_id,
        "sourceType": DataSourceType.TANDEM.value,
        "credentialsEncrypted": credentials_json,
        "isActive": True,
        "syncEnabled": True,
        "priority": 2,
        "providesGlucose": False,
        "providesTreatments": True,
        "lastSyncAt": None,
        "lastSyncStatus": "pending",
        "userId": user_id,
    }

    try:
        source_repo.container.upsert_item(body=source_data)
        logger.info(f"Connected Tandem for user {user_id}")
        return {"message": "Tandem connected successfully", "sourceId": source_id}
    except Exception as e:
        logger.error(f"Error saving Tandem source: {e}")
        raise HTTPException(status_code=500, detail="Failed to save credentials")


@router.delete("/tandem")
async def disconnect_tandem(current_user: User = Depends(get_current_user)):
    """
    Disconnect Tandem account.

    Removes stored credentials but does not delete synced data.
    """
    from database.repositories import ProfileDataSourceRepository

    profile_id = current_user.id
    source_id = f"{profile_id}_tandem"

    try:
        source_repo = ProfileDataSourceRepository()
        source_repo.container.delete_item(item=source_id, partition_key=profile_id)
        logger.info(f"Disconnected Tandem for {profile_id}")
        return {"message": "Tandem disconnected successfully"}
    except Exception as e:
        logger.error(f"Error disconnecting Tandem: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tandem/status")
async def get_tandem_status(current_user: User = Depends(get_current_user)):
    """Get Tandem connection status."""
    from database.repositories import ProfileDataSourceRepository

    profile_id = current_user.id
    source_id = f"{profile_id}_tandem"

    try:
        source_repo = ProfileDataSourceRepository()
        try:
            item = source_repo.container.read_item(item=source_id, partition_key=profile_id)
            return {
                "connected": True,
                "lastSyncAt": item.get("lastSyncAt"),
                "syncEnabled": item.get("syncEnabled", True),
                "lastSyncStatus": item.get("lastSyncStatus", "pending"),
            }
        except Exception:
            return {
                "connected": False,
                "lastSyncAt": None,
                "syncEnabled": False,
                "lastSyncStatus": None,
            }
    except Exception as e:
        logger.error(f"Error getting Tandem status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tandem/sync")
async def sync_tandem(
    full_sync: bool = Query(default=False, description="Perform full sync (7 days) instead of incremental"),
    current_user: User = Depends(get_current_user)
):
    """
    Manually trigger a Tandem sync.
    """
    from services.tandem_sync_service import TandemSyncService
    from database.repositories import ProfileDataSourceRepository
    from utils.encryption import decrypt_secret as decrypt_value
    import json

    user_id = get_data_user_id(current_user.id)
    profile_id = current_user.id
    source_id = f"{profile_id}_tandem"

    # Get stored credentials
    source_repo = ProfileDataSourceRepository()
    try:
        item = source_repo.container.read_item(item=source_id, partition_key=profile_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Tandem not connected")

    creds = json.loads(item.get("credentialsEncrypted", "{}"))
    email = creds.get("email", "")
    encrypted_password = creds.get("password", "")

    if not email or not encrypted_password:
        raise HTTPException(status_code=400, detail="Invalid stored credentials")

    try:
        password = decrypt_value(encrypted_password)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Cannot decrypt stored credentials. Please reconnect Tandem."
        )

    # Always 7-day lookback — Tandem API returns a fixed event block regardless
    # of range, so broader lookback catches site changes and cartridge fills
    since = datetime.now(timezone.utc) - timedelta(days=7)

    # Look up Gluroo credentials so Tandem sync can push bolus/carbs to Gluroo
    gluroo_url = None
    gluroo_api_secret = None
    try:
        gluroo_ds = await datasource_repo.get(user_id, "gluroo")
        if gluroo_ds and gluroo_ds.credentials and gluroo_ds.credentials.apiSecretEncrypted:
            from utils.encryption import decrypt_secret as decrypt_gluroo
            gluroo_url = gluroo_ds.credentials.url
            gluroo_api_secret = decrypt_gluroo(gluroo_ds.credentials.apiSecretEncrypted)
    except Exception as e:
        logger.warning(f"Could not look up Gluroo credentials for Tandem push: {e}")

    service = TandemSyncService()
    try:
        result = await service.sync_for_source(
            email=email,
            password=password,
            user_id=user_id,
            since=since,
            gluroo_url=gluroo_url,
            gluroo_api_secret=gluroo_api_secret,
        )

        # Update last sync timestamp
        item["lastSyncAt"] = datetime.now(timezone.utc).isoformat()
        item["lastSyncStatus"] = "ok"
        source_repo.container.upsert_item(body=item)

        return {
            "basal": result["basal"],
            "bolus": result["bolus"],
            "carbs": result["carbs"],
            "total": result["total"],
            "lastSyncAt": item["lastSyncAt"],
        }
    except Exception as e:
        logger.error(f"Tandem sync failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
