"""
Data Sources API Endpoints for T1D-AI
Manages connections to external data sources (Gluroo).
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

from models.schemas import DataSource, GlurooCredentials
from database.repositories import DataSourceRepository, GlucoseRepository, TreatmentRepository
from services.gluroo_service import GlurooService, create_gluroo_service

logger = logging.getLogger(__name__)
router = APIRouter()

datasource_repo = DataSourceRepository()
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()

# Temporary: hardcoded user ID until auth is implemented
TEMP_USER_ID = "demo_user"


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
    user_id: str = Query(default=TEMP_USER_ID)
):
    """
    Connect Gluroo account.

    Saves encrypted credentials and performs initial data sync.
    """
    try:
        # Test connection first
        service = GlurooService(request.url, request.apiSecret)
        success, message = await service.test_connection()

        if not success:
            raise HTTPException(status_code=400, detail=f"Connection failed: {message}")

        # TODO: Encrypt the API secret before storing
        # For now, we'll hash it (in production, use proper encryption)
        import hashlib
        encrypted_secret = hashlib.sha256(request.apiSecret.encode()).hexdigest()

        credentials = GlurooCredentials(
            url=request.url,
            apiSecretEncrypted=encrypted_secret,  # Should be properly encrypted
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
async def disconnect_gluroo(user_id: str = Query(default=TEMP_USER_ID)):
    """
    Disconnect Gluroo account.

    Removes stored credentials but does not delete synced data.
    """
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
    user_id: str = Query(default=TEMP_USER_ID),
    full_sync: bool = Query(default=False, description="Perform full sync instead of incremental")
):
    """
    Sync data from Gluroo.

    Fetches new glucose readings and treatments from Gluroo API.
    """
    try:
        # Get stored credentials
        datasource = await datasource_repo.get(user_id, "gluroo")
        if not datasource:
            raise HTTPException(status_code=404, detail="Gluroo not connected")

        # TODO: Decrypt the API secret
        # For now, this won't work without the original secret
        # In production, use proper encryption/decryption

        raise HTTPException(
            status_code=501,
            detail="Sync requires decryption of stored credentials (not yet implemented)"
        )

        # When implemented, this would:
        # 1. Decrypt the stored API secret
        # 2. Create GlurooService with decrypted credentials
        # 3. Fetch glucose and treatment data
        # 4. Store in CosmosDB
        # 5. Update lastSyncAt

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing Gluroo: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/gluroo/status")
async def get_gluroo_status(user_id: str = Query(default=TEMP_USER_ID)):
    """
    Get Gluroo connection status.
    """
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
