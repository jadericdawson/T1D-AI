"""
Profiles API Endpoints

CRUD operations for managed profiles and their data sources.
Enables one account to manage multiple people's diabetes data.
"""
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from auth import get_current_user
from models.schemas import (
    User,
    ManagedProfile,
    ManagedProfileCreate,
    ManagedProfileUpdate,
    ProfileDataSource,
    ProfileDataSourceCreate,
    ProfileDataSourceUpdate,
    ProfileSummary,
    ProfileSettings,
    ProfileRelationship,
    DataSourceType,
    SyncStatus
)
from database.repositories import ProfileRepository, ProfileDataSourceRepository
from utils.encryption import encrypt_secret, decrypt_secret

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profiles", tags=["profiles"])


# Response models
class ProfileListResponse(BaseModel):
    """Response for listing profiles."""
    profiles: List[ManagedProfile]
    total: int


class ProfileSummaryResponse(BaseModel):
    """Response for profile summaries (for dropdown selector)."""
    profiles: List[ProfileSummary]
    activeProfileId: Optional[str] = None


class DataSourceListResponse(BaseModel):
    """Response for listing data sources."""
    sources: List[ProfileDataSource]
    total: int


# ==================== Profile CRUD ====================

@router.get("", response_model=ProfileListResponse)
async def list_profiles(
    include_inactive: bool = False,
    user: User = Depends(get_current_user)
):
    """
    List all profiles managed by the current account.

    Returns profiles where the current user's account is the owner.
    """
    profile_repo = ProfileRepository()
    profiles = await profile_repo.get_by_account(user.id, include_inactive=include_inactive)

    return ProfileListResponse(
        profiles=profiles,
        total=len(profiles)
    )


@router.get("/summaries", response_model=ProfileSummaryResponse)
async def get_profile_summaries(
    user: User = Depends(get_current_user)
):
    """
    Get profile summaries for the profile selector dropdown.

    Returns lightweight profile data suitable for UI selection.
    Auto-creates a 'self' profile for existing users who don't have one.
    """
    profile_repo = ProfileRepository()

    # Check if user has any profiles - auto-create "self" profile if not
    self_profile = await profile_repo.get_self_profile(user.id)
    if not self_profile:
        # Auto-create self profile for existing user
        logger.info(f"Auto-creating 'self' profile for user {user.id}")
        display_name = user.displayName or user.email.split('@')[0]
        self_profile = ManagedProfile(
            id=f"profile_{user.id}",
            accountId=user.id,
            displayName=display_name,
            relationship=ProfileRelationship.SELF,
            diabetesType="T1D",
            settings=ProfileSettings(),
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc)
        )
        try:
            self_profile = await profile_repo.create(self_profile)
            logger.info(f"Auto-created 'self' profile {self_profile.id} for user {user.id}")
        except Exception as e:
            logger.error(f"Failed to auto-create profile: {e}")
            # Continue anyway - return empty summaries

    summaries = await profile_repo.get_summaries(user.id)
    active_profile_id = self_profile.id if self_profile else (summaries[0].id if summaries else None)

    return ProfileSummaryResponse(
        profiles=summaries,
        activeProfileId=active_profile_id
    )


@router.post("", response_model=ManagedProfile, status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile_data: ManagedProfileCreate,
    user: User = Depends(get_current_user)
):
    """
    Create a new managed profile.

    Use this to add a new person's diabetes data to manage
    (e.g., adding a child's data to a parent's account).
    """
    profile_repo = ProfileRepository()

    # Check if trying to create a "self" profile when one already exists
    if profile_data.relationship == ProfileRelationship.SELF:
        existing_self = await profile_repo.get_self_profile(user.id)
        if existing_self:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A 'self' profile already exists for this account"
            )

    # Create the profile
    profile = ManagedProfile(
        id=str(uuid.uuid4()),
        accountId=user.id,
        displayName=profile_data.displayName,
        relationship=profile_data.relationship,
        diabetesType=profile_data.diabetesType,
        dateOfBirth=profile_data.dateOfBirth,
        diagnosisDate=profile_data.diagnosisDate,
        settings=profile_data.settings or ProfileSettings(),
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc)
    )

    try:
        created = await profile_repo.create(profile)
        logger.info(f"Created profile {created.id} for account {user.id}")
        return created
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{profile_id}", response_model=ManagedProfile)
async def get_profile(
    profile_id: str,
    user: User = Depends(get_current_user)
):
    """Get a specific profile by ID."""
    profile_repo = ProfileRepository()
    profile = await profile_repo.get_by_id(profile_id, user.id)

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    return profile


@router.put("/{profile_id}", response_model=ManagedProfile)
async def update_profile(
    profile_id: str,
    updates: ManagedProfileUpdate,
    user: User = Depends(get_current_user)
):
    """Update a profile."""
    profile_repo = ProfileRepository()

    # Check profile exists
    existing = await profile_repo.get_by_id(profile_id, user.id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Note: We allow changing relationship (including from "self" to another type)
    # This supports the use case where a parent creates their account but
    # the primary profile is actually for their child's diabetes data

    # Build updates dict excluding None values
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        return existing

    try:
        updated = await profile_repo.update(profile_id, user.id, update_dict)
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    profile_id: str,
    hard_delete: bool = False,
    user: User = Depends(get_current_user)
):
    """
    Delete a profile.

    By default, performs a soft delete (marks as inactive).
    Use hard_delete=true to permanently remove the profile.

    Note: Cannot delete a "self" profile.
    """
    profile_repo = ProfileRepository()

    # Check profile exists
    existing = await profile_repo.get_by_id(profile_id, user.id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Prevent deleting "self" profile
    if existing.relationship == ProfileRelationship.SELF:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a 'self' profile"
        )

    success = await profile_repo.delete(profile_id, user.id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile"
        )


# ==================== Data Source CRUD ====================

@router.get("/{profile_id}/sources", response_model=DataSourceListResponse)
async def list_data_sources(
    profile_id: str,
    include_inactive: bool = False,
    user: User = Depends(get_current_user)
):
    """List all data sources for a profile."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    sources = await source_repo.get_by_profile(
        profile_id,
        user.id,
        include_inactive=include_inactive
    )

    return DataSourceListResponse(
        sources=sources,
        total=len(sources)
    )


@router.post("/{profile_id}/sources", response_model=ProfileDataSource, status_code=status.HTTP_201_CREATED)
async def add_data_source(
    profile_id: str,
    source_data: ProfileDataSourceCreate,
    user: User = Depends(get_current_user)
):
    """
    Add a data source to a profile.

    Encrypts credentials before storage.
    """
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Check if source type already exists for this profile
    existing = await source_repo.get_by_type(profile_id, source_data.sourceType.value)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A {source_data.sourceType.value} data source already exists for this profile"
        )

    # Encrypt credentials
    try:
        credentials_json = json.dumps(source_data.credentials)
        encrypted_credentials = encrypt_secret(credentials_json)
    except Exception as e:
        logger.error(f"Failed to encrypt credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to securely store credentials"
        )

    # Create the data source
    source = ProfileDataSource(
        id=f"{profile_id}_{source_data.sourceType.value}",
        profileId=profile_id,
        sourceType=source_data.sourceType,
        credentialsEncrypted=encrypted_credentials,
        priority=source_data.priority,
        providesGlucose=source_data.providesGlucose,
        providesTreatments=source_data.providesTreatments,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc)
    )

    try:
        created = await source_repo.create(source)

        # Add source to profile's dataSourceIds
        await profile_repo.add_data_source(profile_id, user.id, created.id)

        # Set as primary source if it's the first of its type
        updates = {}
        if source_data.providesGlucose and not profile.primaryGlucoseSourceId:
            updates['primaryGlucoseSourceId'] = created.id
        if source_data.providesTreatments and not profile.primaryTreatmentSourceId:
            updates['primaryTreatmentSourceId'] = created.id

        if updates:
            await profile_repo.update(profile_id, user.id, updates)

        logger.info(f"Added {source_data.sourceType.value} source to profile {profile_id}")
        return created

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{profile_id}/sources/{source_id}", response_model=ProfileDataSource)
async def get_data_source(
    profile_id: str,
    source_id: str,
    user: User = Depends(get_current_user)
):
    """Get a specific data source."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    source = await source_repo.get_by_id(source_id, profile_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    return source


@router.put("/{profile_id}/sources/{source_id}", response_model=ProfileDataSource)
async def update_data_source(
    profile_id: str,
    source_id: str,
    updates: ProfileDataSourceUpdate,
    user: User = Depends(get_current_user)
):
    """Update a data source."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Check source exists
    existing = await source_repo.get_by_id(source_id, profile_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    # Build updates dict excluding None values
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        return existing

    try:
        updated = await source_repo.update(source_id, profile_id, update_dict)
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{profile_id}/sources/{source_id}/credentials")
async def update_data_source_credentials(
    profile_id: str,
    source_id: str,
    credentials: dict,
    user: User = Depends(get_current_user)
):
    """Update credentials for a data source."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Check source exists
    existing = await source_repo.get_by_id(source_id, profile_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    # Encrypt new credentials
    try:
        credentials_json = json.dumps(credentials)
        encrypted_credentials = encrypt_secret(credentials_json)
    except Exception as e:
        logger.error(f"Failed to encrypt credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to securely store credentials"
        )

    try:
        updated = await source_repo.update_credentials(
            source_id,
            profile_id,
            encrypted_credentials
        )
        return {"status": "ok", "message": "Credentials updated successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{profile_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_source(
    profile_id: str,
    source_id: str,
    hard_delete: bool = False,
    user: User = Depends(get_current_user)
):
    """
    Delete a data source.

    By default, deactivates the source instead of deleting.
    Use hard_delete=true to permanently remove.
    """
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Check source exists
    existing = await source_repo.get_by_id(source_id, profile_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    if hard_delete:
        success = await source_repo.delete(source_id, profile_id)
        if success:
            # Remove from profile's dataSourceIds
            await profile_repo.remove_data_source(profile_id, user.id, source_id)
    else:
        await source_repo.deactivate(source_id, profile_id)


@router.post("/{profile_id}/sources/{source_id}/sync")
async def trigger_sync(
    profile_id: str,
    source_id: str,
    user: User = Depends(get_current_user)
):
    """
    Trigger an immediate sync for a data source.

    This queues a sync job that will run as soon as possible.
    """
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Check source exists
    source = await source_repo.get_by_id(source_id, profile_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    if not source.isActive:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot sync an inactive data source"
        )

    # TODO: Implement actual sync trigger via DataSourceManager
    # For now, just update status to pending
    await source_repo.update_sync_status(source_id, profile_id, SyncStatus.PENDING.value)

    return {
        "status": "ok",
        "message": f"Sync queued for {source.sourceType.value}",
        "sourceId": source_id
    }


# ==================== Primary Source Management ====================

@router.put("/{profile_id}/primary-glucose-source")
async def set_primary_glucose_source(
    profile_id: str,
    source_id: str,
    user: User = Depends(get_current_user)
):
    """Set the primary glucose data source for a profile."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Verify source exists and provides glucose
    source = await source_repo.get_by_id(source_id, profile_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    if not source.providesGlucose:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This data source does not provide glucose data"
        )

    await profile_repo.update(profile_id, user.id, {'primaryGlucoseSourceId': source_id})

    return {"status": "ok", "primaryGlucoseSourceId": source_id}


@router.put("/{profile_id}/primary-treatment-source")
async def set_primary_treatment_source(
    profile_id: str,
    source_id: str,
    user: User = Depends(get_current_user)
):
    """Set the primary treatment data source for a profile."""
    profile_repo = ProfileRepository()
    source_repo = ProfileDataSourceRepository()

    # Verify profile ownership
    profile = await profile_repo.get_by_id(profile_id, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile {profile_id} not found"
        )

    # Verify source exists and provides treatments
    source = await source_repo.get_by_id(source_id, profile_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {source_id} not found"
        )

    if not source.providesTreatments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This data source does not provide treatment data"
        )

    await profile_repo.update(profile_id, user.id, {'primaryTreatmentSourceId': source_id})

    return {"status": "ok", "primaryTreatmentSourceId": source_id}
