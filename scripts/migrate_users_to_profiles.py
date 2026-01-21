#!/usr/bin/env python3
"""
Migration Script: Migrate existing users to profile model.

This script creates a 'self' profile for each existing user and migrates their
Gluroo/data source credentials to the new ProfileDataSource model.

Usage:
    python scripts/migrate_users_to_profiles.py [--dry-run]

Options:
    --dry-run    Preview changes without actually migrating
"""
import asyncio
import argparse
import json
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

# Add backend/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

from config import get_settings
from database.cosmos_client import get_cosmos_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_all_users(cosmos):
    """Get all users from CosmosDB."""
    users_container = cosmos._containers.get("users")
    if not users_container:
        logger.error("Users container not found")
        return []

    query = "SELECT * FROM c WHERE c.type = 'user' OR NOT IS_DEFINED(c.type)"
    users = []

    async for item in users_container.query_items(
        query=query,
        enable_cross_partition_query=True
    ):
        users.append(item)

    return users


async def get_user_settings(cosmos, user_id: str):
    """Get user settings from CosmosDB."""
    settings_container = cosmos._containers.get("settings")
    if not settings_container:
        return None

    query = "SELECT * FROM c WHERE c.userId = @userId"
    params = [{"name": "@userId", "value": user_id}]

    try:
        async for item in settings_container.query_items(
            query=query,
            parameters=params,
            partition_key=user_id
        ):
            return item
    except Exception as e:
        logger.warning(f"Could not get settings for user {user_id}: {e}")

    return None


async def get_user_datasource(cosmos, user_id: str):
    """Get user's existing Gluroo/Nightscout data source credentials."""
    datasources_container = cosmos._containers.get("datasources")
    if not datasources_container:
        return None

    query = "SELECT * FROM c WHERE c.userId = @userId"
    params = [{"name": "@userId", "value": user_id}]

    try:
        async for item in datasources_container.query_items(
            query=query,
            parameters=params,
            partition_key=user_id
        ):
            return item
    except Exception as e:
        logger.warning(f"Could not get datasource for user {user_id}: {e}")

    return None


async def profile_exists(cosmos, account_id: str, relationship: str = "self") -> bool:
    """Check if a profile already exists for this account."""
    profiles_container = cosmos._containers.get("profiles")
    if not profiles_container:
        return False

    query = "SELECT VALUE COUNT(1) FROM c WHERE c.accountId = @accountId AND c.relationship = @relationship"
    params = [
        {"name": "@accountId", "value": account_id},
        {"name": "@relationship", "value": relationship}
    ]

    try:
        async for count in profiles_container.query_items(
            query=query,
            parameters=params,
            partition_key=account_id
        ):
            return count > 0
    except Exception:
        pass

    return False


async def create_profile(cosmos, user: dict, settings: Optional[dict], dry_run: bool = False):
    """Create a 'self' profile for a user."""
    profiles_container = cosmos._containers.get("profiles")
    if not profiles_container:
        logger.error("Profiles container not found")
        return None

    user_id = user.get("id")
    email = user.get("email", "")
    display_name = user.get("displayName") or email.split("@")[0]

    profile_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Build profile settings from user settings
    profile_settings = {
        "targetBg": 100,
        "isf": 50,
        "icr": 10,
        "insulinDuration": 180,
        "highThreshold": 180,
        "lowThreshold": 70,
        "criticalHigh": 250,
        "criticalLow": 54,
        "timezone": "America/New_York"
    }

    if settings:
        profile_settings.update({
            "targetBg": settings.get("targetBg", 100),
            "isf": settings.get("isf", 50),
            "icr": settings.get("icr", 10),
            "insulinDuration": settings.get("insulinDuration", 180),
            "highThreshold": settings.get("highThreshold", 180),
            "lowThreshold": settings.get("lowThreshold", 70),
            "criticalHigh": settings.get("criticalHigh", 250),
            "criticalLow": settings.get("criticalLow", 54),
            "timezone": settings.get("timezone", "America/New_York")
        })

    profile = {
        "id": profile_id,
        "accountId": user_id,  # Partition key
        "displayName": display_name,
        "relationship": "self",
        "diabetesType": "T1D",
        "dateOfBirth": None,
        "diagnosisDate": None,
        "avatarUrl": None,
        "settings": profile_settings,
        "dataSourceIds": [],
        "primaryGlucoseSourceId": None,
        "primaryTreatmentSourceId": None,
        "isActive": True,
        "lastDataAt": None,
        "createdAt": now,
        "updatedAt": now,
        # Legacy link for backwards compatibility
        "legacyUserId": user_id
    }

    if dry_run:
        logger.info(f"[DRY RUN] Would create profile: {display_name} ({profile_id}) for user {user_id}")
        return profile

    try:
        await profiles_container.create_item(profile)
        logger.info(f"Created profile: {display_name} ({profile_id}) for user {user_id}")
        return profile
    except Exception as e:
        logger.error(f"Failed to create profile for user {user_id}: {e}")
        return None


async def create_datasource(
    cosmos,
    profile_id: str,
    account_id: str,
    existing_source: dict,
    dry_run: bool = False
):
    """Create a ProfileDataSource from existing data source credentials."""
    sources_container = cosmos._containers.get("profile_data_sources")
    if not sources_container:
        logger.error("ProfileDataSources container not found")
        return None

    source_id = f"{profile_id}_gluroo"
    now = datetime.now(timezone.utc).isoformat()

    # Get encrypted credentials from existing source
    # The existing source should have nightscoutUrl and apiSecret
    credentials_encrypted = existing_source.get("credentialsEncrypted")
    if not credentials_encrypted:
        # Build from raw credentials if not already encrypted
        # Note: In production, these should be encrypted!
        raw_creds = {
            "url": existing_source.get("nightscoutUrl", ""),
            "apiSecret": existing_source.get("apiSecret", "")
        }
        # For migration, we'll store as JSON - should be re-encrypted in production
        credentials_encrypted = json.dumps(raw_creds)
        logger.warning(f"Source for profile {profile_id} has unencrypted credentials - should be re-encrypted")

    data_source = {
        "id": source_id,
        "profileId": profile_id,  # Partition key
        "sourceType": "gluroo",
        "credentialsEncrypted": credentials_encrypted,
        "isActive": True,
        "syncEnabled": True,
        "lastSyncAt": existing_source.get("lastSyncAt"),
        "syncStatus": "pending",
        "syncErrorMessage": None,
        "priority": 1,
        "providesGlucose": True,
        "providesTreatments": True,
        "createdAt": now,
        "updatedAt": now
    }

    if dry_run:
        logger.info(f"[DRY RUN] Would create data source: {source_id}")
        return data_source

    try:
        await sources_container.create_item(data_source)
        logger.info(f"Created data source: {source_id}")
        return data_source
    except Exception as e:
        logger.error(f"Failed to create data source for profile {profile_id}: {e}")
        return None


async def update_profile_datasource(
    cosmos,
    profile: dict,
    source_id: str,
    dry_run: bool = False
):
    """Update profile to link to its data source."""
    profiles_container = cosmos._containers.get("profiles")
    if not profiles_container:
        return

    profile["dataSourceIds"] = [source_id]
    profile["primaryGlucoseSourceId"] = source_id
    profile["primaryTreatmentSourceId"] = source_id
    profile["updatedAt"] = datetime.now(timezone.utc).isoformat()

    if dry_run:
        logger.info(f"[DRY RUN] Would link source {source_id} to profile {profile['id']}")
        return

    try:
        await profiles_container.upsert_item(profile)
        logger.info(f"Linked source {source_id} to profile {profile['id']}")
    except Exception as e:
        logger.error(f"Failed to update profile {profile['id']}: {e}")


async def migrate_users_to_profiles(dry_run: bool = False):
    """Main migration function."""
    logger.info("=" * 60)
    logger.info("T1D-AI: Migrate Users to Profiles")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    # Initialize CosmosDB
    cosmos = get_cosmos_manager()
    await cosmos.initialize_containers()

    # Create profiles container if it doesn't exist
    # Note: The container should be created with /accountId as partition key
    logger.info("Checking for profiles container...")

    # Get all users
    logger.info("Fetching all users...")
    users = await get_all_users(cosmos)
    logger.info(f"Found {len(users)} users")

    # Track stats
    stats = {
        "users_processed": 0,
        "profiles_created": 0,
        "profiles_skipped": 0,
        "datasources_created": 0,
        "errors": 0
    }

    # Process each user
    for user in users:
        user_id = user.get("id")
        if not user_id:
            continue

        stats["users_processed"] += 1
        logger.info(f"\nProcessing user: {user.get('email', user_id)}")

        # Check if profile already exists
        if await profile_exists(cosmos, user_id, "self"):
            logger.info(f"  Profile already exists, skipping")
            stats["profiles_skipped"] += 1
            continue

        # Get user settings
        settings = await get_user_settings(cosmos, user_id)
        if settings:
            logger.info(f"  Found existing settings")

        # Get existing data source
        existing_source = await get_user_datasource(cosmos, user_id)
        if existing_source:
            logger.info(f"  Found existing data source")

        # Create profile
        profile = await create_profile(cosmos, user, settings, dry_run)
        if profile:
            stats["profiles_created"] += 1

            # Create data source if exists
            if existing_source and profile:
                source = await create_datasource(
                    cosmos,
                    profile["id"],
                    user_id,
                    existing_source,
                    dry_run
                )
                if source:
                    stats["datasources_created"] += 1
                    await update_profile_datasource(
                        cosmos,
                        profile,
                        source["id"],
                        dry_run
                    )
        else:
            stats["errors"] += 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Users processed:    {stats['users_processed']}")
    logger.info(f"Profiles created:   {stats['profiles_created']}")
    logger.info(f"Profiles skipped:   {stats['profiles_skipped']}")
    logger.info(f"Data sources:       {stats['datasources_created']}")
    logger.info(f"Errors:             {stats['errors']}")

    if dry_run:
        logger.info("\nThis was a DRY RUN. Run without --dry-run to apply changes.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing users to the new profile model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    args = parser.parse_args()

    asyncio.run(migrate_users_to_profiles(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
