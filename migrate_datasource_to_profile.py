#!/usr/bin/env python3
"""
Migrate old DataSource to new ProfileDataSource for Emrys.
This enables profile-based background sync.
"""
import asyncio
import sys
import os
from datetime import datetime, timezone
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from database.repositories import DataSourceRepository, ProfileDataSourceRepository
    from models.schemas import ProfileDataSource, DataSourceType

    # Account owner ID (where old DataSource is stored)
    account_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"
    # Emrys's profile ID (where ProfileDataSource should be)
    emrys_profile_id = "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    datasource_repo = DataSourceRepository()
    profile_datasource_repo = ProfileDataSourceRepository()

    print("=" * 80)
    print("MIGRATING DATASOURCE TO PROFILEDATASOURCE FOR EMRYS")
    print("=" * 80)

    # Get old DataSource
    old_datasource = await datasource_repo.get(account_id, "gluroo")
    if not old_datasource:
        print(f"❌ No Gluroo connection found under account ID {account_id}")
        return

    print(f"\n✓ Found old DataSource:")
    print(f"  Account ID: {account_id}")
    print(f"  URL: {old_datasource.credentials.url}")

    # Check if ProfileDataSource already exists
    existing = await profile_datasource_repo.get_by_profile(emrys_profile_id, account_id)
    gluroo_sources = [s for s in existing if s.sourceType == DataSourceType.GLUROO]
    if gluroo_sources:
        print(f"\n✓ ProfileDataSource already exists for Emrys")
        print(f"  Skipping migration")
        return

    # Create new ProfileDataSource
    print(f"\n→ Creating ProfileDataSource for Emrys's profile...")

    # Prepare credentials JSON with encrypted API secret
    import json
    credentials_dict = {
        "url": old_datasource.credentials.url,
        "apiSecret": old_datasource.credentials.apiSecretEncrypted  # Already encrypted
    }
    credentials_json = json.dumps(credentials_dict)

    new_profile_datasource = ProfileDataSource(
        id=f"{emrys_profile_id}_gluroo",
        profileId=emrys_profile_id,
        accountId=account_id,
        sourceType=DataSourceType.GLUROO,
        credentialsEncrypted=credentials_json,
        syncEnabled=True,
        isActive=True,
        priority=1,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
        providesGlucose=True,
        providesTreatments=True
    )

    await profile_datasource_repo.create(new_profile_datasource)

    print(f"✅ ProfileDataSource created!")
    print(f"   Profile ID: {emrys_profile_id}")
    print(f"   Background sync will now update Emrys's data every 5 minutes")

if __name__ == "__main__":
    asyncio.run(main())
