#!/usr/bin/env python3
"""
Trigger Gluroo full sync for Emrys's profile to rebuild historical data.
"""
import asyncio
import sys
import os

# Add backend/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from database.repositories import DataSourceRepository
from services.gluroo_service import create_gluroo_service
from utils.encryption import decrypt_secret

async def main():
    # Profile ID for Emrys Dawson
    profile_id = "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    print(f"Triggering full Gluroo sync for profile: {profile_id}")

    # Get Gluroo credentials
    datasource_repo = DataSourceRepository()
    datasource = await datasource_repo.get(profile_id, "gluroo")

    if not datasource:
        print(f"ERROR: No Gluroo connection found for profile {profile_id}")
        return

    print(f"✓ Found Gluroo connection")
    print(f"  URL: {datasource.config.get('url')}")
    print(f"  Last sync: {datasource.lastSyncedAt}")

    # Decrypt credentials
    api_secret = decrypt_secret(datasource.credentials.apiSecretEncrypted)
    url = datasource.config.get('url')

    # Create Gluroo service
    gluroo_service = create_gluroo_service(url, api_secret, profile_id)

    print(f"\nStarting full sync (this may take a minute)...")

    # Perform full sync
    result = await gluroo_service.sync(full_sync=True)

    print(f"\n✓ Sync complete!")
    print(f"  Glucose readings: {result['glucoseReadings']}")
    print(f"  Treatments: {result['treatments']}")
    print(f"  Last sync: {result['lastSyncAt']}")

if __name__ == "__main__":
    asyncio.run(main())
