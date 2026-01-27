#!/usr/bin/env python3
"""Test API endpoints and trigger sync for Emrys's profile."""
import asyncio
import sys
import os
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from database.repositories import DataSourceRepository, GlucoseRepository, TreatmentRepository
from services.gluroo_service import GlurooService
from utils.encryption import decrypt_secret

async def main():
    # Emrys's profile ID
    emrys_profile_id = "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57"
    # Jaderic's account ID (owns the Gluroo connection)
    account_owner_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    print("=" * 80)
    print("TESTING EMRYS DAWSON PROFILE DATA")
    print("=" * 80)

    # Check current data in database
    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()

    print(f"\n1. Checking database for profile: {emrys_profile_id}")

    # Test with profile ID
    latest_glucose = await glucose_repo.get_latest(emrys_profile_id)
    print(f"   Latest glucose (with profile_ prefix): {latest_glucose.value if latest_glucose else 'NONE'}")

    from datetime import datetime, timedelta, timezone
    start_time = datetime.now(timezone.utc) - timedelta(days=7)
    history = await glucose_repo.get_history(emrys_profile_id, start_time, datetime.now(timezone.utc), 5000)
    print(f"   History count (7 days): {len(history)}")

    treatments = await treatment_repo.get_recent(emrys_profile_id, hours=168)
    print(f"   Treatments (7 days): {len(treatments)}")

    # Check Gluroo connection
    print(f"\n2. Checking Gluroo connection (stored under account owner)")
    datasource_repo = DataSourceRepository()

    datasource = await datasource_repo.get(account_owner_id, "gluroo")
    if datasource:
        print(f"   ✓ Gluroo connected")
        print(f"   URL: {datasource.credentials.url}")

        # Trigger sync
        print(f"\n3. Triggering full Gluroo sync...")
        api_secret = decrypt_secret(datasource.credentials.apiSecretEncrypted)
        service = GlurooService(datasource.credentials.url, api_secret)

        # Fetch data from Gluroo
        since = datetime.now(timezone.utc) - timedelta(days=30)
        print(f"   Fetching data since: {since.date()}")

        glucose_readings = await service.fetch_glucose_entries(
            user_id=emrys_profile_id,  # Store under profile ID
            count=5000,
            since=since
        )
        print(f"   Fetched {len(glucose_readings)} glucose readings from Gluroo")

        treatments_data = await service.fetch_all_treatments(
            user_id=emrys_profile_id,  # Store under profile ID
            count=2000,
            since=since
        )
        print(f"   Fetched {len(treatments_data)} treatments from Gluroo")

        # Store in database
        print(f"\n4. Storing data in CosmosDB...")
        glucose_stored = 0
        for reading in glucose_readings[:100]:  # Store first 100 as test
            try:
                await glucose_repo.upsert(reading)
                glucose_stored += 1
            except Exception as e:
                print(f"   Error storing glucose: {e}")
                break

        print(f"   Stored {glucose_stored} glucose readings")

        # Verify
        print(f"\n5. Verifying data...")
        latest_after = await glucose_repo.get_latest(emrys_profile_id)
        print(f"   Latest glucose after sync: {latest_after.value if latest_after else 'NONE'}")

        history_after = await glucose_repo.get_history(emrys_profile_id, start_time, datetime.now(timezone.utc), 5000)
        print(f"   History count after sync: {len(history_after)}")

    else:
        print(f"   ✗ No Gluroo connection found")

if __name__ == "__main__":
    asyncio.run(main())
