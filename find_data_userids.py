#!/usr/bin/env python3
"""Find which user IDs actually have data in CosmosDB."""
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from database.repositories import GlucoseRepository, TreatmentRepository

async def main():
    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()

    # IDs to check
    ids_to_check = [
        "05bf0083-5598-43a5-aa7f-bd70b1f1be57",  # Account owner (Jaderic)
        "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57",  # Emrys child profile
        "c4c77958-3e25-42b6-9f9a-f8a6c6aea098",  # Jaderic self profile
        "emrys",  # Legacy format?
        "demo_user",  # Demo user
    ]

    print("=" * 80)
    print("SEARCHING FOR DATA ACROSS USER IDs")
    print("=" * 80)

    for user_id in ids_to_check:
        print(f"\n[{user_id}]")
        try:
            latest = await glucose_repo.get_latest(user_id)
            if latest:
                print(f"  ✓ Latest glucose: {latest.value} mg/dL at {latest.timestamp}")

            start = datetime.now(timezone.utc) - timedelta(days=7)
            history = await glucose_repo.get_history(user_id, start, datetime.now(timezone.utc), 100)
            if history:
                print(f"  ✓ History: {len(history)} readings (last 7 days)")

            treatments = await treatment_repo.get_recent(user_id, hours=168)
            if treatments:
                print(f"  ✓ Treatments: {len(treatments)} (last 7 days)")

            if not latest and not history and not treatments:
                print(f"  ✗ No data")

        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
