#!/usr/bin/env python3
"""
Verify that ALL API endpoints correctly strip profile_ prefix.
Tests the complete data flow for Emrys's profile.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from database.repositories import (
    GlucoseRepository, TreatmentRepository, InsightRepository
)
from api.v1.glucose import get_data_user_id as glucose_get_data_user_id
from api.v1.treatments import get_data_user_id as treatments_get_data_user_id
from api.v1.calculations import get_data_user_id as calc_get_data_user_id
from api.v1.predictions import get_data_user_id as pred_get_data_user_id
from api.v1.training import get_data_user_id as training_get_data_user_id
from api.v1.websocket import get_data_user_id as ws_get_data_user_id
from api.v1.datasources import get_data_user_id as ds_get_data_user_id
from api.v1.insights import get_data_user_id as insights_get_data_user_id

async def main():
    # Test IDs
    profile_id = "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57"
    expected_data_id = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"

    print("=" * 80)
    print("VERIFYING PROFILE_ PREFIX STRIPPING CONSISTENCY")
    print("=" * 80)

    # Test all get_data_user_id functions
    functions_to_test = [
        ("glucose.py", glucose_get_data_user_id),
        ("treatments.py", treatments_get_data_user_id),
        ("calculations.py", calc_get_data_user_id),
        ("predictions.py", pred_get_data_user_id),
        ("training.py", training_get_data_user_id),
        ("websocket.py", ws_get_data_user_id),
        ("datasources.py", ds_get_data_user_id),
        ("insights.py", insights_get_data_user_id),
    ]

    print(f"\n1. Testing get_data_user_id() functions:")
    print(f"   Input: {profile_id}")
    print(f"   Expected output: {expected_data_id}\n")

    all_correct = True
    for filename, func in functions_to_test:
        result = func(profile_id)
        status = "✓" if result == expected_data_id else "✗"
        print(f"   {status} {filename:20s} → {result}")
        if result != expected_data_id:
            all_correct = False

    if not all_correct:
        print("\n❌ INCONSISTENCY DETECTED!")
        return 1

    # Test actual data access
    print(f"\n2. Testing data access with stripped ID ({expected_data_id}):")

    glucose_repo = GlucoseRepository()
    treatment_repo = TreatmentRepository()
    insight_repo = InsightRepository()

    latest_glucose = await glucose_repo.get_latest(expected_data_id)
    print(f"   ✓ Latest glucose: {latest_glucose.value if latest_glucose else 'NONE'} mg/dL")

    start = datetime.now(timezone.utc) - timedelta(hours=24)
    history = await glucose_repo.get_history(expected_data_id, start, datetime.now(timezone.utc), 1000)
    print(f"   ✓ History (24hr): {len(history)} readings")

    treatments = await treatment_repo.get_recent(expected_data_id, hours=24)
    print(f"   ✓ Treatments (24hr): {len(treatments)} entries")

    insights = await insight_repo.get_by_user(expected_data_id, limit=5)
    print(f"   ✓ Insights: {len(insights)} entries")

    print(f"\n✅ ALL ENDPOINTS CONSISTENT - Profile_ prefix correctly stripped!")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
