#!/usr/bin/env python3
"""
Data Migration Script for T1D-AI
Sets up Jaderic Dawson (parent) and Emrys Dawson (child) accounts,
then migrates existing Gluroo data to Emrys's account in CosmosDB.
"""
import json
import os
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

# Load environment from .env file if present
load_dotenv()

# Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://knowledge2ai-cosmos-serverless.documents.azure.com:443/")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("COSMOS_DATABASE", "T1D-AI-DB")

# Data file paths
SOURCE_DIR = Path("/home/jadericdawson/Documents/AI/dexcom_reader_ML_complete")
GLUCOSE_FILE = SOURCE_DIR / "gluroo_readings.jsonl"

# User info
PARENT_EMAIL = "jadericdawson@gmail.com"
PARENT_NAME = "Jaderic Dawson"
CHILD_EMAIL = "emrys.dawson@t1d-ai.local"  # Internal email for child account
CHILD_NAME = "Emrys Dawson"
CHILD_DOB = datetime(2018, 4, 15)  # Approximate - about 7 years old in 2025
DIAGNOSIS_DATE = datetime(2023, 6, 1)  # Approximate

# Trend mapping
TREND_MAP = {
    0: "Flat",
    1: "FortyFiveUp",
    2: "SingleUp",
    3: "DoubleUp",
    -1: "FortyFiveDown",
    -2: "SingleDown",
    -3: "DoubleDown",
}


def get_cosmos_client():
    """Create CosmosDB client."""
    if not COSMOS_KEY:
        raise ValueError("COSMOS_KEY environment variable not set")
    return CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)


def ensure_containers(database):
    """Ensure all required containers exist."""
    containers = {
        "users": "/id",
        "glucose_readings": "/userId",
        "treatments": "/userId",
        "datasources": "/userId",
        "insights": "/userId",
    }

    for name, partition_key in containers.items():
        try:
            database.create_container(name, PartitionKey(path=partition_key))
            print(f"Created container: {name}")
        except Exception as e:
            if "Conflict" in str(e) or "already exists" in str(e).lower():
                print(f"Container exists: {name}")
            else:
                raise


def create_parent_account(container) -> dict:
    """Create or get parent account."""
    # Check if parent exists
    query = f"SELECT * FROM c WHERE c.email = '{PARENT_EMAIL}'"
    items = list(container.query_items(query, enable_cross_partition_query=True))

    if items:
        print(f"Parent account exists: {items[0]['id']}")
        return items[0]

    # Create parent
    parent = {
        "id": str(uuid.uuid4()),
        "email": PARENT_EMAIL,
        "displayName": PARENT_NAME,
        "passwordHash": None,  # Will be set on first login
        "authProvider": "microsoft",  # Prefer Microsoft login
        "microsoftId": None,  # Will be linked on first Microsoft login
        "createdAt": datetime.utcnow().isoformat(),
        "accountType": "parent",
        "parentId": None,
        "guardianEmail": None,
        "linkedChildIds": [],
        "dateOfBirth": None,
        "diagnosisDate": None,
        "hasT1D": False,
        "avatarUrl": None,
        "theme": None,
        "settings": {
            "timezone": "America/New_York",
            "targetBg": 100,
            "insulinSensitivity": 50.0,
            "carbRatio": 10.0,
            "insulinDuration": 180,
            "carbAbsorptionDuration": 180,
            "highThreshold": 180,
            "lowThreshold": 70,
            "criticalHighThreshold": 250,
            "criticalLowThreshold": 54,
            "enableAlerts": True,
            "enablePredictiveAlerts": True,
            "showInsights": True
        }
    }

    container.create_item(parent)
    print(f"Created parent account: {parent['id']} ({PARENT_NAME})")
    return parent


def create_child_account(container, parent_id: str) -> dict:
    """Create or get child account."""
    # Check if child exists
    query = f"SELECT * FROM c WHERE c.email = '{CHILD_EMAIL}'"
    items = list(container.query_items(query, enable_cross_partition_query=True))

    if items:
        print(f"Child account exists: {items[0]['id']}")
        return items[0]

    # Create child
    child = {
        "id": str(uuid.uuid4()),
        "email": CHILD_EMAIL,
        "displayName": CHILD_NAME,
        "passwordHash": None,
        "authProvider": "email",
        "microsoftId": None,
        "createdAt": datetime.utcnow().isoformat(),
        "accountType": "child",
        "parentId": parent_id,
        "guardianEmail": PARENT_EMAIL,
        "linkedChildIds": [],
        "dateOfBirth": CHILD_DOB.isoformat(),
        "diagnosisDate": DIAGNOSIS_DATE.isoformat(),
        "hasT1D": True,
        "avatarUrl": None,
        "theme": None,
        "settings": {
            "timezone": "America/New_York",
            "targetBg": 100,
            "insulinSensitivity": 60.0,  # Children often more sensitive
            "carbRatio": 15.0,  # Varies for children
            "insulinDuration": 180,
            "carbAbsorptionDuration": 180,
            "highThreshold": 180,
            "lowThreshold": 70,
            "criticalHighThreshold": 250,
            "criticalLowThreshold": 54,
            "enableAlerts": True,
            "enablePredictiveAlerts": True,
            "showInsights": True
        }
    }

    container.create_item(child)
    print(f"Created child account: {child['id']} ({CHILD_NAME})")
    return child


def link_child_to_parent(container, parent: dict, child_id: str):
    """Link child to parent's linkedChildIds."""
    linked = parent.get("linkedChildIds", [])
    if child_id not in linked:
        linked.append(child_id)
        parent["linkedChildIds"] = linked
        container.upsert_item(parent)
        print(f"Linked child {child_id} to parent {parent['id']}")


def migrate_glucose_data(container, user_id: str, max_records: int = None):
    """Migrate glucose readings from JSONL file to CosmosDB."""
    if not GLUCOSE_FILE.exists():
        print(f"Glucose file not found: {GLUCOSE_FILE}")
        return 0

    print(f"Reading glucose data from {GLUCOSE_FILE}...")

    count = 0
    seen_ids = set()
    batch = []

    with open(GLUCOSE_FILE, "r") as f:
        for line in f:
            if max_records and count >= max_records:
                break

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Skip duplicates
            source_id = data.get("_id")
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)

            # Parse timestamp
            ts = data.get("timestamp")
            if isinstance(ts, str):
                # Remove timezone info for parsing
                ts_clean = ts.replace("+00:00", "").replace("Z", "")
                try:
                    timestamp = datetime.fromisoformat(ts_clean)
                except:
                    continue
            else:
                continue

            # Map trend
            trend_val = data.get("trend", 0)
            trend = TREND_MAP.get(trend_val, "Flat")

            # Create glucose reading
            reading = {
                "id": str(uuid.uuid4()),
                "userId": user_id,
                "timestamp": timestamp.isoformat(),
                "value": int(data.get("value", 0)),
                "trend": trend,
                "source": "gluroo",
                "sourceId": source_id,
                "iob": None,
                "cob": None,
                "isf": None,
            }

            batch.append(reading)
            count += 1

            # Insert in batches
            if len(batch) >= 100:
                for item in batch:
                    try:
                        container.upsert_item(item)
                    except Exception as e:
                        print(f"Error inserting reading: {e}")
                print(f"Inserted {count} glucose readings...")
                batch = []

    # Insert remaining
    for item in batch:
        try:
            container.upsert_item(item)
        except Exception as e:
            print(f"Error inserting reading: {e}")

    print(f"Migrated {count} glucose readings for user {user_id}")
    return count


def migrate_treatments(container, user_id: str, max_records: int = None):
    """Extract and migrate treatments from glucose data."""
    if not GLUCOSE_FILE.exists():
        print(f"Glucose file not found: {GLUCOSE_FILE}")
        return 0

    print(f"Extracting treatments from {GLUCOSE_FILE}...")

    count = 0
    seen_ids = set()

    with open(GLUCOSE_FILE, "r") as f:
        for line in f:
            if max_records and count >= max_records:
                break

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Skip if no treatment data
            insulin = float(data.get("insulin", 0) or 0)
            carbs = float(data.get("carbs", 0) or 0)
            protein = float(data.get("protein", 0) or 0)
            fat = float(data.get("fat", 0) or 0)

            if insulin == 0 and carbs == 0:
                continue

            # Parse timestamp
            ts = data.get("timestamp")
            if isinstance(ts, str):
                ts_clean = ts.replace("+00:00", "").replace("Z", "")
                try:
                    timestamp = datetime.fromisoformat(ts_clean)
                except:
                    continue
            else:
                continue

            # Create unique ID based on timestamp and values
            treatment_key = f"{ts}_{insulin}_{carbs}"
            treatment_id = hashlib.sha1(treatment_key.encode()).hexdigest()[:16]

            if treatment_id in seen_ids:
                continue
            seen_ids.add(treatment_id)

            # Determine treatment type
            if insulin > 0 and carbs > 0:
                treatment_type = "Carb Correction"
            elif insulin > 0:
                treatment_type = "Correction Bolus"
            else:
                treatment_type = "carbs"

            treatment = {
                "id": str(uuid.uuid4()),
                "userId": user_id,
                "timestamp": timestamp.isoformat(),
                "type": treatment_type,
                "insulin": insulin if insulin > 0 else None,
                "carbs": carbs if carbs > 0 else None,
                "protein": protein if protein > 0 else None,
                "fat": fat if fat > 0 else None,
                "notes": None,
                "source": "gluroo",
                "sourceId": treatment_id,
            }

            try:
                container.upsert_item(treatment)
                count += 1

                if count % 100 == 0:
                    print(f"Inserted {count} treatments...")
            except Exception as e:
                print(f"Error inserting treatment: {e}")

    print(f"Migrated {count} treatments for user {user_id}")
    return count


def main():
    """Main migration function."""
    print("=" * 60)
    print("T1D-AI Data Migration")
    print("=" * 60)

    # Connect to CosmosDB
    print("\nConnecting to CosmosDB...")
    client = get_cosmos_client()
    database = client.get_database_client(DATABASE_NAME)

    # Ensure containers exist
    print("\nEnsuring containers...")
    ensure_containers(database)

    # Get containers
    users_container = database.get_container_client("users")
    glucose_container = database.get_container_client("glucose_readings")
    treatments_container = database.get_container_client("treatments")

    # Create accounts
    print("\nSetting up accounts...")
    parent = create_parent_account(users_container)
    child = create_child_account(users_container, parent["id"])
    link_child_to_parent(users_container, parent, child["id"])

    # Migrate data
    print("\nMigrating glucose data...")
    glucose_count = migrate_glucose_data(glucose_container, child["id"])

    print("\nMigrating treatments...")
    treatment_count = migrate_treatments(treatments_container, child["id"])

    # Summary
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"\nParent Account:")
    print(f"  ID: {parent['id']}")
    print(f"  Email: {parent['email']}")
    print(f"  Name: {parent['displayName']}")

    print(f"\nChild Account (Emrys):")
    print(f"  ID: {child['id']}")
    print(f"  Email: {child['email']}")
    print(f"  Name: {child['displayName']}")

    print(f"\nData Migrated:")
    print(f"  Glucose readings: {glucose_count}")
    print(f"  Treatments: {treatment_count}")

    print("\nNext Steps:")
    print("1. Deploy the app: Follow DEPLOYMENT.md")
    print("2. Login with Microsoft account (jadericdawson@gmail.com)")
    print("3. View Emrys's data in the dashboard")


if __name__ == "__main__":
    main()
