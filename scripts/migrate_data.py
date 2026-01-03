#!/usr/bin/env python3
"""
Data Migration Script for T1D-AI
Migrates existing gluroo_readings.jsonl to CosmosDB.

Usage:
    python scripts/migrate_data.py --user-id <user_id> [--dry-run]
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Set
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from azure.cosmos import CosmosClient, PartitionKey

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_JSONL_PATH = Path(__file__).parent.parent / "data" / "gluroo_readings.jsonl"
DEFAULT_BOLUS_PATH = Path(__file__).parent.parent / "data" / "bolus_moments.jsonl"


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    import os
                    os.environ.setdefault(key.strip(), value.strip())


def parse_timestamp(ts_str: str) -> datetime:
    """Parse various timestamp formats to datetime."""
    if isinstance(ts_str, datetime):
        return ts_str

    # Try ISO format first
    try:
        if ts_str.endswith("Z"):
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass

    # Try other formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp: {ts_str}")


def parse_glucose_entry(entry: Dict[str, Any], user_id: str, seen_ids: Set[str]) -> Dict[str, Any]:
    """Parse a glucose entry from JSONL format."""
    # Get source ID for deduplication
    source_id = entry.get("_id") or entry.get("id") or str(entry.get("mills", ""))

    # Skip duplicates
    if source_id in seen_ids:
        return None
    seen_ids.add(source_id)

    # Parse glucose value
    value = entry.get("sgv") or entry.get("value")
    if value is None:
        return None

    # Parse timestamp
    timestamp = None
    if entry.get("timestamp"):
        timestamp = parse_timestamp(entry["timestamp"])
    elif entry.get("dateString"):
        timestamp = parse_timestamp(entry["dateString"])
    elif entry.get("date"):
        # Milliseconds timestamp
        timestamp = datetime.fromtimestamp(entry["date"] / 1000, tz=timezone.utc)
    elif entry.get("mills"):
        timestamp = datetime.fromtimestamp(entry["mills"] / 1000, tz=timezone.utc)

    if timestamp is None:
        return None

    # Parse trend
    trend = entry.get("direction") or entry.get("trend") or "Flat"
    if isinstance(trend, int):
        trend_map = {
            1: "DoubleUp", 2: "SingleUp", 3: "FortyFiveUp",
            4: "Flat",
            5: "FortyFiveDown", 6: "SingleDown", 7: "DoubleDown"
        }
        trend = trend_map.get(trend, "Flat")

    # Create document ID
    doc_id = f"{user_id}_{source_id}" if source_id else f"{user_id}_{uuid4().hex[:12]}"

    return {
        "id": doc_id,
        "userId": user_id,
        "timestamp": timestamp.isoformat(),
        "value": int(value),
        "trend": trend,
        "source": "gluroo",
        "sourceId": source_id
    }


def parse_treatment_entry(entry: Dict[str, Any], user_id: str, seen_ids: Set[str]) -> Dict[str, Any]:
    """Parse a treatment entry from JSONL format."""
    source_id = entry.get("_id") or entry.get("id") or str(entry.get("mills", ""))

    if source_id in seen_ids:
        return None
    seen_ids.add(source_id)

    # Get insulin or carbs
    insulin = entry.get("insulin")
    carbs = entry.get("carbs")

    if not insulin and not carbs:
        return None

    # Determine type
    if insulin and float(insulin) > 0:
        treatment_type = "insulin"
    elif carbs and float(carbs) > 0:
        treatment_type = "carbs"
    else:
        return None

    # Parse timestamp
    timestamp = None
    if entry.get("timestamp"):
        timestamp = parse_timestamp(entry["timestamp"])
    elif entry.get("created_at"):
        timestamp = parse_timestamp(entry["created_at"])
    elif entry.get("mills"):
        timestamp = datetime.fromtimestamp(entry["mills"] / 1000, tz=timezone.utc)

    if timestamp is None:
        return None

    doc_id = f"{user_id}_{source_id}" if source_id else f"{user_id}_{uuid4().hex[:12]}"

    return {
        "id": doc_id,
        "userId": user_id,
        "timestamp": timestamp.isoformat(),
        "type": treatment_type,
        "insulin": float(insulin) if insulin else None,
        "carbs": float(carbs) if carbs else None,
        "protein": float(entry.get("protein", 0)) if entry.get("protein") else None,
        "fat": float(entry.get("fat", 0)) if entry.get("fat") else None,
        "notes": entry.get("notes"),
        "source": "gluroo",
        "sourceId": source_id
    }


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load entries from JSONL file."""
    entries = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return entries

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(entries)} entries from {file_path}")
    return entries


def migrate_to_cosmos(
    user_id: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "T1D-AI-DB",
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    dry_run: bool = False
):
    """Migrate data from JSONL to CosmosDB."""
    logger.info(f"Starting migration for user: {user_id}")
    logger.info(f"CosmosDB: {cosmos_endpoint}")
    logger.info(f"Database: {database_name}")
    logger.info(f"Source: {jsonl_path}")
    logger.info(f"Dry run: {dry_run}")

    # Load source data
    entries = load_jsonl(jsonl_path)
    if not entries:
        logger.error("No entries to migrate")
        return

    # Parse entries
    seen_glucose_ids: Set[str] = set()
    seen_treatment_ids: Set[str] = set()

    glucose_docs = []
    treatment_docs = []

    for entry in entries:
        # Try parsing as glucose
        if entry.get("sgv") or entry.get("value"):
            doc = parse_glucose_entry(entry, user_id, seen_glucose_ids)
            if doc:
                glucose_docs.append(doc)

        # Try parsing as treatment
        if entry.get("insulin") or entry.get("carbs"):
            doc = parse_treatment_entry(entry, user_id, seen_treatment_ids)
            if doc:
                treatment_docs.append(doc)

    logger.info(f"Parsed {len(glucose_docs)} glucose readings")
    logger.info(f"Parsed {len(treatment_docs)} treatments")

    if dry_run:
        logger.info("DRY RUN - No data written")
        if glucose_docs:
            logger.info(f"Sample glucose: {glucose_docs[0]}")
        if treatment_docs:
            logger.info(f"Sample treatment: {treatment_docs[0]}")
        return

    # Connect to CosmosDB
    client = CosmosClient(cosmos_endpoint, cosmos_key)
    database = client.create_database_if_not_exists(database_name)

    # Create containers if needed
    glucose_container = database.create_container_if_not_exists(
        id="glucose_readings",
        partition_key=PartitionKey(path="/userId")
    )
    treatment_container = database.create_container_if_not_exists(
        id="treatments",
        partition_key=PartitionKey(path="/userId")
    )

    # Insert glucose readings
    logger.info("Inserting glucose readings...")
    glucose_success = 0
    glucose_errors = 0
    for i, doc in enumerate(glucose_docs):
        try:
            glucose_container.upsert_item(doc)
            glucose_success += 1
            if (i + 1) % 1000 == 0:
                logger.info(f"Progress: {i + 1}/{len(glucose_docs)} glucose readings")
        except Exception as e:
            glucose_errors += 1
            if glucose_errors <= 5:
                logger.error(f"Error inserting glucose {doc['id']}: {e}")

    logger.info(f"Glucose: {glucose_success} inserted, {glucose_errors} errors")

    # Insert treatments
    logger.info("Inserting treatments...")
    treatment_success = 0
    treatment_errors = 0
    for doc in treatment_docs:
        try:
            treatment_container.upsert_item(doc)
            treatment_success += 1
        except Exception as e:
            treatment_errors += 1
            if treatment_errors <= 5:
                logger.error(f"Error inserting treatment {doc['id']}: {e}")

    logger.info(f"Treatments: {treatment_success} inserted, {treatment_errors} errors")
    logger.info("Migration complete!")


def main():
    parser = argparse.ArgumentParser(description="Migrate T1D data to CosmosDB")
    parser.add_argument("--user-id", required=True, help="User ID for the migrated data")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH, help="Path to JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Parse data without writing to CosmosDB")
    parser.add_argument("--cosmos-endpoint", help="CosmosDB endpoint (or set COSMOS_ENDPOINT env)")
    parser.add_argument("--cosmos-key", help="CosmosDB key (or set COSMOS_KEY env)")
    parser.add_argument("--database", default="T1D-AI-DB", help="Database name")

    args = parser.parse_args()

    # Load .env
    load_env()

    import os
    cosmos_endpoint = args.cosmos_endpoint or os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = args.cosmos_key or os.environ.get("COSMOS_KEY")

    if not args.dry_run and (not cosmos_endpoint or not cosmos_key):
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY required for non-dry-run")
        sys.exit(1)

    migrate_to_cosmos(
        user_id=args.user_id,
        cosmos_endpoint=cosmos_endpoint or "",
        cosmos_key=cosmos_key or "",
        database_name=args.database,
        jsonl_path=args.jsonl,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
