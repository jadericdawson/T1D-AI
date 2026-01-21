#!/usr/bin/env python3
"""
Ingest historical glucose readings from existing JSONL file.
Only extracts glucose data - ignores the corrupted treatment columns.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from azure.cosmos import CosmosClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> datetime:
    """Parse various timestamp formats."""
    if not ts:
        return None
    try:
        # Handle +0000 format
        if ts.endswith('+0000'):
            ts = ts[:-5] + '+00:00'
        return datetime.fromisoformat(ts)
    except:
        return None


def ingest_from_jsonl(
    jsonl_path: str,
    user_id: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "T1D-AI-DB",
    batch_size: int = 1000
):
    """Ingest glucose readings from JSONL file."""

    client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = client.get_database_client(database_name)
    glucose_container = database.get_container_client("glucose_readings")

    # Read all records
    logger.info(f"Reading from {jsonl_path}...")
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Total records in file: {len(records)}")

    # Dedupe by _id
    by_id = {r['_id']: r for r in records}
    unique = list(by_id.values())
    logger.info(f"Unique records: {len(unique)}")

    # Sort by timestamp
    unique.sort(key=lambda x: x.get('timestamp', ''))

    if unique:
        first_ts = parse_timestamp(unique[0].get('timestamp'))
        last_ts = parse_timestamp(unique[-1].get('timestamp'))
        logger.info(f"Date range: {first_ts.date() if first_ts else 'N/A'} to {last_ts.date() if last_ts else 'N/A'}")

    # Upload in batches
    success = 0
    errors = 0

    for i, record in enumerate(unique):
        ts = parse_timestamp(record.get('timestamp'))
        if not ts:
            errors += 1
            continue

        # Extract ONLY glucose data - ignore carbs/insulin (corrupted)
        doc = {
            "id": f"{user_id}_{record['_id']}",
            "userId": user_id,
            "timestamp": ts.isoformat(),
            "value": int(record.get('value', 0)),
            "trend": record.get('trend', 0),
            "direction": record.get('direction', 'Flat'),
            "source": "gluroo",
            "sourceId": record['_id']
        }

        try:
            glucose_container.upsert_item(doc)
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Error: {e}")

        if (i + 1) % 5000 == 0:
            logger.info(f"Progress: {i + 1}/{len(unique)} ({success} success, {errors} errors)")

    logger.info(f"Complete: {success} uploaded, {errors} errors")
    return {"success": success, "errors": errors, "total": len(unique)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--database", default="T1D-AI-DB")
    args = parser.parse_args()

    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")

    if not cosmos_endpoint or not cosmos_key:
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY required")
        sys.exit(1)

    result = ingest_from_jsonl(
        jsonl_path=args.jsonl,
        user_id=args.user_id,
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key,
        database_name=args.database
    )
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
