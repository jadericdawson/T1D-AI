#!/usr/bin/env python3
"""
Ingest historical glucose readings from CSV file.
Deduplicates by timestamp (datetime).
Ignores the corrupted treatment columns (carbs, insulin).
"""
import argparse
import csv
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone

from azure.cosmos import CosmosClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_csv_timestamp(ts: str) -> datetime:
    """Parse CSV timestamp format: 2025-07-20 23:36:13+0000"""
    if not ts:
        return None
    try:
        # Handle +0000 format
        if ts.endswith('+0000'):
            ts = ts[:-5] + '+00:00'
        elif '+' in ts and not ts.endswith('+00:00'):
            ts = ts.replace('+', '+00:')
        return datetime.fromisoformat(ts.replace(' ', 'T'))
    except Exception as e:
        logger.debug(f"Failed to parse timestamp {ts}: {e}")
        return None


def generate_id(user_id: str, timestamp: str) -> str:
    """Generate unique ID from user_id and timestamp."""
    content = f"{user_id}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()


def ingest_from_csv(
    csv_path: str,
    user_id: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "T1D-AI-DB"
):
    """Ingest glucose readings from CSV file, deduped by timestamp."""

    client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = client.get_database_client(database_name)
    glucose_container = database.get_container_client("glucose_readings")

    logger.info(f"Reading from {csv_path}...")

    # Read and dedupe by timestamp
    by_timestamp = {}
    row_count = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            ts_str = row.get('timestamp', '').strip()
            ts = parse_csv_timestamp(ts_str)
            if not ts:
                continue

            # Use timestamp as key for dedup
            ts_key = ts.isoformat()
            if ts_key not in by_timestamp:
                by_timestamp[ts_key] = {
                    'timestamp': ts,
                    'value': int(float(row.get('value', 0))),
                    'trend': int(float(row.get('trend', 0)))
                }

            if row_count % 100000 == 0:
                logger.info(f"Read {row_count} rows, {len(by_timestamp)} unique...")

    logger.info(f"Total CSV rows: {row_count}")
    logger.info(f"Unique timestamps: {len(by_timestamp)}")

    # Sort by timestamp
    unique = sorted(by_timestamp.values(), key=lambda x: x['timestamp'])

    if unique:
        logger.info(f"Date range: {unique[0]['timestamp'].date()} to {unique[-1]['timestamp'].date()}")

    # Upload
    success = 0
    errors = 0

    for i, rec in enumerate(unique):
        ts = rec['timestamp']
        doc = {
            "id": f"{user_id}_{generate_id(user_id, ts.isoformat())}",
            "userId": user_id,
            "timestamp": ts.isoformat(),
            "value": rec['value'],
            "trend": rec['trend'],
            "direction": "Flat",  # CSV doesn't have direction
            "source": "gluroo",
            "sourceId": generate_id(user_id, ts.isoformat())
        }

        try:
            glucose_container.upsert_item(doc)
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Error: {e}")

        if (i + 1) % 1000 == 0:
            logger.info(f"Progress: {i + 1}/{len(unique)} ({success} success, {errors} errors)")

    logger.info(f"Complete: {success} uploaded, {errors} errors")
    return {"success": success, "errors": errors, "total": len(unique)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--database", default="T1D-AI-DB")
    args = parser.parse_args()

    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")

    if not cosmos_endpoint or not cosmos_key:
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY required")
        sys.exit(1)

    result = ingest_from_csv(
        csv_path=args.csv,
        user_id=args.user_id,
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key,
        database_name=args.database
    )
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
