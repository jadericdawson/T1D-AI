#!/usr/bin/env python3
"""
Ingest existing Gluroo data from JSONL file into CosmosDB.

Usage:
    python scripts/ingest_gluroo_data.py --user_id emrys --file path/to/gluroo_readings.jsonl
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    # Handle various formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(ts_str.replace("+00:00", "Z").replace("Z", "+0000"), fmt.replace("%z", "%z"))
        except ValueError:
            continue

    # Try pandas as fallback
    try:
        import pandas as pd
        return pd.to_datetime(ts_str).to_pydatetime()
    except:
        pass

    raise ValueError(f"Could not parse timestamp: {ts_str}")


async def ingest_data(
    user_id: str,
    file_path: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "T1D-AI-DB",
    batch_size: int = 100
):
    """Ingest Gluroo data into CosmosDB."""

    # Initialize Cosmos client
    client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = client.get_database_client(database_name)

    # Get containers
    glucose_container = database.get_container_client("glucose_readings")
    treatment_container = database.get_container_client("treatments")

    # Read and process file
    glucose_docs = []
    treatment_docs = []

    logger.info(f"Reading data from {file_path}")

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # Parse timestamp
                ts = parse_timestamp(record['timestamp'])
                ts_iso = ts.isoformat()

                # Create glucose reading document
                source_id = record.get('_id', f"imported_{line_num}")
                glucose_id = f"{user_id}_{source_id}"

                glucose_doc = {
                    "id": glucose_id,
                    "userId": user_id,
                    "timestamp": ts_iso,
                    "value": int(record['value']),
                    "trend": record.get('trend', 0),
                    "source": "gluroo",
                    "sourceId": source_id,
                    "imported": True,
                    "importedAt": datetime.now(timezone.utc).isoformat()
                }
                glucose_docs.append(glucose_doc)

                # Create treatment documents if insulin or carbs present
                insulin = record.get('insulin', 0) or 0
                carbs = record.get('carbs', 0) or 0

                if insulin > 0:
                    treatment_id = f"{user_id}_{source_id}_insulin"
                    treatment_doc = {
                        "id": treatment_id,
                        "userId": user_id,
                        "timestamp": ts_iso,
                        "type": "insulin",
                        "insulin": float(insulin),
                        "source": "gluroo",
                        "sourceId": source_id,
                        "imported": True
                    }
                    treatment_docs.append(treatment_doc)

                if carbs > 0:
                    treatment_id = f"{user_id}_{source_id}_carbs"
                    treatment_doc = {
                        "id": treatment_id,
                        "userId": user_id,
                        "timestamp": ts_iso,
                        "type": "carbs",
                        "carbs": float(carbs),
                        "protein": float(record.get('protein', 0) or 0),
                        "fat": float(record.get('fat', 0) or 0),
                        "source": "gluroo",
                        "sourceId": source_id,
                        "imported": True
                    }
                    treatment_docs.append(treatment_doc)

            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                continue

            # Progress logging
            if line_num % 10000 == 0:
                logger.info(f"Processed {line_num} records...")

    logger.info(f"Parsed {len(glucose_docs)} glucose readings and {len(treatment_docs)} treatments")

    # Upload glucose readings in batches
    logger.info("Uploading glucose readings...")
    glucose_success = 0
    glucose_errors = 0

    for i in range(0, len(glucose_docs), batch_size):
        batch = glucose_docs[i:i + batch_size]
        for doc in batch:
            try:
                glucose_container.upsert_item(doc)
                glucose_success += 1
            except Exception as e:
                glucose_errors += 1
                if glucose_errors <= 5:
                    logger.warning(f"Error upserting glucose: {e}")

        if (i + batch_size) % 5000 == 0:
            logger.info(f"Uploaded {i + batch_size} glucose readings...")

    logger.info(f"Glucose: {glucose_success} uploaded, {glucose_errors} errors")

    # Upload treatments in batches
    logger.info("Uploading treatments...")
    treatment_success = 0
    treatment_errors = 0

    for i in range(0, len(treatment_docs), batch_size):
        batch = treatment_docs[i:i + batch_size]
        for doc in batch:
            try:
                treatment_container.upsert_item(doc)
                treatment_success += 1
            except Exception as e:
                treatment_errors += 1
                if treatment_errors <= 5:
                    logger.warning(f"Error upserting treatment: {e}")

        if (i + batch_size) % 1000 == 0:
            logger.info(f"Uploaded {i + batch_size} treatments...")

    logger.info(f"Treatments: {treatment_success} uploaded, {treatment_errors} errors")

    return {
        "glucose_count": glucose_success,
        "glucose_errors": glucose_errors,
        "treatment_count": treatment_success,
        "treatment_errors": treatment_errors
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Gluroo data into CosmosDB")
    parser.add_argument("--user_id", required=True, help="User ID to associate with data")
    parser.add_argument("--file", required=True, help="Path to JSONL file")
    parser.add_argument("--database", default="T1D-AI-DB", help="CosmosDB database name")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for uploads")

    args = parser.parse_args()

    # Get credentials from environment
    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")

    if not cosmos_endpoint or not cosmos_key:
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY environment variables required")
        sys.exit(1)

    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    # Run ingestion
    result = asyncio.run(ingest_data(
        user_id=args.user_id,
        file_path=args.file,
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key,
        database_name=args.database,
        batch_size=args.batch_size
    ))

    logger.info(f"Ingestion complete: {result}")


if __name__ == "__main__":
    main()
