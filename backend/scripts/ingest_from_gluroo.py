#!/usr/bin/env python3
"""
Ingest data directly from Gluroo API into CosmosDB.
Keeps entries and treatments SEPARATE (proper data model).

Usage:
    python scripts/ingest_from_gluroo.py --user_id emrys
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from azure.cosmos import CosmosClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gluroo API config
BASE_URL = "https://81ca.ns.gluroo.com"
API_SECRET = "81ca21b8-9002-4d24-8492-d7546ff6fade"
HEADERS = {"API-SECRET": hashlib.sha1(API_SECRET.encode()).hexdigest()}


def fetch_entries_batch(last_mills: int = None, count: int = 1000) -> list:
    """Fetch a batch of glucose entries."""
    url = f"{BASE_URL}/api/v1/entries.json?count={count}"
    if last_mills:
        url += f"&find[mills][$lt]={last_mills}"

    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        return resp.json() or []
    return []


def fetch_all_treatments() -> list:
    """Fetch all treatments from Gluroo."""
    all_treatments = []

    for event_type in ["Correction Bolus", "Carb Correction"]:
        url = f"{BASE_URL}/api/v1/treatments.json?count=10000&find[eventType]={event_type.replace(' ', '%20')}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            treatments = resp.json() or []
            logger.info(f"Fetched {len(treatments)} {event_type} treatments")
            all_treatments.extend(treatments)

    return all_treatments


def convert_entry_to_cosmos(entry: dict, user_id: str) -> dict:
    """Convert Gluroo entry to CosmosDB glucose_reading format."""
    source_id = entry.get('_id')
    mills = entry.get('mills', entry.get('date', 0))
    timestamp = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)

    return {
        "id": f"{user_id}_{source_id}",
        "userId": user_id,
        "timestamp": timestamp.isoformat(),
        "value": int(entry.get('sgv', entry.get('value', 0))),
        "trend": entry.get('trend', 0),
        "direction": entry.get('direction', 'Flat'),
        "source": "gluroo",
        "sourceId": source_id
    }


def convert_treatment_to_cosmos(treatment: dict, user_id: str) -> dict:
    """Convert Gluroo treatment to CosmosDB treatment format."""
    source_id = treatment.get('_id')
    created_at = treatment.get('created_at', treatment.get('createdAt'))

    # Parse timestamp
    if created_at:
        try:
            timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except:
            mills = treatment.get('mills', 0)
            timestamp = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)
    else:
        mills = treatment.get('mills', 0)
        timestamp = datetime.fromtimestamp(mills / 1000, tz=timezone.utc)

    # Determine type
    insulin = treatment.get('insulin', 0) or 0
    carbs = treatment.get('carbs', 0) or 0

    if insulin > 0:
        treatment_type = "insulin"
    elif carbs > 0:
        treatment_type = "carbs"
    else:
        return None  # Skip empty treatments

    return {
        "id": f"{user_id}_{source_id}",
        "userId": user_id,
        "timestamp": timestamp.isoformat(),
        "type": treatment_type,
        "insulin": float(insulin) if insulin else None,
        "carbs": float(carbs) if carbs else None,
        "protein": float(treatment.get('protein', 0) or 0) if treatment.get('protein') else None,
        "fat": float(treatment.get('fat', 0) or 0) if treatment.get('fat') else None,
        "notes": treatment.get('notes'),
        "eventType": treatment.get('eventType'),
        "source": "gluroo",
        "sourceId": source_id
    }


def ingest_data(
    user_id: str,
    cosmos_endpoint: str,
    cosmos_key: str,
    database_name: str = "T1D-AI-DB",
    max_entries: int = 100000  # Limit to ~1 year of data
):
    """Ingest data from Gluroo to CosmosDB."""

    # Initialize Cosmos
    client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = client.get_database_client(database_name)
    glucose_container = database.get_container_client("glucose_readings")
    treatment_container = database.get_container_client("treatments")

    # Fetch treatments first (fast, smaller dataset)
    logger.info("Fetching treatments from Gluroo...")
    treatments = fetch_all_treatments()
    logger.info(f"Total treatments: {len(treatments)}")

    # Dedupe by _id
    treatments_by_id = {t['_id']: t for t in treatments}
    unique_treatments = list(treatments_by_id.values())
    logger.info(f"Unique treatments: {len(unique_treatments)}")

    # Convert and upload treatments
    logger.info("Uploading treatments to CosmosDB...")
    treatment_success = 0
    treatment_errors = 0

    for t in unique_treatments:
        doc = convert_treatment_to_cosmos(t, user_id)
        if doc:
            try:
                treatment_container.upsert_item(doc)
                treatment_success += 1
            except Exception as e:
                treatment_errors += 1
                if treatment_errors <= 3:
                    logger.warning(f"Error upserting treatment: {e}")

    logger.info(f"Treatments: {treatment_success} uploaded, {treatment_errors} errors")

    # Fetch entries in batches
    logger.info("Fetching glucose entries from Gluroo...")
    all_entries = []
    last_mills = None

    while len(all_entries) < max_entries:
        entries = fetch_entries_batch(last_mills, count=1000)
        if not entries:
            break

        all_entries.extend(entries)
        last_mills = min(e.get('mills', 0) for e in entries)

        if len(all_entries) % 10000 == 0:
            logger.info(f"Fetched {len(all_entries)} entries...")

        if len(entries) < 1000:
            break

    logger.info(f"Total entries fetched: {len(all_entries)}")

    # Dedupe by _id
    entries_by_id = {e['_id']: e for e in all_entries}
    unique_entries = list(entries_by_id.values())
    logger.info(f"Unique entries: {len(unique_entries)}")

    # Convert and upload entries
    logger.info("Uploading glucose readings to CosmosDB...")
    entry_success = 0
    entry_errors = 0

    for i, e in enumerate(unique_entries):
        doc = convert_entry_to_cosmos(e, user_id)
        try:
            glucose_container.upsert_item(doc)
            entry_success += 1
        except Exception as ex:
            entry_errors += 1
            if entry_errors <= 3:
                logger.warning(f"Error upserting entry: {ex}")

        if (i + 1) % 10000 == 0:
            logger.info(f"Uploaded {i + 1} glucose readings...")

    logger.info(f"Glucose readings: {entry_success} uploaded, {entry_errors} errors")

    # Summary
    if unique_entries:
        first_ts = min(e.get('mills', 0) for e in unique_entries)
        last_ts = max(e.get('mills', 0) for e in unique_entries)
        first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        last_date = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        logger.info(f"Date range: {first_date.date()} to {last_date.date()}")

    return {
        "glucose_count": entry_success,
        "treatment_count": treatment_success,
        "glucose_errors": entry_errors,
        "treatment_errors": treatment_errors
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest Gluroo data into CosmosDB")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--database", default="T1D-AI-DB", help="Database name")
    parser.add_argument("--max_entries", type=int, default=100000, help="Max entries to fetch")

    args = parser.parse_args()

    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")

    if not cosmos_endpoint or not cosmos_key:
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY required")
        sys.exit(1)

    result = ingest_data(
        user_id=args.user_id,
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key,
        database_name=args.database,
        max_entries=args.max_entries
    )

    logger.info(f"Ingestion complete: {result}")


if __name__ == "__main__":
    main()
