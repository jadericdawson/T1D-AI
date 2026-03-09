#!/usr/bin/env python3
"""
One-time cleanup script to remove duplicate treatment entries in CosmosDB.

Duplicate treatments arise because Tandem sync pushes to CosmosDB AND Gluroo,
then Gluroo sync pulls the same entries back with different document IDs.
Additionally, Gluroo API sometimes returns multiple object IDs for the same event.

Strategy:
1. Fetch all treatments for the user
2. Group by (timestamp_rounded_to_2min, carbs_or_insulin_value)
3. For each group with >1 entry, keep the best one (prefer tandem source, then earliest)
4. Delete the rest

Usage:
    cd backend/src && python ../scripts/dedup_treatments.py [--dry-run]
"""
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Add backend/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from azure.cosmos import CosmosClient

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")
COSMOS_KEY = os.getenv("COSMOS_KEY", "")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "T1D-AI-DB")
USER_ID = os.getenv("GLUROO_USER_ID", "")

DRY_RUN = "--dry-run" in sys.argv


def round_timestamp(ts_str: str, minutes: int = 2) -> str:
    """Round a timestamp to the nearest N minutes for grouping."""
    try:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        # Round to nearest 2-minute bucket
        bucket = ts.replace(second=0, microsecond=0)
        bucket = bucket.replace(minute=(bucket.minute // minutes) * minutes)
        return bucket.isoformat()
    except Exception:
        return ts_str


def dedup_key(treatment: dict) -> tuple:
    """Create a dedup key from a treatment based on type, time, and value."""
    ts_bucket = round_timestamp(treatment.get('timestamp', ''), minutes=2)
    carbs = treatment.get('carbs')
    insulin = treatment.get('insulin')

    if carbs and float(carbs) > 0:
        # Round carbs to nearest 1g for matching
        return (ts_bucket, 'carbs', round(float(carbs)))
    elif insulin and float(insulin) > 0:
        # Round insulin to nearest 0.1U for matching
        return (ts_bucket, 'insulin', round(float(insulin), 1))
    else:
        # Basal or other - use exact values
        return (ts_bucket, treatment.get('type', ''), 0)


def pick_best(group: list) -> dict:
    """Pick the best treatment from a group of duplicates.

    Priority:
    1. Tandem source (ground truth from pump)
    2. Has enrichment data (glycemicIndex, notes)
    3. Has user edits
    4. Earliest document (first synced)
    """
    # Prefer tandem source
    tandem = [t for t in group if t.get('source') == 'tandem']
    if tandem:
        # Among tandem entries, prefer one with notes
        with_notes = [t for t in tandem if t.get('notes')]
        return with_notes[0] if with_notes else tandem[0]

    # Among gluroo entries, prefer enriched ones
    enriched = [t for t in group if t.get('glycemicIndex')]
    if enriched:
        return enriched[0]

    # Prefer user-edited
    edited = [t for t in group if t.get('userEdited')]
    if edited:
        return edited[0]

    # Prefer one with notes
    with_notes = [t for t in group if t.get('notes') and
                  not t['notes'].startswith(('Bolus:', 'Food:'))]
    if with_notes:
        return with_notes[0]

    return group[0]


def main():
    if not COSMOS_ENDPOINT or not COSMOS_KEY or not USER_ID:
        print("ERROR: Set COSMOS_ENDPOINT, COSMOS_KEY, and GLUROO_USER_ID env vars")
        sys.exit(1)

    print(f"{'DRY RUN - ' if DRY_RUN else ''}Deduplicating treatments for user: {USER_ID}")

    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.get_database_client(COSMOS_DATABASE)
    container = db.get_container_client("treatments")

    # Fetch all treatments
    print("Fetching all treatments...")
    query = "SELECT * FROM c WHERE c.userId = @userId"
    all_treatments = list(container.query_items(
        query=query,
        parameters=[{"name": "@userId", "value": USER_ID}],
        partition_key=USER_ID,
    ))
    print(f"  Total treatments: {len(all_treatments)}")

    # Group by dedup key
    groups = defaultdict(list)
    for t in all_treatments:
        key = dedup_key(t)
        groups[key].append(t)

    # Find duplicates
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    total_dups = sum(len(v) - 1 for v in dup_groups.values())

    print(f"  Duplicate groups: {len(dup_groups)}")
    print(f"  Total duplicates to remove: {total_dups}")
    print()

    if total_dups == 0:
        print("No duplicates found!")
        return

    # Process each duplicate group
    deleted = 0
    kept = 0
    for key, group in sorted(dup_groups.items()):
        best = pick_best(group)
        to_delete = [t for t in group if t['id'] != best['id']]

        ts_bucket, ttype, value = key
        print(f"  {ts_bucket} | {ttype}={value} | {len(group)} entries -> keeping {best['id'][:40]}... ({best.get('source', '?')})")

        for dup in to_delete:
            print(f"    DELETE {dup['id'][:50]}... (source={dup.get('source', '?')})")
            if not DRY_RUN:
                try:
                    container.delete_item(item=dup['id'], partition_key=USER_ID)
                    deleted += 1
                except Exception as e:
                    print(f"    ERROR deleting {dup['id']}: {e}")
            else:
                deleted += 1
        kept += 1

    print()
    print(f"{'Would delete' if DRY_RUN else 'Deleted'}: {deleted} duplicate treatments")
    print(f"Kept: {kept} canonical treatments (from {kept + deleted} total in dup groups)")
    print(f"Unique (no dups): {len(groups) - len(dup_groups)}")
    print(f"Final treatment count: {len(all_treatments) - deleted}")


if __name__ == "__main__":
    main()
