"""Test Treatment Inference - Run inference detection on last 24 hours of data"""
import asyncio
import os
import sys
import subprocess
import hashlib
from datetime import datetime, timezone, timedelta

# Get secrets from Azure
def get_azure_setting(name):
    result = subprocess.run(
        ["az", "webapp", "config", "appsettings", "list", "--name", "t1d-ai",
         "--resource-group", "rg-knowledge2ai-eastus", "--query", f"[?name=='{name}'].value", "-o", "tsv"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Set environment BEFORE imports
os.environ["COSMOS_ENDPOINT"] = get_azure_setting("COSMOS_ENDPOINT")
os.environ["COSMOS_KEY"] = get_azure_setting("COSMOS_KEY")
os.environ["COSMOS_DATABASE"] = get_azure_setting("COSMOS_DATABASE") or "T1D-AI-DB"
os.environ["GPT41_ENDPOINT"] = get_azure_setting("GPT41_ENDPOINT") or "https://placeholder.openai.azure.com"
os.environ["AZURE_OPENAI_KEY"] = get_azure_setting("AZURE_OPENAI_KEY") or "placeholder"
os.environ["STORAGE_ACCOUNT_URL"] = get_azure_setting("STORAGE_ACCOUNT_URL") or "https://placeholder.blob.core.windows.net"
os.environ["STORAGE_CONNECTION_STRING"] = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING") or "DefaultEndpointsProtocol=https;AccountName=placeholder"
os.environ["JWT_SECRET_KEY"] = get_azure_setting("JWT_SECRET_KEY") or "placeholder"

sys.path.insert(0, "src")

from azure.cosmos import CosmosClient
from models.schemas import GlucoseReading, Treatment, TreatmentType
from services.treatment_inference_service import TreatmentInferenceService

# CosmosDB config
COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
COSMOS_KEY = os.environ["COSMOS_KEY"]
COSMOS_DATABASE = os.environ["COSMOS_DATABASE"]
USER_ID = "05bf0083-5598-43a5-aa7f-bd70b1f1be57"


async def test_inference_on_24h():
    """Test inference detection on last 24 hours of data"""
    print("=" * 60)
    print("TREATMENT INFERENCE TEST - Last 24 Hours")
    print("=" * 60)

    # Connect to CosmosDB
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.get_database_client(COSMOS_DATABASE)
    glucose_container = db.get_container_client("glucose_readings")
    treatment_container = db.get_container_client("treatments")

    # Get last 24 hours of glucose data
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    cutoff_str = cutoff.isoformat()

    print(f"\nFetching data since: {cutoff_str}")

    glucose_query = """
        SELECT * FROM c
        WHERE c.userId = @userId AND c.timestamp >= @cutoff
        ORDER BY c.timestamp ASC
    """
    glucose_items = list(glucose_container.query_items(
        query=glucose_query,
        parameters=[
            {"name": "@userId", "value": USER_ID},
            {"name": "@cutoff", "value": cutoff_str}
        ],
        enable_cross_partition_query=True
    ))

    print(f"Found {len(glucose_items)} glucose readings in last 24h")

    # Convert to GlucoseReading objects
    readings = []
    for item in glucose_items:
        try:
            readings.append(GlucoseReading(
                id=item['id'],
                userId=item['userId'],
                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                value=item['value'],
                trend=item.get('trend'),
                source=item.get('source', 'gluroo'),
                sourceId=item.get('sourceId')
            ))
        except Exception as e:
            continue

    print(f"Parsed {len(readings)} valid readings")

    if readings:
        bg_values = [r.value for r in readings]
        print(f"BG range: {min(bg_values)} - {max(bg_values)} mg/dL")

    # Get treatments in same period
    treatment_query = """
        SELECT * FROM c
        WHERE c.userId = @userId AND c.timestamp >= @cutoff
        ORDER BY c.timestamp ASC
    """
    treatment_items = list(treatment_container.query_items(
        query=treatment_query,
        parameters=[
            {"name": "@userId", "value": USER_ID},
            {"name": "@cutoff", "value": cutoff_str}
        ],
        enable_cross_partition_query=True
    ))

    print(f"Found {len(treatment_items)} treatments in last 24h")

    # Parse treatments
    known_treatments = []
    for item in treatment_items:
        try:
            t_type = item.get('type', 'carbs')
            if t_type == 'insulin':
                t_type = TreatmentType.INSULIN
            else:
                t_type = TreatmentType.CARBS

            known_treatments.append(Treatment(
                id=item['id'],
                userId=item['userId'],
                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                type=t_type,
                insulin=item.get('insulin'),
                carbs=item.get('carbs'),
                notes=item.get('notes'),
                source=item.get('source', 'gluroo'),
                isInferred=item.get('isInferred', False)
            ))
        except Exception as e:
            continue

    # Print known treatments
    print("\n" + "-" * 40)
    print("KNOWN TREATMENTS:")
    print("-" * 40)
    for t in sorted(known_treatments, key=lambda x: x.timestamp):
        time_str = t.timestamp.strftime("%H:%M")
        if t.carbs:
            print(f"  {time_str} - {t.carbs}g carbs {'[INFERRED]' if t.isInferred else ''} {t.notes or ''}")
        if t.insulin:
            print(f"  {time_str} - {t.insulin}U insulin {'[INFERRED]' if t.isInferred else ''}")

    # Run inference in sliding windows across the 24 hours
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE DETECTION...")
    print("=" * 60)

    inference_service = TreatmentInferenceService()

    # Process in 3-hour windows, sliding by 1 hour
    window_hours = 3
    step_hours = 2

    all_inferred = []

    for hour_offset in range(0, 24 - window_hours, step_hours):
        window_start = cutoff + timedelta(hours=hour_offset)
        window_end = window_start + timedelta(hours=window_hours)

        # Get readings in this window
        window_readings = [
            r for r in readings
            if window_start <= r.timestamp < window_end
        ]

        if len(window_readings) < 15:
            continue

        # Get known treatments in this window (non-inferred only)
        window_treatments = [
            t for t in known_treatments
            if window_start <= t.timestamp < window_end and not t.isInferred
        ]

        print(f"\nWindow: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")
        print(f"  Readings: {len(window_readings)}, Known treatments: {len(window_treatments)}")

        # Analyze pattern
        bg_values = [r.value for r in window_readings]
        start_bg = bg_values[0]
        end_bg = bg_values[-1]
        max_bg = max(bg_values)
        min_bg = min(bg_values)

        print(f"  BG: {start_bg} -> {end_bg} (range: {min_bg}-{max_bg})")

        # Check if there's an unexplained pattern
        # Simple heuristic: significant change without corresponding treatment
        has_carbs = any(t.carbs for t in window_treatments)
        has_insulin = any(t.insulin for t in window_treatments)

        rise = max_bg - min_bg
        if rise > 30 and not has_carbs:
            print(f"  ⚠️  BG rose {rise} mg/dL without logged carbs!")

        drop = start_bg - end_bg if start_bg > end_bg else 0
        if drop > 40 and not has_insulin:
            print(f"  ⚠️  BG dropped {drop} mg/dL without logged insulin!")

        # Run inference (dry run - don't actually create treatments)
        try:
            # Temporarily disable treatment creation for testing
            # Just analyze the pattern
            pattern = inference_service._analyze_residual_pattern(
                [r.value - start_bg for r in window_readings],  # Simple residual from start
                window_readings
            )

            print(f"  Pattern: {pattern['type']}, magnitude: {pattern.get('magnitude', 0):.0f}")

            if pattern['type'] in ['rise', 'drop', 'rise_then_drop'] and pattern.get('magnitude', 0) > 25:
                # Run grid search
                best = inference_service._grid_search_best_fit(
                    window_readings,
                    window_treatments,
                    pattern
                )

                if best and best.confidence > 0.5:
                    print(f"  🔍 INFERENCE CANDIDATE:")
                    if best.carbs > 0:
                        print(f"     Carbs: {best.carbs}g at {best.carb_time.strftime('%H:%M')}")
                    if best.insulin > 0:
                        print(f"     Insulin: {best.insulin}U at {best.insulin_time.strftime('%H:%M')}")
                    print(f"     Confidence: {best.confidence:.0%}")
                    print(f"     Residual error: {best.residual_error:.0f}")
                    all_inferred.append(best)

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total windows analyzed: {24 // step_hours}")
    print(f"Inference candidates found: {len(all_inferred)}")

    if all_inferred:
        print("\nAll candidates:")
        for i, inf in enumerate(all_inferred, 1):
            parts = []
            if inf.carbs > 0:
                parts.append(f"{inf.carbs}g carbs @ {inf.carb_time.strftime('%H:%M')}")
            if inf.insulin > 0:
                parts.append(f"{inf.insulin}U insulin @ {inf.insulin_time.strftime('%H:%M')}")
            print(f"  {i}. {', '.join(parts)} (conf: {inf.confidence:.0%})")


if __name__ == "__main__":
    asyncio.run(test_inference_on_24h())
