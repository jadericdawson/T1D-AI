#!/usr/bin/env python3
"""
BG Pressure Analysis Script

This script analyzes the BG Pressure calculation to find data point errors.
It simulates what the frontend receives and identifies discrepancies.

Usage:
    python scripts/analyze_bg_pressure.py

The script outputs a table showing:
- Timestamp
- Actual BG
- IOB, COB
- BG Pressure
- Difference (pressure - actual)

Look for anomalies in the difference column.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

# Import backend services
from models.schemas import Treatment, GlucoseReading
from services.iob_cob_service import IOBCOBService


def format_time(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%H:%M:%S")


def analyze_bg_pressure_calculation(
    glucose_readings: List[GlucoseReading],
    treatments: List[Treatment],
    isf: float = 50.0,
    icr: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Analyze BG Pressure calculation for each glucose reading.

    This replicates the calculation in glucose.py lines 220-256.
    """
    service = IOBCOBService.from_settings()
    bg_per_gram = isf / icr  # BG rise per gram of carbs

    results = []

    for reading in glucose_readings:
        reading_time = reading.timestamp.replace(tzinfo=None)

        # Calculate IOB and COB at this point in time
        hist_iob = service.calculate_iob(treatments, at_time=reading_time)
        hist_cob = service.calculate_cob(treatments, at_time=reading_time)

        # Calculate REMAINING FUTURE effect (what is yet to act on BG)
        remaining_iob_effect = hist_iob * isf
        remaining_cob_effect = hist_cob * bg_per_gram

        # Net future pressure on BG
        net_future_effect = remaining_cob_effect - remaining_iob_effect

        # BG Pressure = current BG + net future effect
        bg_pressure = reading.value + net_future_effect

        results.append({
            'timestamp': reading_time,
            'actual_bg': reading.value,
            'iob': hist_iob,
            'cob': hist_cob,
            'iob_effect': remaining_iob_effect,
            'cob_effect': remaining_cob_effect,
            'net_effect': net_future_effect,
            'bg_pressure': bg_pressure,
            'diff': bg_pressure - reading.value
        })

    return results


def print_analysis_table(results: List[Dict[str, Any]], max_rows: int = 100):
    """Print analysis results as a formatted table."""

    print("\n" + "=" * 120)
    print("BG PRESSURE ANALYSIS - Go number by number to find the error")
    print("=" * 120)
    print()
    print(f"{'Time':<12} {'BG':<8} {'IOB':<8} {'COB':<8} {'IOB Eff':<10} {'COB Eff':<10} {'Net':<10} {'Pressure':<10} {'Diff':<10}")
    print("-" * 120)

    # Sort by timestamp
    sorted_results = sorted(results, key=lambda x: x['timestamp'])

    # Find anomalies (large differences)
    anomalies = []

    for i, r in enumerate(sorted_results[:max_rows]):
        time_str = format_time(r['timestamp'])
        diff = r['diff']

        # Flag anomalies
        is_anomaly = abs(diff) > 100  # Large pressure difference
        marker = " ***" if is_anomaly else ""

        if is_anomaly:
            anomalies.append((r['timestamp'], r))

        print(f"{time_str:<12} {r['actual_bg']:<8.0f} {r['iob']:<8.2f} {r['cob']:<8.1f} "
              f"{r['iob_effect']:<10.1f} {r['cob_effect']:<10.1f} {r['net_effect']:<+10.1f} "
              f"{r['bg_pressure']:<10.0f} {diff:<+10.0f}{marker}")

    print("-" * 120)
    print()

    # Summary statistics
    if results:
        diffs = [r['diff'] for r in results]
        print("STATISTICS:")
        print(f"  Total points: {len(results)}")
        print(f"  Min diff: {min(diffs):+.0f} mg/dL")
        print(f"  Max diff: {max(diffs):+.0f} mg/dL")
        print(f"  Avg diff: {sum(diffs)/len(diffs):+.0f} mg/dL")
        print()

    # List anomalies
    if anomalies:
        print("ANOMALIES (|diff| > 100 mg/dL):")
        for ts, r in anomalies:
            print(f"  {format_time(ts)}: BG={r['actual_bg']:.0f}, IOB={r['iob']:.2f}, COB={r['cob']:.1f} → Pressure={r['bg_pressure']:.0f} (diff={r['diff']:+.0f})")
        print()
    else:
        print("NO ANOMALIES FOUND (all |diff| <= 100 mg/dL)")
        print()


def create_test_data():
    """Create test data to simulate real scenario."""

    # Simulate glucose readings over 24 hours
    now = datetime.utcnow()
    glucose_readings = []

    # Create readings every 5 minutes for 6 hours
    for i in range(72):  # 6 hours * 12 readings/hour
        time = now - timedelta(minutes=i * 5)
        # Simulate a realistic glucose pattern
        base = 120 + 30 * (0.5 - abs((i % 24) / 24 - 0.5))  # Varies 90-150
        glucose_readings.append(GlucoseReading(
            id=f"test-{i}",
            userId="test-user",
            timestamp=time,
            value=int(base),
            trend="Flat"
        ))

    # Simulate treatments
    treatments = []

    # Meal 2 hours ago
    meal_time = now - timedelta(hours=2)
    treatments.append(Treatment(
        id="meal-1",
        userId="test-user",
        timestamp=meal_time,
        carbs=45,
        insulin=4.5,
        mealDescription="Lunch"
    ))

    # Correction 1 hour ago
    correction_time = now - timedelta(hours=1)
    treatments.append(Treatment(
        id="correction-1",
        userId="test-user",
        timestamp=correction_time,
        insulin=1.5
    ))

    # Snack 30 min ago
    snack_time = now - timedelta(minutes=30)
    treatments.append(Treatment(
        id="snack-1",
        userId="test-user",
        timestamp=snack_time,
        carbs=15
    ))

    return glucose_readings, treatments


async def main():
    """Main entry point."""

    print("\n" + "=" * 80)
    print("BG PRESSURE DATA ANALYSIS")
    print("=" * 80)
    print()

    # Check if we can connect to the database
    try:
        from database.repositories import GlucoseRepository, TreatmentRepository, UserRepository
        from config import get_settings

        settings = get_settings()

        # Try to get real data
        user_id = "Jeric Dawson"  # The primary user

        glucose_repo = GlucoseRepository()
        treatment_repo = TreatmentRepository()
        user_repo = UserRepository()

        print(f"Fetching data for user: {user_id}")
        print()

        # Get last 6 hours of data
        start_time = datetime.utcnow() - timedelta(hours=6)

        glucose_readings = await glucose_repo.get_history(user_id, start_time)
        treatments = await treatment_repo.get_recent(user_id, hours=8)

        print(f"Found {len(glucose_readings)} glucose readings")
        print(f"Found {len(treatments)} treatments")
        print()

        # Get user settings for ISF and ICR
        user = await user_repo.get_by_id(user_id)
        isf = user.settings.insulinSensitivity if user and user.settings else 50.0
        icr = user.settings.carbRatio if user and user.settings else 10.0

        print(f"Using ISF={isf}, ICR={icr}")

        # Analyze
        if glucose_readings:
            results = analyze_bg_pressure_calculation(glucose_readings, treatments, isf, icr)
            print_analysis_table(results)
        else:
            print("No glucose data found!")

    except Exception as e:
        print(f"Could not connect to database: {e}")
        print()
        print("Using TEST DATA instead...")
        print()

        glucose_readings, treatments = create_test_data()
        print(f"Created {len(glucose_readings)} test glucose readings")
        print(f"Created {len(treatments)} test treatments")

        # Use default ISF/ICR
        isf = 50.0
        icr = 10.0

        results = analyze_bg_pressure_calculation(glucose_readings, treatments, isf, icr)
        print_analysis_table(results)


if __name__ == "__main__":
    asyncio.run(main())
