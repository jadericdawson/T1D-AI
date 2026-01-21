#!/usr/bin/env python3
"""
Quick test script for Dexcom Share API connection.

Usage:
    python test_dexcom.py <dexcom_username> <dexcom_password>

    # For accounts outside US (EU, etc):
    python test_dexcom.py <dexcom_username> <dexcom_password> --ous

Note: This uses the Dexcom Share credentials (same as Dexcom Follow app),
NOT your Dexcom Clarity username.
"""
import asyncio
import sys
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, 'src')

from services.dexcom_service import DexcomShareService, DexcomAuthError, DexcomAPIError


async def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    username = sys.argv[1]
    password = sys.argv[2]
    ous = "--ous" in sys.argv

    region = "Outside US (OUS)" if ous else "US"
    print(f"\n{'='*50}")
    print(f"Testing Dexcom Share API")
    print(f"Username: {username}")
    print(f"Region: {region}")
    print(f"{'='*50}\n")

    service = DexcomShareService(username=username, password=password, ous=ous)

    try:
        # Step 1: Authenticate
        print("Step 1: Authenticating...")
        session_id = await service.authenticate()
        print(f"   ✓ Got session ID: {session_id[:8]}...\n")

        # Step 2: Get latest reading
        print("Step 2: Fetching latest reading...")
        reading = await service.get_latest_reading()

        if reading:
            age_seconds = (datetime.now(timezone.utc) - reading.timestamp).total_seconds()
            age_min = age_seconds / 60

            print(f"   ✓ Latest BG: {reading.value} mg/dL {reading.trend_arrow}")
            print(f"   ✓ Trend: {reading.trend_description}")
            print(f"   ✓ Timestamp: {reading.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   ✓ Age: {age_min:.1f} minutes\n")

            # Check freshness
            if age_min <= 5:
                print(f"   🟢 Data is FRESH (within 5 minutes)")
            elif age_min <= 10:
                print(f"   🟡 Data is slightly delayed ({age_min:.0f} min old)")
            else:
                print(f"   🔴 Data is STALE ({age_min:.0f} min old - is CGM active?)")
        else:
            print("   ⚠ No recent readings found")

        # Step 3: Get last 30 minutes of readings
        print("\nStep 3: Fetching last 30 minutes of readings...")
        readings = await service.get_glucose_readings(minutes=30, max_count=10)

        if readings:
            print(f"   ✓ Got {len(readings)} readings:\n")
            for r in readings:
                age_min = (datetime.now(timezone.utc) - r.timestamp).total_seconds() / 60
                print(f"      {r.timestamp.strftime('%H:%M:%S')} - {r.value:3d} mg/dL {r.trend_arrow} ({age_min:.0f}m ago)")
        else:
            print("   ⚠ No readings in last 30 minutes")

        print(f"\n{'='*50}")
        print("✓ Dexcom Share connection successful!")
        print(f"{'='*50}\n")

    except DexcomAuthError as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\nTips:")
        print("  - Make sure you're using Dexcom Share credentials")
        print("  - If outside US, try adding --ous flag")
        print("  - Check that sharing is enabled in Dexcom app")
        sys.exit(1)

    except DexcomAPIError as e:
        print(f"\n❌ API error: {e}")
        sys.exit(1)

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
