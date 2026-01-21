"""
Dexcom Share API Service

Pulls glucose data directly from Dexcom Share servers with minimal latency.
Uses pydexcom library for reliable connection to Dexcom Share API.

Note: This is the same API used by the Dexcom Follow app.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pydantic import BaseModel
from pydexcom import Dexcom


# Default credentials (can be overridden)
DEFAULT_DEXCOM_USERNAME = "+19376540197"
DEFAULT_DEXCOM_PASSWORD = "Emrys2018!"


class DexcomGlucoseReading(BaseModel):
    """A glucose reading from Dexcom Share API."""
    value: int  # mg/dL
    trend: int  # 1-7, maps to trend arrows
    trend_description: str
    timestamp: datetime

    @property
    def trend_arrow(self) -> str:
        """Convert trend number to arrow symbol."""
        arrows = {
            1: "⬆⬆",   # DoubleUp
            2: "⬆",    # SingleUp
            3: "↗",    # FortyFiveUp
            4: "→",    # Flat
            5: "↘",    # FortyFiveDown
            6: "⬇",    # SingleDown
            7: "⬇⬇",   # DoubleDown
        }
        return arrows.get(self.trend, "?")


class DexcomShareService:
    """
    Service for pulling glucose data directly from Dexcom Share API.
    Uses pydexcom library for reliable authentication and data fetching.

    Usage:
        service = DexcomShareService()  # Uses default credentials
        readings = service.get_glucose_readings(minutes=30, max_count=6)
    """

    # Trend name to number mapping
    TREND_MAP = {
        "rising quickly": 1,
        "rising": 2,
        "rising slightly": 3,
        "steady": 4,
        "falling slightly": 5,
        "falling": 6,
        "falling quickly": 7,
        "none": 4,  # Default to flat
        "not computable": 8,
        "rate out of range": 9,
    }

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        region: str = "us",  # "us", "ous", or "jp"
    ):
        self.username = username or DEFAULT_DEXCOM_USERNAME
        self.password = password or DEFAULT_DEXCOM_PASSWORD
        self.region = region
        self._dexcom: Optional[Dexcom] = None

    def _get_client(self) -> Dexcom:
        """Get or create pydexcom client."""
        if self._dexcom is None:
            self._dexcom = Dexcom(
                username=self.username,
                password=self.password,
                region=self.region
            )
        return self._dexcom

    def get_glucose_readings(
        self,
        minutes: int = 30,
        max_count: int = 6,
    ) -> List[DexcomGlucoseReading]:
        """
        Get recent glucose readings from Dexcom Share.

        Args:
            minutes: How many minutes back to look (max 1440 = 24 hours)
            max_count: Maximum number of readings to return

        Returns: List of DexcomGlucoseReading objects, newest first
        """
        dexcom = self._get_client()
        raw_readings = dexcom.get_glucose_readings(minutes=minutes, max_count=max_count)

        readings = []
        for r in raw_readings:
            # Convert trend description to number
            trend_desc = r.trend_description.lower() if r.trend_description else "steady"
            trend_num = self.TREND_MAP.get(trend_desc, 4)

            # Ensure timestamp is timezone-aware
            ts = r.datetime
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            reading = DexcomGlucoseReading(
                value=r.value,
                trend=trend_num,
                trend_description=r.trend_description or "Flat",
                timestamp=ts,
            )
            readings.append(reading)

        return readings

    def get_latest_reading(self) -> Optional[DexcomGlucoseReading]:
        """Get the most recent glucose reading."""
        readings = self.get_glucose_readings(minutes=30, max_count=1)
        return readings[0] if readings else None

    async def get_glucose_readings_async(
        self,
        minutes: int = 30,
        max_count: int = 6,
    ) -> List[DexcomGlucoseReading]:
        """Async wrapper for get_glucose_readings."""
        return await asyncio.to_thread(self.get_glucose_readings, minutes, max_count)

    async def get_latest_reading_async(self) -> Optional[DexcomGlucoseReading]:
        """Async wrapper for get_latest_reading."""
        return await asyncio.to_thread(self.get_latest_reading)


class DexcomAuthError(Exception):
    """Raised when Dexcom authentication fails."""
    pass


class DexcomAPIError(Exception):
    """Raised when a Dexcom API call fails."""
    pass


def test_dexcom_connection(username: Optional[str] = None, password: Optional[str] = None, region: str = "us") -> dict:
    """
    Test Dexcom Share connection and return latest reading.

    Returns: Dict with success status and reading info
    """
    try:
        service = DexcomShareService(username=username, password=password, region=region)
        reading = service.get_latest_reading()
        if reading:
            return {
                "success": True,
                "value": reading.value,
                "trend": reading.trend_arrow,
                "trend_description": reading.trend_description,
                "timestamp": reading.timestamp.isoformat(),
                "age_seconds": (datetime.now(timezone.utc) - reading.timestamp).total_seconds(),
            }
        return {"success": True, "message": "Authenticated but no recent readings"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# CLI test
if __name__ == "__main__":
    import sys

    # Use default credentials if not provided
    username = sys.argv[1] if len(sys.argv) > 1 else None
    password = sys.argv[2] if len(sys.argv) > 2 else None
    region = "ous" if "--ous" in sys.argv else "us"

    print(f"\nTesting Dexcom Share connection...")
    print(f"Username: {username or DEFAULT_DEXCOM_USERNAME}")

    result = test_dexcom_connection(username, password, region)

    if result.get("success"):
        if "value" in result:
            age_min = result["age_seconds"] / 60
            print(f"\n✓ Latest reading: {result['value']} mg/dL {result['trend']}")
            print(f"  Trend: {result['trend_description']}")
            print(f"  Timestamp: {result['timestamp']}")
            print(f"  Age: {age_min:.1f} minutes")
        else:
            print(f"✓ {result['message']}")
    else:
        print(f"✗ {result['error']}")
