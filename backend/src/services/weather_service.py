"""
Weather Service for T1D-AI
Fetches weather data from OpenWeatherMap and extracts features for ML predictions.

Weather affects blood glucose through:
1. Temperature: Heat can increase insulin sensitivity
2. Humidity: High humidity can affect glucose readings
3. Barometric pressure: Some studies show correlation with glucose levels
4. Activity score: Weather affects likelihood of outdoor activity
"""
import logging
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """Current weather data from OpenWeatherMap."""
    temperature_c: float  # Temperature in Celsius
    humidity: float  # Humidity percentage (0-100)
    pressure_hpa: float  # Barometric pressure in hPa
    wind_speed_ms: float  # Wind speed in m/s
    cloud_cover: float  # Cloud cover percentage (0-100)
    conditions: str  # Weather description (e.g., "clear sky", "light rain")
    uv_index: Optional[float] = None  # UV index (if available)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=None))
    location: str = ""


@dataclass
class WeatherFeatures:
    """Normalized weather features for ML models."""
    # Temperature deviation from comfort zone (22C)
    # Range: -1 (very cold, <0C) to +1 (very hot, >35C)
    temp_normalized: float

    # Humidity normalized to 0-1 scale
    humidity_normalized: float

    # Pressure deviation from standard (1013.25 hPa)
    # Range: -1 (very low) to +1 (very high)
    pressure_normalized: float

    # Activity-friendliness score
    # Range: -1 (bad for outdoor activity) to +1 (great for activity)
    activity_score: float

    # Raw values for reference
    temperature_c: float
    humidity: float
    pressure_hpa: float


class WeatherService:
    """
    Weather data fetching and feature extraction service.

    Uses OpenWeatherMap API with caching to minimize API calls.
    """

    # OpenWeatherMap free tier limits
    API_BASE_URL = "https://api.openweathermap.org/data/2.5"
    COMFORT_TEMP_C = 22.0
    STANDARD_PRESSURE_HPA = 1013.25

    def __init__(self, api_key: Optional[str] = None, cache_minutes: int = 15):
        """
        Initialize the weather service.

        Args:
            api_key: OpenWeatherMap API key (uses settings if not provided)
            cache_minutes: Minutes to cache weather data
        """
        settings = get_settings()
        self.api_key = api_key or settings.openweathermap_api_key
        self.cache_minutes = cache_minutes
        self._cache: Dict[str, tuple[datetime, WeatherData]] = {}
        self._enabled = bool(self.api_key)

        if not self._enabled:
            logger.warning(
                "OpenWeatherMap API key not configured. "
                "Weather features will use default values."
            )

    @property
    def is_enabled(self) -> bool:
        """Check if weather service is enabled (API key configured)."""
        return self._enabled

    async def get_current_weather(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False
    ) -> Optional[WeatherData]:
        """
        Fetch current weather data for a location.

        Args:
            lat: Latitude
            lon: Longitude
            force_refresh: If True, bypass cache

        Returns:
            WeatherData or None if unavailable
        """
        if not self._enabled:
            return None

        # Check cache
        cache_key = f"{lat:.2f},{lon:.2f}"
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(minutes=self.cache_minutes):
                logger.debug(f"Using cached weather data for {cache_key}")
                return cached_data

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.API_BASE_URL}/weather",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": "metric"
                    }
                )
                response.raise_for_status()
                data = response.json()

            weather = WeatherData(
                temperature_c=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                pressure_hpa=data["main"]["pressure"],
                wind_speed_ms=data["wind"]["speed"],
                cloud_cover=data.get("clouds", {}).get("all", 0),
                conditions=data["weather"][0]["description"] if data.get("weather") else "unknown",
                timestamp=datetime.utcnow(),
                location=data.get("name", "")
            )

            # Cache the result
            self._cache[cache_key] = (datetime.utcnow(), weather)
            logger.info(f"Fetched weather for {weather.location}: {weather.temperature_c}C, {weather.conditions}")

            return weather

        except httpx.HTTPStatusError as e:
            logger.error(f"Weather API HTTP error: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Weather API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            return None

    def extract_features(self, weather: Optional[WeatherData] = None) -> WeatherFeatures:
        """
        Extract normalized features from weather data.

        Args:
            weather: WeatherData object (uses defaults if None)

        Returns:
            WeatherFeatures with normalized values
        """
        if weather is None:
            # Return default (neutral) features when weather unavailable
            return WeatherFeatures(
                temp_normalized=0.0,
                humidity_normalized=0.5,
                pressure_normalized=0.0,
                activity_score=0.0,
                temperature_c=self.COMFORT_TEMP_C,
                humidity=50.0,
                pressure_hpa=self.STANDARD_PRESSURE_HPA
            )

        # Normalize temperature: -1 (very cold) to +1 (very hot)
        # 0C -> -0.5, 22C -> 0, 35C -> 0.5, 44C -> 1.0
        temp_deviation = weather.temperature_c - self.COMFORT_TEMP_C
        temp_normalized = max(-1.0, min(1.0, temp_deviation / 22.0))

        # Normalize humidity: 0-1 scale
        humidity_normalized = weather.humidity / 100.0

        # Normalize pressure: deviation from standard
        # Low pressure (storm): < 1000 hPa
        # High pressure (clear): > 1025 hPa
        pressure_deviation = weather.pressure_hpa - self.STANDARD_PRESSURE_HPA
        pressure_normalized = max(-1.0, min(1.0, pressure_deviation / 25.0))

        # Calculate activity score based on multiple factors
        activity_score = self._calculate_activity_score(weather)

        return WeatherFeatures(
            temp_normalized=round(temp_normalized, 3),
            humidity_normalized=round(humidity_normalized, 3),
            pressure_normalized=round(pressure_normalized, 3),
            activity_score=round(activity_score, 3),
            temperature_c=weather.temperature_c,
            humidity=weather.humidity,
            pressure_hpa=weather.pressure_hpa
        )

    def _calculate_activity_score(self, weather: WeatherData) -> float:
        """
        Calculate how suitable weather is for outdoor activity.

        Returns:
            Score from -1 (bad for activity) to +1 (great for activity)
        """
        score = 0.0

        # Temperature factor (-1 to +1)
        # Ideal: 18-25C
        temp = weather.temperature_c
        if 18 <= temp <= 25:
            temp_score = 1.0
        elif temp < 0 or temp > 40:
            temp_score = -1.0
        elif temp < 18:
            temp_score = (temp - 0) / 18.0 - 0.5  # 0C = -0.5, 18C = 0.5
        else:  # temp > 25
            temp_score = 1.0 - (temp - 25) / 15.0  # 25C = 1.0, 40C = 0.0

        # Humidity factor (-0.5 to +0.5)
        # Ideal: 40-60%
        hum = weather.humidity
        if 40 <= hum <= 60:
            hum_score = 0.5
        elif hum < 20 or hum > 80:
            hum_score = -0.5
        else:
            hum_score = 0.0

        # Wind factor (-0.3 to +0.3)
        wind = weather.wind_speed_ms
        if wind < 5:
            wind_score = 0.3
        elif wind > 15:
            wind_score = -0.3
        else:
            wind_score = 0.3 - (wind - 5) * 0.06

        # Cloud/rain factor (-0.5 to +0.3)
        conditions = weather.conditions.lower()
        if "rain" in conditions or "storm" in conditions or "snow" in conditions:
            cond_score = -0.5
        elif "clear" in conditions:
            cond_score = 0.3
        elif "cloud" in conditions:
            cond_score = 0.0
        else:
            cond_score = 0.0

        # Combine factors
        score = temp_score * 0.4 + hum_score + wind_score + cond_score
        return max(-1.0, min(1.0, score))

    def get_feature_dict(self, weather: Optional[WeatherData] = None) -> Dict[str, float]:
        """
        Get weather features as a dictionary for DataFrame insertion.

        Args:
            weather: WeatherData object (uses defaults if None)

        Returns:
            Dictionary with feature names and values
        """
        features = self.extract_features(weather)
        return {
            "weather_temp_normalized": features.temp_normalized,
            "weather_humidity_normalized": features.humidity_normalized,
            "weather_pressure_normalized": features.pressure_normalized,
            "weather_activity_score": features.activity_score,
        }


# Singleton instance
_weather_service: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    """Get or create the global weather service instance."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


async def get_weather_features(lat: float, lon: float) -> Dict[str, float]:
    """
    Convenience function to get weather features for a location.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary with weather feature values
    """
    service = get_weather_service()
    weather = await service.get_current_weather(lat, lon)
    return service.get_feature_dict(weather)
