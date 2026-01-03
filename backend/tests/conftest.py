"""
Pytest configuration and fixtures for T1D-AI tests.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    with patch("config.get_settings") as mock:
        settings = MagicMock()
        settings.cosmos_endpoint = "https://test.documents.azure.com:443/"
        settings.cosmos_key = "test_key"
        settings.cosmos_database = "test_db"
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_openai_key = "test_key"
        settings.azure_openai_deployment = "gpt-4.1-test"
        settings.azure_openai_api_version = "2024-12-01-preview"
        settings.target_bg = 100
        settings.high_bg_threshold = 180
        settings.low_bg_threshold = 70
        settings.critical_high_threshold = 250
        settings.critical_low_threshold = 54
        settings.insulin_half_life_minutes = 81.0
        settings.carb_half_life_minutes = 45.0
        settings.carb_bg_factor = 4.0
        mock.return_value = settings
        yield settings


@pytest.fixture
def sample_glucose_readings():
    """Generate sample glucose readings for testing."""
    now = datetime.utcnow()
    return [
        {"value": 120, "timestamp": now - timedelta(minutes=5), "trend": "Flat"},
        {"value": 125, "timestamp": now - timedelta(minutes=10), "trend": "FortyFiveUp"},
        {"value": 118, "timestamp": now - timedelta(minutes=15), "trend": "Flat"},
        {"value": 115, "timestamp": now - timedelta(minutes=20), "trend": "FortyFiveDown"},
        {"value": 110, "timestamp": now - timedelta(minutes=25), "trend": "Flat"},
        {"value": 105, "timestamp": now - timedelta(minutes=30), "trend": "FortyFiveDown"},
        {"value": 100, "timestamp": now - timedelta(minutes=35), "trend": "Flat"},
        {"value": 98, "timestamp": now - timedelta(minutes=40), "trend": "Flat"},
        {"value": 95, "timestamp": now - timedelta(minutes=45), "trend": "FortyFiveDown"},
        {"value": 92, "timestamp": now - timedelta(minutes=50), "trend": "Flat"},
    ]


@pytest.fixture
def sample_treatments():
    """Generate sample treatments for testing."""
    now = datetime.utcnow()
    return [
        {"type": "insulin", "insulin": 3.0, "timestamp": now - timedelta(hours=1)},
        {"type": "carbs", "carbs": 45, "timestamp": now - timedelta(hours=2)},
        {"type": "insulin", "insulin": 2.5, "timestamp": now - timedelta(hours=3)},
        {"type": "carbs", "carbs": 30, "timestamp": now - timedelta(hours=4)},
    ]


@pytest.fixture
def sample_user_settings():
    """Generate sample user settings for testing."""
    return {
        "targetBg": 100,
        "highThreshold": 180,
        "lowThreshold": 70,
        "criticalHighThreshold": 250,
        "criticalLowThreshold": 54,
        "insulinSensitivity": 50,
        "carbRatio": 10,
    }


@pytest.fixture
def mock_cosmos_client():
    """Mock CosmosDB client."""
    with patch("azure.cosmos.CosmosClient") as mock:
        client = MagicMock()
        database = MagicMock()
        container = MagicMock()

        client.get_database_client.return_value = database
        database.get_container_client.return_value = container

        mock.return_value = client
        yield container


@pytest.fixture
def mock_openai_client():
    """Mock Azure OpenAI client."""
    with patch("openai.AsyncAzureOpenAI") as mock:
        client = AsyncMock()

        # Mock chat completions
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = '{"insights": [{"content": "Test insight", "category": "recommendation"}]}'

        client.chat.completions.create = AsyncMock(return_value=completion)

        mock.return_value = client
        yield client
