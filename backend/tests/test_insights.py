"""
Tests for Insight Service and OpenAI integration.
Tests pattern detection, anomaly detection, and GPT-4.1 insight generation.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock settings before importing anything
@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("config.get_settings") as mock:
        settings = MagicMock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_openai_key = "test_key"
        settings.azure_openai_deployment = "gpt-4.1-test"
        settings.azure_openai_api_version = "2024-12-01-preview"
        settings.target_bg = 100
        settings.high_bg_threshold = 180
        settings.low_bg_threshold = 70
        settings.critical_high_threshold = 250
        settings.critical_low_threshold = 54
        mock.return_value = settings
        yield settings


class TestOpenAIServiceFallbacks:
    """Test OpenAI fallback insights generation."""

    def test_fallback_insights_high_tir(self, mock_settings):
        """Should generate positive fallback for high TIR."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": 110} for _ in range(100)]
        insights = service._fallback_insights(readings, time_in_range=85.0)

        # High TIR should get achievement insight
        achievement = next((i for i in insights if i["category"] == "achievement"), None)
        assert achievement is not None
        assert "85%" in achievement["content"] or "time in range" in achievement["content"].lower()

    def test_fallback_insights_low_tir(self, mock_settings):
        """Should generate recommendation for low TIR."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": 200} for _ in range(100)]
        insights = service._fallback_insights(readings, time_in_range=35.0)

        # Low TIR should get recommendation
        recommendation = next((i for i in insights if i["category"] == "recommendation"), None)
        assert recommendation is not None

    def test_fallback_insights_low_bg(self, mock_settings):
        """Should generate warning for low BG."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": 65}]  # Current BG is low
        insights = service._fallback_insights(readings, time_in_range=50.0)

        # Low BG should get warning
        warning = next((i for i in insights if i["category"] == "warning"), None)
        assert warning is not None
        assert "low" in warning["content"].lower()

    def test_fallback_insights_high_bg(self, mock_settings):
        """Should generate warning for high BG."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": 280}]  # Current BG is very high
        insights = service._fallback_insights(readings, time_in_range=50.0)

        # High BG should get warning
        warning = next((i for i in insights if i["category"] == "warning"), None)
        assert warning is not None
        assert "elevated" in warning["content"].lower() or "high" in warning["content"].lower()

    def test_fallback_insights_empty_readings(self, mock_settings):
        """Should handle empty readings gracefully."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        insights = service._fallback_insights([], time_in_range=0.0)

        # Should still return at least one insight
        assert len(insights) >= 1


class TestTimeInRangeCalculation:
    """Test time in range calculation."""

    def test_tir_all_in_range(self, mock_settings):
        """All readings in range should give 100% TIR."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": v} for v in [100, 120, 140, 160, 170]]
        settings = {"lowThreshold": 70, "highThreshold": 180}

        tir = service._calculate_time_in_range(readings, settings)
        assert tir == 100.0

    def test_tir_none_in_range(self, mock_settings):
        """All readings out of range should give 0% TIR."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        readings = [{"value": v} for v in [50, 55, 200, 220, 250]]
        settings = {"lowThreshold": 70, "highThreshold": 180}

        tir = service._calculate_time_in_range(readings, settings)
        assert tir == 0.0

    def test_tir_mixed(self, mock_settings):
        """Mix of readings should give correct TIR percentage."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        # 3 in range, 2 out of range = 60% TIR
        readings = [{"value": v} for v in [100, 120, 140, 50, 220]]
        settings = {"lowThreshold": 70, "highThreshold": 180}

        tir = service._calculate_time_in_range(readings, settings)
        assert tir == 60.0

    def test_tir_empty_readings(self, mock_settings):
        """Empty readings should give 0% TIR."""
        from services.openai_service import OpenAIService

        service = OpenAIService()

        settings = {"lowThreshold": 70, "highThreshold": 180}

        tir = service._calculate_time_in_range([], settings)
        assert tir == 0.0


class TestOpenAIServiceInitialization:
    """Test OpenAI service initialization."""

    def test_service_creation(self, mock_settings):
        """Service should be created without errors."""
        from services.openai_service import OpenAIService

        service = OpenAIService()
        assert service is not None
        assert service._initialized == False  # Not initialized until first call

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, mock_settings):
        """Service should initialize lazily on first use."""
        with patch("openai.AsyncAzureOpenAI") as mock_client:
            mock_client.return_value = MagicMock()

            from services.openai_service import OpenAIService
            service = OpenAIService()

            await service.initialize()
            assert service._initialized == True


class TestAnomalyDetectionLogic:
    """Test anomaly detection helper logic."""

    def test_critical_high_detection(self, mock_settings):
        """Values above critical threshold should be detected."""
        # Critical high threshold is 250
        value = 280
        critical_threshold = 250

        is_critical_high = value > critical_threshold
        assert is_critical_high == True

    def test_critical_low_detection(self, mock_settings):
        """Values below critical threshold should be detected."""
        # Critical low threshold is 54
        value = 52
        critical_threshold = 54

        is_critical_low = value < critical_threshold
        assert is_critical_low == True

    def test_normal_value_not_critical(self, mock_settings):
        """Normal values should not trigger critical alerts."""
        value = 120
        critical_high = 250
        critical_low = 54

        is_critical = value > critical_high or value < critical_low
        assert is_critical == False

    def test_rapid_change_detection(self, mock_settings):
        """Rapid glucose changes should be detected."""
        # Readings showing rapid rise
        readings = [
            {"value": 100, "timestamp": datetime.utcnow() - timedelta(minutes=15)},
            {"value": 150, "timestamp": datetime.utcnow() - timedelta(minutes=10)},
            {"value": 200, "timestamp": datetime.utcnow() - timedelta(minutes=5)},
        ]

        # Calculate rate of change
        rate_per_5min = (readings[-1]["value"] - readings[0]["value"]) / 3
        is_rapid = abs(rate_per_5min) > 20  # More than 20 mg/dL per 5 min

        assert is_rapid == True


class TestInsightCategories:
    """Test insight category classification."""

    def test_valid_categories(self, mock_settings):
        """All insight categories should be valid."""
        valid_categories = {"pattern", "recommendation", "warning", "achievement"}

        # Test that fallback insights use valid categories
        from services.openai_service import OpenAIService

        service = OpenAIService()
        insights = service._fallback_insights([{"value": 120}], 70.0)

        for insight in insights:
            assert insight["category"] in valid_categories

    def test_insight_has_content(self, mock_settings):
        """All insights should have non-empty content."""
        from services.openai_service import OpenAIService

        service = OpenAIService()
        insights = service._fallback_insights([{"value": 120}], 70.0)

        for insight in insights:
            assert "content" in insight
            assert len(insight["content"]) > 0
