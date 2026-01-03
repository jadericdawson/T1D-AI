"""
Tests for IOB (Insulin on Board) and COB (Carbs on Board) calculations.
These calculations are critical for accurate glucose management.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock settings before importing anything that uses it
@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("config.get_settings") as mock:
        settings = MagicMock()
        settings.insulin_action_duration_minutes = 180
        settings.insulin_half_life_minutes = 81.0
        settings.carb_absorption_duration_minutes = 180
        settings.carb_half_life_minutes = 45.0
        settings.carb_bg_factor = 4.0
        settings.target_bg = 100
        mock.return_value = settings
        yield settings


class TestIOBCOBService:
    """Test the IOB/COB service class."""

    def test_service_creation(self, mock_settings):
        """Test service instantiation with default values."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()
        assert service.insulin_half_life_min == 81.0
        assert service.carb_half_life_min == 45.0
        assert service.target_bg == 100

    def test_service_custom_params(self, mock_settings):
        """Test service with custom parameters."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(
            insulin_half_life_min=60.0,
            carb_half_life_min=30.0,
            target_bg=120
        )
        assert service.insulin_half_life_min == 60.0
        assert service.carb_half_life_min == 30.0
        assert service.target_bg == 120


class TestIOBCalculations:
    """Test IOB calculations using the service."""

    def test_iob_empty_treatments(self, mock_settings):
        """Empty treatment list should return 0 IOB."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()
        iob = service.calculate_iob([])
        assert iob == 0.0

    def test_iob_with_mock_treatment(self, mock_settings):
        """Test IOB calculation with mocked treatment."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()

        # Create a mock treatment
        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.insulin = 3.0
        treatment.timestamp = now - timedelta(minutes=30)
        treatment.carbs = None

        iob = service.calculate_iob([treatment])

        # After 30 min with 81 min half-life, ~77% should remain
        # 3.0 * 0.5^(30/81) = 3.0 * 0.77 ≈ 2.31
        assert 2.0 < iob < 2.8, f"Expected IOB ~2.3U, got {iob}"

    def test_iob_at_half_life(self, mock_settings):
        """At exactly one half-life, 50% should remain."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.insulin = 4.0
        treatment.timestamp = now - timedelta(minutes=81)  # exactly one half-life
        treatment.carbs = None

        iob = service.calculate_iob([treatment], at_time=now)

        # Should be exactly 2.0 units (50% of 4.0)
        assert 1.9 < iob < 2.1, f"Expected IOB ~2.0U at half-life, got {iob}"

    def test_iob_old_bolus(self, mock_settings):
        """Old bolus (past action duration) should have zero IOB."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(insulin_duration_min=180)

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.insulin = 5.0
        treatment.timestamp = now - timedelta(hours=4)  # 240 min > 180 duration
        treatment.carbs = None

        iob = service.calculate_iob([treatment], at_time=now)
        assert iob == 0.0, "IOB should be 0 for bolus past action duration"


class TestCOBCalculations:
    """Test COB calculations using the service."""

    def test_cob_empty_treatments(self, mock_settings):
        """Empty treatment list should return 0 COB."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()
        cob = service.calculate_cob([])
        assert cob == 0.0

    def test_cob_with_mock_treatment(self, mock_settings):
        """Test COB calculation with mocked treatment."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.carbs = 45
        treatment.timestamp = now - timedelta(minutes=20)
        treatment.insulin = None

        cob = service.calculate_cob([treatment])

        # After 20 min with 45 min half-life, ~74% should remain
        # 45 * 0.5^(20/45) = 45 * 0.74 ≈ 33
        assert 28 < cob < 38, f"Expected COB ~33g, got {cob}"

    def test_cob_at_half_life(self, mock_settings):
        """At exactly one half-life, 50% should remain."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.carbs = 50
        treatment.timestamp = now - timedelta(minutes=45)  # exactly one half-life
        treatment.insulin = None

        cob = service.calculate_cob([treatment], at_time=now)

        # Should be exactly 25g (50% of 50)
        assert 24 < cob < 26, f"Expected COB ~25g at half-life, got {cob}"


class TestDoseRecommendations:
    """Test dose recommendation calculations."""

    def test_correction_dose_high_bg(self, mock_settings):
        """Calculate correction for high BG."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(target_bg=100)

        dose, effective_bg = service.calculate_dose_recommendation(
            current_bg=200,
            iob=0.0,
            cob=0.0,
            isf=50  # 50 mg/dL per unit
        )

        # (200 - 100) / 50 = 2.0 units needed
        assert abs(dose - 2.0) < 0.1
        assert effective_bg == 200

    def test_correction_with_iob(self, mock_settings):
        """IOB should reduce recommended dose."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(target_bg=100)

        dose, effective_bg = service.calculate_dose_recommendation(
            current_bg=200,
            iob=1.0,  # 1 unit still active -> -50 mg/dL
            cob=0.0,
            isf=50
        )

        # effective_bg = 200 - (1 * 50) = 150
        # dose = (150 - 100) / 50 = 1.0 units
        assert abs(dose - 1.0) < 0.1
        assert effective_bg == 150

    def test_correction_with_cob(self, mock_settings):
        """COB should increase recommended dose."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(target_bg=100, carb_bg_factor=4.0)

        dose, effective_bg = service.calculate_dose_recommendation(
            current_bg=120,
            iob=0.0,
            cob=25,  # 25g carbs -> +100 mg/dL expected
            isf=50
        )

        # effective_bg = 120 + (25 * 4) = 220
        # dose = (220 - 100) / 50 = 2.4 units
        assert abs(dose - 2.4) < 0.2

    def test_correction_negative_capped(self, mock_settings):
        """Negative dose should be capped at 0."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(target_bg=100)

        dose, effective_bg = service.calculate_dose_recommendation(
            current_bg=80,  # Already low
            iob=2.0,  # Plenty of insulin active
            cob=0.0,
            isf=50
        )

        # effective_bg = 80 - (2 * 50) = -20 (below target)
        assert dose == 0.0


class TestBGImpactCalculations:
    """Test BG impact calculations."""

    def test_cob_bg_impact(self, mock_settings):
        """Calculate expected BG rise from COB."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(carb_bg_factor=4.0)

        # 30g carbs * 4 mg/dL per gram = 120 mg/dL
        cob = 30
        impact = cob * service.carb_bg_factor
        assert impact == 120

    def test_iob_bg_impact(self, mock_settings):
        """Calculate expected BG drop from IOB."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService()

        # 2 units * 50 ISF = 100 mg/dL drop
        iob = 2.0
        isf = 50
        impact = iob * isf
        assert impact == 100


class TestCurrentMetricsCalculation:
    """Test comprehensive current metrics calculation."""

    def test_full_metrics_calculation(self, mock_settings):
        """Calculate all current metrics together."""
        from services.iob_cob_service import IOBCOBService

        service = IOBCOBService(target_bg=100, carb_bg_factor=4.0)

        now = datetime.utcnow()

        # Create mock treatments
        insulin_treatment = MagicMock()
        insulin_treatment.insulin = 3.0
        insulin_treatment.timestamp = now - timedelta(minutes=30)
        insulin_treatment.carbs = None

        carb_treatment = MagicMock()
        carb_treatment.carbs = 40
        carb_treatment.timestamp = now - timedelta(minutes=20)
        carb_treatment.insulin = None

        iob = service.calculate_iob([insulin_treatment])
        cob = service.calculate_cob([carb_treatment])

        # Both should have positive values
        assert iob > 0
        assert cob > 0

        # Dose recommendation should account for both
        dose, effective_bg = service.calculate_dose_recommendation(
            current_bg=150,
            iob=iob,
            cob=cob,
            isf=50
        )

        # Should return numeric values
        assert isinstance(dose, float)
        assert isinstance(effective_bg, int)
