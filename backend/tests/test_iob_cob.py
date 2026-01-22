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
        """Test service instantiation with LEARNED default values."""
        from services.iob_cob_service import IOBCOBService, LEARNED_ABSORPTION_PARAMS

        service = IOBCOBService()
        # IOBCOBService uses LEARNED_ABSORPTION_PARAMS which have real-world observed values
        assert service.insulin_half_life_min == LEARNED_ABSORPTION_PARAMS['iob_half_life_min']
        assert service.carb_half_life_min == LEARNED_ABSORPTION_PARAMS['cob_half_life_min']
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
        """Test IOB at roughly one half-life using three-phase model."""
        from services.iob_cob_service import IOBCOBService, LEARNED_ABSORPTION_PARAMS

        service = IOBCOBService()
        half_life = LEARNED_ABSORPTION_PARAMS['iob_half_life_min']  # 81 min
        onset = LEARNED_ABSORPTION_PARAMS['iob_onset_min']  # 15 min
        ramp = LEARNED_ABSORPTION_PARAMS['iob_ramp_min']  # 25 min

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.insulin = 4.0
        # Put treatment at onset + ramp + half_life (when 50% of post-ramp remaining decays)
        # At onset+ramp: 50% remains, at onset+ramp+half_life: 25% remains
        treatment.timestamp = now - timedelta(minutes=onset + ramp + half_life)
        treatment.carbs = None

        iob = service.calculate_iob([treatment], at_time=now)

        # After onset(15) + ramp(25) + half_life(81) = 121 min:
        # At ramp end: 50% (2.0U), after one more half-life: 25% (1.0U)
        assert 0.8 < iob < 1.2, f"Expected IOB ~1.0U at onset+ramp+half-life, got {iob}"

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
        """Test COB calculation with mocked treatment using three-phase model."""
        from services.iob_cob_service import IOBCOBService, gi_to_absorption_params

        service = IOBCOBService()

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.carbs = 45
        treatment.timestamp = now - timedelta(minutes=20)
        treatment.insulin = None
        treatment.glycemicIndex = 55  # Medium GI
        treatment.isLiquid = False
        treatment.protein = None

        cob = service.calculate_cob([treatment])

        # Three-phase model with GI-based absorption:
        # For GI=55: onset~10min, ramp~15min (combined ~25min to reach ramp end)
        # At 20 min (in ramp phase): about 65% still remains due to ramp decay
        # 45g * 0.65 ≈ 29g
        # COB should be positive and decreasing as carbs are absorbed
        assert 25 < cob < 35, f"Expected COB ~29g at 20min (ramp phase), got {cob}"

    def test_cob_at_half_life(self, mock_settings):
        """Test COB at a point where significant absorption has occurred."""
        from services.iob_cob_service import IOBCOBService, gi_to_absorption_params

        service = IOBCOBService()

        # For GI=55: onset~10min, ramp~15min, half_life~40min, duration~180min
        # At onset+ramp+half_life = 10+15+40 = 65 min:
        # At ramp end: 50% remains, after one more half-life: 25% remains
        gi_params = gi_to_absorption_params(55, False)
        onset = gi_params['onset_min']
        ramp = gi_params['ramp_min']
        half_life = gi_params['half_life_min']

        now = datetime.utcnow()
        treatment = MagicMock()
        treatment.carbs = 50
        treatment.timestamp = now - timedelta(minutes=onset + ramp + half_life)
        treatment.insulin = None
        treatment.glycemicIndex = 55
        treatment.isLiquid = False
        treatment.protein = None

        cob = service.calculate_cob([treatment], at_time=now)

        # At ramp end: 50% (25g), after one more half-life: 25% (12.5g)
        assert 10 < cob < 16, f"Expected COB ~12.5g at onset+ramp+half-life, got {cob}"


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

        # Create mock treatments with all required attributes
        insulin_treatment = MagicMock()
        insulin_treatment.insulin = 3.0
        insulin_treatment.timestamp = now - timedelta(minutes=30)
        insulin_treatment.carbs = None
        insulin_treatment.protein = None
        insulin_treatment.glycemicIndex = None
        insulin_treatment.isLiquid = False

        carb_treatment = MagicMock()
        carb_treatment.carbs = 40
        carb_treatment.timestamp = now - timedelta(minutes=20)
        carb_treatment.insulin = None
        carb_treatment.protein = None
        carb_treatment.glycemicIndex = 55  # Medium GI
        carb_treatment.isLiquid = False

        iob = service.calculate_iob([insulin_treatment])
        cob = service.calculate_cob([carb_treatment])

        # Both should have positive values
        assert iob > 0, f"IOB should be positive, got {iob}"
        assert cob > 0, f"COB should be positive, got {cob}"

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
