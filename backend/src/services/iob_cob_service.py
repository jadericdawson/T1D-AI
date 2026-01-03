"""
IOB/COB Calculation Service for T1D-AI
Ported from dexcom_reader_predict_v2.3.py lines 468-606

Implements:
- Insulin on Board (IOB) using exponential decay with configurable half-life
- Carbs on Board (COB) using exponential decay
- Dose recommendation with IOB/COB adjustments
"""
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from models.schemas import Treatment, GlucoseReading, CurrentMetrics
from config import get_settings

logger = logging.getLogger(__name__)


class IOBCOBService:
    """Service for calculating Insulin on Board and Carbs on Board."""

    def __init__(
        self,
        insulin_duration_min: int = 180,
        insulin_half_life_min: float = 81.0,
        carb_duration_min: int = 180,
        carb_half_life_min: float = 45.0,
        carb_bg_factor: float = 4.0,
        target_bg: int = 100
    ):
        """
        Initialize IOB/COB service with configurable parameters.

        Args:
            insulin_duration_min: Total duration of insulin action (default: 180 min for Novolog)
            insulin_half_life_min: Half-life for insulin decay (default: 81 min for Novolog)
            carb_duration_min: Total duration of carb absorption (default: 180 min)
            carb_half_life_min: Half-life for carb absorption (default: 45 min)
            carb_bg_factor: BG rise per gram of carbs (default: 4.0 mg/dL)
            target_bg: Target blood glucose level (default: 100 mg/dL)
        """
        self.insulin_duration_min = insulin_duration_min
        self.insulin_half_life_min = insulin_half_life_min
        self.carb_duration_min = carb_duration_min
        self.carb_half_life_min = carb_half_life_min
        self.carb_bg_factor = carb_bg_factor
        self.target_bg = target_bg

    @classmethod
    def from_settings(cls) -> "IOBCOBService":
        """Create service instance from application settings."""
        settings = get_settings()
        return cls(
            insulin_duration_min=settings.insulin_action_duration_minutes,
            insulin_half_life_min=settings.insulin_half_life_minutes,
            carb_duration_min=settings.carb_absorption_duration_minutes,
            carb_half_life_min=settings.carb_half_life_minutes,
            carb_bg_factor=settings.carb_bg_factor,
            target_bg=settings.target_bg
        )

    def calculate_iob(
        self,
        treatments: List[Treatment],
        at_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate Insulin on Board at a specific time.

        Uses exponential decay model:
        IOB = sum(bolus * 0.5^(time_elapsed / half_life))

        Args:
            treatments: List of insulin treatments
            at_time: Time to calculate IOB for (default: now)

        Returns:
            Total IOB in units
        """
        if not treatments:
            return 0.0

        at_time = at_time or datetime.utcnow()
        total_iob = 0.0

        # Filter to insulin treatments only
        insulin_treatments = [t for t in treatments if t.insulin and t.insulin > 0]

        for treatment in insulin_treatments:
            # Calculate time elapsed since bolus
            time_elapsed = (at_time - treatment.timestamp.replace(tzinfo=None)).total_seconds() / 60

            # Only count if within duration and after the bolus
            if 0 <= time_elapsed <= self.insulin_duration_min:
                # Exponential decay: remaining = initial * 0.5^(t/half_life)
                decay_factor = 0.5 ** (time_elapsed / self.insulin_half_life_min)
                total_iob += treatment.insulin * decay_factor

        return round(total_iob, 2)

    def calculate_cob(
        self,
        treatments: List[Treatment],
        at_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate Carbs on Board at a specific time.

        Uses exponential decay model similar to IOB.

        Args:
            treatments: List of carb treatments
            at_time: Time to calculate COB for (default: now)

        Returns:
            Total COB in grams
        """
        if not treatments:
            return 0.0

        at_time = at_time or datetime.utcnow()
        total_cob = 0.0

        # Filter to carb treatments only
        carb_treatments = [t for t in treatments if t.carbs and t.carbs > 0]

        for treatment in carb_treatments:
            # Calculate time elapsed since eating
            time_elapsed = (at_time - treatment.timestamp.replace(tzinfo=None)).total_seconds() / 60

            # Only count if within duration and after eating
            if 0 <= time_elapsed <= self.carb_duration_min:
                # Exponential decay for remaining carbs
                decay_factor = 0.5 ** (time_elapsed / self.carb_half_life_min)
                total_cob += treatment.carbs * decay_factor

        return round(total_cob, 1)

    def calculate_dose_recommendation(
        self,
        current_bg: int,
        iob: float,
        cob: float,
        isf: float
    ) -> Tuple[float, int]:
        """
        Calculate recommended correction dose.

        Formula from dexcom_reader_predict_v2.3.py:1049-1055:
        effective_bg = current_bg + (cob * carb_bg_factor) - (iob * isf)
        dose = (effective_bg - target_bg) / isf

        Args:
            current_bg: Current blood glucose in mg/dL
            iob: Current Insulin on Board
            cob: Current Carbs on Board
            isf: Insulin Sensitivity Factor (mg/dL per unit)

        Returns:
            Tuple of (recommended_dose, effective_bg)
        """
        if isf <= 0:
            logger.warning("ISF must be positive for dose calculation")
            return 0.0, current_bg

        # Calculate effective BG accounting for IOB and COB
        # COB will raise BG (positive contribution)
        # IOB will lower BG (negative contribution)
        cob_effect = cob * self.carb_bg_factor  # Expected BG rise from remaining carbs
        iob_effect = iob * isf  # Expected BG drop from remaining insulin

        effective_bg = current_bg + cob_effect - iob_effect
        effective_bg = int(round(effective_bg))

        # Calculate correction dose
        correction_needed = effective_bg - self.target_bg

        if correction_needed <= 0:
            # BG is at or below target, no correction needed
            return 0.0, effective_bg

        dose = correction_needed / isf
        dose = round(max(0, dose), 2)

        logger.debug(
            f"Dose calc: BG={current_bg}, COB={cob}g (+{cob_effect:.0f}), "
            f"IOB={iob:.2f}U (-{iob_effect:.0f}), EffBG={effective_bg}, Dose={dose}U"
        )

        return dose, effective_bg

    def get_current_metrics(
        self,
        current_bg: int,
        treatments: List[Treatment],
        isf: float
    ) -> CurrentMetrics:
        """
        Calculate all current metrics for display.

        Args:
            current_bg: Current blood glucose
            treatments: Recent treatments for IOB/COB calculation
            isf: Predicted ISF value

        Returns:
            CurrentMetrics with all calculated values
        """
        iob = self.calculate_iob(treatments)
        cob = self.calculate_cob(treatments)
        dose, effective_bg = self.calculate_dose_recommendation(current_bg, iob, cob, isf)

        return CurrentMetrics(
            iob=iob,
            cob=cob,
            isf=isf,
            recommendedDose=dose,
            effectiveBg=effective_bg
        )


# Helper functions for simpler usage

def calculate_iob_simple(
    insulin_treatments: List[Treatment],
    duration_min: int = 180,
    half_life_min: float = 81.0
) -> float:
    """Simple IOB calculation function."""
    service = IOBCOBService(insulin_duration_min=duration_min, insulin_half_life_min=half_life_min)
    return service.calculate_iob(insulin_treatments)


def calculate_cob_simple(
    carb_treatments: List[Treatment],
    duration_min: int = 180,
    half_life_min: float = 45.0
) -> float:
    """Simple COB calculation function."""
    service = IOBCOBService(carb_duration_min=duration_min, carb_half_life_min=half_life_min)
    return service.calculate_cob(carb_treatments)
