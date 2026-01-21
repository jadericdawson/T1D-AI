"""
Absorption Curve Data Collector

Collects high-resolution data needed to learn personalized IOB/COB/POB activity curves.

Key differences from ml_data_collector.py:
1. Stores FULL BG timeseries (every 5 min for 4 hours)
2. Stores theoretical IOB/COB/POB curves at each time point
3. Calculates BG velocity (derivative) for activity detection
4. Captures what current curves predict vs actual response

This data enables training models to learn:
- When absorption actually starts (onset)
- How fast it ramps up (activity shape)
- When it peaks
- How fast it decays
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from models.schemas import Treatment, GlucoseReading
from database.repositories import GlucoseRepository, TreatmentRepository
from services.iob_cob_service import IOBCOBService, insulin_activity_curve, carb_activity_curve, gi_to_absorption_params

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionCurveDataPoint:
    """Single time point in absorption curve learning dataset"""
    minutesSinceTreatment: int
    actualBG: float
    bgSlope: float  # mg/dL per minute
    theoreticalIOB: float
    theoreticalCOB: float
    theoreticalPOB: float
    insulinActivity: float  # 0-1 from current curve
    carbActivity: float     # 0-1 from current curve
    proteinActivity: float  # 0-1 from current curve
    predictedBG: float  # What current model predicts


@dataclass
class AbsorptionLearningDataset:
    """Complete dataset for learning one treatment's absorption curve"""
    treatmentId: str
    userId: str
    treatmentTime: datetime

    # Treatment details
    insulinDose: Optional[float]
    carbs: Optional[float]
    protein: Optional[float]
    fat: Optional[float]
    glycemicIndex: int

    # High-resolution timeseries (every 5 min for 4 hours = 48 points)
    dataPoints: List[AbsorptionCurveDataPoint]

    # Quality flags
    isClean: bool  # No overlapping treatments
    hasFullCoverage: bool  # No CGM gaps
    preTreatmentStable: bool  # BG was stable before

    # Derived labels (ground truth - calculated from actual BG response)
    actualOnsetMin: Optional[float] = None  # When BG first moved
    actualPeakMin: Optional[float] = None   # When BG change was fastest
    actualHalfLifeMin: Optional[float] = None  # When effect dropped to 50%


class AbsorptionCurveDataCollector:
    """
    Collects high-resolution data for learning personalized absorption curves.

    This runs after a treatment and collects:
    1. Complete BG timeseries (every 5 min)
    2. Theoretical IOB/COB/POB at each time
    3. BG velocity (slope) at each time
    4. What current curves predict

    Usage:
        collector = AbsorptionCurveDataCollector(user_id)
        dataset = await collector.collect_after_treatment(treatment_id)
        # Store dataset for ML training
    """

    def __init__(
        self,
        user_id: str,
        glucose_repo: Optional[GlucoseRepository] = None,
        treatment_repo: Optional[TreatmentRepository] = None,
        iob_cob_service: Optional[IOBCOBService] = None
    ):
        self.user_id = user_id
        self.glucose_repo = glucose_repo or GlucoseRepository()
        self.treatment_repo = treatment_repo or TreatmentRepository()
        self.iob_cob_service = iob_cob_service or IOBCOBService()

    async def collect_after_treatment(
        self,
        treatment_id: str,
        isf: float = 50.0,  # Typical ISF in mg/dL per unit
        icr: float = 10.0,
        wait_hours: int = 4
    ) -> Optional[AbsorptionLearningDataset]:
        """
        Collect complete absorption curve data after a treatment.

        Call this 4 hours after a treatment to get the full response curve.

        Args:
            treatment_id: Treatment to analyze
            isf: Insulin sensitivity factor
            icr: Insulin to carb ratio
            wait_hours: How long to wait for full absorption (default 4)

        Returns:
            Complete dataset ready for ML training, or None if data is invalid
        """
        # Get treatment (requires user_id for partition key)
        treatment = await self.treatment_repo.get_by_id(treatment_id, self.user_id)
        if not treatment:
            logger.warning(f"Treatment {treatment_id} not found for user {self.user_id}")
            return None

        treatment_time = treatment.timestamp.replace(tzinfo=None)

        # Check if enough time has passed
        now = datetime.utcnow()
        if (now - treatment_time).total_seconds() / 3600 < wait_hours:
            logger.info(f"Treatment {treatment_id} not old enough yet, need {wait_hours} hours")
            return None

        # Get CGM data for 4 hours after treatment
        end_time = treatment_time + timedelta(hours=wait_hours)
        bg_readings = await self.glucose_repo.get_history(
            self.user_id,
            start_time=treatment_time,
            end_time=end_time,
            limit=500
        )

        if not bg_readings or len(bg_readings) < 10:
            logger.warning(f"Insufficient BG data for {treatment_id}")
            return None

        # Check for clean window (no overlapping treatments)
        is_clean = await self._check_clean_window(treatment_time, end_time)

        # Check for stable pre-treatment BG
        pre_stable = await self._check_pre_treatment_stability(treatment_time)

        # Check CGM coverage
        has_full_coverage = self._check_cgm_coverage(bg_readings, wait_hours)

        # Build data points every 5 minutes
        data_points = []
        bg_per_gram = isf / icr

        # Get all treatments for theoretical IOB/COB calculation
        all_treatments = await self.treatment_repo.get_by_user(
            self.user_id,
            start_time=treatment_time - timedelta(hours=5),  # Look back 5 hours for IOB/COB context
            end_time=end_time
        )

        for i, reading in enumerate(bg_readings):
            time_since_treatment = (reading.timestamp.replace(tzinfo=None) - treatment_time).total_seconds() / 60

            if time_since_treatment < 0:
                continue  # Skip pre-treatment readings

            # Calculate BG slope (velocity)
            bg_slope = self._calculate_bg_slope(bg_readings, i)

            # Calculate theoretical IOB/COB/POB at this time
            reading_time = reading.timestamp.replace(tzinfo=None)
            theoretical_iob = self.iob_cob_service.calculate_iob(all_treatments, at_time=reading_time)
            theoretical_cob = self.iob_cob_service.calculate_cob(all_treatments, at_time=reading_time)
            theoretical_pob = self.iob_cob_service.calculate_pob(all_treatments, at_time=reading_time)

            # Calculate activity levels from current curves
            insulin_activity = 0.0
            carb_activity = 0.0
            protein_activity = 0.0

            if treatment.insulin and treatment.insulin > 0:
                insulin_activity = insulin_activity_curve(
                    time_since_treatment,
                    peak_min=75,
                    dia_min=self.iob_cob_service.insulin_duration_min
                )

            if treatment.carbs and treatment.carbs > 0:
                gi = getattr(treatment, 'glycemicIndex', None) or 55
                is_liquid = getattr(treatment, 'isLiquid', False)
                gi_params = gi_to_absorption_params(gi, is_liquid)
                carb_activity = carb_activity_curve(
                    time_since_treatment,
                    peak_min=45,
                    duration_min=gi_params['duration_min'],
                    glycemic_index=gi,
                    is_liquid=is_liquid
                )

            if treatment.protein and treatment.protein > 0:
                protein_activity = carb_activity_curve(
                    time_since_treatment,
                    peak_min=180,  # Protein peaks much later
                    duration_min=300,
                    glycemic_index=30,
                    is_liquid=False
                )

            # Calculate what current model predicts at this time
            predicted_bg = self._calculate_predicted_bg(
                treatment,
                time_since_treatment,
                isf,
                icr,
                bg_readings[0].value if bg_readings else 120
            )

            data_point = AbsorptionCurveDataPoint(
                minutesSinceTreatment=int(time_since_treatment),
                actualBG=reading.value,
                bgSlope=bg_slope,
                theoreticalIOB=theoretical_iob,
                theoreticalCOB=theoretical_cob,
                theoreticalPOB=theoretical_pob,
                insulinActivity=insulin_activity,
                carbActivity=carb_activity,
                proteinActivity=protein_activity,
                predictedBG=predicted_bg
            )
            data_points.append(data_point)

        # Detect ground truth labels from actual BG response
        actual_onset = self._detect_onset(data_points)
        actual_peak = self._detect_peak_activity(data_points)
        actual_half_life = self._detect_half_life(data_points, actual_peak)

        dataset = AbsorptionLearningDataset(
            treatmentId=treatment_id,
            userId=self.user_id,
            treatmentTime=treatment_time,
            insulinDose=treatment.insulin,
            carbs=treatment.carbs,
            protein=treatment.protein or 0,
            fat=getattr(treatment, 'fat', 0) or 0,
            glycemicIndex=getattr(treatment, 'glycemicIndex', 55) or 55,
            dataPoints=data_points,
            isClean=is_clean,
            hasFullCoverage=has_full_coverage,
            preTreatmentStable=pre_stable,
            actualOnsetMin=actual_onset,
            actualPeakMin=actual_peak,
            actualHalfLifeMin=actual_half_life
        )

        logger.info(f"Collected absorption curve data for {treatment_id}: "
                   f"{len(data_points)} points, clean={is_clean}, "
                   f"onset={actual_onset:.1f}min, peak={actual_peak:.1f}min")

        return dataset

    def _calculate_bg_slope(self, readings: List[GlucoseReading], index: int) -> float:
        """Calculate BG slope (mg/dL per minute) using central difference"""
        if index == 0 or index >= len(readings) - 1:
            return 0.0

        prev_reading = readings[index - 1]
        next_reading = readings[index + 1]

        time_diff = (next_reading.timestamp - prev_reading.timestamp).total_seconds() / 60
        if time_diff == 0:
            return 0.0

        bg_diff = next_reading.value - prev_reading.value
        return bg_diff / time_diff

    async def _check_clean_window(self, start_time: datetime, end_time: datetime) -> bool:
        """Check if no other treatments occurred in window"""
        treatments = await self.treatment_repo.get_by_user(
            self.user_id,
            start_time=start_time,
            end_time=end_time
        )
        return len(treatments) <= 1  # Only the treatment itself

    async def _check_pre_treatment_stability(self, treatment_time: datetime) -> bool:
        """Check if BG was stable for 20 min before treatment"""
        start = treatment_time - timedelta(minutes=20)
        readings = await self.glucose_repo.get_history(
            self.user_id,
            start_time=start,
            end_time=treatment_time,
            limit=10
        )

        if len(readings) < 3:
            return False

        values = [r.value for r in readings]
        bg_range = max(values) - min(values)
        return bg_range <= 20  # Within 20 mg/dL = stable

    def _check_cgm_coverage(self, readings: List[GlucoseReading], hours: int) -> bool:
        """Check if we have good CGM coverage (>90% of expected readings)"""
        expected_readings = hours * 12  # 12 readings per hour at 5-min intervals
        actual_readings = len(readings)
        coverage = actual_readings / expected_readings
        return coverage >= 0.9

    def _calculate_predicted_bg(
        self,
        treatment: Treatment,
        time_since_min: float,
        isf: float,
        icr: float,
        baseline_bg: float
    ) -> float:
        """Calculate what current model predicts BG will be at this time.

        Uses proper absorption fraction calculation based on activity curve integration.
        Activity curve represents instantaneous absorption RATE, so we need to integrate
        to get total absorbed fraction.
        """
        import math
        bg_change = 0.0

        if treatment.insulin and treatment.insulin > 0:
            # Insulin effect using exponential absorption model
            # This approximates the integral of the activity curve
            dia_min = 240.0
            # Use 1 - exp(-k*t) model where half the insulin is absorbed by ~75 min
            k = math.log(2) / 75.0  # Half-life constant
            absorbed_fraction = 1.0 - math.exp(-k * time_since_min)
            # Cap at 95% for numerical stability
            absorbed_fraction = min(absorbed_fraction, 0.95)
            bg_change -= absorbed_fraction * treatment.insulin * isf

        if treatment.carbs and treatment.carbs > 0:
            # Carb effect using exponential absorption model
            gi = getattr(treatment, 'glycemicIndex', 55) or 55
            gi_params = gi_to_absorption_params(gi, False)
            # Faster carbs = shorter half-life
            carb_half_life = gi_params.get('duration_min', 180) / 4.0  # ~45 min for standard
            k = math.log(2) / carb_half_life
            absorbed_fraction = 1.0 - math.exp(-k * time_since_min)
            absorbed_fraction = min(absorbed_fraction, 0.95)
            bg_change += absorbed_fraction * treatment.carbs * (isf / icr)

        return baseline_bg + bg_change

    def _detect_onset(self, data_points: List[AbsorptionCurveDataPoint]) -> Optional[float]:
        """Detect when absorption actually started (BG first moved).

        Uses a moving average to filter out CGM noise (~2-3 mg/dL jitter)
        and requires sustained movement in the same direction.
        """
        if len(data_points) < 5:
            return None

        # Use 3-point moving average to smooth noise
        smoothed_slopes = []
        for i in range(1, len(data_points) - 1):
            avg_slope = (
                data_points[i-1].bgSlope +
                data_points[i].bgSlope +
                data_points[i+1].bgSlope
            ) / 3.0
            smoothed_slopes.append((data_points[i].minutesSinceTreatment, avg_slope))

        # Threshold: 0.2 mg/dL per minute = 12 mg/dL per hour (clear movement)
        threshold = 0.2

        # Look for 3 consecutive readings above threshold in same direction
        for i in range(len(smoothed_slopes) - 2):
            slopes = [smoothed_slopes[i][1], smoothed_slopes[i+1][1], smoothed_slopes[i+2][1]]

            # All same sign (all rising or all falling)
            if all(s > threshold for s in slopes) or all(s < -threshold for s in slopes):
                return float(smoothed_slopes[i][0])

        return None

    def _detect_peak_activity(self, data_points: List[AbsorptionCurveDataPoint]) -> Optional[float]:
        """Detect when absorption peaked (maximum |BG slope|)"""
        if len(data_points) < 3:
            return None

        max_slope = 0.0
        peak_time = 0.0

        for point in data_points:
            if abs(point.bgSlope) > max_slope:
                max_slope = abs(point.bgSlope)
                peak_time = point.minutesSinceTreatment

        return float(peak_time) if max_slope > 0 else None

    def _detect_half_life(
        self,
        data_points: List[AbsorptionCurveDataPoint],
        peak_time: Optional[float]
    ) -> Optional[float]:
        """Detect half-life (when BG effect recovered to 50% of peak change).

        Half-life measures when the REMAINING effect drops to 50%, not when
        the slope drops to 50%. This is calculated from the BG values, not slopes.

        For carbs: When BG has come down 50% from peak rise
        For insulin: When BG has risen 50% from maximum drop

        Returns minutes from peak to half-recovery.
        """
        if not peak_time or len(data_points) < 5:
            return None

        # Get baseline BG (first reading)
        baseline_bg = data_points[0].actualBG

        # Find peak BG (maximum deviation from baseline)
        peak_bg = baseline_bg
        peak_idx = 0
        max_deviation = 0.0

        for i, point in enumerate(data_points):
            deviation = abs(point.actualBG - baseline_bg)
            if deviation > max_deviation:
                max_deviation = deviation
                peak_bg = point.actualBG
                peak_idx = i

        if max_deviation < 10:  # Less than 10 mg/dL change = no clear effect
            return None

        # Calculate the halfway point
        # If BG rose: target = baseline + 50% of rise
        # If BG fell: target = baseline - 50% of fall
        half_effect_bg = baseline_bg + (peak_bg - baseline_bg) / 2.0

        # Find when BG crossed halfway back to baseline (after peak)
        peak_minutes = data_points[peak_idx].minutesSinceTreatment

        for point in data_points:
            if point.minutesSinceTreatment > peak_minutes:
                # Check if we've crossed halfway back
                if peak_bg > baseline_bg:  # BG rose (carbs)
                    if point.actualBG <= half_effect_bg:
                        return float(point.minutesSinceTreatment - peak_minutes)
                else:  # BG fell (insulin)
                    if point.actualBG >= half_effect_bg:
                        return float(point.minutesSinceTreatment - peak_minutes)

        return None
