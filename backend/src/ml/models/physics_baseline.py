"""
Physics Baseline Calculator

Combines IOB and COB forcing functions with ISF to compute deterministic
BG predictions. This is the "physics layer" - the neural residual model
only learns adjustments on top of this baseline.

Core formula:
    predicted_bg = current_bg + iob_pressure + cob_pressure

Where:
    iob_pressure = -absorbed_insulin * isf * isf_adjustment  (NEGATIVE)
    cob_pressure = absorbed_carbs * (isf / icr)              (POSITIVE)

ISF adjustments by time of day (dawn phenomenon, etc.):
    - 4-8 AM:   +15-25%  (dawn phenomenon, cortisol)
    - 8-11 AM:  +5-10%   (residual morning hormones)
    - 11 AM-5 PM: baseline
    - 5-10 PM:  -5-10%   (evening sensitivity)
    - 10 PM-4 AM: -10-15% (night sensitivity)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ml.models.iob_forcing import IOBForcingService, get_iob_forcing_service
from ml.models.cob_forcing import COBForcingService, get_cob_forcing_service

logger = logging.getLogger(__name__)


@dataclass
class PhysicsPrediction:
    """Result of physics baseline prediction."""
    horizon_min: int
    predicted_bg: float
    iob_pressure: float  # Negative (insulin lowers BG)
    cob_pressure: float  # Positive (carbs raise BG)
    net_pressure: float  # Sum of IOB + COB pressures
    absorbed_insulin: float
    absorbed_carbs: float
    isf_adjustment: float
    confidence: float


class PhysicsBaseline:
    """
    Computes deterministic BG predictions from IOB/COB forcing functions.

    This is the "physics layer" that respects known diabetes physiology:
    - IOB always applies negative pressure (lowers BG)
    - COB always applies positive pressure (raises BG)
    - ISF varies by time of day

    The neural residual model learns adjustments to these predictions
    for factors the physics doesn't capture (activity, stress, etc.).
    """

    # ISF time-of-day adjustment factors
    ISF_ADJUSTMENTS = {
        # Hour range: (start_hour, end_hour, adjustment_factor)
        'dawn': (4, 8, 1.20),      # Dawn phenomenon: +20% ISF needed
        'morning': (8, 11, 1.08),   # Residual hormones: +8%
        'midday': (11, 17, 1.00),   # Baseline
        'evening': (17, 22, 0.95),  # More sensitive: -5%
        'night': (22, 4, 0.88),     # Most sensitive: -12%
    }

    def __init__(
        self,
        iob_service: Optional[IOBForcingService] = None,
        cob_service: Optional[COBForcingService] = None,
        default_isf: float = 55.0,
        default_icr: float = 10.0,
    ):
        """
        Initialize physics baseline calculator.

        Args:
            iob_service: Service for IOB forcing predictions
            cob_service: Service for COB forcing predictions
            default_isf: Default insulin sensitivity factor
            default_icr: Default insulin-to-carb ratio
        """
        self.iob_service = iob_service or get_iob_forcing_service()
        self.cob_service = cob_service or get_cob_forcing_service()
        self.default_isf = default_isf
        self.default_icr = default_icr

    def get_isf_adjustment(self, hour: int) -> float:
        """
        Get ISF adjustment factor for time of day.

        Args:
            hour: Hour of day (0-23)

        Returns:
            adjustment: Multiplier for ISF (>1 = more insulin needed)
        """
        # Handle night crossing midnight
        if hour >= 22 or hour < 4:
            return self.ISF_ADJUSTMENTS['night'][2]
        elif 4 <= hour < 8:
            return self.ISF_ADJUSTMENTS['dawn'][2]
        elif 8 <= hour < 11:
            return self.ISF_ADJUSTMENTS['morning'][2]
        elif 11 <= hour < 17:
            return self.ISF_ADJUSTMENTS['midday'][2]
        else:  # 17-22
            return self.ISF_ADJUSTMENTS['evening'][2]

    def is_dawn_window(self, hour: int) -> bool:
        """Check if current hour is in dawn phenomenon window."""
        return 4 <= hour < 8

    def predict(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        horizon_min: int,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None,
    ) -> PhysicsPrediction:
        """
        Compute physics-based BG prediction at a single horizon.

        Args:
            current_bg: Current blood glucose (mg/dL)
            iob: Current insulin on board (units)
            cob: Current carbs on board (grams)
            horizon_min: Prediction horizon (minutes)
            isf: Insulin sensitivity factor (mg/dL per unit)
            icr: Insulin-to-carb ratio (grams per unit)
            glycemic_index: GI of food (affects COB absorption)
            fat_grams: Fat content (slows COB absorption)
            protein_grams: Protein content
            hour: Current hour for ISF adjustment
            metabolic_state: Current metabolic state (sick, resistant, normal, sensitive, very_sensitive)
                            Adjusts insulin kinetics
            absorption_state: Current absorption state (very_slow, slow, normal, fast, very_fast)
                            Adjusts carb absorption kinetics

        Returns:
            PhysicsPrediction with all calculation details
        """
        if hour is None:
            hour = datetime.utcnow().hour

        if isf is None:
            isf = self.default_isf

        if icr is None:
            icr = self.default_icr

        # Get ISF adjustment for time of day
        isf_adj = self.get_isf_adjustment(hour)
        is_dawn = self.is_dawn_window(hour)

        # Calculate IOB forcing (negative pressure)
        # Metabolic state affects insulin kinetics (sick/resistant = slower, sensitive = faster)
        absorbed_insulin = self.iob_service.predict_absorbed(
            current_iob=iob,
            horizon_min=horizon_min,
            hour=hour,
            is_dawn_window=is_dawn,
            metabolic_state=metabolic_state,
        )
        iob_pressure = -absorbed_insulin * isf * isf_adj

        # Calculate COB forcing (positive pressure)
        # Absorption state affects carb kinetics (slow = delayed rise, fast = quicker rise)
        absorbed_carbs = self.cob_service.predict_absorbed(
            current_cob=cob,
            horizon_min=horizon_min,
            glycemic_index=glycemic_index,
            fat_grams=fat_grams,
            protein_grams=protein_grams,
            hour=hour,
            absorption_state=absorption_state,
        )
        bg_per_gram = isf / icr
        cob_pressure = absorbed_carbs * bg_per_gram

        # Net pressure (they counteract!)
        net_pressure = iob_pressure + cob_pressure

        # Predicted BG
        predicted_bg = current_bg + net_pressure

        # Clamp to physiological range
        predicted_bg = max(40, min(400, predicted_bg))

        # Confidence decreases with horizon and metabolic activity
        base_confidence = 0.85
        horizon_penalty = horizon_min * 0.002  # 0.2% per minute
        activity_penalty = (abs(iob_pressure) + abs(cob_pressure)) * 0.001
        # Also reduce confidence when metabolic state is abnormal (less predictable)
        state_penalty = 0.05 if metabolic_state and metabolic_state not in (None, "normal") else 0.0
        confidence = max(0.5, base_confidence - horizon_penalty - activity_penalty - state_penalty)

        return PhysicsPrediction(
            horizon_min=horizon_min,
            predicted_bg=round(predicted_bg, 1),
            iob_pressure=round(iob_pressure, 1),
            cob_pressure=round(cob_pressure, 1),
            net_pressure=round(net_pressure, 1),
            absorbed_insulin=round(absorbed_insulin, 2),
            absorbed_carbs=round(absorbed_carbs, 1),
            isf_adjustment=round(isf_adj, 2),
            confidence=round(confidence, 2),
        )

    def predict_multi_horizon(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        horizons: List[int],
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None,
    ) -> Dict[int, PhysicsPrediction]:
        """
        Compute physics predictions at multiple horizons.

        Args:
            current_bg: Current blood glucose
            iob: Current IOB
            cob: Current COB
            horizons: List of horizon minutes [5, 10, 15, 30, 45, 60, 90, 120]
            isf: Insulin sensitivity factor
            icr: Insulin-to-carb ratio
            glycemic_index: Food GI
            fat_grams: Fat content
            protein_grams: Protein content
            hour: Current hour
            metabolic_state: Current metabolic state (affects insulin kinetics)
            absorption_state: Current absorption state (affects carb kinetics)

        Returns:
            Dict mapping horizon_min to PhysicsPrediction
        """
        predictions = {}
        for h in horizons:
            predictions[h] = self.predict(
                current_bg=current_bg,
                iob=iob,
                cob=cob,
                horizon_min=h,
                isf=isf,
                icr=icr,
                glycemic_index=glycemic_index,
                fat_grams=fat_grams,
                protein_grams=protein_grams,
                hour=hour,
                metabolic_state=metabolic_state,
                absorption_state=absorption_state,
            )
        return predictions

    def compute_trajectory(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        duration_min: int = 180,
        step_min: int = 5,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None,
    ) -> List[Dict]:
        """
        Compute full BG trajectory for visualization.

        Returns list of points with all forcing function details.
        """
        trajectory = []
        horizons = list(range(0, duration_min + 1, step_min))

        for h in horizons:
            if h == 0:
                # Current point
                trajectory.append({
                    'minutes': 0,
                    'predicted_bg': current_bg,
                    'iob_pressure': 0,
                    'cob_pressure': 0,
                    'net_pressure': 0,
                    'remaining_iob': iob,
                    'remaining_cob': cob,
                })
            else:
                pred = self.predict(
                    current_bg=current_bg,
                    iob=iob,
                    cob=cob,
                    horizon_min=h,
                    isf=isf,
                    icr=icr,
                    glycemic_index=glycemic_index,
                    fat_grams=fat_grams,
                    protein_grams=protein_grams,
                    hour=hour,
                    metabolic_state=metabolic_state,
                    absorption_state=absorption_state,
                )

                remaining_iob = iob - pred.absorbed_insulin
                remaining_cob = cob - pred.absorbed_carbs

                trajectory.append({
                    'minutes': h,
                    'predicted_bg': pred.predicted_bg,
                    'iob_pressure': pred.iob_pressure,
                    'cob_pressure': pred.cob_pressure,
                    'net_pressure': pred.net_pressure,
                    'remaining_iob': round(remaining_iob, 2),
                    'remaining_cob': round(remaining_cob, 1),
                })

        return trajectory

    def compute_uncertainty(
        self,
        horizon_min: int,
        iob_pressure: float,
        cob_pressure: float,
    ) -> Tuple[float, float]:
        """
        Compute prediction uncertainty bounds.

        Uncertainty grows with:
        - Time horizon
        - Metabolic activity (high IOB/COB = more variability)

        Returns:
            (lower_uncertainty, upper_uncertainty) in mg/dL
        """
        # Base uncertainty grows with horizon
        base_uncertainty = 8 + (horizon_min * 0.25)

        # Metabolic uncertainty from active IOB/COB
        metabolic_factor = (abs(iob_pressure) + abs(cob_pressure)) * 0.08
        total_uncertainty = base_uncertainty + metabolic_factor

        # Asymmetric: slightly more uncertainty on the upside
        lower = total_uncertainty * 0.95
        upper = total_uncertainty * 1.05

        return round(lower, 1), round(upper, 1)


# Singleton instance
_physics_baseline: Optional[PhysicsBaseline] = None


def get_physics_baseline() -> PhysicsBaseline:
    """Get or create the physics baseline singleton."""
    global _physics_baseline
    if _physics_baseline is None:
        _physics_baseline = PhysicsBaseline()
    return _physics_baseline
