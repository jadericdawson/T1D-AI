"""
Forcing Function Ensemble

Main prediction model combining IOB/COB forcing functions with neural residual.

Architecture:
1. IOB Forcing → Remaining IOB → multiply by ISF → negative BG pressure
2. COB Forcing → Remaining COB → multiply by ISF/ICR → positive BG pressure
3. Physics Baseline → current_bg + iob_pressure + cob_pressure
4. Residual Model → secondary factor adjustments (+/- 25 mg/dL max)
5. Final Prediction → physics_baseline + residual

Why this works better than pure ML (per arXiv:2502.00065v1):
- Pure ML on CGM-only achieved RMSE ~22.5 mg/dL at 30 min
- They noted: "BG influenced by insulin dosage and meal consumption"
- Our approach uses IOB/COB as forcing functions = we have the causal factors!
- Expected improvement: 15-25% better predictions

Benchmarks to beat (from research):
- 30 min: MAE < 16 mg/dL, RMSE < 22 mg/dL
- Hypoglycemia: Most difficult (RMSE ~28-35 in pure ML)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from ml.models.iob_forcing import IOBForcingService, get_iob_forcing_service
from ml.models.cob_forcing import COBForcingService, get_cob_forcing_service
from ml.models.physics_baseline import PhysicsBaseline, get_physics_baseline, PhysicsPrediction
from ml.models.residual_tft import ResidualService, get_residual_service, SecondaryFeatures

logger = logging.getLogger(__name__)

# Default prediction horizons
DEFAULT_HORIZONS = [5, 10, 15, 30, 45, 60, 90, 120]


@dataclass
class ForcingPrediction:
    """Complete prediction with all components exposed."""
    horizon_min: int
    final_prediction: float
    physics_baseline: float
    residual_adjustment: float
    iob_pressure: float
    cob_pressure: float
    net_pressure: float
    absorbed_insulin: float
    absorbed_carbs: float
    lower_bound: float
    upper_bound: float
    confidence: float


class ForcingFunctionEnsemble:
    """
    Main prediction model combining physics and neural components.

    Core formula:
        predicted_bg = current_bg + iob_pressure + cob_pressure + residual

    Where:
        iob_pressure = -absorbed_insulin * isf * isf_adj  (NEGATIVE)
        cob_pressure = absorbed_carbs * (isf / icr)       (POSITIVE)
        residual = neural adjustment for secondary factors (+/- 25 max)

    Key insight: IOB/COB are independent forcing functions that counteract!
    """

    def __init__(
        self,
        iob_service: Optional[IOBForcingService] = None,
        cob_service: Optional[COBForcingService] = None,
        physics_baseline: Optional[PhysicsBaseline] = None,
        residual_service: Optional[ResidualService] = None,
        default_isf: float = 55.0,
        default_icr: float = 10.0,
    ):
        """
        Initialize the forcing function ensemble.

        Args:
            iob_service: Service for IOB predictions
            cob_service: Service for COB predictions
            physics_baseline: Physics baseline calculator
            residual_service: Residual adjustment service
            default_isf: Default insulin sensitivity factor
            default_icr: Default insulin-to-carb ratio
        """
        self.iob_service = iob_service or get_iob_forcing_service()
        self.cob_service = cob_service or get_cob_forcing_service()
        self.physics = physics_baseline or get_physics_baseline()
        self.residual_service = residual_service or get_residual_service()

        self.default_isf = default_isf
        self.default_icr = default_icr

        logger.info("Initialized ForcingFunctionEnsemble")

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
        recent_trend: float = 0,
        bg_volatility: float = 0,
        dawn_intensity: float = 0.5,
        hour: Optional[int] = None,
        metabolic_state: Optional[str] = None,
        absorption_state: Optional[str] = None,
    ) -> ForcingPrediction:
        """
        Generate prediction at a single horizon.

        Args:
            current_bg: Current blood glucose (mg/dL)
            iob: Current insulin on board (units)
            cob: Current carbs on board (grams)
            horizon_min: Prediction horizon (minutes)
            isf: Insulin sensitivity factor
            icr: Insulin-to-carb ratio
            glycemic_index: Food GI (affects COB absorption)
            fat_grams: Fat content (slows absorption)
            protein_grams: Protein content
            recent_trend: BG change per 5 min (minor factor)
            bg_volatility: Std dev of recent BG
            dawn_intensity: Dawn phenomenon intensity (0-1)
            hour: Current hour
            metabolic_state: Current metabolic state (sick, resistant, normal, sensitive, very_sensitive)
                            Adjusts insulin kinetics - sick/resistant = slower action
            absorption_state: Current absorption state (very_slow, slow, normal, fast, very_fast)
                            Adjusts carb kinetics - slow = delayed BG rise

        Returns:
            ForcingPrediction with all components
        """
        if hour is None:
            hour = datetime.utcnow().hour

        if isf is None:
            isf = self.default_isf

        if icr is None:
            icr = self.default_icr

        # Step 1: Get physics baseline (IOB/COB forcing functions)
        # Pass metabolic and absorption state for kinetics adjustments
        physics_pred = self.physics.predict(
            current_bg=current_bg,
            iob=iob,
            cob=cob,
            horizon_min=horizon_min,
            isf=isf,
            icr=icr,
            glycemic_index=glycemic_index,
            fat_grams=fat_grams,
            protein_grams=protein_grams,
            hour=hour,
            metabolic_state=metabolic_state,
            absorption_state=absorption_state,
        )

        # Step 2: Extract secondary features for residual model
        secondary_features = self.residual_service.extract_features(
            dt=datetime.utcnow(),
            recent_trend=recent_trend,
            bg_volatility=bg_volatility,
            dawn_intensity=dawn_intensity,
            horizon_min=horizon_min,
            physics_pred=physics_pred.predicted_bg,
        )

        # Step 3: Get residual adjustment (clamped +/- 25 mg/dL)
        residual = self.residual_service.predict(
            features=secondary_features,
            horizon_min=horizon_min,
            physics_pred=physics_pred.predicted_bg,
        )

        # Step 4: Final prediction
        final_prediction = physics_pred.predicted_bg + residual
        final_prediction = max(40, min(400, final_prediction))

        # Step 5: Compute uncertainty bounds
        lower_unc, upper_unc = self.physics.compute_uncertainty(
            horizon_min=horizon_min,
            iob_pressure=physics_pred.iob_pressure,
            cob_pressure=physics_pred.cob_pressure,
        )
        lower_bound = max(40, final_prediction - lower_unc)
        upper_bound = min(400, final_prediction + upper_unc)

        return ForcingPrediction(
            horizon_min=horizon_min,
            final_prediction=round(final_prediction, 1),
            physics_baseline=physics_pred.predicted_bg,
            residual_adjustment=round(residual, 1),
            iob_pressure=physics_pred.iob_pressure,
            cob_pressure=physics_pred.cob_pressure,
            net_pressure=physics_pred.net_pressure,
            absorbed_insulin=physics_pred.absorbed_insulin,
            absorbed_carbs=physics_pred.absorbed_carbs,
            lower_bound=round(lower_bound, 1),
            upper_bound=round(upper_bound, 1),
            confidence=physics_pred.confidence,
        )

    def predict_multi_horizon(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        horizons: Optional[List[int]] = None,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        recent_trend: float = 0,
        bg_volatility: float = 0,
        dawn_intensity: float = 0.5,
        hour: Optional[int] = None,
    ) -> Dict[int, ForcingPrediction]:
        """
        Generate predictions at multiple horizons.

        Args:
            current_bg: Current BG
            iob: Current IOB
            cob: Current COB
            horizons: List of horizons [5, 10, 15, 30, 45, 60, 90, 120]
            isf, icr: Sensitivity factors
            glycemic_index, fat_grams, protein_grams: Food composition
            recent_trend, bg_volatility, dawn_intensity: Secondary factors
            hour: Current hour

        Returns:
            Dict mapping horizon_min to ForcingPrediction
        """
        if horizons is None:
            horizons = DEFAULT_HORIZONS

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
                recent_trend=recent_trend,
                bg_volatility=bg_volatility,
                dawn_intensity=dawn_intensity,
                hour=hour,
            )

        return predictions

    def explain_prediction(
        self,
        prediction: ForcingPrediction,
    ) -> str:
        """
        Generate human-readable explanation of prediction.

        Useful for debugging and user understanding.
        """
        direction = "falling" if prediction.net_pressure < 0 else "rising" if prediction.net_pressure > 0 else "stable"

        explanation = f"""
BG Prediction at +{prediction.horizon_min} min: {prediction.final_prediction} mg/dL

Physics breakdown:
  - IOB effect: {prediction.iob_pressure:+.1f} mg/dL ({prediction.absorbed_insulin:.2f}U absorbed)
  - COB effect: {prediction.cob_pressure:+.1f} mg/dL ({prediction.absorbed_carbs:.1f}g absorbed)
  - Net pressure: {prediction.net_pressure:+.1f} mg/dL (BG {direction})

Adjustments:
  - Residual (secondary factors): {prediction.residual_adjustment:+.1f} mg/dL

Uncertainty:
  - Range: {prediction.lower_bound} - {prediction.upper_bound} mg/dL
  - Confidence: {prediction.confidence:.0%}
"""
        return explanation.strip()

    def get_forcing_summary(
        self,
        current_bg: float,
        iob: float,
        cob: float,
        isf: float,
        icr: float,
    ) -> Dict:
        """
        Get summary of current forcing functions.

        Useful for dashboard display.
        """
        # Calculate effects at key horizons
        effects = {}
        for h in [30, 60, 120]:
            pred = self.predict(current_bg, iob, cob, h, isf, icr)
            effects[h] = {
                'iob_pressure': pred.iob_pressure,
                'cob_pressure': pred.cob_pressure,
                'net_pressure': pred.net_pressure,
            }

        # Determine dominant force
        net_30 = effects[30]['net_pressure']
        if abs(effects[30]['iob_pressure']) > abs(effects[30]['cob_pressure']) * 1.5:
            dominant = 'IOB (insulin lowering BG)'
        elif abs(effects[30]['cob_pressure']) > abs(effects[30]['iob_pressure']) * 1.5:
            dominant = 'COB (carbs raising BG)'
        else:
            dominant = 'Balanced (IOB and COB roughly equal)'

        return {
            'current_bg': current_bg,
            'iob': iob,
            'cob': cob,
            'dominant_force': dominant,
            'net_direction': 'down' if net_30 < -5 else 'up' if net_30 > 5 else 'stable',
            'effects_by_horizon': effects,
        }


# Singleton instance
_forcing_ensemble: Optional[ForcingFunctionEnsemble] = None


def get_forcing_ensemble() -> ForcingFunctionEnsemble:
    """Get or create the forcing function ensemble singleton."""
    global _forcing_ensemble
    if _forcing_ensemble is None:
        _forcing_ensemble = ForcingFunctionEnsemble()
    return _forcing_ensemble
