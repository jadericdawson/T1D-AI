"""
COB Forcing Function Model

Predicts remaining COB at a given horizon, accounting for food composition.
The caller computes BG pressure by multiplying absorbed carbs by ISF/ICR.

Food composition affects absorption rate:
- High GI (>70): Fast absorption (juice, candy) - 30 min half-life
- Medium GI (55-70): Normal absorption (bread, rice) - 45 min half-life
- Low GI (<55): Slow absorption (beans, whole grains) - 60 min half-life
- High fat: Extends duration significantly (pizza: 75+ min half-life)
- Protein: Secondary rise effect at 2-4 hours

Key insight: COB is always a POSITIVE pressure on BG (carbs raise BG).
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Default half-lives by food type (minutes)
FAST_HALF_LIFE_MIN = 30.0    # High GI, low fat (juice, candy)
MEDIUM_HALF_LIFE_MIN = 45.0  # Medium GI (bread, rice)
SLOW_HALF_LIFE_MIN = 60.0    # Low GI (beans, whole grains)
HIGH_FAT_HALF_LIFE_MIN = 90.0  # High fat meals (pizza, pasta with cream)

# Absorption state adjustment factors for carb kinetics
# These modify the half-life based on detected absorption patterns
ABSORPTION_STATE_FACTORS = {
    "very_slow": 1.50,      # Gastroparesis/very slow digestion - 50% slower
    "slow": 1.25,           # Slow digestion - 25% slower
    "normal": 1.0,          # Baseline
    "fast": 0.80,           # Fast absorption - 20% faster
    "very_fast": 0.65,      # Very fast (e.g., glucose tabs) - 35% faster
}


class COBForcingModel(nn.Module):
    """
    Neural network that predicts remaining COB at a future horizon.

    Key innovation: Accounts for food composition from GPT-4.1 enrichment.

    Input features (10):
    - horizon_normalized: horizon_min / 360
    - current_cob_normalized: current_cob / 100
    - glycemic_index_normalized: GI / 100
    - fat_normalized: fat_grams / 50
    - protein_normalized: protein_grams / 50
    - hour_sin, hour_cos: circadian rhythm
    - is_recent_meal: within 30 min of eating
    - is_stacking: multiple meals overlapping
    - estimated_half_life: from food composition

    Output:
    - remaining_fraction: (0-1) multiply by current_cob to get remaining COB
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Food composition branch - learns how composition affects absorption
        self.food_branch = nn.Sequential(
            nn.Linear(5, hidden_size),  # cob, gi, fat, protein, half_life
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Time + context branch
        self.time_branch = nn.Sequential(
            nn.Linear(5, hidden_size),  # horizon, hour_sin, hour_cos, recent, stacking
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Output: remaining fraction 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 10) with features

        Returns:
            remaining_fraction: Tensor of shape (batch, 1) in range [0, 1]
        """
        # Split features
        food_features = x[:, 1:6]  # cob, gi, fat, protein, half_life
        time_features = torch.cat([x[:, :1], x[:, 6:]], dim=1)  # horizon + context

        # Process branches
        food_out = self.food_branch(food_features)
        time_out = self.time_branch(time_features)

        # Combine and predict
        combined = torch.cat([food_out, time_out], dim=1)
        remaining_fraction = self.combine(combined)

        return remaining_fraction


class COBForcingService:
    """
    Service for predicting COB-induced BG pressure.

    Uses food composition from GPT-4.1 enrichment to adjust absorption curves.
    Falls back to GI-based formula when model unavailable.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        default_half_life_min: float = MEDIUM_HALF_LIFE_MIN,
    ):
        self.default_half_life_min = default_half_life_min
        self.model: Optional[COBForcingModel] = None
        self._use_ml = False

        # Try to load trained model
        # Path: src/ml/models/cob_forcing.py -> ../../../.. = backend
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "cob_forcing.pth"

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                self.model = COBForcingModel()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self._use_ml = True
                logger.info("Loaded COB forcing model")
            except Exception as e:
                logger.warning(f"Failed to load COB forcing model: {e}")
        else:
            logger.info(f"No COB forcing model at {model_path}, using formula")

    def estimate_half_life(
        self,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        absorption_state: Optional[str] = None,
    ) -> float:
        """
        Estimate carb absorption half-life based on food composition and absorption state.

        High GI + low fat = fast absorption
        Low GI + high fat = slow absorption (pizza effect)
        Absorption state further modifies based on detected patterns

        Args:
            glycemic_index: GI value (0-100), default 55 (medium)
            fat_grams: Fat content in grams
            protein_grams: Protein content in grams
            absorption_state: Current absorption state (very_slow, slow, normal, fast, very_fast)
                            Adjusts carb kinetics based on detected absorption patterns

        Returns:
            half_life_min: Estimated absorption half-life in minutes
        """
        # Base half-life from GI
        if glycemic_index >= 70:
            base_half_life = FAST_HALF_LIFE_MIN
        elif glycemic_index >= 55:
            base_half_life = MEDIUM_HALF_LIFE_MIN
        else:
            base_half_life = SLOW_HALF_LIFE_MIN

        # Fat significantly extends absorption (gastroparesis effect)
        # Each 10g of fat adds ~15 min to half-life
        fat_extension = (fat_grams / 10.0) * 15.0

        # Protein has minor effect (~5 min per 10g)
        protein_extension = (protein_grams / 10.0) * 5.0

        # Total half-life (capped at reasonable maximum)
        total_half_life = base_half_life + fat_extension + protein_extension

        # Apply absorption state adjustment
        if absorption_state and absorption_state in ABSORPTION_STATE_FACTORS:
            adjustment_factor = ABSORPTION_STATE_FACTORS[absorption_state]
            total_half_life = total_half_life * adjustment_factor
            if absorption_state != "normal":
                logger.debug(f"COB half-life adjusted for {absorption_state}: base -> {total_half_life:.1f} min")

        return min(total_half_life, 240.0)  # Cap at 4 hours (higher for slow absorption)

    def predict_remaining(
        self,
        current_cob: float,
        horizon_min: int,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        is_recent_meal: bool = False,
        is_stacking: bool = False,
        absorption_state: Optional[str] = None,
    ) -> float:
        """
        Predict remaining COB at a future horizon.

        Args:
            current_cob: Current COB in grams
            horizon_min: Minutes into future (5-120)
            glycemic_index: GI of the food (0-100)
            fat_grams: Fat content
            protein_grams: Protein content
            hour: Current hour (0-23)
            is_recent_meal: True if within 30 min of eating
            is_stacking: True if multiple meals overlapping
            absorption_state: Current absorption state (affects carb kinetics)

        Returns:
            remaining_cob: COB that will remain at horizon (grams)
        """
        if current_cob <= 0:
            return 0.0

        if horizon_min <= 0:
            return current_cob

        if horizon_min > 480:  # 8 hours max for very slow meals
            return 0.0

        if hour is None:
            hour = datetime.utcnow().hour

        # Estimate half-life from food composition and absorption state
        half_life = self.estimate_half_life(glycemic_index, fat_grams, protein_grams, absorption_state)

        # Try ML model
        if self._use_ml and self.model is not None:
            try:
                features = self._prepare_features(
                    current_cob, horizon_min, glycemic_index,
                    fat_grams, protein_grams, hour,
                    is_recent_meal, is_stacking, half_life
                )
                with torch.no_grad():
                    remaining_fraction = self.model(features).item()

                # Apply absorption state adjustment to ML prediction as well
                if absorption_state and absorption_state in ABSORPTION_STATE_FACTORS and absorption_state != "normal":
                    factor = ABSORPTION_STATE_FACTORS[absorption_state]
                    # Scale the decay: if factor > 1 (slower), more remains; if factor < 1 (faster), less remains
                    adjusted_fraction = remaining_fraction ** (1.0 / factor)
                    remaining_fraction = max(0.0, min(1.0, adjusted_fraction))

                remaining_cob = current_cob * max(0.0, min(1.0, remaining_fraction))
                return round(remaining_cob, 1)
            except Exception as e:
                logger.warning(f"COB ML prediction failed: {e}")

        # Fallback: exponential decay with absorption-state-adjusted half-life
        remaining_fraction = 0.5 ** (horizon_min / half_life)
        return round(current_cob * remaining_fraction, 1)

    def predict_absorbed(
        self,
        current_cob: float,
        horizon_min: int,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        is_recent_meal: bool = False,
        is_stacking: bool = False,
        absorption_state: Optional[str] = None,
    ) -> float:
        """
        Predict how many carbs will be absorbed by horizon.

        Args:
            current_cob: Current COB in grams
            horizon_min: Minutes into future
            glycemic_index: GI value
            fat_grams: Fat content
            protein_grams: Protein content
            hour: Current hour
            is_recent_meal: Recent meal flag
            is_stacking: Stacking flag
            absorption_state: Current absorption state (affects carb kinetics)

        Returns:
            absorbed_carbs: Grams of carbs absorbed (always positive)
        """
        remaining = self.predict_remaining(
            current_cob, horizon_min, glycemic_index,
            fat_grams, protein_grams, hour,
            is_recent_meal, is_stacking, absorption_state
        )
        return round(current_cob - remaining, 1)

    def get_bg_pressure(
        self,
        current_cob: float,
        horizon_min: int,
        isf: float,
        icr: float,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        hour: Optional[int] = None,
        is_recent_meal: bool = False,
        is_stacking: bool = False,
        absorption_state: Optional[str] = None,
    ) -> float:
        """
        Calculate the BG-raising pressure from COB at horizon.

        Args:
            current_cob: Current COB in grams
            horizon_min: Minutes into future
            isf: Insulin sensitivity factor (mg/dL per unit)
            icr: Insulin-to-carb ratio (grams per unit)
            glycemic_index: GI value
            fat_grams: Fat content
            protein_grams: Protein content
            hour: Current hour
            is_recent_meal: Recent meal flag
            is_stacking: Stacking flag
            absorption_state: Current absorption state (affects carb kinetics)

        Returns:
            bg_pressure: Expected BG change (POSITIVE - carbs raise BG)
        """
        absorbed = self.predict_absorbed(
            current_cob, horizon_min, glycemic_index,
            fat_grams, protein_grams, hour,
            is_recent_meal, is_stacking, absorption_state
        )

        # BG pressure = absorbed carbs * (ISF / ICR)
        # Positive because carbs RAISE BG
        bg_per_gram = isf / icr
        bg_pressure = absorbed * bg_per_gram

        return round(bg_pressure, 1)

    def _prepare_features(
        self,
        current_cob: float,
        horizon_min: int,
        glycemic_index: float,
        fat_grams: float,
        protein_grams: float,
        hour: int,
        is_recent_meal: bool,
        is_stacking: bool,
        half_life: float,
    ) -> torch.Tensor:
        """Prepare feature tensor for model input."""
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        features = torch.tensor([[
            horizon_min / 360.0,
            current_cob / 100.0,
            glycemic_index / 100.0,
            fat_grams / 50.0,
            protein_grams / 50.0,
            half_life / 100.0,
            hour_sin,
            hour_cos,
            1.0 if is_recent_meal else 0.0,
            1.0 if is_stacking else 0.0,
        ]], dtype=torch.float32)

        return features

    def get_absorption_curve(
        self,
        initial_carbs: float = 30.0,
        glycemic_index: float = 55,
        fat_grams: float = 0,
        protein_grams: float = 0,
        duration_min: int = 300,
        step_min: int = 15,
        hour: int = 12,
    ) -> list:
        """
        Generate COB absorption curve for visualization.

        Returns list of {minutes, remaining_cob, absorbed, remaining_fraction}
        """
        curve = []
        half_life = self.estimate_half_life(glycemic_index, fat_grams, protein_grams)

        for t in range(0, duration_min + 1, step_min):
            remaining = self.predict_remaining(
                initial_carbs, t, glycemic_index,
                fat_grams, protein_grams, hour
            )
            absorbed = initial_carbs - remaining

            curve.append({
                'minutes': t,
                'remaining_cob': round(remaining, 1),
                'absorbed_carbs': round(absorbed, 1),
                'remaining_fraction': round(remaining / initial_carbs, 3) if initial_carbs > 0 else 0,
                'half_life_min': round(half_life, 0),
            })

        return curve


# Singleton instance
_cob_forcing_service: Optional[COBForcingService] = None


def get_cob_forcing_service(model_path: Optional[Path] = None) -> COBForcingService:
    """Get or create the COB forcing service singleton."""
    global _cob_forcing_service
    if _cob_forcing_service is None:
        _cob_forcing_service = COBForcingService(model_path)
    return _cob_forcing_service
