"""
Strictly Causal TFT for Blood Glucose Prediction

This model is designed to be CAUSALLY CORRECT:
- BG rises are caused by CARBS (COB)
- BG drops are caused by INSULIN (IOB)
- Dawn phenomenon (4-8 AM) causes rises without carbs
- Neural network can ONLY modulate these effects, not create independent ones

The key constraint: WITHOUT active IOB/COB/dawn, predictions should be FLAT.
The neural network learns HOW MUCH the causal factors affect BG, not WHETHER
arbitrary patterns exist in the data.

Author: T1D-AI
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Optional
from dataclasses import dataclass

# Physics constants
INSULIN_HALF_LIFE_MIN = 81.0  # Fast-acting insulin half-life
DEFAULT_ISF = 55.0  # mg/dL per unit
DEFAULT_ICR = 10.0  # grams per unit

# Prediction horizons
PREDICTION_HORIZONS = [15, 30, 60, 90, 120, 180]


@dataclass
class CausalPrediction:
    """Breakdown of causal contributions."""
    horizon_min: int
    value: float
    iob_effect: float      # Insulin's contribution (negative)
    cob_effect: float      # Carbs' contribution (positive)
    dawn_effect: float     # Dawn phenomenon (positive, morning only)
    trend_effect: float    # Short-term momentum
    confidence: float


class DawnPhenomenonModule(nn.Module):
    """
    Models the dawn phenomenon: early morning BG rise without carbs.

    This is caused by:
    - Cortisol release (peaks around wake time)
    - Growth hormone
    - Natural circadian insulin resistance

    Typically affects 4-8 AM, peaks around 6-7 AM.
    Can cause 20-50 mg/dL rise over 2-3 hours.
    """

    def __init__(self, max_effect: float = 40.0):
        super().__init__()
        self.max_effect = max_effect

        # Learnable dawn window parameters
        # Default: peaks at 6 AM, affects 4-8 AM
        self.peak_hour = nn.Parameter(torch.tensor(6.0))
        self.width = nn.Parameter(torch.tensor(2.0))  # Hours on each side
        self.intensity = nn.Parameter(torch.tensor(0.5))  # 0-1 scale

    def forward(
        self,
        hour: torch.Tensor,           # (batch,) 0-23.99
        horizon_min: int,             # Prediction horizon
        current_bg: torch.Tensor,     # (batch,) Current BG affects dawn intensity
    ) -> torch.Tensor:
        """
        Calculate dawn phenomenon effect.

        Returns positive value (mg/dL rise) during dawn window, 0 otherwise.
        """
        # Calculate future hour
        future_hour = hour + horizon_min / 60.0
        future_hour = future_hour % 24  # Wrap around midnight

        # Gaussian-like dawn window centered on peak_hour
        peak = torch.sigmoid(self.peak_hour - 2) * 8 + 2  # Constrain to 2-10 AM
        width = F.softplus(self.width) + 0.5  # Min 0.5 hours

        # Distance from peak (handle wrap-around)
        dist = torch.abs(future_hour - peak)
        dist = torch.min(dist, 24 - dist)  # Handle midnight wrap

        # Dawn window activation (1 at peak, 0 outside)
        dawn_activation = torch.exp(-(dist ** 2) / (2 * width ** 2))

        # Scale by learned intensity and max effect
        intensity = torch.sigmoid(self.intensity)

        # Higher BG reduces dawn effect (body compensates)
        bg_factor = torch.clamp(1.0 - (current_bg - 100) / 200, 0.3, 1.0)

        # Effect accumulates with horizon (dawn is a gradual rise)
        time_factor = min(1.0, horizon_min / 60.0)  # Full effect at 60 min

        dawn_effect = dawn_activation * intensity * self.max_effect * bg_factor * time_factor

        return dawn_effect


class CausalPhysicsModule(nn.Module):
    """
    Computes strictly causal BG predictions.

    BG change = IOB effect + COB effect + Dawn effect + minimal drift

    The neural network can ONLY adjust:
    - ISF (insulin sensitivity factor) based on time/context
    - ICR (carb ratio) based on meal composition
    - Carb absorption rate based on fat/protein
    - Dawn phenomenon intensity

    It CANNOT create arbitrary BG changes.
    """

    def __init__(self, default_isf: float = DEFAULT_ISF, default_icr: float = DEFAULT_ICR):
        super().__init__()
        self.default_isf = default_isf
        self.default_icr = default_icr

        # Dawn phenomenon
        self.dawn = DawnPhenomenonModule(max_effect=40.0)

        # ISF time-of-day adjustment (captures circadian insulin sensitivity)
        self.isf_tod_adjustment = nn.Sequential(
            nn.Linear(2, 16),  # sin/cos of hour
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )
        self.isf_scale = 0.25  # Max ±25% ISF adjustment

    def forward(
        self,
        current_bg: torch.Tensor,      # (batch,)
        iob: torch.Tensor,             # (batch,) Active insulin
        cob: torch.Tensor,             # (batch,) Active carbs
        trend: torch.Tensor,           # (batch,) Rate per 5 min
        hour: torch.Tensor,            # (batch,) 0-23.99
        carb_half_life: torch.Tensor,  # (batch,) Carb absorption speed
        isf_base: torch.Tensor,        # (batch,) Base ISF
        icr_base: torch.Tensor,        # (batch,) Base ICR
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal predictions for all horizons.
        """
        batch_size = current_bg.shape[0]
        device = current_bg.device

        # Time-of-day ISF adjustment
        hour_features = torch.stack([
            torch.sin(2 * math.pi * hour / 24),
            torch.cos(2 * math.pi * hour / 24),
        ], dim=-1)
        isf_adj = self.isf_tod_adjustment(hour_features).squeeze(-1)
        adjusted_isf = isf_base * (1 + isf_adj * self.isf_scale)

        results = {}

        for horizon in PREDICTION_HORIZONS:
            # ===== IOB EFFECT (CAUSAL: insulin → BG drop) =====
            iob_remaining = iob * torch.pow(
                torch.tensor(0.5, device=device),
                horizon / INSULIN_HALF_LIFE_MIN
            )
            insulin_absorbed = iob - iob_remaining
            iob_effect = -insulin_absorbed * adjusted_isf  # Negative (lowers BG)

            # ===== COB EFFECT (CAUSAL: carbs → BG rise) =====
            cob_remaining = cob * torch.pow(
                torch.tensor(0.5, device=device),
                horizon / carb_half_life
            )
            carbs_absorbed = cob - cob_remaining
            cob_effect = carbs_absorbed * (adjusted_isf / icr_base)  # Positive (raises BG)

            # ===== DAWN EFFECT (PHYSIOLOGICAL: morning rise) =====
            dawn_effect = self.dawn(hour, horizon, current_bg)

            # ===== TREND EFFECT (SHORT-TERM ONLY, EXPLAINED BY METABOLISM) =====
            # Trend is only meaningful if not already explained by IOB/COB/dawn
            has_explanation = (torch.abs(iob_effect) > 5) | (torch.abs(cob_effect) > 5) | (dawn_effect > 5)

            # For short horizons, allow small trend contribution if unexplained
            if horizon <= 15:
                trend_factor = (horizon / 5) * 0.3 * (~has_explanation).float()
            elif horizon <= 30:
                trend_factor = 3.0 * 0.2 * (~has_explanation).float()
            else:
                # Beyond 30 min, trend should be negligible
                trend_factor = torch.zeros(batch_size, device=device)

            trend_effect = trend * trend_factor

            # ===== TOTAL PREDICTION =====
            predicted_bg = current_bg + iob_effect + cob_effect + dawn_effect + trend_effect
            predicted_bg = torch.clamp(predicted_bg, 40, 400)

            # Store results
            results[f'pred_{horizon}'] = predicted_bg
            results[f'iob_effect_{horizon}'] = iob_effect
            results[f'cob_effect_{horizon}'] = cob_effect
            results[f'dawn_effect_{horizon}'] = dawn_effect
            results[f'trend_effect_{horizon}'] = trend_effect

            # Uncertainty scales with metabolic activity
            base_uncertainty = 5 + horizon * 0.15
            metabolic_uncertainty = (torch.abs(iob_effect) + torch.abs(cob_effect)) * 0.15
            results[f'uncertainty_{horizon}'] = base_uncertainty + metabolic_uncertainty

        return results


class CausalTFT(nn.Module):
    """
    Strictly Causal Temporal Fusion Transformer.

    Key constraint: The neural network can ONLY learn to MODULATE causal factors,
    not create independent predictions.

    Modulations learned:
    1. ISF adjustment based on time-of-day (dawn phenomenon, evening sensitivity)
    2. Carb absorption rate based on meal composition (fat/protein slow it down)
    3. Dawn phenomenon intensity (personalized)

    The physics baseline is the ONLY source of BG change predictions.
    """

    def __init__(
        self,
        n_features: int = 20,
        hidden_size: int = 32,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        default_isf: float = DEFAULT_ISF,
        default_icr: float = DEFAULT_ICR,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.default_isf = default_isf
        self.default_icr = default_icr

        # Causal physics module
        self.physics = CausalPhysicsModule(default_isf, default_icr)

        # LSTM for processing glucose history (learns patterns for MODULATION only)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )

        # Carb absorption rate predictor (based on meal features + history)
        self.absorption_predictor = nn.Sequential(
            nn.Linear(hidden_size + 4, 32),  # +4 for carbs, fat, protein, GI
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output 0-1, maps to half-life
        )
        self.min_half_life = 25.0
        self.max_half_life = 90.0

        # Small uncertainty adjustment (context-aware)
        self.uncertainty_adj = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, len(PREDICTION_HORIZONS)),
            nn.Softplus(),  # Always positive
        )

    def forward(
        self,
        features: torch.Tensor,        # (batch, seq_len, n_features)
        current_bg: torch.Tensor,      # (batch,)
        iob: torch.Tensor,             # (batch,)
        cob: torch.Tensor,             # (batch,)
        trend: torch.Tensor,           # (batch,) rate per 5 min
        hour: torch.Tensor,            # (batch,) 0-23.99
        isf: Optional[torch.Tensor] = None,  # (batch,) user's ISF
        icr: Optional[torch.Tensor] = None,  # (batch,) user's ICR
        carbs: Optional[torch.Tensor] = None,  # (batch,) recent carbs
        fat: Optional[torch.Tensor] = None,    # (batch,) fat content
        protein: Optional[torch.Tensor] = None, # (batch,) protein content
        glycemic_index: Optional[torch.Tensor] = None,  # (batch,) GI
        **kwargs,  # Ignore extra args (day_of_week, month, etc.)
    ) -> Dict[str, torch.Tensor]:
        """Generate strictly causal predictions."""
        batch_size = features.shape[0]
        device = features.device

        # Defaults
        if isf is None:
            isf = torch.full((batch_size,), self.default_isf, device=device)
        if icr is None:
            icr = torch.full((batch_size,), self.default_icr, device=device)
        if carbs is None:
            carbs = cob  # Use COB if carbs not provided
        if fat is None:
            fat = torch.zeros(batch_size, device=device)
        if protein is None:
            protein = torch.zeros(batch_size, device=device)
        if glycemic_index is None:
            glycemic_index = torch.full((batch_size,), 55.0, device=device)

        # Process history through LSTM
        lstm_out, _ = self.lstm(features)
        context = lstm_out[:, -1, :]  # Last hidden state

        # Predict carb absorption rate from context + meal features
        meal_features = torch.stack([
            carbs / 50,
            fat / 20,
            protein / 30,
            glycemic_index / 100,
        ], dim=-1)
        absorption_input = torch.cat([context, meal_features], dim=-1)
        absorption_factor = self.absorption_predictor(absorption_input).squeeze(-1)
        carb_half_life = self.min_half_life + absorption_factor * (self.max_half_life - self.min_half_life)

        # Get causal physics predictions
        physics_results = self.physics(
            current_bg=current_bg,
            iob=iob,
            cob=cob,
            trend=trend,
            hour=hour,
            carb_half_life=carb_half_life,
            isf_base=isf,
            icr_base=icr,
        )

        # Context-aware uncertainty adjustment
        uncertainty_adj = self.uncertainty_adj(context)

        # Package predictions
        predictions = []
        for i, horizon in enumerate(PREDICTION_HORIZONS):
            pred_value = physics_results[f'pred_{horizon}']
            base_uncertainty = physics_results[f'uncertainty_{horizon}']

            # Add learned uncertainty adjustment (can only increase, not decrease)
            total_uncertainty = base_uncertainty + uncertainty_adj[:, i]

            predictions.append({
                'horizon_min': horizon,
                'value': pred_value,
                'lower': torch.clamp(pred_value - total_uncertainty, 40, 400),
                'upper': torch.clamp(pred_value + total_uncertainty, 40, 400),
                'iob_effect': physics_results[f'iob_effect_{horizon}'],
                'cob_effect': physics_results[f'cob_effect_{horizon}'],
                'dawn_effect': physics_results[f'dawn_effect_{horizon}'],
                'trend_effect': physics_results[f'trend_effect_{horizon}'],
                'confidence': max(0.5, 0.9 - horizon / 300),
            })

        return {
            'predictions': predictions,
            'carb_half_life': carb_half_life,
        }

    def predict_single(
        self,
        features: torch.Tensor,
        current_bg: float,
        iob: float,
        cob: float,
        trend: float,
        hour: float,
        isf: float = DEFAULT_ISF,
        icr: float = DEFAULT_ICR,
        carbs: float = 0,
        fat: float = 0,
        protein: float = 0,
        glycemic_index: float = 55,
    ) -> List[Dict]:
        """Single-sample prediction for inference."""
        self.eval()
        device = next(self.parameters()).device

        def to_t(x):
            return torch.tensor([x], device=device, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(
                features=features.unsqueeze(0) if features.dim() == 2 else features,
                current_bg=to_t(current_bg),
                iob=to_t(iob),
                cob=to_t(cob),
                trend=to_t(trend),
                hour=to_t(hour),
                isf=to_t(isf),
                icr=to_t(icr),
                carbs=to_t(carbs),
                fat=to_t(fat),
                protein=to_t(protein),
                glycemic_index=to_t(glycemic_index),
            )

        results = []
        for pred in output['predictions']:
            results.append({
                'horizon_min': pred['horizon_min'],
                'value': pred['value'].item(),
                'lower': pred['lower'].item(),
                'upper': pred['upper'].item(),
                'iob_effect': pred['iob_effect'].item(),
                'cob_effect': pred['cob_effect'].item(),
                'dawn_effect': pred['dawn_effect'].item(),
                'confidence': pred['confidence'],
            })

        return results


def create_causal_tft(
    n_features: int = 20,
    hidden_size: int = 32,
    device: str = 'cpu',
) -> CausalTFT:
    """Create a causal TFT model."""
    model = CausalTFT(
        n_features=n_features,
        hidden_size=hidden_size,
    )
    return model.to(device)


# Alias for compatibility
PhysicsInformedTFT = CausalTFT
create_physics_tft = create_causal_tft
