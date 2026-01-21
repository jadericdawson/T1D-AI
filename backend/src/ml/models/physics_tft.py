"""
Physics-Informed Temporal Fusion Transformer for Blood Glucose Prediction

This model combines physics-based understanding of glucose metabolism with
neural network learning to make accurate BG predictions.

Key insight: BG changes are primarily driven by:
1. IOB (Insulin on Board) - lowers BG via ISF
2. COB (Carbs on Board) - raises BG via ICR
3. Current trend - reflects immediate momentum

The neural network learns ADJUSTMENTS to the physics baseline based on:
- Time of day (circadian rhythms, dawn phenomenon)
- Day of week (weekday vs weekend patterns)
- Season/month (seasonal sensitivity changes)
- Weather (temperature, pressure affect insulin sensitivity)
- Food composition (fat/protein slow carb absorption)
- Recent BG variability (affects prediction confidence)

Author: T1D-AI
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Physics constants for insulin and carb absorption
INSULIN_HALF_LIFE_MIN = 81.0  # Fast-acting (Humalog/Novolog) half-life
CARB_HALF_LIFE_FAST = 30.0    # Fast GI carbs (juice, candy)
CARB_HALF_LIFE_MEDIUM = 45.0  # Medium GI carbs (bread, rice)
CARB_HALF_LIFE_SLOW = 75.0    # Slow GI carbs (pizza, high-fat meals)

# Default metabolic parameters (can be personalized)
DEFAULT_ISF = 55.0  # mg/dL per unit
DEFAULT_ICR = 10.0  # grams per unit

# Prediction horizons in minutes
PREDICTION_HORIZONS = [15, 30, 60, 90, 120, 180]


@dataclass
class PhysicsPrediction:
    """Physics-based baseline prediction."""
    horizon_min: int
    value: float
    iob_effect: float
    cob_effect: float
    trend_effect: float
    confidence: float


@dataclass
class PhysicsTFTOutput:
    """Output from Physics-Informed TFT."""
    predictions: List[Dict]  # {horizon_min, value, lower, upper, confidence}
    physics_baseline: List[PhysicsPrediction]
    neural_adjustment: torch.Tensor
    feature_importance: Optional[np.ndarray] = None


class PhysicsBaseline(nn.Module):
    """
    Computes physics-based baseline predictions.

    This module encodes the known pharmacokinetics of insulin and carb absorption.
    It's not learned - it's fixed physics that the neural network adjusts.
    """

    def __init__(
        self,
        default_isf: float = DEFAULT_ISF,
        default_icr: float = DEFAULT_ICR,
    ):
        super().__init__()
        self.default_isf = default_isf
        self.default_icr = default_icr

    def forward(
        self,
        current_bg: torch.Tensor,      # (batch,)
        iob: torch.Tensor,             # (batch,)
        cob: torch.Tensor,             # (batch,)
        trend: torch.Tensor,           # (batch,) rate per 5 min
        isf: torch.Tensor,             # (batch,) personalized ISF
        icr: torch.Tensor,             # (batch,) personalized ICR
        carb_half_life: torch.Tensor,  # (batch,) based on meal composition
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based predictions for all horizons.

        Returns dict with keys for each horizon (e.g., 'pred_30', 'pred_60')
        and the effect components ('iob_effect_30', 'cob_effect_30', etc.)
        """
        batch_size = current_bg.shape[0]
        device = current_bg.device

        results = {}

        for horizon in PREDICTION_HORIZONS:
            # IOB effect: insulin absorbed from now to horizon
            # IOB_remaining = IOB * 0.5^(t/half_life)
            iob_remaining = iob * torch.pow(
                torch.tensor(0.5, device=device),
                horizon / INSULIN_HALF_LIFE_MIN
            )
            insulin_absorbed = iob - iob_remaining
            iob_effect = -insulin_absorbed * isf  # Negative (lowers BG)

            # COB effect: carbs absorbed from now to horizon
            # Half-life varies by food composition
            cob_remaining = cob * torch.pow(
                torch.tensor(0.5, device=device),
                horizon / carb_half_life
            )
            carbs_absorbed = cob - cob_remaining
            # Carb effect in mg/dL = carbs * (ISF / ICR)
            cob_effect = carbs_absorbed * (isf / icr)  # Positive (raises BG)

            # Trend effect: extrapolate short-term, dampen long-term
            # When IOB/COB active, trend is already explained by metabolism
            has_metabolism = (iob > 0.5) | (cob > 10)

            # Short-term trend (< 30 min): partial contribution
            # Long-term trend (> 30 min): heavily damped
            if horizon <= 30:
                trend_factor = (horizon / 5) * 0.5 * (~has_metabolism).float() + \
                              (horizon / 5) * 0.2 * has_metabolism.float()
            else:
                # Beyond 30 min, cap trend contribution
                trend_factor = 3.0 * 0.3 * torch.exp(
                    torch.tensor(-(horizon - 30) / 60, device=device)
                )

            trend_effect = trend * trend_factor

            # Total prediction
            predicted_bg = current_bg + iob_effect + cob_effect + trend_effect

            # Clamp to physiological range
            predicted_bg = torch.clamp(predicted_bg, 40, 400)

            # Store results
            results[f'pred_{horizon}'] = predicted_bg
            results[f'iob_effect_{horizon}'] = iob_effect
            results[f'cob_effect_{horizon}'] = cob_effect
            results[f'trend_effect_{horizon}'] = trend_effect

            # Baseline uncertainty grows with horizon and metabolic activity
            base_uncertainty = 8 + horizon * 0.2
            metabolic_uncertainty = (torch.abs(iob_effect) + torch.abs(cob_effect)) * 0.1
            results[f'uncertainty_{horizon}'] = base_uncertainty + metabolic_uncertainty

        return results


class TimeEncoder(nn.Module):
    """
    Encodes time features with cyclical representations.

    Time of day is crucial for capturing:
    - Dawn phenomenon (4-8 AM: higher insulin resistance)
    - Circadian meal patterns
    - Sleep vs wake periods
    """

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim

        # Learnable projections for time features
        self.time_projection = nn.Linear(8, output_dim)

    def forward(
        self,
        hour: torch.Tensor,        # (batch,) 0-23
        day_of_week: torch.Tensor, # (batch,) 0-6
        month: torch.Tensor,       # (batch,) 1-12
        lunar_phase: Optional[torch.Tensor] = None,  # (batch,) 0-1
    ) -> torch.Tensor:
        """Encode time features into cyclical representation."""
        batch_size = hour.shape[0]
        device = hour.device

        # Hour of day (24-hour cycle)
        hour_sin = torch.sin(2 * math.pi * hour / 24)
        hour_cos = torch.cos(2 * math.pi * hour / 24)

        # Day of week (7-day cycle)
        dow_sin = torch.sin(2 * math.pi * day_of_week / 7)
        dow_cos = torch.cos(2 * math.pi * day_of_week / 7)

        # Month (12-month cycle for seasonal patterns)
        month_sin = torch.sin(2 * math.pi * month / 12)
        month_cos = torch.cos(2 * math.pi * month / 12)

        # Lunar phase (29.5-day cycle) - optional
        if lunar_phase is not None:
            lunar_sin = torch.sin(2 * math.pi * lunar_phase)
            lunar_cos = torch.cos(2 * math.pi * lunar_phase)
        else:
            lunar_sin = torch.zeros(batch_size, device=device)
            lunar_cos = torch.zeros(batch_size, device=device)

        # Combine all time features
        time_features = torch.stack([
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            month_sin, month_cos,
            lunar_sin, lunar_cos,
        ], dim=-1)

        return self.time_projection(time_features)


class ISFAdjustmentModule(nn.Module):
    """
    Learns time-of-day adjustments to ISF.

    Key patterns to capture:
    - Dawn phenomenon: 4-8 AM typically requires 15-30% more insulin
    - Evening sensitivity: Some people are more sensitive after 6 PM
    - Seasonal variation: Winter vs summer
    """

    def __init__(self, time_dim: int = 16, hidden_dim: int = 32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1], representing ISF adjustment %
        )

        # Scale factor: max 30% adjustment
        self.scale = 0.3

    def forward(self, time_encoding: torch.Tensor) -> torch.Tensor:
        """
        Returns ISF multiplier (0.7 to 1.3).

        A value > 1 means higher ISF (less insulin effect) = insulin resistance
        A value < 1 means lower ISF (more insulin effect) = higher sensitivity
        """
        adjustment = self.network(time_encoding) * self.scale
        return 1.0 + adjustment.squeeze(-1)


class AbsorptionRateModule(nn.Module):
    """
    Predicts carb absorption rate based on meal features.

    Fat and protein slow carb absorption:
    - Low fat: ~30 min half-life
    - Medium fat: ~45 min half-life
    - High fat (pizza): ~75+ min half-life
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()

        # Input: [carbs, fat, protein, gi]
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Maps [0, 1] to half-life range [30, 90] minutes
        self.min_half_life = 30.0
        self.max_half_life = 90.0

    def forward(
        self,
        carbs: torch.Tensor,
        fat: torch.Tensor,
        protein: torch.Tensor,
        glycemic_index: torch.Tensor,
    ) -> torch.Tensor:
        """Returns carb absorption half-life in minutes."""
        # Normalize inputs
        meal_features = torch.stack([
            carbs / 50,           # Normalize to ~1 for typical meal
            fat / 20,             # High fat > 15g
            protein / 30,         # High protein > 20g
            glycemic_index / 100, # GI 0-100
        ], dim=-1)

        # Network predicts slow-down factor
        slowdown = self.network(meal_features).squeeze(-1)

        # Map to half-life range
        half_life = self.min_half_life + slowdown * (self.max_half_life - self.min_half_life)

        return half_life


class PhysicsInformedTFT(nn.Module):
    """
    Physics-Informed Temporal Fusion Transformer for BG Prediction.

    Architecture:
    1. Physics baseline computes expected BG from IOB/COB/trend
    2. Time encoder captures circadian patterns
    3. ISF adjustment module learns time-varying sensitivity
    4. Absorption rate module predicts carb absorption speed
    5. Neural adjustment network learns residual corrections

    The neural network learns to ADJUST the physics baseline, not replace it.
    This ensures predictions always respect known glucose metabolism.
    """

    def __init__(
        self,
        n_features: int = 32,
        hidden_size: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        default_isf: float = DEFAULT_ISF,
        default_icr: float = DEFAULT_ICR,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        # Physics baseline (not learned)
        self.physics = PhysicsBaseline(default_isf, default_icr)

        # Time encoding
        self.time_encoder = TimeEncoder(output_dim=16)

        # ISF adjustment based on time
        self.isf_adjustment = ISFAdjustmentModule(time_dim=16, hidden_dim=32)

        # Absorption rate predictor
        self.absorption_rate = AbsorptionRateModule(input_dim=4)

        # Feature processing LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Multi-head attention for feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Neural adjustment heads (one per horizon)
        self.adjustment_heads = nn.ModuleDict({
            f'adj_{h}': nn.Sequential(
                nn.Linear(hidden_size + 16, hidden_size // 2),  # +16 for time encoding
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 3),  # [adjustment, lower_adj, upper_adj]
            )
            for h in PREDICTION_HORIZONS
        })

        # Uncertainty scaling (learned)
        self.uncertainty_scale = nn.Parameter(torch.ones(len(PREDICTION_HORIZONS)))

    def forward(
        self,
        features: torch.Tensor,        # (batch, seq_len, n_features)
        current_bg: torch.Tensor,      # (batch,)
        iob: torch.Tensor,             # (batch,)
        cob: torch.Tensor,             # (batch,)
        trend: torch.Tensor,           # (batch,) rate per 5 min
        hour: torch.Tensor,            # (batch,) 0-23
        day_of_week: torch.Tensor,     # (batch,) 0-6
        month: torch.Tensor,           # (batch,) 1-12
        isf: Optional[torch.Tensor] = None,  # (batch,) user's ISF
        icr: Optional[torch.Tensor] = None,  # (batch,) user's ICR
        carbs: Optional[torch.Tensor] = None,  # (batch,) recent carbs
        fat: Optional[torch.Tensor] = None,    # (batch,) fat content
        protein: Optional[torch.Tensor] = None, # (batch,) protein content
        glycemic_index: Optional[torch.Tensor] = None,  # (batch,) GI
        lunar_phase: Optional[torch.Tensor] = None,     # (batch,) 0-1
    ) -> PhysicsTFTOutput:
        """
        Generate predictions combining physics baseline with neural adjustments.
        """
        batch_size = features.shape[0]
        device = features.device

        # Default values
        if isf is None:
            isf = torch.full((batch_size,), self.physics.default_isf, device=device)
        if icr is None:
            icr = torch.full((batch_size,), self.physics.default_icr, device=device)
        if carbs is None:
            carbs = torch.zeros(batch_size, device=device)
        if fat is None:
            fat = torch.zeros(batch_size, device=device)
        if protein is None:
            protein = torch.zeros(batch_size, device=device)
        if glycemic_index is None:
            glycemic_index = torch.full((batch_size,), 55.0, device=device)  # Medium GI default

        # 1. Time encoding
        time_encoding = self.time_encoder(hour, day_of_week, month, lunar_phase)

        # 2. ISF adjustment based on time of day
        isf_multiplier = self.isf_adjustment(time_encoding)
        adjusted_isf = isf * isf_multiplier

        # 3. Carb absorption rate based on meal composition
        carb_half_life = self.absorption_rate(carbs, fat, protein, glycemic_index)

        # 4. Physics baseline predictions
        physics_results = self.physics(
            current_bg=current_bg,
            iob=iob,
            cob=cob,
            trend=trend,
            isf=adjusted_isf,
            icr=icr,
            carb_half_life=carb_half_life,
        )

        # 5. Process features through LSTM
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 6. Self-attention for feature importance
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        context = attn_out[:, -1, :]  # (batch, hidden_size)

        # 7. Combine context with time encoding for adjustments
        combined = torch.cat([context, time_encoding], dim=-1)

        # 8. Generate neural adjustments for each horizon
        predictions = []
        physics_baseline = []

        for i, horizon in enumerate(PREDICTION_HORIZONS):
            # Physics baseline
            baseline = physics_results[f'pred_{horizon}']
            base_uncertainty = physics_results[f'uncertainty_{horizon}']

            # Neural adjustment
            adj_output = self.adjustment_heads[f'adj_{horizon}'](combined)
            adjustment = adj_output[:, 0] * 20  # Scale adjustment to ±20 mg/dL max
            lower_adj = adj_output[:, 1] * 10
            upper_adj = adj_output[:, 2] * 10

            # Final prediction = baseline + adjustment
            final_pred = baseline + adjustment
            final_pred = torch.clamp(final_pred, 40, 400)

            # Uncertainty bounds
            scaled_uncertainty = base_uncertainty * self.uncertainty_scale[i]
            lower = final_pred - scaled_uncertainty - torch.abs(lower_adj)
            upper = final_pred + scaled_uncertainty + torch.abs(upper_adj)

            # Confidence (decreases with horizon)
            confidence = max(0.5, 0.9 - horizon / 300)

            predictions.append({
                'horizon_min': horizon,
                'value': final_pred,
                'lower': torch.clamp(lower, 40, 400),
                'upper': torch.clamp(upper, 40, 400),
                'confidence': confidence,
            })

            physics_baseline.append(PhysicsPrediction(
                horizon_min=horizon,
                value=baseline.mean().item(),
                iob_effect=physics_results[f'iob_effect_{horizon}'].mean().item(),
                cob_effect=physics_results[f'cob_effect_{horizon}'].mean().item(),
                trend_effect=physics_results[f'trend_effect_{horizon}'].mean().item(),
                confidence=confidence,
            ))

        # Feature importance from attention weights
        feature_importance = attn_weights.mean(dim=1).detach().cpu().numpy() if attn_weights is not None else None

        return PhysicsTFTOutput(
            predictions=predictions,
            physics_baseline=physics_baseline,
            neural_adjustment=adjustment,
            feature_importance=feature_importance,
        )

    def predict(
        self,
        features: torch.Tensor,
        current_bg: float,
        iob: float,
        cob: float,
        trend: float,
        hour: int,
        day_of_week: int,
        month: int,
        isf: float = DEFAULT_ISF,
        icr: float = DEFAULT_ICR,
        carbs: float = 0,
        fat: float = 0,
        protein: float = 0,
        glycemic_index: float = 55,
    ) -> List[Dict]:
        """
        Convenience method for single-sample prediction.

        Returns list of predictions in the format expected by the API.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert scalars to tensors
        def to_tensor(x):
            return torch.tensor([x], device=device, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(
                features=features.unsqueeze(0) if features.dim() == 2 else features,
                current_bg=to_tensor(current_bg),
                iob=to_tensor(iob),
                cob=to_tensor(cob),
                trend=to_tensor(trend),
                hour=to_tensor(hour),
                day_of_week=to_tensor(day_of_week),
                month=to_tensor(month),
                isf=to_tensor(isf),
                icr=to_tensor(icr),
                carbs=to_tensor(carbs),
                fat=to_tensor(fat),
                protein=to_tensor(protein),
                glycemic_index=to_tensor(glycemic_index),
            )

        # Convert to list of dicts
        results = []
        for pred in output.predictions:
            results.append({
                'horizon_min': pred['horizon_min'],
                'value': pred['value'].item(),
                'lower': pred['lower'].item(),
                'upper': pred['upper'].item(),
                'confidence': pred['confidence'],
            })

        return results


def create_physics_tft(
    n_features: int = 32,
    hidden_size: int = 64,
    device: str = 'cpu',
) -> PhysicsInformedTFT:
    """Create a physics-informed TFT model."""
    model = PhysicsInformedTFT(
        n_features=n_features,
        hidden_size=hidden_size,
    )
    return model.to(device)
