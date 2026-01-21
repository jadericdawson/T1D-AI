"""
BG Pressure Predictor Model

Predicts the "pressure" on blood glucose from active insulin and carbs.
This is the combined cumulative effect of all active treatments on BG trajectory.

Key insight: BG Pressure = where BG is being pushed by IOB (down) and COB (up)
- Positive pressure = BG trending up (carbs > insulin)
- Negative pressure = BG trending down (insulin > carbs)

This model learns the ACTUAL pressure pattern from historical data,
accounting for individual physiology, food types, time of day, etc.

The output of this model directly feeds into TFT predictions, ensuring
the TFT is influenced by the current IOB/COB state.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)


# Lunar phase calculation
def get_lunar_phase(dt: datetime) -> Tuple[float, float]:
    """
    Calculate lunar phase as sin/cos for cyclical feature.

    The lunar cycle affects fluid retention, hormones, and potentially
    insulin sensitivity in some individuals.

    Returns:
        Tuple of (lunar_sin, lunar_cos) for cyclical encoding
    """
    # Known new moon reference: Jan 1, 2000 was close to new moon
    ref_date = datetime(2000, 1, 6, 18, 14)  # New moon
    lunar_cycle_days = 29.530588853  # Average synodic month

    days_since_ref = (dt - ref_date).total_seconds() / 86400
    phase = (days_since_ref % lunar_cycle_days) / lunar_cycle_days

    # Convert to sin/cos for cyclical encoding
    theta = 2 * math.pi * phase
    return math.sin(theta), math.cos(theta)


class BGPressureModel(nn.Module):
    """
    Neural network that predicts BG Pressure from current metabolic state.

    Architecture: Transformer-based with attention over treatment history

    Input features:
    - Current metabolic state (BG, trend, IOB, COB)
    - Treatment history sequence (insulin and carb events)
    - Temporal features (hour, day of week, month, lunar phase)
    - Individual factors (recent ISF observed, carb sensitivity)

    Output:
    - Predicted BG pressure at multiple future horizons (5, 10, 15, ..., 120 min)
    - Uncertainty bounds for each prediction
    """

    def __init__(
        self,
        state_size: int = 32,       # Metabolic state features
        treatment_size: int = 16,   # Per-treatment features
        hidden_size: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_treatments: int = 20,   # Max active treatments to consider
        n_horizons: int = 24,       # Predict up to 120 min (24 * 5 min)
    ):
        super().__init__()

        self.state_size = state_size
        self.treatment_size = treatment_size
        self.hidden_size = hidden_size
        self.n_horizons = n_horizons
        self.max_treatments = max_treatments

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # Treatment encoder (per treatment)
        self.treatment_encoder = nn.Sequential(
            nn.Linear(treatment_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # Treatment attention (which treatments matter most for current pressure)
        self.treatment_attention = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )

        # Temporal fusion layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output heads for each horizon
        # Outputs: median, lower bound, upper bound for each horizon
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_horizons * 3)  # median, lower, upper per horizon
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,           # (batch, state_size)
        treatments: torch.Tensor,       # (batch, max_treatments, treatment_size)
        treatment_mask: torch.Tensor,   # (batch, max_treatments) - False for padding
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Current metabolic state features
            treatments: Sequence of active treatment features
            treatment_mask: Boolean mask (False = valid, True = padding)

        Returns:
            Dict with 'predictions' (batch, n_horizons, 3) and attention weights
        """
        batch_size = state.shape[0]

        # Encode state
        state_encoded = self.state_encoder(state)  # (batch, hidden)

        # Encode treatments
        treatments_encoded = self.treatment_encoder(treatments)  # (batch, max_t, hidden)

        # Apply attention: state attends to treatments
        # Query: state, Key/Value: treatments
        state_query = state_encoded.unsqueeze(1)  # (batch, 1, hidden)

        attended, attn_weights = self.treatment_attention(
            state_query,
            treatments_encoded,
            treatments_encoded,
            key_padding_mask=treatment_mask
        )

        # Combine state with attended treatment context
        combined = state_encoded + attended.squeeze(1)  # (batch, hidden)

        # Temporal encoding (single step for now, could expand)
        combined = combined.unsqueeze(1)  # (batch, 1, hidden)
        temporal_out = self.temporal_encoder(combined)

        # Generate pressure predictions
        pressure_out = self.pressure_head(temporal_out.squeeze(1))  # (batch, n_horizons * 3)
        pressure_out = pressure_out.view(batch_size, self.n_horizons, 3)

        return {
            'predictions': pressure_out,  # (batch, n_horizons, 3) - [median, lower, upper]
            'attention_weights': attn_weights,  # (batch, 1, max_treatments)
            'encoded_state': state_encoded,
        }

    def predict_pressure_curve(
        self,
        state_features: np.ndarray,
        treatment_features: np.ndarray,
        treatment_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict full BG pressure curve for the next 2 hours.

        Args:
            state_features: (state_size,) current metabolic state
            treatment_features: (max_treatments, treatment_size) active treatments
            treatment_mask: (max_treatments,) True for padding

        Returns:
            Dict with 'horizons' (minutes), 'median', 'lower', 'upper' arrays
        """
        self.eval()

        with torch.no_grad():
            state = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            treatments = torch.tensor(treatment_features, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(treatment_mask, dtype=torch.bool).unsqueeze(0)

            output = self.forward(state, treatments, mask)
            preds = output['predictions'].squeeze(0).numpy()

        horizons = [(i + 1) * 5 for i in range(self.n_horizons)]  # 5, 10, 15, ..., 120 min

        return {
            'horizons_min': np.array(horizons),
            'median': preds[:, 0],
            'lower': preds[:, 1],
            'upper': preds[:, 2],
        }


# Feature extraction helpers

def extract_state_features(
    current_bg: float,
    trend: int,
    iob: float,
    cob: float,
    isf: float = 50.0,
    icr: float = 10.0,
    hour: int = 12,
    day_of_week: int = 3,
    day_of_month: int = 15,
    month: int = 6,
    rate_of_change: float = 0.0,
    recent_variability: float = 20.0,
    is_fasting: bool = False,
    activity_level: float = 0.0,
) -> np.ndarray:
    """
    Extract state features for the BG Pressure model.

    Total features: 32
    - BG features: 6 (value, trend, rate_of_change, variability, is_low, is_high)
    - IOB/COB features: 6 (iob, cob, iob_effect, cob_effect, net_effect, balance_ratio)
    - Sensitivity features: 4 (isf_scaled, icr_scaled, effective_ratio, sensitivity_zone)
    - Time features: 12 (hour sin/cos, dow sin/cos, dom sin/cos, month sin/cos, lunar sin/cos, year_progress sin/cos)
    - State features: 4 (is_fasting, activity, time_since_wake, post_meal_hours)
    """
    features = []

    # BG features
    features.append(current_bg / 200.0)  # Scaled BG
    features.append(trend / 3.0)  # Scaled trend (-1 to 1)
    features.append(rate_of_change / 10.0)  # Scaled rate
    features.append(recent_variability / 100.0)  # Scaled variability
    features.append(1.0 if current_bg < 70 else 0.0)  # is_low
    features.append(1.0 if current_bg > 180 else 0.0)  # is_high

    # IOB/COB features
    features.append(iob / 10.0)  # Scaled IOB
    features.append(cob / 100.0)  # Scaled COB
    iob_effect = -(iob * isf)
    cob_effect = (cob * isf / icr)
    features.append(iob_effect / 200.0)  # Scaled IOB effect
    features.append(cob_effect / 200.0)  # Scaled COB effect
    net_effect = iob_effect + cob_effect
    features.append(net_effect / 200.0)  # Scaled net effect
    balance_ratio = iob_effect / (cob_effect + 1e-6) if cob_effect > 0 else -1.0
    features.append(np.tanh(balance_ratio))  # Balance ratio (-1 to 1)

    # Sensitivity features
    features.append(isf / 100.0)  # Scaled ISF
    features.append(icr / 20.0)  # Scaled ICR
    features.append(isf / (icr + 1e-6) / 10.0)  # Effective ratio
    # ISF zone based on time of day
    if 6 <= hour < 11:
        isf_zone = 0.0  # Morning (often resistant)
    elif 11 <= hour < 17:
        isf_zone = 0.5  # Afternoon
    elif 17 <= hour < 22:
        isf_zone = 0.75  # Evening
    else:
        isf_zone = 1.0  # Night (often sensitive)
    features.append(isf_zone)

    # Time features (cyclical encoding)
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    features.append(np.sin(2 * np.pi * day_of_week / 7))
    features.append(np.cos(2 * np.pi * day_of_week / 7))
    features.append(np.sin(2 * np.pi * day_of_month / 31))
    features.append(np.cos(2 * np.pi * day_of_month / 31))
    features.append(np.sin(2 * np.pi * month / 12))
    features.append(np.cos(2 * np.pi * month / 12))

    # Lunar phase
    now = datetime(2024, month, day_of_month, hour)
    lunar_sin, lunar_cos = get_lunar_phase(now)
    features.append(lunar_sin)
    features.append(lunar_cos)

    # Year progress (captures seasonal patterns)
    day_of_year = (datetime(2024, month, day_of_month) - datetime(2024, 1, 1)).days + 1
    features.append(np.sin(2 * np.pi * day_of_year / 365))
    features.append(np.cos(2 * np.pi * day_of_year / 365))

    # State features
    features.append(1.0 if is_fasting else 0.0)
    features.append(activity_level / 3.0)  # 0-3 scaled to 0-1

    # Time since wake (estimated from hour)
    if 6 <= hour < 22:
        time_since_wake = (hour - 6) / 16.0  # Assuming wake at 6am
    else:
        time_since_wake = 0.0
    features.append(time_since_wake)

    # Placeholder for post-meal hours (would come from treatment history)
    features.append(0.5)  # Default

    return np.array(features, dtype=np.float32)


def extract_treatment_features(
    treatment: Dict,
    current_time: datetime,
    isf: float = 50.0,
    icr: float = 10.0,
) -> np.ndarray:
    """
    Extract features for a single treatment.

    Total features: 16
    - Type features: 2 (is_insulin, is_carbs)
    - Amount features: 4 (insulin_scaled, carbs_scaled, protein_scaled, fat_scaled)
    - Time features: 4 (minutes_since_scaled, absorption_progress, time_to_peak, remaining_duration)
    - Food features: 4 (gi_scaled, gl_scaled, is_high_fat, absorption_rate_code)
    - Effect features: 2 (potential_bg_effect, current_activity)
    """
    features = []

    # Parse timestamp
    t_time = treatment.get('timestamp')
    if isinstance(t_time, str):
        try:
            t_time = datetime.fromisoformat(t_time.replace('Z', '+00:00'))
            t_time = t_time.replace(tzinfo=None)
        except:
            t_time = current_time
    elif isinstance(t_time, datetime):
        t_time = t_time.replace(tzinfo=None)
    else:
        t_time = current_time

    current_time_naive = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
    minutes_since = (current_time_naive - t_time).total_seconds() / 60

    # Get values
    insulin = treatment.get('insulin', 0) or 0
    carbs = treatment.get('carbs', 0) or 0
    protein = treatment.get('protein', 0) or 0
    fat = treatment.get('fat', 0) or 0
    gi = treatment.get('glycemicIndex', 55) or 55

    # Type features
    features.append(1.0 if insulin > 0 else 0.0)  # is_insulin
    features.append(1.0 if carbs > 0 else 0.0)  # is_carbs

    # Amount features (scaled)
    features.append(insulin / 10.0)
    features.append(carbs / 100.0)
    features.append(protein / 50.0)
    features.append(fat / 50.0)

    # Time features
    features.append(min(minutes_since / 360.0, 1.0))  # Capped at 6 hours

    # Absorption progress depends on treatment type
    if insulin > 0:
        # Insulin: peaks ~75 min, DIA ~300 min
        absorption_progress = min(minutes_since / 300.0, 1.0)
        time_to_peak = max(0, (75 - minutes_since) / 75.0)
        remaining_duration = max(0, (300 - minutes_since) / 300.0)
    elif carbs > 0:
        # Carbs: depends on GI and fat
        gi_factor = gi / 55.0
        fat_delay = 60 if fat > 15 else (30 if fat > 5 else 0)
        duration = (180 + fat_delay) / gi_factor
        peak_time = (45 + fat_delay) / gi_factor

        absorption_progress = min(minutes_since / duration, 1.0)
        time_to_peak = max(0, (peak_time - minutes_since) / peak_time) if peak_time > 0 else 0
        remaining_duration = max(0, (duration - minutes_since) / duration)
    else:
        absorption_progress = 0.0
        time_to_peak = 0.0
        remaining_duration = 0.0

    features.append(absorption_progress)
    features.append(time_to_peak)
    features.append(remaining_duration)

    # Food features
    features.append(gi / 100.0)  # GI scaled
    gl = (carbs * gi / 100) / 50.0 if carbs > 0 else 0  # Glycemic load scaled
    features.append(gl)
    features.append(1.0 if fat > 15 else 0.0)  # is_high_fat

    # Absorption rate code
    absorption_rate = treatment.get('absorptionRate', 'medium')
    rate_code = {'fast': 0.0, 'medium': 0.5, 'slow': 1.0}.get(absorption_rate, 0.5)
    features.append(rate_code)

    # Effect features
    if insulin > 0:
        potential_effect = -insulin * isf  # Negative (lowering BG)
        # Activity curve for insulin (bell curve peaking at ~75 min)
        if minutes_since < 300:
            activity = np.exp(-0.5 * ((minutes_since - 75) / 50) ** 2)
        else:
            activity = 0.0
    elif carbs > 0:
        potential_effect = (carbs * isf / icr)  # Positive (raising BG)
        # Activity based on absorption progress
        if absorption_progress < 1.0:
            # Bell curve centered at peak time
            peak_min = (45 + (60 if fat > 15 else 0)) / (gi / 55.0)
            activity = np.exp(-0.5 * ((minutes_since - peak_min) / 30) ** 2)
        else:
            activity = 0.0
    else:
        potential_effect = 0.0
        activity = 0.0

    features.append(potential_effect / 200.0)  # Scaled
    features.append(activity)

    return np.array(features, dtype=np.float32)


# Model configuration
BG_PRESSURE_MODEL_CONFIG = {
    "state_size": 32,
    "treatment_size": 16,
    "hidden_size": 128,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "max_treatments": 20,
    "n_horizons": 24,  # 5, 10, 15, ..., 120 min
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
}


class BGPressureService:
    """
    Service for calculating BG Pressure using the ML model with formula fallback.

    When the ML model is trained, it provides accurate personalized pressure predictions.
    Until then, falls back to exponential decay formulas with GI adjustments.
    """

    def __init__(
        self,
        model: Optional[BGPressureModel] = None,
        isf: float = 50.0,
        icr: float = 10.0,
    ):
        self.model = model
        self.isf = isf
        self.icr = icr
        self._use_ml = model is not None

    def predict_pressure_curve(
        self,
        current_bg: float,
        trend: int,
        iob: float,
        cob: float,
        treatments: List[Dict],
        current_time: Optional[datetime] = None,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
    ) -> Dict:
        """
        Predict BG pressure curve for the next 2 hours.

        Args:
            current_bg: Current glucose value
            trend: CGM trend (-3 to +3)
            iob: Current IOB (units)
            cob: Current COB (grams)
            treatments: List of active treatments
            current_time: Current time (default: now)
            isf: Insulin sensitivity factor (optional override)
            icr: Insulin to carb ratio (optional override)

        Returns:
            Dict with pressure predictions at multiple horizons
        """
        current_time = current_time or datetime.utcnow()
        isf = isf or self.isf
        icr = icr or self.icr

        if self._use_ml and self.model is not None:
            try:
                return self._ml_predict(
                    current_bg, trend, iob, cob, treatments,
                    current_time, isf, icr
                )
            except Exception as e:
                logger.warning(f"ML BG Pressure prediction failed: {e}")

        # Fallback to formula-based calculation
        return self._formula_predict(
            current_bg, trend, iob, cob, treatments,
            current_time, isf, icr
        )

    def _ml_predict(
        self,
        current_bg: float,
        trend: int,
        iob: float,
        cob: float,
        treatments: List[Dict],
        current_time: datetime,
        isf: float,
        icr: float,
    ) -> Dict:
        """Use the ML model for prediction."""
        # Extract state features
        state = extract_state_features(
            current_bg=current_bg,
            trend=trend,
            iob=iob,
            cob=cob,
            isf=isf,
            icr=icr,
            hour=current_time.hour,
            day_of_week=current_time.weekday(),
            day_of_month=current_time.day,
            month=current_time.month,
        )

        # Extract treatment features
        max_treatments = BG_PRESSURE_MODEL_CONFIG["max_treatments"]
        treatment_features = np.zeros((max_treatments, BG_PRESSURE_MODEL_CONFIG["treatment_size"]))
        treatment_mask = np.ones(max_treatments, dtype=bool)  # True = padding

        for i, t in enumerate(treatments[:max_treatments]):
            treatment_features[i] = extract_treatment_features(t, current_time, isf, icr)
            treatment_mask[i] = False  # Valid treatment

        # Get predictions
        result = self.model.predict_pressure_curve(state, treatment_features, treatment_mask)

        return {
            'horizons_min': result['horizons_min'].tolist(),
            'pressure_median': result['median'].tolist(),
            'pressure_lower': result['lower'].tolist(),
            'pressure_upper': result['upper'].tolist(),
            'method': 'ml',
        }

    def _formula_predict(
        self,
        current_bg: float,
        trend: int,
        iob: float,
        cob: float,
        treatments: List[Dict],
        current_time: datetime,
        isf: float,
        icr: float,
    ) -> Dict:
        """
        Formula-based calculation using TREATMENTS directly.

        CRITICAL: Must match historical bgPressure calculation exactly!
        Both historical and projected use the same formula:
        - Iterate through all treatments
        - Calculate cumulative absorbed effect at each timestamp
        - This ensures continuity between historical and projected lines

        The pressure at each timestamp is the CUMULATIVE absorbed effect:
        - Insulin: negative (absorbed insulin has lowered BG)
        - Carbs: positive (absorbed carbs have raised BG)
        """
        horizons = [(i + 1) * 5 for i in range(24)]  # 5, 10, ..., 120 min
        pressures = []

        bg_per_gram = isf / icr

        logger.debug(f"BG Pressure formula: IOB={iob:.2f}U, COB={cob:.0f}g, ISF={isf}, ICR={icr}")

        for horizon in horizons:
            # Calculate CUMULATIVE ABSORBED effect at future timestamp
            # This matches the historical calculation exactly
            future_time = current_time + timedelta(minutes=horizon)

            cumulative_insulin_effect = 0.0
            cumulative_carb_effect = 0.0

            for t in treatments:
                # Parse treatment timestamp
                t_time = t.get('timestamp')
                if isinstance(t_time, str):
                    try:
                        t_time = datetime.fromisoformat(t_time.replace('Z', '+00:00'))
                        t_time = t_time.replace(tzinfo=None)
                    except:
                        continue
                elif isinstance(t_time, datetime):
                    t_time = t_time.replace(tzinfo=None)
                else:
                    continue

                future_time_naive = future_time.replace(tzinfo=None) if future_time.tzinfo else future_time
                time_since_dose = (future_time_naive - t_time).total_seconds() / 60

                if time_since_dose <= 0:
                    continue  # Treatment is in the future

                insulin = t.get('insulin', 0) or 0
                carbs = t.get('carbs', 0) or 0

                if insulin > 0:
                    # Insulin: delayed onset, peak ~75 min, DIA ~300 min
                    initial_dose = insulin
                    remaining = initial_dose * (0.5 ** (time_since_dose / 81.0)) if time_since_dose < 300 else 0
                    absorbed = initial_dose - remaining

                    # Apply onset delay - effect builds up slowly in first 30 min
                    if time_since_dose < 30:
                        onset_factor = time_since_dose / 30.0
                        absorbed *= onset_factor * onset_factor  # Quadratic onset

                    cumulative_insulin_effect += absorbed * isf

                if carbs > 0:
                    # Carbs: onset depends on glycemic index
                    gi = t.get('glycemicIndex', 55) or 55
                    initial_carbs = carbs

                    # Adjust timing based on GI
                    gi_factor = gi / 55.0
                    onset_delay = 20 / gi_factor
                    half_life = 45 / gi_factor

                    remaining = initial_carbs * (0.5 ** (time_since_dose / half_life)) if time_since_dose < 180 else 0
                    absorbed = initial_carbs - remaining

                    # Apply onset delay
                    if time_since_dose < onset_delay:
                        onset_factor = time_since_dose / onset_delay
                        absorbed *= onset_factor * onset_factor

                    cumulative_carb_effect += absorbed * bg_per_gram

            # Net pressure = carb effect - insulin effect (same formula as historical!)
            pressure = cumulative_carb_effect - cumulative_insulin_effect
            pressures.append(pressure)

            if horizon in [30, 60, 120]:
                logger.debug(f"  Horizon +{horizon}min: pressure={pressure:.1f} (carb={cumulative_carb_effect:.1f}, insulin={cumulative_insulin_effect:.1f})")

        # Uncertainty estimation - grows with horizon and metabolic activity
        base_uncertainty = 10.0
        metabolic_activity = max(iob, cob / 20)
        uncertainties = [
            base_uncertainty * (1 + h / 120) * (1 + 0.2 * metabolic_activity)
            for h in horizons
        ]

        return {
            'horizons_min': horizons,
            'pressure_median': pressures,
            'pressure_lower': [p - u for p, u in zip(pressures, uncertainties)],
            'pressure_upper': [p + u for p, u in zip(pressures, uncertainties)],
            'method': 'formula',
        }


def create_bg_pressure_model() -> BGPressureModel:
    """Create a new BG Pressure model with default configuration."""
    return BGPressureModel(
        state_size=BG_PRESSURE_MODEL_CONFIG["state_size"],
        treatment_size=BG_PRESSURE_MODEL_CONFIG["treatment_size"],
        hidden_size=BG_PRESSURE_MODEL_CONFIG["hidden_size"],
        n_heads=BG_PRESSURE_MODEL_CONFIG["n_heads"],
        n_layers=BG_PRESSURE_MODEL_CONFIG["n_layers"],
        dropout=BG_PRESSURE_MODEL_CONFIG["dropout"],
        max_treatments=BG_PRESSURE_MODEL_CONFIG["max_treatments"],
        n_horizons=BG_PRESSURE_MODEL_CONFIG["n_horizons"],
    )
