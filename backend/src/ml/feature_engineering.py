"""
Feature Engineering Module
Ported from train_bg_predictor.py and dexcom_reader_predict.py

Creates all features required for BG and ISF prediction models.

Version 2.0: Extended to ~65 features for TFT model including:
- Core glucose features (value, trend, rate of change, acceleration)
- ML-based IOB/COB features
- Food features (glycemic index, absorption rate)
- Multi-window rolling statistics
- Glucose variability metrics
- Event timing features
- Cyclical time features
- Pattern detection features
- Derived metabolic features
- Polynomial trend features
- Historical pattern features
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


# Feature configuration matching the trained models
SAMPLING_MIN = 5  # Data granularity in minutes
POLY_DEGREE = 2
ROLL_WINDOW_SHORT = 6   # 30 min / 5 min = 6 steps
ROLL_WINDOW_LONG = 18   # 90 min / 5 min = 18 steps
POLY_WINDOW_SIZE = 24   # 120 min / 5 min = 24 steps
SEQ_LENGTH = 24         # 120 min / 5 min = 24 steps for LSTM input

# Rolling window configurations (in number of 5-min steps)
ROLL_WINDOWS = {
    15: 3,    # 15 min = 3 steps
    30: 6,    # 30 min = 6 steps
    60: 12,   # 60 min = 12 steps
    90: 18,   # 90 min = 18 steps
    180: 36,  # 180 min = 36 steps
}

# Original feature columns for LSTM (backwards compatibility)
BG_FEATURE_COLUMNS = [
    "value", "trend",
    "carbs", "protein", "fat",
    "iob",
    "roll30_mean", "roll30_std", "roll90_mean", "roll90_std",
    "secs_since_start",
    "min5_sin", "min5_cos", "min15_sin", "min15_cos",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "mon_sin", "mon_cos", "doy_sin", "doy_cos",
    "poly0", "poly1", "poly2"
]

# Extended feature columns for TFT model (~65 features)
TFT_FEATURE_COLUMNS = [
    # Category 1: Core Glucose Features (4)
    "value", "trend", "rate_of_change", "acceleration",

    # Category 2: ML-Based IOB/COB Features (5)
    "ml_iob", "ml_cob", "ml_iob_effect", "ml_cob_effect", "net_effect",

    # Category 3: Food Features (6)
    "glycemic_index", "glycemic_load",
    "absorption_fast", "absorption_medium", "absorption_slow",
    "fat_high",

    # Category 4: Macro Features (5)
    "carbs", "protein", "fat", "carb_ratio", "fat_ratio",

    # Category 5: Rolling Statistics Multi-Window (10)
    "roll15_mean", "roll15_std",
    "roll30_mean", "roll30_std",
    "roll60_mean", "roll60_std",
    "roll90_mean", "roll90_std",
    "roll180_mean", "roll180_std",

    # Category 6: Glucose Variability Metrics (6)
    "coefficient_of_variation", "glucose_range",
    "time_in_range", "time_low", "time_high",
    "consecutive_in_range",

    # Category 7: Event Timing Features (6)
    "minutes_since_meal", "minutes_since_bolus", "minutes_since_correction",
    "meals_today", "total_carbs_today", "total_insulin_today",

    # Category 8: Cyclical Time Features (14)
    "min5_sin", "min5_cos", "min15_sin", "min15_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "dom_sin", "dom_cos",
    "mon_sin", "mon_cos",
    "doy_sin", "doy_cos",

    # Category 9: Pattern Detection Features (5)
    "is_fasting", "is_sleeping_hours", "dawn_phenomenon_zone",
    "post_meal_window", "stacking_risk",

    # Category 10: Derived Metabolic Features (4)
    "effective_carb_ratio", "recent_isf_observed",
    "insulin_sensitivity_zone", "carb_absorption_delay",

    # Category 11: Polynomial Trend Features (3)
    "poly0", "poly1", "poly2",

    # Category 12: Time Context (1)
    "secs_since_start",

    # Category 13: Pump Features (5)
    "basal_rate",              # Current basal rate in U/hr
    "auto_correction_count",   # Rolling 1-hour count of auto-correction events
    "auto_correction_insulin", # Rolling 1-hour sum of auto-correction insulin
    "basal_deviation",         # Current basal vs historical median (Control-IQ adjustments)
    "is_pump_user",            # Binary flag
]


def sincos(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    """Convert a cyclical feature to sin/cos components."""
    theta = 2 * np.pi * series / period
    return np.sin(theta), np.cos(theta)


def calculate_iob_for_features(
    insulin_series: pd.Series,
    duration_min: int = 180,
    sampling_min: int = 5
) -> pd.Series:
    """
    Calculate Insulin On Board (IOB) using linear decay model.

    This is a simplified version for feature engineering.
    For actual IOB calculations, use the IOBCOBService with exponential decay.

    Args:
        insulin_series: Series of insulin doses
        duration_min: Duration of insulin action (default: 180 min)
        sampling_min: Time between samples (default: 5 min)

    Returns:
        Series of IOB values
    """
    iob = pd.Series(0.0, index=insulin_series.index)
    decay_steps = duration_min // sampling_min

    if decay_steps <= 0:
        logger.warning("IOB decay steps <= 0. Returning zero IOB.")
        return iob

    # Find indices where insulin > 0
    bolus_indices = insulin_series[insulin_series > 0].index

    for idx in bolus_indices:
        bolus_amount = insulin_series[idx]
        # Apply linear decay for 'decay_steps' into the future
        for i in range(decay_steps):
            future_idx = idx + i
            if future_idx < len(iob):
                decay_factor = max(0.0, 1.0 - (i / decay_steps))
                iob.iloc[future_idx] += bolus_amount * decay_factor

    return iob


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features required for BG prediction model.

    Ported from train_bg_predictor.py lines 157-248.

    Args:
        df_in: DataFrame with columns: timestamp, value, trend, carbs,
               protein, fat, insulin

    Returns:
        DataFrame with all engineered features
    """
    logger.debug(f"Starting feature engineering on DF shape: {df_in.shape if df_in is not None else 'None'}")

    if df_in is None or df_in.empty:
        logger.warning("engineer_features received empty or None DataFrame.")
        return pd.DataFrame()

    required_inputs = ["timestamp", "value"]
    if not all(col in df_in.columns for col in required_inputs):
        missing = [col for col in required_inputs if col not in df_in.columns]
        logger.error(f"Input DataFrame missing required columns: {missing}. Aborting.")
        return pd.DataFrame()

    df = df_in.copy()

    # Ensure essential columns exist with defaults
    for col in ["trend", "carbs", "protein", "fat", "insulin"]:
        if col not in df:
            df[col] = 0.0

    # Coerce to numeric
    for col in ["value", "trend", "carbs", "protein", "fat", "insulin"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'trend' in df.columns:
        df['trend'] = df['trend'].astype(int)

    # Parse timestamp - handle both string and datetime objects
    # model_dump() from Pydantic returns datetime objects, not strings
    try:
        if df["timestamp"].dtype == 'object':
            # Check if first element is already a datetime
            first_ts = df["timestamp"].iloc[0]
            if hasattr(first_ts, 'tzinfo'):
                # Already datetime objects in object dtype column
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')
            else:
                # String timestamps
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        else:
            # Already datetime64 dtype - ensure UTC timezone
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert('UTC')
    except Exception as e:
        logger.error(f"Timestamp parsing error: {e}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    df.dropna(subset=["timestamp", "value"], inplace=True)

    if df.empty:
        logger.error("DataFrame empty after initial check/dropna.")
        return pd.DataFrame()

    df.sort_values("timestamp", inplace=True, ignore_index=True)

    # --- Calculate IOB ---
    df['iob'] = calculate_iob_for_features(
        df['insulin'],
        duration_min=180,
        sampling_min=SAMPLING_MIN
    )

    # --- Time Features ---
    start_time_utc = df["timestamp"].iloc[0]
    df["secs_since_start"] = (df["timestamp"] - start_time_utc).dt.total_seconds()

    dt_col = df["timestamp"].dt
    df["minute"] = dt_col.minute
    df["hour"] = dt_col.hour
    df["day_of_week"] = dt_col.dayofweek
    df["month"] = dt_col.month
    df["day_of_year"] = dt_col.dayofyear

    # Cyclical time features
    df["min5_sin"], df["min5_cos"] = sincos(df["minute"] % 5, 5)
    df["min15_sin"], df["min15_cos"] = sincos(df["minute"] % 15, 15)
    df["hour_sin"], df["hour_cos"] = sincos(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = sincos(df["day_of_week"], 7)
    df["mon_sin"], df["mon_cos"] = sincos(df["month"], 12)
    df["doy_sin"], df["doy_cos"] = sincos(df["day_of_year"], 365)

    # --- Rolling Statistics ---
    df["roll30_mean"] = df["value"].rolling(window=ROLL_WINDOW_SHORT, min_periods=1).mean()
    df["roll30_std"] = df["value"].rolling(window=ROLL_WINDOW_SHORT, min_periods=1).std().fillna(0)
    df["roll90_mean"] = df["value"].rolling(window=ROLL_WINDOW_LONG, min_periods=1).mean()
    df["roll90_std"] = df["value"].rolling(window=ROLL_WINDOW_LONG, min_periods=1).std().fillna(0)

    # --- Polynomial Features ---
    current_poly_win_size = min(POLY_WINDOW_SIZE, len(df))

    if current_poly_win_size >= POLY_DEGREE + 1:
        for k in range(POLY_DEGREE + 1):
            col = f"poly{k}"
            try:
                df[col] = (
                    df["value"]
                    .rolling(current_poly_win_size, min_periods=POLY_DEGREE + 1)
                    .apply(
                        lambda y: _safe_polyfit(y, POLY_DEGREE, k),
                        raw=True
                    )
                    .ffill().bfill().fillna(0)
                )
            except Exception as e:
                logger.error(f"Error calculating polynomial feature {col}: {e}")
                df[col] = 0.0
    else:
        logger.warning(f"Not enough data ({len(df)}) for poly features. Setting to 0.")
        for k in range(POLY_DEGREE + 1):
            df[f"poly{k}"] = 0.0

    logger.debug(f"Feature engineering complete. Output DF shape: {df.shape}")
    return df


def _safe_polyfit(y: np.ndarray, degree: int, coef_index: int) -> float:
    """Safely compute polynomial coefficient with error handling."""
    try:
        y_clean = y[~np.isnan(y)]
        if len(y_clean) >= degree + 1:
            coeffs = np.polyfit(np.arange(len(y_clean)), y_clean, degree)
            return coeffs[coef_index]
    except Exception:
        pass
    return np.nan


def extract_feature_sequence(
    df: pd.DataFrame,
    feature_columns: List[str] = None,
    seq_length: int = SEQ_LENGTH
) -> Optional[np.ndarray]:
    """
    Extract the last seq_length rows as a feature sequence for LSTM input.

    Args:
        df: DataFrame with engineered features
        feature_columns: List of feature column names (default: BG_FEATURE_COLUMNS)
        seq_length: Number of time steps (default: 24 = 120 min)

    Returns:
        Numpy array of shape (1, seq_length, n_features) or None if insufficient data
    """
    if feature_columns is None:
        feature_columns = BG_FEATURE_COLUMNS

    if len(df) < seq_length:
        logger.warning(f"Not enough data for sequence. Have {len(df)}, need {seq_length}")
        return None

    # Check all feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return None

    # Extract last seq_length rows
    sequence = df[feature_columns].tail(seq_length).values.astype(np.float32)

    # Add batch dimension: (1, seq_length, n_features)
    return sequence.reshape(1, seq_length, len(feature_columns))


def prepare_realtime_features(
    glucose_readings: List[dict],
    treatments: List[dict],
    iob_value: float = 0.0
) -> Optional[pd.DataFrame]:
    """
    Prepare features from realtime glucose and treatment data.

    Args:
        glucose_readings: List of glucose reading dicts with 'timestamp', 'value', 'trend'
        treatments: List of treatment dicts with 'timestamp', 'type', 'insulin', 'carbs', etc.
        iob_value: Pre-calculated IOB value from IOBCOBService

    Returns:
        DataFrame ready for feature engineering or None if insufficient data
    """
    if not glucose_readings:
        logger.warning("No glucose readings provided")
        return None

    logger.debug(f"prepare_realtime_features: {len(glucose_readings)} readings, {len(treatments) if treatments else 0} treatments")

    # Convert to DataFrame
    df = pd.DataFrame(glucose_readings)

    logger.debug(f"DataFrame columns after creation: {list(df.columns)}")

    # Ensure required columns
    if 'value' not in df.columns and 'sgv' in df.columns:
        df['value'] = df['sgv']

    if 'trend' not in df.columns:
        df['trend'] = 0

    # Add treatment data
    df['insulin'] = 0.0
    df['carbs'] = 0.0
    df['protein'] = 0.0
    df['fat'] = 0.0

    # Merge treatments with closest glucose reading
    if treatments:
        treatments_df = pd.DataFrame(treatments)
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], errors='coerce', utc=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

        # For each treatment, add to the closest glucose reading
        for _, treat in treatments_df.iterrows():
            if pd.isna(treat['timestamp']):
                continue

            # Find closest glucose reading
            time_diffs = abs(df['timestamp'] - treat['timestamp'])
            closest_idx = time_diffs.idxmin()

            # Helper to convert None to 0
            def safe_float(val, default=0.0):
                return float(val) if val is not None else default

            insulin_val = safe_float(treat.get('insulin'))
            carbs_val = safe_float(treat.get('carbs'))
            protein_val = safe_float(treat.get('protein'))
            fat_val = safe_float(treat.get('fat'))

            if treat.get('type') == 'insulin' or insulin_val > 0:
                df.loc[closest_idx, 'insulin'] += insulin_val
            elif treat.get('type') == 'carbs' or carbs_val > 0:
                df.loc[closest_idx, 'carbs'] += carbs_val
                df.loc[closest_idx, 'protein'] += protein_val
                df.loc[closest_idx, 'fat'] += fat_val

    # Override IOB with pre-calculated value if provided
    # (The IOB from IOBCOBService uses exponential decay which is more accurate)
    df['iob'] = iob_value

    return df


class FeatureEngineer:
    """
    Stateful feature engineering for real-time predictions.

    Maintains a buffer of recent readings and treatments for
    efficient feature computation.
    """

    def __init__(
        self,
        history_min: int = 180,  # 3 hours of history
        sampling_min: int = 5
    ):
        self.history_min = history_min
        self.sampling_min = sampling_min
        self.max_readings = history_min // sampling_min

        self._glucose_buffer: List[dict] = []
        self._treatment_buffer: List[dict] = []

    def add_glucose_reading(self, reading: dict) -> None:
        """Add a glucose reading to the buffer."""
        self._glucose_buffer.append(reading)

        # Trim to max size
        if len(self._glucose_buffer) > self.max_readings:
            self._glucose_buffer = self._glucose_buffer[-self.max_readings:]

    def add_treatment(self, treatment: dict) -> None:
        """Add a treatment to the buffer."""
        self._treatment_buffer.append(treatment)

        # Remove old treatments (older than history window)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.history_min)
        self._treatment_buffer = [
            t for t in self._treatment_buffer
            if pd.to_datetime(t.get('timestamp'), utc=True) > cutoff
        ]

    def get_feature_sequence(
        self,
        iob_value: float = 0.0,
        seq_length: int = SEQ_LENGTH
    ) -> Optional[np.ndarray]:
        """
        Get the current feature sequence for prediction.

        Args:
            iob_value: Pre-calculated IOB from IOBCOBService
            seq_length: Sequence length for LSTM

        Returns:
            Numpy array of shape (1, seq_length, n_features) or None
        """
        buffer_len = len(self._glucose_buffer)
        logger.debug(f"get_feature_sequence: buffer has {buffer_len} readings, need {seq_length}")

        if buffer_len < seq_length:
            logger.warning(
                f"Not enough glucose readings. Have {buffer_len}, "
                f"need {seq_length}"
            )
            return None

        # Log sample of buffer contents for debugging
        if buffer_len > 0:
            sample = self._glucose_buffer[0]
            logger.debug(f"Sample buffer reading keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")

        # Prepare DataFrame from buffer
        df = prepare_realtime_features(
            self._glucose_buffer,
            self._treatment_buffer,
            iob_value
        )

        if df is None:
            logger.warning("prepare_realtime_features returned None")
            return None

        logger.debug(f"After prepare_realtime_features: shape={df.shape}, columns={list(df.columns)[:5]}...")

        # Engineer features
        df = engineer_features(df)

        if df.empty:
            logger.warning("engineer_features returned empty DataFrame")
            return None

        logger.debug(f"After engineer_features: shape={df.shape}")

        # Extract sequence
        seq = extract_feature_sequence(df, BG_FEATURE_COLUMNS, seq_length)
        if seq is None:
            logger.warning("extract_feature_sequence returned None")
        else:
            logger.debug(f"Feature sequence shape: {seq.shape}")

        return seq

    def clear(self) -> None:
        """Clear all buffers."""
        self._glucose_buffer.clear()
        self._treatment_buffer.clear()


# =============================================================================
# Extended Feature Engineering for TFT Model
# =============================================================================

def _ensure_tz_consistent(ts1: pd.Timestamp, ts2: pd.Timestamp) -> tuple:
    """Ensure two timestamps have consistent timezone handling."""
    # Convert both to timezone-naive for comparison
    if hasattr(ts1, 'tz') and ts1.tz is not None:
        ts1 = ts1.tz_localize(None)
    if hasattr(ts2, 'tz') and ts2.tz is not None:
        ts2 = ts2.tz_localize(None)
    return ts1, ts2


def compute_per_timestep_iob(
    df: pd.DataFrame,
    treatments_df: Optional[pd.DataFrame] = None,
    insulin_action_minutes: int = 180,
    peak_minutes: int = 75
) -> pd.Series:
    """
    Compute Insulin On Board (IOB) for each timestep using bilinear decay curve.

    This is CRITICAL for training - each timestep must have accurate IOB
    reflecting all active insulin at that moment, not a static value.

    The bilinear model uses:
    - Linear ramp up to peak at ~75 minutes
    - Linear decay from peak to zero at ~180 minutes

    Args:
        df: DataFrame with 'timestamp' column
        treatments_df: DataFrame with 'timestamp' and 'insulin' columns
        insulin_action_minutes: Total insulin action time (default 180 = 3 hours)
        peak_minutes: Time to peak insulin activity (default 75 min)

    Returns:
        Series of IOB values aligned to df index
    """
    iob_values = pd.Series(0.0, index=df.index)

    if treatments_df is None or treatments_df.empty:
        return iob_values

    # Ensure timestamp is datetime and timezone-naive for consistent comparison
    df_timestamps = pd.to_datetime(df['timestamp'])
    if df_timestamps.dt.tz is not None:
        df_timestamps = df_timestamps.dt.tz_localize(None)

    # Get insulin treatments
    treatments = treatments_df.copy()
    if 'timestamp' not in treatments.columns:
        return iob_values

    treatments['timestamp'] = pd.to_datetime(treatments['timestamp'])
    if treatments['timestamp'].dt.tz is not None:
        treatments['timestamp'] = treatments['timestamp'].dt.tz_localize(None)

    # Filter to insulin treatments
    if 'insulin' in treatments.columns:
        insulin_treatments = treatments[treatments['insulin'] > 0].copy()
    else:
        return iob_values

    if insulin_treatments.empty:
        return iob_values

    # For each glucose reading, calculate total IOB from all active insulin
    for idx, row in df.iterrows():
        current_time = df_timestamps.loc[idx]
        total_iob = 0.0

        for _, treat in insulin_treatments.iterrows():
            treat_time = treat['timestamp']
            insulin_amount = treat['insulin']

            # Time since bolus in minutes
            minutes_since = (current_time - treat_time).total_seconds() / 60

            if minutes_since < 0 or minutes_since > insulin_action_minutes:
                continue  # Not yet given or fully absorbed

            # Bilinear decay model
            if minutes_since <= peak_minutes:
                # Rising phase: linear from 0 to 1
                activity = minutes_since / peak_minutes
            else:
                # Falling phase: linear from 1 to 0
                remaining = insulin_action_minutes - peak_minutes
                time_past_peak = minutes_since - peak_minutes
                activity = 1.0 - (time_past_peak / remaining)

            # IOB = amount * remaining activity
            # More accurate: IOB decreases as insulin is absorbed
            iob_fraction = 1.0 - (minutes_since / insulin_action_minutes)
            total_iob += insulin_amount * max(0, iob_fraction)

        iob_values.loc[idx] = total_iob

    return iob_values


def compute_per_timestep_cob(
    df: pd.DataFrame,
    treatments_df: Optional[pd.DataFrame] = None,
    absorption_minutes: int = 180,
    gi_based: bool = True
) -> pd.Series:
    """
    Compute Carbs On Board (COB) for each timestep using exponential decay.

    This is CRITICAL for training - each timestep must have accurate COB
    reflecting remaining carbs being absorbed at that moment.

    Args:
        df: DataFrame with 'timestamp' column
        treatments_df: DataFrame with 'timestamp', 'carbs', and optionally 'glycemicIndex'
        absorption_minutes: Base absorption time (modified by GI if gi_based=True)
        gi_based: If True, adjust absorption time based on glycemic index

    Returns:
        Series of COB values aligned to df index
    """
    cob_values = pd.Series(0.0, index=df.index)

    if treatments_df is None or treatments_df.empty:
        return cob_values

    # Ensure timestamp is datetime and timezone-naive for consistent comparison
    df_timestamps = pd.to_datetime(df['timestamp'])
    if df_timestamps.dt.tz is not None:
        df_timestamps = df_timestamps.dt.tz_localize(None)

    treatments = treatments_df.copy()
    if 'timestamp' not in treatments.columns:
        return cob_values

    treatments['timestamp'] = pd.to_datetime(treatments['timestamp'])
    if treatments['timestamp'].dt.tz is not None:
        treatments['timestamp'] = treatments['timestamp'].dt.tz_localize(None)

    # Filter to carb treatments
    if 'carbs' in treatments.columns:
        carb_treatments = treatments[treatments['carbs'] > 0].copy()
    else:
        return cob_values

    if carb_treatments.empty:
        return cob_values

    # For each glucose reading, calculate total COB from all active carbs
    for idx, row in df.iterrows():
        current_time = df_timestamps.loc[idx]
        total_cob = 0.0

        for _, treat in carb_treatments.iterrows():
            treat_time = treat['timestamp']
            carb_amount = treat['carbs']

            # Get glycemic index for absorption rate (default 55 = medium)
            gi = treat.get('glycemicIndex', 55) or 55

            # Adjust absorption time based on GI
            if gi_based:
                # High GI (>70): faster absorption (150 min)
                # Medium GI (55-70): normal (180 min)
                # Low GI (<55): slower absorption (240 min)
                if gi > 70:
                    carb_absorption = 150
                elif gi < 55:
                    carb_absorption = 240
                else:
                    carb_absorption = absorption_minutes
            else:
                carb_absorption = absorption_minutes

            # Time since meal in minutes
            minutes_since = (current_time - treat_time).total_seconds() / 60

            if minutes_since < 0 or minutes_since > carb_absorption:
                continue  # Not yet eaten or fully absorbed

            # Exponential decay model for carb absorption
            # COB = carbs * exp(-k * t) where k = ln(20) / absorption_time
            # This means 5% remaining at absorption_time
            decay_constant = np.log(20) / carb_absorption
            remaining_fraction = np.exp(-decay_constant * minutes_since)
            total_cob += carb_amount * remaining_fraction

        cob_values.loc[idx] = total_cob

    return cob_values


def engineer_extended_features(
    df_in: pd.DataFrame,
    treatments_df: Optional[pd.DataFrame] = None,
    ml_iob: float = 0.0,
    ml_cob: float = 0.0,
    isf: float = 50.0,
    icr: float = 10.0,
    compute_per_timestep: bool = True
) -> pd.DataFrame:
    """
    Engineer extended ~65 features for TFT model.

    This extends the basic features with:
    - Rate of change and acceleration
    - ML-based IOB/COB effects (per-timestep if compute_per_timestep=True)
    - Food features (GI, absorption rate)
    - Multi-window rolling statistics
    - Variability metrics
    - Event timing
    - Pattern detection

    CRITICAL: For training, set compute_per_timestep=True to get accurate
    IOB/COB for each timestep. For real-time inference with pre-calculated
    values, set compute_per_timestep=False.

    Args:
        df_in: DataFrame with timestamp, value, trend, carbs, protein, fat, insulin
        treatments_df: Optional treatments DataFrame for event timing
        ml_iob: Pre-calculated ML-based IOB (used if compute_per_timestep=False)
        ml_cob: Pre-calculated ML-based COB (used if compute_per_timestep=False)
        isf: Insulin sensitivity factor
        icr: Insulin to carb ratio
        compute_per_timestep: If True, compute IOB/COB for each row from treatments

    Returns:
        DataFrame with all ~65 features
    """
    # Start with basic features
    df = engineer_features(df_in)

    if df.empty:
        return df

    # --- Category 1: Core Glucose Features ---
    # Rate of change (mg/dL per 5 min)
    df['rate_of_change'] = df['value'].diff().fillna(0)

    # Acceleration (2nd derivative)
    df['acceleration'] = df['rate_of_change'].diff().fillna(0)

    # --- Category 2: ML-Based IOB/COB Features ---
    # CRITICAL: For training, compute per-timestep values from treatment history
    if compute_per_timestep and treatments_df is not None and not treatments_df.empty:
        logger.debug("Computing per-timestep IOB/COB from treatment history")
        df['ml_iob'] = compute_per_timestep_iob(df, treatments_df)
        df['ml_cob'] = compute_per_timestep_cob(df, treatments_df)
    else:
        # Use static values (for real-time inference)
        df['ml_iob'] = ml_iob
        df['ml_cob'] = ml_cob

    # Calculate effects based on IOB/COB
    df['ml_iob_effect'] = -(df['ml_iob'] * isf)  # Negative = lowering BG
    df['ml_cob_effect'] = (df['ml_cob'] / icr) * isf  # Positive = raising BG
    df['net_effect'] = df['ml_iob_effect'] + df['ml_cob_effect']

    # --- Category 3: Food Features ---
    # Default values - should be overridden with actual treatment data
    df['glycemic_index'] = 55 / 100.0  # Scaled 0-1
    df['glycemic_load'] = 0.0
    df['absorption_fast'] = 0
    df['absorption_medium'] = 1  # Default to medium
    df['absorption_slow'] = 0
    df['fat_high'] = 0

    # Update food features from treatment data if available
    if treatments_df is not None and not treatments_df.empty:
        df = _add_food_features_from_treatments(df, treatments_df)

    # --- Category 4: Macro Features (already have carbs, protein, fat) ---
    total_macros = df['carbs'] + df['protein'] + df['fat']
    df['carb_ratio'] = np.where(total_macros > 0, df['carbs'] / total_macros, 0)
    df['fat_ratio'] = np.where(total_macros > 0, df['fat'] / total_macros, 0)

    # --- Category 5: Multi-Window Rolling Statistics ---
    for window_min, window_steps in ROLL_WINDOWS.items():
        col_mean = f'roll{window_min}_mean'
        col_std = f'roll{window_min}_std'

        if col_mean not in df.columns:
            df[col_mean] = df['value'].rolling(window=window_steps, min_periods=1).mean()
        if col_std not in df.columns:
            df[col_std] = df['value'].rolling(window=window_steps, min_periods=1).std().fillna(0)

    # --- Category 6: Glucose Variability Metrics ---
    # 24-hour window for TIR calculations (288 readings at 5-min intervals)
    window_24h = min(288, len(df))

    # Coefficient of variation
    roll_mean = df['value'].rolling(window=window_24h, min_periods=1).mean()
    roll_std = df['value'].rolling(window=window_24h, min_periods=1).std().fillna(0)
    df['coefficient_of_variation'] = np.where(roll_mean > 0, (roll_std / roll_mean) * 100, 0)

    # Glucose range (max - min in window)
    df['glucose_range'] = (
        df['value'].rolling(window=window_24h, min_periods=1).max() -
        df['value'].rolling(window=window_24h, min_periods=1).min()
    ).fillna(0)

    # Time in range (70-180 mg/dL)
    in_range = (df['value'] >= 70) & (df['value'] <= 180)
    df['time_in_range'] = in_range.rolling(window=window_24h, min_periods=1).mean().fillna(0)

    # Time low (< 70)
    is_low = df['value'] < 70
    df['time_low'] = is_low.rolling(window=window_24h, min_periods=1).mean().fillna(0)

    # Time high (> 180)
    is_high = df['value'] > 180
    df['time_high'] = is_high.rolling(window=window_24h, min_periods=1).mean().fillna(0)

    # Consecutive readings in range
    df['consecutive_in_range'] = _count_consecutive(in_range)

    # --- Category 7: Event Timing Features ---
    df['minutes_since_meal'] = 360  # Default: 6 hours (fasting)
    df['minutes_since_bolus'] = 360
    df['minutes_since_correction'] = 360
    df['meals_today'] = 0
    df['total_carbs_today'] = 0
    df['total_insulin_today'] = 0

    if treatments_df is not None and not treatments_df.empty:
        df = _add_event_timing_features(df, treatments_df)

    # --- Category 8: Additional Cyclical Time Features ---
    dt_col = df['timestamp'].dt
    df['dom_sin'], df['dom_cos'] = sincos(dt_col.day, 31)  # Day of month

    # --- Category 9: Pattern Detection Features ---
    # Is fasting (no carbs in past 3 hours = 36 steps)
    carbs_window = 36
    df['is_fasting'] = (
        df['carbs'].rolling(window=carbs_window, min_periods=1).sum() == 0
    ).astype(int)

    # Is sleeping hours (11pm - 6am)
    df['is_sleeping_hours'] = ((df['hour'] >= 23) | (df['hour'] < 6)).astype(int)

    # Dawn phenomenon zone (4am - 8am)
    df['dawn_phenomenon_zone'] = ((df['hour'] >= 4) & (df['hour'] < 8)).astype(int)

    # Post-meal window (0-3 hours after meal)
    df['post_meal_window'] = (df['minutes_since_meal'] <= 180).astype(int)

    # Stacking risk (IOB > 3 units with recent bolus)
    df['stacking_risk'] = ((df['ml_iob'] > 3) & (df['minutes_since_bolus'] < 60)).astype(int)

    # --- Category 10: Derived Metabolic Features ---
    # These are placeholders - would need historical data to compute properly
    df['effective_carb_ratio'] = 1.0  # Recent carbs / insulin ratio
    df['recent_isf_observed'] = isf  # BG drop per unit (last 24h)

    # Insulin sensitivity zone (0=morning, 1=afternoon, 2=evening, 3=night)
    df['insulin_sensitivity_zone'] = df['hour'].apply(_get_isf_zone)

    # Carb absorption delay (estimated)
    df['carb_absorption_delay'] = 0  # Time to BG peak after meal

    # --- Category 13: Pump Features ---
    df['basal_rate'] = 0.0
    df['auto_correction_count'] = 0
    df['auto_correction_insulin'] = 0.0
    df['basal_deviation'] = 0.0
    df['is_pump_user'] = 0

    if treatments_df is not None and not treatments_df.empty:
        df = _add_pump_features(df, treatments_df)

    # Ensure all features exist with defaults
    for col in TFT_FEATURE_COLUMNS:
        if col not in df.columns:
            logger.warning(f"Missing TFT feature {col}, adding with default 0")
            df[col] = 0.0

    logger.debug(f"Extended feature engineering complete. {len(TFT_FEATURE_COLUMNS)} features created.")
    return df


def _count_consecutive(series: pd.Series) -> pd.Series:
    """Count consecutive True values in a boolean series."""
    # Group by changes and count within groups
    groups = (~series).cumsum()
    result = series.groupby(groups).cumcount() + 1
    result[~series] = 0
    return result


def _get_isf_zone(hour: int) -> int:
    """Map hour to ISF zone (insulin sensitivity varies by time of day)."""
    if 6 <= hour < 11:
        return 0  # Morning (often more resistant)
    elif 11 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 22:
        return 2  # Evening
    else:
        return 3  # Night


def _add_food_features_from_treatments(
    df: pd.DataFrame,
    treatments_df: pd.DataFrame
) -> pd.DataFrame:
    """Add food features from recent treatments to glucose DataFrame."""
    treatments_df = treatments_df.copy()
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)

    # Get most recent carb treatment for each glucose reading
    for idx, row in df.iterrows():
        glucose_time = row['timestamp']

        # Look back up to 3 hours for relevant carb treatments
        window_start = glucose_time - timedelta(hours=3)

        recent_carbs = treatments_df[
            (treatments_df['timestamp'] >= window_start) &
            (treatments_df['timestamp'] <= glucose_time) &
            (treatments_df.get('carbs', 0) > 0)
        ]

        if not recent_carbs.empty:
            latest_carb = recent_carbs.iloc[-1]

            # Update food features
            gi = latest_carb.get('glycemicIndex', 55) or 55
            carbs = latest_carb.get('carbs', 0) or 0
            fat = latest_carb.get('fat', 0) or 0
            absorption = latest_carb.get('absorptionRate', 'medium') or 'medium'

            df.loc[idx, 'glycemic_index'] = gi / 100.0
            df.loc[idx, 'glycemic_load'] = (carbs * gi / 100) / 50.0  # Scaled

            # One-hot absorption rate
            df.loc[idx, 'absorption_fast'] = 1 if absorption == 'fast' else 0
            df.loc[idx, 'absorption_medium'] = 1 if absorption == 'medium' else 0
            df.loc[idx, 'absorption_slow'] = 1 if absorption == 'slow' else 0

            # High fat indicator
            df.loc[idx, 'fat_high'] = 1 if fat > 15 else 0

    return df


def _add_event_timing_features(
    df: pd.DataFrame,
    treatments_df: pd.DataFrame
) -> pd.DataFrame:
    """Add event timing features from treatments."""
    treatments_df = treatments_df.copy()
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)

    for idx, row in df.iterrows():
        glucose_time = row['timestamp']
        glucose_date = glucose_time.date()

        # Minutes since last meal
        carb_treatments = treatments_df[
            (treatments_df['timestamp'] <= glucose_time) &
            ((treatments_df.get('carbs', 0) > 0) | (treatments_df.get('type') == 'carbs'))
        ]
        if not carb_treatments.empty:
            last_meal = carb_treatments['timestamp'].max()
            df.loc[idx, 'minutes_since_meal'] = (glucose_time - last_meal).total_seconds() / 60

        # Minutes since last bolus
        insulin_treatments = treatments_df[
            (treatments_df['timestamp'] <= glucose_time) &
            ((treatments_df.get('insulin', 0) > 0) | (treatments_df.get('type') == 'insulin'))
        ]
        if not insulin_treatments.empty:
            last_bolus = insulin_treatments['timestamp'].max()
            df.loc[idx, 'minutes_since_bolus'] = (glucose_time - last_bolus).total_seconds() / 60

            # Check if it was a correction (no carbs nearby)
            correction_window = timedelta(minutes=30)
            nearby_carbs = carb_treatments[
                (carb_treatments['timestamp'] >= last_bolus - correction_window) &
                (carb_treatments['timestamp'] <= last_bolus + correction_window)
            ]
            if nearby_carbs.empty:
                df.loc[idx, 'minutes_since_correction'] = df.loc[idx, 'minutes_since_bolus']

        # Daily totals
        today_treatments = treatments_df[
            treatments_df['timestamp'].dt.date == glucose_date
        ]

        df.loc[idx, 'meals_today'] = len(today_treatments[
            (today_treatments.get('carbs', 0) > 0) | (today_treatments.get('type') == 'carbs')
        ])

        carbs_col = today_treatments.get('carbs', pd.Series([0]))
        df.loc[idx, 'total_carbs_today'] = carbs_col.fillna(0).sum()

        insulin_col = today_treatments.get('insulin', pd.Series([0]))
        df.loc[idx, 'total_insulin_today'] = insulin_col.fillna(0).sum()

    return df


def _add_pump_features(
    df: pd.DataFrame,
    treatments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add pump-specific features from treatment data.

    For non-pump users, all pump features remain 0.0 (safe default).
    Detects pump usage from deliveryMethod field on treatments.
    """
    treatments_df = treatments_df.copy()
    treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)

    # Check if user has pump data
    has_delivery_method = 'deliveryMethod' in treatments_df.columns
    if not has_delivery_method:
        return df

    pump_methods = {'pump_basal', 'pump_bolus', 'pump_auto_correction'}
    pump_treatments = treatments_df[
        treatments_df['deliveryMethod'].isin(pump_methods)
    ] if has_delivery_method else pd.DataFrame()

    if pump_treatments.empty:
        return df

    # Mark as pump user
    df['is_pump_user'] = 1

    # Get basal and auto-correction subsets
    basal_treatments = pump_treatments[
        pump_treatments['deliveryMethod'] == 'pump_basal'
    ].copy()
    auto_corrections = pump_treatments[
        pump_treatments['deliveryMethod'] == 'pump_auto_correction'
    ].copy()

    # Calculate historical median basal rate for deviation feature
    median_basal_rate = 0.0
    if not basal_treatments.empty and 'basalRate' in basal_treatments.columns:
        rates = basal_treatments['basalRate'].dropna()
        if not rates.empty:
            median_basal_rate = float(rates.median())

    for idx, row in df.iterrows():
        glucose_time = row['timestamp']

        # 1. Basal rate — most recent basal rate before this timestamp
        recent_basal = basal_treatments[basal_treatments['timestamp'] <= glucose_time]
        if not recent_basal.empty and 'basalRate' in recent_basal.columns:
            latest = recent_basal.iloc[-1]
            rate = latest.get('basalRate', 0) or 0
            df.loc[idx, 'basal_rate'] = rate
            # Deviation from historical median
            if median_basal_rate > 0:
                df.loc[idx, 'basal_deviation'] = rate - median_basal_rate

        # 2. Rolling 1-hour auto-correction count and insulin sum
        window_start = glucose_time - timedelta(hours=1)
        recent_auto = auto_corrections[
            (auto_corrections['timestamp'] >= window_start) &
            (auto_corrections['timestamp'] <= glucose_time)
        ]
        if not recent_auto.empty:
            df.loc[idx, 'auto_correction_count'] = len(recent_auto)
            insulin_col = recent_auto.get('insulin', pd.Series([0]))
            df.loc[idx, 'auto_correction_insulin'] = insulin_col.fillna(0).sum()

    return df


def extract_tft_feature_sequence(
    df: pd.DataFrame,
    seq_length: int = SEQ_LENGTH
) -> Optional[np.ndarray]:
    """
    Extract feature sequence for TFT model.

    Args:
        df: DataFrame with extended features
        seq_length: Sequence length

    Returns:
        Numpy array of shape (1, seq_length, n_features) or None
    """
    available_features = [col for col in TFT_FEATURE_COLUMNS if col in df.columns]

    if len(available_features) < len(TFT_FEATURE_COLUMNS):
        missing = set(TFT_FEATURE_COLUMNS) - set(available_features)
        logger.warning(f"Missing TFT features: {missing}")

        # Add missing features with defaults
        for col in missing:
            df[col] = 0.0
        available_features = TFT_FEATURE_COLUMNS

    if len(df) < seq_length:
        logger.warning(f"Not enough data for TFT sequence. Have {len(df)}, need {seq_length}")
        return None

    sequence = df[available_features].tail(seq_length).values.astype(np.float32)
    return sequence.reshape(1, seq_length, len(available_features))
