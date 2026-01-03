"""
Feature Engineering Module
Ported from train_bg_predictor.py and dexcom_reader_predict.py

Creates all features required for BG and ISF prediction models.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# Feature configuration matching the trained models
SAMPLING_MIN = 5  # Data granularity in minutes
POLY_DEGREE = 2
ROLL_WINDOW_SHORT = 6   # 30 min / 5 min = 6 steps
ROLL_WINDOW_LONG = 18   # 90 min / 5 min = 18 steps
POLY_WINDOW_SIZE = 24   # 120 min / 5 min = 24 steps
SEQ_LENGTH = 24         # 120 min / 5 min = 24 steps for LSTM input

# Feature columns matching bg_feature_list_v2.pkl
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

    # Parse timestamp
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

    # Convert to DataFrame
    df = pd.DataFrame(glucose_readings)

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

            if treat.get('type') == 'insulin' or treat.get('insulin', 0) > 0:
                df.loc[closest_idx, 'insulin'] += treat.get('insulin', 0)
            elif treat.get('type') == 'carbs' or treat.get('carbs', 0) > 0:
                df.loc[closest_idx, 'carbs'] += treat.get('carbs', 0)
                df.loc[closest_idx, 'protein'] += treat.get('protein', 0)
                df.loc[closest_idx, 'fat'] += treat.get('fat', 0)

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
        cutoff = datetime.utcnow() - timedelta(minutes=self.history_min)
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
        if len(self._glucose_buffer) < seq_length:
            logger.warning(
                f"Not enough glucose readings. Have {len(self._glucose_buffer)}, "
                f"need {seq_length}"
            )
            return None

        # Prepare DataFrame from buffer
        df = prepare_realtime_features(
            self._glucose_buffer,
            self._treatment_buffer,
            iob_value
        )

        if df is None:
            return None

        # Engineer features
        df = engineer_features(df)

        if df.empty:
            return None

        # Extract sequence
        return extract_feature_sequence(df, BG_FEATURE_COLUMNS, seq_length)

    def clear(self) -> None:
        """Clear all buffers."""
        self._glucose_buffer.clear()
        self._treatment_buffer.clear()
