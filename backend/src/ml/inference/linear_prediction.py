"""
Linear Prediction Module
Simple linear extrapolation for BG predictions when LSTM is unavailable.

Ported from dexcom_reader_predict.py linear prediction logic.
"""
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LinearPredictor:
    """
    Linear extrapolation predictor for blood glucose.

    Uses recent glucose values to fit a line and extrapolate
    future values. Serves as fallback when LSTM is unavailable.
    """

    def __init__(
        self,
        history_points: int = 6,  # 30 min of history at 5-min intervals
        prediction_horizons: List[int] = None
    ):
        """
        Initialize linear predictor.

        Args:
            history_points: Number of recent points to use for fitting
            prediction_horizons: Minutes ahead to predict (default: [5, 10, 15])
        """
        self.history_points = history_points
        self.prediction_horizons = prediction_horizons or [5, 10, 15]

    def predict(
        self,
        glucose_values: List[float],
        timestamps: Optional[List[datetime]] = None,
        interval_min: int = 5
    ) -> Tuple[List[float], float, float]:
        """
        Predict future glucose values using linear extrapolation.

        Args:
            glucose_values: Recent glucose values (newest last)
            timestamps: Optional timestamps for each value
            interval_min: Time interval between readings in minutes

        Returns:
            Tuple of (predictions, slope, intercept)
            Predictions are for +5, +10, +15 minutes.
        """
        if len(glucose_values) < 2:
            logger.warning("Not enough glucose values for linear prediction")
            # Return last value as all predictions
            last_val = glucose_values[-1] if glucose_values else 100.0
            return [last_val] * len(self.prediction_horizons), 0.0, last_val

        # Use last N points for fitting
        recent_values = glucose_values[-self.history_points:]
        n_points = len(recent_values)

        # Create time indices (in minutes from first point)
        if timestamps and len(timestamps) >= n_points:
            # Use actual timestamps
            recent_ts = timestamps[-n_points:]
            base_time = recent_ts[0]
            x = np.array([
                (t - base_time).total_seconds() / 60.0
                for t in recent_ts
            ])
        else:
            # Assume fixed intervals
            x = np.arange(n_points) * interval_min

        y = np.array(recent_values)

        # Fit linear regression
        try:
            # Using numpy polyfit for robustness
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]  # mg/dL per minute
            intercept = coeffs[1]

        except Exception as e:
            logger.warning(f"Linear fit failed: {e}, using simple average")
            slope = 0.0
            intercept = np.mean(y)

        # Current time is at the end of the series
        current_x = x[-1]

        # Predict at each horizon
        predictions = []
        for horizon_min in self.prediction_horizons:
            future_x = current_x + horizon_min
            predicted_value = slope * future_x + intercept

            # Clamp to reasonable range
            predicted_value = max(40.0, min(400.0, predicted_value))
            predictions.append(float(predicted_value))

        return predictions, float(slope), float(intercept)

    def predict_with_trend(
        self,
        current_bg: float,
        trend: int,
        current_time: Optional[datetime] = None
    ) -> List[float]:
        """
        Predict using current BG and CGM trend arrow.

        This is a simpler prediction method using trend arrows.

        Trend values:
            -3: DoubleDown (~-3 mg/dL per minute)
            -2: SingleDown (~-2 mg/dL per minute)
            -1: FortyFiveDown (~-1 mg/dL per minute)
             0: Flat (~0 mg/dL per minute)
            +1: FortyFiveUp (~+1 mg/dL per minute)
            +2: SingleUp (~+2 mg/dL per minute)
            +3: DoubleUp (~+3 mg/dL per minute)

        Args:
            current_bg: Current glucose value in mg/dL
            trend: Trend arrow value (-3 to +3)
            current_time: Optional current timestamp

        Returns:
            List of predictions for configured horizons
        """
        # Approximate rate of change based on trend
        # These values are approximate mg/dL per minute
        trend_rates = {
            -3: -3.0,   # DoubleDown
            -2: -2.0,   # SingleDown
            -1: -1.0,   # FortyFiveDown
            0: 0.0,     # Flat
            1: 1.0,     # FortyFiveUp
            2: 2.0,     # SingleUp
            3: 3.0,     # DoubleUp
        }

        rate = trend_rates.get(int(trend), 0.0)

        predictions = []
        for horizon_min in self.prediction_horizons:
            predicted = current_bg + (rate * horizon_min)
            # Clamp to reasonable range
            predicted = max(40.0, min(400.0, predicted))
            predictions.append(float(predicted))

        return predictions

    def get_trend_from_values(
        self,
        glucose_values: List[float],
        interval_min: int = 5
    ) -> Tuple[int, float]:
        """
        Calculate trend arrow from recent glucose values.

        Args:
            glucose_values: Recent glucose values (newest last)
            interval_min: Time interval between readings

        Returns:
            Tuple of (trend_int, rate_per_min)
        """
        if len(glucose_values) < 2:
            return 0, 0.0

        # Use last 3 values (15 min) for trend calculation
        recent = glucose_values[-3:] if len(glucose_values) >= 3 else glucose_values
        n = len(recent)

        # Calculate average rate of change
        total_change = recent[-1] - recent[0]
        total_time = (n - 1) * interval_min
        rate = total_change / total_time if total_time > 0 else 0.0

        # Convert rate to trend arrow
        if rate <= -3.0:
            trend = -3
        elif rate <= -2.0:
            trend = -2
        elif rate <= -1.0:
            trend = -1
        elif rate < 1.0:
            trend = 0
        elif rate < 2.0:
            trend = 1
        elif rate < 3.0:
            trend = 2
        else:
            trend = 3

        return trend, rate


def calculate_prediction_accuracy(
    predictions: List[float],
    actuals: List[float]
) -> dict:
    """
    Calculate prediction accuracy metrics.

    Args:
        predictions: Predicted values
        actuals: Actual observed values

    Returns:
        Dictionary with accuracy metrics
    """
    if not predictions or not actuals:
        return {"mae": None, "rmse": None, "mape": None}

    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    preds = np.array(predictions[:min_len])
    acts = np.array(actuals[:min_len])

    # Calculate errors
    errors = preds - acts
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_errors = np.abs(errors / acts) * 100
        percentage_errors = np.where(np.isinf(percentage_errors), np.nan, percentage_errors)
        mape = float(np.nanmean(percentage_errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "n_samples": min_len
    }


def compare_predictions(
    linear_preds: List[float],
    lstm_preds: List[float],
    actuals: List[float]
) -> dict:
    """
    Compare linear vs LSTM prediction accuracy.

    Args:
        linear_preds: Linear model predictions
        lstm_preds: LSTM model predictions
        actuals: Actual observed values

    Returns:
        Dictionary with comparison metrics
    """
    linear_accuracy = calculate_prediction_accuracy(linear_preds, actuals)
    lstm_accuracy = calculate_prediction_accuracy(lstm_preds, actuals)

    # Determine winner
    linear_mae = linear_accuracy.get("mae", float("inf"))
    lstm_mae = lstm_accuracy.get("mae", float("inf"))

    if linear_mae is None:
        linear_mae = float("inf")
    if lstm_mae is None:
        lstm_mae = float("inf")

    winner = "linear" if linear_mae < lstm_mae else "lstm"
    improvement = abs(linear_mae - lstm_mae)

    return {
        "linear": linear_accuracy,
        "lstm": lstm_accuracy,
        "winner": winner,
        "improvement_mg_dl": improvement,
        "improvement_pct": (improvement / max(linear_mae, 0.001)) * 100
    }
