"""
Prediction Tracker Service
Tracks predictions vs actual glucose readings for model evaluation.
Integrates with MLflow for persistent tracking.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class PendingPrediction:
    """A prediction waiting for actual value comparison."""
    user_id: str
    prediction_time: datetime  # When the prediction was made
    target_time: datetime  # When we expect the actual reading
    horizon_min: int
    model_type: str  # 'linear', 'lstm', 'tft'
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


@dataclass
class AccuracyResult:
    """Result of comparing prediction to actual."""
    user_id: str
    model_type: str
    horizon_min: int
    predicted: float
    actual: float
    error: float  # predicted - actual
    abs_error: float
    percentage_error: float
    within_bounds: Optional[bool] = None  # For TFT predictions with bounds
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PredictionTracker:
    """
    Track predictions vs actuals for model evaluation.

    Stores pending predictions and compares them to actual glucose
    readings when they arrive. Aggregates accuracy metrics and
    optionally logs to MLflow.
    """

    def __init__(self, mlflow_enabled: bool = False):
        """
        Initialize the prediction tracker.

        Args:
            mlflow_enabled: Whether to log metrics to MLflow
        """
        self.mlflow_enabled = mlflow_enabled
        self._mlflow_tracker = None

        # Pending predictions keyed by (user_id, target_time_bucket)
        # Time bucket = timestamp rounded to nearest 5 minutes
        self._pending: Dict[str, List[PendingPrediction]] = defaultdict(list)

        # Completed accuracy results by model type
        self._results: Dict[str, List[AccuracyResult]] = defaultdict(list)

        # Rolling accuracy metrics (last 100 predictions per model)
        self._rolling_window = 100

        if mlflow_enabled:
            try:
                from ml.mlflow_tracking import ModelTracker
                self._mlflow_tracker = ModelTracker(model_type="prediction_eval")
            except Exception as e:
                logger.warning(f"MLflow tracking disabled: {e}")
                self.mlflow_enabled = False

    def _time_bucket(self, dt: datetime) -> str:
        """Round datetime to nearest 5-minute bucket for matching."""
        # Round to nearest 5 minutes
        minute = (dt.minute // 5) * 5
        bucket_time = dt.replace(minute=minute, second=0, microsecond=0)
        return bucket_time.isoformat()

    def log_prediction(
        self,
        user_id: str,
        prediction_time: datetime,
        model_type: str,
        horizon_min: int,
        predicted_value: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> None:
        """
        Log a prediction to be compared with actual later.

        Args:
            user_id: User ID
            prediction_time: When the prediction was made
            model_type: Model type ('linear', 'lstm', 'tft')
            horizon_min: Prediction horizon in minutes
            predicted_value: Predicted glucose value
            lower_bound: Lower uncertainty bound (for TFT)
            upper_bound: Upper uncertainty bound (for TFT)
        """
        target_time = prediction_time + timedelta(minutes=horizon_min)
        bucket = self._time_bucket(target_time)

        pred = PendingPrediction(
            user_id=user_id,
            prediction_time=prediction_time,
            target_time=target_time,
            horizon_min=horizon_min,
            model_type=model_type,
            predicted_value=predicted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        key = f"{user_id}_{bucket}"
        self._pending[key].append(pred)

        # Clean up old pending predictions (> 3 hours old)
        self._cleanup_old_pending()

        logger.debug(
            f"Logged {model_type} prediction for {user_id}: "
            f"+{horizon_min}min = {predicted_value:.1f}"
        )

    def log_predictions_batch(
        self,
        user_id: str,
        prediction_time: datetime,
        linear: List[float],
        lstm: Optional[List[float]] = None,
        tft: Optional[List[Dict]] = None,
    ) -> None:
        """
        Log a batch of predictions from a single inference call.

        Args:
            user_id: User ID
            prediction_time: When predictions were made
            linear: Linear predictions [+5, +10, +15]
            lstm: LSTM predictions [+5, +10, +15] (optional)
            tft: TFT predictions with horizons and bounds (optional)
        """
        # Log linear predictions
        for i, horizon in enumerate([5, 10, 15]):
            if i < len(linear):
                self.log_prediction(
                    user_id=user_id,
                    prediction_time=prediction_time,
                    model_type="linear",
                    horizon_min=horizon,
                    predicted_value=linear[i],
                )

        # Log LSTM predictions
        if lstm:
            for i, horizon in enumerate([5, 10, 15]):
                if i < len(lstm):
                    self.log_prediction(
                        user_id=user_id,
                        prediction_time=prediction_time,
                        model_type="lstm",
                        horizon_min=horizon,
                        predicted_value=lstm[i],
                    )

        # Log TFT predictions
        if tft:
            for pred in tft:
                self.log_prediction(
                    user_id=user_id,
                    prediction_time=prediction_time,
                    model_type="tft",
                    horizon_min=pred.get("horizon_min", 30),
                    predicted_value=pred.get("value", 0),
                    lower_bound=pred.get("lower"),
                    upper_bound=pred.get("upper"),
                )

    def compare_with_actual(
        self,
        user_id: str,
        actual_timestamp: datetime,
        actual_bg: float,
    ) -> List[AccuracyResult]:
        """
        Compare pending predictions against actual BG when it arrives.

        Args:
            user_id: User ID
            actual_timestamp: Timestamp of actual reading
            actual_bg: Actual glucose value

        Returns:
            List of AccuracyResult for matched predictions
        """
        bucket = self._time_bucket(actual_timestamp)
        key = f"{user_id}_{bucket}"

        if key not in self._pending:
            return []

        results = []
        matched_indices = []

        for i, pred in enumerate(self._pending[key]):
            # Check if this prediction is close enough to the actual time
            time_diff = abs((actual_timestamp - pred.target_time).total_seconds())
            if time_diff > 150:  # More than 2.5 minutes off
                continue

            matched_indices.append(i)

            error = pred.predicted_value - actual_bg
            abs_error = abs(error)
            pct_error = (abs_error / actual_bg * 100) if actual_bg > 0 else 0

            within_bounds = None
            if pred.lower_bound is not None and pred.upper_bound is not None:
                within_bounds = pred.lower_bound <= actual_bg <= pred.upper_bound

            result = AccuracyResult(
                user_id=user_id,
                model_type=pred.model_type,
                horizon_min=pred.horizon_min,
                predicted=pred.predicted_value,
                actual=actual_bg,
                error=error,
                abs_error=abs_error,
                percentage_error=pct_error,
                within_bounds=within_bounds,
            )

            results.append(result)
            self._results[pred.model_type].append(result)

            # Trim to rolling window
            if len(self._results[pred.model_type]) > self._rolling_window:
                self._results[pred.model_type] = self._results[pred.model_type][-self._rolling_window:]

            logger.debug(
                f"{pred.model_type} +{pred.horizon_min}min: "
                f"predicted={pred.predicted_value:.1f}, "
                f"actual={actual_bg:.1f}, "
                f"error={error:.1f}"
            )

        # Remove matched predictions
        for i in sorted(matched_indices, reverse=True):
            del self._pending[key][i]

        # Log to MLflow if enabled
        if self.mlflow_enabled and results and self._mlflow_tracker:
            self._log_to_mlflow(results)

        return results

    def _log_to_mlflow(self, results: List[AccuracyResult]) -> None:
        """Log accuracy results to MLflow."""
        if not self._mlflow_tracker:
            return

        try:
            # Start a run if not already active
            if not self._mlflow_tracker.run_id:
                self._mlflow_tracker.start_run(run_name="prediction_tracking")

            # Aggregate by model type and horizon
            metrics = {}
            for result in results:
                key = f"{result.model_type}_{result.horizon_min}min"
                metrics[f"abs_error_{key}"] = result.abs_error
                metrics[f"pct_error_{key}"] = result.percentage_error

                if result.within_bounds is not None:
                    metrics[f"within_bounds_{key}"] = 1.0 if result.within_bounds else 0.0

            self._mlflow_tracker.log_metrics(metrics)

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def _cleanup_old_pending(self, max_age_hours: int = 3) -> None:
        """Remove predictions older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        keys_to_remove = []

        for key, predictions in self._pending.items():
            self._pending[key] = [
                p for p in predictions
                if p.prediction_time > cutoff
            ]
            if not self._pending[key]:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._pending[key]

    def get_accuracy_stats(
        self,
        model_type: Optional[str] = None,
        horizon_min: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get accuracy statistics for predictions.

        Args:
            model_type: Filter by model type (None for all)
            horizon_min: Filter by horizon (None for all)

        Returns:
            Dictionary with accuracy statistics
        """
        results = []

        if model_type:
            results = self._results.get(model_type, [])
        else:
            for r_list in self._results.values():
                results.extend(r_list)

        if horizon_min:
            results = [r for r in results if r.horizon_min == horizon_min]

        if not results:
            return {
                "count": 0,
                "mae": None,
                "rmse": None,
                "mape": None,
                "coverage_80": None,
            }

        errors = [r.abs_error for r in results]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        mape = sum(r.percentage_error for r in results) / len(results)

        # Coverage for TFT predictions
        tft_results = [r for r in results if r.within_bounds is not None]
        coverage = None
        if tft_results:
            coverage = sum(1 for r in tft_results if r.within_bounds) / len(tft_results)

        return {
            "count": len(results),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "coverage_80": round(coverage, 3) if coverage else None,
        }

    def get_model_comparison(self) -> Dict[str, Dict]:
        """
        Compare accuracy across all model types.

        Returns:
            Dictionary with stats for each model type
        """
        comparison = {}
        for model_type in ["linear", "lstm", "tft"]:
            comparison[model_type] = self.get_accuracy_stats(model_type=model_type)
        return comparison

    def close(self) -> None:
        """Clean up resources and end MLflow run if active."""
        if self._mlflow_tracker and self._mlflow_tracker.run_id:
            self._mlflow_tracker.end_run()


# Singleton instance
_prediction_tracker: Optional[PredictionTracker] = None


def get_prediction_tracker(mlflow_enabled: bool = False) -> PredictionTracker:
    """Get or create the global prediction tracker."""
    global _prediction_tracker
    if _prediction_tracker is None:
        _prediction_tracker = PredictionTracker(mlflow_enabled=mlflow_enabled)
    return _prediction_tracker
