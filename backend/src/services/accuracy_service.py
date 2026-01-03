"""
Prediction Accuracy Tracking Service
Tracks and compares linear vs LSTM prediction accuracy.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from collections import deque

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a prediction for accuracy tracking."""
    timestamp: datetime
    current_bg: float
    linear_5: float
    linear_10: float
    linear_15: float
    lstm_5: Optional[float]
    lstm_10: Optional[float]
    lstm_15: Optional[float]
    actual_5: Optional[float] = None
    actual_10: Optional[float] = None
    actual_15: Optional[float] = None
    evaluated: bool = False


class AccuracyMetrics(BaseModel):
    """Accuracy metrics for a prediction method."""
    mae_5: float = 0.0  # Mean Absolute Error at 5 min
    mae_10: float = 0.0
    mae_15: float = 0.0
    mae_overall: float = 0.0
    rmse_5: float = 0.0  # Root Mean Square Error
    rmse_10: float = 0.0
    rmse_15: float = 0.0
    rmse_overall: float = 0.0
    sample_count: int = 0
    within_10_pct: float = 0.0  # % of predictions within 10 mg/dL
    within_20_pct: float = 0.0  # % within 20 mg/dL


class AccuracyComparison(BaseModel):
    """Comparison between linear and LSTM predictions."""
    linear: AccuracyMetrics
    lstm: Optional[AccuracyMetrics] = None
    winner: str = "linear"
    linear_wins: int = 0
    lstm_wins: int = 0
    total_comparisons: int = 0
    period_hours: int = 24
    timestamp: datetime


class AccuracyTracker:
    """
    Tracks prediction accuracy over time.

    Stores predictions and evaluates them against actual values
    when they become available.
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._predictions: deque[PredictionRecord] = deque(maxlen=max_records)
        self._linear_errors: List[Tuple[float, float, float]] = []  # (5m, 10m, 15m)
        self._lstm_errors: List[Tuple[float, float, float]] = []
        self._linear_wins = 0
        self._lstm_wins = 0

    def record_prediction(
        self,
        current_bg: float,
        linear: List[float],
        lstm: Optional[List[float]] = None
    ) -> None:
        """
        Record a new prediction for later evaluation.

        Args:
            current_bg: Current BG when prediction was made
            linear: Linear predictions [5m, 10m, 15m]
            lstm: LSTM predictions [5m, 10m, 15m] or None
        """
        record = PredictionRecord(
            timestamp=datetime.utcnow(),
            current_bg=current_bg,
            linear_5=linear[0],
            linear_10=linear[1],
            linear_15=linear[2],
            lstm_5=lstm[0] if lstm else None,
            lstm_10=lstm[1] if lstm else None,
            lstm_15=lstm[2] if lstm else None
        )
        self._predictions.append(record)

    def record_actual(self, timestamp: datetime, value: float) -> None:
        """
        Record an actual glucose value and evaluate pending predictions.

        Args:
            timestamp: When the reading was taken
            value: Actual glucose value
        """
        now = timestamp

        for record in self._predictions:
            if record.evaluated:
                continue

            time_diff = (now - record.timestamp).total_seconds() / 60

            # Check if this reading matches a prediction horizon
            if 4 <= time_diff <= 6 and record.actual_5 is None:
                record.actual_5 = value
            elif 9 <= time_diff <= 11 and record.actual_10 is None:
                record.actual_10 = value
            elif 14 <= time_diff <= 16 and record.actual_15 is None:
                record.actual_15 = value

            # Mark as evaluated if all actuals are filled
            if (record.actual_5 is not None and
                record.actual_10 is not None and
                record.actual_15 is not None):
                record.evaluated = True
                self._evaluate_record(record)

    def _evaluate_record(self, record: PredictionRecord) -> None:
        """Evaluate a completed prediction record."""
        # Calculate linear errors
        linear_err_5 = abs(record.linear_5 - record.actual_5)
        linear_err_10 = abs(record.linear_10 - record.actual_10)
        linear_err_15 = abs(record.linear_15 - record.actual_15)
        self._linear_errors.append((linear_err_5, linear_err_10, linear_err_15))

        linear_avg = (linear_err_5 + linear_err_10 + linear_err_15) / 3

        # Calculate LSTM errors if available
        if record.lstm_5 is not None:
            lstm_err_5 = abs(record.lstm_5 - record.actual_5)
            lstm_err_10 = abs(record.lstm_10 - record.actual_10)
            lstm_err_15 = abs(record.lstm_15 - record.actual_15)
            self._lstm_errors.append((lstm_err_5, lstm_err_10, lstm_err_15))

            lstm_avg = (lstm_err_5 + lstm_err_10 + lstm_err_15) / 3

            # Determine winner
            if lstm_avg < linear_avg:
                self._lstm_wins += 1
            else:
                self._linear_wins += 1

        # Trim error lists
        max_errors = 500
        if len(self._linear_errors) > max_errors:
            self._linear_errors = self._linear_errors[-max_errors:]
        if len(self._lstm_errors) > max_errors:
            self._lstm_errors = self._lstm_errors[-max_errors:]

    def get_accuracy(self, hours: int = 24) -> AccuracyComparison:
        """
        Get accuracy metrics for the specified period.

        Args:
            hours: Hours of history to analyze

        Returns:
            AccuracyComparison with linear and LSTM metrics
        """
        linear_metrics = self._calculate_metrics(self._linear_errors)
        lstm_metrics = None

        if self._lstm_errors:
            lstm_metrics = self._calculate_metrics(self._lstm_errors)

        winner = "linear"
        if lstm_metrics and lstm_metrics.mae_overall < linear_metrics.mae_overall:
            winner = "lstm"

        return AccuracyComparison(
            linear=linear_metrics,
            lstm=lstm_metrics,
            winner=winner,
            linear_wins=self._linear_wins,
            lstm_wins=self._lstm_wins,
            total_comparisons=self._linear_wins + self._lstm_wins,
            period_hours=hours,
            timestamp=datetime.utcnow()
        )

    def _calculate_metrics(
        self,
        errors: List[Tuple[float, float, float]]
    ) -> AccuracyMetrics:
        """Calculate accuracy metrics from error list."""
        if not errors:
            return AccuracyMetrics()

        import math

        n = len(errors)
        errors_5 = [e[0] for e in errors]
        errors_10 = [e[1] for e in errors]
        errors_15 = [e[2] for e in errors]
        all_errors = errors_5 + errors_10 + errors_15

        mae_5 = sum(errors_5) / len(errors_5)
        mae_10 = sum(errors_10) / len(errors_10)
        mae_15 = sum(errors_15) / len(errors_15)
        mae_overall = sum(all_errors) / len(all_errors)

        rmse_5 = math.sqrt(sum(e**2 for e in errors_5) / len(errors_5))
        rmse_10 = math.sqrt(sum(e**2 for e in errors_10) / len(errors_10))
        rmse_15 = math.sqrt(sum(e**2 for e in errors_15) / len(errors_15))
        rmse_overall = math.sqrt(sum(e**2 for e in all_errors) / len(all_errors))

        within_10 = sum(1 for e in all_errors if e <= 10) / len(all_errors) * 100
        within_20 = sum(1 for e in all_errors if e <= 20) / len(all_errors) * 100

        return AccuracyMetrics(
            mae_5=round(mae_5, 1),
            mae_10=round(mae_10, 1),
            mae_15=round(mae_15, 1),
            mae_overall=round(mae_overall, 1),
            rmse_5=round(rmse_5, 1),
            rmse_10=round(rmse_10, 1),
            rmse_15=round(rmse_15, 1),
            rmse_overall=round(rmse_overall, 1),
            sample_count=n,
            within_10_pct=round(within_10, 1),
            within_20_pct=round(within_20, 1)
        )

    def get_recent_predictions(self, limit: int = 10) -> List[dict]:
        """Get recent predictions with their evaluation status."""
        results = []
        for record in list(self._predictions)[-limit:]:
            results.append({
                "timestamp": record.timestamp.isoformat(),
                "current_bg": record.current_bg,
                "linear": [record.linear_5, record.linear_10, record.linear_15],
                "lstm": [record.lstm_5, record.lstm_10, record.lstm_15] if record.lstm_5 else None,
                "actual": [record.actual_5, record.actual_10, record.actual_15] if record.evaluated else None,
                "evaluated": record.evaluated
            })
        return results


# Singleton tracker
_accuracy_tracker: Optional[AccuracyTracker] = None


def get_accuracy_tracker() -> AccuracyTracker:
    """Get or create the global accuracy tracker."""
    global _accuracy_tracker
    if _accuracy_tracker is None:
        _accuracy_tracker = AccuracyTracker()
    return _accuracy_tracker
