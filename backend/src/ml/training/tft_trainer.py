"""
Comprehensive TFT Training Pipeline for T1D-AI

This module provides a production-ready training system for the Temporal Fusion
Transformer (TFT) glucose prediction model. Key features:

1. Proper scaler management (saved with model checkpoint)
2. Per-timestep IOB/COB computation (not static values)
3. Data quality filtering (excludes periods without treatments)
4. Temporal-aware train/val/test splitting (no data leakage)
5. Weighted quantile loss (prioritizes hypoglycemia detection)
6. Comprehensive evaluation metrics with backtesting
7. MLflow integration for experiment tracking

Usage:
    from ml.training.tft_trainer import TFTTrainingPipeline

    pipeline = TFTTrainingPipeline(
        user_id="user123",
        data_days=90,
        device="cuda"
    )

    # Load and prepare data
    await pipeline.load_data()

    # Train with cross-validation
    results = pipeline.train(epochs=100, patience=15)

    # Evaluate and save
    metrics = pipeline.evaluate()
    pipeline.save_checkpoint("models/tft_v2.pth")
"""
import logging
import pickle
import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TimeExclusionPattern:
    """
    Define a time-of-day pattern to exclude from training.

    Useful for excluding periods with known missing data:
    - School hours when treatments aren't logged
    - Work hours
    - Specific days of the week
    """
    name: str  # e.g., "school_hours", "work_hours"
    start_hour: int  # 0-23
    end_hour: int  # 0-23
    days_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday. None = all days
    enabled: bool = True

    def is_excluded(self, dt: datetime) -> bool:
        """Check if a datetime falls within this exclusion pattern."""
        if not self.enabled:
            return False

        # Check day of week
        if self.days_of_week is not None:
            if dt.weekday() not in self.days_of_week:
                return False

        hour = dt.hour

        # Handle patterns that span midnight (e.g., 23:00-06:00)
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:
            return hour >= self.start_hour or hour < self.end_hour


@dataclass
class TFTTrainingConfig:
    """Configuration for TFT training pipeline."""

    # Model architecture
    n_features: int = 69
    hidden_size: int = 64
    n_heads: int = 4
    n_lstm_layers: int = 2
    dropout: float = 0.1

    # Sequence configuration
    encoder_length: int = 24  # 120 min history (24 × 5 min)
    prediction_length: int = 12  # 60 min prediction (12 × 5 min)
    horizons_minutes: List[int] = field(default_factory=lambda: [15, 30, 45, 60, 90, 120])

    # Quantile configuration
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    quantile_weights: List[float] = field(default_factory=lambda: [1.5, 1.0, 0.8])
    # Higher weight on lower quantile for hypoglycemia detection

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    grad_clip: float = 1.0
    weight_decay: float = 1e-5

    # Data quality thresholds
    min_completeness_score: float = 0.7
    max_glucose_gap_minutes: int = 15  # Max gap between readings
    min_treatments_per_day: float = 2.0  # Minimum insulin/carb events

    # Time-of-day exclusion patterns
    # Example: Exclude school hours (8am-3pm) on weekdays where treatments may not be logged
    time_exclusion_patterns: List[TimeExclusionPattern] = field(default_factory=list)

    # Validation
    val_split: float = 0.15
    test_split: float = 0.15
    n_cv_folds: int = 5

    # Feature engineering
    use_weather: bool = False
    use_inferred_treatments: bool = False

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert TimeExclusionPattern objects to dicts for serialization
        result['time_exclusion_patterns'] = [
            {'name': p.name, 'start_hour': p.start_hour, 'end_hour': p.end_hour,
             'days_of_week': p.days_of_week, 'enabled': p.enabled}
            for p in self.time_exclusion_patterns
        ]
        return result

    @staticmethod
    def with_school_hours_excluded(
        school_start: int = 8,
        school_end: int = 15,
        weekdays_only: bool = True
    ) -> 'TFTTrainingConfig':
        """
        Create a config with school hours excluded.

        Args:
            school_start: Hour school starts (default 8 = 8am)
            school_end: Hour school ends (default 15 = 3pm)
            weekdays_only: If True, only exclude Mon-Fri

        Returns:
            TFTTrainingConfig with school exclusion pattern
        """
        config = TFTTrainingConfig()
        config.time_exclusion_patterns = [
            TimeExclusionPattern(
                name="school_hours",
                start_hour=school_start,
                end_hour=school_end,
                days_of_week=[0, 1, 2, 3, 4] if weekdays_only else None,
                enabled=True
            )
        ]
        return config


# ==============================================================================
# Weighted Quantile Loss
# ==============================================================================

class WeightedQuantileLoss(nn.Module):
    """
    Weighted quantile loss for probabilistic forecasting.

    Allows different weights for each quantile to emphasize certain predictions:
    - Higher weight on lower quantile (0.1) → better hypoglycemia detection
    - Higher weight on median (0.5) → better point prediction
    - Lower weight on upper quantile (0.9) → less focus on high bounds

    Also supports horizon-specific weighting (closer predictions more important).
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        quantile_weights: List[float] = [1.5, 1.0, 0.8],
        horizon_decay: float = 0.95,  # Weight decay per horizon
    ):
        super().__init__()
        self.quantiles = quantiles
        self.quantile_weights = torch.tensor(quantile_weights, dtype=torch.float32)
        self.horizon_decay = horizon_decay

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate weighted quantile loss.

        Args:
            predictions: (batch, n_horizons, n_quantiles)
            targets: (batch, n_horizons) or (batch, n_horizons, 1)
            mask: (batch, n_horizons) binary mask for valid targets

        Returns:
            Scalar loss tensor
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        # Move weights to same device as predictions
        quantile_weights = self.quantile_weights.to(predictions.device)

        # Calculate horizon weights (closer = more important)
        n_horizons = predictions.shape[1]
        horizon_weights = torch.tensor(
            [self.horizon_decay ** i for i in range(n_horizons)],
            device=predictions.device,
            dtype=torch.float32
        )

        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, :, i:i+1]
            error = targets - pred

            # Pinball loss
            loss = torch.where(
                error >= 0,
                q * error,
                (q - 1) * error
            )

            # Apply horizon weights
            loss = loss.squeeze(-1) * horizon_weights.unsqueeze(0)

            # Apply mask if provided
            if mask is not None:
                loss = loss * mask

            # Apply quantile weight
            total_loss = total_loss + quantile_weights[i] * loss.mean()

        return total_loss / len(self.quantiles)


# ==============================================================================
# Data Quality Filtering
# ==============================================================================

@dataclass
class DataQualityMetrics:
    """Metrics for assessing training data quality."""
    total_windows: int = 0
    valid_windows: int = 0
    glucose_completeness: float = 0.0
    treatment_density: float = 0.0  # Treatments per day
    avg_gap_minutes: float = 0.0
    excluded_reasons: Dict[str, int] = field(default_factory=dict)
    # New: Track windows flagged for missing treatments (for later inference)
    flagged_for_inference: List[Dict] = field(default_factory=list)


@dataclass
class MissingTreatmentWindow:
    """A window flagged as having likely missing treatment data."""
    start_time: datetime
    end_time: datetime
    reason: str
    confidence: float  # 0-1, how confident we are treatment is missing
    suspected_type: str  # 'insulin', 'carbs', 'both'
    bg_change: float  # mg/dL change that triggered suspicion
    details: Dict[str, Any] = field(default_factory=dict)


class DataQualityFilter:
    """
    Filter training windows based on data quality criteria.

    CRITICAL: This filter EXCLUDES periods where insulin/carb data appears to be
    missing. These periods can later be used for treatment inference, but should
    NOT be used for training as they would teach the model incorrect patterns.

    Criteria for EXCLUSION (training data quality):
    1. Insufficient glucose readings (large gaps)
    2. Missing treatment coverage (no insulin/carbs when expected)
    3. Unexplained BG changes (suggests missing treatment logs)
    4. Anomalous glucose values
    5. Incomplete feature vectors (NaN values)

    Detection of missing treatments:
    - Unexplained BG rise >40 mg/dL in 60 min with no carbs → likely missing carbs
    - Unexplained BG drop >50 mg/dL in 90 min with no insulin → likely missing insulin
    - Extended periods (>4 hours) with zero treatments during waking hours
    - Time-of-day patterns (e.g., school hours with known missing data)
    """

    # Thresholds for detecting likely missing treatments
    UNEXPLAINED_RISE_THRESHOLD = 40  # mg/dL rise suggesting missing carbs
    UNEXPLAINED_RISE_WINDOW_MIN = 60  # minutes
    UNEXPLAINED_DROP_THRESHOLD = 50  # mg/dL drop suggesting missing insulin
    UNEXPLAINED_DROP_WINDOW_MIN = 90  # minutes
    MIN_TREATMENT_GAP_HOURS = 4  # Max hours without treatments during waking hours
    WAKING_HOURS = (6, 22)  # 6 AM to 10 PM

    def __init__(self, config: TFTTrainingConfig):
        self.config = config
        self.metrics = DataQualityMetrics()
        # Track windows flagged for later inference
        self._flagged_windows: List[MissingTreatmentWindow] = []

    def _check_time_exclusion(self, window_start: datetime, window_end: datetime) -> Optional[str]:
        """
        Check if window overlaps with any time exclusion patterns.

        Returns:
            Exclusion reason string if excluded, None if OK
        """
        for pattern in self.config.time_exclusion_patterns:
            if not pattern.enabled:
                continue

            # Check multiple points within the window for pattern overlap
            current = window_start
            while current <= window_end:
                if pattern.is_excluded(current):
                    return f"time_exclusion_{pattern.name}"
                current += timedelta(minutes=15)  # Check every 15 min

        return None

    def calculate_completeness_score(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime
    ) -> Tuple[float, str]:
        """
        Calculate data completeness score for a time window.

        Returns:
            Tuple of (score 0-1, reason if excluded)
        """
        # Check time exclusion patterns FIRST (e.g., school hours)
        time_exclusion = self._check_time_exclusion(window_start, window_end)
        if time_exclusion:
            return 0.0, time_exclusion

        # Check glucose coverage
        window_mask = (
            (glucose_df['timestamp'] >= window_start) &
            (glucose_df['timestamp'] <= window_end)
        )
        window_glucose = glucose_df[window_mask].copy()

        expected_readings = (window_end - window_start).total_seconds() / 300  # 5 min intervals
        actual_readings = len(window_glucose)
        glucose_completeness = min(1.0, actual_readings / max(1, expected_readings))

        if glucose_completeness < 0.8:
            return 0.0, "insufficient_glucose_readings"

        # Check for large gaps
        if len(window_glucose) >= 2:
            timestamps = pd.to_datetime(window_glucose['timestamp']).sort_values()
            gaps = timestamps.diff().dt.total_seconds() / 60
            max_gap = gaps.max()
            if max_gap > self.config.max_glucose_gap_minutes:
                return 0.0, f"gap_too_large_{max_gap:.0f}min"

        # Check for anomalous glucose values
        if len(window_glucose) > 0:
            glucose_values = window_glucose['value']
            if (glucose_values < 20).any() or (glucose_values > 600).any():
                return 0.0, "anomalous_glucose_values"

        # Get treatments in window
        treatment_mask = (
            (treatments_df['timestamp'] >= window_start) &
            (treatments_df['timestamp'] <= window_end)
        )
        window_treatments = treatments_df[treatment_mask].copy()

        # CRITICAL: Check for missing treatment data
        missing_reason = self._detect_missing_treatments(
            window_glucose, window_treatments, window_start, window_end
        )
        if missing_reason:
            return 0.0, missing_reason

        # Calculate treatment density score
        window_days = max(0.5, (window_end - window_start).total_seconds() / 86400)

        # Count meaningful treatments (insulin > 0.5U or carbs > 5g)
        meaningful_treatments = 0
        if len(window_treatments) > 0:
            insulin_col = 'insulin' if 'insulin' in window_treatments.columns else None
            carbs_col = 'carbs' if 'carbs' in window_treatments.columns else None

            for _, row in window_treatments.iterrows():
                insulin_val = row.get(insulin_col, 0) if insulin_col else 0
                carbs_val = row.get(carbs_col, 0) if carbs_col else 0

                if (insulin_val or 0) > 0.5 or (carbs_val or 0) > 5:
                    meaningful_treatments += 1

        treatment_density = meaningful_treatments / window_days

        # For short windows (< 3 hours), don't strictly require treatments
        if window_days < 0.125:  # 3 hours
            treatment_score = 1.0
        else:
            treatment_score = min(1.0, treatment_density / self.config.min_treatments_per_day)

        # Combined score
        score = 0.6 * glucose_completeness + 0.4 * treatment_score

        if score < self.config.min_completeness_score:
            return score, f"low_completeness_{score:.2f}"

        return score, ""

    def _detect_missing_treatments(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime
    ) -> Optional[str]:
        """
        Detect likely missing treatment data based on glucose patterns.

        Returns:
            Exclusion reason string if missing treatments detected, None if OK
        """
        if len(glucose_df) < 3:
            return None

        # Sort glucose by timestamp
        glucose_sorted = glucose_df.sort_values('timestamp')
        timestamps = pd.to_datetime(glucose_sorted['timestamp'])
        values = glucose_sorted['value'].values

        # Get insulin and carb timestamps
        insulin_times = []
        carb_times = []

        if len(treatments_df) > 0:
            for _, row in treatments_df.iterrows():
                ts = pd.to_datetime(row['timestamp'])
                insulin_val = row.get('insulin', 0) or 0
                carbs_val = row.get('carbs', 0) or 0

                if insulin_val > 0.5:
                    insulin_times.append(ts)
                if carbs_val > 5:
                    carb_times.append(ts)

        # Check 1: Unexplained BG rises (likely missing carbs)
        missing_carbs = self._detect_unexplained_rise(
            timestamps, values, carb_times, window_start, window_end
        )
        if missing_carbs:
            self._flag_for_inference(
                window_start, window_end,
                reason="unexplained_bg_rise",
                suspected_type="carbs",
                confidence=missing_carbs['confidence'],
                bg_change=missing_carbs['rise'],
                details=missing_carbs
            )
            return f"missing_carbs_likely_{missing_carbs['rise']:.0f}mg_rise"

        # Check 2: Unexplained BG drops (likely missing insulin)
        missing_insulin = self._detect_unexplained_drop(
            timestamps, values, insulin_times, window_start, window_end
        )
        if missing_insulin:
            self._flag_for_inference(
                window_start, window_end,
                reason="unexplained_bg_drop",
                suspected_type="insulin",
                confidence=missing_insulin['confidence'],
                bg_change=missing_insulin['drop'],
                details=missing_insulin
            )
            return f"missing_insulin_likely_{missing_insulin['drop']:.0f}mg_drop"

        # Check 3: Extended treatment gap during waking hours
        treatment_gap = self._detect_treatment_gap(
            insulin_times + carb_times, window_start, window_end
        )
        if treatment_gap:
            self._flag_for_inference(
                window_start, window_end,
                reason="extended_treatment_gap",
                suspected_type="both",
                confidence=treatment_gap['confidence'],
                bg_change=0,
                details=treatment_gap
            )
            return f"treatment_gap_{treatment_gap['gap_hours']:.1f}h"

        return None

    def _detect_unexplained_rise(
        self,
        timestamps: pd.Series,
        values: np.ndarray,
        carb_times: List[datetime],
        window_start: datetime,
        window_end: datetime
    ) -> Optional[Dict]:
        """
        Detect BG rises that aren't explained by logged carbs.

        A rise is unexplained if:
        - BG increases > THRESHOLD in WINDOW minutes
        - No carbs logged within 30 min before the rise started
        - No carbs logged during the rise period
        """
        window_min = self.UNEXPLAINED_RISE_WINDOW_MIN
        threshold = self.UNEXPLAINED_RISE_THRESHOLD

        for i in range(len(values) - 1):
            # Find max BG within window
            start_time = timestamps.iloc[i]
            start_bg = values[i]

            # Skip if starting BG is already high (could be post-meal)
            if start_bg > 200:
                continue

            for j in range(i + 1, len(values)):
                end_time = timestamps.iloc[j]
                time_diff_min = (end_time - start_time).total_seconds() / 60

                if time_diff_min > window_min:
                    break

                rise = values[j] - start_bg

                if rise >= threshold:
                    # Check if carbs were logged nearby
                    carbs_nearby = False
                    check_start = start_time - timedelta(minutes=30)
                    check_end = end_time + timedelta(minutes=15)

                    for carb_time in carb_times:
                        if check_start <= carb_time <= check_end:
                            carbs_nearby = True
                            break

                    if not carbs_nearby:
                        # Calculate confidence based on rise magnitude
                        confidence = min(1.0, rise / 80)  # 80 mg/dL rise = 100% confidence

                        return {
                            'rise': rise,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_min': time_diff_min,
                            'start_bg': start_bg,
                            'peak_bg': values[j],
                            'confidence': confidence
                        }

        return None

    def _detect_unexplained_drop(
        self,
        timestamps: pd.Series,
        values: np.ndarray,
        insulin_times: List[datetime],
        window_start: datetime,
        window_end: datetime
    ) -> Optional[Dict]:
        """
        Detect BG drops that aren't explained by logged insulin.

        A drop is unexplained if:
        - BG decreases > THRESHOLD in WINDOW minutes
        - No insulin logged within 3 hours before the drop
        - IOB would be negligible from any earlier insulin
        """
        window_min = self.UNEXPLAINED_DROP_WINDOW_MIN
        threshold = self.UNEXPLAINED_DROP_THRESHOLD

        for i in range(len(values) - 1):
            start_time = timestamps.iloc[i]
            start_bg = values[i]

            # Skip if starting BG is already low
            if start_bg < 100:
                continue

            for j in range(i + 1, len(values)):
                end_time = timestamps.iloc[j]
                time_diff_min = (end_time - start_time).total_seconds() / 60

                if time_diff_min > window_min:
                    break

                drop = start_bg - values[j]

                if drop >= threshold:
                    # Check if insulin was logged recently (within 3 hours before drop)
                    insulin_nearby = False
                    check_start = start_time - timedelta(hours=3)
                    check_end = end_time

                    for insulin_time in insulin_times:
                        if check_start <= insulin_time <= check_end:
                            insulin_nearby = True
                            break

                    if not insulin_nearby:
                        # Calculate confidence based on drop magnitude and BG level
                        confidence = min(1.0, drop / 100)  # 100 mg/dL drop = 100% confidence

                        return {
                            'drop': drop,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_min': time_diff_min,
                            'start_bg': start_bg,
                            'end_bg': values[j],
                            'confidence': confidence
                        }

        return None

    def _detect_treatment_gap(
        self,
        all_treatment_times: List[datetime],
        window_start: datetime,
        window_end: datetime
    ) -> Optional[Dict]:
        """
        Detect extended periods without any treatments during waking hours.

        This is important because diabetics on insulin typically need
        treatment every 4-6 hours while awake.
        """
        # Only check if window is during waking hours
        start_hour = window_start.hour
        end_hour = window_end.hour

        # Check if window overlaps waking hours
        wake_start, wake_end = self.WAKING_HOURS
        if end_hour < wake_start or start_hour > wake_end:
            # Entirely during sleeping hours - OK to have no treatments
            return None

        # Window duration in hours
        window_hours = (window_end - window_start).total_seconds() / 3600

        if window_hours < self.MIN_TREATMENT_GAP_HOURS:
            # Window too short to require treatments
            return None

        if not all_treatment_times:
            # No treatments in entire window during waking hours
            confidence = min(1.0, window_hours / 8)  # 8 hours = 100% confidence

            return {
                'gap_hours': window_hours,
                'start_time': window_start,
                'end_time': window_end,
                'confidence': confidence
            }

        # Check for large gaps between treatments
        sorted_times = sorted(all_treatment_times)

        # Check gap from window start to first treatment
        first_gap = (sorted_times[0] - window_start).total_seconds() / 3600
        if first_gap > self.MIN_TREATMENT_GAP_HOURS:
            # Check if this gap is during waking hours
            gap_end_hour = sorted_times[0].hour
            if wake_start <= gap_end_hour <= wake_end:
                return {
                    'gap_hours': first_gap,
                    'start_time': window_start,
                    'end_time': sorted_times[0],
                    'confidence': min(1.0, first_gap / 8)
                }

        # Check gaps between treatments
        for i in range(len(sorted_times) - 1):
            gap_hours = (sorted_times[i + 1] - sorted_times[i]).total_seconds() / 3600

            if gap_hours > self.MIN_TREATMENT_GAP_HOURS:
                # Check if gap is during waking hours
                gap_mid_hour = (sorted_times[i] + timedelta(hours=gap_hours/2)).hour
                if wake_start <= gap_mid_hour <= wake_end:
                    return {
                        'gap_hours': gap_hours,
                        'start_time': sorted_times[i],
                        'end_time': sorted_times[i + 1],
                        'confidence': min(1.0, gap_hours / 8)
                    }

        # Check gap from last treatment to window end
        last_gap = (window_end - sorted_times[-1]).total_seconds() / 3600
        if last_gap > self.MIN_TREATMENT_GAP_HOURS:
            gap_start_hour = sorted_times[-1].hour
            if wake_start <= gap_start_hour <= wake_end:
                return {
                    'gap_hours': last_gap,
                    'start_time': sorted_times[-1],
                    'end_time': window_end,
                    'confidence': min(1.0, last_gap / 8)
                }

        return None

    def _flag_for_inference(
        self,
        start_time: datetime,
        end_time: datetime,
        reason: str,
        suspected_type: str,
        confidence: float,
        bg_change: float,
        details: Dict
    ) -> None:
        """Flag a window for later treatment inference."""
        flagged = MissingTreatmentWindow(
            start_time=start_time,
            end_time=end_time,
            reason=reason,
            confidence=confidence,
            suspected_type=suspected_type,
            bg_change=bg_change,
            details=details
        )
        self._flagged_windows.append(flagged)

    def get_flagged_windows(self) -> List[MissingTreatmentWindow]:
        """Get all windows flagged for treatment inference."""
        return self._flagged_windows

    def filter_sequences(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        timestamps: List[datetime],
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime], DataQualityMetrics]:
        """
        Filter training sequences based on quality criteria.

        CRITICAL: Sequences with missing treatment data are EXCLUDED from training.
        These periods are flagged for later inference but should NOT be used to
        train the model as they would teach incorrect BG-treatment relationships.

        Args:
            sequences: (n_samples, seq_len, n_features) feature arrays
            targets: (n_samples, n_horizons) target glucose values
            timestamps: List of sequence end timestamps
            glucose_df: Full glucose DataFrame
            treatments_df: Full treatments DataFrame

        Returns:
            Filtered sequences, targets, timestamps, and metrics
        """
        # Reset metrics and flagged windows for this run
        self.metrics = DataQualityMetrics()
        self.metrics.total_windows = len(sequences)
        self.metrics.excluded_reasons = {}
        self._flagged_windows = []  # Reset flagged windows

        valid_indices = []

        # Window duration: encoder (120 min) + prediction (60 min) = 180 min
        window_duration = timedelta(
            minutes=(self.config.encoder_length + self.config.prediction_length) * 5
        )

        logger.info(
            f"Filtering {len(sequences)} sequences for training data quality...\n"
            f"  - Excluding windows with missing insulin/carb data\n"
            f"  - Flagging periods for later treatment inference"
        )

        for i, ts in enumerate(timestamps):
            window_end = ts
            window_start = ts - window_duration

            # Check for NaN in features
            if np.isnan(sequences[i]).any():
                self.metrics.excluded_reasons['nan_features'] = \
                    self.metrics.excluded_reasons.get('nan_features', 0) + 1
                continue

            # Check for NaN in targets
            if np.isnan(targets[i]).any():
                self.metrics.excluded_reasons['nan_targets'] = \
                    self.metrics.excluded_reasons.get('nan_targets', 0) + 1
                continue

            # Check completeness score (includes missing treatment detection)
            score, reason = self.calculate_completeness_score(
                glucose_df, treatments_df, window_start, window_end
            )

            if reason:
                # Categorize the exclusion reason
                if reason.startswith('missing_carbs'):
                    key = 'missing_carbs'
                elif reason.startswith('missing_insulin'):
                    key = 'missing_insulin'
                elif reason.startswith('treatment_gap'):
                    key = 'treatment_gap'
                elif reason.startswith('gap_too_large'):
                    key = 'glucose_gap'
                elif reason.startswith('low_completeness'):
                    key = 'low_completeness'
                elif reason.startswith('time_exclusion'):
                    # Extract the pattern name (e.g., "time_exclusion_school_hours" -> "school_hours")
                    key = reason.replace('time_exclusion_', '')
                else:
                    key = reason.split('_')[0] if '_' in reason else reason

                self.metrics.excluded_reasons[key] = \
                    self.metrics.excluded_reasons.get(key, 0) + 1
                continue

            valid_indices.append(i)

        self.metrics.valid_windows = len(valid_indices)
        self.metrics.glucose_completeness = (
            self.metrics.valid_windows / max(1, self.metrics.total_windows)
        )

        # Add flagged windows to metrics for later inference
        self.metrics.flagged_for_inference = [
            {
                'start': w.start_time.isoformat(),
                'end': w.end_time.isoformat(),
                'reason': w.reason,
                'type': w.suspected_type,
                'confidence': w.confidence,
                'bg_change': w.bg_change
            }
            for w in self._flagged_windows
        ]

        # Log detailed filtering summary
        logger.info(
            f"\n{'='*60}\n"
            f"DATA QUALITY FILTERING SUMMARY\n"
            f"{'='*60}\n"
            f"Total windows: {self.metrics.total_windows}\n"
            f"Valid for training: {self.metrics.valid_windows} "
            f"({self.metrics.glucose_completeness:.1%})\n"
            f"Excluded: {self.metrics.total_windows - self.metrics.valid_windows}"
        )

        if self.metrics.excluded_reasons:
            logger.info("\nExclusion breakdown:")
            for reason, count in sorted(
                self.metrics.excluded_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / self.metrics.total_windows * 100
                logger.info(f"  - {reason}: {count} ({pct:.1f}%)")

        # Log flagged windows for inference
        flagged_by_type = {}
        for w in self._flagged_windows:
            flagged_by_type[w.suspected_type] = flagged_by_type.get(w.suspected_type, 0) + 1

        if flagged_by_type:
            logger.info(f"\nFlagged for treatment inference:")
            for treatment_type, count in flagged_by_type.items():
                logger.info(f"  - Likely missing {treatment_type}: {count} windows")

        logger.info(f"{'='*60}\n")

        if len(valid_indices) == 0:
            # Provide helpful error with details
            error_msg = (
                f"No valid training windows after quality filtering.\n"
                f"Exclusion reasons: {self.metrics.excluded_reasons}\n"
                f"Consider:\n"
                f"  - Collecting more complete treatment data\n"
                f"  - Adjusting quality thresholds in TFTTrainingConfig\n"
                f"  - Using treatment inference to fill gaps before training"
            )
            raise ValueError(error_msg)

        valid_indices = np.array(valid_indices)
        filtered_timestamps = [timestamps[i] for i in valid_indices]

        return (
            sequences[valid_indices],
            targets[valid_indices],
            filtered_timestamps,
            self.metrics
        )

    def export_flagged_windows_for_inference(self, output_path: Optional[Path] = None) -> List[Dict]:
        """
        Export flagged windows for treatment inference.

        These are periods excluded from training due to likely missing treatments.
        They can be used to:
        1. Display to user for confirmation
        2. Run treatment inference algorithms
        3. Add inferred treatments and re-train

        Args:
            output_path: Optional path to save JSON file

        Returns:
            List of flagged window dictionaries
        """
        export_data = []

        for w in self._flagged_windows:
            export_data.append({
                'start_time': w.start_time.isoformat(),
                'end_time': w.end_time.isoformat(),
                'reason': w.reason,
                'suspected_treatment_type': w.suspected_type,
                'confidence': round(w.confidence, 3),
                'bg_change_mg_dl': round(w.bg_change, 1),
                'details': {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in w.details.items()
                }
            })

        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'flagged_windows': export_data,
                    'total_count': len(export_data),
                    'by_type': {
                        'carbs': len([w for w in export_data if w['suspected_treatment_type'] == 'carbs']),
                        'insulin': len([w for w in export_data if w['suspected_treatment_type'] == 'insulin']),
                        'both': len([w for w in export_data if w['suspected_treatment_type'] == 'both']),
                    },
                    'exported_at': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Exported {len(export_data)} flagged windows to {output_path}")

        return export_data


# ==============================================================================
# Temporal-Aware Data Splitting
# ==============================================================================

class TemporalDataSplitter:
    """
    Split time series data with proper temporal ordering to prevent data leakage.

    Rules:
    1. Training data must be BEFORE validation data
    2. Validation data must be BEFORE test data
    3. Gap between splits to prevent information leakage
    4. Optional: rolling window cross-validation
    """

    def __init__(
        self,
        val_split: float = 0.15,
        test_split: float = 0.15,
        gap_hours: int = 6,  # Gap between train/val and val/test
    ):
        self.val_split = val_split
        self.test_split = test_split
        self.gap_samples = gap_hours * 12  # 12 samples per hour (5-min intervals)

    def split(
        self,
        n_samples: int,
        timestamps: Optional[List[datetime]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test with temporal ordering.

        Args:
            n_samples: Total number of samples
            timestamps: Optional timestamps for validation

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Calculate split points
        test_size = int(n_samples * self.test_split)
        val_size = int(n_samples * self.val_split)

        # Test set: last portion
        test_start = n_samples - test_size

        # Validation set: before test with gap
        val_end = test_start - self.gap_samples
        val_start = val_end - val_size

        # Training set: before validation with gap
        train_end = val_start - self.gap_samples

        if train_end < 100:
            raise ValueError(
                f"Not enough training data after temporal split. "
                f"Need at least 100 samples, got {train_end}"
            )

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)
        test_indices = np.arange(test_start, n_samples)

        logger.info(
            f"Temporal split: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )

        if timestamps:
            logger.info(
                f"  Train: {timestamps[0]} to {timestamps[train_end-1]}"
            )
            logger.info(
                f"  Val: {timestamps[val_start]} to {timestamps[val_end-1]}"
            )
            logger.info(
                f"  Test: {timestamps[test_start]} to {timestamps[-1]}"
            )

        return train_indices, val_indices, test_indices

    def time_series_cv(
        self,
        n_samples: int,
        n_splits: int = 5,
        min_train_size: int = 500,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series cross-validation splits.

        Uses expanding window: each fold has more training data than the previous.

        Returns:
            List of (train_indices, val_indices) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.gap_samples)

        splits = []
        for train_idx, val_idx in tscv.split(np.arange(n_samples)):
            if len(train_idx) >= min_train_size:
                splits.append((train_idx, val_idx))

        logger.info(f"Time series CV: {len(splits)} valid folds")
        return splits


# ==============================================================================
# PyTorch Dataset
# ==============================================================================

class TFTDataset(Dataset):
    """
    PyTorch Dataset for TFT training.

    Features:
    - Proper handling of missing values
    - Optional data augmentation
    - Efficient memory usage
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        augment: bool = False,
    ):
        """
        Args:
            sequences: (n_samples, seq_len, n_features)
            targets: (n_samples, n_horizons)
            scaler: Optional fitted scaler for features
            augment: Whether to apply data augmentation
        """
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.scaler = scaler
        self.augment = augment

        # Apply scaling if scaler provided
        if scaler is not None:
            n_samples, seq_len, n_features = self.sequences.shape
            flat = self.sequences.reshape(-1, n_features)
            scaled = scaler.transform(flat)
            self.sequences = scaled.reshape(n_samples, seq_len, n_features).astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        target = self.targets[idx]

        # Optional augmentation
        if self.augment:
            seq = self._augment_sequence(seq)

        return torch.from_numpy(seq), torch.from_numpy(target)

    def _augment_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply light data augmentation."""
        # Add small Gaussian noise to non-categorical features
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, seq.shape).astype(np.float32)

        # Don't add noise to binary/categorical features (last 10)
        noise[:, -10:] = 0

        return seq + noise


# ==============================================================================
# Model Checkpoint Management
# ==============================================================================

@dataclass
class TFTCheckpoint:
    """
    Complete model checkpoint including scalers and metadata.
    """
    model_state_dict: Dict
    scaler_state: bytes  # Pickled StandardScaler
    config: Dict
    training_metrics: Dict
    feature_columns: List[str]
    created_at: str
    training_data_range: Dict  # start/end timestamps
    version: str = "2.0"

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        checkpoint = {
            'model_state_dict': self.model_state_dict,
            'scaler_state': self.scaler_state,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_columns': self.feature_columns,
            'created_at': self.created_at,
            'training_data_range': self.training_data_range,
            'version': self.version,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved TFT checkpoint to {path}")

        # Also save metadata as JSON for easy inspection
        metadata_path = path.with_suffix('.json')
        metadata = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'scaler_state']}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'TFTCheckpoint':
        """Load checkpoint from disk."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return cls(
            model_state_dict=checkpoint['model_state_dict'],
            scaler_state=checkpoint['scaler_state'],
            config=checkpoint['config'],
            training_metrics=checkpoint.get('training_metrics', {}),
            feature_columns=checkpoint.get('feature_columns', []),
            created_at=checkpoint.get('created_at', ''),
            training_data_range=checkpoint.get('training_data_range', {}),
            version=checkpoint.get('version', '1.0'),
        )

    def get_scaler(self) -> StandardScaler:
        """Deserialize the feature scaler."""
        return pickle.loads(self.scaler_state)


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

class TFTEvaluator:
    """
    Comprehensive evaluation metrics for TFT model.

    Metrics computed:
    - Per-horizon MAE, RMSE, MAPE
    - Quantile calibration (coverage)
    - Clarke Error Grid analysis
    - Time-in-range prediction accuracy
    """

    def __init__(self, horizons_minutes: List[int] = [15, 30, 45, 60, 90, 120]):
        self.horizons = horizons_minutes

    def evaluate(
        self,
        predictions: np.ndarray,  # (n_samples, n_horizons, n_quantiles)
        targets: np.ndarray,  # (n_samples, n_horizons)
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Get median predictions (50th percentile)
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else 1
        median_pred = predictions[:, :, median_idx]

        # Per-horizon metrics
        for h_idx, horizon in enumerate(self.horizons[:predictions.shape[1]]):
            h_pred = median_pred[:, h_idx]
            h_target = targets[:, h_idx]

            # Remove NaN
            valid = ~(np.isnan(h_pred) | np.isnan(h_target))
            h_pred = h_pred[valid]
            h_target = h_target[valid]

            if len(h_pred) == 0:
                continue

            # MAE
            mae = np.mean(np.abs(h_pred - h_target))
            results[f'mae_{horizon}min'] = float(mae)

            # RMSE
            rmse = np.sqrt(np.mean((h_pred - h_target) ** 2))
            results[f'rmse_{horizon}min'] = float(rmse)

            # MAPE (mean absolute percentage error)
            mape = np.mean(np.abs((h_pred - h_target) / np.maximum(h_target, 1))) * 100
            results[f'mape_{horizon}min'] = float(mape)

            # Bias (mean error - positive = over-predicting)
            bias = np.mean(h_pred - h_target)
            results[f'bias_{horizon}min'] = float(bias)

        # Overall metrics
        all_pred = median_pred.flatten()
        all_target = targets.flatten()
        valid = ~(np.isnan(all_pred) | np.isnan(all_target))

        results['overall_mae'] = float(np.mean(np.abs(all_pred[valid] - all_target[valid])))
        results['overall_rmse'] = float(np.sqrt(np.mean((all_pred[valid] - all_target[valid]) ** 2)))

        # Quantile calibration (coverage)
        for q_idx, q in enumerate(quantiles):
            if q == 0.5:
                continue

            q_pred = predictions[:, :, q_idx].flatten()

            if q < 0.5:
                # Lower quantile: % of actuals above prediction
                coverage = np.mean(all_target[valid] > q_pred[valid])
                expected = 1 - q
            else:
                # Upper quantile: % of actuals below prediction
                coverage = np.mean(all_target[valid] < q_pred[valid])
                expected = q

            results[f'coverage_q{int(q*100)}'] = float(coverage)
            results[f'coverage_error_q{int(q*100)}'] = float(abs(coverage - expected))

        # 80% interval coverage (10th to 90th)
        if 0.1 in quantiles and 0.9 in quantiles:
            lower = predictions[:, :, quantiles.index(0.1)].flatten()[valid]
            upper = predictions[:, :, quantiles.index(0.9)].flatten()[valid]
            in_interval = (all_target[valid] >= lower) & (all_target[valid] <= upper)
            results['coverage_80pct'] = float(np.mean(in_interval))

        # Clarke Error Grid zones (simplified)
        results['clarke_zone_a'] = self._clarke_zone_a_percentage(
            median_pred.flatten()[valid],
            all_target[valid]
        )

        # Hypoglycemia detection metrics
        results.update(self._hypo_detection_metrics(
            predictions, targets, threshold=70
        ))

        return results

    def _clarke_zone_a_percentage(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        Calculate percentage of predictions in Clarke Error Grid Zone A.

        Zone A: Clinically accurate (within 20% for BG > 70, within 20 mg/dL for BG <= 70)
        """
        n = len(predicted)
        if n == 0:
            return 0.0

        zone_a = 0
        for p, a in zip(predicted, actual):
            if a <= 70:
                # For low BG, allow 20 mg/dL error
                if abs(p - a) <= 20:
                    zone_a += 1
            else:
                # For normal/high BG, allow 20% error
                if abs(p - a) / a <= 0.20:
                    zone_a += 1

        return float(zone_a / n)

    def _hypo_detection_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: int = 70
    ) -> Dict[str, float]:
        """
        Calculate hypoglycemia detection sensitivity and specificity.
        """
        # Use 30-min prediction if available, else first horizon
        h_idx = min(1, predictions.shape[1] - 1)

        pred = predictions[:, h_idx, 1]  # Median
        actual = targets[:, h_idx]

        valid = ~(np.isnan(pred) | np.isnan(actual))
        pred = pred[valid]
        actual = actual[valid]

        # Binary classification: is BG < threshold?
        pred_hypo = pred < threshold
        actual_hypo = actual < threshold

        # True positives, false positives, etc.
        tp = np.sum(pred_hypo & actual_hypo)
        fp = np.sum(pred_hypo & ~actual_hypo)
        fn = np.sum(~pred_hypo & actual_hypo)
        tn = np.sum(~pred_hypo & ~actual_hypo)

        sensitivity = tp / max(1, tp + fn)  # Recall
        specificity = tn / max(1, tn + fp)
        precision = tp / max(1, tp + fp)

        return {
            'hypo_sensitivity': float(sensitivity),
            'hypo_specificity': float(specificity),
            'hypo_precision': float(precision),
            'hypo_f1': float(2 * precision * sensitivity / max(0.001, precision + sensitivity)),
        }


# ==============================================================================
# Main Training Pipeline
# ==============================================================================

class TFTTrainingPipeline:
    """
    Complete TFT training pipeline with all best practices.

    Usage:
        pipeline = TFTTrainingPipeline(config)
        pipeline.prepare_data(glucose_df, treatments_df)
        metrics = pipeline.train()
        pipeline.save_checkpoint("models/tft_v2.pth")
    """

    def __init__(
        self,
        config: Optional[TFTTrainingConfig] = None,
        device: str = "auto",
    ):
        self.config = config or TFTTrainingConfig()

        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"TFT Training Pipeline initialized on {self.device}")

        # Components
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.data_filter = DataQualityFilter(self.config)
        self.splitter = TemporalDataSplitter(
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            gap_hours=1  # Reduced gap for sparse data (was 6)
        )
        self.evaluator = TFTEvaluator(self.config.horizons_minutes)

        # Data
        self.train_sequences: Optional[np.ndarray] = None
        self.train_targets: Optional[np.ndarray] = None
        self.val_sequences: Optional[np.ndarray] = None
        self.val_targets: Optional[np.ndarray] = None
        self.test_sequences: Optional[np.ndarray] = None
        self.test_targets: Optional[np.ndarray] = None
        self.timestamps: List[datetime] = []
        self.feature_columns: List[str] = []

        # Training state
        self.training_history: List[Dict] = []
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict] = None

    def prepare_data(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> DataQualityMetrics:
        """
        Prepare training data with quality filtering and splitting.

        Args:
            glucose_df: DataFrame with 'timestamp', 'value' columns
            treatments_df: DataFrame with 'timestamp', 'insulin', 'carbs', etc.
            feature_columns: List of feature column names (uses TFT_FEATURE_COLUMNS if None)

        Returns:
            DataQualityMetrics
        """
        from ml.feature_engineering import (
            TFT_FEATURE_COLUMNS,
            prepare_realtime_features,
            engineer_extended_features,
        )

        self.feature_columns = feature_columns or list(TFT_FEATURE_COLUMNS)

        logger.info(f"Preparing TFT training data with {len(self.feature_columns)} features")
        logger.info(f"Glucose readings: {len(glucose_df)}, Treatments: {len(treatments_df)}")

        # Convert to dict format for feature engineering
        glucose_records = glucose_df.to_dict('records')
        treatment_records = treatments_df.to_dict('records') if len(treatments_df) > 0 else []

        # Prepare features
        df = prepare_realtime_features(glucose_records, treatment_records)

        if df is None or len(df) < self.config.encoder_length + self.config.prediction_length:
            raise ValueError("Insufficient data for feature engineering")

        # Engineer extended features
        df_extended = engineer_extended_features(
            df,
            treatments_df=treatments_df,
            ml_iob=0.0,  # Will be computed per-timestep
            ml_cob=0.0,
            isf=50.0,
            icr=10.0,
        )

        # Create sequences
        sequences, targets, timestamps = self._create_sequences(df_extended, glucose_df)

        logger.info(f"Created {len(sequences)} sequences")

        # Apply quality filtering
        sequences, targets, timestamps, metrics = self.data_filter.filter_sequences(
            sequences, targets, timestamps, glucose_df, treatments_df
        )

        # Fit scaler on filtered data
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        n_samples, seq_len, n_features = sequences.shape
        flat_sequences = sequences.reshape(-1, n_features)
        self.scaler.fit(flat_sequences)

        # Temporal split
        train_idx, val_idx, test_idx = self.splitter.split(len(sequences), timestamps)

        self.train_sequences = sequences[train_idx]
        self.train_targets = targets[train_idx]
        self.val_sequences = sequences[val_idx]
        self.val_targets = targets[val_idx]
        self.test_sequences = sequences[test_idx]
        self.test_targets = targets[test_idx]
        self.timestamps = timestamps

        logger.info(
            f"Data prepared: train={len(self.train_sequences)}, "
            f"val={len(self.val_sequences)}, test={len(self.test_sequences)}"
        )

        return metrics

    def _create_sequences(
        self,
        df: pd.DataFrame,
        glucose_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """Create input sequences and targets from feature DataFrame."""

        # Ensure features are available
        available_features = [f for f in self.feature_columns if f in df.columns]
        if len(available_features) < len(self.feature_columns) * 0.8:
            logger.warning(
                f"Only {len(available_features)}/{len(self.feature_columns)} features available. "
                f"Missing: {set(self.feature_columns) - set(available_features)}"
            )

        # Fill missing features with 0
        for f in self.feature_columns:
            if f not in df.columns:
                df[f] = 0.0

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Extract feature matrix
        feature_matrix = df[self.feature_columns].values
        timestamps = pd.to_datetime(df['timestamp']).tolist()
        glucose_values = df['value'].values if 'value' in df.columns else None

        if glucose_values is None:
            # Merge glucose values from original df
            df_merged = df.merge(
                glucose_df[['timestamp', 'value']],
                on='timestamp',
                how='left'
            )
            glucose_values = df_merged['value'].values

        # Create sequences
        encoder_len = self.config.encoder_length
        pred_len = self.config.prediction_length
        total_len = encoder_len + pred_len

        sequences = []
        targets_list = []
        seq_timestamps = []

        # Horizon indices (relative to encoder end)
        horizon_indices = [h // 5 for h in self.config.horizons_minutes]

        for i in range(len(feature_matrix) - total_len + 1):
            # Input sequence: encoder_len timesteps
            seq = feature_matrix[i:i + encoder_len]

            # Target: glucose at each horizon
            target = []
            for h_idx in horizon_indices:
                target_idx = i + encoder_len + h_idx - 1
                if target_idx < len(glucose_values):
                    target.append(glucose_values[target_idx])
                else:
                    target.append(np.nan)

            sequences.append(seq)
            targets_list.append(target)
            seq_timestamps.append(timestamps[i + encoder_len - 1])

        return (
            np.array(sequences, dtype=np.float32),
            np.array(targets_list, dtype=np.float32),
            seq_timestamps
        )

    def train(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        mlflow_tracking: bool = False,
    ) -> Dict[str, float]:
        """
        Train the TFT model.

        Args:
            epochs: Number of epochs (uses config if None)
            patience: Early stopping patience (uses config if None)
            mlflow_tracking: Whether to log to MLflow

        Returns:
            Dictionary of final metrics
        """
        epochs = epochs or self.config.epochs
        patience = patience or self.config.patience

        if self.train_sequences is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Create model
        from ml.models.tft_predictor import TemporalFusionTransformer

        self.model = TemporalFusionTransformer(
            n_features=self.config.n_features,
            hidden_size=self.config.hidden_size,
            n_heads=self.config.n_heads,
            n_lstm_layers=self.config.n_lstm_layers,
            dropout=self.config.dropout,
            encoder_length=self.config.encoder_length,
            prediction_length=self.config.prediction_length,
            quantiles=self.config.quantiles,
            horizons_minutes=self.config.horizons_minutes,
        ).to(self.device)

        # Create datasets and loaders
        train_dataset = TFTDataset(
            self.train_sequences,
            self.train_targets,
            scaler=self.scaler,
            augment=True
        )
        val_dataset = TFTDataset(
            self.val_sequences,
            self.val_targets,
            scaler=self.scaler,
            augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == 'cuda'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0
        )

        # Loss and optimizer
        criterion = WeightedQuantileLoss(
            quantiles=self.config.quantiles,
            quantile_weights=self.config.quantile_weights,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # MLflow tracking
        mlflow_run = None
        if mlflow_tracking:
            try:
                from ml.mlflow_tracking import ModelTracker
                tracker = ModelTracker(model_type="tft")
                mlflow_run = tracker.start_run(run_name=f"tft_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                tracker.log_params(self.config.to_dict())
            except Exception as e:
                logger.warning(f"MLflow tracking disabled: {e}")
                mlflow_tracking = False

        # Training loop
        epochs_without_improvement = 0
        self.training_history = []

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = self.model(batch_x)
                predictions = output['predictions']

                # Calculate loss
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = self.model(batch_x)
                    predictions = output['predictions']
                    loss = criterion(predictions, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Update scheduler
            scheduler.step(avg_val_loss)

            # Record history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

            # Log to MLflow
            if mlflow_tracking:
                try:
                    tracker.log_metrics({
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'lr': optimizer.param_groups[0]['lr']
                    }, step=epoch)
                except:
                    pass

            # Check for improvement
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f} (new best)"
                )
            else:
                epochs_without_improvement += 1
                if epoch % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={avg_val_loss:.4f}"
                    )

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final evaluation on test set
        test_metrics = self.evaluate()

        # End MLflow run
        if mlflow_tracking and mlflow_run:
            try:
                tracker.log_metrics(test_metrics)
                tracker.end_run()
            except:
                pass

        return test_metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.model is None or self.test_sequences is None:
            raise ValueError("Model not trained or test data not available")

        self.model.eval()

        # Create test dataset
        test_dataset = TFTDataset(
            self.test_sequences,
            self.test_targets,
            scaler=self.scaler,
            augment=False
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                all_predictions.append(output['predictions'].cpu().numpy())
                all_targets.append(batch_y.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        logger.info(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

        metrics = self.evaluator.evaluate(
            predictions,
            targets,
            quantiles=self.config.quantiles
        )

        logger.info("Test set evaluation:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save complete model checkpoint including scalers."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained")

        checkpoint = TFTCheckpoint(
            model_state_dict=self.model.state_dict(),
            scaler_state=pickle.dumps(self.scaler),
            config=self.config.to_dict(),
            training_metrics={
                'best_val_loss': self.best_val_loss,
                'final_history': self.training_history[-1] if self.training_history else {},
            },
            feature_columns=self.feature_columns,
            created_at=datetime.now().isoformat(),
            training_data_range={
                'start': str(self.timestamps[0]) if self.timestamps else '',
                'end': str(self.timestamps[-1]) if self.timestamps else '',
                'n_samples': len(self.timestamps),
            }
        )

        checkpoint.save(Path(path))

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "auto") -> 'TFTTrainingPipeline':
        """Load a trained pipeline from checkpoint."""
        checkpoint = TFTCheckpoint.load(Path(path))

        config = TFTTrainingConfig(**checkpoint.config)
        pipeline = cls(config=config, device=device)

        # Create model
        from ml.models.tft_predictor import TemporalFusionTransformer

        pipeline.model = TemporalFusionTransformer(
            n_features=config.n_features,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_lstm_layers=config.n_lstm_layers,
            dropout=config.dropout,
            encoder_length=config.encoder_length,
            prediction_length=config.prediction_length,
            quantiles=config.quantiles,
            horizons_minutes=config.horizons_minutes,
        ).to(pipeline.device)

        pipeline.model.load_state_dict(checkpoint.model_state_dict)
        pipeline.model.eval()

        pipeline.scaler = checkpoint.get_scaler()
        pipeline.feature_columns = checkpoint.feature_columns

        logger.info(f"Loaded TFT checkpoint from {path}")
        logger.info(f"  Version: {checkpoint.version}")
        logger.info(f"  Created: {checkpoint.created_at}")
        logger.info(f"  Features: {len(pipeline.feature_columns)}")

        return pipeline
