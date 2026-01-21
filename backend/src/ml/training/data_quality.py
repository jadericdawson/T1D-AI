"""
Training Data Quality Filter for T1D-AI

Filters training data to exclude periods with incomplete treatment logging.
This is critical because missing insulin/carb data would teach the model
wrong patterns (e.g., BG rising without logged carbs).

Key filtering criteria:
1. School hours (7:30 AM - 3:30 PM weekdays) - treatments often not logged
2. Long gaps in glucose readings (> 30 min)
3. Unexplained BG movements (rise/fall without treatments)
4. Completeness score < threshold

Author: T1D-AI
Date: 2026-01-07
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TrainingDataQualityFilter:
    """
    Filters training data to ensure high quality for model training.

    Excludes periods where treatment logging is incomplete, which would
    otherwise teach the model incorrect patterns.
    """

    def __init__(
        self,
        school_start: time = time(7, 30),
        school_end: time = time(15, 30),
        exclude_school_weekdays: bool = True,
        max_glucose_gap_minutes: int = 30,
        min_completeness_score: float = 0.6,
        unexplained_rise_threshold: float = 40.0,  # mg/dL rise without carbs
        unexplained_drop_threshold: float = 50.0,  # mg/dL drop without insulin
    ):
        self.school_start = school_start
        self.school_end = school_end
        self.exclude_school_weekdays = exclude_school_weekdays
        self.max_glucose_gap_minutes = max_glucose_gap_minutes
        self.min_completeness_score = min_completeness_score
        self.unexplained_rise_threshold = unexplained_rise_threshold
        self.unexplained_drop_threshold = unexplained_drop_threshold

    def is_school_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within school hours (weekday 7:30 AM - 3:30 PM)."""
        if not self.exclude_school_weekdays:
            return False

        # Check if weekday (0=Monday, 6=Sunday)
        if timestamp.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check time
        t = timestamp.time()
        return self.school_start <= t <= self.school_end

    def calculate_completeness_score(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime,
    ) -> Tuple[float, Dict]:
        """
        Calculate completeness score for a time window.

        Factors:
        1. Glucose coverage: % of expected readings present
        2. Treatment logging rate: treatments per day
        3. Unexplained movements: BG changes without logged treatments
        4. School hours overlap: penalize windows with school time

        Returns:
            score: 0-1 completeness score
            details: Dict with component scores
        """
        details = {}
        duration_hours = (window_end - window_start).total_seconds() / 3600

        # 1. Glucose coverage (expect reading every 5 min)
        expected_readings = int(duration_hours * 12)  # 12 per hour
        actual_readings = len(glucose_df[
            (glucose_df['timestamp'] >= window_start) &
            (glucose_df['timestamp'] <= window_end)
        ])
        glucose_coverage = min(1.0, actual_readings / max(1, expected_readings))
        details['glucose_coverage'] = glucose_coverage

        # 2. Check for gaps > max_glucose_gap_minutes
        window_glucose = glucose_df[
            (glucose_df['timestamp'] >= window_start) &
            (glucose_df['timestamp'] <= window_end)
        ].sort_values('timestamp')

        if len(window_glucose) >= 2:
            gaps = window_glucose['timestamp'].diff().dt.total_seconds() / 60
            max_gap = gaps.max()
            gap_penalty = 1.0 if max_gap <= self.max_glucose_gap_minutes else 0.5
        else:
            gap_penalty = 0.0
        details['gap_penalty'] = gap_penalty

        # 3. Treatment logging rate
        window_treatments = treatments_df[
            (treatments_df['timestamp'] >= window_start) &
            (treatments_df['timestamp'] <= window_end)
        ]

        # Expect at least 4 treatments per day (3 meals + corrections)
        expected_treatments = max(1, duration_hours / 6)  # 4 per 24h
        treatment_rate = min(1.0, len(window_treatments) / expected_treatments)
        details['treatment_rate'] = treatment_rate

        # 4. School hours overlap
        school_hours_in_window = 0
        current = window_start
        while current < window_end:
            if self.is_school_hours(current):
                school_hours_in_window += 5  # minutes
            current += timedelta(minutes=5)

        total_minutes = (window_end - window_start).total_seconds() / 60
        school_overlap = school_hours_in_window / max(1, total_minutes)
        school_penalty = 1.0 - (school_overlap * 0.5)  # Max 50% penalty
        details['school_penalty'] = school_penalty

        # 5. Check for unexplained movements
        unexplained_penalty = self._check_unexplained_movements(
            window_glucose, window_treatments
        )
        details['unexplained_penalty'] = unexplained_penalty

        # Combine scores with weights
        weights = {
            'glucose_coverage': 0.25,
            'gap_penalty': 0.15,
            'treatment_rate': 0.25,
            'school_penalty': 0.20,
            'unexplained_penalty': 0.15,
        }

        score = sum(details[k] * weights[k] for k in weights)
        details['final_score'] = score

        return score, details

    def _check_unexplained_movements(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
    ) -> float:
        """
        Check for unexplained BG movements (changes without treatments).

        Returns penalty score (1.0 = no issues, 0.0 = major issues).
        """
        if len(glucose_df) < 10:
            return 1.0  # Not enough data to evaluate

        unexplained_count = 0
        total_windows = 0

        # Check 45-minute windows for unexplained changes
        glucose_sorted = glucose_df.sort_values('timestamp')
        values = glucose_sorted['value'].values
        timestamps = glucose_sorted['timestamp'].values

        for i in range(len(values) - 9):  # 45 min = 9 readings
            start_idx, end_idx = i, i + 9
            start_ts = pd.Timestamp(timestamps[start_idx])
            end_ts = pd.Timestamp(timestamps[end_idx])

            bg_change = values[end_idx] - values[start_idx]
            total_windows += 1

            # Get treatments in this window
            window_treatments = treatments_df[
                (treatments_df['timestamp'] >= start_ts) &
                (treatments_df['timestamp'] <= end_ts)
            ]

            has_carbs = window_treatments['carbs'].sum() > 0 if 'carbs' in window_treatments.columns else False
            has_insulin = window_treatments['insulin'].sum() > 0 if 'insulin' in window_treatments.columns else False

            # Unexplained rise: BG up > threshold without carbs
            if bg_change > self.unexplained_rise_threshold and not has_carbs:
                unexplained_count += 1

            # Unexplained drop: BG down > threshold without insulin
            if bg_change < -self.unexplained_drop_threshold and not has_insulin:
                unexplained_count += 1

        if total_windows == 0:
            return 1.0

        unexplained_ratio = unexplained_count / total_windows
        return max(0.0, 1.0 - unexplained_ratio * 2)  # Penalize heavily

    def filter_training_windows(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        window_hours: int = 4,
        stride_hours: int = 1,
    ) -> List[Tuple[datetime, datetime, float]]:
        """
        Filter data into training windows that meet quality threshold.

        Args:
            glucose_df: DataFrame with 'timestamp' and 'value' columns
            treatments_df: DataFrame with 'timestamp', 'carbs', 'insulin' columns
            window_hours: Size of each training window
            stride_hours: Hours to advance between windows

        Returns:
            List of (start, end, score) tuples for valid windows
        """
        if len(glucose_df) == 0:
            return []

        # Ensure timestamps are datetime
        glucose_df = glucose_df.copy()
        treatments_df = treatments_df.copy()

        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'])

        # Get date range
        min_ts = glucose_df['timestamp'].min()
        max_ts = glucose_df['timestamp'].max()

        valid_windows = []
        current_start = min_ts

        while current_start + timedelta(hours=window_hours) <= max_ts:
            window_end = current_start + timedelta(hours=window_hours)

            # Calculate completeness score
            score, details = self.calculate_completeness_score(
                glucose_df, treatments_df, current_start, window_end
            )

            if score >= self.min_completeness_score:
                valid_windows.append((current_start, window_end, score))
                logger.debug(
                    f"Valid window: {current_start} - {window_end}, "
                    f"score={score:.2f}, details={details}"
                )
            else:
                logger.debug(
                    f"Rejected window: {current_start} - {window_end}, "
                    f"score={score:.2f} < {self.min_completeness_score}"
                )

            current_start += timedelta(hours=stride_hours)

        logger.info(
            f"Found {len(valid_windows)} valid training windows "
            f"(out of {(max_ts - min_ts).total_seconds() / 3600 / stride_hours:.0f} total)"
        )

        return valid_windows

    def create_filtered_dataset(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        window_hours: int = 4,
        stride_hours: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[datetime, datetime, float]]]:
        """
        Create filtered dataset excluding low-quality periods.

        Returns:
            filtered_glucose: Glucose readings from valid windows only
            filtered_treatments: Treatments from valid windows only
            windows: List of (start, end, score) for valid windows
        """
        windows = self.filter_training_windows(
            glucose_df, treatments_df, window_hours, stride_hours
        )

        if not windows:
            logger.warning("No valid training windows found!")
            return pd.DataFrame(), pd.DataFrame(), []

        # Combine all valid windows
        glucose_dfs = []
        treatment_dfs = []

        for start, end, score in windows:
            window_glucose = glucose_df[
                (glucose_df['timestamp'] >= start) &
                (glucose_df['timestamp'] <= end)
            ].copy()
            window_glucose['window_score'] = score

            window_treatments = treatments_df[
                (treatments_df['timestamp'] >= start) &
                (treatments_df['timestamp'] <= end)
            ].copy()
            window_treatments['window_score'] = score

            glucose_dfs.append(window_glucose)
            treatment_dfs.append(window_treatments)

        filtered_glucose = pd.concat(glucose_dfs, ignore_index=True)
        filtered_treatments = pd.concat(treatment_dfs, ignore_index=True)

        # Remove duplicates (overlapping windows may include same readings)
        filtered_glucose = filtered_glucose.drop_duplicates(subset=['timestamp', 'value'])
        filtered_treatments = filtered_treatments.drop_duplicates(subset=['timestamp'])

        logger.info(
            f"Filtered dataset: {len(filtered_glucose)} glucose readings, "
            f"{len(filtered_treatments)} treatments "
            f"(from {len(windows)} valid windows)"
        )

        return filtered_glucose, filtered_treatments, windows

    def generate_quality_report(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
    ) -> Dict:
        """Generate a report on data quality."""
        # Get all windows and their scores
        window_hours = 4
        stride_hours = 1

        all_windows = []
        min_ts = glucose_df['timestamp'].min()
        max_ts = glucose_df['timestamp'].max()

        current = min_ts
        while current + timedelta(hours=window_hours) <= max_ts:
            end = current + timedelta(hours=window_hours)
            score, details = self.calculate_completeness_score(
                glucose_df, treatments_df, current, end
            )
            all_windows.append({
                'start': current,
                'end': end,
                'score': score,
                **details
            })
            current += timedelta(hours=stride_hours)

        if not all_windows:
            return {'error': 'No data windows found'}

        # Aggregate statistics
        scores = [w['score'] for w in all_windows]
        valid_count = sum(1 for s in scores if s >= self.min_completeness_score)

        # Time breakdown
        school_rejected = sum(
            1 for w in all_windows
            if w['school_penalty'] < 0.9 and w['score'] < self.min_completeness_score
        )

        unexplained_rejected = sum(
            1 for w in all_windows
            if w['unexplained_penalty'] < 0.8 and w['score'] < self.min_completeness_score
        )

        report = {
            'total_windows': len(all_windows),
            'valid_windows': valid_count,
            'valid_percentage': valid_count / len(all_windows) * 100,
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'school_rejected_count': school_rejected,
            'unexplained_rejected_count': unexplained_rejected,
            'total_glucose_readings': len(glucose_df),
            'total_treatments': len(treatments_df),
            'date_range': {
                'start': str(min_ts),
                'end': str(max_ts),
                'days': (max_ts - min_ts).days,
            },
        }

        return report


def filter_for_training(
    glucose_df: pd.DataFrame,
    treatments_df: pd.DataFrame,
    exclude_school: bool = True,
    min_score: float = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to filter data for training.

    Args:
        glucose_df: Raw glucose data
        treatments_df: Raw treatment data
        exclude_school: Whether to exclude school hours (default True)
        min_score: Minimum completeness score (default 0.6)

    Returns:
        Filtered glucose and treatment DataFrames
    """
    filter = TrainingDataQualityFilter(
        exclude_school_weekdays=exclude_school,
        min_completeness_score=min_score,
    )

    filtered_glucose, filtered_treatments, _ = filter.create_filtered_dataset(
        glucose_df, treatments_df
    )

    return filtered_glucose, filtered_treatments
