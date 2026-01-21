"""
CosmosDB Training Data Loader

Fetches user glucose and treatment data from CosmosDB for personalized model training.
Replaces local JSONL file-based training with cloud-native data access.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from database.repositories import GlucoseRepository, TreatmentRepository, UserRepository
from models.schemas import GlucoseReading, Treatment

logger = logging.getLogger(__name__)


class CosmosTrainingDataLoader:
    """
    Load training data from CosmosDB for a specific user.

    Provides pandas DataFrames compatible with the TFT training pipeline.
    """

    def __init__(self):
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()
        self.user_repo = UserRepository()

    async def get_user_training_data(
        self,
        user_id: str,
        days: int = 90,
        min_readings: int = 500,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Fetch glucose and treatment data for a user.

        Args:
            user_id: User ID to fetch data for
            days: Number of days of history to fetch
            min_readings: Minimum readings required for training

        Returns:
            Tuple of (glucose_df, treatments_df, metadata)

        Raises:
            ValueError: If insufficient data for training
        """
        logger.info(f"Loading training data for user {user_id} (last {days} days)")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Fetch glucose readings
        glucose_readings = await self.glucose_repo.get_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=50000
        )

        if len(glucose_readings) < min_readings:
            raise ValueError(
                f"Insufficient data for training. Need {min_readings} readings, "
                f"found {len(glucose_readings)}"
            )

        # Fetch treatments
        treatments = await self._get_treatments_history(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )

        # Convert to DataFrames
        glucose_df = self._readings_to_dataframe(glucose_readings)
        treatments_df = self._treatments_to_dataframe(treatments)

        # Calculate metadata
        metadata = self._calculate_metadata(glucose_df, treatments_df, user_id)

        logger.info(
            f"Loaded {len(glucose_df)} glucose readings and "
            f"{len(treatments_df)} treatments for user {user_id}"
        )

        return glucose_df, treatments_df, metadata

    def _normalize_timestamp(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-naive UTC for consistent comparison."""
        if dt is None:
            return None
        # If timezone-aware, convert to UTC and remove tzinfo
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    async def _get_treatments_history(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> list:
        """Fetch all treatments in time range."""
        # Normalize times to naive UTC for comparison
        start_naive = self._normalize_timestamp(start_time)
        end_naive = self._normalize_timestamp(end_time)

        # Get treatments in batches by day to avoid query limits
        all_treatments = []
        current = start_time

        while current < end_time:
            batch_end = min(current + timedelta(days=7), end_time)
            hours = int((batch_end - current).total_seconds() / 3600)

            # Temporarily adjust for query
            batch = await self.treatment_repo.get_recent(
                user_id=user_id,
                hours=hours + int((datetime.now(timezone.utc) - batch_end).total_seconds() / 3600)
            )

            # Filter to actual range (using normalized timestamps)
            for t in batch:
                t_time = self._normalize_timestamp(t.timestamp)
                if t_time and start_naive <= t_time <= end_naive:
                    all_treatments.append(t)

            current = batch_end

        return all_treatments

    def _readings_to_dataframe(self, readings: list) -> pd.DataFrame:
        """Convert glucose readings to training DataFrame format."""
        records = []
        for r in readings:
            # Map trend string to numeric value for training
            trend_val = 0
            if r.trend:
                trend_map = {
                    "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                    "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
                }
                trend_val = trend_map.get(str(r.trend), 0)

            # Normalize timestamp to naive UTC
            ts = self._normalize_timestamp(r.timestamp)

            records.append({
                'timestamp': ts,
                'value': r.value,
                'trend': trend_val,
                # IOB/COB from reading if populated (for enriched readings)
                'iob': getattr(r, 'iob', None) or 0.0,
                'cob': getattr(r, 'cob', None) or 0.0,
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp', 'value'])

        return df

    def _treatments_to_dataframe(self, treatments: list) -> pd.DataFrame:
        """Convert treatments to training DataFrame format."""
        records = []
        for t in treatments:
            # Normalize timestamp to naive UTC
            ts = self._normalize_timestamp(t.timestamp)
            records.append({
                'timestamp': ts,
                'type': t.type,
                'insulin': t.insulin or 0.0,
                'carbs': t.carbs or 0.0,
                'protein': t.protein or 0.0,
                'fat': t.fat or 0.0,
            })

        if not records:
            return pd.DataFrame(columns=[
                'timestamp', 'type', 'insulin', 'carbs', 'protein', 'fat'
            ])

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def _calculate_metadata(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        user_id: str
    ) -> Dict[str, Any]:
        """Calculate training metadata and statistics."""
        # Calculate data span in days
        data_span_days = 0
        if len(glucose_df) > 0:
            time_diff = glucose_df['timestamp'].max() - glucose_df['timestamp'].min()
            data_span_days = max(1, int(time_diff.total_seconds() / 86400))

        # Calculate totals
        total_insulin = float(treatments_df['insulin'].sum()) if len(treatments_df) > 0 else 0.0
        total_carbs = float(treatments_df['carbs'].sum()) if len(treatments_df) > 0 else 0.0

        # Calculate per-day averages
        readings_per_day = len(glucose_df) / max(1, data_span_days)
        treatments_per_day = len(treatments_df) / max(1, data_span_days)

        return {
            'user_id': user_id,
            # Frontend-expected field names
            'totalReadings': len(glucose_df),
            'totalTreatments': len(treatments_df),
            'totalInsulin': round(total_insulin, 1),
            'totalCarbs': round(total_carbs, 1),
            'dataSpanDays': data_span_days,
            'readingsPerDay': round(readings_per_day, 1),
            'treatmentsPerDay': round(treatments_per_day, 1),
            'oldestData': glucose_df['timestamp'].min().isoformat() if len(glucose_df) > 0 else None,
            'newestData': glucose_df['timestamp'].max().isoformat() if len(glucose_df) > 0 else None,
            # Additional stats (keep for backwards compatibility)
            'glucose_count': len(glucose_df),
            'treatment_count': len(treatments_df),
            'bg_mean': float(glucose_df['value'].mean()) if len(glucose_df) > 0 else None,
            'bg_std': float(glucose_df['value'].std()) if len(glucose_df) > 0 else None,
            'bg_min': float(glucose_df['value'].min()) if len(glucose_df) > 0 else None,
            'bg_max': float(glucose_df['value'].max()) if len(glucose_df) > 0 else None,
            'insulin_boluses': int((treatments_df['insulin'] > 0).sum()) if len(treatments_df) > 0 else 0,
            'carb_entries': int((treatments_df['carbs'] > 0).sum()) if len(treatments_df) > 0 else 0,
            'loaded_at': datetime.now(timezone.utc).isoformat(),
        }

    async def check_training_eligibility(
        self,
        user_id: str,
        min_days: int = 7,
        min_readings: int = 500,
        min_treatments: int = 20
    ) -> Tuple[bool, str]:
        """
        Check if a user has enough data for model training.

        Returns:
            Tuple of (is_eligible, reason)
        """
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=min_days)

            # Check glucose readings
            readings = await self.glucose_repo.get_history(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time,
                limit=min_readings + 1
            )

            if len(readings) < min_readings:
                return False, f"Need {min_readings} readings, have {len(readings)}"

            # Check treatments
            treatments = await self.treatment_repo.get_recent(
                user_id=user_id,
                hours=min_days * 24
            )

            if len(treatments) < min_treatments:
                return False, f"Need {min_treatments} treatments, have {len(treatments)}"

            return True, "Eligible for training"

        except Exception as e:
            logger.error(f"Error checking eligibility for {user_id}: {e}")
            return False, f"Error: {str(e)}"

    async def get_training_stats(self, user_id: str) -> Dict[str, Any]:
        """Get summary statistics about user's training data."""
        try:
            glucose_df, treatments_df, metadata = await self.get_user_training_data(
                user_id=user_id,
                days=90,
                min_readings=0  # Don't fail, just return what we have
            )
            return metadata
        except Exception as e:
            logger.error(f"Error getting training stats for {user_id}: {e}")
            return {
                'user_id': user_id,
                'error': str(e),
                'totalReadings': 0,
                'totalTreatments': 0,
                'totalInsulin': 0,
                'totalCarbs': 0,
                'dataSpanDays': 0,
                'readingsPerDay': 0,
                'treatmentsPerDay': 0,
                'oldestData': None,
                'newestData': None,
                # Legacy fields
                'glucose_count': 0,
                'treatment_count': 0,
            }
