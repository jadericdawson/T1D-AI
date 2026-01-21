"""
Training Pipeline for Personalized Absorption Models

Trains IOB and COB models that learn individual insulin/carb absorption curves
from actual BG response data, replacing fixed exponential decay formulas.

Training Strategy:
- IOB Model: Find "clean" correction boluses (no carbs nearby) and track BG response
- COB Model: Find meal events and track BG response, accounting for insulin
- Both use the observed BG changes to infer actual absorption curves
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import tempfile
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ml.models.iob_model import (
    PersonalizedIOBModel,
    IOB_MODEL_CONFIG,
    create_iob_training_sample,
    estimate_remaining_iob_from_bg
)
from ml.models.cob_model import (
    PersonalizedCOBModel,
    COB_MODEL_CONFIG,
    create_cob_training_sample,
    estimate_remaining_cob_from_bg,
    get_absorption_rate_from_gi
)
from ml.models.tft_predictor import (
    TemporalFusionTransformer,
    TFT_MODEL_CONFIG,
    QuantileLoss,
    create_tft_model
)
from database.repositories import GlucoseRepository, TreatmentRepository

logger = logging.getLogger(__name__)


class AbsorptionModelTrainer:
    """
    Trains personalized IOB and COB models from user data.

    Key insight: We can infer actual absorption curves by analyzing how BG
    responds to insulin and carbs over time.
    """

    def __init__(
        self,
        device: str = "cpu",
        default_isf: float = 50.0,
        default_icr: float = 10.0,
        min_iob_samples: int = 50,
        min_cob_samples: int = 30,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize trainer.

        Args:
            device: Training device
            default_isf: Default insulin sensitivity factor (mg/dL per unit)
            default_icr: Default insulin to carb ratio (carbs per unit)
            min_iob_samples: Minimum correction events for IOB training
            min_cob_samples: Minimum meal events for COB training
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        self.device = device
        self.default_isf = default_isf
        self.default_icr = default_icr
        self.min_iob_samples = min_iob_samples
        self.min_cob_samples = min_cob_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._glucose_repo: Optional[GlucoseRepository] = None
        self._treatment_repo: Optional[TreatmentRepository] = None

    async def _get_glucose_repo(self) -> GlucoseRepository:
        if self._glucose_repo is None:
            self._glucose_repo = GlucoseRepository()
        return self._glucose_repo

    async def _get_treatment_repo(self) -> TreatmentRepository:
        if self._treatment_repo is None:
            self._treatment_repo = TreatmentRepository()
        return self._treatment_repo

    async def fetch_data(
        self,
        user_id: str,
        days: int = 90
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch glucose and treatment data."""
        since = datetime.now(timezone.utc) - timedelta(days=days)

        glucose_repo = await self._get_glucose_repo()
        treatment_repo = await self._get_treatment_repo()

        # Use get_history for glucose readings
        glucose_readings = await glucose_repo.get_history(
            user_id=user_id,
            start_time=since,
            limit=50000  # Get enough data
        )
        glucose_df = pd.DataFrame([r.model_dump() for r in glucose_readings])

        # Use get_recent for treatments
        treatments = await treatment_repo.get_recent(
            user_id=user_id,
            hours=days * 24
        )
        treatments_df = pd.DataFrame([t.model_dump() for t in treatments])

        logger.info(f"Fetched {len(glucose_df)} glucose, {len(treatments_df)} treatments")
        return glucose_df, treatments_df

    def find_correction_boluses(
        self,
        treatments_df: pd.DataFrame,
        glucose_df: pd.DataFrame,
        carb_free_window_hours: float = 2.0
    ) -> List[Dict]:
        """
        Find "clean" correction boluses with no carbs nearby.

        These are ideal for learning IOB curves because BG change
        is purely from insulin, not carbs.

        Args:
            treatments_df: Treatment data
            glucose_df: Glucose data
            carb_free_window_hours: Hours before/after with no carbs

        Returns:
            List of correction bolus events
        """
        if treatments_df.empty:
            return []

        # Ensure timestamps are datetime
        treatments_df = treatments_df.copy()
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
        glucose_df = glucose_df.copy()
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)

        # Get insulin and carb events
        insulin_events = treatments_df[
            (treatments_df.get('insulin', 0) > 0) |
            (treatments_df.get('type') == 'insulin')
        ].copy()

        carb_events = treatments_df[
            (treatments_df.get('carbs', 0) > 0) |
            (treatments_df.get('type') == 'carbs')
        ].copy()

        if insulin_events.empty:
            return []

        window_delta = timedelta(hours=carb_free_window_hours)
        clean_corrections = []

        for _, bolus in insulin_events.iterrows():
            bolus_time = bolus['timestamp']
            insulin_units = bolus.get('insulin', 0) or 0

            if insulin_units <= 0:
                continue

            # Check if any carbs within window
            nearby_carbs = carb_events[
                (carb_events['timestamp'] >= bolus_time - window_delta) &
                (carb_events['timestamp'] <= bolus_time + window_delta)
            ]

            if len(nearby_carbs) == 0:
                # This is a clean correction bolus!
                # Get BG at time of bolus
                bg_at_bolus = self._get_nearest_bg(glucose_df, bolus_time, max_min=5)

                if bg_at_bolus is not None:
                    clean_corrections.append({
                        'timestamp': bolus_time,
                        'insulin': insulin_units,
                        'bg_before': bg_at_bolus,
                        'hour': bolus_time.hour
                    })

        logger.info(f"Found {len(clean_corrections)} clean correction boluses")
        return clean_corrections

    def find_meal_events(
        self,
        treatments_df: pd.DataFrame,
        glucose_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Find meal events for COB training.

        Args:
            treatments_df: Treatment data
            glucose_df: Glucose data

        Returns:
            List of meal events with food composition
        """
        if treatments_df.empty:
            return []

        treatments_df = treatments_df.copy()
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
        glucose_df = glucose_df.copy()
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)

        # Get carb events
        carb_events = treatments_df[
            (treatments_df.get('carbs', 0) > 0) |
            (treatments_df.get('type') == 'carbs')
        ].copy()

        if carb_events.empty:
            return []

        meals = []
        for _, meal in carb_events.iterrows():
            meal_time = meal['timestamp']
            carbs = meal.get('carbs', 0) or 0

            if carbs <= 0:
                continue

            bg_at_meal = self._get_nearest_bg(glucose_df, meal_time, max_min=5)

            if bg_at_meal is not None:
                # Handle NaN values properly - pd.isna() catches NaN, None, NaT
                protein_val = meal.get('protein', 0)
                protein = 0.0 if pd.isna(protein_val) else float(protein_val)

                fat_val = meal.get('fat', 0)
                fat = 0.0 if pd.isna(fat_val) else float(fat_val)

                gi_val = meal.get('glycemicIndex', 55)
                glycemic_index = 55 if pd.isna(gi_val) else int(gi_val)

                meals.append({
                    'timestamp': meal_time,
                    'carbs': float(carbs),
                    'protein': protein,
                    'fat': fat,
                    'glycemic_index': glycemic_index,
                    'bg_before': bg_at_meal,
                    'hour': meal_time.hour
                })

        logger.info(f"Found {len(meals)} meal events")
        return meals

    def _get_nearest_bg(
        self,
        glucose_df: pd.DataFrame,
        target_time: datetime,
        max_min: int = 5
    ) -> Optional[float]:
        """Get BG value nearest to a target time."""
        if glucose_df.empty:
            return None

        time_diffs = abs(glucose_df['timestamp'] - target_time)
        min_diff = time_diffs.min()

        if min_diff > timedelta(minutes=max_min):
            return None

        closest_idx = time_diffs.idxmin()
        return float(glucose_df.loc[closest_idx, 'value'])

    def _get_bg_at_time(
        self,
        glucose_df: pd.DataFrame,
        target_time: datetime
    ) -> Optional[float]:
        """Get BG at a specific time (within 5 min)."""
        return self._get_nearest_bg(glucose_df, target_time, max_min=5)

    def create_iob_training_data(
        self,
        corrections: List[Dict],
        glucose_df: pd.DataFrame,
        isf: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data for IOB model from correction boluses.

        For each correction:
        - Track BG at 15, 30, 45, 60, 90, 120, 180, 240 minutes
        - Estimate remaining IOB from BG drop vs expected drop

        Args:
            corrections: List of correction bolus events
            glucose_df: Glucose DataFrame
            isf: Insulin sensitivity factor

        Returns:
            Tuple of (X, y) training arrays
        """
        glucose_df = glucose_df.copy()
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)

        features_list = []
        targets_list = []

        time_points = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360]  # Minutes

        for correction in corrections:
            bolus_time = correction['timestamp']
            insulin = correction['insulin']
            bg_before = correction['bg_before']
            hour = correction['hour']

            for minutes in time_points:
                target_time = bolus_time + timedelta(minutes=minutes)
                bg_at_t = self._get_bg_at_time(glucose_df, target_time)

                if bg_at_t is not None:
                    # Estimate remaining IOB from BG response
                    remaining_fraction = estimate_remaining_iob_from_bg(
                        bg_before, bg_at_t, insulin, isf
                    )

                    # Create training sample
                    X, y = create_iob_training_sample(
                        bolus_units=insulin,
                        minutes_since_bolus=minutes,
                        remaining_fraction=remaining_fraction,
                        hour=hour,
                        activity_level=0.0  # Default, could be enhanced
                    )

                    features_list.append(X)
                    targets_list.append(y)

        if not features_list:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(targets_list)

    def create_cob_training_data(
        self,
        meals: List[Dict],
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame,
        isf: float = 50.0,
        icr: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data for COB model from meal events.

        For each meal:
        - Track BG at multiple time points
        - Account for any insulin given
        - Estimate remaining COB from BG rise

        Args:
            meals: List of meal events
            glucose_df: Glucose DataFrame
            treatments_df: Treatment DataFrame (for insulin correction)
            isf: Insulin sensitivity factor
            icr: Insulin to carb ratio

        Returns:
            Tuple of (X, y) training arrays
        """
        glucose_df = glucose_df.copy()
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'], utc=True)
        treatments_df = treatments_df.copy()
        treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)

        features_list = []
        targets_list = []

        # Different time points for carbs (can be slower, especially high fat)
        time_points = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360]

        bg_per_gram = isf / icr  # Expected BG rise per gram of carbs

        for meal in meals:
            meal_time = meal['timestamp']
            carbs = meal['carbs']
            protein = meal.get('protein', 0)
            fat = meal.get('fat', 0)
            gi = meal.get('glycemic_index', 55)
            bg_before = meal['bg_before']
            hour = meal['hour']

            for minutes in time_points:
                target_time = meal_time + timedelta(minutes=minutes)
                bg_at_t = self._get_bg_at_time(glucose_df, target_time)

                if bg_at_t is not None:
                    # Calculate insulin effect during this period
                    insulin_effect = self._calculate_insulin_effect(
                        treatments_df, meal_time, target_time, isf
                    )

                    # Estimate remaining COB
                    remaining_fraction = estimate_remaining_cob_from_bg(
                        bg_before, bg_at_t, carbs, insulin_effect, bg_per_gram
                    )

                    # Create training sample
                    X, y = create_cob_training_sample(
                        carbs=carbs,
                        minutes_since_meal=minutes,
                        remaining_fraction=remaining_fraction,
                        protein=protein,
                        fat=fat,
                        glycemic_index=gi,
                        hour=hour
                    )

                    features_list.append(X)
                    targets_list.append(y)

        if not features_list:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(targets_list)

    def _calculate_insulin_effect(
        self,
        treatments_df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        isf: float
    ) -> float:
        """
        Calculate total insulin effect (BG lowering) between two times.

        Uses simple exponential decay model.
        """
        # Find insulin given near the meal
        window_start = start_time - timedelta(minutes=30)
        window_end = start_time + timedelta(hours=1)

        insulin_events = treatments_df[
            (treatments_df['timestamp'] >= window_start) &
            (treatments_df['timestamp'] <= window_end) &
            ((treatments_df.get('insulin', 0) > 0) |
             (treatments_df.get('type') == 'insulin'))
        ]

        total_effect = 0.0
        elapsed_min = (end_time - start_time).total_seconds() / 60

        for _, insulin_event in insulin_events.iterrows():
            insulin = insulin_event.get('insulin', 0) or 0
            if insulin > 0:
                event_time = insulin_event['timestamp']
                time_since_insulin = (end_time - event_time).total_seconds() / 60

                if time_since_insulin > 0:
                    # Simple exponential decay estimate
                    decay = 0.5 ** (time_since_insulin / 81.0)
                    absorbed = insulin * (1 - decay)
                    total_effect += absorbed * isf

        return total_effect

    def train_iob_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[PersonalizedIOBModel, Dict[str, float]]:
        """
        Train the personalized IOB model.

        Args:
            X: Features (n_samples, 5)
            y: Targets (n_samples, 1)

        Returns:
            Tuple of (trained_model, metrics)
        """
        n_samples = len(X)
        logger.info(f"Training IOB model with {n_samples} samples")

        # Split data
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create model
        model = PersonalizedIOBModel(
            input_size=IOB_MODEL_CONFIG["input_size"],
            hidden_sizes=IOB_MODEL_CONFIG["hidden_sizes"],
            dropout=IOB_MODEL_CONFIG["dropout"]
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 20 == 0:
                logger.info(f"IOB Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Calculate metrics
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            mae = np.mean(np.abs(val_preds - y_val))
            rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_val_loss": best_val_loss,
            "mae": float(mae),
            "rmse": float(rmse)
        }

        logger.info(f"IOB model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return model, metrics

    def train_cob_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[PersonalizedCOBModel, Dict[str, float]]:
        """
        Train the personalized COB model.

        Args:
            X: Features (n_samples, 7)
            y: Targets (n_samples, 1)

        Returns:
            Tuple of (trained_model, metrics)
        """
        n_samples = len(X)
        logger.info(f"Training COB model with {n_samples} samples")

        # Split data
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create model
        model = PersonalizedCOBModel(
            input_size=COB_MODEL_CONFIG["input_size"],
            hidden_sizes=COB_MODEL_CONFIG["hidden_sizes"],
            dropout=COB_MODEL_CONFIG["dropout"]
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 20 == 0:
                logger.info(f"COB Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Calculate metrics
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            mae = np.mean(np.abs(val_preds - y_val))
            rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_val_loss": best_val_loss,
            "mae": float(mae),
            "rmse": float(rmse)
        }

        logger.info(f"COB model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return model, metrics

    async def train_user_absorption_models(
        self,
        user_id: str,
        days: int = 90,
        isf: float = 50.0,
        icr: float = 10.0
    ) -> Dict[str, Any]:
        """
        Train personalized IOB and COB models for a user.

        Args:
            user_id: User ID
            days: Days of data to use
            isf: User's insulin sensitivity factor
            icr: User's insulin to carb ratio

        Returns:
            Training result with model paths and metrics
        """
        logger.info(f"Training absorption models for user {user_id}")

        # Fetch data
        glucose_df, treatments_df = await self.fetch_data(user_id, days)

        if glucose_df.empty or treatments_df.empty:
            return {
                "success": False,
                "error": "Insufficient data",
                "user_id": user_id
            }

        results = {"user_id": user_id, "success": True}

        # Train IOB model
        corrections = self.find_correction_boluses(treatments_df, glucose_df)

        if len(corrections) >= self.min_iob_samples:
            X_iob, y_iob = self.create_iob_training_data(
                corrections, glucose_df, isf
            )

            if len(X_iob) >= self.min_iob_samples:
                iob_model, iob_metrics = self.train_iob_model(X_iob, y_iob)
                results["iob_model_trained"] = True
                results["iob_metrics"] = iob_metrics

                # Save model
                temp_path = Path(tempfile.mktemp(suffix=".pth"))
                torch.save(iob_model.state_dict(), temp_path)
                results["iob_model_path"] = str(temp_path)
            else:
                results["iob_model_trained"] = False
                results["iob_error"] = f"Not enough training samples: {len(X_iob)}"
        else:
            results["iob_model_trained"] = False
            results["iob_error"] = f"Not enough correction boluses: {len(corrections)}"

        # Train COB model
        meals = self.find_meal_events(treatments_df, glucose_df)

        if len(meals) >= self.min_cob_samples:
            X_cob, y_cob = self.create_cob_training_data(
                meals, glucose_df, treatments_df, isf, icr
            )

            if len(X_cob) >= self.min_cob_samples:
                cob_model, cob_metrics = self.train_cob_model(X_cob, y_cob)
                results["cob_model_trained"] = True
                results["cob_metrics"] = cob_metrics

                # Save model
                temp_path = Path(tempfile.mktemp(suffix=".pth"))
                torch.save(cob_model.state_dict(), temp_path)
                results["cob_model_path"] = str(temp_path)
            else:
                results["cob_model_trained"] = False
                results["cob_error"] = f"Not enough training samples: {len(X_cob)}"
        else:
            results["cob_model_trained"] = False
            results["cob_error"] = f"Not enough meal events: {len(meals)}"

        results["success"] = results.get("iob_model_trained", False) or results.get("cob_model_trained", False)

        return results

    async def close(self):
        """Clean up resources."""
        pass


class TFTTrainer:
    """
    Trainer for Temporal Fusion Transformer model.

    Trains the TFT for long-horizon glucose predictions (30/45/60 min)
    with uncertainty quantification.
    """

    def __init__(
        self,
        device: str = "cpu",
        n_features: int = 65,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        encoder_length: int = 24,
        prediction_length: int = 12
    ):
        self.device = device
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length

    def create_tft_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for TFT training.

        Args:
            df: DataFrame with all features
            feature_columns: List of feature column names

        Returns:
            Tuple of (X, y) where:
            - X: (n_samples, encoder_length, n_features)
            - y: (n_samples, 3) for 30, 45, 60 min predictions
        """
        total_length = self.encoder_length + self.prediction_length

        # Ensure all features exist
        available_features = [c for c in feature_columns if c in df.columns]

        if len(available_features) < len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            logger.warning(f"Missing features for TFT: {missing}")

        feature_data = df[available_features].values.astype(np.float32)
        target_data = df['value'].values.astype(np.float32)

        sequences = []
        targets = []

        # Horizon indices: 30, 45, 60 min = steps 6, 9, 12 (at 5 min sampling)
        horizon_steps = [5, 8, 11]  # 0-indexed from end of encoder

        for i in range(len(feature_data) - total_length + 1):
            seq = feature_data[i:i + self.encoder_length]

            # Get targets at 30, 45, 60 min horizons
            target_30 = target_data[i + self.encoder_length + horizon_steps[0]]
            target_45 = target_data[i + self.encoder_length + horizon_steps[1]]
            target_60 = target_data[i + self.encoder_length + horizon_steps[2]]

            if not np.isnan(seq).any() and not np.isnan([target_30, target_45, target_60]).any():
                sequences.append(seq)
                targets.append([target_30, target_45, target_60])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        logger.info(f"Created {len(X)} TFT training sequences")
        return X, y

    def train_tft(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[TemporalFusionTransformer, Dict[str, float]]:
        """
        Train the TFT model.

        Args:
            X: Input sequences (n_samples, encoder_length, n_features)
            y: Targets (n_samples, 3) for 30/45/60 min

        Returns:
            Tuple of (trained_model, metrics)
        """
        n_samples, _, n_features = X.shape
        logger.info(f"Training TFT with {n_samples} samples, {n_features} features")

        # Split data
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
        X_val_2d = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)

        # Scale targets (important for stable training)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        self.y_scaler = y_scaler  # Save for inverse transform
        self.y_mean = y_scaler.mean_
        self.y_std = y_scaler.scale_
        logger.info(f"Target scaling: mean={self.y_mean}, std={self.y_std}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)

        # Expand targets for 3 quantiles (10th, 50th, 90th)
        # y shape: (batch, 3) -> need (batch, 3, 1) for loss calculation
        y_train_t = torch.FloatTensor(y_train_scaled).unsqueeze(-1).to(self.device)
        y_val_t = torch.FloatTensor(y_val_scaled).unsqueeze(-1).to(self.device)

        # Create model
        model = create_tft_model(n_features=n_features).to(self.device)

        # Training setup
        criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize MLflow tracking
        mlflow_run = None
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("T1D-AI-TFT-Training")
            mlflow_run = mlflow.start_run(run_name=f"tft_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                "n_samples": n_samples,
                "n_features": n_features,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "device": str(self.device)
            })

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                predictions = output['predictions']  # (batch, 3, 3) - 3 horizons, 3 quantiles

                # Use median (50th percentile) for loss calculation
                loss = criterion(predictions, batch_y.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Validation (batched to avoid OOM)
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_batch_size = min(256, len(X_val_t))
                for i in range(0, len(X_val_t), val_batch_size):
                    batch_X = X_val_t[i:i+val_batch_size]
                    batch_y = y_val_t[i:i+val_batch_size]
                    val_output = model(batch_X)
                    val_preds = val_output['predictions']
                    batch_loss = criterion(val_preds, batch_y.squeeze(-1)).item()
                    val_losses.append(batch_loss * len(batch_X))
                val_loss = sum(val_losses) / len(X_val_t)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Log every epoch for visibility
            logger.info(f"TFT Epoch {epoch+1}/{self.epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

            # Log to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)

            if patience_counter >= 15:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Calculate metrics (using median predictions, batched to avoid OOM)
        model.eval()
        all_preds = []
        with torch.no_grad():
            val_batch_size = min(256, len(X_val_t))
            for i in range(0, len(X_val_t), val_batch_size):
                batch_X = X_val_t[i:i+val_batch_size]
                val_output = model(batch_X)
                batch_preds = val_output['predictions'][:, :, 1].cpu().numpy()  # Median
                all_preds.append(batch_preds)
            val_preds_scaled = np.concatenate(all_preds, axis=0)

            # Inverse transform predictions back to original scale
            val_preds = self.y_scaler.inverse_transform(val_preds_scaled)
            y_val_np = y_val  # Original unscaled values

            mae = np.mean(np.abs(val_preds - y_val_np))
            rmse = np.sqrt(np.mean((val_preds - y_val_np) ** 2))

            # Per-horizon MAE
            mae_30 = np.mean(np.abs(val_preds[:, 0] - y_val_np[:, 0]))
            mae_45 = np.mean(np.abs(val_preds[:, 1] - y_val_np[:, 1]))
            mae_60 = np.mean(np.abs(val_preds[:, 2] - y_val_np[:, 2]))

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "best_val_loss": best_val_loss,
            "mae": float(mae),
            "rmse": float(rmse),
            "mae_30min": float(mae_30),
            "mae_45min": float(mae_45),
            "mae_60min": float(mae_60)
        }

        logger.info(f"TFT trained. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        logger.info(f"  30min MAE: {mae_30:.2f}, 45min MAE: {mae_45:.2f}, 60min MAE: {mae_60:.2f}")

        # Log final metrics and model to MLflow
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({
                "final_mae": float(mae),
                "final_rmse": float(rmse),
                "mae_30min": float(mae_30),
                "mae_45min": float(mae_45),
                "mae_60min": float(mae_60),
                "best_val_loss": best_val_loss
            })
            # Log model artifact
            mlflow.pytorch.log_model(model, "tft_model")
            mlflow.end_run()
            logger.info("MLflow run completed - metrics and model logged")

        return model, metrics


async def train_all_models(
    user_id: str,
    days: int = 90,
    isf: float = 50.0,
    icr: float = 10.0
) -> Dict[str, Any]:
    """
    Train all ML models for a user: IOB, COB, and TFT.

    Args:
        user_id: User ID
        days: Days of data
        isf: Insulin sensitivity factor
        icr: Insulin to carb ratio

    Returns:
        Combined training results
    """
    results = {"user_id": user_id}

    # Train absorption models
    absorption_trainer = AbsorptionModelTrainer()
    absorption_results = await absorption_trainer.train_user_absorption_models(
        user_id, days, isf, icr
    )
    results["absorption"] = absorption_results

    # TFT training would need feature-engineered data
    # This is typically done as part of the main trainer pipeline

    results["success"] = absorption_results.get("success", False)
    return results
