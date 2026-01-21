"""
Per-User Model Trainer for T1D-AI
Trains personalized blood glucose prediction models using user-specific data.

Usage:
    # As a script
    python -m ml.training.trainer --user_id USER_ID --days 30

    # As a module
    from ml.training.trainer import UserModelTrainer
    trainer = UserModelTrainer()
    result = await trainer.train_user_model(user_id, days=30)
"""
import argparse
import asyncio
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ml.models.bg_predictor import BG_PredictorNet, BG_MODEL_CONFIG
from ml.feature_engineering import (
    engineer_features,
    BG_FEATURE_COLUMNS,
    SEQ_LENGTH,
    SAMPLING_MIN
)
from ml.training.model_manager import get_model_manager, ModelManager
from ml.training.isf_learner import ISFLearner
from database.repositories import GlucoseRepository, TreatmentRepository

logger = logging.getLogger(__name__)


class UserModelTrainer:
    """
    Trains personalized BG prediction models for individual users.

    Uses data from CosmosDB (glucose_readings and treatments) to fine-tune
    models starting from base weights, or train from scratch if the user
    has enough data (>7 days).
    """

    def __init__(
        self,
        device: str = "cpu",
        min_days: int = 7,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2
    ):
        """
        Initialize the trainer.

        Args:
            device: Training device ("cpu" or "cuda")
            min_days: Minimum days of data required for training
            epochs: Maximum training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            early_stopping_patience: Epochs without improvement before stopping
            validation_split: Fraction of data for validation
        """
        self.device = device
        self.min_days = min_days
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split

        self._model_manager: Optional[ModelManager] = None
        self._glucose_repo: Optional[GlucoseRepository] = None
        self._treatment_repo: Optional[TreatmentRepository] = None

    async def _get_model_manager(self) -> ModelManager:
        """Get or create model manager."""
        if self._model_manager is None:
            self._model_manager = get_model_manager()
        return self._model_manager

    async def _get_glucose_repo(self) -> GlucoseRepository:
        """Get or create glucose repository."""
        if self._glucose_repo is None:
            self._glucose_repo = GlucoseRepository()
        return self._glucose_repo

    async def _get_treatment_repo(self) -> TreatmentRepository:
        """Get or create treatment repository."""
        if self._treatment_repo is None:
            self._treatment_repo = TreatmentRepository()
        return self._treatment_repo

    async def fetch_user_data(
        self,
        user_id: str,
        days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch glucose and treatment data for a user from CosmosDB.

        Args:
            user_id: User ID
            days: Number of days of data to fetch

        Returns:
            Tuple of (glucose_df, treatments_df)
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)

        glucose_repo = await self._get_glucose_repo()
        treatment_repo = await self._get_treatment_repo()

        # Fetch glucose readings
        glucose_readings = await glucose_repo.get_by_user(user_id, since=since)
        glucose_df = pd.DataFrame([r.dict() for r in glucose_readings])

        # Fetch treatments
        treatments = await treatment_repo.get_by_user(user_id, since=since)
        treatments_df = pd.DataFrame([t.dict() for t in treatments])

        logger.info(
            f"Fetched {len(glucose_df)} glucose readings and "
            f"{len(treatments_df)} treatments for user {user_id}"
        )

        return glucose_df, treatments_df

    def prepare_data(
        self,
        glucose_df: pd.DataFrame,
        treatments_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge glucose and treatment data and prepare for feature engineering.

        Args:
            glucose_df: DataFrame of glucose readings
            treatments_df: DataFrame of treatments

        Returns:
            Combined DataFrame ready for feature engineering
        """
        if glucose_df.empty:
            return pd.DataFrame()

        # Prepare glucose data
        df = glucose_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Initialize treatment columns
        df['insulin'] = 0.0
        df['carbs'] = 0.0
        df['protein'] = 0.0
        df['fat'] = 0.0

        # Merge treatments with nearest glucose reading
        if not treatments_df.empty:
            treatments_df['timestamp'] = pd.to_datetime(
                treatments_df['timestamp'], utc=True
            )

            for _, treat in treatments_df.iterrows():
                treat_time = treat['timestamp']

                # Find closest glucose reading within 5 minutes
                time_diffs = abs(df['timestamp'] - treat_time)
                closest_idx = time_diffs.idxmin()
                min_diff = time_diffs.min()

                if min_diff <= timedelta(minutes=SAMPLING_MIN):
                    if treat.get('type') == 'insulin':
                        df.loc[closest_idx, 'insulin'] += treat.get('insulin', 0) or 0
                    else:
                        df.loc[closest_idx, 'carbs'] += treat.get('carbs', 0) or 0
                        df.loc[closest_idx, 'protein'] += treat.get('protein', 0) or 0
                        df.loc[closest_idx, 'fat'] += treat.get('fat', 0) or 0

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        seq_length: int = SEQ_LENGTH,
        out_steps: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input/output sequences for LSTM training.

        Args:
            df: DataFrame with engineered features
            seq_length: Input sequence length (default: 24 = 2 hours)
            out_steps: Output prediction steps (default: 3 = +5, +10, +15 min)

        Returns:
            Tuple of (X, y) arrays
        """
        # Ensure all feature columns exist
        missing_cols = [c for c in BG_FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.error(f"Missing feature columns: {missing_cols}")
            return np.array([]), np.array([])

        # Get feature matrix
        feature_data = df[BG_FEATURE_COLUMNS].values.astype(np.float32)
        target_data = df['value'].values.astype(np.float32)

        sequences = []
        targets = []

        # Create sequences
        for i in range(len(feature_data) - seq_length - out_steps + 1):
            seq = feature_data[i:i + seq_length]
            target = target_data[i + seq_length:i + seq_length + out_steps]

            # Skip if any NaN
            if not np.isnan(seq).any() and not np.isnan(target).any():
                sequences.append(seq)
                targets.append(target)

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        logger.info(f"Created {len(X)} training sequences")
        return X, y

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, dict, StandardScaler, StandardScaler]:
        """
        Train the BG prediction model.

        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Target values (n_samples, out_steps)
            base_model: Optional pre-trained model for fine-tuning

        Returns:
            Tuple of (trained_model, metrics, features_scaler, targets_scaler)
        """
        n_samples, seq_length, n_features = X.shape
        _, out_steps = y.shape

        # Scale features
        X_reshaped = X.reshape(-1, n_features)
        features_scaler = StandardScaler()
        X_scaled = features_scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_length, n_features)

        # Scale targets
        targets_scaler = StandardScaler()
        y_scaled = targets_scaler.fit_transform(y)

        # Split into train/validation
        split_idx = int(n_samples * (1 - self.validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create dataloaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create model
        if base_model is not None:
            model = base_model.to(self.device)
            logger.info("Fine-tuning from base model")
        else:
            model = BG_PredictorNet(
                n_features=n_features,
                out_steps=out_steps,
                **{k: v for k, v in BG_MODEL_CONFIG.items()
                   if k in ['hidden_size', 'num_layers', 'dropout_prob']}
            ).to(self.device)
            logger.info("Training model from scratch")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Calculate metrics
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            val_preds_unscaled = targets_scaler.inverse_transform(val_preds)
            y_val_unscaled = targets_scaler.inverse_transform(y_val)

            mae = np.mean(np.abs(val_preds_unscaled - y_val_unscaled))
            rmse = np.sqrt(np.mean((val_preds_unscaled - y_val_unscaled) ** 2))

            # Per-horizon MAE
            mae_5 = np.mean(np.abs(val_preds_unscaled[:, 0] - y_val_unscaled[:, 0]))
            mae_10 = np.mean(np.abs(val_preds_unscaled[:, 1] - y_val_unscaled[:, 1]))
            mae_15 = np.mean(np.abs(val_preds_unscaled[:, 2] - y_val_unscaled[:, 2]))

        metrics = {
            "epochs_trained": epoch + 1,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "final_train_loss": train_losses[-1],
            "best_val_loss": best_val_loss,
            "mae": float(mae),
            "rmse": float(rmse),
            "mae_5min": float(mae_5),
            "mae_10min": float(mae_10),
            "mae_15min": float(mae_15),
        }

        logger.info(f"Training complete. MAE: {mae:.2f} mg/dL, RMSE: {rmse:.2f} mg/dL")

        return model, metrics, features_scaler, targets_scaler

    async def train_user_model(
        self,
        user_id: str,
        days: int = 30,
        learn_isf: bool = True
    ) -> dict:
        """
        Train a personalized model for a user.

        Args:
            user_id: User ID
            days: Days of data to use for training
            learn_isf: Whether to also learn ISF

        Returns:
            Training result dictionary
        """
        logger.info(f"Starting model training for user {user_id}")

        # Fetch data
        glucose_df, treatments_df = await self.fetch_user_data(user_id, days)

        if len(glucose_df) < SEQ_LENGTH + 3:
            return {
                "success": False,
                "error": f"Insufficient data: {len(glucose_df)} readings "
                        f"(need at least {SEQ_LENGTH + 3})",
                "user_id": user_id
            }

        # Check minimum days
        if not glucose_df.empty:
            time_range = (
                glucose_df['timestamp'].max() - glucose_df['timestamp'].min()
            )
            data_days = time_range.days if hasattr(time_range, 'days') else 0

            if data_days < self.min_days:
                return {
                    "success": False,
                    "error": f"Insufficient data: {data_days} days (need {self.min_days})",
                    "user_id": user_id
                }

        # Prepare and engineer features
        df = self.prepare_data(glucose_df, treatments_df)
        df = engineer_features(df)

        if df.empty:
            return {
                "success": False,
                "error": "Feature engineering failed",
                "user_id": user_id
            }

        # Create training sequences
        X, y = self.create_sequences(df)

        if len(X) < 100:
            return {
                "success": False,
                "error": f"Too few training sequences: {len(X)} (need at least 100)",
                "user_id": user_id
            }

        # Try to load base model for fine-tuning
        model_manager = await self._get_model_manager()
        base_model = None
        try:
            base_model_path = await model_manager.get_base_model_path()
            if base_model_path:
                base_model = torch.load(base_model_path, map_location=self.device)
                logger.info("Loaded base model for fine-tuning")
        except Exception as e:
            logger.warning(f"Could not load base model: {e}. Training from scratch.")

        # Train model
        model, metrics, features_scaler, targets_scaler = self.train_model(
            X, y, base_model
        )

        # Save model to temp file
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / "bg_predictor.pth"
        torch.save(model.state_dict(), model_path)

        # Upload to blob storage
        metadata = {
            "userId": user_id,
            "trainedAt": datetime.now(timezone.utc).isoformat(),
            "dataRangeDays": days,
            "totalSequences": len(X),
            **metrics
        }

        await model_manager.upload_user_model(user_id, model_path, metadata)
        await model_manager.upload_user_scalers(
            user_id, features_scaler, targets_scaler
        )

        # Learn ISF if requested
        isf_result = None
        if learn_isf:
            try:
                isf_learner = ISFLearner()
                isf_result = await isf_learner.learn_all_isf(user_id, days)
                logger.info(f"Learned ISF for user {user_id}")
            except Exception as e:
                logger.warning(f"ISF learning failed: {e}")

        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "success": True,
            "user_id": user_id,
            "metrics": metrics,
            "isf_learned": isf_result is not None,
            "model_url": f"users/{user_id}/bg_predictor.pth"
        }

    async def close(self):
        """Close resources."""
        if self._model_manager:
            await self._model_manager.close()


async def train_all_users(min_days: int = 7, days: int = 30) -> List[dict]:
    """
    Train models for all users with sufficient data.

    Args:
        min_days: Minimum days of data required
        days: Days of data to use for training

    Returns:
        List of training results
    """
    from database.repositories import DataSourceRepository

    datasource_repo = DataSourceRepository()
    model_manager = get_model_manager()

    # Get all users with connected datasources
    # This would query CosmosDB for all connected datasources
    # For now, we'll list users from existing models + datasources
    users = await model_manager.list_user_models()

    # Get users from datasources
    # TODO: Query datasources container for all connected users

    trainer = UserModelTrainer(min_days=min_days)
    results = []

    for user_id in users:
        try:
            result = await trainer.train_user_model(user_id, days)
            results.append(result)
        except Exception as e:
            logger.error(f"Training failed for user {user_id}: {e}")
            results.append({
                "success": False,
                "user_id": user_id,
                "error": str(e)
            })

    await trainer.close()

    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Training complete: {successful}/{len(results)} successful")

    return results


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train personalized BG prediction models"
    )
    parser.add_argument(
        "--user_id",
        type=str,
        help="User ID to train model for (optional, trains all if not specified)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of data to use for training (default: 30)"
    )
    parser.add_argument(
        "--min_days",
        type=int,
        default=7,
        help="Minimum days of data required (default: 7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Training device (default: cpu)"
    )
    parser.add_argument(
        "--no-isf",
        action="store_true",
        help="Skip ISF learning"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.user_id:
        # Train single user
        trainer = UserModelTrainer(
            device=args.device,
            min_days=args.min_days
        )
        result = await trainer.train_user_model(
            args.user_id,
            days=args.days,
            learn_isf=not args.no_isf
        )
        await trainer.close()

        if result["success"]:
            print(f"Training successful for user {args.user_id}")
            print(f"  MAE: {result['metrics']['mae']:.2f} mg/dL")
            print(f"  RMSE: {result['metrics']['rmse']:.2f} mg/dL")
        else:
            print(f"Training failed: {result.get('error')}")

    else:
        # Train all users
        results = await train_all_users(
            min_days=args.min_days,
            days=args.days
        )

        successful = sum(1 for r in results if r.get("success"))
        print(f"\nTraining complete: {successful}/{len(results)} successful")

        for result in results:
            status = "OK" if result["success"] else "FAILED"
            print(f"  {result['user_id']}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
