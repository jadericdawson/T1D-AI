"""
BG Inference Service
LSTM-based blood glucose prediction at +5, +10, +15 minutes.
"""
import torch
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from ..models.bg_predictor import BG_PredictorNet, BG_MODEL_CONFIG, BG_FEATURE_COLUMNS
from ..feature_engineering import (
    engineer_features,
    extract_feature_sequence,
    SEQ_LENGTH,
)

logger = logging.getLogger(__name__)


class BGInferenceService:
    """
    Blood Glucose prediction service using LSTM model.

    Predicts BG values at +5, +10, +15 minutes from current reading.
    Uses MinMaxScaler for feature and target normalization.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        features_scaler_path: Optional[Path] = None,
        targets_scaler_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        """
        Initialize the BG inference service.

        Args:
            model_path: Path to trained model (.pth file)
            features_scaler_path: Path to feature scaler (.pkl)
            targets_scaler_path: Path to target scaler (.pkl)
            feature_list_path: Path to feature list (.pkl)
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.model: Optional[BG_PredictorNet] = None
        self.features_scaler = None
        self.targets_scaler = None
        self.feature_columns: List[str] = BG_FEATURE_COLUMNS
        self._loaded = False

        # Default paths (can be overridden)
        self.model_path = model_path
        self.features_scaler_path = features_scaler_path
        self.targets_scaler_path = targets_scaler_path
        self.feature_list_path = feature_list_path

    def load(self) -> bool:
        """
        Load model and scalers from disk.

        Returns:
            True if loading successful, False otherwise
        """
        try:
            # Load model
            if self.model_path and self.model_path.exists():
                logger.info(f"Loading BG model from {self.model_path}")

                # Initialize model with config
                self.model = BG_PredictorNet(
                    n_features=BG_MODEL_CONFIG["n_features"],
                    out_steps=BG_MODEL_CONFIG["out_steps"],
                    hidden_size=BG_MODEL_CONFIG["hidden_size"],
                    num_layers=BG_MODEL_CONFIG["num_layers"],
                    dropout_prob=BG_MODEL_CONFIG["dropout_prob"]
                )

                # Load state dict
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True
                )
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info("BG model loaded successfully")
            else:
                logger.warning(f"BG model not found at {self.model_path}")
                return False

            # Load feature scaler
            if self.features_scaler_path and self.features_scaler_path.exists():
                with open(self.features_scaler_path, 'rb') as f:
                    self.features_scaler = pickle.load(f)
                logger.info("Feature scaler loaded")
            else:
                logger.warning(f"Feature scaler not found at {self.features_scaler_path}")
                return False

            # Load target scaler
            if self.targets_scaler_path and self.targets_scaler_path.exists():
                with open(self.targets_scaler_path, 'rb') as f:
                    self.targets_scaler = pickle.load(f)
                logger.info("Target scaler loaded")
            else:
                logger.warning(f"Target scaler not found at {self.targets_scaler_path}")
                return False

            # Load feature list (optional - we have defaults)
            if self.feature_list_path and self.feature_list_path.exists():
                with open(self.feature_list_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_columns)} features from file")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load BG model: {e}", exc_info=True)
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._loaded and self.model is not None

    def predict(
        self,
        feature_sequence: np.ndarray
    ) -> Optional[Tuple[List[float], List[float]]]:
        """
        Make BG prediction from feature sequence.

        Args:
            feature_sequence: Numpy array of shape (1, seq_length, n_features)
                              Already scaled if using raw values, otherwise
                              will be scaled internally.

        Returns:
            Tuple of (predictions in mg/dL, scaled predictions) or None on error.
            Predictions are for +5, +10, +15 minutes.
        """
        if not self.is_loaded:
            logger.error("Model not loaded. Call load() first.")
            return None

        try:
            # Scale features if scaler is available
            original_shape = feature_sequence.shape
            flat_features = feature_sequence.reshape(-1, original_shape[-1])

            if self.features_scaler is not None:
                scaled_features = self.features_scaler.transform(flat_features)
            else:
                scaled_features = flat_features

            # Reshape back to sequence
            scaled_sequence = scaled_features.reshape(original_shape)

            # Convert to tensor
            input_tensor = torch.tensor(
                scaled_sequence,
                dtype=torch.float32,
                device=self.device
            )

            # Run inference
            with torch.no_grad():
                scaled_output = self.model(input_tensor)

            # Convert to numpy
            scaled_predictions = scaled_output.cpu().numpy()

            # Inverse transform to get mg/dL values
            if self.targets_scaler is not None:
                predictions_mgdl = self.targets_scaler.inverse_transform(scaled_predictions)
            else:
                predictions_mgdl = scaled_predictions

            # Flatten to list
            predictions_list = predictions_mgdl[0].tolist()
            scaled_list = scaled_predictions[0].tolist()

            # Clamp to reasonable range (40-400 mg/dL)
            predictions_list = [
                max(40.0, min(400.0, p)) for p in predictions_list
            ]

            return predictions_list, scaled_list

        except Exception as e:
            logger.error(f"BG prediction failed: {e}", exc_info=True)
            return None

    def predict_from_dataframe(
        self,
        df,
        seq_length: int = SEQ_LENGTH
    ) -> Optional[Tuple[List[float], List[float]]]:
        """
        Make prediction from a DataFrame with raw glucose/treatment data.

        Args:
            df: DataFrame with columns matching feature requirements
            seq_length: Sequence length (default: 24 = 120 min)

        Returns:
            Tuple of (predictions in mg/dL, scaled predictions) or None
        """
        # Engineer features
        df_features = engineer_features(df)

        if df_features.empty:
            logger.error("Feature engineering returned empty DataFrame")
            return None

        # Extract sequence
        sequence = extract_feature_sequence(
            df_features,
            self.feature_columns,
            seq_length
        )

        if sequence is None:
            logger.error("Failed to extract feature sequence")
            return None

        return self.predict(sequence)

    def get_prediction_horizons(self) -> List[int]:
        """
        Get prediction time horizons in minutes.

        Returns:
            List of prediction horizons: [5, 10, 15]
        """
        return [5, 10, 15]


def create_bg_service(
    models_dir: Path,
    device: str = "cpu"
) -> BGInferenceService:
    """
    Factory function to create and load a BG inference service.

    Args:
        models_dir: Directory containing model files
        device: Device for inference

    Returns:
        Loaded BGInferenceService
    """
    service = BGInferenceService(
        model_path=models_dir / "bg_predictor_3step_v2.pth",
        features_scaler_path=models_dir / "bg_features_scaler_v2.pkl",
        targets_scaler_path=models_dir / "bg_targets_scaler_v2.pkl",
        feature_list_path=models_dir / "bg_feature_list_v2.pkl",
        device=device
    )

    if not service.load():
        logger.warning("Failed to load BG service - predictions will be unavailable")

    return service
