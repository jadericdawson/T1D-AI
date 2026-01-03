"""
ISF Inference Service
LSTM-based Insulin Sensitivity Factor prediction.
"""
import torch
import pickle
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np

from ..models.isf_net import ISFNet, ISF_MODEL_CONFIG

logger = logging.getLogger(__name__)


class ISFInferenceService:
    """
    Insulin Sensitivity Factor (ISF) prediction service.

    Predicts how much 1 unit of insulin will lower blood glucose.
    Uses LSTM with softplus output to ensure positive values.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        features_scaler_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
        device: str = "cpu"
    ):
        """
        Initialize the ISF inference service.

        Args:
            model_path: Path to trained model (.pth file)
            features_scaler_path: Path to feature scaler (.pkl)
            feature_list_path: Path to feature list (.pkl)
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.model: Optional[ISFNet] = None
        self.features_scaler = None
        self.feature_columns: List[str] = []
        self._loaded = False

        self.model_path = model_path
        self.features_scaler_path = features_scaler_path
        self.feature_list_path = feature_list_path

        # Default ISF fallback (if model fails)
        self.default_isf = 50.0  # 1U drops BG by 50 mg/dL

    def load(self) -> bool:
        """
        Load model and scalers from disk.

        Returns:
            True if loading successful, False otherwise
        """
        try:
            # Load feature list first to get n_feat
            n_feat = ISF_MODEL_CONFIG["n_feat"]

            if self.feature_list_path and self.feature_list_path.exists():
                with open(self.feature_list_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                n_feat = len(self.feature_columns)
                logger.info(f"Loaded {n_feat} ISF features from file")

            # Load model
            if self.model_path and self.model_path.exists():
                logger.info(f"Loading ISF model from {self.model_path}")

                # Try to infer dimensions from saved model
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True
                )

                # Infer dimensions from state dict
                if 'lstm.weight_ih_l0' in state_dict:
                    n_feat = state_dict['lstm.weight_ih_l0'].shape[1]
                    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                else:
                    hidden_size = ISF_MODEL_CONFIG["hidden_size"]

                # Count LSTM layers
                num_layers = 1
                for key in state_dict.keys():
                    if key.startswith('lstm.weight_ih_l'):
                        layer_num = int(key.split('_l')[1])
                        num_layers = max(num_layers, layer_num + 1)

                logger.info(
                    f"Inferred model config: n_feat={n_feat}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}"
                )

                # Initialize model
                self.model = ISFNet(
                    n_feat=n_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=ISF_MODEL_CONFIG["dropout"]
                )

                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info("ISF model loaded successfully")
            else:
                logger.warning(f"ISF model not found at {self.model_path}")
                return False

            # Load feature scaler
            if self.features_scaler_path and self.features_scaler_path.exists():
                with open(self.features_scaler_path, 'rb') as f:
                    self.features_scaler = pickle.load(f)
                logger.info("ISF feature scaler loaded")
            else:
                logger.warning(
                    f"ISF feature scaler not found at {self.features_scaler_path}"
                )
                # Continue without scaler - may still work if data is pre-scaled
                pass

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load ISF model: {e}", exc_info=True)
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._loaded and self.model is not None

    def predict(
        self,
        feature_sequence: np.ndarray
    ) -> Optional[float]:
        """
        Predict ISF from feature sequence.

        Args:
            feature_sequence: Numpy array of shape (1, seq_length, n_feat)

        Returns:
            ISF value in mg/dL per unit, or None on error
        """
        if not self.is_loaded:
            logger.warning("ISF model not loaded, returning default ISF")
            return self.default_isf

        try:
            # Scale features if scaler available
            original_shape = feature_sequence.shape
            flat_features = feature_sequence.reshape(-1, original_shape[-1])

            if self.features_scaler is not None:
                try:
                    scaled_features = self.features_scaler.transform(flat_features)
                except Exception as e:
                    logger.warning(f"Feature scaling failed: {e}, using unscaled")
                    scaled_features = flat_features
            else:
                scaled_features = flat_features

            scaled_sequence = scaled_features.reshape(original_shape)

            # Convert to tensor
            input_tensor = torch.tensor(
                scaled_sequence,
                dtype=torch.float32,
                device=self.device
            )

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)

            # Get ISF value (softplus ensures > 0)
            isf_value = output.cpu().numpy()[0, 0]

            # Clamp to reasonable range (10-150 mg/dL per unit)
            isf_value = max(10.0, min(150.0, float(isf_value)))

            return isf_value

        except Exception as e:
            logger.error(f"ISF prediction failed: {e}", exc_info=True)
            return self.default_isf

    def get_default_isf(self) -> float:
        """
        Get the default ISF value used when model is unavailable.

        Returns:
            Default ISF in mg/dL per unit
        """
        return self.default_isf

    def set_default_isf(self, value: float) -> None:
        """
        Set the default ISF value.

        Args:
            value: New default ISF
        """
        self.default_isf = max(10.0, min(150.0, value))


def create_isf_service(
    models_dir: Path,
    device: str = "cpu"
) -> ISFInferenceService:
    """
    Factory function to create and load an ISF inference service.

    Args:
        models_dir: Directory containing model files
        device: Device for inference

    Returns:
        Loaded ISFInferenceService
    """
    service = ISFInferenceService(
        model_path=models_dir / "best_isf_net.pth",
        features_scaler_path=models_dir / "scaler_X_isf.pkl",
        feature_list_path=models_dir / "isf_feature_list.pkl",
        device=device
    )

    if not service.load():
        logger.warning(
            "Failed to load ISF service - using default ISF value"
        )

    return service
