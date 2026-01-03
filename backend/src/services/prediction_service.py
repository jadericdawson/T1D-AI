"""
Prediction Service
Unified service for BG and ISF predictions.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from ml.inference import (
    BGInferenceService,
    ISFInferenceService,
    LinearPredictor,
)
from ml.inference.bg_inference import create_bg_service
from ml.inference.isf_inference import create_isf_service
from ml.feature_engineering import FeatureEngineer, SEQ_LENGTH

logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Result of a glucose prediction."""
    linear: List[float]  # Linear predictions [+5, +10, +15]
    lstm: Optional[List[float]] = None  # LSTM predictions [+5, +10, +15]
    horizons_min: List[int] = [5, 10, 15]  # Prediction horizons in minutes
    timestamp: datetime  # When prediction was made
    current_bg: float
    trend: int
    isf: float  # Predicted or default ISF
    method: str = "linear"  # "linear", "lstm", or "hybrid"


class AccuracyStats(BaseModel):
    """Prediction accuracy statistics."""
    linear_mae: float
    lstm_mae: Optional[float] = None
    linear_count: int = 0
    lstm_count: int = 0
    winner: str = "linear"


class PredictionService:
    """
    Unified prediction service combining LSTM and linear models.

    Manages model loading, feature engineering, and prediction generation.
    Tracks prediction accuracy over time.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = "cpu"
    ):
        """
        Initialize the prediction service.

        Args:
            models_dir: Directory containing trained models and scalers
            device: Device for inference ("cpu" or "cuda")
        """
        self.models_dir = models_dir
        self.device = device

        # Services (initialized lazily)
        self._bg_service: Optional[BGInferenceService] = None
        self._isf_service: Optional[ISFInferenceService] = None
        self._linear_predictor = LinearPredictor()
        self._feature_engineer = FeatureEngineer()

        # Accuracy tracking
        self._linear_errors: List[float] = []
        self._lstm_errors: List[float] = []
        self._max_tracking_samples = 1000

        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize and load all models.

        Returns:
            True if at least linear prediction is available
        """
        if self._initialized:
            return True

        if self.models_dir and self.models_dir.exists():
            logger.info(f"Loading models from {self.models_dir}")

            # Load BG service
            self._bg_service = create_bg_service(self.models_dir, self.device)

            # Load ISF service
            self._isf_service = create_isf_service(self.models_dir, self.device)

            logger.info(
                f"Models loaded - BG: {self._bg_service.is_loaded}, "
                f"ISF: {self._isf_service.is_loaded}"
            )
        else:
            logger.warning(
                f"Models directory not found: {self.models_dir}. "
                "Using linear prediction only."
            )

        self._initialized = True
        return True

    @property
    def lstm_available(self) -> bool:
        """Check if LSTM predictions are available."""
        return (
            self._bg_service is not None and
            self._bg_service.is_loaded
        )

    @property
    def isf_available(self) -> bool:
        """Check if ISF predictions are available."""
        return (
            self._isf_service is not None and
            self._isf_service.is_loaded
        )

    def add_glucose_reading(self, reading: dict) -> None:
        """Add a glucose reading to the feature buffer."""
        self._feature_engineer.add_glucose_reading(reading)

    def add_treatment(self, treatment: dict) -> None:
        """Add a treatment to the feature buffer."""
        self._feature_engineer.add_treatment(treatment)

    def predict(
        self,
        current_bg: float,
        trend: int,
        iob: float = 0.0,
        glucose_history: Optional[List[dict]] = None,
        treatments: Optional[List[dict]] = None,
    ) -> PredictionResult:
        """
        Generate glucose predictions.

        Args:
            current_bg: Current glucose value in mg/dL
            trend: CGM trend arrow (-3 to +3)
            iob: Current insulin on board
            glucose_history: Optional list of recent glucose readings
            treatments: Optional list of recent treatments

        Returns:
            PredictionResult with linear and optionally LSTM predictions
        """
        if not self._initialized:
            self.initialize()

        now = datetime.utcnow()

        # Always compute linear prediction (fast, reliable)
        if glucose_history and len(glucose_history) >= 2:
            # Extract values from history
            glucose_values = [r.get('value', r.get('sgv', 0)) for r in glucose_history]
            linear_preds, slope, intercept = self._linear_predictor.predict(glucose_values)
        else:
            # Fallback to trend-based prediction
            linear_preds = self._linear_predictor.predict_with_trend(current_bg, trend)

        # Get ISF (use default if model unavailable)
        isf = self._get_isf(iob)

        # Try LSTM prediction if available
        lstm_preds = None
        method = "linear"

        if self.lstm_available and glucose_history:
            try:
                # Update feature engineer with history
                if glucose_history:
                    self._feature_engineer.clear()
                    for reading in glucose_history:
                        self._feature_engineer.add_glucose_reading(reading)

                if treatments:
                    for treatment in treatments:
                        self._feature_engineer.add_treatment(treatment)

                # Get feature sequence
                sequence = self._feature_engineer.get_feature_sequence(
                    iob_value=iob,
                    seq_length=SEQ_LENGTH
                )

                if sequence is not None:
                    result = self._bg_service.predict(sequence)
                    if result:
                        lstm_preds, _ = result
                        method = "lstm"
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        return PredictionResult(
            linear=linear_preds,
            lstm=lstm_preds,
            horizons_min=[5, 10, 15],
            timestamp=now,
            current_bg=current_bg,
            trend=trend,
            isf=isf,
            method=method if lstm_preds else "linear"
        )

    def _get_isf(self, iob: float = 0.0) -> float:
        """
        Get current ISF prediction or default.

        Args:
            iob: Current IOB (may affect ISF in some models)

        Returns:
            ISF value in mg/dL per unit
        """
        if self.isf_available:
            try:
                sequence = self._feature_engineer.get_feature_sequence(iob)
                if sequence is not None:
                    isf = self._isf_service.predict(sequence)
                    if isf is not None:
                        return isf
            except Exception as e:
                logger.warning(f"ISF prediction failed: {e}")

        # Return default
        return self._isf_service.default_isf if self._isf_service else 50.0

    def record_actual(
        self,
        prediction_time: datetime,
        actual_values: List[float],
        method: str = "linear"
    ) -> None:
        """
        Record actual values for accuracy tracking.

        Args:
            prediction_time: When the prediction was made
            actual_values: Actual observed values [+5, +10, +15]
            method: Which prediction method to attribute
        """
        # This would typically compare against stored predictions
        # For now, we just track overall accuracy
        pass

    def get_accuracy_stats(self) -> AccuracyStats:
        """Get current prediction accuracy statistics."""
        linear_mae = (
            sum(self._linear_errors) / len(self._linear_errors)
            if self._linear_errors else 0.0
        )
        lstm_mae = (
            sum(self._lstm_errors) / len(self._lstm_errors)
            if self._lstm_errors else None
        )

        winner = "linear"
        if lstm_mae is not None and lstm_mae < linear_mae:
            winner = "lstm"

        return AccuracyStats(
            linear_mae=linear_mae,
            lstm_mae=lstm_mae,
            linear_count=len(self._linear_errors),
            lstm_count=len(self._lstm_errors),
            winner=winner
        )

    def calculate_dose_correction(
        self,
        current_bg: float,
        target_bg: float,
        isf: float,
        iob: float,
        cob: float,
        cob_ratio: float = 4.0  # mg/dL rise per gram of carbs
    ) -> Dict[str, Any]:
        """
        Calculate correction dose considering IOB and COB.

        Ported from dexcom_reader_predict_v2.3.py lines 1049-1055.

        Args:
            current_bg: Current glucose in mg/dL
            target_bg: Target glucose in mg/dL
            isf: Insulin sensitivity factor (mg/dL per unit)
            iob: Current insulin on board (units)
            cob: Current carbs on board (grams)
            cob_ratio: mg/dL rise per gram (default: 4.0)

        Returns:
            Dictionary with dose calculation details
        """
        # Calculate effective BG considering active insulin and carbs
        # COB will raise BG, IOB will lower it
        cob_effect = cob * cob_ratio  # How much COB will raise BG
        iob_effect = iob * isf  # How much IOB will lower BG

        effective_bg = current_bg + cob_effect - iob_effect

        # Calculate raw correction needed
        raw_correction = (effective_bg - target_bg) / isf

        # Recommended dose (floor at 0 - don't recommend negative insulin)
        recommended_dose = max(0.0, raw_correction)

        return {
            "current_bg": current_bg,
            "target_bg": target_bg,
            "effective_bg": round(effective_bg, 1),
            "iob": round(iob, 2),
            "cob": round(cob, 1),
            "isf": round(isf, 1),
            "iob_effect_mgdl": round(iob_effect, 1),
            "cob_effect_mgdl": round(cob_effect, 1),
            "raw_correction_units": round(raw_correction, 2),
            "recommended_dose_units": round(recommended_dose, 2),
            "formula": f"({effective_bg} - {target_bg}) / {isf} = {raw_correction:.2f}U"
        }


# Singleton instances
_prediction_service: Optional[PredictionService] = None

# Create a default instance (lazy initialization)
prediction_service = PredictionService()


def get_prediction_service(
    models_dir: Optional[Path] = None,
    device: str = "cpu"
) -> PredictionService:
    """
    Get or create the global prediction service.

    Args:
        models_dir: Directory containing trained models
        device: Device for inference

    Returns:
        PredictionService instance
    """
    global _prediction_service

    if _prediction_service is None:
        _prediction_service = PredictionService(models_dir, device)
        _prediction_service.initialize()

    return _prediction_service
