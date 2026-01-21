"""
Prediction Service
Unified service for BG and ISF predictions.
Supports per-user personalized models stored in Azure Blob Storage.
"""
import asyncio
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
from ml.feature_engineering import (
    FeatureEngineer, SEQ_LENGTH, TFT_FEATURE_COLUMNS,
    prepare_realtime_features, engineer_extended_features, extract_tft_feature_sequence
)
from ml.training.model_manager import get_model_manager, ModelManager
from ml.models.tft_predictor import (
    TemporalFusionTransformer,
    TFTOutput,
    TFTPrediction,
    create_tft_model,
    TFT_MODEL_CONFIG,
)
from database.repositories import LearnedISFRepository, LearnedICRRepository, LearnedPIRRepository
from services.personalized_iob_service import get_personalized_iob_service, PersonalizedIOBService
from services.iob_cob_service import IOBCOBService
from services.metabolic_params_service import (
    get_metabolic_params_service,
    MetabolicParamsService,
    EffectiveISF,
    EffectiveICR,
    EffectivePIR,
    MetabolicParams,
)
import torch

# Forcing function imports for physics-informed predictions
try:
    from ml.models.forcing_ensemble import ForcingFunctionEnsemble, get_forcing_ensemble
    FORCING_ENSEMBLE_AVAILABLE = True
except ImportError:
    FORCING_ENSEMBLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TFTPredictionItem(BaseModel):
    """Single TFT prediction with uncertainty bounds."""
    horizon_min: int  # Prediction horizon in minutes (30, 45, 60)
    value: float  # Median prediction (50th percentile)
    lower: float  # Lower bound (10th percentile)
    upper: float  # Upper bound (90th percentile)
    confidence: float = 0.8  # Confidence level
    tft_delta: Optional[float] = None  # TFT modifier delta (mg/dL) - how much TFT adjusted physics


class PredictionResult(BaseModel):
    """Result of a glucose prediction."""
    linear: List[float]  # Linear predictions [+5, +10, +15]
    lstm: Optional[List[float]] = None  # LSTM predictions [+5, +10, +15]
    tft: Optional[List[TFTPredictionItem]] = None  # TFT predictions [+30, +45, +60]
    horizons_min: List[int] = [5, 10, 15]  # Prediction horizons in minutes
    timestamp: datetime  # When prediction was made
    current_bg: float
    trend: int
    isf: float  # Predicted or default ISF
    method: str = "linear"  # "linear", "lstm", "tft", or "hybrid"


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
        self._tft_model: Optional[TemporalFusionTransformer] = None
        self._tft_scaler = None  # RobustScaler for TFT feature normalization
        self._linear_predictor = LinearPredictor()
        self._feature_engineer = FeatureEngineer()

        # Personalized IOB service (uses ML model trained on this person's BG data)
        # This child's insulin acts 30% faster than adult formula (54 min vs 81 min half-life)
        self._personalized_iob = get_personalized_iob_service()

        # IOB/COB service with LEARNED absorption curves from BG response data
        # Model-based: onset=15min, ramp=75min, half-life=120min (learned from 54 correction boluses)
        self._iob_cob_service = IOBCOBService()

        # Metabolic params service for learned ISF/ICR/PIR values
        # Provides effective values combining baseline + short-term deviation (illness detection)
        self._metabolic_params_service = get_metabolic_params_service()

        # Cache for user-specific parameters (cleared when user_id changes)
        self._cached_user_id: Optional[str] = None
        self._cached_isf: Optional[float] = None
        self._cached_icr: Optional[float] = None
        self._cached_pir: Optional[float] = None

        # Forcing function ensemble for physics-informed predictions
        # Uses IOB/COB as primary forcing functions with neural residual
        self._forcing_ensemble: Optional[ForcingFunctionEnsemble] = None
        if FORCING_ENSEMBLE_AVAILABLE:
            try:
                self._forcing_ensemble = get_forcing_ensemble()
                logger.info("Forcing function ensemble initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize forcing ensemble: {e}")

        # Accuracy tracking
        self._linear_errors: List[float] = []
        self._lstm_errors: List[float] = []
        self._max_tracking_samples = 1000

        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize and load BG model only.
        ISF model is loaded lazily on first use (it's 431MB and takes ~5 min).

        Returns:
            True if at least linear prediction is available
        """
        if self._initialized:
            return True

        if self.models_dir and self.models_dir.exists():
            logger.info(f"Loading models from {self.models_dir}")

            # Load BG service (fast - ~1MB model)
            self._bg_service = create_bg_service(self.models_dir, self.device)

            # DON'T load ISF model at startup - it's 431MB and takes ~5 minutes
            # It will be loaded lazily on first ISF prediction request
            # self._isf_service = create_isf_service(self.models_dir, self.device)

            # Load TFT model if available (5MB, fast to load)
            self._load_tft_model()

            logger.info(
                f"Models loaded - BG: {self._bg_service.is_loaded}, "
                f"TFT: {self._tft_model is not None}, "
                f"ISF: deferred (loaded on first use)"
            )
        else:
            logger.warning(
                f"Models directory not found: {self.models_dir}. "
                "Using linear prediction only."
            )

        self._initialized = True
        return True

    def _load_tft_model(self) -> None:
        """Load TFT model if available."""
        if self.models_dir is None:
            return

        tft_path = self.models_dir / "tft_glucose_predictor.pth"
        if not tft_path.exists():
            logger.info(f"TFT model not found at {tft_path}")
            return

        try:
            # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(tft_path, map_location=self.device, weights_only=False)

            # Extract config from checkpoint to create model with correct architecture
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                horizons = config.get('horizons_minutes', [30, 45, 60])
                n_features = config.get('n_features', TFT_MODEL_CONFIG["n_features"])
                hidden_size = config.get('hidden_size', TFT_MODEL_CONFIG["hidden_size"])
                n_heads = config.get('n_heads', TFT_MODEL_CONFIG["n_heads"])
                n_lstm_layers = config.get('n_lstm_layers', TFT_MODEL_CONFIG["n_lstm_layers"])
                dropout = config.get('dropout', TFT_MODEL_CONFIG["dropout"])
                encoder_length = config.get('encoder_length', TFT_MODEL_CONFIG["encoder_length"])
                prediction_length = config.get('prediction_length', TFT_MODEL_CONFIG["prediction_length"])
                quantiles = config.get('quantiles', TFT_MODEL_CONFIG["quantiles"])

                logger.info(f"Loading TFT model with config: horizons={horizons}, n_features={n_features}")

                # Import TFT model class directly
                from ml.models.tft_predictor import TemporalFusionTransformer
                self._tft_model = TemporalFusionTransformer(
                    n_features=n_features,
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    n_lstm_layers=n_lstm_layers,
                    dropout=dropout,
                    encoder_length=encoder_length,
                    prediction_length=prediction_length,
                    quantiles=quantiles,
                    horizons_minutes=horizons,
                )
            else:
                # Fallback to default config
                self._tft_model = create_tft_model(n_features=TFT_MODEL_CONFIG["n_features"])

            # Load trained weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self._tft_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._tft_model.load_state_dict(checkpoint)

            self._tft_model.to(self.device)
            self._tft_model.eval()

            # Load scaler for feature normalization (CRITICAL for correct predictions!)
            if isinstance(checkpoint, dict) and 'scaler_state' in checkpoint:
                import pickle
                self._tft_scaler = pickle.loads(checkpoint['scaler_state'])
                logger.info(f"TFT scaler loaded: {type(self._tft_scaler).__name__}")
            else:
                logger.warning("TFT scaler not found in checkpoint - predictions may be inaccurate!")

            logger.info(f"TFT model loaded successfully from {tft_path}")
        except Exception as e:
            logger.warning(f"Failed to load TFT model: {e}")
            import traceback
            traceback.print_exc()
            self._tft_model = None

    def _ensure_isf_loaded(self) -> None:
        """Lazy load ISF model on first use."""
        if self._isf_service is not None:
            return

        # Search for ISF model in multiple locations
        # ISF model files are: best_isf_net.pth, scaler_X_isf.pkl, isf_feature_list.pkl
        isf_search_paths = [
            Path("ml/models_data"),  # Primary location for ISF models
            Path("./ml/models_data"),
            Path("../ml/models_data"),  # If running from src/ directory
            Path(__file__).parent.parent.parent / "ml" / "models_data",  # Relative to this file
            Path("/app/ml/models_data"),  # Docker container
            self.models_dir,  # Also check general models directory
            Path("./models"),
            Path("./data/models"),
            Path("/app/models"),
        ]

        isf_dir = None
        for path in isf_search_paths:
            if path and path.exists() and (path / "best_isf_net.pth").exists():
                isf_dir = path
                logger.info(f"Found ISF model at {path}")
                break

        if isf_dir:
            logger.info(f"Lazy loading ISF model from {isf_dir}...")
            self._isf_service = create_isf_service(isf_dir, self.device)
            logger.info(f"ISF model loaded: {self._isf_service.is_loaded}")
        else:
            logger.warning(
                "ISF model not found in any search path. "
                "Searched: ml/models_data, ./models, ./data/models, /app/models. "
                "Using default ISF value."
            )
            # Create a service with default ISF (will return 55.0)
            from ml.inference.isf_inference import ISFInferenceService
            self._isf_service = ISFInferenceService(device=self.device)

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

    @property
    def tft_available(self) -> bool:
        """Check if TFT predictions are available."""
        return self._tft_model is not None

    @property
    def forcing_available(self) -> bool:
        """Check if forcing function ensemble predictions are available."""
        return self._forcing_ensemble is not None

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
        cob: float = 0.0,
        pob: float = 0.0,
        glucose_history: Optional[List[dict]] = None,
        treatments: Optional[List[dict]] = None,
    ) -> PredictionResult:
        """
        Generate glucose predictions.

        Args:
            current_bg: Current glucose value in mg/dL
            trend: CGM trend arrow (-3 to +3)
            iob: Current insulin on board
            cob: Current carbs on board
            pob: Current protein on board
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
            # Sort by timestamp (oldest first) - linear predictor expects newest last
            sorted_history = sorted(
                glucose_history,
                key=lambda x: x.get('timestamp', x.get('dateString', ''))
            )
            # Extract values from sorted history (now oldest first, newest last)
            glucose_values = [r.get('value', r.get('sgv', 0)) for r in sorted_history]
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
                    logger.debug(f"LSTM: Added {len(glucose_history)} glucose readings to feature engineer")

                if treatments:
                    for treatment in treatments:
                        self._feature_engineer.add_treatment(treatment)
                    logger.debug(f"LSTM: Added {len(treatments)} treatments to feature engineer")

                # Get feature sequence
                sequence = self._feature_engineer.get_feature_sequence(
                    iob_value=iob,
                    seq_length=SEQ_LENGTH
                )

                if sequence is not None:
                    logger.debug(f"LSTM: Got feature sequence shape {sequence.shape}")
                    result = self._bg_service.predict(sequence)
                    if result:
                        lstm_preds, _ = result
                        method = "lstm"
                        logger.info(f"LSTM prediction successful: {lstm_preds}")
                    else:
                        logger.warning("LSTM: bg_service.predict() returned None")
                else:
                    logger.warning(f"LSTM: sequence is None - need {SEQ_LENGTH} readings, have {len(glucose_history)}")
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}", exc_info=True)
        else:
            if not self.lstm_available:
                logger.debug("LSTM not available - model not loaded")
            if not glucose_history:
                logger.debug("LSTM skipped - no glucose history provided")

        # Try TFT prediction if available (longer horizon: 30, 45, 60 min)
        tft_preds = None
        if self.tft_available and glucose_history:
            try:
                tft_preds = self._predict_tft(glucose_history, treatments, iob, cob, pob)
                if tft_preds:
                    method = "tft" if not lstm_preds else "hybrid"
            except Exception as e:
                logger.warning(f"TFT prediction failed: {e}")

        return PredictionResult(
            linear=linear_preds,
            lstm=lstm_preds,
            tft=tft_preds,
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
        # Lazy load ISF model on first use
        self._ensure_isf_loaded()

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
        return self._isf_service.default_isf if self._isf_service else 55.0

    async def _get_effective_isf_async(
        self,
        user_id: str,
        is_fasting: bool = True,
        include_short_term: bool = True
    ) -> float:
        """
        Get effective ISF for a user from learned data.

        This is the CORRECT ISF to use for predictions and dose calculations.
        It combines long-term baseline with short-term deviation (illness detection).

        Args:
            user_id: User ID to get ISF for
            is_fasting: Whether to prefer fasting or meal ISF
            include_short_term: Whether to include illness/sensitivity detection

        Returns:
            Effective ISF value in mg/dL per unit
        """
        try:
            effective = await self._metabolic_params_service.get_effective_isf(
                user_id, is_fasting, include_short_term
            )
            if effective.is_sick or effective.is_resistant:
                logger.info(
                    f"User {user_id} ISF deviation detected: {effective.deviation_percent:.1f}% "
                    f"(baseline={effective.baseline:.1f}, effective={effective.value:.1f})"
                )
            return effective.value
        except Exception as e:
            logger.warning(f"Failed to get effective ISF for user {user_id}: {e}")
            return self._get_isf(0.0)

    async def _get_effective_icr_async(
        self,
        user_id: str,
        meal_type: Optional[str] = None
    ) -> float:
        """
        Get effective ICR (Insulin-to-Carb Ratio) for a user from learned data.

        Args:
            user_id: User ID
            meal_type: Optional meal type (breakfast, lunch, dinner)

        Returns:
            Effective ICR value in grams per unit
        """
        try:
            effective = await self._metabolic_params_service.get_effective_icr(user_id, meal_type)
            return effective.value
        except Exception as e:
            logger.warning(f"Failed to get effective ICR for user {user_id}: {e}")
            return 10.0  # Default

    async def _get_effective_pir_async(
        self,
        user_id: str,
        meal_type: Optional[str] = None
    ) -> tuple:
        """
        Get effective PIR (Protein-to-Insulin Ratio) for a user from learned data.

        Args:
            user_id: User ID
            meal_type: Optional meal type

        Returns:
            Tuple of (pir_value, onset_minutes, peak_minutes)
        """
        try:
            effective = await self._metabolic_params_service.get_effective_pir(user_id, meal_type)
            return effective.value, effective.onset_minutes, effective.peak_minutes
        except Exception as e:
            logger.warning(f"Failed to get effective PIR for user {user_id}: {e}")
            return 25.0, 120, 210  # Defaults

    async def _get_metabolic_params_async(
        self,
        user_id: str,
        is_fasting: bool = False,
        meal_type: Optional[str] = None,
        include_short_term: bool = True
    ) -> MetabolicParams:
        """
        Get all metabolic parameters for a user.

        This is the main method for getting complete metabolic context.

        Args:
            user_id: User ID
            is_fasting: Whether this is a fasting context
            meal_type: Optional meal type for ICR/PIR
            include_short_term: Whether to include illness detection

        Returns:
            MetabolicParams with all parameters and metabolic state
        """
        return await self._metabolic_params_service.get_all_params(
            user_id, is_fasting, meal_type, include_short_term
        )

    def _predict_tft(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float,
        cob: float = 0.0,
        pob: float = 0.0,
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Generate predictions using physics-first approach with TFT as a modifier.

        Architecture: Physics-First with TFT Residual
        1. Physics baseline (forcing functions) is ALWAYS the primary prediction
        2. TFT neural network provides a modifier/delta (clamped to +/- 20 mg/dL)
        3. Final = physics + TFT_delta

        This ensures predictions are grounded in physical reality (IOB/COB/POB effects)
        while TFT captures secondary patterns the physics layer doesn't model.

        Args:
            glucose_history: List of recent glucose readings
            treatments: List of recent treatments (with glycemic data)
            iob: Current insulin on board (from IOB model)
            cob: Current carbs on board (from COB model)
            pob: Current protein on board (from POB model)

        Returns:
            List of TFTPredictionItem at multiple horizons, or None if prediction fails
        """
        import numpy as np
        import math

        # Maximum TFT delta (mg/dL) - allows TFT larger adjustments to physics baseline
        MAX_TFT_DELTA = 40.0

        logger.info(f"TFT prediction starting (physics-first): IOB={iob}, COB={cob}, POB={pob}, treatments={len(treatments or [])}")

        if len(glucose_history) < 6:
            logger.warning(f"TFT: insufficient glucose history ({len(glucose_history)} < 6)")
            return None

        # Step 1: ALWAYS get physics baseline (forcing functions)
        physics_result = None
        if self._forcing_ensemble is not None:
            try:
                physics_result = self._predict_with_forcing_functions(glucose_history, treatments, iob, cob, pob)
                if physics_result:
                    logger.info(f"Physics baseline generated: {len(physics_result)} predictions")
            except Exception as e:
                logger.warning(f"Physics forcing ensemble failed: {e}")

        # If physics fails, fall back to formula-based physics
        if physics_result is None:
            logger.info("Using formula-based physics as baseline")
            physics_result = self._predict_tft_formula(glucose_history, treatments, iob, cob, pob)
            if physics_result is None:
                logger.warning("Formula-based physics also failed")
                return None

        # Step 2: If TFT neural network available, use it as a modifier
        if self._tft_model is not None:
            try:
                tft_raw = self._predict_with_tft_model_raw(glucose_history, treatments, iob, cob)
                if tft_raw:
                    # Create mapping of physics predictions by horizon
                    physics_by_horizon = {p.horizon_min: p for p in physics_result}

                    # Apply TFT delta to physics baseline
                    modified_results = []
                    for tft_pred in tft_raw:
                        horizon = tft_pred.horizon_min
                        if horizon in physics_by_horizon:
                            physics_pred = physics_by_horizon[horizon]

                            # Calculate delta and clamp it
                            delta = tft_pred.value - physics_pred.value
                            clamped_delta = max(-MAX_TFT_DELTA, min(MAX_TFT_DELTA, delta))

                            # Final = physics + clamped TFT delta
                            final_value = physics_pred.value + clamped_delta

                            # Expand uncertainty bounds if TFT and physics disagree
                            delta_magnitude = abs(delta)
                            extra_uncertainty = min(10, delta_magnitude * 0.3)

                            modified_results.append(TFTPredictionItem(
                                horizon_min=horizon,
                                value=round(max(40, min(400, final_value)), 1),
                                lower=round(max(40, physics_pred.lower - extra_uncertainty), 1),
                                upper=round(max(40, min(400, physics_pred.upper + extra_uncertainty)), 1),
                                confidence=round(physics_pred.confidence * 0.95, 2),  # Slight confidence reduction
                                tft_delta=round(clamped_delta, 1)  # TFT modifier delta
                            ))

                            if abs(clamped_delta) > 5:
                                logger.info(
                                    f"  +{horizon}min: physics={physics_pred.value:.0f}, "
                                    f"TFT_raw={tft_pred.value:.0f}, delta={delta:.1f}, "
                                    f"clamped={clamped_delta:.1f}, final={final_value:.0f}"
                                )
                        else:
                            # No physics prediction at this horizon, use physics result directly
                            modified_results.append(physics_by_horizon.get(horizon, tft_pred))

                    # Fill in any horizons only in physics
                    tft_horizons = {p.horizon_min for p in tft_raw}
                    for horizon, physics_pred in physics_by_horizon.items():
                        if horizon not in tft_horizons:
                            modified_results.append(physics_pred)

                    # Sort by horizon
                    modified_results.sort(key=lambda x: x.horizon_min)

                    logger.info(f"Physics + TFT modifier generated {len(modified_results)} predictions")
                    return modified_results

            except Exception as e:
                logger.warning(f"TFT modifier failed: {e}, using physics-only")

        # Return pure physics if TFT unavailable or failed
        logger.info(f"Using physics-only predictions ({len(physics_result)} horizons)")
        return physics_result

    def _predict_with_tft_model_raw(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float,
        cob: float = 0.0,
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Generate RAW predictions using the TFT neural network.

        These are absolute BG predictions from the neural network, not yet
        combined with physics. Used as input to the physics-modifier architecture.

        Args:
            glucose_history: List of recent glucose readings
            treatments: List of recent treatments
            iob: Current insulin on board
            cob: Current carbs on board

        Returns:
            List of TFTPredictionItem (raw TFT output) or None if prediction fails
        """
        import numpy as np
        import math

        # Prepare features for TFT model
        logger.debug(f"Preparing TFT features from {len(glucose_history)} readings")

        # Step 1: Create base DataFrame from glucose readings
        df = prepare_realtime_features(
            glucose_readings=glucose_history,
            treatments=treatments or [],
            iob_value=iob
        )

        if df is None or len(df) < SEQ_LENGTH:
            logger.warning(f"TFT: insufficient data after prepare_realtime_features (need {SEQ_LENGTH})")
            return None

        # Step 2: Prepare treatments DataFrame
        treatments_df = None
        if treatments:
            import pandas as pd
            treatments_df = pd.DataFrame(treatments)
            if 'timestamp' in treatments_df.columns:
                treatments_df['timestamp'] = pd.to_datetime(treatments_df['timestamp'], utc=True)
                treatments_df['timestamp'] = treatments_df['timestamp'].dt.tz_localize(None)

        # Step 3: Engineer extended features (65 TFT features)
        # Use cached learned values if available, otherwise fall back to model/defaults
        isf = self._cached_isf if self._cached_isf else self._get_isf(iob)
        icr = self._cached_icr if self._cached_icr else 10.0  # Use learned ICR or default

        df_extended = engineer_extended_features(
            df_in=df,
            treatments_df=treatments_df,
            ml_iob=iob,
            ml_cob=cob,
            isf=isf,
            icr=icr
        )

        logger.debug(f"Extended features shape: {df_extended.shape}, columns: {len(df_extended.columns)}")

        # Step 4: Extract TFT feature sequence
        sequence = extract_tft_feature_sequence(df_extended, seq_length=SEQ_LENGTH)

        if sequence is None:
            logger.warning("TFT: failed to extract feature sequence")
            return None

        # Step 4.5: Apply scaler normalization (CRITICAL for correct predictions!)
        if self._tft_scaler is not None:
            # sequence shape: (1, seq_len, n_features)
            # scaler expects (n_samples, n_features), so reshape
            orig_shape = sequence.shape
            sequence_2d = sequence.reshape(-1, sequence.shape[-1])
            sequence_scaled = self._tft_scaler.transform(sequence_2d)
            sequence = sequence_scaled.reshape(orig_shape)
            logger.debug(f"TFT features scaled with {type(self._tft_scaler).__name__}")
        else:
            logger.warning("TFT scaler not available - using raw features (predictions may be inaccurate)")

        logger.debug(f"TFT sequence shape: {sequence.shape}")

        # Step 5: Convert to tensor and predict
        x = torch.from_numpy(sequence).float()

        # Move to same device as model
        if next(self._tft_model.parameters()).device.type != 'cpu':
            x = x.to(next(self._tft_model.parameters()).device)

        tft_output = self._tft_model.predict_with_uncertainty(x)

        # Step 6: Convert TFT output to TFTPredictionItem list
        # TFT model outputs at horizons [30, 45, 60]
        results = []

        # Get current BG and trend for interpolation
        sorted_readings = sorted(
            glucose_history,
            key=lambda x: x.get('timestamp', x.get('dateString', '')),
            reverse=True
        )
        current_bg = float(sorted_readings[0].get('value', sorted_readings[0].get('sgv', 120)))
        recent_values = [float(r.get('value', r.get('sgv', 120))) for r in sorted_readings[:6]]
        rate_per_5min = (recent_values[0] - recent_values[-1]) / max(1, len(recent_values) - 1) if len(recent_values) >= 2 else 0

        # TFT model outputs at 30, 45, 60 min
        tft_horizons = [30, 45, 60]
        tft_predictions = {}
        for pred in tft_output.predictions:
            tft_predictions[pred.horizon_min] = pred

        # Generate predictions at all target horizons
        # For horizons < 30: interpolate from current BG to 30min prediction
        # For horizons > 60: extrapolate using 45-60 slope
        target_horizons = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]

        for horizon in target_horizons:
            if horizon in tft_predictions:
                # Direct TFT prediction available
                pred = tft_predictions[horizon]
                results.append(TFTPredictionItem(
                    horizon_min=horizon,
                    value=round(pred.value, 1),
                    lower=round(pred.lower, 1),
                    upper=round(pred.upper, 1),
                    confidence=pred.confidence
                ))
            elif horizon < 30:
                # Interpolate between current BG and 30min prediction
                pred_30 = tft_predictions.get(30)
                if pred_30:
                    t_ratio = horizon / 30.0
                    value = current_bg + t_ratio * (pred_30.value - current_bg)
                    # Uncertainty grows with time but less than TFT uncertainty at 30min
                    uncertainty = 8 + (horizon * 0.25)
                    results.append(TFTPredictionItem(
                        horizon_min=horizon,
                        value=round(max(40, min(400, value)), 1),
                        lower=round(max(40, min(400, value - uncertainty)), 1),
                        upper=round(max(40, min(400, value + uncertainty)), 1),
                        confidence=round(0.85 - (horizon * 0.005), 2)
                    ))
            else:
                # Extrapolate beyond 60min using 45-60 slope
                pred_45 = tft_predictions.get(45)
                pred_60 = tft_predictions.get(60)
                if pred_45 and pred_60:
                    # Rate of change from 45 to 60 min
                    rate_per_min = (pred_60.value - pred_45.value) / 15.0
                    # Apply damping for extrapolation (trends don't continue forever)
                    extra_minutes = horizon - 60
                    damping = math.exp(-extra_minutes / 60)  # Dampen extrapolation
                    value = pred_60.value + rate_per_min * extra_minutes * damping

                    # Uncertainty grows significantly for extrapolation
                    base_uncertainty = (pred_60.upper - pred_60.lower) / 2
                    extra_uncertainty = extra_minutes * 0.5
                    total_uncertainty = base_uncertainty + extra_uncertainty

                    results.append(TFTPredictionItem(
                        horizon_min=horizon,
                        value=round(max(40, min(400, value)), 1),
                        lower=round(max(40, min(400, value - total_uncertainty)), 1),
                        upper=round(max(40, min(400, value + total_uncertainty)), 1),
                        confidence=round(max(0.5, 0.7 - (extra_minutes * 0.003)), 2)
                    ))

        logger.info(f"TFT neural network generated {len(results)} predictions at horizons: {[r.horizon_min for r in results]}")
        return results

    def _predict_with_forcing_functions(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float,
        cob: float = 0.0,
        pob: float = 0.0,
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Generate predictions using the physics-informed forcing function ensemble.

        This is the new architecture where IOB, COB, and POB are independent forcing functions:
        - IOB applies negative pressure (insulin lowers BG)
        - COB applies positive pressure (carbs raise BG)
        - POB applies delayed positive pressure (protein raises BG after 2-5 hours)
        - When multiple are present, their effects combine
        - Neural residual adds secondary factor adjustments (+/- 25 mg/dL max)

        Formula:
            predicted_bg = current_bg + iob_pressure + cob_pressure + pob_pressure + residual

        Args:
            glucose_history: List of recent glucose readings
            treatments: List of recent treatments (with glycemic data)
            iob: Current insulin on board (from personalized IOB model)
            cob: Current carbs on board
            pob: Current protein on board

        Returns:
            List of TFTPredictionItem at multiple horizons, or None if prediction fails
        """
        if self._forcing_ensemble is None:
            return None

        if len(glucose_history) < 3:
            logger.warning("Forcing ensemble: insufficient glucose history")
            return None

        try:
            # Sort by timestamp to get current BG
            sorted_readings = sorted(
                glucose_history,
                key=lambda x: x.get('timestamp', x.get('dateString', '')),
                reverse=True
            )

            current_bg = float(sorted_readings[0].get('value', sorted_readings[0].get('sgv', 120)))
            current_hour = datetime.utcnow().hour

            # Calculate recent trend and volatility for secondary features
            recent_values = [float(r.get('value', r.get('sgv', 120))) for r in sorted_readings[:6]]
            if len(recent_values) >= 2:
                recent_trend = (recent_values[0] - recent_values[-1]) / max(1, len(recent_values) - 1)
            else:
                recent_trend = 0.0

            import numpy as np
            bg_volatility = np.std(recent_values) if len(recent_values) > 1 else 0.0

            # Get ISF/ICR - use cached learned values if available, otherwise model/defaults
            isf = self._cached_isf if self._cached_isf else self._get_isf(iob)
            icr = self._cached_icr if self._cached_icr else 10.0

            # Get food composition from treatments if available
            glycemic_index = 55.0  # Default medium GI
            fat_grams = 0.0
            protein_grams = 0.0

            if treatments:
                # Find most recent carb treatment within 3 hours
                now = datetime.utcnow()
                for treat in reversed(treatments):
                    carbs = treat.get('carbs', 0)
                    if carbs and carbs > 0:
                        t_timestamp = treat.get('timestamp')
                        if isinstance(t_timestamp, str):
                            try:
                                t_time = datetime.fromisoformat(t_timestamp.replace('Z', '+00:00'))
                                t_time = t_time.replace(tzinfo=None)
                                if (now - t_time).total_seconds() < 3 * 3600:  # Within 3 hours
                                    glycemic_index = treat.get('glycemicIndex', 55) or 55
                                    fat_grams = treat.get('fat', 0) or 0
                                    protein_grams = treat.get('protein', 0) or 0
                                    break
                            except:
                                pass

            # Dawn phenomenon intensity (stronger 4-8 AM)
            dawn_intensity = 0.7 if 4 <= current_hour < 8 else 0.0

            # Generate predictions at all target horizons
            target_horizons = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]
            results = []

            # Get multi-horizon predictions from forcing ensemble
            predictions = self._forcing_ensemble.predict_multi_horizon(
                current_bg=current_bg,
                iob=iob,
                cob=cob,
                horizons=target_horizons,
                isf=isf,
                icr=icr,
                glycemic_index=glycemic_index,
                fat_grams=fat_grams,
                protein_grams=protein_grams,
                recent_trend=recent_trend,
                bg_volatility=bg_volatility,
                dawn_intensity=dawn_intensity,
                hour=current_hour,
            )

            # Convert to TFTPredictionItem format
            for horizon in target_horizons:
                if horizon in predictions:
                    pred = predictions[horizon]
                    results.append(TFTPredictionItem(
                        horizon_min=horizon,
                        value=round(pred.final_prediction, 1),
                        lower=round(pred.lower_bound, 1),
                        upper=round(pred.upper_bound, 1),
                        confidence=round(pred.confidence, 2)
                    ))

            if results:
                # Log key predictions for debugging
                for h in [30, 60, 120]:
                    if h in predictions:
                        p = predictions[h]
                        logger.info(
                            f"  Forcing +{h}min: {p.final_prediction:.0f} "
                            f"(IOB={p.iob_pressure:.1f}, COB={p.cob_pressure:.1f}, "
                            f"residual={p.residual_adjustment:.1f})"
                        )

            logger.info(f"Forcing ensemble generated {len(results)} predictions")
            return results

        except Exception as e:
            logger.warning(f"Forcing ensemble prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_tft_formula(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float,
        cob: float = 0.0,
        pob: float = 0.0,
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Formula-based TFT predictions (fallback when neural network unavailable).

        Uses IOB/COB/POB decay formulas with physically correct insulin/carb/protein absorption curves.

        Key insight: Predictions should be driven by REMAINING IOB/COB/POB effects, not trend damping.
        The trend only reflects what's ALREADY happening - metabolic effects drive future BG.

        POB (Protein on Board):
        - Protein has delayed absorption: onset ~120 min, peak ~210 min, duration ~300 min
        - Protein raises BG at ~50% the rate of carbs (bg_per_gram_protein = ISF / PIR)
        - Half-life is ~75 min (vs 45 min for carbs)
        """
        try:
            import numpy as np
            import math

            # Sort by timestamp
            sorted_readings = sorted(
                glucose_history,
                key=lambda x: x.get('timestamp', x.get('dateString', '')),
                reverse=True
            )

            # Get current glucose and recent trend
            current_bg = float(sorted_readings[0].get('value', sorted_readings[0].get('sgv', 120)))
            raw_trend = sorted_readings[0].get('trend', 0)
            if isinstance(raw_trend, str):
                trend_map = {
                    "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                    "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
                }
                trend = trend_map.get(raw_trend, 0)
            else:
                trend = int(raw_trend or 0)

            # Calculate rate of change from last 30 min (6 readings)
            recent_values = [float(r.get('value', r.get('sgv', 120))) for r in sorted_readings[:6]]
            rate_per_5min = (recent_values[0] - recent_values[-1]) / max(1, len(recent_values) - 1) if len(recent_values) >= 2 else trend * 1.5

            # Use cached learned values if available, otherwise fall back to defaults
            # These are populated by predict_for_user() with effective (illness-adjusted) values
            isf = self._cached_isf if self._cached_isf else 55.0
            icr = self._cached_icr if self._cached_icr else 10.0
            pir = self._cached_pir if self._cached_pir else 25.0  # PIR: grams protein per unit

            logger.info(f"TFT formula: current_bg={current_bg}, rate={rate_per_5min:.1f}/5min, IOB={iob:.1f}U, COB={cob:.0f}g, POB={pob:.0f}g, ISF={isf} (learned={self._cached_isf is not None})")

            target_horizons = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]
            results = []

            # Use PERSONALIZED insulin decay model (learned from this child's BG data)
            # This 7-year-old has ~54 min half-life vs 81 min adult formula (30% faster!)
            # The personalized_iob service uses the ML model trained on actual BG responses

            # Carb absorption: half-life ~45 min for medium GI foods
            carb_half_life_min = 45.0

            # Protein absorption parameters (delayed, slower than carbs)
            # Protein starts affecting BG ~2 hours after eating, peaks at ~3.5 hours
            protein_onset_min = 120.0  # Protein starts affecting BG
            protein_half_life_min = 75.0  # Slower decay than carbs (75 vs 45 min)

            # Get current hour for personalized IOB model
            current_hour = datetime.utcnow().hour

            for horizon in target_horizons:
                # Calculate FUTURE IOB effect using PERSONALIZED model
                # This accounts for this child's faster insulin absorption
                insulin_effect = self._personalized_iob.get_insulin_effect_at_horizon(
                    current_iob=iob,
                    horizon_min=horizon,
                    isf=isf,
                    hour=current_hour,
                )

                # Calculate FUTURE COB effect: carbs that will be absorbed from NOW to horizon
                cob_remaining_at_horizon = cob * (0.5 ** (horizon / carb_half_life_min))
                carbs_to_be_absorbed = cob - cob_remaining_at_horizon
                carb_effect = carbs_to_be_absorbed * (isf / icr)

                # Calculate FUTURE POB effect: protein has delayed onset (~2 hours)
                # Protein only starts affecting BG after onset period
                protein_effect = 0.0
                if pob > 0 and horizon >= protein_onset_min:
                    # Time since protein effect started (relative to now + horizon)
                    effective_protein_time = horizon - protein_onset_min
                    # POB remaining at horizon (after onset)
                    pob_remaining_at_horizon = pob * (0.5 ** (effective_protein_time / protein_half_life_min))
                    protein_absorbed = pob - pob_remaining_at_horizon
                    # Protein raises BG at ~50% the rate of carbs (PIR is typically 2x ICR)
                    protein_effect = protein_absorbed * (isf / pir)
                elif pob > 0 and horizon < protein_onset_min:
                    # Before onset: protein is being digested but not yet affecting BG
                    # Small anticipatory effect as we approach onset
                    approach_factor = horizon / protein_onset_min
                    protein_effect = pob * (isf / pir) * approach_factor * 0.1  # 10% anticipatory

                # Net metabolic effect (insulin lowers, carbs and protein raise)
                net_metabolic_effect = insulin_effect + carb_effect + protein_effect

                # Trend contribution: the current trend reflects CURRENT metabolic state
                # It should NOT be extrapolated far into the future - IOB/COB/POB effects dominate
                # Use minimal trend contribution that decays quickly
                if iob > 0.5 or cob > 10 or pob > 10:
                    # When metabolic activity is high, trend is already explained by IOB/COB/POB
                    # Only use trend for immediate short-term (< 15 min)
                    if horizon <= 15:
                        trend_contribution = rate_per_5min * (horizon / 5) * 0.5
                    else:
                        # Beyond 15 min, let IOB/COB/POB drive the prediction
                        trend_contribution = rate_per_5min * 3 * 0.3  # Cap at ~15 min worth
                else:
                    # Low metabolic activity - use normal trend extrapolation with damping
                    trend_damping = math.exp(-horizon / 60)
                    trend_contribution = rate_per_5min * (horizon / 5) * trend_damping

                # Predicted BG = current + trend (short-term) + metabolic effects (from IOB/COB/POB)
                predicted_bg = current_bg + trend_contribution + net_metabolic_effect

                # Uncertainty grows with horizon and metabolic activity
                base_uncertainty = 10 + (horizon * 0.25)
                if abs(rate_per_5min) > 3:
                    base_uncertainty *= 1.2  # Volatile BG
                if iob > 2 or cob > 30 or pob > 20:
                    base_uncertainty *= 1.15  # High metabolic uncertainty

                lower = predicted_bg - base_uncertainty
                upper = predicted_bg + base_uncertainty

                median = max(40, min(400, predicted_bg))
                lower = max(40, min(400, lower))
                upper = max(40, min(400, upper))

                confidence = max(0.5, 0.8 - (horizon / 200))

                if horizon in [30, 60, 120]:
                    logger.info(f"  +{horizon}min: pred={median:.0f} (trend={trend_contribution:.1f}, insulin={insulin_effect:.1f}, carb={carb_effect:.1f}, protein={protein_effect:.1f})")

                results.append(TFTPredictionItem(
                    horizon_min=horizon,
                    value=round(median, 1),
                    lower=round(lower, 1),
                    upper=round(upper, 1),
                    confidence=round(confidence, 2)
                ))

            logger.info(f"TFT formula-based generated {len(results)} predictions")
            return results

        except Exception as e:
            logger.warning(f"TFT formula-based prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_tft_fallback(glucose_history, treatments, iob)

    def _predict_tft_fallback(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Fallback TFT prediction when formula-based calculation fails.
        Uses simpler IOB-driven prediction with personalized IOB model.
        """
        try:
            import math

            if len(glucose_history) < 6:
                return None

            sorted_readings = sorted(
                glucose_history,
                key=lambda x: x.get('timestamp', x.get('dateString', '')),
                reverse=True
            )

            current_bg = float(sorted_readings[0].get('value', sorted_readings[0].get('sgv', 120)))
            recent_values = [float(r.get('value', r.get('sgv', 120))) for r in sorted_readings[:6]]
            rate_per_5min = (recent_values[0] - recent_values[-1]) / max(1, len(recent_values) - 1)

            # Use cached learned ISF if available (includes illness-adjusted effective ISF)
            isf = self._cached_isf if self._cached_isf else 55.0
            current_hour = datetime.utcnow().hour

            # Return all target horizons for consistency
            results = []
            for horizon in [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]:
                # Use PERSONALIZED IOB model (this child has 54 min half-life vs 81 min adult)
                iob_effect = self._personalized_iob.get_insulin_effect_at_horizon(
                    current_iob=iob,
                    horizon_min=horizon,
                    isf=isf,
                    hour=current_hour,
                )

                # Short-term trend with quick decay
                if iob > 0.5:
                    trend_effect = rate_per_5min * min(horizon / 5, 3) * 0.3
                else:
                    trend_effect = rate_per_5min * (horizon / 5) * math.exp(-horizon / 60)

                predicted_bg = current_bg + trend_effect + iob_effect
                uncertainty = 12 + (horizon * 0.3)

                results.append(TFTPredictionItem(
                    horizon_min=horizon,
                    value=round(max(40, min(400, predicted_bg)), 1),
                    lower=round(max(40, min(400, predicted_bg - uncertainty)), 1),
                    upper=round(max(40, min(400, predicted_bg + uncertainty)), 1),
                    confidence=max(0.5, 0.7 - (horizon / 200))
                ))

            return results

        except Exception as e:
            logger.warning(f"TFT fallback prediction error: {e}")
            return None

    def _predict_tft_original(
        self,
        glucose_history: List[dict],
        treatments: Optional[List[dict]],
        iob: float
    ) -> Optional[List[TFTPredictionItem]]:
        """
        Original TFT prediction method (kept for reference).
        """
        try:
            import numpy as np
            import math

            if len(glucose_history) < 6:
                return None

            sorted_readings = sorted(
                glucose_history,
                key=lambda x: x.get('timestamp', x.get('dateString', '')),
                reverse=True
            )

            current_bg = float(sorted_readings[0].get('value', sorted_readings[0].get('sgv', 120)))
            raw_trend = sorted_readings[0].get('trend', 0)
            if isinstance(raw_trend, str):
                trend_map = {
                    "DoubleDown": -3, "SingleDown": -2, "FortyFiveDown": -1,
                    "Flat": 0, "FortyFiveUp": 1, "SingleUp": 2, "DoubleUp": 3
                }
                trend = trend_map.get(raw_trend, 0)
            else:
                trend = int(raw_trend or 0)

            recent_values = [float(r.get('value', r.get('sgv', 120))) for r in sorted_readings[:6]]
            if len(recent_values) >= 2:
                rate_per_5min = (recent_values[0] - recent_values[-1]) / (len(recent_values) - 1)
            else:
                rate_per_5min = trend * 1.5

            carb_effect_by_horizon = {30: 0, 45: 0, 60: 0}
            insulin_effect_by_horizon = {30: 0, 45: 0, 60: 0}
            uncertainty_adjustment = 1.0
            # Use cached learned ISF if available (includes illness-adjusted effective ISF)
            isf = self._cached_isf if self._cached_isf else 50.0

            if iob > 0:
                iob_half_life_min = 90
                for horizon in [30, 45, 60]:
                    remaining_iob = iob * math.exp(-0.693 * horizon / iob_half_life_min)
                    activity = min(1.0, 0.5 + (horizon / 150))
                    absorbed_this_period = iob - remaining_iob
                    insulin_effect = absorbed_this_period * isf * activity
                    insulin_effect_by_horizon[horizon] = insulin_effect

            # Process carb treatments for glycemic data
            if treatments:
                now = datetime.utcnow()
                for treatment in treatments:
                    # Parse treatment timestamp
                    t_timestamp = treatment.get('timestamp')
                    if isinstance(t_timestamp, str):
                        try:
                            t_time = datetime.fromisoformat(t_timestamp.replace('Z', '+00:00'))
                            t_time = t_time.replace(tzinfo=None)
                        except:
                            continue
                    elif isinstance(t_timestamp, datetime):
                        t_time = t_timestamp.replace(tzinfo=None)
                    else:
                        continue

                    minutes_since = (now - t_time).total_seconds() / 60

                    # Process carb treatments with glycemic data
                    carbs = treatment.get('carbs', 0) or 0
                    if carbs > 0 and minutes_since < 300:  # Extended to 5 hours for high-fat
                        # Get glycemic data (with defaults)
                        gi = treatment.get('glycemicIndex', 55) or 55
                        absorption_rate = treatment.get('absorptionRate', 'medium') or 'medium'
                        fat_content = treatment.get('fatContent', 'low') or 'low'

                        # Calculate peak time based on GI (higher GI = faster peak)
                        gi_timing_factor = gi / 55
                        base_peak_min = 45
                        peak_time = base_peak_min / gi_timing_factor

                        # Fat delays absorption significantly
                        fat_delay = {'low': 0, 'medium': 30, 'high': 60}.get(fat_content, 0)
                        peak_time += fat_delay

                        # Duration of effect based on absorption rate and fat
                        base_duration = {'fast': 90, 'medium': 150, 'slow': 240}.get(absorption_rate, 150)
                        duration_min = base_duration + fat_delay

                        # BG rise per gram: 10g raises BG by ISF mg/dL
                        bg_per_gram = isf / 10.0

                        for horizon in [30, 45, 60]:
                            # Time from treatment to prediction horizon
                            time_at_horizon = minutes_since + horizon

                            if 0 < time_at_horizon < duration_min:
                                # Bell-curve absorption model
                                sigma = duration_min / 4
                                activity = math.exp(-0.5 * ((time_at_horizon - peak_time) / sigma) ** 2)

                                # Remaining carbs to absorb
                                absorbed_fraction = min(1.0, time_at_horizon / duration_min)
                                remaining_carbs = carbs * (1 - absorbed_fraction)

                                # Effect from carbs absorbing in this period
                                carb_effect = remaining_carbs * bg_per_gram * activity * 0.3
                                carb_effect_by_horizon[horizon] += carb_effect

                        # Higher uncertainty for high-GI or unknown foods
                        if gi > 70:
                            uncertainty_adjustment *= 1.2
                        if treatment.get('enrichedAt') is None:
                            uncertainty_adjustment *= 1.3

            # Build continuous prediction curve that is TANGENT to current BG
            # Biological systems are continuous - predictions must flow smoothly from current point
            #
            # Approach: Track cumulative insulin/carb absorption and apply incremental effects
            # This ensures predictions are tangent at t=0 and physically correct

            horizons = [30, 45, 60]
            results = []

            # Current rate of change (mg/dL per 5 min interval)
            # This is the baseline trend from recent glucose history

            logger.info(f"TFT: current_bg={current_bg}, rate_per_5min={rate_per_5min}, iob={iob}, isf={isf}")

            # Calculate prediction by tracking cumulative effects
            for horizon in horizons:
                predicted_bg = current_bg

                # Track totals for logging
                total_insulin_delta = 0.0
                total_trend_delta = 0.0
                total_carb_delta = 0.0

                for t in range(0, horizon, 5):
                    t_end = t + 5  # End of this interval

                    # Base trend contribution (decays over time - mean reversion)
                    # Trends don't continue forever in biological systems
                    damping = math.exp(-t / 60)  # ~37% remaining at 60 min
                    trend_delta = rate_per_5min * damping

                    # IOB effect: track CUMULATIVE absorption and apply incremental
                    # Total effect of IOB = IOB * ISF (e.g., 5U * 50 = 250 mg/dL total drop)
                    # But it's distributed over ~5 hours following absorption curve
                    insulin_delta = 0.0
                    if iob > 0:
                        # IOB decays as more insulin is absorbed and acts
                        # Use half-life model: IOB(t) = IOB(0) * exp(-0.693 * t / half_life)
                        # Fast-acting insulin half-life is ~60-90 minutes
                        half_life_min = 90.0

                        # Calculate IOB remaining at start and end of interval
                        iob_at_start = iob * math.exp(-0.693 * t / half_life_min)
                        iob_at_end = iob * math.exp(-0.693 * t_end / half_life_min)

                        # Insulin absorbed this interval
                        insulin_absorbed_this_interval = iob_at_start - iob_at_end

                        # BG drop = (insulin absorbed this interval) * ISF
                        insulin_delta = -insulin_absorbed_this_interval * isf

                    total_insulin_delta += insulin_delta
                    total_trend_delta += trend_delta

                    # COB effect: similar incremental approach
                    # Already mostly factored into current trend, but add pending carb effects
                    carb_delta = 0.0
                    carb_effect = carb_effect_by_horizon.get(horizon, 0)
                    if carb_effect > 0:
                        # Spread carb effect over the horizon proportionally
                        carb_delta = (carb_effect / horizon) * 5  # Per 5-min interval

                    total_carb_delta += carb_delta

                    # Total BG change this interval
                    delta_bg = trend_delta + insulin_delta + carb_delta
                    predicted_bg += delta_bg

                logger.info(f"TFT horizon={horizon}: predicted_bg={predicted_bg:.1f}, trend={total_trend_delta:.1f}, insulin={total_insulin_delta:.1f}, carbs={total_carb_delta:.1f}")

                # Uncertainty grows with time horizon
                base_uncertainty = 12 + (horizon * 0.4)

                # More uncertainty if glucose is unstable
                if abs(rate_per_5min) > 3:
                    base_uncertainty *= 1.3

                # Apply treatment-based uncertainty adjustment
                base_uncertainty *= uncertainty_adjustment

                lower = predicted_bg - base_uncertainty
                upper = predicted_bg + base_uncertainty

                # Clamp to reasonable glucose range
                median = max(40, min(400, predicted_bg))
                lower = max(40, min(400, lower))
                upper = max(40, min(400, upper))

                # Confidence based on data quality
                confidence = 0.7
                if treatments and any(t.get('enrichedAt') for t in treatments if t.get('carbs')):
                    confidence = 0.75

                results.append(TFTPredictionItem(
                    horizon_min=horizon,
                    value=round(median, 1),
                    lower=round(lower, 1),
                    upper=round(upper, 1),
                    confidence=confidence
                ))

            return results

        except Exception as e:
            logger.warning(f"TFT prediction error: {e}")
            return None

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

    def get_dual_bg_predictions(
        self,
        current_bg: float,
        treatments: List[dict],
        isf: float,
        icr: float,
        bg_trend: float = 0.0,
        duration_min: int = 120,
        step_min: int = 5
    ) -> Dict[str, Any]:
        """
        Get BOTH model-based and hardcoded BG predictions for comparison.

        Returns two prediction lines:
        1. Model-based: Uses LEARNED absorption curves (from this person's data)
           - IOB: onset=15min, ramp=75min, half-life=120min
           - COB: onset=5min, ramp=10min, half-life=45min
        2. Hardcoded: Uses standard textbook parameters
           - IOB: onset=20min, half-life=81min (adult average)
           - COB: onset=15min, half-life=45min

        Over time, you can see which prediction is more accurate!

        Args:
            current_bg: Current blood glucose (mg/dL)
            treatments: List of recent treatments as dicts
            isf: Insulin Sensitivity Factor (mg/dL per unit)
            icr: Insulin to Carb Ratio (grams per unit)
            bg_trend: Current BG trend (mg/dL per 5 min)
            duration_min: Prediction horizon (default: 120 min)
            step_min: Time step (default: 5 min)

        Returns:
            Dict with 'model' and 'hardcoded' prediction arrays, plus learned parameters
        """
        from models.schemas import Treatment
        from datetime import datetime

        # Convert dict treatments to Treatment objects
        treatment_objects = []
        for t in treatments:
            try:
                timestamp = t.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = timestamp.replace(tzinfo=None)
                elif timestamp is None:
                    continue

                treatment_obj = Treatment(
                    id=t.get('id', ''),
                    userId=t.get('userId', ''),
                    timestamp=timestamp,
                    carbs=t.get('carbs', 0) or 0,
                    insulin=t.get('insulin', 0) or 0,
                    glycemicIndex=t.get('glycemicIndex', 55) or 55,
                    notes=t.get('notes', ''),
                )
                treatment_objects.append(treatment_obj)
            except Exception as e:
                logger.warning(f"Failed to convert treatment: {e}")
                continue

        # Get model-based predictions (using learned parameters)
        model_predictions = self._iob_cob_service.predict_bg_physics_based(
            current_bg=current_bg,
            treatments=treatment_objects,
            isf=isf,
            icr=icr,
            bg_trend=bg_trend,
            duration_min=duration_min,
            step_min=step_min
        )

        # Get hardcoded predictions (using standard textbook parameters)
        hardcoded_predictions = self._iob_cob_service.predict_bg_simple_physics(
            current_bg=current_bg,
            treatments=treatment_objects,
            isf=isf,
            icr=icr,
            bg_trend=bg_trend,
            duration_min=duration_min,
            step_min=step_min
        )

        # Include the learned parameters for reference
        from services.iob_cob_service import LEARNED_ABSORPTION_PARAMS

        return {
            'model_predictions': model_predictions,
            'hardcoded_predictions': hardcoded_predictions,
            'learned_parameters': {
                'iob_onset_min': LEARNED_ABSORPTION_PARAMS['iob_onset_min'],
                'iob_ramp_min': LEARNED_ABSORPTION_PARAMS['iob_ramp_min'],
                'iob_half_life_min': LEARNED_ABSORPTION_PARAMS['iob_half_life_min'],
                'cob_onset_min': LEARNED_ABSORPTION_PARAMS['cob_onset_min'],
                'cob_ramp_min': LEARNED_ABSORPTION_PARAMS['cob_ramp_min'],
                'cob_half_life_min': LEARNED_ABSORPTION_PARAMS['cob_half_life_min'],
            },
            'hardcoded_parameters': {
                'iob_onset_min': 20.0,
                'iob_half_life_min': 81.0,
                'cob_onset_min': 15.0,
                'cob_half_life_min': 45.0,
            },
            'description': {
                'model': 'Learned from BG response data (54 boluses, 193 meals)',
                'hardcoded': 'Standard textbook parameters (adult average)'
            }
        }


class UserPredictionService:
    """
    Per-user prediction service that loads personalized models from Azure Blob Storage.

    Falls back to base model if user doesn't have a personalized model.
    Caches loaded models in memory to avoid repeated blob downloads.
    """

    def __init__(
        self,
        base_models_dir: Optional[Path] = None,
        device: str = "cpu"
    ):
        """
        Initialize the user prediction service.

        Args:
            base_models_dir: Directory containing base (fallback) models
            device: Device for inference ("cpu" or "cuda")
        """
        self.base_models_dir = base_models_dir
        self.device = device
        self._model_manager: Optional[ModelManager] = None
        self._isf_repository: Optional[LearnedISFRepository] = None

        # Base service for users without personalized models
        self._base_service: Optional[PredictionService] = None

        # Cache of per-user services
        self._user_services: Dict[str, PredictionService] = {}
        self._user_isf_cache: Dict[str, Dict[str, float]] = {}

        # Track when models were loaded (for cache invalidation)
        self._model_load_times: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)  # Reload models every hour

    async def _get_model_manager(self) -> ModelManager:
        """Get or create the model manager."""
        if self._model_manager is None:
            self._model_manager = get_model_manager()
        return self._model_manager

    async def _get_isf_repository(self) -> LearnedISFRepository:
        """Get or create the ISF repository."""
        if self._isf_repository is None:
            self._isf_repository = LearnedISFRepository()
        return self._isf_repository

    def _get_base_service(self) -> PredictionService:
        """Get or create the base prediction service."""
        if self._base_service is None:
            self._base_service = PredictionService(self.base_models_dir, self.device)
            self._base_service.initialize()
        return self._base_service

    async def get_service_for_user(self, user_id: str) -> PredictionService:
        """
        Get the prediction service for a specific user.

        Loads personalized model from blob storage if available.
        Falls back to base model otherwise.

        Args:
            user_id: User ID

        Returns:
            PredictionService configured for the user
        """
        # Check if we have a cached service that's still valid
        if user_id in self._user_services:
            load_time = self._model_load_times.get(user_id)
            if load_time and (datetime.utcnow() - load_time) < self._cache_ttl:
                return self._user_services[user_id]

        # Check blob storage for personalized model
        model_manager = await self._get_model_manager()

        if await model_manager.has_user_model(user_id):
            try:
                # Download user's model
                model_path = await model_manager.download_user_model(user_id)

                if model_path:
                    # Create service with user's model
                    user_service = PredictionService(
                        models_dir=model_path.parent,
                        device=self.device
                    )
                    user_service.initialize()

                    # Cache the service
                    self._user_services[user_id] = user_service
                    self._model_load_times[user_id] = datetime.utcnow()

                    logger.info(f"Loaded personalized model for user {user_id}")
                    return user_service

            except Exception as e:
                logger.warning(f"Failed to load personalized model for {user_id}: {e}")

        # Fall back to base service
        logger.debug(f"Using base model for user {user_id}")
        return self._get_base_service()

    async def get_learned_isf(self, user_id: str) -> Dict[str, Optional[float]]:
        """
        Get learned ISF values for a user from CosmosDB.

        Args:
            user_id: User ID

        Returns:
            Dictionary with fasting, meal, and default ISF values
        """
        # Check cache
        if user_id in self._user_isf_cache:
            return self._user_isf_cache[user_id]

        # Query from CosmosDB
        isf_repo = await self._get_isf_repository()

        try:
            isf_data = await isf_repo.get_both(user_id)

            result = {
                "fasting": isf_data["fasting"].value if isf_data.get("fasting") else None,
                "meal": isf_data["meal"].value if isf_data.get("meal") else None,
                "default": (
                    isf_data["fasting"].value if isf_data.get("fasting")
                    else isf_data["meal"].value if isf_data.get("meal")
                    else 50.0
                )
            }

            # Cache the result
            self._user_isf_cache[user_id] = result
            return result

        except Exception as e:
            logger.warning(f"Failed to get learned ISF for {user_id}: {e}")
            return {"fasting": None, "meal": None, "default": 50.0}

    async def predict_for_user(
        self,
        user_id: str,
        current_bg: float,
        trend: int,
        iob: float = 0.0,
        cob: float = 0.0,
        pob: float = 0.0,
        glucose_history: Optional[List[dict]] = None,
        treatments: Optional[List[dict]] = None,
        include_illness_detection: bool = True,
    ) -> PredictionResult:
        """
        Generate predictions for a specific user.

        Uses personalized model if available, otherwise falls back to base model.
        Uses LEARNED ISF/ICR/PIR from CosmosDB, with short-term deviation for illness detection.

        IMPORTANT: This method uses effective metabolic parameters that account for:
        - Long-term learned baselines (from historical data)
        - Short-term deviations (illness/sensitivity detection from last 2-3 days)

        If a user is sick (insulin resistant), the effective ISF will be LOWER,
        meaning predictions will show less BG drop from insulin.

        Args:
            user_id: User ID
            current_bg: Current glucose value in mg/dL
            trend: CGM trend arrow (-3 to +3)
            iob: Current insulin on board
            cob: Current carbs on board
            pob: Current protein on board
            glucose_history: Optional list of recent glucose readings
            treatments: Optional list of recent treatments
            include_illness_detection: Whether to use short-term ISF deviation (default True)

        Returns:
            PredictionResult with predictions using learned metabolic parameters
        """
        # Get effective metabolic parameters (ISF, ICR, PIR) with illness detection
        try:
            params = await self._metabolic_params_service.get_all_params(
                user_id=user_id,
                is_fasting=(cob == 0 and pob == 0),
                meal_type=None,
                include_short_term=include_illness_detection
            )

            # Cache the values for use in internal prediction methods
            self._cached_user_id = user_id
            self._cached_isf = params.isf.value
            self._cached_icr = params.icr.value
            self._cached_pir = params.pir.value

            # Log if illness detected
            if params.isf.is_sick or params.isf.is_resistant:
                logger.info(
                    f"Metabolic state for user {user_id}: {params.metabolic_state.value} - "
                    f"ISF deviation: {params.isf.deviation_percent:.1f}% "
                    f"(baseline={params.isf.baseline:.1f}, effective={params.isf.value:.1f})"
                )

        except Exception as e:
            logger.warning(f"Failed to get metabolic params for user {user_id}: {e}")
            # Fall back to old method
            isf_values = await self.get_learned_isf(user_id)
            self._cached_user_id = user_id
            self._cached_isf = isf_values.get("default", 50.0)
            self._cached_icr = 10.0
            self._cached_pir = 25.0

        # Get service for this user
        service = await self.get_service_for_user(user_id)

        # Pass cached params to the service instance
        service._cached_user_id = self._cached_user_id
        service._cached_isf = self._cached_isf
        service._cached_icr = self._cached_icr
        service._cached_pir = self._cached_pir

        # Make prediction
        result = service.predict(
            current_bg=current_bg,
            trend=trend,
            iob=iob,
            cob=cob,
            pob=pob,
            glucose_history=glucose_history,
            treatments=treatments
        )

        # Set the effective ISF in the result
        result.isf = self._cached_isf

        return result

    async def invalidate_user_cache(self, user_id: str) -> None:
        """
        Invalidate cached model and ISF for a user.

        Call this after training a new model or updating ISF.

        Args:
            user_id: User ID to invalidate
        """
        if user_id in self._user_services:
            del self._user_services[user_id]
        if user_id in self._model_load_times:
            del self._model_load_times[user_id]
        if user_id in self._user_isf_cache:
            del self._user_isf_cache[user_id]

        logger.info(f"Invalidated cache for user {user_id}")

    async def preload_user_model(self, user_id: str) -> bool:
        """
        Preload a user's model into memory.

        Useful for warming up the cache before predictions are needed.

        Args:
            user_id: User ID

        Returns:
            True if model was loaded successfully
        """
        try:
            service = await self.get_service_for_user(user_id)
            return service.lstm_available
        except Exception as e:
            logger.warning(f"Failed to preload model for {user_id}: {e}")
            return False

    def get_cached_users(self) -> List[str]:
        """Get list of users with cached models."""
        return list(self._user_services.keys())

    async def close(self) -> None:
        """Close resources and clear caches."""
        self._user_services.clear()
        self._model_load_times.clear()
        self._user_isf_cache.clear()

        if self._model_manager:
            await self._model_manager.close()


# Singleton instances
_prediction_service: Optional[PredictionService] = None
_user_prediction_service: Optional[UserPredictionService] = None

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


def get_user_prediction_service(
    base_models_dir: Optional[Path] = None,
    device: str = "cpu"
) -> UserPredictionService:
    """
    Get or create the global user prediction service.

    This service supports per-user personalized models.

    Args:
        base_models_dir: Directory containing base (fallback) models
        device: Device for inference

    Returns:
        UserPredictionService instance
    """
    global _user_prediction_service

    if _user_prediction_service is None:
        _user_prediction_service = UserPredictionService(base_models_dir, device)

    return _user_prediction_service
