# T1D-AI ML Models Package
from .bg_predictor import BG_PredictorNet, BG_MODEL_CONFIG, BG_FEATURE_COLUMNS
from .isf_net import ISFNet, ISF_MODEL_CONFIG
from .iob_model import PersonalizedIOBModel, IOBModelService, IOB_MODEL_CONFIG
from .cob_model import PersonalizedCOBModel, COBModelService, COB_MODEL_CONFIG
from .tft_predictor import (
    TemporalFusionTransformer,
    TFT_MODEL_CONFIG,
    TFTOutput,
    TFTPrediction,
    QuantileLoss,
    create_tft_model
)

# Forcing function models (physics-informed BG prediction)
from .iob_forcing import IOBForcingModel, IOBForcingService, get_iob_forcing_service
from .cob_forcing import COBForcingModel, COBForcingService, get_cob_forcing_service
from .physics_baseline import PhysicsBaseline, PhysicsPrediction, get_physics_baseline
from .residual_tft import ResidualModel, ResidualService, get_residual_service, SecondaryFeatures
from .forcing_ensemble import ForcingFunctionEnsemble, ForcingPrediction, get_forcing_ensemble

# Absorption curve learners (learn onset/ramp-up from actual data)
from .absorption_learner import (
    InsulinAbsorptionLearner,
    CarbAbsorptionLearner,
    AbsorptionCurveParams,
    get_insulin_absorption_learner,
    get_carb_absorption_learner,
    extract_absorption_training_data,
    fit_absorption_curve,
)

__all__ = [
    # BG Prediction (LSTM - short term)
    "BG_PredictorNet",
    "BG_MODEL_CONFIG",
    "BG_FEATURE_COLUMNS",
    # BG Prediction (TFT - long term with uncertainty)
    "TemporalFusionTransformer",
    "TFT_MODEL_CONFIG",
    "TFTOutput",
    "TFTPrediction",
    "QuantileLoss",
    "create_tft_model",
    # ISF Learning
    "ISFNet",
    "ISF_MODEL_CONFIG",
    # IOB Model (personalized absorption)
    "PersonalizedIOBModel",
    "IOBModelService",
    "IOB_MODEL_CONFIG",
    # COB Model (personalized absorption)
    "PersonalizedCOBModel",
    "COBModelService",
    "COB_MODEL_CONFIG",
    # Forcing Function Models (physics-informed prediction)
    "IOBForcingModel",
    "IOBForcingService",
    "get_iob_forcing_service",
    "COBForcingModel",
    "COBForcingService",
    "get_cob_forcing_service",
    "PhysicsBaseline",
    "PhysicsPrediction",
    "get_physics_baseline",
    "ResidualModel",
    "ResidualService",
    "get_residual_service",
    "SecondaryFeatures",
    "ForcingFunctionEnsemble",
    "ForcingPrediction",
    "get_forcing_ensemble",
    # Absorption curve learners
    "InsulinAbsorptionLearner",
    "CarbAbsorptionLearner",
    "AbsorptionCurveParams",
    "get_insulin_absorption_learner",
    "get_carb_absorption_learner",
    "extract_absorption_training_data",
    "fit_absorption_curve",
]
