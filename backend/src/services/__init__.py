# T1D-AI Services Package
# Note: Import specific items as needed to avoid circular imports

__all__ = [
    # Gluroo
    "GlurooService",
    "create_gluroo_service",
    # IOB/COB
    "IOBCOBService",
    "calculate_iob_simple",
    "calculate_cob_simple",
    # Predictions
    "PredictionService",
    "get_prediction_service",
    # Accuracy
    "AccuracyTracker",
    "get_accuracy_tracker",
    # OpenAI
    "OpenAIService",
    "openai_service",
    # Insights
    "InsightService",
    "insight_service",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "GlurooService":
        from services.gluroo_service import GlurooService
        return GlurooService
    elif name == "create_gluroo_service":
        from services.gluroo_service import create_gluroo_service
        return create_gluroo_service
    elif name == "IOBCOBService":
        from services.iob_cob_service import IOBCOBService
        return IOBCOBService
    elif name == "calculate_iob_simple":
        from services.iob_cob_service import calculate_iob_simple
        return calculate_iob_simple
    elif name == "calculate_cob_simple":
        from services.iob_cob_service import calculate_cob_simple
        return calculate_cob_simple
    elif name == "PredictionService":
        from services.prediction_service import PredictionService
        return PredictionService
    elif name == "get_prediction_service":
        from services.prediction_service import get_prediction_service
        return get_prediction_service
    elif name == "AccuracyTracker":
        from services.accuracy_service import AccuracyTracker
        return AccuracyTracker
    elif name == "get_accuracy_tracker":
        from services.accuracy_service import get_accuracy_tracker
        return get_accuracy_tracker
    elif name == "OpenAIService":
        from services.openai_service import OpenAIService
        return OpenAIService
    elif name == "openai_service":
        from services.openai_service import openai_service
        return openai_service
    elif name == "InsightService":
        from services.insight_service import InsightService
        return InsightService
    elif name == "insight_service":
        from services.insight_service import insight_service
        return insight_service
    raise AttributeError(f"module 'services' has no attribute '{name}'")
