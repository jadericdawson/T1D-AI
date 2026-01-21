"""
MLflow Tracking Configuration for T1D-AI Models

Centralizes MLflow experiment tracking for all ML models:
- TFT (Temporal Fusion Transformer) for glucose prediction
- IOB Model (Personalized insulin absorption)
- COB Model (Personalized carb absorption)
- BG Pressure Model (Combined IOB+COB effect prediction)
- ISF Model (Insulin Sensitivity Factor prediction)

Supports multiple backends:
- Local: file:// or http://localhost:5002 (development)
- Azure ML: azureml://... URI (production)

Azure ML Workspace provides:
- Persistent experiment tracking across deployments
- Model registry with versioning
- Per-user experiment namespacing
- ~$5-10/month cost (storage + container registry)
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _get_mlflow_config() -> tuple[str, str]:
    """
    Get MLflow tracking URI and experiment prefix from config.

    Priority:
    1. Environment variable MLFLOW_TRACKING_URI
    2. Config settings (Azure ML or local)
    3. Default to local file store

    Returns:
        (tracking_uri, experiment_prefix)
    """
    # Try to import config settings
    try:
        from config import get_settings
        settings = get_settings()

        # Check for explicit tracking URI in config
        if settings.mlflow_tracking_uri:
            return settings.mlflow_tracking_uri, settings.mlflow_experiment_prefix

        # Build Azure ML URI if workspace is configured
        if settings.azure_ml_workspace_name and settings.azure_ml_subscription_id:
            azure_uri = (
                f"azureml://{settings.azure_ml_region}.api.azureml.ms/mlflow/v1.0/"
                f"subscriptions/{settings.azure_ml_subscription_id}/"
                f"resourceGroups/{settings.azure_ml_resource_group}/"
                f"providers/Microsoft.MachineLearningServices/workspaces/"
                f"{settings.azure_ml_workspace_name}"
            )
            logger.info(f"Using Azure ML for MLflow tracking: {settings.azure_ml_workspace_name}")
            return azure_uri, settings.mlflow_experiment_prefix

        # Return experiment prefix from config even if using env var for URI
        return (
            os.getenv("MLFLOW_TRACKING_URI", ""),
            settings.mlflow_experiment_prefix
        )

    except Exception as e:
        logger.debug(f"Config not available, using environment variables: {e}")

    # Fallback to environment variables
    return (
        os.getenv("MLFLOW_TRACKING_URI", ""),
        os.getenv("MLFLOW_EXPERIMENT_PREFIX", "T1D-AI")
    )


# Get configuration at module load
_uri, _prefix = _get_mlflow_config()
MLFLOW_TRACKING_URI = _uri
MLFLOW_EXPERIMENT_PREFIX = _prefix

# Log the configuration
if MLFLOW_TRACKING_URI:
    if "azureml://" in MLFLOW_TRACKING_URI:
        logger.info("MLflow configured for Azure ML Workspace")
    elif MLFLOW_TRACKING_URI.startswith("http"):
        logger.info(f"MLflow configured for remote server: {MLFLOW_TRACKING_URI}")
    else:
        logger.info(f"MLflow configured for local store: {MLFLOW_TRACKING_URI}")
else:
    logger.info("MLflow tracking disabled (no URI configured)")

# Experiment names for each model type
EXPERIMENTS = {
    "tft": f"{MLFLOW_EXPERIMENT_PREFIX}/TFT-GlucosePredictor",
    "iob": f"{MLFLOW_EXPERIMENT_PREFIX}/IOB-Personalized",
    "cob": f"{MLFLOW_EXPERIMENT_PREFIX}/COB-Personalized",
    "bg_pressure": f"{MLFLOW_EXPERIMENT_PREFIX}/BG-Pressure",
    "isf": f"{MLFLOW_EXPERIMENT_PREFIX}/ISF-Personalized",
    "icr": f"{MLFLOW_EXPERIMENT_PREFIX}/ICR-Personalized",
    "pir": f"{MLFLOW_EXPERIMENT_PREFIX}/PIR-Personalized",
    "lstm": f"{MLFLOW_EXPERIMENT_PREFIX}/LSTM-GlucosePredictor",
    # Forcing function models (physics-informed BG prediction)
    "iob_forcing": f"{MLFLOW_EXPERIMENT_PREFIX}/IOB-Forcing",
    "cob_forcing": f"{MLFLOW_EXPERIMENT_PREFIX}/COB-Forcing",
    "residual_tft": f"{MLFLOW_EXPERIMENT_PREFIX}/Residual-TFT",
    "forcing_ensemble": f"{MLFLOW_EXPERIMENT_PREFIX}/Forcing-Ensemble",
}


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is enabled."""
    return bool(MLFLOW_TRACKING_URI)


def setup_mlflow(experiment_name: str = None, user_id: str = None) -> str:
    """
    Set up MLflow tracking for the given experiment.

    For Azure ML, experiments are namespaced by user:
    - Base models: T1D-AI/global/{model_type}
    - User models: T1D-AI/users/{user_id}/{model_type}

    Args:
        experiment_name: Name of the experiment (uses default if not provided)
        user_id: Optional user ID for per-user experiment namespacing

    Returns:
        Experiment ID (or "0" if tracking is disabled)
    """
    if not MLFLOW_TRACKING_URI:
        logger.debug("MLflow tracking disabled (no URI configured)")
        return "0"

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

        # Build namespaced experiment name for Azure ML
        if experiment_name:
            if user_id and "azureml://" in MLFLOW_TRACKING_URI:
                # Per-user namespacing for Azure ML
                full_name = f"{MLFLOW_EXPERIMENT_PREFIX}/users/{user_id}/{experiment_name}"
            else:
                full_name = experiment_name
            mlflow.set_experiment(full_name)
        else:
            full_name = "Default"

        experiment = mlflow.get_experiment_by_name(full_name)
        experiment_id = experiment.experiment_id if experiment else "0"

        logger.info(f"MLflow experiment '{full_name}' ready, ID: {experiment_id}")
        return experiment_id

    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Tracking disabled.")
        return "0"


def get_mlflow_client() -> Optional[MlflowClient]:
    """
    Get MLflow client for API operations.

    Returns:
        MlflowClient or None if tracking is disabled
    """
    if not MLFLOW_TRACKING_URI:
        logger.debug("MLflow tracking disabled, client not available")
        return None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


class ModelTracker:
    """
    MLflow model tracking wrapper for T1D-AI models.

    Provides standardized logging for:
    - Model parameters
    - Training metrics
    - Model artifacts (weights, scalers)
    - Feature importance
    - Performance comparison

    Supports per-user experiment namespacing for Azure ML:
    - Base models: T1D-AI/global/{model_type}
    - User models: T1D-AI/users/{user_id}/{model_type}
    """

    def __init__(
        self,
        model_type: str,
        user_id: Optional[str] = None,
    ):
        """
        Initialize model tracker.

        Args:
            model_type: One of 'tft', 'iob', 'cob', 'bg_pressure', 'isf', 'lstm'
            user_id: User ID for personalized models (None for base models)
        """
        self.model_type = model_type
        self.user_id = user_id or "base"
        self.enabled = is_mlflow_enabled()

        # Build experiment name with user namespacing for Azure ML
        base_experiment = EXPERIMENTS.get(model_type, f"{MLFLOW_EXPERIMENT_PREFIX}/{model_type}")

        if self.user_id != "base" and "azureml://" in MLFLOW_TRACKING_URI:
            # Per-user namespacing: T1D-AI/users/{user_id}/{model_type}
            self.experiment_name = f"{MLFLOW_EXPERIMENT_PREFIX}/users/{self.user_id}/{model_type}"
        else:
            self.experiment_name = base_experiment

        self.run_id = None
        self._setup()

    def _setup(self):
        """Set up MLflow for this tracker."""
        if not self.enabled:
            logger.debug(f"MLflow disabled - tracker for {self.model_type} will not record")
            return

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"ModelTracker ready for {self.model_type} (user: {self.user_id}, experiment: {self.experiment_name})")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.enabled = False

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run

        Returns:
            Run ID (empty string if tracking is disabled)
        """
        if not self.enabled:
            logger.debug(f"MLflow tracking disabled, skipping run start for {self.model_type}")
            return ""

        try:
            run_name = run_name or f"{self.model_type}_{self.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            default_tags = {
                "model_type": self.model_type,
                "user_id": self.user_id,
                "framework": "pytorch",
            }
            if tags:
                default_tags.update(tags)

            run = mlflow.start_run(run_name=run_name, tags=default_tags)
            self.run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")
            return self.run_id

        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.run_id = None
            return ""

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters."""
        if not self.enabled or not self.run_id:
            return

        try:
            # Flatten nested dicts
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.debug(f"Logged {len(flat_params)} parameters")
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training/evaluation metrics."""
        if not self.run_id:
            return

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged {len(metrics)} metrics at step {step}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_name: Optional[str] = None,
    ) -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within the run to store the model
            registered_name: If provided, register model in MLflow registry
        """
        if not self.run_id:
            return

        try:
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name,
            )
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file (e.g., scaler, config)."""
        if not self.run_id:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def log_feature_importance(self, importance: Dict[str, float]) -> None:
        """Log feature importance scores."""
        if not self.run_id:
            return

        try:
            # Log as metrics
            for feature, score in importance.items():
                safe_name = feature.replace("/", "_").replace(" ", "_")
                mlflow.log_metric(f"feature_importance_{safe_name}", score)

            # Also log as artifact
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(importance, f, indent=2)
                f.flush()
                mlflow.log_artifact(f.name, "feature_importance")

            logger.info(f"Logged feature importance for {len(importance)} features")
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    def log_training_curve(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log training curve data point."""
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        self.log_metrics(metrics, step=epoch)

    def log_prediction_accuracy(
        self,
        horizon_min: int,
        mae: float,
        rmse: float,
        mape: Optional[float] = None,
    ) -> None:
        """Log prediction accuracy for a specific horizon."""
        metrics = {
            f"mae_{horizon_min}min": mae,
            f"rmse_{horizon_min}min": rmse,
        }
        if mape is not None:
            metrics[f"mape_{horizon_min}min"] = mape

        self.log_metrics(metrics)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if not self.run_id:
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id} with status {status}")
        except Exception as e:
            logger.warning(f"Failed to end run: {e}")
        finally:
            self.run_id = None

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def get_best_model_run(
    model_type: str,
    user_id: Optional[str] = None,
    metric: str = "val_loss",
    ascending: bool = True,
) -> Optional[Dict]:
    """
    Get the best run for a model type based on a metric.

    Args:
        model_type: Model type ('tft', 'iob', 'cob', etc.)
        user_id: User ID (None for base models)
        metric: Metric to sort by
        ascending: Sort order (True for loss, False for accuracy)

    Returns:
        Dict with run info and metrics, or None if no runs found or tracking disabled
    """
    if not is_mlflow_enabled():
        return None

    try:
        experiment_name = EXPERIMENTS.get(model_type)
        if not experiment_name:
            return None

        client = get_mlflow_client()
        if not client:
            return None
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        # Build filter string
        filter_str = ""
        if user_id:
            filter_str = f"tags.user_id = '{user_id}'"
        else:
            filter_str = "tags.user_id = 'base'"

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_str,
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            return None

        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "artifact_uri": run.info.artifact_uri,
        }

    except Exception as e:
        logger.warning(f"Failed to get best model run: {e}")
        return None


def compare_model_runs(
    model_type: str,
    run_ids: List[str],
    metrics: List[str] = None,
) -> Dict:
    """
    Compare multiple model runs.

    Args:
        model_type: Model type
        run_ids: List of run IDs to compare
        metrics: List of metrics to compare (None for all)

    Returns:
        Dict mapping run_id to metrics
    """
    try:
        client = get_mlflow_client()
        comparison = {}

        for run_id in run_ids:
            run = client.get_run(run_id)
            run_metrics = run.data.metrics

            if metrics:
                run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

            comparison[run_id] = {
                "name": run.info.run_name,
                "metrics": run_metrics,
                "params": run.data.params,
            }

        return comparison

    except Exception as e:
        logger.warning(f"Failed to compare runs: {e}")
        return {}


def start_mlflow_server(port: int = 5002, backend_dir: str = None):
    """
    Start the MLflow tracking server.

    This is typically run as a separate process, not from within the app.

    Args:
        port: Port to run the server on
        backend_dir: Directory for MLflow backend store
    """
    backend_dir = backend_dir or str(Path(__file__).parent.parent.parent / "mlruns")

    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--backend-store-uri", f"file://{backend_dir}",
        "--default-artifact-root", f"file://{backend_dir}",
    ]

    logger.info(f"Starting MLflow server on port {port}")
    logger.info(f"Backend store: {backend_dir}")
    logger.info(f"Command: {' '.join(cmd)}")

    import subprocess
    subprocess.run(cmd)


# Example usage in training scripts:
"""
from ml.mlflow_tracking import ModelTracker

# Initialize tracker
tracker = ModelTracker(model_type="tft", user_id="demo_user")

# Start a run
tracker.start_run(run_name="tft_training_v2")

# Log parameters
tracker.log_params({
    "n_features": 69,
    "hidden_size": 64,
    "n_heads": 4,
    "learning_rate": 0.001,
})

# Training loop
for epoch in range(100):
    train_loss, val_loss = train_epoch(model, train_loader, val_loader)
    tracker.log_training_curve(epoch, train_loss, val_loss)

# Log final metrics
tracker.log_prediction_accuracy(30, mae=15.2, rmse=19.1)
tracker.log_prediction_accuracy(60, mae=28.5, rmse=35.2)

# Log model
tracker.log_model(model, artifact_path="tft_model")

# End run
tracker.end_run()
"""
