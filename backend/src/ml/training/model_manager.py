"""
Model Manager for T1D-AI
Handles storage and retrieval of per-user ML models in Azure Blob Storage.

Structure in t1d-ai-models container:
├── base/                           # Population baseline model
│   ├── bg_predictor_3step_v2.pth
│   └── scalers/
│       ├── features_scaler.pkl
│       └── targets_scaler.pkl
└── users/
    └── {user_id}/
        ├── bg_predictor.pth        # Personalized model
        ├── scalers/
        │   ├── features_scaler.pkl
        │   └── targets_scaler.pkl
        ├── isf_fasting.json        # Cached ISF data
        ├── isf_meal.json
        └── training_metadata.json  # Training date, metrics
"""
import io
import json
import logging
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, BinaryIO, Any

import torch
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model storage in Azure Blob Storage."""

    def __init__(self):
        self.settings = get_settings()
        self.container_name = self.settings.models_container
        self._client: Optional[BlobServiceClient] = None
        self._container: Optional[ContainerClient] = None

    async def _get_container(self) -> ContainerClient:
        """Get or create the blob container client."""
        if self._container is None:
            self._client = BlobServiceClient.from_connection_string(
                self.settings.storage_connection_string
            )
            self._container = self._client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                await self._container.get_container_properties()
            except ResourceNotFoundError:
                await self._container.create_container()
                logger.info(f"Created blob container: {self.container_name}")

        return self._container

    async def close(self):
        """Close the blob client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._container = None

    # ==================== User Model Operations ====================

    async def has_user_model(self, user_id: str) -> bool:
        """Check if a user has a personalized model."""
        container = await self._get_container()
        blob_name = f"users/{user_id}/bg_predictor.pth"

        blob_client = container.get_blob_client(blob_name)
        return await blob_client.exists()

    async def upload_user_model(
        self,
        user_id: str,
        model_path: Path,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Upload a user's personalized model to blob storage.

        Args:
            user_id: User ID
            model_path: Local path to the model file
            metadata: Optional training metadata

        Returns:
            Blob URL of the uploaded model
        """
        container = await self._get_container()
        blob_name = f"users/{user_id}/bg_predictor.pth"

        # Upload model file
        blob_client = container.get_blob_client(blob_name)
        with open(model_path, "rb") as f:
            await blob_client.upload_blob(f, overwrite=True)

        logger.info(f"Uploaded model for user {user_id}: {blob_name}")

        # Upload metadata if provided
        if metadata:
            metadata["uploadedAt"] = datetime.now(timezone.utc).isoformat()
            await self._upload_json(
                f"users/{user_id}/training_metadata.json",
                metadata
            )

        return blob_client.url

    async def download_user_model(self, user_id: str) -> Optional[Path]:
        """
        Download a user's personalized model to a temp file.

        Args:
            user_id: User ID

        Returns:
            Path to downloaded model file, or None if not found
        """
        container = await self._get_container()
        blob_name = f"users/{user_id}/bg_predictor.pth"

        blob_client = container.get_blob_client(blob_name)

        if not await blob_client.exists():
            logger.info(f"No personalized model for user {user_id}")
            return None

        # Download to temp file
        temp_dir = Path(tempfile.gettempdir()) / "t1d-ai-models" / user_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        model_path = temp_dir / "bg_predictor.pth"

        download_stream = await blob_client.download_blob()
        with open(model_path, "wb") as f:
            data = await download_stream.readall()
            f.write(data)

        logger.info(f"Downloaded model for user {user_id}: {model_path}")
        return model_path

    async def load_user_model(self, user_id: str, device: str = "cpu") -> Optional[torch.nn.Module]:
        """
        Load a user's personalized PyTorch model.

        Args:
            user_id: User ID
            device: Device to load model on ("cpu" or "cuda")

        Returns:
            Loaded PyTorch model or None if not found
        """
        model_path = await self.download_user_model(user_id)
        if not model_path:
            return None

        try:
            model = torch.load(model_path, map_location=device)
            model.eval()
            logger.info(f"Loaded model for user {user_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for user {user_id}: {e}")
            return None

    async def delete_user_model(self, user_id: str) -> bool:
        """Delete all model files for a user."""
        container = await self._get_container()
        prefix = f"users/{user_id}/"

        deleted = False
        async for blob in container.list_blobs(name_starts_with=prefix):
            blob_client = container.get_blob_client(blob.name)
            await blob_client.delete_blob()
            deleted = True
            logger.info(f"Deleted blob: {blob.name}")

        return deleted

    # ==================== Scaler Operations ====================

    async def upload_user_scalers(
        self,
        user_id: str,
        features_scaler: Any,
        targets_scaler: Any
    ):
        """Upload user's feature and target scalers."""
        container = await self._get_container()

        # Upload features scaler
        features_blob = f"users/{user_id}/scalers/features_scaler.pkl"
        await self._upload_pickle(features_blob, features_scaler)

        # Upload targets scaler
        targets_blob = f"users/{user_id}/scalers/targets_scaler.pkl"
        await self._upload_pickle(targets_blob, targets_scaler)

        logger.info(f"Uploaded scalers for user {user_id}")

    async def download_user_scalers(self, user_id: str) -> tuple[Any, Any]:
        """
        Download user's scalers.

        Returns:
            Tuple of (features_scaler, targets_scaler) or (None, None) if not found
        """
        features_scaler = await self._download_pickle(
            f"users/{user_id}/scalers/features_scaler.pkl"
        )
        targets_scaler = await self._download_pickle(
            f"users/{user_id}/scalers/targets_scaler.pkl"
        )

        return features_scaler, targets_scaler

    # ==================== Base Model Operations ====================

    async def get_base_model_path(self) -> Optional[Path]:
        """
        Get path to the base (population) model.

        Downloads from blob storage if not cached locally.
        """
        container = await self._get_container()
        blob_name = "base/bg_predictor_3step_v2.pth"

        blob_client = container.get_blob_client(blob_name)

        if not await blob_client.exists():
            logger.warning("Base model not found in blob storage")
            return None

        # Download to cache
        cache_dir = Path(tempfile.gettempdir()) / "t1d-ai-models" / "base"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "bg_predictor_3step_v2.pth"

        # Check if already cached
        if model_path.exists():
            return model_path

        download_stream = await blob_client.download_blob()
        with open(model_path, "wb") as f:
            data = await download_stream.readall()
            f.write(data)

        logger.info(f"Downloaded base model: {model_path}")
        return model_path

    async def upload_base_model(self, model_path: Path):
        """Upload or update the base model."""
        container = await self._get_container()
        blob_name = "base/bg_predictor_3step_v2.pth"

        blob_client = container.get_blob_client(blob_name)
        with open(model_path, "rb") as f:
            await blob_client.upload_blob(f, overwrite=True)

        logger.info(f"Uploaded base model: {blob_name}")

    # ==================== ISF Cache Operations ====================

    async def cache_user_isf(
        self,
        user_id: str,
        isf_type: str,
        isf_data: dict
    ):
        """
        Cache learned ISF data to blob storage.

        This is a backup/export of the CosmosDB data.
        """
        blob_name = f"users/{user_id}/isf_{isf_type}.json"
        await self._upload_json(blob_name, isf_data)
        logger.info(f"Cached {isf_type} ISF for user {user_id}")

    async def get_cached_isf(
        self,
        user_id: str,
        isf_type: str
    ) -> Optional[dict]:
        """Get cached ISF data from blob storage."""
        blob_name = f"users/{user_id}/isf_{isf_type}.json"
        return await self._download_json(blob_name)

    # ==================== Training Metadata ====================

    async def get_training_metadata(self, user_id: str) -> Optional[dict]:
        """Get training metadata for a user's model."""
        blob_name = f"users/{user_id}/training_metadata.json"
        return await self._download_json(blob_name)

    async def update_training_metadata(
        self,
        user_id: str,
        metadata: dict
    ):
        """Update training metadata for a user."""
        blob_name = f"users/{user_id}/training_metadata.json"

        # Merge with existing
        existing = await self._download_json(blob_name) or {}
        existing.update(metadata)
        existing["lastUpdated"] = datetime.now(timezone.utc).isoformat()

        await self._upload_json(blob_name, existing)

    # ==================== Helper Methods ====================

    async def _upload_json(self, blob_name: str, data: dict):
        """Upload JSON data to blob."""
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        await blob_client.upload_blob(json_bytes, overwrite=True)

    async def _download_json(self, blob_name: str) -> Optional[dict]:
        """Download JSON data from blob."""
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        try:
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            return json.loads(data.decode("utf-8"))
        except ResourceNotFoundError:
            return None

    async def _upload_pickle(self, blob_name: str, obj: Any):
        """Upload pickled object to blob."""
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)

        await blob_client.upload_blob(buffer, overwrite=True)

    async def _download_pickle(self, blob_name: str) -> Any:
        """Download and unpickle object from blob."""
        container = await self._get_container()
        blob_client = container.get_blob_client(blob_name)

        try:
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            return pickle.loads(data)
        except ResourceNotFoundError:
            return None

    # ==================== Listing Operations ====================

    async def list_user_models(self) -> list[str]:
        """List all users who have personalized models."""
        container = await self._get_container()
        users = set()

        async for blob in container.list_blobs(name_starts_with="users/"):
            # Extract user ID from path like "users/{user_id}/..."
            parts = blob.name.split("/")
            if len(parts) >= 2:
                users.add(parts[1])

        return sorted(list(users))

    async def get_model_stats(self, user_id: str) -> Optional[dict]:
        """Get statistics about a user's model."""
        container = await self._get_container()

        has_model = await self.has_user_model(user_id)
        metadata = await self.get_training_metadata(user_id)

        if not has_model:
            return None

        # Get model file size
        blob_client = container.get_blob_client(f"users/{user_id}/bg_predictor.pth")
        props = await blob_client.get_blob_properties()

        return {
            "userId": user_id,
            "hasModel": has_model,
            "modelSizeBytes": props.size,
            "lastModified": props.last_modified.isoformat() if props.last_modified else None,
            "trainingMetadata": metadata
        }


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the model manager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
