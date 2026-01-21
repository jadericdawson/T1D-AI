"""
Azure Blob Storage Model Store

Manages trained ML models in Azure Blob Storage.
Supports per-user personalized models with versioning.
"""
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, BinaryIO
from pathlib import Path
import tempfile
import io

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from config import get_settings

logger = logging.getLogger(__name__)


class BlobStorageModelStore:
    """
    Store and retrieve ML models from Azure Blob Storage.

    Blob structure:
    - models/global/{model_type}/latest.pth - Global shared models
    - models/users/{user_id}/{model_type}/v{version}.pth - Per-user models
    - models/users/{user_id}/{model_type}/metadata.json - Model metadata
    """

    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None):
        settings = get_settings()
        self.connection_string = connection_string or settings.storage_connection_string
        self.container_name = container_name or settings.models_container

        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

        # Ensure container exists
        self._ensure_container()

    def _ensure_container(self):
        """Create container if it doesn't exist."""
        try:
            self.container_client.create_container()
            logger.info(f"Created blob container: {self.container_name}")
        except Exception as e:
            if "ContainerAlreadyExists" not in str(e):
                logger.debug(f"Container check: {e}")

    def _get_user_model_path(self, user_id: str, model_type: str, version: Optional[int] = None) -> str:
        """Get blob path for user model."""
        if version:
            return f"models/users/{user_id}/{model_type}/v{version}.pth"
        return f"models/users/{user_id}/{model_type}/latest.pth"

    def _get_metadata_path(self, user_id: str, model_type: str) -> str:
        """Get blob path for model metadata."""
        return f"models/users/{user_id}/{model_type}/metadata.json"

    def _get_global_model_path(self, model_type: str) -> str:
        """Get blob path for global model."""
        return f"models/global/{model_type}/latest.pth"

    async def upload_user_model(
        self,
        user_id: str,
        model_type: str,
        model_data: bytes,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Upload a trained model for a user.

        Args:
            user_id: User ID
            model_type: Type of model (tft, isf, iob, cob)
            model_data: Serialized model bytes
            metrics: Training metrics
            config: Training configuration
            version: Optional version number (auto-increments if not provided)

        Returns:
            Dict with upload details
        """
        try:
            # Get next version if not provided
            if version is None:
                version = await self._get_next_version(user_id, model_type)

            # Upload versioned model
            versioned_path = self._get_user_model_path(user_id, model_type, version)
            blob_client = self.container_client.get_blob_client(versioned_path)
            blob_client.upload_blob(model_data, overwrite=True)

            # Also save as latest
            latest_path = self._get_user_model_path(user_id, model_type)
            latest_blob = self.container_client.get_blob_client(latest_path)
            latest_blob.upload_blob(model_data, overwrite=True)

            # Save metadata
            metadata = {
                "user_id": user_id,
                "model_type": model_type,
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "config": config or {},
                "blob_path": versioned_path,
                "size_bytes": len(model_data)
            }

            metadata_path = self._get_metadata_path(user_id, model_type)
            metadata_blob = self.container_client.get_blob_client(metadata_path)
            metadata_blob.upload_blob(
                json.dumps(metadata, indent=2),
                overwrite=True,
                content_settings={"content_type": "application/json"}
            )

            logger.info(f"Uploaded model for user {user_id}: {model_type} v{version}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to upload model for {user_id}: {e}")
            raise

    async def download_user_model(
        self,
        user_id: str,
        model_type: str,
        version: Optional[int] = None
    ) -> Optional[bytes]:
        """
        Download a user's trained model.

        Args:
            user_id: User ID
            model_type: Type of model
            version: Optional specific version (latest if not provided)

        Returns:
            Model bytes or None if not found
        """
        try:
            blob_path = self._get_user_model_path(user_id, model_type, version)
            blob_client = self.container_client.get_blob_client(blob_path)

            download = blob_client.download_blob()
            model_data = download.readall()

            logger.info(f"Downloaded model for user {user_id}: {model_type}")
            return model_data

        except ResourceNotFoundError:
            logger.debug(f"No model found for user {user_id}: {model_type}")
            return None
        except Exception as e:
            logger.error(f"Failed to download model for {user_id}: {e}")
            return None

    async def download_global_model(self, model_type: str) -> Optional[bytes]:
        """Download the global shared model."""
        try:
            blob_path = self._get_global_model_path(model_type)
            blob_client = self.container_client.get_blob_client(blob_path)

            download = blob_client.download_blob()
            return download.readall()

        except ResourceNotFoundError:
            logger.debug(f"No global model found: {model_type}")
            return None
        except Exception as e:
            logger.error(f"Failed to download global model {model_type}: {e}")
            return None

    async def upload_global_model(
        self,
        model_type: str,
        model_data: bytes,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Upload a global shared model."""
        try:
            blob_path = self._get_global_model_path(model_type)
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(model_data, overwrite=True)

            # Save metadata
            metadata = {
                "model_type": model_type,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "blob_path": blob_path,
                "size_bytes": len(model_data)
            }

            metadata_path = f"models/global/{model_type}/metadata.json"
            metadata_blob = self.container_client.get_blob_client(metadata_path)
            metadata_blob.upload_blob(
                json.dumps(metadata, indent=2),
                overwrite=True,
                content_settings={"content_type": "application/json"}
            )

            logger.info(f"Uploaded global model: {model_type}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to upload global model {model_type}: {e}")
            raise

    async def get_user_model_metadata(
        self,
        user_id: str,
        model_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a user's model."""
        try:
            metadata_path = self._get_metadata_path(user_id, model_type)
            blob_client = self.container_client.get_blob_client(metadata_path)

            download = blob_client.download_blob()
            return json.loads(download.readall())

        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for {user_id}: {e}")
            return None

    async def _get_next_version(self, user_id: str, model_type: str) -> int:
        """Get next version number for a user's model."""
        metadata = await self.get_user_model_metadata(user_id, model_type)
        if metadata:
            return metadata.get("version", 0) + 1
        return 1

    async def list_user_models(self, user_id: str) -> List[Dict[str, Any]]:
        """List all models for a user."""
        models = []
        prefix = f"models/users/{user_id}/"

        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)

            # Group by model type
            model_types = set()
            for blob in blobs:
                parts = blob.name.split("/")
                if len(parts) >= 4:
                    model_types.add(parts[3])  # model_type

            # Get metadata for each model type
            for model_type in model_types:
                metadata = await self.get_user_model_metadata(user_id, model_type)
                if metadata:
                    models.append(metadata)

            return models

        except Exception as e:
            logger.error(f"Failed to list models for {user_id}: {e}")
            return []

    async def delete_user_model(self, user_id: str, model_type: str) -> bool:
        """Delete a user's model and all versions."""
        try:
            prefix = f"models/users/{user_id}/{model_type}/"
            blobs = self.container_client.list_blobs(name_starts_with=prefix)

            deleted = 0
            for blob in blobs:
                self.container_client.delete_blob(blob.name)
                deleted += 1

            logger.info(f"Deleted {deleted} blobs for user {user_id} model {model_type}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Failed to delete model for {user_id}: {e}")
            return False

    async def has_personalized_model(self, user_id: str, model_type: str) -> bool:
        """Check if user has a personalized model."""
        metadata = await self.get_user_model_metadata(user_id, model_type)
        return metadata is not None

    async def download_to_file(
        self,
        user_id: str,
        model_type: str,
        local_path: Path
    ) -> bool:
        """Download model to a local file."""
        model_data = await self.download_user_model(user_id, model_type)
        if model_data:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(model_data)
            return True
        return False

    async def upload_from_file(
        self,
        user_id: str,
        model_type: str,
        local_path: Path,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload model from a local file."""
        model_data = local_path.read_bytes()
        return await self.upload_user_model(
            user_id=user_id,
            model_type=model_type,
            model_data=model_data,
            metrics=metrics,
            config=config
        )


# Singleton instance
_model_store: Optional[BlobStorageModelStore] = None


def get_model_store() -> BlobStorageModelStore:
    """Get or create the model store singleton."""
    global _model_store
    if _model_store is None:
        _model_store = BlobStorageModelStore()
    return _model_store
