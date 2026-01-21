"""
Azure Container Instances Training Service

Triggers TFT model training in an Azure Container Instance.
Training runs in a separate container with more resources than the web app.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup, Container, ContainerGroupNetworkProtocol,
    ResourceRequests, ResourceRequirements, EnvironmentVariable,
    OperatingSystemTypes, ContainerGroupRestartPolicy,
    GpuResource
)

logger = logging.getLogger(__name__)

# Azure configuration
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-knowledge2ai-eastus")
ACR_SERVER = os.getenv("ACR_SERVER", "knowledge2aiacr.azurecr.io")
ACR_USERNAME = os.getenv("ACR_USERNAME", "knowledge2aiacr")
ACR_PASSWORD = os.getenv("ACR_PASSWORD", "")

# Container configuration
TRAINING_IMAGE = f"{ACR_SERVER}/t1d-ai:latest"

# CPU configuration (default, always available)
TRAINING_CPU = 4.0  # 4 CPU cores
TRAINING_MEMORY = 16.0  # 16 GB RAM

# GPU configuration (faster but more expensive, ~$3/hr vs ~$0.05/hr)
# NCv3 series with NVIDIA Tesla V100
GPU_SKU = "V100"  # K80, P100, or V100
GPU_COUNT = 1


class ACITrainingService:
    """Service to trigger model training on Azure Container Instances."""

    def __init__(self):
        """Initialize the ACI client."""
        self._client = None

    @property
    def client(self) -> ContainerInstanceManagementClient:
        """Lazy initialization of ACI client."""
        if self._client is None:
            if not SUBSCRIPTION_ID:
                raise ValueError("AZURE_SUBSCRIPTION_ID environment variable required")
            credential = DefaultAzureCredential()
            self._client = ContainerInstanceManagementClient(credential, SUBSCRIPTION_ID)
        return self._client

    async def start_training(
        self,
        user_id: str,
        model_type: str = "tft",
        job_id: Optional[str] = None,
        use_gpu: bool = True
    ) -> dict:
        """
        Start a training container on ACI.

        Tries GPU first for faster training, falls back to CPU if GPU unavailable.

        Args:
            user_id: User ID to train model for
            model_type: Type of model to train
            job_id: Optional job ID for tracking
            use_gpu: Try GPU first (default True, ~5-10x faster but ~$3/hr vs ~$0.05/hr)

        Returns:
            dict with container group info
        """
        # Try GPU first, then fallback to CPU
        if use_gpu:
            try:
                return await self._start_container(
                    user_id=user_id,
                    model_type=model_type,
                    job_id=job_id,
                    use_gpu=True
                )
            except Exception as gpu_error:
                logger.warning(f"GPU container failed, falling back to CPU: {gpu_error}")
                # Fall through to CPU

        return await self._start_container(
            user_id=user_id,
            model_type=model_type,
            job_id=job_id,
            use_gpu=False
        )

    async def _start_container(
        self,
        user_id: str,
        model_type: str,
        job_id: Optional[str],
        use_gpu: bool
    ) -> dict:
        """Start a container with specific GPU/CPU configuration."""
        # Generate unique container group name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        gpu_suffix = "-gpu" if use_gpu else "-cpu"
        container_group_name = f"t1d-train-{user_id[:8]}-{timestamp}{gpu_suffix}"

        logger.info(f"Starting training container: {container_group_name} (GPU: {use_gpu})")

        # Environment variables for training script
        # Need all required Settings fields even if not used by training
        # Note: Using 'value' for non-secrets and 'secure_value' for secrets
        cosmos_endpoint = os.getenv("COSMOS_ENDPOINT", "")
        cosmos_key = os.getenv("COSMOS_KEY", "")
        storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("STORAGE_CONNECTION_STRING", "")
        openai_endpoint = os.getenv("GPT41_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", "https://placeholder.openai.azure.com")
        openai_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("GPT41_API_KEY", "placeholder-key")
        jwt_key = os.getenv("JWT_SECRET_KEY", "training-placeholder-key")

        env_vars = [
            EnvironmentVariable(name="USER_ID", value=user_id),
            EnvironmentVariable(name="MODEL_TYPE", value=model_type),
            # CosmosDB (required for data loading) - endpoint is public, key is secret
            EnvironmentVariable(name="COSMOS_ENDPOINT", value=cosmos_endpoint),
            EnvironmentVariable(name="COSMOS_KEY", secure_value=cosmos_key),
            EnvironmentVariable(name="COSMOS_DATABASE", value=os.getenv("COSMOS_DATABASE", "T1D-AI-DB")),
            # Blob Storage (required for model upload)
            EnvironmentVariable(name="STORAGE_ACCOUNT_URL", value=os.getenv("STORAGE_ACCOUNT_URL", "")),
            EnvironmentVariable(name="STORAGE_CONNECTION_STRING", secure_value=storage_conn),
            # Azure OpenAI (required by Settings, not used in training)
            EnvironmentVariable(name="GPT41_ENDPOINT", value=openai_endpoint),
            EnvironmentVariable(name="AZURE_OPENAI_KEY", secure_value=openai_key),
            # JWT (required by Settings, not used in training)
            EnvironmentVariable(name="JWT_SECRET_KEY", secure_value=jwt_key),
        ]

        if job_id:
            env_vars.append(EnvironmentVariable(name="JOB_ID", value=job_id))

        # Container resource requirements
        if use_gpu:
            # GPU configuration - V100 with 4 CPUs and 16GB RAM
            resources = ResourceRequirements(
                requests=ResourceRequests(
                    cpu=TRAINING_CPU,
                    memory_in_gb=TRAINING_MEMORY,
                    gpu=GpuResource(count=GPU_COUNT, sku=GPU_SKU)
                )
            )
            # GPU containers only available in specific regions
            location = "eastus"  # NCv3 available in eastus, eastus2, westus2, westeurope
        else:
            # CPU-only configuration
            resources = ResourceRequirements(
                requests=ResourceRequests(
                    cpu=TRAINING_CPU,
                    memory_in_gb=TRAINING_MEMORY
                )
            )
            location = "eastus"

        # Container definition
        container = Container(
            name="trainer",
            image=TRAINING_IMAGE,
            resources=resources,
            environment_variables=env_vars,
            command=["python", "scripts/azure_train.py"]
        )

        # Container group definition
        container_group = ContainerGroup(
            location=location,
            containers=[container],
            os_type=OperatingSystemTypes.LINUX,
            restart_policy=ContainerGroupRestartPolicy.NEVER,
            image_registry_credentials=[{
                "server": ACR_SERVER,
                "username": ACR_USERNAME,
                "password": ACR_PASSWORD
            }]
        )

        # Create container group
        try:
            result = self.client.container_groups.begin_create_or_update(
                RESOURCE_GROUP,
                container_group_name,
                container_group
            )

            logger.info(f"Training container started: {container_group_name}")

            return {
                "container_group_name": container_group_name,
                "status": "starting",
                "user_id": user_id,
                "model_type": model_type,
                "job_id": job_id,
                "use_gpu": use_gpu,
                "location": location
            }

        except Exception as e:
            logger.error(f"Failed to start training container: {e}")
            raise

    async def get_training_status(self, container_group_name: str) -> dict:
        """Get the status of a training container."""
        try:
            container_group = self.client.container_groups.get(
                RESOURCE_GROUP,
                container_group_name
            )

            container = container_group.containers[0]
            instance_view = container.instance_view

            return {
                "container_group_name": container_group_name,
                "state": instance_view.current_state.state if instance_view else "Unknown",
                "start_time": instance_view.current_state.start_time if instance_view else None,
                "finish_time": instance_view.current_state.finish_time if instance_view else None,
                "exit_code": instance_view.current_state.exit_code if instance_view else None,
                "provisioning_state": container_group.provisioning_state
            }

        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {
                "container_group_name": container_group_name,
                "state": "Error",
                "error": str(e)
            }

    async def cleanup_container(self, container_group_name: str) -> bool:
        """Delete a completed training container."""
        try:
            self.client.container_groups.begin_delete(
                RESOURCE_GROUP,
                container_group_name
            )
            logger.info(f"Deleted container group: {container_group_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete container group: {e}")
            return False


# Singleton instance
_aci_service: Optional[ACITrainingService] = None


def get_aci_training_service() -> ACITrainingService:
    """Get the ACI training service singleton."""
    global _aci_service
    if _aci_service is None:
        _aci_service = ACITrainingService()
    return _aci_service
