"""
CosmosDB Client for T1D-AI
Manages connection to Azure CosmosDB Serverless
"""
import logging
from typing import Optional
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy

from config import get_settings

logger = logging.getLogger(__name__)


class CosmosDBManager:
    """Manages CosmosDB database and containers for T1D-AI."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[CosmosClient] = None
        self._database: Optional[DatabaseProxy] = None
        self._containers: dict[str, ContainerProxy] = {}

    @property
    def client(self) -> CosmosClient:
        """Lazy initialization of CosmosDB client."""
        if self._client is None:
            self._client = CosmosClient(
                url=self.settings.cosmos_endpoint,
                credential=self.settings.cosmos_key
            )
            logger.info("CosmosDB client initialized")
        return self._client

    @property
    def database(self) -> DatabaseProxy:
        """Get or create the T1D-AI database."""
        if self._database is None:
            try:
                self._database = self.client.create_database_if_not_exists(
                    id=self.settings.cosmos_database
                )
                logger.info(f"Database '{self.settings.cosmos_database}' ready")
            except exceptions.CosmosHttpResponseError as e:
                logger.error(f"Failed to create database: {e}")
                raise
        return self._database

    def get_container(self, container_name: str, partition_key: str = "/userId") -> ContainerProxy:
        """Get or create a container with the specified partition key."""
        if container_name not in self._containers:
            try:
                self._containers[container_name] = self.database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path=partition_key),
                    offer_throughput=None  # Serverless mode
                )
                logger.info(f"Container '{container_name}' ready")
            except exceptions.CosmosHttpResponseError as e:
                logger.error(f"Failed to create container '{container_name}': {e}")
                raise
        return self._containers[container_name]

    async def initialize_containers(self):
        """Initialize all required containers for T1D-AI."""
        containers = [
            ("users", "/id"),
            ("glucose_readings", "/userId"),
            ("treatments", "/userId"),
            ("predictions", "/userId"),
            ("insights", "/userId"),
            ("datasources", "/userId"),
        ]

        for container_name, partition_key in containers:
            self.get_container(container_name, partition_key)

        logger.info("All containers initialized")

    def close(self):
        """Close the CosmosDB client connection."""
        if self._client:
            # CosmosClient doesn't have explicit close, but we clear references
            self._client = None
            self._database = None
            self._containers.clear()
            logger.info("CosmosDB client closed")


# Singleton instance
_cosmos_manager: Optional[CosmosDBManager] = None


def get_cosmos_manager() -> CosmosDBManager:
    """Get the singleton CosmosDB manager instance."""
    global _cosmos_manager
    if _cosmos_manager is None:
        _cosmos_manager = CosmosDBManager()
    return _cosmos_manager
