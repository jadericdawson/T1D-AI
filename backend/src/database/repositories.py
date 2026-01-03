"""
CosmosDB Repositories for T1D-AI
CRUD operations for all data models.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from azure.cosmos import exceptions
from azure.cosmos.container import ContainerProxy

from database.cosmos_client import get_cosmos_manager
from models.schemas import (
    User, GlucoseReading, Treatment, DataSource, AIInsight,
    GlucoseWithPredictions, GlucosePrediction
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, container_name: str, partition_key_field: str = "userId"):
        self.manager = get_cosmos_manager()
        self.container_name = container_name
        self.partition_key_field = partition_key_field

    @property
    def container(self) -> ContainerProxy:
        return self.manager.get_container(self.container_name)

    def _get_partition_key(self, item: dict) -> str:
        return item.get(self.partition_key_field, item.get("id"))


class UserRepository(BaseRepository):
    """Repository for user accounts."""

    def __init__(self):
        super().__init__("users", "id")

    async def create(self, email: str, display_name: Optional[str] = None) -> User:
        """Create a new user."""
        import uuid
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            displayName=display_name
        )
        try:
            result = self.container.create_item(body=user.model_dump(mode='json'))
            logger.info(f"Created user: {user.id}")
            return User(**result)
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"User already exists: {user.id}")
            raise ValueError(f"User {user.id} already exists")

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            result = self.container.read_item(item=user_id, partition_key=user_id)
            return User(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.email = @email
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@email", "value": email}],
            enable_cross_partition_query=True
        ))
        return User(**items[0]) if items else None

    async def update(self, user_id: str, updates: dict) -> User:
        """Update an existing user."""
        user = await self.get_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        user_data = user.model_dump()
        user_data.update(updates)
        result = self.container.upsert_item(body=user_data)
        logger.info(f"Updated user: {user_id}")
        return User(**result)

    async def delete(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            self.container.delete_item(item=user_id, partition_key=user_id)
            logger.info(f"Deleted user: {user_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False


class GlucoseRepository(BaseRepository):
    """Repository for glucose readings."""

    def __init__(self):
        super().__init__("glucose_readings", "userId")

    async def create(self, reading: GlucoseReading) -> GlucoseReading:
        """Create a new glucose reading."""
        result = self.container.upsert_item(body=reading.model_dump(mode='json'))
        return GlucoseReading(**result)

    async def create_many(self, readings: List[GlucoseReading]) -> int:
        """Bulk create glucose readings."""
        count = 0
        for reading in readings:
            try:
                self.container.upsert_item(body=reading.model_dump(mode='json'))
                count += 1
            except Exception as e:
                logger.error(f"Failed to insert reading {reading.id}: {e}")
        logger.info(f"Inserted {count} glucose readings")
        return count

    async def get_latest(self, user_id: str) -> Optional[GlucoseReading]:
        """Get the most recent glucose reading for a user."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.userId = @userId
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))
        return GlucoseReading(**items[0]) if items else None

    async def get_history(
        self,
        user_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[GlucoseReading]:
        """Get glucose readings within a time range."""
        end_time = end_time or datetime.utcnow()

        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.userId = @userId
              AND c.timestamp >= @startTime
              AND c.timestamp <= @endTime
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@startTime", "value": start_time.isoformat()},
                {"name": "@endTime", "value": end_time.isoformat()},
                {"name": "@limit", "value": limit}
            ],
            partition_key=user_id
        ))
        return [GlucoseReading(**item) for item in items]

    async def get_for_calculation(
        self,
        user_id: str,
        hours: int = 24
    ) -> List[GlucoseReading]:
        """Get readings for IOB/COB/feature calculations."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        return await self.get_history(user_id, start_time, limit=2000)


class TreatmentRepository(BaseRepository):
    """Repository for treatments (insulin, carbs)."""

    def __init__(self):
        super().__init__("treatments", "userId")

    async def create(self, treatment: Treatment) -> Treatment:
        """Create a new treatment."""
        result = self.container.upsert_item(body=treatment.model_dump(mode='json'))
        return Treatment(**result)

    async def create_many(self, treatments: List[Treatment]) -> int:
        """Bulk create treatments."""
        count = 0
        for treatment in treatments:
            try:
                self.container.upsert_item(body=treatment.model_dump(mode='json'))
                count += 1
            except Exception as e:
                logger.error(f"Failed to insert treatment {treatment.id}: {e}")
        logger.info(f"Inserted {count} treatments")
        return count

    async def get_recent(
        self,
        user_id: str,
        hours: int = 6,
        treatment_type: Optional[str] = None
    ) -> List[Treatment]:
        """Get recent treatments for a user."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        if treatment_type:
            query = """
                SELECT *
                FROM c
                WHERE c.userId = @userId
                  AND c.timestamp >= @startTime
                  AND c.type = @type
                ORDER BY c.timestamp DESC
            """
            params = [
                {"name": "@userId", "value": user_id},
                {"name": "@startTime", "value": start_time.isoformat()},
                {"name": "@type", "value": treatment_type}
            ]
        else:
            query = """
                SELECT *
                FROM c
                WHERE c.userId = @userId
                  AND c.timestamp >= @startTime
                ORDER BY c.timestamp DESC
            """
            params = [
                {"name": "@userId", "value": user_id},
                {"name": "@startTime", "value": start_time.isoformat()}
            ]

        items = list(self.container.query_items(
            query=query,
            parameters=params,
            partition_key=user_id
        ))
        return [Treatment(**item) for item in items]

    async def get_for_iob_calculation(self, user_id: str, duration_minutes: int = 180) -> List[Treatment]:
        """Get insulin treatments for IOB calculation."""
        start_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        return await self.get_recent(user_id, hours=duration_minutes // 60 + 1, treatment_type="insulin")

    async def get_for_cob_calculation(self, user_id: str, duration_minutes: int = 180) -> List[Treatment]:
        """Get carb treatments for COB calculation."""
        start_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        return await self.get_recent(user_id, hours=duration_minutes // 60 + 1, treatment_type="carbs")

    async def get_by_user(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Treatment]:
        """Get treatments for a user within a time range."""
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=24))

        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.userId = @userId
              AND c.timestamp >= @startTime
              AND c.timestamp <= @endTime
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@startTime", "value": start_time.isoformat()},
                {"name": "@endTime", "value": end_time.isoformat()},
                {"name": "@limit", "value": limit}
            ],
            partition_key=user_id
        ))
        return [Treatment(**item) for item in items]


class DataSourceRepository(BaseRepository):
    """Repository for data source configurations."""

    def __init__(self):
        super().__init__("datasources", "userId")

    async def create(self, datasource: DataSource) -> DataSource:
        """Create or update a data source."""
        result = self.container.upsert_item(body=datasource.model_dump(mode='json'))
        return DataSource(**result)

    async def get(self, user_id: str, source_type: str = "gluroo") -> Optional[DataSource]:
        """Get a user's data source configuration."""
        try:
            doc_id = f"{user_id}_{source_type}"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return DataSource(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def delete(self, user_id: str, source_type: str = "gluroo") -> bool:
        """Delete a data source configuration."""
        try:
            doc_id = f"{user_id}_{source_type}"
            self.container.delete_item(item=doc_id, partition_key=user_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False


class InsightRepository(BaseRepository):
    """Repository for AI-generated insights."""

    def __init__(self):
        super().__init__("insights", "userId")

    async def create(self, insight: AIInsight) -> AIInsight:
        """Create a new insight."""
        result = self.container.upsert_item(body=insight.model_dump(mode='json'))
        return AIInsight(**result)

    async def get_by_user(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[AIInsight]:
        """Get insights for a user with optional category filter."""
        if category:
            query = """
                SELECT *
                FROM c
                WHERE c.userId = @userId AND c.category = @category
                ORDER BY c.createdAt DESC
                OFFSET @offset LIMIT @limit
            """
            params = [
                {"name": "@userId", "value": user_id},
                {"name": "@category", "value": category},
                {"name": "@offset", "value": offset},
                {"name": "@limit", "value": limit}
            ]
        else:
            query = """
                SELECT *
                FROM c
                WHERE c.userId = @userId
                ORDER BY c.createdAt DESC
                OFFSET @offset LIMIT @limit
            """
            params = [
                {"name": "@userId", "value": user_id},
                {"name": "@offset", "value": offset},
                {"name": "@limit", "value": limit}
            ]

        items = list(self.container.query_items(
            query=query,
            parameters=params,
            partition_key=user_id
        ))
        return [AIInsight(**item) for item in items]

    async def get_recent(self, user_id: str, limit: int = 10) -> List[AIInsight]:
        """Get recent insights for a user."""
        return await self.get_by_user(user_id, limit=limit)

    async def delete_expired(self, user_id: str) -> int:
        """Delete expired insights."""
        now = datetime.utcnow()
        query = """
            SELECT c.id
            FROM c
            WHERE c.userId = @userId AND c.expiresAt < @now
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@now", "value": now.isoformat()}
            ],
            partition_key=user_id
        ))
        count = 0
        for item in items:
            try:
                self.container.delete_item(item=item["id"], partition_key=user_id)
                count += 1
            except Exception:
                pass
        return count
