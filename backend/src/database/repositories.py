"""
CosmosDB Repositories for T1D-AI
CRUD operations for all data models.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from azure.cosmos import exceptions
from azure.cosmos.container import ContainerProxy

from database.cosmos_client import get_cosmos_manager
from models.schemas import (
    User, GlucoseReading, Treatment, DataSource, AIInsight,
    GlucoseWithPredictions, GlucosePrediction, LearnedISF,
    AccountShare, ShareInvitation, UserModel, TrainingJob,
    MLTrainingDataPoint, FoodAbsorptionProfile,
    LearnedICR, ICRDataPoint, LearnedPIR, PIRDataPoint,
    UserAbsorptionProfile, AbsorptionCurveDataPoint,
    ManagedProfile, ProfileDataSource, ProfileSummary, SyncStatus
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

    async def create(
        self,
        email: str,
        display_name: Optional[str] = None,
        password_hash: Optional[str] = None,
        auth_provider: str = "email",
        microsoft_id: Optional[str] = None,
        account_type: str = "personal",
        parent_id: Optional[str] = None,
        guardian_email: Optional[str] = None,
        date_of_birth: Optional[datetime] = None,
        diagnosis_date: Optional[datetime] = None,
        has_t1d: bool = True,
        # Email verification fields
        email_verified: bool = False,
        email_verification_token: Optional[str] = None,
        email_verification_expires: Optional[datetime] = None
    ) -> User:
        """Create a new user.

        Args:
            email: User's email address
            display_name: Display name
            password_hash: Hashed password for email/password auth
            auth_provider: "email", "microsoft", or "google"
            microsoft_id: Microsoft account ID for OAuth
            account_type: "personal", "parent", or "child"
            parent_id: Parent user ID for child accounts
            guardian_email: Guardian email for child accounts
            date_of_birth: Date of birth for age-appropriate features
            diagnosis_date: Date of T1D diagnosis
            has_t1d: Whether this user has T1D themselves
            email_verified: Whether email has been verified
            email_verification_token: Token for email verification
            email_verification_expires: When verification token expires
        """
        import uuid
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            displayName=display_name,
            passwordHash=password_hash,
            authProvider=auth_provider,
            microsoftId=microsoft_id,
            accountType=account_type,
            parentId=parent_id,
            guardianEmail=guardian_email,
            dateOfBirth=date_of_birth,
            diagnosisDate=diagnosis_date,
            hasT1D=has_t1d,
            emailVerified=email_verified,
            emailVerificationToken=email_verification_token,
            emailVerificationExpires=email_verification_expires
        )
        try:
            result = self.container.create_item(body=user.model_dump(mode='json'))
            logger.info(f"Created user: {user.id} ({account_type})")
            return User(**result)
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"User already exists: {user.id}")
            raise ValueError(f"User {user.id} already exists")

    async def link_child_to_parent(self, parent_id: str, child_id: str) -> User:
        """Link a child account to a parent account."""
        parent = await self.get_by_id(parent_id)
        if not parent:
            raise ValueError(f"Parent user {parent_id} not found")

        # Add child to parent's linkedChildIds
        linked_children = parent.linkedChildIds or []
        if child_id not in linked_children:
            linked_children.append(child_id)
            parent = await self.update(parent_id, {"linkedChildIds": linked_children})

        # Update child's parentId
        await self.update(child_id, {"parentId": parent_id})

        return parent

    async def get_children(self, parent_id: str) -> List[User]:
        """Get all children linked to a parent account."""
        parent = await self.get_by_id(parent_id)
        if not parent or not parent.linkedChildIds:
            return []

        children = []
        for child_id in parent.linkedChildIds:
            child = await self.get_by_id(child_id)
            if child:
                children.append(child)
        return children

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

    async def get_by_verification_token(self, token: str) -> Optional[User]:
        """Get a user by email verification token."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.emailVerificationToken = @token
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@token", "value": token}],
            enable_cross_partition_query=True
        ))
        return User(**items[0]) if items else None

    async def update(self, user_id: str, updates: dict) -> User:
        """Update an existing user."""
        user = await self.get_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        user_data = user.model_dump(mode='json')
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

    async def list_all_users(
        self,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None
    ) -> List[User]:
        """List all users with pagination and optional search.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Optional search string (searches email and displayName)
        """
        if search:
            query = """
                SELECT *
                FROM c
                WHERE CONTAINS(LOWER(c.email), @search) OR CONTAINS(LOWER(c.displayName), @search)
                ORDER BY c.createdAt DESC
                OFFSET @skip LIMIT @limit
            """
            params = [
                {"name": "@search", "value": search.lower()},
                {"name": "@skip", "value": skip},
                {"name": "@limit", "value": limit}
            ]
        else:
            query = """
                SELECT *
                FROM c
                ORDER BY c.createdAt DESC
                OFFSET @skip LIMIT @limit
            """
            params = [
                {"name": "@skip", "value": skip},
                {"name": "@limit", "value": limit}
            ]

        items = list(self.container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        return [User(**item) for item in items]

    async def count_all_users(self) -> int:
        """Get total count of all users."""
        query = "SELECT VALUE COUNT(1) FROM c"
        results = list(self.container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        return results[0] if results else 0


class GlucoseRepository(BaseRepository):
    """Repository for glucose readings."""

    def __init__(self):
        super().__init__("glucose_readings", "userId")

    async def create(self, reading: GlucoseReading) -> GlucoseReading:
        """Create a new glucose reading."""
        result = self.container.upsert_item(body=reading.model_dump(mode='json'))
        return GlucoseReading(**result)

    async def upsert(self, reading: GlucoseReading) -> GlucoseReading:
        """Create or update a glucose reading."""
        return await self.create(reading)

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

    async def get_recent_by_source(
        self,
        user_id: str,
        source: str,
        hours: int = 24
    ) -> List[GlucoseReading]:
        """
        Get recent glucose readings from a specific source (e.g., 'dexcom', 'gluroo').
        Used to check if a user has data from a particular source configured.
        """
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = """
            SELECT TOP 10 *
            FROM c
            WHERE c.userId = @userId
              AND c.source = @source
              AND c.timestamp >= @startTime
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@source", "value": source},
                {"name": "@startTime", "value": start_time.isoformat()}
            ],
            partition_key=user_id
        ))
        return [GlucoseReading(**item) for item in items]

    async def get_history(
        self,
        user_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[GlucoseReading]:
        """Get glucose readings within a time range."""
        end_time = end_time or datetime.now(timezone.utc)

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
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return await self.get_history(user_id, start_time, limit=2000)

    async def count_for_user(self, user_id: str) -> int:
        """Count total glucose readings for a user."""
        query = """
            SELECT VALUE COUNT(1)
            FROM c
            WHERE c.userId = @userId
        """
        results = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))
        return results[0] if results else 0

    async def get_date_range(self, user_id: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the oldest and newest reading timestamps for a user.

        Returns:
            Tuple of (oldest_timestamp, newest_timestamp), or (None, None) if no readings
        """
        # Get oldest
        oldest_query = """
            SELECT TOP 1 c.timestamp
            FROM c
            WHERE c.userId = @userId
            ORDER BY c.timestamp ASC
        """
        oldest_items = list(self.container.query_items(
            query=oldest_query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))

        # Get newest
        newest_query = """
            SELECT TOP 1 c.timestamp
            FROM c
            WHERE c.userId = @userId
            ORDER BY c.timestamp DESC
        """
        newest_items = list(self.container.query_items(
            query=newest_query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))

        oldest = None
        newest = None

        if oldest_items:
            oldest_ts = oldest_items[0].get("timestamp")
            if oldest_ts:
                oldest = datetime.fromisoformat(oldest_ts.replace("Z", "+00:00")) if isinstance(oldest_ts, str) else oldest_ts

        if newest_items:
            newest_ts = newest_items[0].get("timestamp")
            if newest_ts:
                newest = datetime.fromisoformat(newest_ts.replace("Z", "+00:00")) if isinstance(newest_ts, str) else newest_ts

        return oldest, newest


class TreatmentRepository(BaseRepository):
    """Repository for treatments (insulin, carbs)."""

    def __init__(self):
        super().__init__("treatments", "userId")

    async def create(self, treatment: Treatment) -> Treatment:
        """Create a new treatment."""
        result = self.container.upsert_item(body=treatment.model_dump(mode='json'))
        return Treatment(**result)

    async def upsert(self, treatment: Treatment) -> Treatment:
        """Create or update a treatment."""
        return await self.create(treatment)

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
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

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
        start_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        return await self.get_recent(user_id, hours=duration_minutes // 60 + 1, treatment_type="insulin")

    async def get_for_cob_calculation(self, user_id: str, duration_minutes: int = 180) -> List[Treatment]:
        """Get carb treatments for COB calculation."""
        start_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        return await self.get_recent(user_id, hours=duration_minutes // 60 + 1, treatment_type="carbs")

    async def get_by_user(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Treatment]:
        """Get treatments for a user within a time range."""
        end_time = end_time or datetime.now(timezone.utc)
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

    async def get_by_id(self, treatment_id: str, user_id: str) -> Optional[Treatment]:
        """Get a specific treatment by ID."""
        try:
            result = self.container.read_item(item=treatment_id, partition_key=user_id)
            return Treatment(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def delete(self, treatment_id: str, user_id: str) -> bool:
        """Delete a treatment."""
        try:
            self.container.delete_item(item=treatment_id, partition_key=user_id)
            logger.info(f"Deleted treatment {treatment_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    async def count_for_user(self, user_id: str) -> int:
        """Count total treatments for a user."""
        query = """
            SELECT VALUE COUNT(1)
            FROM c
            WHERE c.userId = @userId
        """
        results = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))
        return results[0] if results else 0


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

    async def update(self, datasource: DataSource) -> DataSource:
        """Update an existing data source."""
        result = self.container.upsert_item(body=datasource.model_dump(mode='json'))
        return DataSource(**result)

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
        now = datetime.now(timezone.utc)
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


class LearnedISFRepository(BaseRepository):
    """Repository for learned ISF values."""

    def __init__(self):
        super().__init__("learned_isf", "userId")

    async def get(self, user_id: str, isf_type: str) -> Optional[LearnedISF]:
        """Get learned ISF for a user and type (fasting or meal)."""
        try:
            doc_id = f"{user_id}_{isf_type}"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return LearnedISF(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_both(self, user_id: str) -> dict:
        """Get both fasting and meal ISF for a user."""
        fasting = await self.get(user_id, "fasting")
        meal = await self.get(user_id, "meal")
        return {
            "fasting": fasting,
            "meal": meal
        }

    async def upsert(self, learned_isf: LearnedISF) -> LearnedISF:
        """Create or update learned ISF."""
        result = self.container.upsert_item(body=learned_isf.model_dump(mode='json'))
        return LearnedISF(**result)

    async def update_isf(
        self,
        user_id: str,
        isf_type: str,
        new_value: float,
        data_point: dict,
        confidence: float = 0.5
    ) -> LearnedISF:
        """Update ISF with a new observation."""
        existing = await self.get(user_id, isf_type)

        if existing:
            # Add to history (keep last 30)
            history = existing.history[-29:] if existing.history else []
            from models.schemas import ISFDataPoint
            history.append(ISFDataPoint(**data_point))

            # Calculate weighted average (more recent = higher weight)
            if history:
                import numpy as np
                values = [dp.value for dp in history]
                weights = np.exp(np.linspace(-1, 0, len(values)))  # Exponential decay
                weighted_avg = np.average(values, weights=weights)
                new_value = weighted_avg

                # Calculate stats
                existing.meanISF = float(np.mean(values))
                existing.stdISF = float(np.std(values)) if len(values) > 1 else 0.0
                existing.minISF = float(min(values))
                existing.maxISF = float(max(values))

            existing.value = new_value
            existing.history = history
            existing.sampleCount = len(history)
            existing.confidence = min(1.0, len(history) / 10)  # Full confidence at 10+ samples
            existing.lastUpdated = datetime.now(timezone.utc)

            return await self.upsert(existing)
        else:
            # Create new record
            from models.schemas import ISFDataPoint
            new_isf = LearnedISF(
                id=f"{user_id}_{isf_type}",
                userId=user_id,
                isfType=isf_type,
                value=new_value,
                confidence=confidence,
                sampleCount=1,
                history=[ISFDataPoint(**data_point)],
                meanISF=new_value,
                stdISF=0.0,
                minISF=new_value,
                maxISF=new_value
            )
            return await self.upsert(new_isf)

    async def delete(self, user_id: str, isf_type: str) -> bool:
        """Delete a learned ISF record."""
        try:
            doc_id = f"{user_id}_{isf_type}"
            self.container.delete_item(item=doc_id, partition_key=user_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False


class LearnedICRRepository(BaseRepository):
    """Repository for learned ICR (Insulin-to-Carb Ratio) values."""

    def __init__(self):
        super().__init__("learned_icr", "userId")

    async def get(self, user_id: str, meal_type: str = "overall") -> Optional[LearnedICR]:
        """Get learned ICR for a user and meal type (breakfast, lunch, dinner, overall)."""
        try:
            doc_id = f"{user_id}_{meal_type}"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return LearnedICR(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_all(self, user_id: str) -> dict:
        """Get all learned ICR values for a user (breakfast, lunch, dinner, overall)."""
        result = {}
        for meal_type in ["overall", "breakfast", "lunch", "dinner"]:
            icr = await self.get(user_id, meal_type)
            if icr:
                result[meal_type] = icr
        return result

    async def upsert(self, learned_icr: LearnedICR) -> LearnedICR:
        """Create or update learned ICR."""
        result = self.container.upsert_item(body=learned_icr.model_dump(mode='json'))
        return LearnedICR(**result)

    async def update_icr(
        self,
        user_id: str,
        meal_type: str,
        new_value: float,
        data_point: dict,
        confidence: float = 0.5
    ) -> LearnedICR:
        """Update ICR with a new observation."""
        existing = await self.get(user_id, meal_type)

        if existing:
            # Add to history (keep last 30)
            history = existing.history[-29:] if existing.history else []
            history.append(ICRDataPoint(**data_point))

            # Calculate weighted average (more recent = higher weight)
            if history:
                import numpy as np
                values = [dp.value for dp in history]
                weights = np.exp(np.linspace(-1, 0, len(values)))  # Exponential decay
                weighted_avg = np.average(values, weights=weights)
                new_value = weighted_avg

                # Calculate stats
                existing.meanICR = float(np.mean(values))
                existing.stdICR = float(np.std(values)) if len(values) > 1 else 0.0
                existing.minICR = float(min(values))
                existing.maxICR = float(max(values))

            existing.value = new_value
            existing.history = history
            existing.sampleCount = len(history)
            existing.confidence = min(1.0, len(history) / 10)  # Full confidence at 10+ samples
            existing.lastUpdated = datetime.now(timezone.utc)

            return await self.upsert(existing)
        else:
            # Create new record
            new_icr = LearnedICR(
                id=f"{user_id}_{meal_type}",
                userId=user_id,
                mealType=meal_type,
                value=new_value,
                confidence=confidence,
                sampleCount=1,
                history=[ICRDataPoint(**data_point)],
                meanICR=new_value,
                stdICR=0.0,
                minICR=new_value,
                maxICR=new_value
            )
            return await self.upsert(new_icr)

    async def delete(self, user_id: str, meal_type: str) -> bool:
        """Delete a learned ICR record."""
        try:
            doc_id = f"{user_id}_{meal_type}"
            self.container.delete_item(item=doc_id, partition_key=user_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False


class LearnedPIRRepository(BaseRepository):
    """Repository for learned PIR (Protein-to-Insulin Ratio) values."""

    def __init__(self):
        super().__init__("learned_pir", "userId")

    async def get(self, user_id: str, meal_type: str = "overall") -> Optional[LearnedPIR]:
        """Get learned PIR for a user and meal type."""
        try:
            doc_id = f"{user_id}_{meal_type}"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return LearnedPIR(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_all(self, user_id: str) -> dict:
        """Get all learned PIR values for a user."""
        result = {}
        for meal_type in ["overall", "breakfast", "lunch", "dinner"]:
            pir = await self.get(user_id, meal_type)
            if pir:
                result[meal_type] = pir
        return result

    async def upsert(self, learned_pir: LearnedPIR) -> LearnedPIR:
        """Create or update learned PIR."""
        result = self.container.upsert_item(body=learned_pir.model_dump(mode='json'))
        return LearnedPIR(**result)

    async def update_pir(
        self,
        user_id: str,
        meal_type: str,
        new_value: float,
        data_point: dict,
        confidence: float = 0.5
    ) -> LearnedPIR:
        """Update PIR with a new observation."""
        existing = await self.get(user_id, meal_type)

        if existing:
            # Add to history (keep last 30)
            history = existing.history[-29:] if existing.history else []
            history.append(PIRDataPoint(**data_point))

            # Calculate weighted average (more recent = higher weight)
            if history:
                import numpy as np
                values = [dp.value for dp in history]
                weights = np.exp(np.linspace(-1, 0, len(values)))  # Exponential decay
                weighted_avg = np.average(values, weights=weights)
                new_value = weighted_avg

                # Calculate stats
                existing.meanPIR = float(np.mean(values))
                existing.stdPIR = float(np.std(values)) if len(values) > 1 else 0.0
                existing.minPIR = float(min(values))
                existing.maxPIR = float(max(values))

                # Update timing stats from data points with timing info
                onset_times = [dp.proteinOnsetMinutes for dp in history if dp.proteinOnsetMinutes]
                peak_times = [dp.proteinPeakMinutes for dp in history if dp.proteinPeakMinutes]
                if onset_times:
                    existing.avgOnsetMinutes = float(np.mean(onset_times))
                if peak_times:
                    existing.avgPeakMinutes = float(np.mean(peak_times))

            existing.value = new_value
            existing.history = history
            existing.sampleCount = len(history)
            existing.confidence = min(1.0, len(history) / 10)  # Full confidence at 10+ samples
            existing.lastUpdated = datetime.now(timezone.utc)

            return await self.upsert(existing)
        else:
            # Create new record
            new_pir = LearnedPIR(
                id=f"{user_id}_{meal_type}",
                userId=user_id,
                mealType=meal_type,
                value=new_value,
                confidence=confidence,
                sampleCount=1,
                history=[PIRDataPoint(**data_point)],
                meanPIR=new_value,
                stdPIR=0.0,
                minPIR=new_value,
                maxPIR=new_value
            )
            return await self.upsert(new_pir)

    async def delete(self, user_id: str, meal_type: str) -> bool:
        """Delete a learned PIR record."""
        try:
            doc_id = f"{user_id}_{meal_type}"
            self.container.delete_item(item=doc_id, partition_key=user_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False


class SharingRepository(BaseRepository):
    """Repository for account sharing."""

    def __init__(self):
        super().__init__("account_shares", "ownerId")

    async def create_share(self, share: AccountShare) -> AccountShare:
        """Create a new share."""
        result = self.container.upsert_item(body=share.model_dump(mode='json'))
        logger.info(f"Created share: {share.ownerId} -> {share.sharedWithId}")
        return AccountShare(**result)

    async def get_share(self, owner_id: str, shared_with_id: str) -> Optional[AccountShare]:
        """Get a specific share between two users."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.ownerId = @ownerId AND c.sharedWithId = @sharedWithId AND c.isActive = true
        """
        # Use cross-partition query because some shares may have been created
        # with partition keys that don't match the ownerId field value
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@ownerId", "value": owner_id},
                {"name": "@sharedWithId", "value": shared_with_id}
            ],
            enable_cross_partition_query=True
        ))
        return AccountShare(**items[0]) if items else None

    async def get_shares_by_owner(self, owner_id: str) -> List[AccountShare]:
        """Get all shares created by a user (who they're sharing with)."""
        query = """
            SELECT *
            FROM c
            WHERE c.ownerId = @ownerId AND c.isActive = true
            ORDER BY c.createdAt DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@ownerId", "value": owner_id}],
            partition_key=owner_id
        ))
        return [AccountShare(**item) for item in items]

    async def get_shares_with_user(self, shared_with_id: str) -> List[AccountShare]:
        """Get all shares where this user has been granted access (who shared with them)."""
        query = """
            SELECT *
            FROM c
            WHERE c.sharedWithId = @sharedWithId AND c.isActive = true
            ORDER BY c.createdAt DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@sharedWithId", "value": shared_with_id}],
            enable_cross_partition_query=True
        ))
        return [AccountShare(**item) for item in items]

    async def get_share_for_profile(self, profile_id: str, shared_with_id: str) -> Optional[AccountShare]:
        """
        Check if a user has been granted access to view a specific profile's data.

        This handles the case where:
        - A parent (ownerId) shares their child's profile (profileId) with someone (sharedWithId)
        - The share record has ownerId=parent, profileId=child, sharedWithId=viewer
        - When viewer tries to access child's data, we need to find this share by profileId

        Args:
            profile_id: The user ID whose data is being accessed (could be the ownerId OR a profileId)
            shared_with_id: The user requesting access

        Returns:
            AccountShare if access is granted, None otherwise
        """
        # First check: direct share (ownerId = profile_id)
        # This handles cases where someone shares their own data directly
        direct_share = await self.get_share(profile_id, shared_with_id)
        if direct_share:
            return direct_share

        # Second check: profile share (profileId = profile_id)
        # This handles cases where a parent shares a child's profile
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.profileId = @profileId AND c.sharedWithId = @sharedWithId AND c.isActive = true
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@profileId", "value": profile_id},
                {"name": "@sharedWithId", "value": shared_with_id}
            ],
            enable_cross_partition_query=True
        ))
        return AccountShare(**items[0]) if items else None

    async def revoke_share(self, share_id: str, owner_id: str) -> bool:
        """Revoke a share (soft delete by setting isActive=False)."""
        try:
            result = self.container.read_item(item=share_id, partition_key=owner_id)
            result['isActive'] = False
            self.container.upsert_item(body=result)
            logger.info(f"Revoked share: {share_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    async def update_permissions(
        self,
        share_id: str,
        owner_id: str,
        permissions: List[str]
    ) -> Optional[AccountShare]:
        """Update permissions for an existing share."""
        try:
            result = self.container.read_item(item=share_id, partition_key=owner_id)
            result['permissions'] = permissions
            updated = self.container.upsert_item(body=result)
            return AccountShare(**updated)
        except exceptions.CosmosResourceNotFoundError:
            return None


class InvitationRepository(BaseRepository):
    """Repository for share invitations."""

    def __init__(self):
        super().__init__("share_invitations", "ownerId")

    async def create(self, invitation: ShareInvitation) -> ShareInvitation:
        """Create a new invitation."""
        result = self.container.upsert_item(body=invitation.model_dump(mode='json'))
        logger.info(f"Created invitation: {invitation.id} for {invitation.inviteeEmail}")
        return ShareInvitation(**result)

    async def get_by_token(self, token: str) -> Optional[ShareInvitation]:
        """Get an invitation by its token (ID)."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.id = @token AND c.isUsed = false
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@token", "value": token}],
            enable_cross_partition_query=True
        ))
        return ShareInvitation(**items[0]) if items else None

    async def get_pending_for_email(self, email: str) -> List[ShareInvitation]:
        """Get all pending invitations for an email address."""
        now = datetime.now(timezone.utc)
        query = """
            SELECT *
            FROM c
            WHERE c.inviteeEmail = @email AND c.isUsed = false AND c.expiresAt > @now
            ORDER BY c.createdAt DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@email", "value": email},
                {"name": "@now", "value": now.isoformat()}
            ],
            enable_cross_partition_query=True
        ))
        return [ShareInvitation(**item) for item in items]

    async def mark_used(self, token: str, owner_id: str) -> bool:
        """Mark an invitation as used."""
        try:
            result = self.container.read_item(item=token, partition_key=owner_id)
            result['isUsed'] = True
            result['acceptedAt'] = datetime.now(timezone.utc).isoformat()
            self.container.upsert_item(body=result)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    async def get_by_owner_and_email(self, owner_id: str, invitee_email: str) -> List[ShareInvitation]:
        """Get pending invitations from owner to specific email."""
        query = """
            SELECT *
            FROM c
            WHERE c.ownerId = @ownerId
                AND c.inviteeEmail = @email
                AND c.isUsed = false
            ORDER BY c.createdAt DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@ownerId", "value": owner_id},
                {"name": "@email", "value": invitee_email.lower()}
            ],
            partition_key=owner_id
        ))
        return [ShareInvitation(**item) for item in items]

    async def delete(self, invitation_id: str, owner_id: str) -> bool:
        """Delete an invitation."""
        try:
            self.container.delete_item(item=invitation_id, partition_key=owner_id)
            logger.info(f"Deleted invitation: {invitation_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Invitation {invitation_id} not found for deletion")
            return False


class UserModelRepository(BaseRepository):
    """Repository for user-specific ML models."""

    def __init__(self):
        super().__init__("user_models", "userId")

    async def get(self, user_id: str, model_type: str) -> Optional[UserModel]:
        """Get a user's model by type."""
        try:
            doc_id = f"{user_id}_{model_type}"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return UserModel(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_all_for_user(self, user_id: str) -> List[UserModel]:
        """Get all models for a user."""
        query = """
            SELECT *
            FROM c
            WHERE c.userId = @userId
            ORDER BY c.modelType
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))
        return [UserModel(**item) for item in items]

    async def upsert(self, model: UserModel) -> UserModel:
        """Create or update a user model."""
        model.updatedAt = datetime.now(timezone.utc)
        result = self.container.upsert_item(body=model.model_dump(mode='json'))
        logger.info(f"Upserted model: {model.userId}/{model.modelType}")
        return UserModel(**result)

    async def update_status(
        self,
        user_id: str,
        model_type: str,
        status: str,
        metrics: Optional[dict] = None,
        error: Optional[str] = None
    ) -> Optional[UserModel]:
        """Update model status."""
        model = await self.get(user_id, model_type)
        if not model:
            return None

        model.status = status
        if metrics:
            model.metrics = metrics
        if error:
            model.lastError = error
        if status == "active":
            model.trainedAt = datetime.now(timezone.utc)

        return await self.upsert(model)


class TrainingJobRepository(BaseRepository):
    """Repository for training jobs."""

    def __init__(self):
        super().__init__("training_jobs", "userId")

    async def create(self, job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        result = self.container.upsert_item(body=job.model_dump(mode='json'))
        logger.info(f"Created training job: {job.id}")
        return TrainingJob(**result)

    async def get_pending(self, limit: int = 100) -> List[TrainingJob]:
        """Get pending training jobs."""
        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.status = 'queued'
            ORDER BY c.createdAt ASC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True
        ))
        return [TrainingJob(**item) for item in items]

    async def update_status(
        self,
        job_id: str,
        user_id: str,
        status: str,
        metrics: Optional[dict] = None,
        error: Optional[str] = None
    ) -> Optional[TrainingJob]:
        """Update job status."""
        try:
            result = self.container.read_item(item=job_id, partition_key=user_id)
            result['status'] = status
            if status == 'running':
                result['startedAt'] = datetime.now(timezone.utc).isoformat()
            if status in ('completed', 'failed'):
                result['completedAt'] = datetime.now(timezone.utc).isoformat()
            if metrics:
                result['metrics'] = metrics
            if error:
                result['error'] = error
            updated = self.container.upsert_item(body=result)
            return TrainingJob(**updated)
        except exceptions.CosmosResourceNotFoundError:
            return None


class MLTrainingDataRepository(BaseRepository):
    """Repository for ML training data points."""

    def __init__(self):
        super().__init__("ml_training_data", "userId")

    async def create(self, data_point: MLTrainingDataPoint) -> MLTrainingDataPoint:
        """Create a new training data point."""
        try:
            result = self.container.upsert_item(body=data_point.model_dump(mode='json'))
            logger.info(f"Created ML training data point: {data_point.id}")
            return MLTrainingDataPoint(**result)
        except Exception as e:
            logger.error(f"Failed to create training data point: {e}")
            raise

    async def get_by_treatment(self, user_id: str, treatment_id: str) -> Optional[MLTrainingDataPoint]:
        """Get training data for a specific treatment."""
        query = """
            SELECT * FROM c
            WHERE c.userId = @userId AND c.treatmentId = @treatmentId
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@treatmentId", "value": treatment_id}
            ],
            partition_key=user_id
        ))
        return MLTrainingDataPoint(**items[0]) if items else None

    async def get_incomplete(self, user_id: str, limit: int = 100) -> List[MLTrainingDataPoint]:
        """Get incomplete data points that need checkpoint collection."""
        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.userId = @userId AND c.isComplete = false
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@limit", "value": limit},
                {"name": "@userId", "value": user_id}
            ],
            partition_key=user_id
        ))
        return [MLTrainingDataPoint(**item) for item in items]

    async def get_by_food(self, user_id: str, food_id: str, limit: int = 50) -> List[MLTrainingDataPoint]:
        """Get all training data for a specific food."""
        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.userId = @userId AND c.foodId = @foodId AND c.isComplete = true
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@limit", "value": limit},
                {"name": "@userId", "value": user_id},
                {"name": "@foodId", "value": food_id}
            ],
            partition_key=user_id
        ))
        return [MLTrainingDataPoint(**item) for item in items]

    async def get_recent_complete(self, user_id: str, days: int = 30, limit: int = 500) -> List[MLTrainingDataPoint]:
        """Get recent completed training data for model training."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.userId = @userId
                AND c.isComplete = true
                AND c.timestamp >= @cutoff
            ORDER BY c.timestamp DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@limit", "value": limit},
                {"name": "@userId", "value": user_id},
                {"name": "@cutoff", "value": cutoff.isoformat()}
            ],
            partition_key=user_id
        ))
        return [MLTrainingDataPoint(**item) for item in items]

    async def update_checkpoint(
        self,
        data_point_id: str,
        user_id: str,
        checkpoint_min: int,
        actual_bg: float
    ) -> Optional[MLTrainingDataPoint]:
        """Update a checkpoint with actual BG reading."""
        try:
            result = self.container.read_item(item=data_point_id, partition_key=user_id)

            now = datetime.now(timezone.utc).isoformat()

            if checkpoint_min == 30:
                result['actualBg30'] = actual_bg
                result['error30'] = actual_bg - result['predictedBg30']
                result['collectedAt30'] = now
            elif checkpoint_min == 60:
                result['actualBg60'] = actual_bg
                result['error60'] = actual_bg - result['predictedBg60']
                result['collectedAt60'] = now
            elif checkpoint_min == 90:
                result['actualBg90'] = actual_bg
                result['error90'] = actual_bg - result['predictedBg90']
                result['collectedAt90'] = now

            # Check if complete
            if all([result.get('actualBg30'), result.get('actualBg60'), result.get('actualBg90')]):
                result['isComplete'] = True

            result['updatedAt'] = now

            updated = self.container.upsert_item(body=result)
            return MLTrainingDataPoint(**updated)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def count_by_food(self, user_id: str, food_id: str) -> int:
        """Count completed observations for a food."""
        query = """
            SELECT VALUE COUNT(1)
            FROM c
            WHERE c.userId = @userId AND c.foodId = @foodId AND c.isComplete = true
        """
        results = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@foodId", "value": food_id}
            ],
            partition_key=user_id
        ))
        return results[0] if results else 0


class FoodAbsorptionProfileRepository(BaseRepository):
    """Repository for learned food absorption profiles."""

    def __init__(self):
        super().__init__("food_profiles", "userId")

    async def upsert(self, profile: FoodAbsorptionProfile) -> FoodAbsorptionProfile:
        """Create or update a food absorption profile."""
        try:
            result = self.container.upsert_item(body=profile.model_dump(mode='json'))
            logger.info(f"Upserted food profile: {profile.id}")
            return FoodAbsorptionProfile(**result)
        except Exception as e:
            logger.error(f"Failed to upsert food profile: {e}")
            raise

    async def get_by_food(self, user_id: str, food_id: str) -> Optional[FoodAbsorptionProfile]:
        """Get profile for a specific food."""
        profile_id = f"{user_id}_{food_id}"
        try:
            result = self.container.read_item(item=profile_id, partition_key=user_id)
            return FoodAbsorptionProfile(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_all_for_user(self, user_id: str) -> List[FoodAbsorptionProfile]:
        """Get all food profiles for a user."""
        query = """
            SELECT * FROM c
            WHERE c.userId = @userId
            ORDER BY c.sampleCount DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))
        return [FoodAbsorptionProfile(**item) for item in items]

    async def get_learned_profiles(self, user_id: str, min_samples: int = 5) -> List[FoodAbsorptionProfile]:
        """Get profiles with enough samples to be considered 'learned'."""
        query = """
            SELECT * FROM c
            WHERE c.userId = @userId AND c.sampleCount >= @minSamples
            ORDER BY c.confidence DESC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[
                {"name": "@userId", "value": user_id},
                {"name": "@minSamples", "value": min_samples}
            ],
            partition_key=user_id
        ))
        return [FoodAbsorptionProfile(**item) for item in items]

    async def search_similar(self, user_id: str, keywords: List[str]) -> List[FoodAbsorptionProfile]:
        """Search for similar foods by keywords."""
        # Simple keyword matching - could be enhanced with embeddings later
        # For now, look for any profile containing any of the keywords
        query = """
            SELECT * FROM c
            WHERE c.userId = @userId AND c.sampleCount >= 3
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user_id}],
            partition_key=user_id
        ))

        # Filter by keywords in description
        matches = []
        for item in items:
            desc = item.get('foodDescription', '').lower()
            if any(kw.lower() in desc for kw in keywords):
                matches.append(FoodAbsorptionProfile(**item))

        return sorted(matches, key=lambda x: x.confidence, reverse=True)[:5]


class UserAbsorptionProfileRepository(BaseRepository):
    """Repository for personalized absorption curve profiles."""

    def __init__(self):
        super().__init__("absorption_profiles", "userId")

    async def get(self, user_id: str) -> Optional[UserAbsorptionProfile]:
        """Get absorption profile for a user."""
        try:
            doc_id = f"{user_id}_absorption_profile"
            result = self.container.read_item(item=doc_id, partition_key=user_id)
            return UserAbsorptionProfile(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def upsert(self, profile: UserAbsorptionProfile) -> UserAbsorptionProfile:
        """Create or update an absorption profile."""
        result = self.container.upsert_item(body=profile.model_dump(mode='json'))
        logger.info(f"Upserted absorption profile for user {profile.userId}")
        return UserAbsorptionProfile(**result)

    async def delete(self, user_id: str) -> bool:
        """Delete an absorption profile."""
        try:
            doc_id = f"{user_id}_absorption_profile"
            self.container.delete_item(item=doc_id, partition_key=user_id)
            logger.info(f"Deleted absorption profile for user {user_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    async def get_or_create_default(self, user_id: str) -> UserAbsorptionProfile:
        """Get existing profile or create one with default values."""
        existing = await self.get(user_id)
        if existing:
            return existing

        # Create default profile
        default_profile = UserAbsorptionProfile(
            id=f"{user_id}_absorption_profile",
            userId=user_id
            # All other fields use Pydantic defaults
        )
        return await self.upsert(default_profile)


class ProfileRepository(BaseRepository):
    """Repository for managed profiles.

    A managed profile represents a person whose diabetes data is managed by an account.
    This enables one account (e.g., a parent) to manage multiple people's data.

    Partition key: accountId - ensures all profiles for an account are stored together.
    """

    def __init__(self):
        super().__init__("profiles", "accountId")

    async def create(self, profile: ManagedProfile) -> ManagedProfile:
        """Create a new managed profile."""
        try:
            result = self.container.upsert_item(body=profile.model_dump(mode='json'))
            logger.info(f"Created managed profile: {profile.id} for account {profile.accountId}")
            return ManagedProfile(**result)
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"Profile already exists: {profile.id}")
            raise ValueError(f"Profile {profile.id} already exists")

    async def get_by_id(self, profile_id: str, account_id: str) -> Optional[ManagedProfile]:
        """Get a profile by ID."""
        try:
            result = self.container.read_item(item=profile_id, partition_key=account_id)
            return ManagedProfile(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_by_account(self, account_id: str, include_inactive: bool = False) -> List[ManagedProfile]:
        """Get all profiles for an account."""
        if include_inactive:
            query = """
                SELECT *
                FROM c
                WHERE c.accountId = @accountId
                ORDER BY c.createdAt ASC
            """
        else:
            query = """
                SELECT *
                FROM c
                WHERE c.accountId = @accountId AND c.isActive = true
                ORDER BY c.createdAt ASC
            """

        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@accountId", "value": account_id}],
            partition_key=account_id
        ))
        return [ManagedProfile(**item) for item in items]

    async def get_summaries(self, account_id: str) -> List[ProfileSummary]:
        """Get profile summaries for the profile selector dropdown."""
        profiles = await self.get_by_account(account_id)
        summaries = []

        for profile in profiles:
            # Count data sources and determine overall sync status
            data_source_repo = ProfileDataSourceRepository()
            sources = await data_source_repo.get_by_profile(profile.id, account_id)

            # Determine overall sync status
            overall_status = SyncStatus.OK
            active_sources = [s for s in sources if s.isActive]
            if not active_sources:
                overall_status = SyncStatus.PENDING
            elif any(s.syncStatus == SyncStatus.ERROR for s in active_sources):
                overall_status = SyncStatus.ERROR
            elif all(s.syncStatus == SyncStatus.OK for s in active_sources):
                overall_status = SyncStatus.OK
            else:
                overall_status = SyncStatus.PENDING

            summaries.append(ProfileSummary(
                id=profile.id,
                displayName=profile.displayName,
                relationship=profile.relationship,
                avatarUrl=profile.avatarUrl,
                diabetesType=profile.diabetesType,
                isActive=profile.isActive,
                lastDataAt=profile.lastDataAt,
                dataSourceCount=len(active_sources),
                syncStatus=overall_status
            ))

        return summaries

    async def get_self_profile(self, account_id: str) -> Optional[ManagedProfile]:
        """Get the 'self' profile for an account (the account owner's own data)."""
        query = """
            SELECT TOP 1 *
            FROM c
            WHERE c.accountId = @accountId AND c.relationship = 'self' AND c.isActive = true
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@accountId", "value": account_id}],
            partition_key=account_id
        ))
        return ManagedProfile(**items[0]) if items else None

    async def update(self, profile_id: str, account_id: str, updates: dict) -> ManagedProfile:
        """Update a profile."""
        profile = await self.get_by_id(profile_id, account_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        profile_data = profile.model_dump(mode='json')
        profile_data.update(updates)
        profile_data['updatedAt'] = datetime.now(timezone.utc).isoformat()

        result = self.container.upsert_item(body=profile_data)
        logger.info(f"Updated managed profile: {profile_id}")
        return ManagedProfile(**result)

    async def update_last_data_at(self, profile_id: str, account_id: str) -> None:
        """Update the lastDataAt timestamp when new data is received."""
        try:
            result = self.container.read_item(item=profile_id, partition_key=account_id)
            result['lastDataAt'] = datetime.now(timezone.utc).isoformat()
            result['updatedAt'] = datetime.now(timezone.utc).isoformat()
            self.container.upsert_item(body=result)
        except exceptions.CosmosResourceNotFoundError:
            pass

    async def delete(self, profile_id: str, account_id: str, hard_delete: bool = False) -> bool:
        """Delete a profile (soft delete by default)."""
        if hard_delete:
            try:
                self.container.delete_item(item=profile_id, partition_key=account_id)
                logger.info(f"Hard deleted profile: {profile_id}")
                return True
            except exceptions.CosmosResourceNotFoundError:
                return False
        else:
            # Soft delete - just mark as inactive
            try:
                await self.update(profile_id, account_id, {'isActive': False})
                logger.info(f"Soft deleted profile: {profile_id}")
                return True
            except ValueError:
                return False

    async def add_data_source(self, profile_id: str, account_id: str, source_id: str) -> ManagedProfile:
        """Add a data source to a profile."""
        profile = await self.get_by_id(profile_id, account_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        if source_id not in profile.dataSourceIds:
            profile.dataSourceIds.append(source_id)
            return await self.update(profile_id, account_id, {
                'dataSourceIds': profile.dataSourceIds
            })
        return profile

    async def remove_data_source(self, profile_id: str, account_id: str, source_id: str) -> ManagedProfile:
        """Remove a data source from a profile."""
        profile = await self.get_by_id(profile_id, account_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        if source_id in profile.dataSourceIds:
            profile.dataSourceIds.remove(source_id)
            updates = {'dataSourceIds': profile.dataSourceIds}

            # Clear primary source references if this was the primary
            if profile.primaryGlucoseSourceId == source_id:
                updates['primaryGlucoseSourceId'] = None
            if profile.primaryTreatmentSourceId == source_id:
                updates['primaryTreatmentSourceId'] = None

            return await self.update(profile_id, account_id, updates)
        return profile

    async def get_all_active(self, limit: int = 1000) -> List[ManagedProfile]:
        """Get all active profiles across all accounts (for background sync)."""
        query = """
            SELECT TOP @limit *
            FROM c
            WHERE c.isActive = true
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True
        ))
        return [ManagedProfile(**item) for item in items]


class ProfileDataSourceRepository(BaseRepository):
    """Repository for profile data sources.

    A profile data source represents a connection to an external data provider
    (e.g., Gluroo, Dexcom) for a specific profile.

    Partition key: profileId - ensures all sources for a profile are stored together.
    Note: We also store accountId for cross-partition queries during sync.
    """

    def __init__(self):
        super().__init__("profile_data_sources", "profileId")

    async def create(self, source: ProfileDataSource) -> ProfileDataSource:
        """Create a new data source."""
        try:
            result = self.container.upsert_item(body=source.model_dump(mode='json'))
            logger.info(f"Created data source: {source.id} for profile {source.profileId}")
            return ProfileDataSource(**result)
        except exceptions.CosmosResourceExistsError:
            logger.warning(f"Data source already exists: {source.id}")
            raise ValueError(f"Data source {source.id} already exists")

    async def get_by_id(self, source_id: str, profile_id: str) -> Optional[ProfileDataSource]:
        """Get a data source by ID."""
        try:
            result = self.container.read_item(item=source_id, partition_key=profile_id)
            return ProfileDataSource(**result)
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def get_by_profile(
        self,
        profile_id: str,
        account_id: str,
        include_inactive: bool = False
    ) -> List[ProfileDataSource]:
        """Get all data sources for a profile."""
        if include_inactive:
            query = """
                SELECT *
                FROM c
                WHERE c.profileId = @profileId
                ORDER BY c.priority ASC
            """
        else:
            query = """
                SELECT *
                FROM c
                WHERE c.profileId = @profileId AND c.isActive = true
                ORDER BY c.priority ASC
            """

        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@profileId", "value": profile_id}],
            partition_key=profile_id
        ))
        return [ProfileDataSource(**item) for item in items]

    async def get_by_type(
        self,
        profile_id: str,
        source_type: str
    ) -> Optional[ProfileDataSource]:
        """Get a data source by type for a profile."""
        source_id = f"{profile_id}_{source_type}"
        return await self.get_by_id(source_id, profile_id)

    async def update(self, source_id: str, profile_id: str, updates: dict) -> ProfileDataSource:
        """Update a data source."""
        source = await self.get_by_id(source_id, profile_id)
        if not source:
            raise ValueError(f"Data source {source_id} not found")

        source_data = source.model_dump(mode='json')
        source_data.update(updates)
        source_data['updatedAt'] = datetime.now(timezone.utc).isoformat()

        result = self.container.upsert_item(body=source_data)
        logger.info(f"Updated data source: {source_id}")
        return ProfileDataSource(**result)

    async def update_sync_status(
        self,
        source_id: str,
        profile_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> ProfileDataSource:
        """Update sync status after a sync attempt."""
        updates = {
            'syncStatus': status,
            'syncErrorMessage': error_message
        }
        if status == 'ok':
            updates['lastSyncAt'] = datetime.now(timezone.utc).isoformat()
            updates['syncErrorMessage'] = None

        return await self.update(source_id, profile_id, updates)

    async def update_credentials(
        self,
        source_id: str,
        profile_id: str,
        encrypted_credentials: str
    ) -> ProfileDataSource:
        """Update encrypted credentials for a data source."""
        return await self.update(source_id, profile_id, {
            'credentialsEncrypted': encrypted_credentials,
            'syncStatus': 'pending'  # Reset sync status when credentials change
        })

    async def delete(self, source_id: str, profile_id: str) -> bool:
        """Delete a data source."""
        try:
            self.container.delete_item(item=source_id, partition_key=profile_id)
            logger.info(f"Deleted data source: {source_id}")
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

    async def deactivate(self, source_id: str, profile_id: str) -> ProfileDataSource:
        """Deactivate a data source instead of deleting."""
        return await self.update(source_id, profile_id, {
            'isActive': False,
            'syncEnabled': False
        })

    async def get_all_active_for_sync(self, source_type: Optional[str] = None) -> List[ProfileDataSource]:
        """Get all active data sources for background sync.

        Returns sources that are both active and have sync enabled.
        """
        if source_type:
            query = """
                SELECT *
                FROM c
                WHERE c.isActive = true
                    AND c.syncEnabled = true
                    AND c.sourceType = @sourceType
            """
            params = [{"name": "@sourceType", "value": source_type}]
        else:
            query = """
                SELECT *
                FROM c
                WHERE c.isActive = true AND c.syncEnabled = true
            """
            params = []

        items = list(self.container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        return [ProfileDataSource(**item) for item in items]

    async def get_glucose_sources_for_profile(self, profile_id: str) -> List[ProfileDataSource]:
        """Get data sources that provide glucose data, sorted by priority."""
        query = """
            SELECT *
            FROM c
            WHERE c.profileId = @profileId
                AND c.isActive = true
                AND c.providesGlucose = true
            ORDER BY c.priority ASC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@profileId", "value": profile_id}],
            partition_key=profile_id
        ))
        return [ProfileDataSource(**item) for item in items]

    async def get_treatment_sources_for_profile(self, profile_id: str) -> List[ProfileDataSource]:
        """Get data sources that provide treatment data, sorted by priority."""
        query = """
            SELECT *
            FROM c
            WHERE c.profileId = @profileId
                AND c.isActive = true
                AND c.providesTreatments = true
            ORDER BY c.priority ASC
        """
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@profileId", "value": profile_id}],
            partition_key=profile_id
        ))
        return [ProfileDataSource(**item) for item in items]
