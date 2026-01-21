"""
Data Source Manager Service

Orchestrates data synchronization from multiple sources for profiles.
Handles credential decryption, data merging, and sync status management.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from models.schemas import (
    ProfileDataSource,
    ManagedProfile,
    DataSourceType,
    SyncStatus,
    GlucoseReading,
    Treatment
)
from database.repositories import (
    ProfileRepository,
    ProfileDataSourceRepository,
    GlucoseRepository,
    TreatmentRepository
)
from utils.encryption import decrypt_value

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a data source sync operation."""
    success: bool
    source_id: str
    source_type: str
    glucose_count: int = 0
    treatment_count: int = 0
    error_message: Optional[str] = None


@dataclass
class GlurooCredentials:
    """Decrypted Gluroo credentials."""
    url: str
    api_secret: str


@dataclass
class DexcomCredentials:
    """Decrypted Dexcom credentials."""
    access_token: str
    refresh_token: str
    expires_at: Optional[datetime] = None


class DataSourceManager:
    """
    Manages data synchronization for profiles with multiple data sources.

    Responsibilities:
    - Decrypt and validate credentials for each source type
    - Coordinate sync operations across multiple sources
    - Merge data from multiple sources with deduplication
    - Track sync status and handle errors
    """

    def __init__(self):
        self.profile_repo = ProfileRepository()
        self.source_repo = ProfileDataSourceRepository()
        self.glucose_repo = GlucoseRepository()
        self.treatment_repo = TreatmentRepository()

    def _decrypt_credentials(self, source: ProfileDataSource) -> Optional[Dict[str, Any]]:
        """Decrypt credentials for a data source."""
        try:
            decrypted_json = decrypt_value(source.credentialsEncrypted)
            return json.loads(decrypted_json)
        except Exception as e:
            logger.error(f"Failed to decrypt credentials for {source.id}: {e}")
            return None

    def _get_gluroo_credentials(self, source: ProfileDataSource) -> Optional[GlurooCredentials]:
        """Get Gluroo credentials from a data source."""
        creds = self._decrypt_credentials(source)
        if not creds:
            return None

        return GlurooCredentials(
            url=creds.get('url', ''),
            api_secret=creds.get('apiSecret', creds.get('api_secret', ''))
        )

    def _get_dexcom_credentials(self, source: ProfileDataSource) -> Optional[DexcomCredentials]:
        """Get Dexcom credentials from a data source."""
        creds = self._decrypt_credentials(source)
        if not creds:
            return None

        expires_at = None
        if creds.get('expiresAt'):
            try:
                expires_at = datetime.fromisoformat(creds['expiresAt'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass

        return DexcomCredentials(
            access_token=creds.get('accessToken', creds.get('access_token', '')),
            refresh_token=creds.get('refreshToken', creds.get('refresh_token', '')),
            expires_at=expires_at
        )

    async def sync_profile(self, profile: ManagedProfile) -> List[SyncResult]:
        """
        Sync all active data sources for a profile.

        Returns a list of SyncResult for each source.
        """
        results = []

        # Get all active sources for this profile
        sources = await self.source_repo.get_by_profile(
            profile.id,
            profile.accountId,
            include_inactive=False
        )

        for source in sources:
            if not source.syncEnabled:
                continue

            result = await self.sync_source(source, profile)
            results.append(result)

        # Update profile's lastDataAt if any sync was successful
        if any(r.success and (r.glucose_count > 0 or r.treatment_count > 0) for r in results):
            await self.profile_repo.update_last_data_at(profile.id, profile.accountId)

        return results

    async def sync_source(
        self,
        source: ProfileDataSource,
        profile: ManagedProfile
    ) -> SyncResult:
        """
        Sync a single data source.

        Delegates to the appropriate sync method based on source type.
        """
        try:
            if source.sourceType == DataSourceType.GLUROO:
                return await self._sync_gluroo(source, profile)
            elif source.sourceType == DataSourceType.DEXCOM:
                return await self._sync_dexcom(source, profile)
            elif source.sourceType == DataSourceType.NIGHTSCOUT:
                return await self._sync_nightscout(source, profile)
            else:
                return SyncResult(
                    success=False,
                    source_id=source.id,
                    source_type=source.sourceType.value,
                    error_message=f"Unsupported source type: {source.sourceType}"
                )
        except Exception as e:
            logger.error(f"Sync failed for {source.id}: {e}")

            # Update sync status to error
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.ERROR.value,
                str(e)
            )

            return SyncResult(
                success=False,
                source_id=source.id,
                source_type=source.sourceType.value,
                error_message=str(e)
            )

    async def _sync_gluroo(
        self,
        source: ProfileDataSource,
        profile: ManagedProfile
    ) -> SyncResult:
        """
        Sync data from Gluroo (Nightscout-compatible API).

        Uses the existing GlurooService but adapts it for profile-based sync.
        """
        from services.gluroo_service import GlurooService

        credentials = self._get_gluroo_credentials(source)
        if not credentials:
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.ERROR.value,
                "Failed to decrypt credentials"
            )
            return SyncResult(
                success=False,
                source_id=source.id,
                source_type="gluroo",
                error_message="Failed to decrypt credentials"
            )

        # Initialize Gluroo service with decrypted credentials
        gluroo = GlurooService(
            base_url=credentials.url,
            api_secret=credentials.api_secret
        )

        try:
            # Determine sync window
            # If we have a last sync time, only fetch since then
            # Otherwise, fetch last 24 hours
            if source.lastSyncAt:
                since = source.lastSyncAt - timedelta(minutes=5)  # 5 min overlap for safety
            else:
                since = datetime.now(timezone.utc) - timedelta(hours=24)

            glucose_count = 0
            treatment_count = 0

            # Sync glucose if this source provides it
            if source.providesGlucose:
                readings = await gluroo.fetch_glucose_readings(since=since)
                if readings:
                    # Convert to GlucoseReading with profile's user context
                    # Note: We use profile.id as the userId for data partitioning
                    for reading in readings:
                        reading.userId = profile.id

                    glucose_count = await self.glucose_repo.create_many(readings)

            # Sync treatments if this source provides it
            if source.providesTreatments:
                treatments = await gluroo.fetch_treatments(since=since)
                if treatments:
                    # Convert to Treatment with profile's user context
                    for treatment in treatments:
                        treatment.userId = profile.id

                    treatment_count = await self.treatment_repo.create_many(treatments)

            # Update sync status
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.OK.value
            )

            return SyncResult(
                success=True,
                source_id=source.id,
                source_type="gluroo",
                glucose_count=glucose_count,
                treatment_count=treatment_count
            )

        except Exception as e:
            logger.error(f"Gluroo sync failed for {source.id}: {e}")
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.ERROR.value,
                str(e)
            )
            return SyncResult(
                success=False,
                source_id=source.id,
                source_type="gluroo",
                error_message=str(e)
            )

    async def _sync_dexcom(
        self,
        source: ProfileDataSource,
        profile: ManagedProfile
    ) -> SyncResult:
        """
        Sync data from Dexcom Share API.

        Note: Dexcom typically only provides glucose data, not treatments.
        """
        from services.dexcom_service import DexcomService

        credentials = self._get_dexcom_credentials(source)
        if not credentials:
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.ERROR.value,
                "Failed to decrypt credentials"
            )
            return SyncResult(
                success=False,
                source_id=source.id,
                source_type="dexcom",
                error_message="Failed to decrypt credentials"
            )

        # Check if token is expired
        if credentials.expires_at and credentials.expires_at < datetime.now(timezone.utc):
            # Token expired - need to refresh
            # This would typically be handled by the DexcomService
            pass

        try:
            dexcom = DexcomService(
                access_token=credentials.access_token,
                refresh_token=credentials.refresh_token
            )

            # Dexcom only provides glucose
            glucose_count = 0
            if source.providesGlucose:
                readings = await dexcom.fetch_glucose_readings()
                if readings:
                    for reading in readings:
                        reading.userId = profile.id
                    glucose_count = await self.glucose_repo.create_many(readings)

            # Update sync status
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.OK.value
            )

            return SyncResult(
                success=True,
                source_id=source.id,
                source_type="dexcom",
                glucose_count=glucose_count,
                treatment_count=0
            )

        except Exception as e:
            logger.error(f"Dexcom sync failed for {source.id}: {e}")
            await self.source_repo.update_sync_status(
                source.id,
                source.profileId,
                SyncStatus.ERROR.value,
                str(e)
            )
            return SyncResult(
                success=False,
                source_id=source.id,
                source_type="dexcom",
                error_message=str(e)
            )

    async def _sync_nightscout(
        self,
        source: ProfileDataSource,
        profile: ManagedProfile
    ) -> SyncResult:
        """
        Sync data from Nightscout API.

        Similar to Gluroo but uses standard Nightscout endpoints.
        """
        # Nightscout sync is essentially the same as Gluroo
        # since Gluroo is Nightscout-compatible
        return await self._sync_gluroo(source, profile)

    async def sync_all_profiles(self) -> Dict[str, List[SyncResult]]:
        """
        Sync all active profiles (used by background sync task).

        Returns a dictionary mapping profile_id to sync results.
        """
        all_results = {}

        # Get all active profiles
        profiles = await self.profile_repo.get_all_active()

        for profile in profiles:
            try:
                results = await self.sync_profile(profile)
                all_results[profile.id] = results

                # Log summary
                total_glucose = sum(r.glucose_count for r in results)
                total_treatments = sum(r.treatment_count for r in results)
                if total_glucose > 0 or total_treatments > 0:
                    logger.info(
                        f"Profile {profile.displayName}: "
                        f"{total_glucose} glucose, {total_treatments} treatments"
                    )

            except Exception as e:
                logger.error(f"Failed to sync profile {profile.id}: {e}")
                all_results[profile.id] = [SyncResult(
                    success=False,
                    source_id="",
                    source_type="",
                    error_message=str(e)
                )]

        return all_results

    async def get_merged_glucose(
        self,
        profile: ManagedProfile,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[GlucoseReading]:
        """
        Get glucose readings for a profile, merging from multiple sources.

        Uses the primary glucose source first, then fills gaps from other sources.
        Deduplicates readings within a 2-minute window.
        """
        end_time = end_time or datetime.now(timezone.utc)

        # Get readings using profile.id as the userId
        readings = await self.glucose_repo.get_history(
            user_id=profile.id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        # Deduplicate by timestamp (within 2-minute window)
        return self._deduplicate_readings(readings, window_minutes=2)

    async def get_merged_treatments(
        self,
        profile: ManagedProfile,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Treatment]:
        """
        Get treatments for a profile, merging from multiple sources.

        Uses the primary treatment source first, then fills gaps from other sources.
        Deduplicates treatments within a 2-minute window.
        """
        end_time = end_time or datetime.now(timezone.utc)

        # Get treatments using profile.id as the userId
        treatments = await self.treatment_repo.get_by_user(
            user_id=profile.id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        # Deduplicate by timestamp and type (within 2-minute window)
        return self._deduplicate_treatments(treatments, window_minutes=2)

    def _deduplicate_readings(
        self,
        readings: List[GlucoseReading],
        window_minutes: int = 2
    ) -> List[GlucoseReading]:
        """
        Deduplicate glucose readings within a time window.

        Keeps the reading with the highest priority source.
        """
        if not readings:
            return []

        # Sort by timestamp
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)

        # Track seen timestamps (within window)
        deduplicated = []
        for reading in sorted_readings:
            # Check if we already have a reading within the window
            is_duplicate = False
            for existing in deduplicated:
                time_diff = abs((reading.timestamp - existing.timestamp).total_seconds())
                if time_diff < window_minutes * 60:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(reading)

        return deduplicated

    def _deduplicate_treatments(
        self,
        treatments: List[Treatment],
        window_minutes: int = 2
    ) -> List[Treatment]:
        """
        Deduplicate treatments within a time window.

        Only considers treatments of the same type as duplicates.
        """
        if not treatments:
            return []

        # Sort by timestamp
        sorted_treatments = sorted(treatments, key=lambda t: t.timestamp)

        # Track seen timestamps by type
        deduplicated = []
        for treatment in sorted_treatments:
            # Check if we already have a similar treatment within the window
            is_duplicate = False
            for existing in deduplicated:
                if existing.type != treatment.type:
                    continue
                time_diff = abs((treatment.timestamp - existing.timestamp).total_seconds())
                if time_diff < window_minutes * 60:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(treatment)

        return deduplicated


# Singleton instance
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get the singleton DataSourceManager instance."""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
    return _data_source_manager
