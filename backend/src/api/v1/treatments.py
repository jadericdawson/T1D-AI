"""
Treatments API Endpoints for T1D-AI
Provides insulin and carb treatment data.
Triggers prediction refresh after treatment logging for real-time updates.
Auto-syncs treatments to Gluroo when logged.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Body, Depends, BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4

from models.schemas import Treatment, TreatmentType, User, AbsorptionRate, FatContent
from database.repositories import TreatmentRepository, DataSourceRepository, SharingRepository, UserRepository
from auth import get_current_user
from services.food_enrichment_service import food_enrichment_service
from services.gluroo_service import GlurooService
from utils.encryption import decrypt_secret
from api.v1.websocket import manager as ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_data_user_id(profile_id: str) -> str:
    """
    Convert a profile ID to the actual data user ID.

    For 'self' profiles, the profile ID is 'profile_{user_id}' but data
    is stored with just the raw user_id. This function strips the prefix.
    """
    if profile_id.startswith("profile_"):
        return profile_id[8:]  # Strip "profile_" prefix (8 chars)
    return profile_id


treatment_repo = TreatmentRepository()
datasource_repo = DataSourceRepository()
sharing_repo = SharingRepository()
user_repo = UserRepository()


async def validate_user_access(requester_id: str, target_user_id: str) -> bool:
    """
    Check if requester has access to target user's data.
    Access is granted if:
    - requester_id == target_user_id (viewing own data)
    - requester owns the target profile (managed profile)
    - requester is a parent with target_user_id in their linkedChildIds
    - target_user has shared their data with requester via the sharing system

    Returns False on any error to fail safely.
    """
    try:
        # Normalize both IDs for comparison
        # Profile IDs may have 'profile_' prefix but user IDs don't
        normalized_requester = get_data_user_id(requester_id)
        normalized_target = get_data_user_id(target_user_id)

        if normalized_requester == normalized_target:
            return True

        # Also check raw IDs in case one has prefix and one doesn't
        if requester_id == target_user_id:
            return True

        # Check if requester OWNS this profile (managed profile system)
        try:
            from database.repositories import ProfileRepository
            profile_repo = ProfileRepository()
            profile = await profile_repo.get_by_id(target_user_id, normalized_requester)
            if profile:
                logger.info(f"Access granted via profile ownership: {normalized_requester} owns profile {target_user_id}")
                return True
        except Exception as e:
            logger.warning(f"Error checking profile ownership: {e}")

        # Check if requester is a parent of the target
        try:
            requester = await user_repo.get_by_id(requester_id)
            if requester and requester.linkedChildIds:
                # Check both normalized and raw target IDs
                if target_user_id in requester.linkedChildIds or normalized_target in requester.linkedChildIds:
                    return True
        except Exception as e:
            logger.warning(f"Error checking parent-child access: {e}")

        # Check if target user has shared data with requester
        # This handles two cases:
        # 1. Direct share: ownerId = target_user_id (user shares their own data)
        # 2. Profile share: profileId = target_user_id (parent shares child's data)
        try:
            # Try with original target_user_id first (for profile shares)
            share = await sharing_repo.get_share_for_profile(target_user_id, requester_id)
            if share and share.isActive:
                role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                logger.info(f"Access granted via share: {target_user_id} shared with {requester_id} (role: {role_str})")
                return True
            # Also try with normalized target (for direct user shares)
            if normalized_target != target_user_id:
                share = await sharing_repo.get_share_for_profile(normalized_target, requester_id)
                if share and share.isActive:
                    role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                    logger.info(f"Access granted via share: {normalized_target} shared with {requester_id} (role: {role_str})")
                    return True
        except Exception as e:
            logger.warning(f"Error checking share access: {e}")

        return False
    except Exception as e:
        logger.error(f"Unexpected error in validate_user_access: {e}")
        return False


async def get_gluroo_service(user_id: str) -> Optional[GlurooService]:
    """
    Get a configured GlurooService for a user, or None if not available.
    """
    try:
        datasource = await datasource_repo.get(user_id, "gluroo")
        if not datasource:
            return None

        creds = datasource.credentials
        if not creds or not creds.syncEnabled:
            return None

        url = creds.url
        api_secret_encrypted = creds.apiSecretEncrypted

        if not url or not api_secret_encrypted:
            return None

        api_secret = decrypt_secret(api_secret_encrypted)
        return GlurooService(base_url=url, api_secret=api_secret)
    except Exception as e:
        logger.error(f"Error getting Gluroo service: {e}")
        return None


async def push_treatment_to_gluroo(user_id: str, treatment: Treatment):
    """
    Background task to push a treatment to Gluroo.
    Silently fails if user has no Gluroo datasource or push fails.
    """
    try:
        # Get treatment type as string for robust comparison
        treatment_type_str = str(treatment.type).lower()
        logger.info(f"Gluroo push: type={treatment_type_str}, insulin={treatment.insulin}, carbs={treatment.carbs}")

        service = await get_gluroo_service(user_id)
        if not service:
            logger.warning(f"No Gluroo service available for user {user_id}, skipping push (check datasource config)")
            return

        # Determine treatment type and value (robust string comparison)
        is_insulin = treatment_type_str in ("insulin", "correction bolus") and treatment.insulin
        is_carbs = treatment_type_str in ("carbs", "carb correction") and treatment.carbs

        if is_insulin:
            logger.info(f"Pushing insulin treatment: {treatment.insulin}U at {treatment.timestamp}")
            success, message, response = await service.push_treatment(
                treatment_type="insulin",
                value=treatment.insulin,
                timestamp=treatment.timestamp,
                notes=treatment.notes
            )
        elif is_carbs:
            logger.info(f"Pushing carbs treatment: {treatment.carbs}g at {treatment.timestamp}")
            success, message, response = await service.push_treatment(
                treatment_type="carbs",
                value=treatment.carbs,
                timestamp=treatment.timestamp,
                notes=treatment.notes,
                protein=treatment.protein,
                fat=treatment.fat,
                glycemic_index=treatment.glycemicIndex,
                absorption_rate=treatment.absorptionRate,
                is_liquid=treatment.isLiquid
            )
        else:
            logger.warning(f"Unknown treatment type for push: {treatment.type} (str: {treatment_type_str}), insulin={treatment.insulin}, carbs={treatment.carbs}")
            return

        # Build enrichment details for notification
        enrichment_data = {
            "notes": treatment.notes,
            "protein": treatment.protein,
            "fat": treatment.fat,
            "glycemicIndex": treatment.glycemicIndex,
            "absorptionRate": str(treatment.absorptionRate) if treatment.absorptionRate else None,
            "isLiquid": treatment.isLiquid,
        }

        if success:
            logger.info(f"Successfully pushed {treatment_type_str} treatment to Gluroo for user {user_id}")
            # Send success notification via WebSocket with enrichment data
            await ws_manager.broadcast_to_user(user_id, {
                "type": "gluroo_sync",
                "status": "success",
                "message": f"Synced {treatment.insulin or treatment.carbs}{'U insulin' if is_insulin else 'g carbs'} to Gluroo",
                "treatment_type": treatment_type_str,
                "value": treatment.insulin if is_insulin else treatment.carbs,
                "carbs": treatment.carbs,
                "insulin": treatment.insulin,
                **enrichment_data
            })
        else:
            logger.error(f"Gluroo push FAILED for {treatment_type_str}: {message}")
            # Send failure notification via WebSocket with enrichment data
            await ws_manager.broadcast_to_user(user_id, {
                "type": "gluroo_sync",
                "status": "error",
                "message": f"Failed to sync to Gluroo: {message}",
                "treatment_type": treatment_type_str,
                "value": treatment.insulin if is_insulin else treatment.carbs,
                "carbs": treatment.carbs,
                "insulin": treatment.insulin,
                **enrichment_data
            })

    except Exception as e:
        logger.error(f"Error pushing treatment to Gluroo: {e}", exc_info=True)
        # Send error notification via WebSocket
        try:
            await ws_manager.broadcast_to_user(user_id, {
                "type": "gluroo_sync",
                "status": "error",
                "message": f"Gluroo sync error: {str(e)}",
                "treatment_type": str(treatment.type).lower() if treatment else "unknown"
            })
        except:
            pass  # Don't fail if WebSocket notification fails


async def delete_treatment_from_gluroo(user_id: str, treatment: Treatment):
    """
    Background task to delete a treatment from Gluroo.
    Silently fails if user has no Gluroo datasource or delete fails.
    """
    try:
        service = await get_gluroo_service(user_id)
        if not service:
            logger.debug(f"No Gluroo service available for user {user_id}, skipping delete")
            return

        # Determine treatment type and value for matching
        if treatment.type == TreatmentType.INSULIN and treatment.insulin:
            treatment_type = "insulin"
            value = treatment.insulin
        elif treatment.type == TreatmentType.CARBS and treatment.carbs:
            treatment_type = "carbs"
            value = treatment.carbs
        else:
            logger.warning(f"Unknown treatment type for delete: {treatment.type}")
            return

        # Use sourceId for direct deletion if treatment came from Gluroo
        # sourceId contains the Gluroo/Nightscout _id
        nightscout_id = treatment.sourceId if treatment.source == "gluroo" else None
        logger.info(f"Gluroo delete: treatment.source={treatment.source}, sourceId={treatment.sourceId}")

        success, message = await service.delete_treatment(
            nightscout_id=nightscout_id,
            timestamp=treatment.timestamp,
            treatment_type=treatment_type,
            value=value
        )

        if success:
            logger.info(f"Deleted {treatment.type} treatment from Gluroo for user {user_id}")
        else:
            logger.warning(f"Gluroo delete failed (may not exist there): {message}")

    except Exception as e:
        logger.error(f"Error deleting treatment from Gluroo: {e}")


class LogTreatmentRequest(BaseModel):
    """Request to log a new treatment."""
    type: TreatmentType
    insulin: Optional[float] = None
    carbs: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    notes: Optional[str] = None  # Food description for AI glycemic prediction
    timestamp: Optional[datetime] = None


@router.get("/recent", response_model=List[Treatment])
async def get_recent_treatments(
    hours: int = Query(default=24, ge=1, le=168),
    treatment_type: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None, description="User ID whose treatments to view (optional, defaults to current user)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent treatments.

    Returns insulin and/or carb treatments for the specified time period.
    If user_id is provided, validates that the current user has access to view that user's treatments.
    """
    # Use current user if no user_id provided
    target_user_id = user_id or current_user.id

    # Validate access if viewing another user's data
    if target_user_id != current_user.id:
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's treatments")

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(target_user_id)

    try:
        treatments = await treatment_repo.get_recent(
            user_id=data_user_id,
            hours=hours,
            treatment_type=treatment_type
        )
        return treatments

    except Exception as e:
        logger.error(f"Error getting treatments: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class TreatmentResponse(BaseModel):
    """Treatment response with prediction refresh flag."""
    treatment: Treatment
    predictionsStale: bool = True  # Indicates predictions should be refreshed


@router.post("/log", response_model=TreatmentResponse)
async def log_treatment(
    request: LogTreatmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Log a new treatment (insulin or carbs).

    For carb treatments with food description (notes), AI glycemic prediction
    will analyze the food and set glycemic index, load, and absorption rate.

    Returns predictionsStale=True to signal frontend to refresh TFT predictions.
    """
    user_id = current_user.id
    try:
        # Validate request
        if request.type == TreatmentType.INSULIN and not request.insulin:
            raise HTTPException(status_code=400, detail="Insulin amount required for insulin treatment")
        if request.type == TreatmentType.CARBS and not request.carbs:
            raise HTTPException(status_code=400, detail="Carbs amount required for carb treatment")

        # Initialize treatment fields
        glycemic_index = None
        glycemic_load = None
        absorption_rate = None
        fat_content = None
        is_liquid = None
        enriched_at = None
        estimated_protein = request.protein
        estimated_fat = request.fat

        # For carb treatments, enrich with AI macro estimation and glycemic prediction
        if request.type == TreatmentType.CARBS and request.carbs:
            try:
                # Step 1: If protein/fat not provided and we have food notes, estimate them with GPT
                if request.notes and (not request.protein or not request.fat):
                    await food_enrichment_service.initialize()
                    macro_estimate = await food_enrichment_service.estimate_macros_from_description(
                        food_description=request.notes,
                        known_carbs=request.carbs
                    )
                    # Use estimated values if not provided by user
                    if not request.protein:
                        estimated_protein = macro_estimate.protein_g
                    if not request.fat:
                        estimated_fat = macro_estimate.fat_g

                    logger.info(
                        f"AI macro estimation for '{request.notes}': "
                        f"protein={estimated_protein:.1f}g, fat={estimated_fat:.1f}g "
                        f"(confidence={macro_estimate.confidence:.2f})"
                    )

                # Step 2: Get glycemic features (GI, absorption rate, etc.)
                features = await food_enrichment_service.extract_food_features(
                    food_text=request.notes or "",
                    carbs=request.carbs,
                    protein=estimated_protein or 0,
                    fat=estimated_fat or 0
                )
                glycemic_index = features.glycemic_index
                glycemic_load = features.glycemic_load
                absorption_rate = features.absorption_rate
                fat_content = features.fat_content
                is_liquid = features.is_liquid
                enriched_at = datetime.now(timezone.utc)

                logger.info(
                    f"AI glycemic prediction for '{request.notes}': "
                    f"GI={glycemic_index}, GL={glycemic_load:.1f}, "
                    f"absorption={absorption_rate}, fat={fat_content}, liquid={is_liquid}"
                )
            except Exception as e:
                logger.warning(f"Food enrichment failed, using defaults: {e}")
                # Default values for unknown food
                glycemic_index = 55
                glycemic_load = request.carbs * 55 / 100 if request.carbs else 0
                absorption_rate = "medium"
                fat_content = "low"
                is_liquid = False

        treatment = Treatment(
            id=f"{user_id}_{uuid4().hex[:12]}",
            userId=user_id,
            timestamp=request.timestamp or datetime.now(timezone.utc),
            type=request.type,
            insulin=request.insulin,
            carbs=request.carbs,
            protein=estimated_protein,  # Use AI-estimated if not provided
            fat=estimated_fat,          # Use AI-estimated if not provided
            notes=request.notes,
            source="manual",
            glycemicIndex=glycemic_index,
            glycemicLoad=glycemic_load,
            absorptionRate=absorption_rate,
            fatContent=fat_content,
            isLiquid=is_liquid,
            enrichedAt=enriched_at
        )

        created = await treatment_repo.create(treatment)
        logger.info(f"Logged treatment: {request.type} for user {user_id}")

        # Push to Gluroo in background (non-blocking)
        background_tasks.add_task(push_treatment_to_gluroo, user_id, created)

        # Return with flag indicating predictions need refresh
        return TreatmentResponse(treatment=created, predictionsStale=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging treatment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class UpdateTreatmentRequest(BaseModel):
    """Request to update a treatment."""
    carbs: Optional[float] = None
    insulin: Optional[float] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None


@router.put("/{treatment_id}", response_model=TreatmentResponse)
async def update_treatment(
    treatment_id: str,
    request: UpdateTreatmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing treatment.
    Also pushes updated treatment to Gluroo if connected.
    """
    user_id = current_user.id
    try:
        # Get existing treatment
        existing = await treatment_repo.get_by_id(treatment_id, user_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Treatment not found")

        # Update fields
        if request.carbs is not None:
            existing.carbs = request.carbs
        if request.insulin is not None:
            existing.insulin = request.insulin
        if request.protein is not None:
            existing.protein = request.protein
        if request.fat is not None:
            existing.fat = request.fat
        if request.notes is not None:
            existing.notes = request.notes
        if request.timestamp is not None:
            existing.timestamp = request.timestamp

        # Re-enrich carb treatments if notes changed
        if existing.type == TreatmentType.CARBS and request.notes and existing.carbs:
            try:
                features = await food_enrichment_service.extract_food_features(
                    food_text=request.notes,
                    carbs=existing.carbs,
                    protein=existing.protein or 0,
                    fat=existing.fat or 0
                )
                existing.glycemicIndex = features.glycemic_index
                existing.glycemicLoad = features.glycemic_load
                existing.absorptionRate = features.absorption_rate
                existing.fatContent = features.fat_content
                existing.enrichedAt = datetime.now(timezone.utc)
            except Exception as e:
                logger.warning(f"Food enrichment failed during update: {e}")

        # Mark as user-edited to prevent Gluroo sync from overwriting
        existing.userEdited = True

        updated = await treatment_repo.upsert(existing)
        logger.info(f"Updated treatment {treatment_id} for user {user_id}")

        # Push updated treatment to Gluroo in background
        background_tasks.add_task(push_treatment_to_gluroo, user_id, updated)

        return TreatmentResponse(treatment=updated, predictionsStale=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating treatment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{treatment_id}")
async def delete_treatment(
    treatment_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a treatment.
    Also deletes from Gluroo if connected.
    """
    user_id = current_user.id
    try:
        # Verify treatment exists and belongs to user
        existing = await treatment_repo.get_by_id(treatment_id, user_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Treatment not found")

        # Delete from local database
        await treatment_repo.delete(treatment_id, user_id)
        logger.info(f"Deleted treatment {treatment_id} for user {user_id}")

        # Delete from Gluroo in background (best effort)
        background_tasks.add_task(delete_treatment_from_gluroo, user_id, existing)

        return {"message": "Treatment deleted successfully", "predictionsStale": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting treatment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/iob")
async def get_iob(current_user: User = Depends(get_current_user)):
    """
    Get current Insulin on Board (IOB).
    """
    from services.iob_cob_service import IOBCOBService

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(current_user.id)
    try:
        service = IOBCOBService.from_settings()
        treatments = await treatment_repo.get_for_iob_calculation(data_user_id)
        iob = service.calculate_iob(treatments)

        return {
            "iob": iob,
            "unit": "U",
            "calculatedAt": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating IOB: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cob")
async def get_cob(current_user: User = Depends(get_current_user)):
    """
    Get current Carbs on Board (COB).
    """
    from services.iob_cob_service import IOBCOBService

    # Normalize profile ID to data user ID
    data_user_id = get_data_user_id(current_user.id)
    try:
        service = IOBCOBService.from_settings()
        treatments = await treatment_repo.get_for_cob_calculation(data_user_id)
        cob = service.calculate_cob(treatments)

        return {
            "cob": cob,
            "unit": "g",
            "calculatedAt": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating COB: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class EnrichmentResponse(BaseModel):
    """Response from batch enrichment."""
    total_found: int
    enriched_count: int
    error_count: int
    skipped_count: int
    dry_run: bool


@router.post("/enrich-batch", response_model=EnrichmentResponse)
async def batch_enrich_treatments(
    dry_run: bool = Query(default=True, description="Preview changes without updating"),
    limit: int = Query(default=50, ge=1, le=500, description="Max treatments to process"),
    current_user: User = Depends(get_current_user)
):
    """
    Batch enrich existing carb treatments with GPT-4.1 macro estimates.

    Finds treatments with food notes that haven't been enriched yet and
    uses GPT-4.1 to estimate fat, protein, glycemic index, and absorption rate.

    This enriched data improves COB predictions for the ML models.

    Set dry_run=false to actually update treatments.
    """
    user_id = current_user.id

    try:
        # Initialize enrichment service
        await food_enrichment_service.initialize()

        # Find treatments needing enrichment (have notes but no enrichment)
        # Get recent carb treatments
        all_treatments = await treatment_repo.get_recent(
            user_id=user_id,
            hours=720,  # 30 days
            treatment_type="carbs"
        )

        # Filter to those with notes but not enriched
        to_enrich = [
            t for t in all_treatments
            if t.notes and t.notes.strip() and t.enrichedAt is None
        ][:limit]

        logger.info(f"Found {len(to_enrich)} treatments needing enrichment (limit: {limit})")

        enriched_count = 0
        error_count = 0
        skipped_count = 0

        for treatment in to_enrich:
            try:
                # Get macro estimates from GPT-4.1
                from services.food_enrichment_service import MacroEstimate
                estimate = await food_enrichment_service.estimate_macros_from_description(
                    food_description=treatment.notes,
                    known_carbs=treatment.carbs
                )

                if estimate.confidence < 0.3:
                    skipped_count += 1
                    logger.info(f"Skipping low-confidence estimate for '{treatment.notes[:30]}'")
                    continue

                if not dry_run:
                    # Update treatment with enriched data
                    treatment.protein = estimate.protein_g
                    treatment.fat = estimate.fat_g
                    treatment.glycemicIndex = estimate.glycemic_index
                    treatment.absorptionRate = estimate.absorption_rate
                    treatment.enrichedAt = datetime.now(timezone.utc)

                    await treatment_repo.upsert(treatment)

                logger.info(
                    f"{'[DRY RUN] ' if dry_run else ''}Enriched '{treatment.notes[:40]}': "
                    f"protein={estimate.protein_g:.0f}g, fat={estimate.fat_g:.0f}g, "
                    f"GI={estimate.glycemic_index}, rate={estimate.absorption_rate}"
                )
                enriched_count += 1

            except Exception as e:
                logger.error(f"Failed to enrich treatment {treatment.id}: {e}")
                error_count += 1

        return EnrichmentResponse(
            total_found=len(to_enrich),
            enriched_count=enriched_count,
            error_count=error_count,
            skipped_count=skipped_count,
            dry_run=dry_run
        )

    except Exception as e:
        logger.error(f"Batch enrichment error: {e}")
        raise HTTPException(status_code=500, detail=f"Enrichment error: {str(e)}")


@router.post("/admin/enrich-all", response_model=EnrichmentResponse)
async def admin_batch_enrich_all_treatments(
    admin_key: str = Query(..., description="Admin API key for authentication"),
    dry_run: bool = Query(default=True, description="Preview changes without updating"),
    limit: int = Query(default=100, ge=1, le=500, description="Max treatments to process"),
    user_id: Optional[str] = Query(default=None, description="Specific user ID to enrich, or all users")
):
    """
    Admin endpoint to batch enrich treatments across all users.

    Requires admin_key for authentication (no user login needed).
    Use for scheduled enrichment jobs.
    """
    from config import get_settings
    settings = get_settings()

    # Verify admin key (use jwt_secret_key as admin key for simplicity)
    if admin_key != settings.jwt_secret_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    try:
        # Initialize enrichment service
        await food_enrichment_service.initialize()

        # Get all treatments needing enrichment
        from azure.cosmos import CosmosClient
        cosmos_client = CosmosClient(settings.cosmos_endpoint, settings.cosmos_key)
        database = cosmos_client.get_database_client(settings.cosmos_database)
        treatments_container = database.get_container_client("treatments")

        # Query for treatments with notes but not enriched
        if user_id:
            query = """
                SELECT TOP @limit *
                FROM c
                WHERE c.userId = @userId
                  AND IS_DEFINED(c.notes) AND c.notes != null AND c.notes != ""
                  AND (NOT IS_DEFINED(c.enrichedAt) OR c.enrichedAt = null)
                ORDER BY c.timestamp DESC
            """
            params = [
                {"name": "@limit", "value": limit},
                {"name": "@userId", "value": user_id}
            ]
        else:
            query = """
                SELECT TOP @limit *
                FROM c
                WHERE IS_DEFINED(c.notes) AND c.notes != null AND c.notes != ""
                  AND (NOT IS_DEFINED(c.enrichedAt) OR c.enrichedAt = null)
                ORDER BY c.timestamp DESC
            """
            params = [{"name": "@limit", "value": limit}]

        items = list(treatments_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))

        logger.info(f"Admin enrichment: Found {len(items)} treatments needing enrichment")

        enriched_count = 0
        error_count = 0
        skipped_count = 0

        for item in items:
            notes = item.get('notes', '')
            carbs = item.get('carbs', 0)

            if not notes or not notes.strip():
                skipped_count += 1
                continue

            try:
                estimate = await food_enrichment_service.estimate_macros_from_description(
                    food_description=notes,
                    known_carbs=carbs
                )

                if estimate.confidence < 0.3:
                    skipped_count += 1
                    continue

                if not dry_run:
                    # Update item
                    item['protein'] = estimate.protein_g
                    item['fat'] = estimate.fat_g
                    item['glycemicIndex'] = estimate.glycemic_index
                    item['absorptionRate'] = estimate.absorption_rate
                    item['enrichedAt'] = datetime.now(timezone.utc).isoformat()

                    treatments_container.upsert_item(body=item)

                logger.info(
                    f"{'[DRY RUN] ' if dry_run else ''}Enriched '{notes[:40]}': "
                    f"protein={estimate.protein_g:.0f}g, fat={estimate.fat_g:.0f}g, "
                    f"GI={estimate.glycemic_index}"
                )
                enriched_count += 1

            except Exception as e:
                logger.error(f"Failed to enrich: {e}")
                error_count += 1

        return EnrichmentResponse(
            total_found=len(items),
            enriched_count=enriched_count,
            error_count=error_count,
            skipped_count=skipped_count,
            dry_run=dry_run
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin enrichment error: {e}")
        raise HTTPException(status_code=500, detail=f"Enrichment error: {str(e)}")
