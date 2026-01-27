"""
Training API Endpoints
Manage personalized model training for users.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List
import asyncio

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from models.schemas import User, UserModel, TrainingJob, UserModelStatus, LearnedISF, LearnedICR, LearnedPIR, UserAbsorptionProfile
from database.repositories import (
    UserModelRepository, TrainingJobRepository, GlucoseRepository, TreatmentRepository,
    LearnedISFRepository, LearnedICRRepository, LearnedPIRRepository,
    UserAbsorptionProfileRepository, UserRepository, SharingRepository
)
from services.blob_model_store import get_model_store
from services.cosmos_training_loader import CosmosTrainingDataLoader
from ml.training.isf_learner import ISFLearner
from services.metabolic_params_service import get_metabolic_params_service, MetabolicState

# Optional ACI import - may not be available in all environments
try:
    from services.aci_training_service import get_aci_training_service
    ACI_AVAILABLE = True
except ImportError:
    ACI_AVAILABLE = False
    get_aci_training_service = None
from ml.training.enhanced_isf_learner import EnhancedISFLearner, import_and_learn_isf
from ml.training.icr_learner import ICRLearner
from ml.training.pir_learner import PIRLearner
from ml.training.absorption_curve_learner import AbsorptionCurveLearner, learn_absorption_curves
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])


def get_data_user_id(profile_id: str) -> str:
    """Get the userId for database queries. Profile IDs are used AS-IS."""
    return profile_id


# Repository instances
model_repo = UserModelRepository()
job_repo = TrainingJobRepository()
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()
isf_repo = LearnedISFRepository()
icr_repo = LearnedICRRepository()
pir_repo = LearnedPIRRepository()
absorption_profile_repo = UserAbsorptionProfileRepository()
user_repo = UserRepository()
sharing_repo = SharingRepository()
data_loader = CosmosTrainingDataLoader()
isf_learner = ISFLearner()
enhanced_isf_learner = EnhancedISFLearner()
icr_learner = ICRLearner()
pir_learner = PIRLearner()


async def validate_user_access(requester_id: str, target_user_id: str) -> bool:
    """
    Check if requester has access to target user's data.
    Access is granted if:
    - requester_id == target_user_id (viewing own data)
    - requester owns the target profile (managed profile)
    - requester is a parent with target_user_id in their linkedChildIds
    - target_user has shared their data with requester via the sharing system
    """
    try:
        # Normalize both IDs for comparison
        # Profile IDs may have 'profile_' prefix but user IDs don't
        normalized_requester = get_data_user_id(requester_id)
        normalized_target = get_data_user_id(target_user_id)

        if normalized_requester == normalized_target:
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

        # Check if requester is a parent of the target (try both normalized and raw IDs)
        try:
            requester = await user_repo.get_by_id(normalized_requester)
            if requester and requester.linkedChildIds:
                # Check both normalized and raw target IDs
                if normalized_target in requester.linkedChildIds or target_user_id in requester.linkedChildIds:
                    return True
        except Exception as e:
            logger.warning(f"Error checking parent-child access: {e}")

        # Check if target user has shared data with requester (try both ID forms)
        try:
            share = await sharing_repo.get_share_for_profile(normalized_target, normalized_requester)
            if not share:
                # Try with profile_ prefix
                share = await sharing_repo.get_share_for_profile(target_user_id, requester_id)
            if share and share.isActive:
                role_str = share.role.value if hasattr(share.role, 'value') else str(share.role)
                logger.info(f"Access granted via share: {target_user_id} shared with {requester_id} (role: {role_str})")
                return True
        except Exception as e:
            logger.warning(f"Error checking share access: {e}")

        return False
    except Exception as e:
        logger.error(f"Unexpected error in validate_user_access: {e}")
        return False

# Training requirements
MIN_READINGS_FOR_TRAINING = 500
MIN_TREATMENTS_FOR_TRAINING = 20
MIN_DAYS_FOR_TRAINING = 7


# Response Models
class EligibilityResponse(BaseModel):
    """Training eligibility response."""
    eligible: bool
    reason: str
    stats: dict
    requirements: dict


class TrainingStatusResponse(BaseModel):
    """Training status response."""
    hasPersonalizedModel: bool
    modelStatus: Optional[str] = None
    modelVersion: Optional[int] = None
    modelMetrics: Optional[dict] = None
    lastTrainedAt: Optional[datetime] = None
    activeJob: Optional[dict] = None
    recentJobs: List[dict] = []


class StartTrainingResponse(BaseModel):
    """Response for starting training."""
    jobId: str
    status: str
    message: str


class ModelListResponse(BaseModel):
    """List of user models."""
    models: List[dict]
    count: int


@router.get("/eligibility", response_model=EligibilityResponse)
async def check_eligibility(
    model_type: str = Query(default="tft", description="Model type to check eligibility for"),
    current_user: User = Depends(get_current_user)
):
    """
    Check if user is eligible for personalized model training.

    Requires:
    - Minimum 500 glucose readings
    - Minimum 20 treatments (insulin/carbs)
    - At least 7 days of data
    """
    user_id = current_user.id
    try:
        eligible, reason = await data_loader.check_training_eligibility(
            user_id=user_id,
            min_days=MIN_DAYS_FOR_TRAINING,
            min_readings=MIN_READINGS_FOR_TRAINING,
            min_treatments=MIN_TREATMENTS_FOR_TRAINING
        )

        stats = await data_loader.get_training_stats(user_id)

        return EligibilityResponse(
            eligible=eligible,
            reason=reason,
            stats=stats,
            requirements={
                "min_readings": MIN_READINGS_FOR_TRAINING,
                "min_treatments": MIN_TREATMENTS_FOR_TRAINING,
                "min_days": MIN_DAYS_FOR_TRAINING
            }
        )

    except Exception as e:
        logger.error(f"Error checking eligibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(
    model_type: str = Query(default="tft", description="Model type to check status for"),
    current_user: User = Depends(get_current_user)
):
    """
    Get training status and model info for the user.

    Returns information about:
    - Whether user has a personalized model
    - Current model version and metrics
    - Active training jobs
    - Recent training history
    """
    user_id = current_user.id
    try:
        # Get user's model
        user_model = await model_repo.get(user_id, model_type)

        # Get active/recent jobs
        all_jobs = await _get_user_jobs(user_id, model_type)
        active_job = next((j for j in all_jobs if j.status in ("queued", "running")), None)
        recent_jobs = sorted(all_jobs, key=lambda j: j.createdAt, reverse=True)[:5]

        return TrainingStatusResponse(
            hasPersonalizedModel=user_model is not None and user_model.status == UserModelStatus.ACTIVE,
            modelStatus=user_model.status if user_model else None,
            modelVersion=user_model.version if user_model else None,
            modelMetrics=user_model.metrics if user_model else None,
            lastTrainedAt=user_model.trainedAt if user_model else None,
            activeJob=active_job.model_dump() if active_job else None,
            recentJobs=[j.model_dump() for j in recent_jobs]
        )

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=StartTrainingResponse)
async def start_training(
    model_type: str = Query(default="tft", description="Model type to train"),
    force: bool = Query(default=False, description="Force training even if not eligible"),
    use_gpu: bool = Query(default=True, description="Use GPU container (faster, ~$3/hr vs ~$0.05/hr)"),
    use_aci: bool = Query(default=True, description="Use Azure Container Instances (required for TFT training)"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Start training a personalized model.

    For TFT models, training runs on Azure Container Instances to avoid App Service timeouts.
    GPU containers are faster (~5-10min vs ~30-60min) but cost more (~$3/hr vs ~$0.05/hr).

    Check status with /training/status.
    """
    user_id = current_user.id
    try:
        # Check eligibility unless forced
        if not force:
            eligible, reason = await data_loader.check_training_eligibility(
                user_id=user_id,
                min_days=MIN_DAYS_FOR_TRAINING,
                min_readings=MIN_READINGS_FOR_TRAINING,
                min_treatments=MIN_TREATMENTS_FOR_TRAINING
            )
            if not eligible:
                raise HTTPException(status_code=400, detail=f"Not eligible for training: {reason}")

        # Check for existing active job
        existing_jobs = await _get_user_jobs(user_id, model_type)
        active_job = next((j for j in existing_jobs if j.status in ("queued", "running")), None)
        if active_job:
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {active_job.id}"
            )

        # Create training job
        job = TrainingJob(
            id=str(uuid.uuid4()),
            userId=user_id,
            modelType=model_type,
            status="queued",
            createdAt=datetime.now(timezone.utc)
        )
        await job_repo.create(job)

        # Use ACI for TFT models (required to avoid timeout)
        if use_aci and model_type == "tft" and ACI_AVAILABLE:
            try:
                aci_service = get_aci_training_service()
                aci_result = await aci_service.start_training(
                    user_id=user_id,
                    model_type=model_type,
                    job_id=job.id,
                    use_gpu=use_gpu
                )
                logger.info(f"Started ACI training: {aci_result}")

                # Update job with ACI container info
                await job_repo.update_status(
                    job.id, user_id, "running",
                    container_group=aci_result.get("container_group_name")
                )

                return StartTrainingResponse(
                    jobId=job.id,
                    status="running",
                    message=f"Training started on Azure Container Instance ({('GPU' if use_gpu else 'CPU')}). Check status with /training/status"
                )

            except Exception as aci_error:
                logger.error(f"ACI training failed to start: {aci_error}")
                # Fall back to local training (will likely timeout but try anyway)
                logger.warning("Falling back to local training (may timeout)")

        # Local training (for non-TFT models or ACI fallback)
        if background_tasks:
            background_tasks.add_task(_run_training, job)
        else:
            asyncio.create_task(_run_training(job))

        logger.info(f"Started training job {job.id} for user {user_id}")

        return StartTrainingResponse(
            jobId=job.id,
            status="queued",
            message="Training job started. Check status with /training/status"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_details(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get details of a specific training job."""
    user_id = current_user.id
    try:
        # Query for job
        jobs = await _get_user_jobs(user_id)
        job = next((j for j in jobs if j.id == job_id), None)

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        return job.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """List all personalized models for the user."""
    user_id = current_user.id
    try:
        models = await model_repo.get_all_for_user(user_id)

        return ModelListResponse(
            models=[m.model_dump() for m in models],
            count=len(models)
        )

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_type}")
async def delete_model(
    model_type: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a personalized model."""
    user_id = current_user.id
    try:
        # Delete from blob storage
        model_store = get_model_store()
        await model_store.delete_user_model(user_id, model_type)

        # Update model status in DB
        model = await model_repo.get(user_id, model_type)
        if model:
            model.status = UserModelStatus.PENDING
            model.blobPath = None
            model.metrics = {}
            await model_repo.upsert(model)

        return {"message": f"Model {model_type} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-stats")
async def get_data_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed statistics about user's training data.

    Useful for understanding data quality and coverage.
    """
    user_id = current_user.id
    try:
        stats = await data_loader.get_training_stats(user_id)
        return stats

    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def _get_user_jobs(user_id: str, model_type: Optional[str] = None) -> List[TrainingJob]:
    """Get all training jobs for a user."""
    # This would ideally have a better query but for now we'll get pending and filter
    pending = await job_repo.get_pending(limit=100)
    user_jobs = [j for j in pending if j.userId == user_id]
    if model_type:
        user_jobs = [j for j in user_jobs if j.modelType == model_type]
    return user_jobs


async def _run_training(job: TrainingJob):
    """Run training job in background."""
    try:
        logger.info(f"Starting training job {job.id}")

        # Update status to running
        await job_repo.update_status(job.id, job.userId, "running")

        # Update model status
        model = await model_repo.get(job.userId, job.modelType)
        if not model:
            model = UserModel(
                id=f"{job.userId}_{job.modelType}",
                userId=job.userId,
                modelType=job.modelType,
                status=UserModelStatus.TRAINING
            )
        else:
            model.status = UserModelStatus.TRAINING
        await model_repo.upsert(model)

        # Load training data
        glucose_df, treatments_df, metadata = await data_loader.get_user_training_data(
            user_id=job.userId,
            days=90,
            min_readings=100  # Lower threshold for actual training
        )

        # Import and run training
        from ml.training.tft_trainer import TFTTrainingPipeline, TFTTrainingConfig

        config = TFTTrainingConfig(
            n_features=69,
            hidden_size=32,  # Smaller for per-user models
            n_heads=2,
            n_lstm_layers=1,
            dropout=0.1,
            encoder_length=24,
            prediction_length=12,
            horizons_minutes=[15, 30, 45, 60],
            quantiles=[0.1, 0.5, 0.9],
            quantile_weights=[1.5, 1.0, 0.8],
            learning_rate=0.001,
            batch_size=16,
            epochs=50,
            patience=10,
            grad_clip=1.0,
            weight_decay=1e-5,
            min_completeness_score=0.3,
            max_glucose_gap_minutes=30,
            min_treatments_per_day=0.3,
            time_exclusion_patterns=[],
            val_split=0.15,
            test_split=0.15,
        )

        pipeline = TFTTrainingPipeline(config=config, device="auto")

        # Prepare data
        data_metrics = pipeline.prepare_data(glucose_df, treatments_df)

        if data_metrics.valid_windows < 50:
            raise ValueError(f"Not enough valid training windows: {data_metrics.valid_windows}")

        # Train
        test_metrics = pipeline.train(
            epochs=config.epochs,
            patience=config.patience,
            mlflow_tracking=False
        )

        # Save to blob storage
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = Path(f.name)

        pipeline.save_checkpoint(str(temp_path))
        model_data = temp_path.read_bytes()
        temp_path.unlink()

        # Upload to blob storage
        model_store = get_model_store()
        blob_metadata = await model_store.upload_user_model(
            user_id=job.userId,
            model_type=job.modelType,
            model_data=model_data,
            metrics=test_metrics,
            config=config.to_dict() if hasattr(config, 'to_dict') else {}
        )

        # Update model record
        model.status = UserModelStatus.ACTIVE
        model.version = blob_metadata.get("version", 1)
        model.blobPath = blob_metadata.get("blob_path")
        model.metrics = test_metrics
        model.dataPoints = len(glucose_df)
        model.trainedAt = datetime.now(timezone.utc)
        await model_repo.upsert(model)

        # Update job as completed
        await job_repo.update_status(job.id, job.userId, "completed", metrics=test_metrics)

        logger.info(f"Training completed for job {job.id}. MAE@30min: {test_metrics.get('mae_30min', 'N/A')}")

    except Exception as e:
        logger.error(f"Training failed for job {job.id}: {e}")
        import traceback
        traceback.print_exc()

        # Update status to failed
        await job_repo.update_status(job.id, job.userId, "failed", error=str(e))

        # Update model status
        model = await model_repo.get(job.userId, job.modelType)
        if model:
            model.status = UserModelStatus.FAILED
            model.lastError = str(e)
            await model_repo.upsert(model)


# ==================== ISF Learning Endpoints ====================

class ISFLearningResponse(BaseModel):
    """Response from ISF learning."""
    success: bool
    message: str
    fasting_isf: Optional[dict] = None
    meal_isf: Optional[dict] = None
    default_isf: float
    sample_count: int


class ISFStatusResponse(BaseModel):
    """Current ISF status for user."""
    has_learned_isf: bool
    fasting_isf: Optional[float] = None
    fasting_confidence: Optional[float] = None
    fasting_samples: int = 0
    meal_isf: Optional[float] = None
    meal_confidence: Optional[float] = None
    meal_samples: int = 0
    default_isf: float = 55.0
    last_updated: Optional[datetime] = None
    time_of_day_pattern: Optional[dict] = None
    # Short-term ISF (recent 2-3 days) for detecting illness/resistance changes
    current_isf: Optional[float] = None
    current_isf_confidence: Optional[float] = None
    current_isf_samples: int = 0
    isf_deviation: Optional[float] = None  # Percentage deviation from baseline (-30 = 30% more resistant)
    recent_data_points: Optional[list] = None  # Recent ISF observations for debugging


@router.post("/isf/learn", response_model=ISFLearningResponse)
async def learn_isf(
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Learn ISF from user's insulin dose and BG response data.

    Analyzes:
    - Fasting corrections (insulin with no carbs within ±2 hours)
    - Meal boluses (insulin given with carbs)

    Requires at least 7 days of data with insulin doses.
    """
    user_id = current_user.id
    try:
        logger.info(f"Learning ISF for user {user_id} over {days} days")

        # Learn both fasting and meal ISF
        result = await isf_learner.learn_all_isf(user_id, days)

        fasting = result.get("fasting")
        meal = result.get("meal")
        default = result.get("default", 55.0)

        sample_count = 0
        if fasting:
            sample_count += fasting.sampleCount
        if meal:
            sample_count += meal.sampleCount

        if sample_count == 0:
            return ISFLearningResponse(
                success=False,
                message="Not enough valid insulin/BG data to learn ISF. Need clean correction boluses (no carbs within 2 hours).",
                fasting_isf=None,
                meal_isf=None,
                default_isf=default,
                sample_count=0
            )

        return ISFLearningResponse(
            success=True,
            message=f"Learned ISF from {sample_count} observations",
            fasting_isf=fasting.model_dump() if fasting else None,
            meal_isf=meal.model_dump() if meal else None,
            default_isf=default,
            sample_count=sample_count
        )

    except Exception as e:
        logger.error(f"Error learning ISF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/isf/status", response_model=ISFStatusResponse)
async def get_isf_status(
    user_id: Optional[str] = Query(default=None, description="User ID to get ISF status for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current learned ISF status for user.

    Returns both fasting and meal ISF if available,
    plus time-of-day patterns and short-term ISF for detecting
    temporary resistance changes (illness, stress, etc).
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's ISF status"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        # Get both ISF types
        isf_data = await isf_repo.get_both(data_user_id)

        fasting = isf_data.get("fasting")
        meal = isf_data.get("meal")

        has_learned = bool(fasting or meal)

        # Determine default ISF (prefer fasting, then meal, then 55)
        default_isf = 55.0
        if fasting and fasting.value:
            default_isf = fasting.value
        elif meal and meal.value:
            default_isf = meal.value

        # Get most recent update time
        last_updated = None
        if fasting and fasting.lastUpdated:
            last_updated = fasting.lastUpdated
        if meal and meal.lastUpdated:
            if not last_updated or meal.lastUpdated > last_updated:
                last_updated = meal.lastUpdated

        # Get time of day pattern
        tod_pattern = None
        if fasting and fasting.timeOfDayPattern:
            tod_pattern = fasting.timeOfDayPattern

        # Calculate short-term ISF from recent data (last 3 days)
        short_term = await isf_learner.calculate_short_term_isf(data_user_id, days=3)
        current_isf = short_term.get("current_isf")
        current_isf_confidence = short_term.get("confidence")
        current_isf_samples = short_term.get("sample_count", 0)
        recent_data_points = short_term.get("data_points")

        # Calculate deviation from baseline
        # Positive = more sensitive (needs less insulin)
        # Negative = more resistant (needs more insulin, like when sick)
        isf_deviation = None
        if current_isf and default_isf > 0:
            # Deviation as percentage: (current - baseline) / baseline * 100
            isf_deviation = round((current_isf - default_isf) / default_isf * 100, 1)

        return ISFStatusResponse(
            has_learned_isf=has_learned,
            fasting_isf=fasting.value if fasting else None,
            fasting_confidence=fasting.confidence if fasting else None,
            fasting_samples=fasting.sampleCount if fasting else 0,
            meal_isf=meal.value if meal else None,
            meal_confidence=meal.confidence if meal else None,
            meal_samples=meal.sampleCount if meal else 0,
            default_isf=default_isf,
            last_updated=last_updated,
            time_of_day_pattern=tod_pattern,
            current_isf=current_isf,
            current_isf_confidence=current_isf_confidence,
            current_isf_samples=current_isf_samples,
            isf_deviation=isf_deviation,
            recent_data_points=recent_data_points
        )

    except Exception as e:
        logger.error(f"Error getting ISF status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/isf/reset")
async def reset_learned_isf(
    current_user: User = Depends(get_current_user)
):
    """
    Reset learned ISF data for user.

    Use this if you want to re-learn ISF from scratch,
    for example after changing insulin or becoming more active.
    """
    user_id = current_user.id
    try:
        # Delete both ISF types
        deleted = []

        fasting = await isf_repo.get(user_id, "fasting")
        if fasting:
            # We don't have a delete method, so we'll reset the values
            fasting.value = 50.0
            fasting.confidence = 0.0
            fasting.sampleCount = 0
            fasting.history = []
            fasting.timeOfDayPattern = {"morning": None, "afternoon": None, "evening": None, "night": None}
            await isf_repo.upsert(fasting)
            deleted.append("fasting")

        meal = await isf_repo.get(user_id, "meal")
        if meal:
            meal.value = 50.0
            meal.confidence = 0.0
            meal.sampleCount = 0
            meal.history = []
            meal.timeOfDayPattern = {"morning": None, "afternoon": None, "evening": None, "night": None}
            await isf_repo.upsert(meal)
            deleted.append("meal")

        return {
            "message": f"Reset ISF data: {', '.join(deleted) if deleted else 'No ISF data to reset'}",
            "reset_types": deleted
        }

    except Exception as e:
        logger.error(f"Error resetting ISF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Enhanced ISF Learning Endpoints ====================

class EnhancedISFResponse(BaseModel):
    """Response from enhanced ISF learning."""
    success: bool
    message: str
    imported: int = 0
    validated_clean: int = 0
    rejected: int = 0
    learned_isf: Optional[float] = None
    confidence: Optional[float] = None
    sample_count: int = 0
    time_of_day_pattern: Optional[dict] = None
    isf_range: Optional[dict] = None


@router.post("/isf/import-historic", response_model=EnhancedISFResponse)
async def import_historic_isf_data(
    current_user: User = Depends(get_current_user)
):
    """
    Import and learn ISF from historic bolus_moments.jsonl data.

    This uses the enhanced ISF learner which:
    1. Validates each bolus as "clean" (no undocumented carbs)
    2. Detects anomalies like BG rising after insulin
    3. Tracks contextual features (time of day, lunar phase, etc.)
    4. Uses validated clean boluses as reference for future detection
    """
    user_id = current_user.id
    try:
        logger.info(f"Importing historic ISF data for user {user_id}")

        # Import from bolus_moments.jsonl
        result = await enhanced_isf_learner.import_historic_bolus_data(user_id)

        imported = result.get("imported", 0)
        validated = result.get("validated", 0)
        rejected = result.get("rejected", 0)
        clean_values = result.get("clean_isf_values", [])

        if not clean_values:
            return EnhancedISFResponse(
                success=False,
                message=f"Imported {imported} bolus moments but none passed clean bolus validation",
                imported=imported,
                validated_clean=0,
                rejected=rejected
            )

        # Get the learned ISF from repository
        learned = await isf_repo.get(user_id, "fasting")

        # Calculate ISF range
        isf_values = [v["isf"] for v in clean_values]
        isf_range = {
            "min": round(min(isf_values), 1),
            "max": round(max(isf_values), 1),
            "mean": round(sum(isf_values) / len(isf_values), 1)
        }

        return EnhancedISFResponse(
            success=True,
            message=f"Successfully learned ISF from {validated} clean boluses",
            imported=imported,
            validated_clean=validated,
            rejected=rejected,
            learned_isf=round(learned.value, 1) if learned else None,
            confidence=round(learned.confidence, 2) if learned else None,
            sample_count=len(clean_values),
            time_of_day_pattern=learned.timeOfDayPattern if learned else None,
            isf_range=isf_range
        )

    except Exception as e:
        logger.error(f"Error importing historic ISF data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/isf/learn-enhanced", response_model=EnhancedISFResponse)
async def learn_isf_enhanced(
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze"),
    import_historic: bool = Query(default=True, description="Also import from bolus_moments.jsonl"),
    current_user: User = Depends(get_current_user)
):
    """
    Learn ISF using enhanced clean bolus detection.

    Features:
    - Smart validation: detects undocumented carbs (BG rises after insulin)
    - Contextual learning: time of day, lunar phase, day of year
    - Uses validated profiles to improve future detection
    - Optionally imports from historic bolus_moments.jsonl data
    """
    user_id = current_user.id
    try:
        logger.info(f"Enhanced ISF learning for user {user_id}")

        total_imported = 0
        total_validated = 0
        total_rejected = 0

        # Optionally import historic data first
        if import_historic:
            import_result = await enhanced_isf_learner.import_historic_bolus_data(user_id)
            total_imported += import_result.get("imported", 0)
            total_validated += import_result.get("validated", 0)
            total_rejected += import_result.get("rejected", 0)

        # Learn from recent real-time data
        await enhanced_isf_learner.learn_from_realtime_data(user_id, days)

        # Get final learned ISF
        learned = await isf_repo.get(user_id, "fasting")

        if not learned or learned.sampleCount == 0:
            return EnhancedISFResponse(
                success=False,
                message="Could not learn ISF - no clean boluses found. Need insulin doses without nearby carbs where BG drops as expected.",
                imported=total_imported,
                validated_clean=total_validated,
                rejected=total_rejected
            )

        return EnhancedISFResponse(
            success=True,
            message=f"Learned ISF {learned.value:.1f} from {learned.sampleCount} clean boluses",
            imported=total_imported,
            validated_clean=total_validated,
            rejected=total_rejected,
            learned_isf=round(learned.value, 1),
            confidence=round(learned.confidence, 2),
            sample_count=learned.sampleCount,
            time_of_day_pattern=learned.timeOfDayPattern,
            isf_range={
                "min": round(learned.minISF, 1),
                "max": round(learned.maxISF, 1),
                "mean": round(learned.meanISF, 1),
                "std": round(learned.stdISF, 1)
            }
        )

    except Exception as e:
        logger.error(f"Error in enhanced ISF learning: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/isf/context")
async def get_isf_for_context(
    timestamp: Optional[datetime] = Query(default=None, description="Timestamp for context-aware ISF"),
    is_fasting: bool = Query(default=True, description="Use fasting ISF (true) or meal ISF (false)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get ISF for a specific context (time of day, fasting state).

    Returns the best ISF estimate based on:
    - Time of day patterns (if learned)
    - Fasting vs meal state
    - Confidence level
    """
    user_id = current_user.id
    try:
        isf_value, context_info = await enhanced_isf_learner.get_isf_for_context(
            user_id=user_id,
            timestamp=timestamp,
            is_fasting=is_fasting
        )

        return {
            "isf": round(isf_value, 1),
            "source": context_info.get("source", "default"),
            "confidence": context_info.get("confidence", 0.0),
            "sample_count": context_info.get("sample_count", 0),
            "timestamp": timestamp or datetime.now(timezone.utc),
            "is_fasting": is_fasting
        }

    except Exception as e:
        logger.error(f"Error getting ISF for context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ICR Learning Endpoints ====================

class ICRLearningResponse(BaseModel):
    """Response from ICR learning."""
    success: bool
    message: str
    overall_icr: Optional[dict] = None
    breakfast_icr: Optional[dict] = None
    lunch_icr: Optional[dict] = None
    dinner_icr: Optional[dict] = None
    default_icr: float
    sample_count: int


class ICRStatusResponse(BaseModel):
    """Current ICR status for user."""
    has_learned_icr: bool
    overall_icr: Optional[float] = None
    overall_confidence: Optional[float] = None
    overall_samples: int = 0
    breakfast_icr: Optional[float] = None
    lunch_icr: Optional[float] = None
    dinner_icr: Optional[float] = None
    default_icr: float = 10.0
    last_updated: Optional[datetime] = None
    meal_type_pattern: Optional[dict] = None
    # Short-term ICR (last 3 days) for detecting temporary changes
    current_icr: Optional[float] = None
    current_icr_confidence: Optional[float] = None
    current_icr_samples: int = 0
    icr_deviation: Optional[float] = None  # Percentage deviation from baseline


@router.post("/icr/learn", response_model=ICRLearningResponse)
async def learn_icr(
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze"),
    meal_type: Optional[str] = Query(default=None, description="Specific meal type to learn (breakfast, lunch, dinner)"),
    current_user: User = Depends(get_current_user)
):
    """
    Learn ICR (Carb-to-Insulin Ratio) from user's meal data.

    Analyzes meal boluses where:
    - Carbs and insulin are logged together
    - BG returns close to target after meal
    - Accounts for correction insulin when BG was high

    Returns ICR in grams of carbs per unit of insulin.
    """
    user_id = current_user.id
    try:
        logger.info(f"Learning ICR for user {user_id} over {days} days")

        if meal_type:
            # Learn specific meal type
            result = await icr_learner.learn_icr(user_id, days, meal_type=meal_type)
            sample_count = result.sampleCount if result else 0
            default_icr = result.value if result else 10.0

            return ICRLearningResponse(
                success=result is not None,
                message=f"Learned {meal_type} ICR from {sample_count} meals" if result else f"Not enough {meal_type} data",
                overall_icr=None,
                breakfast_icr=result.model_dump() if result and meal_type == "breakfast" else None,
                lunch_icr=result.model_dump() if result and meal_type == "lunch" else None,
                dinner_icr=result.model_dump() if result and meal_type == "dinner" else None,
                default_icr=default_icr,
                sample_count=sample_count
            )
        else:
            # Learn all meal types
            results = await icr_learner.learn_all_icr(user_id, days)

            overall = results.get("overall")
            breakfast = results.get("breakfast")
            lunch = results.get("lunch")
            dinner = results.get("dinner")
            default_icr = results.get("default", 10.0)

            sample_count = 0
            if overall:
                sample_count = overall.sampleCount

            if sample_count == 0:
                return ICRLearningResponse(
                    success=False,
                    message="Not enough meal data to learn ICR. Need meals with both carbs and insulin logged.",
                    overall_icr=None,
                    breakfast_icr=None,
                    lunch_icr=None,
                    dinner_icr=None,
                    default_icr=10.0,
                    sample_count=0
                )

            return ICRLearningResponse(
                success=True,
                message=f"Learned ICR from {sample_count} meal observations",
                overall_icr=overall.model_dump() if overall else None,
                breakfast_icr=breakfast.model_dump() if breakfast else None,
                lunch_icr=lunch.model_dump() if lunch else None,
                dinner_icr=dinner.model_dump() if dinner else None,
                default_icr=default_icr,
                sample_count=sample_count
            )

    except Exception as e:
        logger.error(f"Error learning ICR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/icr/status", response_model=ICRStatusResponse)
async def get_icr_status(
    user_id: Optional[str] = Query(default=None, description="User ID to get ICR status for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current learned ICR status for user.

    Returns overall and meal-specific ICR values if available.
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's ICR status"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        # Get all ICR types
        icr_data = await icr_repo.get_all(data_user_id)

        overall = icr_data.get("overall")
        breakfast = icr_data.get("breakfast")
        lunch = icr_data.get("lunch")
        dinner = icr_data.get("dinner")

        has_learned = bool(overall or breakfast or lunch or dinner)

        # Determine default ICR
        default_icr = 10.0
        if overall and overall.value:
            default_icr = overall.value

        # Get most recent update time
        last_updated = None
        for icr in [overall, breakfast, lunch, dinner]:
            if icr and icr.lastUpdated:
                if not last_updated or icr.lastUpdated > last_updated:
                    last_updated = icr.lastUpdated

        # Get meal type pattern
        meal_type_pattern = None
        if overall and overall.mealTypePattern:
            meal_type_pattern = overall.mealTypePattern

        # Calculate short-term ICR derived from ISF
        # ICR and ISF both depend on insulin sensitivity, so they should track together
        # This is more reliable than meal-based calculation which gets corrupted when users adjust doses
        current_icr = None
        current_icr_confidence = None
        current_icr_samples = 0
        icr_deviation = None

        try:
            # Get ISF deviation to derive ICR
            from ml.training.isf_learner import ISFLearner
            isf_learner_local = ISFLearner()
            isf_short_term = await isf_learner_local.calculate_short_term_isf(data_user_id, days=3)

            isf_current = isf_short_term.get("current_isf")
            isf_baseline = isf_short_term.get("baseline_isf")
            isf_confidence = isf_short_term.get("confidence", 0)

            if isf_current and isf_baseline and isf_baseline > 0 and isf_confidence > 0.3:
                # Calculate ISF deviation
                isf_deviation_pct = ((isf_current - isf_baseline) / isf_baseline) * 100

                # ICR tracks ISF: if ISF is -23%, ICR should also be -23%
                # Lower ISF = need more insulin = lower ICR (fewer grams per unit)
                icr_deviation = round(isf_deviation_pct, 1)
                current_icr = round(default_icr * (1 + isf_deviation_pct / 100), 1)
                current_icr_confidence = isf_confidence
                current_icr_samples = isf_short_term.get("sample_count", 0)

                logger.info(
                    f"ICR derived from ISF: baseline_icr={default_icr}, current_icr={current_icr}, "
                    f"deviation={icr_deviation}% (ISF: {isf_baseline:.1f} -> {isf_current:.1f})"
                )
        except Exception as e:
            logger.warning(f"Failed to calculate short-term ICR from ISF: {e}")

        return ICRStatusResponse(
            has_learned_icr=has_learned,
            overall_icr=overall.value if overall else None,
            overall_confidence=overall.confidence if overall else None,
            overall_samples=overall.sampleCount if overall else 0,
            breakfast_icr=breakfast.value if breakfast else None,
            lunch_icr=lunch.value if lunch else None,
            dinner_icr=dinner.value if dinner else None,
            default_icr=default_icr,
            last_updated=last_updated,
            meal_type_pattern=meal_type_pattern,
            current_icr=current_icr,
            current_icr_confidence=current_icr_confidence,
            current_icr_samples=current_icr_samples,
            icr_deviation=icr_deviation
        )

    except Exception as e:
        logger.error(f"Error getting ICR status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/icr/reset")
async def reset_learned_icr(
    current_user: User = Depends(get_current_user)
):
    """
    Reset learned ICR data for user.

    Use this to re-learn ICR from scratch if insulin needs have changed.
    """
    user_id = current_user.id
    try:
        deleted = []

        for meal_type in ["overall", "breakfast", "lunch", "dinner"]:
            icr = await icr_repo.get(user_id, meal_type)
            if icr:
                await icr_repo.delete(user_id, meal_type)
                deleted.append(meal_type)

        return {
            "message": f"Reset ICR data: {', '.join(deleted) if deleted else 'No ICR data to reset'}",
            "reset_types": deleted
        }

    except Exception as e:
        logger.error(f"Error resetting ICR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PIR Learning Endpoints ====================

class PIRLearningResponse(BaseModel):
    """Response from PIR learning."""
    success: bool
    message: str
    overall_pir: Optional[dict] = None
    breakfast_pir: Optional[dict] = None
    lunch_pir: Optional[dict] = None
    dinner_pir: Optional[dict] = None
    default_pir: float
    sample_count: int
    timing: Optional[dict] = None


class PIRStatusResponse(BaseModel):
    """Current PIR status for user."""
    has_learned_pir: bool
    overall_pir: Optional[float] = None
    overall_confidence: Optional[float] = None
    overall_samples: int = 0
    breakfast_pir: Optional[float] = None
    lunch_pir: Optional[float] = None
    dinner_pir: Optional[float] = None
    default_pir: float = 25.0
    last_updated: Optional[datetime] = None
    avg_onset_minutes: Optional[int] = None
    avg_peak_minutes: Optional[int] = None
    protein_onset_min: Optional[int] = None
    protein_peak_min: Optional[int] = None
    protein_duration_min: Optional[int] = None
    # Short-term PIR fields for detecting temporary changes (derived from ISF like ICR)
    current_pir: Optional[float] = None
    current_pir_confidence: Optional[float] = None
    current_pir_samples: int = 0
    pir_deviation: Optional[float] = None  # Percentage deviation from baseline
    pir_range: Optional[dict] = None


class PIRTimingResponse(BaseModel):
    """Protein timing details."""
    pir_value: float
    onset_minutes: Optional[int] = None
    peak_minutes: Optional[int] = None
    duration_minutes: int = 180
    timing_advice: str


@router.post("/pir/learn", response_model=PIRLearningResponse)
async def learn_pir(
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze"),
    meal_type: Optional[str] = Query(default=None, description="Specific meal type to learn"),
    current_user: User = Depends(get_current_user)
):
    """
    Learn PIR (Protein-to-Insulin Ratio) from user's protein intake and BG response.

    Analyzes high-protein meals where:
    - Significant protein is consumed (≥15g)
    - A late BG rise pattern is detected (2-4h after meal)

    Also learns protein timing:
    - Onset: When protein starts affecting BG
    - Peak: When protein effect is strongest

    Returns PIR in grams of protein per unit of insulin.
    """
    user_id = current_user.id
    try:
        logger.info(f"Learning PIR for user {user_id} over {days} days")

        if meal_type:
            result = await pir_learner.learn_pir(user_id, days, meal_type=meal_type)
            sample_count = result.sampleCount if result else 0
            default_pir = result.value if result else 25.0

            timing = None
            if result:
                timing = {
                    "onset_minutes": result.avgOnsetMinutes,
                    "peak_minutes": result.avgPeakMinutes
                }

            return PIRLearningResponse(
                success=result is not None,
                message=f"Learned {meal_type} PIR from {sample_count} meals" if result else f"Not enough {meal_type} protein data",
                overall_pir=None,
                breakfast_pir=result.model_dump() if result and meal_type == "breakfast" else None,
                lunch_pir=result.model_dump() if result and meal_type == "lunch" else None,
                dinner_pir=result.model_dump() if result and meal_type == "dinner" else None,
                default_pir=default_pir,
                sample_count=sample_count,
                timing=timing
            )
        else:
            results = await pir_learner.learn_all_pir(user_id, days)

            overall = results.get("overall")
            breakfast = results.get("breakfast")
            lunch = results.get("lunch")
            dinner = results.get("dinner")
            default_pir = results.get("default", 25.0)
            timing = results.get("timing")

            sample_count = 0
            if overall:
                sample_count = overall.sampleCount

            if sample_count == 0:
                return PIRLearningResponse(
                    success=False,
                    message="Not enough high-protein meal data to learn PIR. Need meals with protein that show late BG rise.",
                    overall_pir=None,
                    breakfast_pir=None,
                    lunch_pir=None,
                    dinner_pir=None,
                    default_pir=25.0,
                    sample_count=0,
                    timing=None
                )

            return PIRLearningResponse(
                success=True,
                message=f"Learned PIR from {sample_count} protein observations",
                overall_pir=overall.model_dump() if overall else None,
                breakfast_pir=breakfast.model_dump() if breakfast else None,
                lunch_pir=lunch.model_dump() if lunch else None,
                dinner_pir=dinner.model_dump() if dinner else None,
                default_pir=default_pir,
                sample_count=sample_count,
                timing=timing
            )

    except Exception as e:
        logger.error(f"Error learning PIR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pir/status", response_model=PIRStatusResponse)
async def get_pir_status(
    user_id: Optional[str] = Query(default=None, description="User ID to get PIR status for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current learned PIR status for user.

    Returns overall and meal-specific PIR values plus timing info.
    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's PIR status"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        pir_data = await pir_repo.get_all(data_user_id)

        overall = pir_data.get("overall")
        breakfast = pir_data.get("breakfast")
        lunch = pir_data.get("lunch")
        dinner = pir_data.get("dinner")

        has_learned = bool(overall or breakfast or lunch or dinner)

        default_pir = 25.0
        if overall and overall.value:
            default_pir = overall.value

        last_updated = None
        for pir in [overall, breakfast, lunch, dinner]:
            if pir and pir.lastUpdated:
                if not last_updated or pir.lastUpdated > last_updated:
                    last_updated = pir.lastUpdated

        # Calculate short-term PIR derived from ISF (like ICR)
        # PIR and ISF both depend on insulin sensitivity, so they should track together
        current_pir = None
        current_pir_confidence = None
        current_pir_samples = 0
        pir_deviation = None

        try:
            # Get ISF deviation to derive PIR
            from ml.training.isf_learner import ISFLearner
            isf_learner_instance = ISFLearner()
            isf_short_term = await isf_learner_instance.calculate_short_term_isf(data_user_id, days=3)

            isf_current = isf_short_term.get("current_isf")
            isf_baseline = isf_short_term.get("baseline_isf")
            isf_confidence = isf_short_term.get("confidence", 0)

            if isf_current and isf_baseline and isf_baseline > 0 and isf_confidence > 0.3:
                # Calculate ISF deviation
                isf_deviation_pct = ((isf_current - isf_baseline) / isf_baseline) * 100

                # PIR tracks ISF: if ISF is -23%, PIR should also be -23%
                # Lower ISF = need more insulin = lower PIR (fewer grams per unit)
                pir_deviation = round(isf_deviation_pct, 1)
                current_pir = round(default_pir * (1 + isf_deviation_pct / 100), 1)
                current_pir_confidence = isf_confidence
                current_pir_samples = isf_short_term.get("sample_count", 0)

                logger.info(
                    f"PIR derived from ISF: baseline_pir={default_pir}, current_pir={current_pir}, "
                    f"deviation={pir_deviation}% (ISF: {isf_baseline:.1f} -> {isf_current:.1f})"
                )
        except Exception as e:
            logger.warning(f"Failed to calculate short-term PIR from ISF: {e}")

        # Get PIR range if available
        pir_range = None
        if overall and hasattr(overall, 'minPIR') and overall.minPIR is not None:
            pir_range = {
                "min": overall.minPIR,
                "max": overall.maxPIR,
                "mean": overall.meanPIR,
                "std": getattr(overall, 'stdPIR', None)
            }

        return PIRStatusResponse(
            has_learned_pir=has_learned,
            overall_pir=overall.value if overall else None,
            overall_confidence=overall.confidence if overall else None,
            overall_samples=overall.sampleCount if overall else 0,
            breakfast_pir=breakfast.value if breakfast else None,
            lunch_pir=lunch.value if lunch else None,
            dinner_pir=dinner.value if dinner else None,
            default_pir=default_pir,
            last_updated=last_updated,
            avg_onset_minutes=int(overall.proteinOnsetMin) if overall and overall.proteinOnsetMin else None,
            avg_peak_minutes=int(overall.proteinPeakMin) if overall and overall.proteinPeakMin else None,
            protein_onset_min=int(overall.proteinOnsetMin) if overall and overall.proteinOnsetMin else None,
            protein_peak_min=int(overall.proteinPeakMin) if overall and overall.proteinPeakMin else None,
            protein_duration_min=int(overall.proteinDurationMin) if overall and hasattr(overall, 'proteinDurationMin') and overall.proteinDurationMin else None,
            current_pir=current_pir,
            current_pir_confidence=current_pir_confidence,
            current_pir_samples=current_pir_samples,
            pir_deviation=pir_deviation,
            pir_range=pir_range
        )

    except Exception as e:
        logger.error(f"Error getting PIR status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pir/timing", response_model=PIRTimingResponse)
async def get_protein_timing(
    meal_type: Optional[str] = Query(default=None, description="Specific meal type"),
    current_user: User = Depends(get_current_user)
):
    """
    Get protein timing details for dosing advice.

    Returns:
    - PIR value (grams per unit)
    - Onset time (when protein starts affecting BG)
    - Peak time (when protein effect is strongest)
    - Timing advice for extended bolus
    """
    user_id = current_user.id
    try:
        pir_value, onset, peak = await pir_learner.get_current_pir(user_id, meal_type)

        # Generate timing advice
        if onset and peak:
            advice = f"Consider extended bolus: give carb insulin at meal, then protein insulin over {onset}-{peak} min"
        else:
            advice = "Consider extended bolus: give carb insulin at meal, then protein insulin 2-3 hours later"

        return PIRTimingResponse(
            pir_value=round(pir_value, 1),
            onset_minutes=onset,
            peak_minutes=peak,
            duration_minutes=peak - onset if onset and peak else 180,
            timing_advice=advice
        )

    except Exception as e:
        logger.error(f"Error getting protein timing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pir/reset")
async def reset_learned_pir(
    current_user: User = Depends(get_current_user)
):
    """
    Reset learned PIR data for user.

    Use this to re-learn PIR from scratch if protein sensitivity has changed.
    """
    user_id = current_user.id
    try:
        deleted = []

        for meal_type in ["overall", "breakfast", "lunch", "dinner"]:
            pir = await pir_repo.get(user_id, meal_type)
            if pir:
                await pir_repo.delete(user_id, meal_type)
                deleted.append(meal_type)

        return {
            "message": f"Reset PIR data: {', '.join(deleted) if deleted else 'No PIR data to reset'}",
            "reset_types": deleted
        }

    except Exception as e:
        logger.error(f"Error resetting PIR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Absorption Curve Learning Endpoints ====================

class AbsorptionProfileResponse(BaseModel):
    """Response with learned absorption curve timing."""
    profile: Optional[dict] = None
    learned: bool = False
    message: str = ""
    insulin_timing: Optional[dict] = None
    carb_timing: Optional[dict] = None
    protein_timing: Optional[dict] = None


class LearnAbsorptionCurvesResponse(BaseModel):
    """Response for absorption curve learning."""
    success: bool
    profile: Optional[dict] = None
    message: str = ""
    samples_analyzed: dict = {}


@router.get("/absorption-curves", response_model=AbsorptionProfileResponse)
async def get_absorption_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get the user's learned absorption curve timing profile.

    Returns personalized timing parameters for:
    - Insulin: onset, peak, duration (replaces default peak_min=75)
    - Carbs: onset, peak, duration (replaces default peak_min=45)
    - Protein: onset, peak, duration (replaces default peak_min=180)

    These parameters are used to calculate IOB/COB/POB and BG pressure visualization.
    """
    user_id = current_user.id
    try:
        profile = await absorption_profile_repo.get(user_id)

        if profile is None:
            return AbsorptionProfileResponse(
                profile=None,
                learned=False,
                message="No absorption profile yet. Use POST /training/absorption-curves/learn to create one.",
                insulin_timing={"onset": 15, "peak": 75, "duration": 240},
                carb_timing={"onset": 10, "peak": 45, "duration": 180},
                protein_timing={"onset": 90, "peak": 180, "duration": 300}
            )

        return AbsorptionProfileResponse(
            profile=profile.model_dump(),
            learned=profile.confidence > 0.1,
            message=f"Profile confidence: {profile.confidence:.0%}",
            insulin_timing={
                "onset": profile.insulinOnsetMin,
                "peak": profile.insulinPeakMin,
                "duration": profile.insulinDurationMin,
                "samples": profile.insulinSampleCount
            },
            carb_timing={
                "onset": profile.carbOnsetMin,
                "peak": profile.carbPeakMin,
                "duration": profile.carbDurationMin,
                "samples": profile.carbSampleCount
            },
            protein_timing={
                "onset": profile.proteinOnsetMin,
                "peak": profile.proteinPeakMin,
                "duration": profile.proteinDurationMin,
                "samples": profile.proteinSampleCount
            }
        )

    except Exception as e:
        logger.error(f"Error getting absorption profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/absorption-curves/learn", response_model=LearnAbsorptionCurvesResponse)
async def learn_absorption_curves_endpoint(
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze"),
    isf: float = Query(default=50.0, ge=10, le=200, description="Insulin sensitivity factor"),
    icr: float = Query(default=10.0, ge=3, le=30, description="Insulin-to-carb ratio"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """
    Learn personalized absorption curve timing from historical data.

    This analyzes your treatment history to determine when insulin, carbs, and protein
    actually peak and how long they last, replacing the hardcoded population averages.

    The learned timing improves:
    - BG pressure visualization accuracy
    - IOB/COB/POB calculations
    - Prediction accuracy

    Requirements:
    - At least 7 days of data
    - Ideally 20+ treatments (isolated, no overlapping doses)
    - CGM data with good coverage (>90%)

    Note: This may take 10-30 seconds as it analyzes many treatments.
    """
    user_id = current_user.id
    try:
        logger.info(f"Learning absorption curves for user {user_id}, {days} days")

        # Run learning
        learner = AbsorptionCurveLearner(
            user_id=user_id,
            min_samples_per_type=5
        )

        profile = await learner.learn_from_recent_treatments(
            days=days,
            isf=isf,
            icr=icr
        )

        if profile is None:
            return LearnAbsorptionCurvesResponse(
                success=False,
                message="Not enough data to learn absorption curves. Need more isolated treatments with clean CGM data.",
                samples_analyzed={"insulin": 0, "carbs": 0, "protein": 0}
            )

        # Save the profile
        saved_profile = await absorption_profile_repo.upsert(profile)

        return LearnAbsorptionCurvesResponse(
            success=True,
            profile=saved_profile.model_dump(),
            message=f"Learned absorption curves with {profile.confidence:.0%} confidence",
            samples_analyzed={
                "insulin": profile.insulinSampleCount,
                "carbs": profile.carbSampleCount,
                "protein": profile.proteinSampleCount
            }
        )

    except Exception as e:
        logger.error(f"Error learning absorption curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/absorption-curves/reset")
async def reset_absorption_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Reset learned absorption curves for user.

    Use this to re-learn curves from scratch if absorption has changed
    (e.g., new insulin type, injection site changes, etc.)
    """
    user_id = current_user.id
    try:
        deleted = await absorption_profile_repo.delete(user_id)

        if deleted:
            return {
                "message": "Absorption profile reset successfully. Use POST /training/absorption-curves/learn to re-learn.",
                "deleted": True
            }
        else:
            return {
                "message": "No absorption profile to reset",
                "deleted": False
            }

    except Exception as e:
        logger.error(f"Error resetting absorption profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# METABOLIC STATE ENDPOINTS
# ============================================================================

class MetabolicStateResponse(BaseModel):
    """Response for metabolic state endpoint."""
    state: str = Field(..., description="Overall metabolic state (sick, resistant, normal, sensitive, very_sensitive)")
    state_description: str = Field(..., description="Human-readable description of metabolic state")

    # ISF (Insulin Sensitivity Factor)
    isf_value: float = Field(..., description="Effective ISF to use for calculations")
    isf_baseline: float = Field(..., description="Long-term learned ISF baseline")
    isf_deviation_percent: float = Field(..., description="ISF deviation from baseline")
    isf_source: str = Field(..., description="Source of ISF value (learned, short_term, default)")
    is_resistant: bool = Field(..., description="True if ISF deviation < -10%")
    is_sick: bool = Field(..., description="True if ISF deviation < -15%")

    # ICR (Insulin-to-Carb Ratio)
    icr_value: float = Field(..., description="Effective ICR to use for calculations")
    icr_baseline: float = Field(..., description="Long-term learned ICR baseline")
    icr_deviation_percent: float = Field(..., description="ICR deviation from baseline")
    icr_source: str = Field(..., description="Source of ICR value")

    # PIR (Protein-to-Insulin Ratio)
    pir_value: float = Field(..., description="Effective PIR to use for calculations")
    pir_baseline: float = Field(..., description="Long-term learned PIR baseline")

    # Absorption (optional - may be None if insufficient data)
    absorption_time_to_peak: Optional[float] = Field(None, description="Current time-to-peak in minutes")
    absorption_baseline_time_to_peak: Optional[float] = Field(None, description="Baseline time-to-peak")
    absorption_deviation_percent: Optional[float] = Field(None, description="Absorption rate deviation")
    absorption_state: Optional[str] = Field(None, description="Absorption state (very_slow, slow, normal, fast, very_fast)")
    absorption_is_slow: Optional[bool] = Field(None, description="True if absorption is slower than normal")

    confidence: float = Field(..., description="Overall confidence in metabolic state (0-1)")


@router.get("/metabolic-state", response_model=MetabolicStateResponse)
async def get_metabolic_state(
    is_fasting: bool = Query(False, description="Whether the user is in a fasting state"),
    meal_type: Optional[str] = Query(None, description="Meal type for ICR/PIR (breakfast, lunch, dinner)"),
    user_id: Optional[str] = Query(default=None, description="User ID to get metabolic state for (for shared access)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current metabolic state with all effective parameters.

    This endpoint provides:
    - Overall metabolic state (sick, resistant, normal, sensitive)
    - Effective ISF with short-term deviation (illness detection)
    - Effective ICR with short-term deviation
    - Effective PIR
    - Absorption rate state (if sufficient meal data)

    The metabolic state is determined by:
    - ISF deviation < -15%: SICK
    - ISF deviation < -10% AND slow absorption: SICK
    - ISF deviation < -10%: RESISTANT
    - ISF deviation > +20%: VERY_SENSITIVE
    - ISF deviation > +10%: SENSITIVE
    - Otherwise: NORMAL

    Use this endpoint for:
    - Dashboard metabolic state panel
    - Pre-dose calculations to adjust for illness
    - Alerts about unusual metabolic states

    Supports viewing shared accounts when user_id is provided.
    """
    target_user_id = user_id if user_id else current_user.id

    # Validate access if viewing another user's data
    # Must normalize IDs for comparison (profile_xxx vs xxx)
    if get_data_user_id(target_user_id) != get_data_user_id(current_user.id):
        has_access = await validate_user_access(current_user.id, target_user_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to view this user's metabolic state"
            )

    # Normalize for data queries - data is stored with raw user ID, not profile_ prefix
    data_user_id = get_data_user_id(target_user_id)

    try:
        service = get_metabolic_params_service()
        params = await service.get_all_params(
            data_user_id,
            is_fasting=is_fasting,
            meal_type=meal_type,
            include_short_term=True
        )

        # Calculate overall confidence
        confidences = [params.isf.confidence, params.icr.confidence]
        if params.absorption:
            confidences.append(params.absorption.confidence)
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        response = MetabolicStateResponse(
            state=params.metabolic_state.value,
            state_description=params.state_description,
            isf_value=params.isf.value,
            isf_baseline=params.isf.baseline,
            isf_deviation_percent=params.isf.deviation_percent,
            isf_source=params.isf.source,
            is_resistant=params.isf.is_resistant,
            is_sick=params.isf.is_sick,
            icr_value=params.icr.value,
            icr_baseline=params.icr.baseline,
            icr_deviation_percent=params.icr.deviation_percent,
            icr_source=params.icr.source,
            pir_value=params.pir.value,
            pir_baseline=params.pir.baseline,
            confidence=overall_confidence
        )

        # Add absorption data if available
        if params.absorption:
            response.absorption_time_to_peak = params.absorption.time_to_peak
            response.absorption_baseline_time_to_peak = params.absorption.baseline_time_to_peak
            response.absorption_deviation_percent = params.absorption.deviation_percent
            response.absorption_state = params.absorption.state
            response.absorption_is_slow = params.absorption.is_slow

        return response

    except Exception as e:
        logger.error(f"Error getting metabolic state for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
