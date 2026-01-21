"""
Users API Endpoints
User management and settings.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field, EmailStr

from database.repositories import UserRepository, LearnedISFRepository
from models.schemas import User, UserSettings, LearnedISF
from ml.training.isf_learner import ISFLearner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])

# Repository instances
user_repo = UserRepository()
isf_repo = LearnedISFRepository()


# Request/Response Models
class UserCreateRequest(BaseModel):
    """Request to create a new user."""
    email: EmailStr
    displayName: Optional[str] = None


class UserUpdateRequest(BaseModel):
    """Request to update user profile."""
    displayName: Optional[str] = None


class UserSettingsUpdateRequest(BaseModel):
    """Request to update user settings."""
    timezone: Optional[str] = None
    targetBg: Optional[int] = Field(None, ge=70, le=150)
    insulinSensitivity: Optional[float] = Field(None, ge=10, le=200)
    carbRatio: Optional[float] = Field(None, ge=1, le=50)
    insulinDuration: Optional[int] = Field(None, ge=120, le=360)
    carbAbsorptionDuration: Optional[int] = Field(None, ge=60, le=360)
    highThreshold: Optional[int] = Field(None, ge=120, le=300)
    lowThreshold: Optional[int] = Field(None, ge=50, le=100)
    criticalHighThreshold: Optional[int] = Field(None, ge=200, le=400)
    criticalLowThreshold: Optional[int] = Field(None, ge=40, le=70)
    enableAlerts: Optional[bool] = None
    enablePredictiveAlerts: Optional[bool] = None
    showInsights: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    displayName: Optional[str]
    createdAt: datetime
    settings: UserSettings


class UserSettingsResponse(BaseModel):
    """User settings response."""
    settings: UserSettings
    updatedAt: datetime


# Endpoints
@router.get("/me", response_model=UserResponse)
async def get_current_user(user_id: str = Query(..., description="User ID")):
    """
    Get current user profile.

    Returns user information and settings.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=user.id,
            email=user.email,
            displayName=user.displayName,
            createdAt=user.createdAt,
            settings=user.settings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=UserResponse)
async def create_user(request: UserCreateRequest):
    """
    Create a new user.

    Note: In production, this would be called after Azure AD B2C authentication.
    """
    try:
        # Check if user already exists
        existing = await user_repo.get_by_email(request.email)
        if existing:
            raise HTTPException(status_code=409, detail="User already exists")

        # Create new user
        user = await user_repo.create(
            email=request.email,
            display_name=request.displayName
        )

        return UserResponse(
            id=user.id,
            email=user.email,
            displayName=user.displayName,
            createdAt=user.createdAt,
            settings=user.settings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/me", response_model=UserResponse)
async def update_user(
    request: UserUpdateRequest,
    user_id: str = Query(..., description="User ID")
):
    """
    Update user profile.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields
        update_data = request.model_dump(exclude_unset=True)
        updated_user = await user_repo.update(user_id, update_data)

        return UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            displayName=updated_user.displayName,
            createdAt=updated_user.createdAt,
            settings=updated_user.settings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/settings", response_model=UserSettingsResponse)
async def get_user_settings(user_id: str = Query(..., description="User ID")):
    """
    Get user settings.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserSettingsResponse(
            settings=user.settings,
            updatedAt=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    request: UserSettingsUpdateRequest,
    user_id: str = Query(..., description="User ID")
):
    """
    Update user settings.

    Allows partial updates - only specified fields are changed.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get current settings and update
        current_settings = user.settings.model_dump()
        update_data = request.model_dump(exclude_unset=True)

        # Merge updates
        for key, value in update_data.items():
            if value is not None:
                current_settings[key] = value

        # Validate and save
        new_settings = UserSettings(**current_settings)
        await user_repo.update(user_id, {"settings": new_settings.model_dump()})

        return UserSettingsResponse(
            settings=new_settings,
            updatedAt=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/me")
async def delete_user(user_id: str = Query(..., description="User ID")):
    """
    Delete user account.

    This will delete all user data including glucose readings and treatments.
    Use with caution - this action cannot be undone.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        await user_repo.delete(user_id)

        return {"message": "User deleted successfully", "userId": user_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/settings/defaults")
async def get_default_settings():
    """
    Get default user settings.

    Returns the default values for all settings.
    Useful for displaying defaults in the UI.
    """
    defaults = UserSettings()
    return {
        "settings": defaults.model_dump(),
        "description": {
            "timezone": "User's timezone for time-based features",
            "targetBg": "Target blood glucose in mg/dL",
            "insulinSensitivity": "How much 1 unit of insulin lowers BG (ISF)",
            "carbRatio": "Grams of carbs covered by 1 unit of insulin",
            "insulinDuration": "Duration of insulin action in minutes",
            "carbAbsorptionDuration": "Duration of carb absorption in minutes",
            "highThreshold": "BG threshold for high alerts",
            "lowThreshold": "BG threshold for low alerts",
            "criticalHighThreshold": "BG threshold for critical high alerts",
            "criticalLowThreshold": "BG threshold for critical low alerts",
            "enableAlerts": "Enable glucose alerts",
            "enablePredictiveAlerts": "Enable alerts based on predictions",
            "showInsights": "Show AI-generated insights"
        }
    }


# ==================== Family/Child Management ====================

class ChildCreateRequest(BaseModel):
    """Request to create a child account."""
    email: EmailStr = Field(..., description="Email for the child account (can be parent's email with +child suffix)")
    displayName: str = Field(..., description="Child's name")
    dateOfBirth: Optional[datetime] = Field(None, description="Child's date of birth")
    diagnosisDate: Optional[datetime] = Field(None, description="Date of T1D diagnosis")


class ChildResponse(BaseModel):
    """Child account response."""
    id: str
    email: str
    displayName: Optional[str]
    dateOfBirth: Optional[datetime]
    diagnosisDate: Optional[datetime]
    createdAt: datetime
    settings: UserSettings


class FamilyResponse(BaseModel):
    """Response with parent and linked children."""
    parent: UserResponse
    children: list[ChildResponse]


@router.post("/children", response_model=ChildResponse)
async def create_child_account(
    request: ChildCreateRequest,
    parent_id: str = Query(..., description="Parent user ID")
):
    """
    Create a child account linked to a parent.

    This creates a child account that the parent can monitor.
    The child's data (glucose, treatments) will be accessible to the parent.
    """
    try:
        # Verify parent exists
        parent = await user_repo.get_by_id(parent_id)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent account not found")

        # Check if email already in use
        existing = await user_repo.get_by_email(request.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already in use")

        # Create child account
        child = await user_repo.create(
            email=request.email,
            display_name=request.displayName,
            account_type="child",
            parent_id=parent_id,
            guardian_email=parent.email,
            date_of_birth=request.dateOfBirth,
            diagnosis_date=request.diagnosisDate,
            has_t1d=True
        )

        # Link child to parent
        await user_repo.link_child_to_parent(parent_id, child.id)

        # Update parent's account type if not already set
        if parent.accountType == "personal":
            await user_repo.update(parent_id, {"accountType": "parent"})

        logger.info(f"Created child account {child.id} linked to parent {parent_id}")

        return ChildResponse(
            id=child.id,
            email=child.email,
            displayName=child.displayName,
            dateOfBirth=child.dateOfBirth,
            diagnosisDate=child.diagnosisDate,
            createdAt=child.createdAt,
            settings=child.settings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating child account: {e}")
        raise HTTPException(status_code=500, detail="Failed to create child account")


@router.get("/children", response_model=list[ChildResponse])
async def get_children(parent_id: str = Query(..., description="Parent user ID")):
    """
    Get all children linked to a parent account.

    Returns list of child accounts the parent can monitor.
    """
    try:
        children = await user_repo.get_children(parent_id)

        return [
            ChildResponse(
                id=child.id,
                email=child.email,
                displayName=child.displayName,
                dateOfBirth=child.dateOfBirth,
                diagnosisDate=child.diagnosisDate,
                createdAt=child.createdAt,
                settings=child.settings
            )
            for child in children
        ]

    except Exception as e:
        logger.error(f"Error getting children: {e}")
        raise HTTPException(status_code=500, detail="Failed to get children")


@router.get("/family", response_model=FamilyResponse)
async def get_family(parent_id: str = Query(..., description="Parent user ID")):
    """
    Get full family information including parent and all children.

    Returns the parent account and all linked child accounts.
    """
    try:
        parent = await user_repo.get_by_id(parent_id)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent account not found")

        children = await user_repo.get_children(parent_id)

        return FamilyResponse(
            parent=UserResponse(
                id=parent.id,
                email=parent.email,
                displayName=parent.displayName,
                createdAt=parent.createdAt,
                settings=parent.settings
            ),
            children=[
                ChildResponse(
                    id=child.id,
                    email=child.email,
                    displayName=child.displayName,
                    dateOfBirth=child.dateOfBirth,
                    diagnosisDate=child.diagnosisDate,
                    createdAt=child.createdAt,
                    settings=child.settings
                )
                for child in children
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting family: {e}")
        raise HTTPException(status_code=500, detail="Failed to get family")


@router.patch("/children/{child_id}", response_model=ChildResponse)
async def update_child(
    child_id: str,
    request: UserSettingsUpdateRequest,
    parent_id: str = Query(..., description="Parent user ID")
):
    """
    Update a child's settings.

    Parents can update their child's diabetes settings.
    """
    try:
        # Verify parent has access to this child
        parent = await user_repo.get_by_id(parent_id)
        if not parent or child_id not in (parent.linkedChildIds or []):
            raise HTTPException(status_code=403, detail="Not authorized to update this child")

        child = await user_repo.get_by_id(child_id)
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")

        # Update settings
        current_settings = child.settings.model_dump()
        update_data = request.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            if value is not None:
                current_settings[key] = value

        new_settings = UserSettings(**current_settings)
        updated_child = await user_repo.update(child_id, {"settings": new_settings.model_dump()})

        return ChildResponse(
            id=updated_child.id,
            email=updated_child.email,
            displayName=updated_child.displayName,
            dateOfBirth=updated_child.dateOfBirth,
            diagnosisDate=updated_child.diagnosisDate,
            createdAt=updated_child.createdAt,
            settings=updated_child.settings
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating child: {e}")
        raise HTTPException(status_code=500, detail="Failed to update child")


# ==================== ISF Learning ====================

class LearnedISFResponse(BaseModel):
    """Response containing learned ISF values."""
    fastingISF: Optional[float] = Field(None, description="Fasting ISF (mg/dL per unit)")
    mealISF: Optional[float] = Field(None, description="Meal ISF (mg/dL per unit)")
    defaultISF: float = Field(50.0, description="Default ISF fallback")
    fastingConfidence: Optional[float] = Field(None, description="Confidence in fasting ISF")
    mealConfidence: Optional[float] = Field(None, description="Confidence in meal ISF")
    lastUpdated: Optional[datetime] = None


@router.get("/{user_id}/isf", response_model=LearnedISFResponse)
async def get_learned_isf(user_id: str):
    """
    Get learned ISF values for a user.

    Returns both fasting ISF and meal ISF, along with confidence scores.
    The default ISF is used when no learned values are available.
    """
    try:
        result = await isf_repo.get_both(user_id)

        fasting = result.get("fasting")
        meal = result.get("meal")

        return LearnedISFResponse(
            fastingISF=fasting.value if fasting else None,
            mealISF=meal.value if meal else None,
            defaultISF=fasting.value if fasting else (meal.value if meal else 50.0),
            fastingConfidence=fasting.confidence if fasting else None,
            mealConfidence=meal.confidence if meal else None,
            lastUpdated=fasting.lastUpdated if fasting else (meal.lastUpdated if meal else None)
        )

    except Exception as e:
        logger.error(f"Error getting learned ISF: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ISF data")


@router.post("/{user_id}/isf/learn")
async def learn_isf(user_id: str, days: int = Query(default=30, ge=7, le=90)):
    """
    Trigger ISF learning for a user.

    Analyzes bolus and glucose data to learn the user's actual ISF.
    Requires at least 7 days of data.

    Args:
        user_id: User ID to learn ISF for
        days: Number of days of history to analyze (default 30, max 90)
    """
    try:
        # Verify user exists
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Run ISF learning
        learner = ISFLearner()
        result = await learner.learn_all_isf(user_id, days)

        fasting = result.get("fasting")
        meal = result.get("meal")

        return {
            "success": True,
            "message": "ISF learning completed",
            "fastingISF": fasting.value if fasting else None,
            "mealISF": meal.value if meal else None,
            "fastingSamples": fasting.sampleCount if fasting else 0,
            "mealSamples": meal.sampleCount if meal else 0,
            "defaultISF": result.get("default", 50.0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error learning ISF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ISF learning failed: {str(e)}")


# ==================== Onboarding ====================

class UserPreferencesRequest(BaseModel):
    """Request to update user preferences."""
    preferredDataSources: Optional[list[str]] = Field(None, description="Data sources user wants supported")


@router.post("/{user_id}/onboarding/complete")
async def complete_onboarding(user_id: str):
    """
    Mark user's onboarding as complete.

    This is called when the user finishes the onboarding wizard.
    After completion, users will have full access to the dashboard.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        await user_repo.update(user_id, {
            "onboardingCompleted": True,
            "onboardingCompletedAt": datetime.utcnow().isoformat()
        })

        return {
            "success": True,
            "message": "Onboarding completed successfully",
            "onboardingCompleted": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing onboarding: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete onboarding")


@router.put("/{user_id}/preferences")
async def update_preferences(user_id: str, request: UserPreferencesRequest):
    """
    Update user preferences.

    Currently supports:
    - preferredDataSources: List of data source IDs the user wants supported.
      Used for feature prioritization and analytics.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        update_data = {}
        if request.preferredDataSources is not None:
            update_data["preferredDataSources"] = request.preferredDataSources

        if update_data:
            await user_repo.update(user_id, update_data)

        return {
            "success": True,
            "message": "Preferences updated",
            "preferredDataSources": request.preferredDataSources or user.preferredDataSources
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.put("/{user_id}/settings")
async def update_user_settings_by_id(
    user_id: str,
    request: UserSettingsUpdateRequest
):
    """
    Update user settings by user ID (path parameter).

    This endpoint is used by the onboarding wizard.
    Allows partial updates - only specified fields are changed.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get current settings and update
        current_settings = user.settings.model_dump()
        update_data = request.model_dump(exclude_unset=True)

        # Merge updates
        for key, value in update_data.items():
            if value is not None:
                current_settings[key] = value

        # Validate and save
        new_settings = UserSettings(**current_settings)
        await user_repo.update(user_id, {"settings": new_settings.model_dump()})

        return UserSettingsResponse(
            settings=new_settings,
            updatedAt=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
