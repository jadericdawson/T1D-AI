"""
Users API Endpoints
User management and settings.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field, EmailStr

from database.repositories import UserRepository
from models.schemas import User, UserSettings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])

# Repository instance
user_repo = UserRepository()


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
