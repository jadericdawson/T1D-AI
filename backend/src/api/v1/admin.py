"""
Admin API Routes.
Provides admin-only endpoints for user management, analytics, and platform monitoring.
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from auth.routes import get_current_user
from database.repositories import UserRepository, GlucoseRepository, TreatmentRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

# Repositories
user_repo = UserRepository()
glucose_repo = GlucoseRepository()
treatment_repo = TreatmentRepository()


# ==================== Admin Dependency ====================

async def require_admin(current_user=Depends(get_current_user)):
    """Dependency that requires admin privileges."""
    if not getattr(current_user, 'isAdmin', False):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return current_user


# ==================== Response Models ====================

class UserSummary(BaseModel):
    """Summary of a user for admin view."""
    id: str
    email: str
    displayName: Optional[str]
    createdAt: datetime
    lastLoginAt: Optional[datetime]
    emailVerified: bool
    onboardingCompleted: bool
    authProvider: str
    isAdmin: bool
    # Stats
    profileCount: int = 0
    glucoseReadingCount: int = 0


class PlatformStats(BaseModel):
    """Platform-wide statistics."""
    totalUsers: int
    verifiedUsers: int
    onboardedUsers: int
    activeUsers7d: int
    activeUsers30d: int
    totalGlucoseReadings: int
    totalTreatments: int
    newUsersToday: int
    newUsersThisWeek: int


class UserDetail(BaseModel):
    """Detailed user information for admin view."""
    id: str
    email: str
    displayName: Optional[str]
    createdAt: datetime
    lastLoginAt: Optional[datetime]
    emailVerified: bool
    onboardingCompleted: bool
    authProvider: str
    isAdmin: bool
    accountType: str
    hasT1D: bool
    # Stats
    glucoseReadingCount: int
    treatmentCount: int
    oldestReading: Optional[datetime]
    newestReading: Optional[datetime]


# ==================== Endpoints ====================

@router.get("/users", response_model=List[UserSummary])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    admin_user=Depends(require_admin)
):
    """
    List all users with pagination and optional search.
    Admin only.
    """
    try:
        users = await user_repo.list_all_users(skip=skip, limit=limit, search=search)

        summaries = []
        for user in users:
            # Get basic stats for each user
            glucose_count = await glucose_repo.count_for_user(user.id)

            summaries.append(UserSummary(
                id=user.id,
                email=user.email,
                displayName=user.displayName,
                createdAt=user.createdAt,
                lastLoginAt=getattr(user, 'lastLoginAt', None),
                emailVerified=getattr(user, 'emailVerified', True),
                onboardingCompleted=getattr(user, 'onboardingCompleted', False),
                authProvider=getattr(user, 'authProvider', 'email'),
                isAdmin=getattr(user, 'isAdmin', False),
                glucoseReadingCount=glucose_count
            ))

        return summaries

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.get("/users/{user_id}", response_model=UserDetail)
async def get_user_detail(
    user_id: str,
    admin_user=Depends(require_admin)
):
    """
    Get detailed information about a specific user.
    Admin only.
    """
    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get stats
        glucose_count = await glucose_repo.count_for_user(user_id)
        treatment_count = await treatment_repo.count_for_user(user_id)

        # Get date range of readings
        oldest, newest = await glucose_repo.get_date_range(user_id)

        return UserDetail(
            id=user.id,
            email=user.email,
            displayName=user.displayName,
            createdAt=user.createdAt,
            lastLoginAt=getattr(user, 'lastLoginAt', None),
            emailVerified=getattr(user, 'emailVerified', True),
            onboardingCompleted=getattr(user, 'onboardingCompleted', False),
            authProvider=getattr(user, 'authProvider', 'email'),
            isAdmin=getattr(user, 'isAdmin', False),
            accountType=getattr(user, 'accountType', 'personal'),
            hasT1D=getattr(user, 'hasT1D', True),
            glucoseReadingCount=glucose_count,
            treatmentCount=treatment_count,
            oldestReading=oldest,
            newestReading=newest
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user details")


@router.get("/stats", response_model=PlatformStats)
async def get_platform_stats(admin_user=Depends(require_admin)):
    """
    Get platform-wide statistics.
    Admin only.
    """
    try:
        # Get all users for stats calculation
        all_users = await user_repo.list_all_users(skip=0, limit=10000)

        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        total_users = len(all_users)
        verified_users = sum(1 for u in all_users if getattr(u, 'emailVerified', True))
        onboarded_users = sum(1 for u in all_users if getattr(u, 'onboardingCompleted', False))

        # Active users (by lastLoginAt)
        active_7d = sum(1 for u in all_users
                       if getattr(u, 'lastLoginAt', None) and u.lastLoginAt > week_ago)
        active_30d = sum(1 for u in all_users
                        if getattr(u, 'lastLoginAt', None) and u.lastLoginAt > month_ago)

        # New users
        new_today = sum(1 for u in all_users if u.createdAt >= today_start)
        new_this_week = sum(1 for u in all_users if u.createdAt >= week_ago)

        # Total readings and treatments (aggregate across all users)
        total_glucose = 0
        total_treatments = 0
        for user in all_users:
            total_glucose += await glucose_repo.count_for_user(user.id)
            total_treatments += await treatment_repo.count_for_user(user.id)

        return PlatformStats(
            totalUsers=total_users,
            verifiedUsers=verified_users,
            onboardedUsers=onboarded_users,
            activeUsers7d=active_7d,
            activeUsers30d=active_30d,
            totalGlucoseReadings=total_glucose,
            totalTreatments=total_treatments,
            newUsersToday=new_today,
            newUsersThisWeek=new_this_week
        )

    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get platform stats")


@router.put("/users/{user_id}/admin")
async def toggle_admin(
    user_id: str,
    is_admin: bool,
    admin_user=Depends(require_admin)
):
    """
    Toggle admin status for a user.
    Admin only. Cannot remove own admin status.
    """
    if user_id == admin_user.id and not is_admin:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove your own admin status"
        )

    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        await user_repo.update(user_id, {"isAdmin": is_admin})

        return {"message": f"Admin status {'granted' if is_admin else 'revoked'} for {user.email}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling admin: {e}")
        raise HTTPException(status_code=500, detail="Failed to update admin status")


class DeleteUserResponse(BaseModel):
    """Response for user deletion."""
    message: str
    userId: str
    email: str
    dataRetained: bool
    glucoseReadings: int
    treatments: int


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
async def delete_user(
    user_id: str,
    admin_user=Depends(require_admin)
):
    """
    Delete a user account.
    Admin only. Cannot delete your own account.

    The user's glucose readings and treatments are retained for ML training purposes.
    Only the user account/profile is deleted.
    """
    if user_id == admin_user.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete your own account"
        )

    try:
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get stats before deletion for response
        glucose_count = await glucose_repo.count_for_user(user_id)
        treatment_count = await treatment_repo.count_for_user(user_id)
        user_email = user.email

        # Delete the user account (data stays for ML training)
        await user_repo.delete(user_id)

        logger.info(f"Admin {admin_user.email} deleted user {user_email} (id: {user_id}). "
                   f"Retained {glucose_count} glucose readings and {treatment_count} treatments for ML training.")

        return DeleteUserResponse(
            message=f"User {user_email} deleted. Data retained for ML training.",
            userId=user_id,
            email=user_email,
            dataRetained=True,
            glucoseReadings=glucose_count,
            treatments=treatment_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/resend-verification")
async def resend_verification_email(
    email: str = Query(..., description="Email address to resend verification to"),
    copy_to: Optional[str] = Query(None, description="Optional email to send a copy to"),
    admin_user: User = Depends(require_admin)
):
    """
    Resend verification email to a user (admin only).
    Optionally send a copy to another email for testing.
    """
    try:
        from services.email_service import get_email_service

        # Find the user
        user = await user_repo.get_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {email} not found")

        # Get or generate verification token
        if not user.emailVerificationToken:
            user.emailVerificationToken = secrets.token_urlsafe(32)
            user.emailVerificationExpires = datetime.utcnow() + timedelta(hours=24)
            await user_repo.update(user.id, {
                "emailVerificationToken": user.emailVerificationToken,
                "emailVerificationExpires": user.emailVerificationExpires.isoformat()
            })

        email_service = get_email_service()

        # Send to original user
        try:
            await email_service.send_verification_email(
                to_email=user.email,
                display_name=user.displayName or user.email.split('@')[0],
                verification_token=user.emailVerificationToken
            )
            logger.info(f"Admin {admin_user.email} resent verification to {user.email}")
        except Exception as e:
            logger.error(f"Failed to send to {user.email}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

        # Send copy if requested
        if copy_to:
            try:
                await email_service.send_verification_email(
                    to_email=copy_to,
                    display_name=f"{user.displayName or user.email.split('@')[0]} (Copy)",
                    verification_token=user.emailVerificationToken
                )
                logger.info(f"Sent copy to {copy_to}")
            except Exception as e:
                logger.warning(f"Failed to send copy to {copy_to}: {e}")

        return {
            "message": f"Verification email sent to {user.email}" + (f" and {copy_to}" if copy_to else ""),
            "token": user.emailVerificationToken[:20] + "..."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resending verification: {e}")
        raise HTTPException(status_code=500, detail="Failed to resend verification email")
