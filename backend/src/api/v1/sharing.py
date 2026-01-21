"""
Account Sharing API Routes.
Allows users to share their data with other registered users.
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr

from auth.routes import get_current_user
from database.repositories import (
    SharingRepository, InvitationRepository, UserRepository
)
from models.schemas import (
    AccountShare, ShareInvitation, ShareRole, SharePermission
)
from services.email_service import get_email_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sharing", tags=["sharing"])

# Repositories
sharing_repo = SharingRepository()
invitation_repo = InvitationRepository()
user_repo = UserRepository()


# ==================== Request/Response Models ====================

class InviteRequest(BaseModel):
    """Request to invite a user to view your data."""
    email: EmailStr
    profileId: Optional[str] = None  # Specific profile to share (None = all profiles)
    profileName: Optional[str] = None  # Profile display name for invitation
    role: str = "viewer"
    permissions: List[str] = []


class InviteResponse(BaseModel):
    """Response after sending invitation."""
    invitationId: str
    inviteeEmail: str
    expiresAt: datetime
    message: str


class ShareResponse(BaseModel):
    """Share information for display."""
    id: str
    ownerId: str
    ownerEmail: Optional[str] = None
    ownerName: Optional[str] = None
    profileId: Optional[str] = None  # Specific profile shared (None = all profiles)
    profileName: Optional[str] = None  # Profile display name
    sharedWithId: str
    sharedWithEmail: str
    sharedWithName: Optional[str] = None
    role: str
    permissions: List[str]
    createdAt: datetime
    isActive: bool


class UpdatePermissionsRequest(BaseModel):
    """Request to update share permissions."""
    permissions: List[str]


class AcceptInviteResponse(BaseModel):
    """Response after accepting invitation."""
    shareId: str
    ownerId: str
    ownerEmail: str
    message: str


# ==================== Endpoints ====================

@router.post("/invite", response_model=InviteResponse)
async def invite_user(request: InviteRequest, current_user=Depends(get_current_user)):
    """
    Invite another user to view your glucose data.
    Creates a pending invitation that the invitee can accept.
    """
    # Check if user is trying to invite themselves
    if request.email.lower() == current_user.email.lower():
        raise HTTPException(status_code=400, detail="Cannot invite yourself")

    # Check if share already exists
    invitee = await user_repo.get_by_email(request.email)
    if invitee:
        existing_share = await sharing_repo.get_share(current_user.id, invitee.id)
        if existing_share:
            raise HTTPException(
                status_code=409,
                detail="You are already sharing with this user"
            )

    # Validate role
    try:
        role = ShareRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")

    # Set default permissions based on role
    if not request.permissions:
        if role == ShareRole.VIEWER:
            permissions = [
                SharePermission.VIEW_GLUCOSE.value,
                SharePermission.VIEW_TREATMENTS.value,
                SharePermission.VIEW_PREDICTIONS.value,
                SharePermission.VIEW_INSIGHTS.value
            ]
        elif role == ShareRole.CAREGIVER:
            permissions = [
                SharePermission.VIEW_GLUCOSE.value,
                SharePermission.VIEW_TREATMENTS.value,
                SharePermission.VIEW_PREDICTIONS.value,
                SharePermission.VIEW_INSIGHTS.value,
                SharePermission.ADD_TREATMENTS.value,
                SharePermission.RECEIVE_ALERTS.value
            ]
        else:  # admin
            permissions = [p.value for p in SharePermission]
    else:
        permissions = request.permissions

    # Create invitation
    invitation = ShareInvitation(
        id=str(uuid.uuid4()),
        ownerId=current_user.id,
        ownerEmail=current_user.email,
        ownerName=current_user.displayName,
        profileId=request.profileId,
        profileName=request.profileName,
        inviteeEmail=request.email.lower(),
        role=role,
        permissions=[SharePermission(p) for p in permissions],
        expiresAt=datetime.now(timezone.utc) + timedelta(days=7)
    )

    created = await invitation_repo.create(invitation)

    # Send email notification to invitee
    email_service = get_email_service()
    if email_service.is_configured:
        try:
            await email_service.send_sharing_invitation(
                to_email=request.email,
                owner_name=current_user.displayName,
                owner_email=current_user.email,
                role=role.value,
                invitation_token=created.id
            )
            logger.info(f"Sharing invitation email sent to {request.email}")
        except Exception as e:
            # Log but don't fail the invitation creation
            logger.error(f"Failed to send invitation email: {e}")
    else:
        logger.warning("Email service not configured - invitation created without email")

    return InviteResponse(
        invitationId=created.id,
        inviteeEmail=created.inviteeEmail,
        expiresAt=created.expiresAt,
        message=f"Invitation sent to {created.inviteeEmail}. They can accept it from their Settings page."
    )


class InvitationDetailsResponse(BaseModel):
    """Public invitation details (for display on accept page)."""
    id: str
    ownerEmail: str
    ownerName: Optional[str] = None
    profileId: Optional[str] = None
    profileName: Optional[str] = None
    role: str
    expiresAt: datetime


@router.get("/invitation/{token}", response_model=InvitationDetailsResponse)
async def get_invitation_details(token: str):
    """
    Get invitation details by token.
    This endpoint does not require authentication to allow showing
    invitation details to users who haven't logged in yet.
    """
    invitation = await invitation_repo.get_by_token(token)
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or expired")

    # Check if expired
    if datetime.now(timezone.utc) > invitation.expiresAt:
        raise HTTPException(status_code=410, detail="Invitation has expired")

    return InvitationDetailsResponse(
        id=invitation.id,
        ownerEmail=invitation.ownerEmail,
        ownerName=invitation.ownerName,
        profileId=getattr(invitation, 'profileId', None),
        profileName=getattr(invitation, 'profileName', None),
        role=invitation.role.value if hasattr(invitation.role, 'value') else invitation.role,
        expiresAt=invitation.expiresAt
    )


@router.post("/accept/{token}", response_model=AcceptInviteResponse)
async def accept_invitation(token: str, current_user=Depends(get_current_user)):
    """
    Accept a sharing invitation using the invitation token.
    """
    # Get invitation
    invitation = await invitation_repo.get_by_token(token)
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or expired")

    # Check if invitation is for this user
    if invitation.inviteeEmail.lower() != current_user.email.lower():
        raise HTTPException(
            status_code=403,
            detail="This invitation is for a different email address"
        )

    # Check if expired
    if datetime.now(timezone.utc) > invitation.expiresAt:
        raise HTTPException(status_code=410, detail="Invitation has expired")

    # Check if already shared
    existing = await sharing_repo.get_share(invitation.ownerId, current_user.id)
    if existing:
        raise HTTPException(status_code=409, detail="Share already exists")

    # Create the share
    share = AccountShare(
        id=str(uuid.uuid4()),
        ownerId=invitation.ownerId,
        sharedWithId=current_user.id,
        sharedWithEmail=current_user.email,
        profileId=invitation.profileId,
        profileName=invitation.profileName,
        role=invitation.role,
        permissions=invitation.permissions
    )

    created = await sharing_repo.create_share(share)

    # Mark invitation as used
    await invitation_repo.mark_used(token, invitation.ownerId)

    # Get owner info for response
    owner = await user_repo.get_by_id(invitation.ownerId)

    # Build response message - mention profile name if sharing a specific profile
    if invitation.profileName:
        message = f"You now have access to {invitation.profileName}'s glucose data"
    else:
        message = f"You now have access to {owner.displayName or owner.email}'s glucose data"

    return AcceptInviteResponse(
        shareId=created.id,
        ownerId=invitation.ownerId,
        ownerEmail=invitation.ownerEmail,
        message=message
    )


@router.get("/my-shares", response_model=List[ShareResponse])
async def get_my_shares(current_user=Depends(get_current_user)):
    """
    Get list of users I'm sharing my data with.
    """
    shares = await sharing_repo.get_shares_by_owner(current_user.id)

    responses = []
    for share in shares:
        # Get shared user info
        shared_user = await user_repo.get_by_id(share.sharedWithId)
        responses.append(ShareResponse(
            id=share.id,
            ownerId=share.ownerId,
            ownerEmail=current_user.email,
            ownerName=current_user.displayName,
            profileId=getattr(share, 'profileId', None),
            profileName=getattr(share, 'profileName', None),
            sharedWithId=share.sharedWithId,
            sharedWithEmail=share.sharedWithEmail,
            sharedWithName=shared_user.displayName if shared_user else None,
            role=share.role,
            permissions=share.permissions,
            createdAt=share.createdAt,
            isActive=share.isActive
        ))

    return responses


@router.get("/shared-with-me", response_model=List[ShareResponse])
async def get_shared_with_me(current_user=Depends(get_current_user)):
    """
    Get list of users who have shared their data with me.
    """
    shares = await sharing_repo.get_shares_with_user(current_user.id)

    responses = []
    for share in shares:
        # Get owner info
        owner = await user_repo.get_by_id(share.ownerId)
        responses.append(ShareResponse(
            id=share.id,
            ownerId=share.ownerId,
            ownerEmail=owner.email if owner else None,
            ownerName=owner.displayName if owner else None,
            profileId=getattr(share, 'profileId', None),
            profileName=getattr(share, 'profileName', None),
            sharedWithId=share.sharedWithId,
            sharedWithEmail=share.sharedWithEmail,
            sharedWithName=current_user.displayName,
            role=share.role,
            permissions=share.permissions,
            createdAt=share.createdAt,
            isActive=share.isActive
        ))

    return responses


@router.get("/pending-invitations")
async def get_pending_invitations(current_user=Depends(get_current_user)):
    """
    Get pending invitations for the current user.
    """
    invitations = await invitation_repo.get_pending_for_email(current_user.email)

    return [{
        "id": inv.id,
        "ownerEmail": inv.ownerEmail,
        "ownerName": inv.ownerName,
        "role": inv.role,
        "permissions": inv.permissions,
        "expiresAt": inv.expiresAt
    } for inv in invitations]


@router.delete("/revoke/{share_id}")
async def revoke_share(share_id: str, current_user=Depends(get_current_user)):
    """
    Revoke access for a user you're sharing with.
    """
    success = await sharing_repo.revoke_share(share_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Share not found")

    return {"message": "Share revoked successfully"}


@router.put("/permissions/{share_id}", response_model=ShareResponse)
async def update_share_permissions(
    share_id: str,
    request: UpdatePermissionsRequest,
    current_user=Depends(get_current_user)
):
    """
    Update permissions for an existing share.
    """
    # Validate permissions
    valid_permissions = [p.value for p in SharePermission]
    for p in request.permissions:
        if p not in valid_permissions:
            raise HTTPException(status_code=400, detail=f"Invalid permission: {p}")

    updated = await sharing_repo.update_permissions(
        share_id, current_user.id, request.permissions
    )

    if not updated:
        raise HTTPException(status_code=404, detail="Share not found")

    shared_user = await user_repo.get_by_id(updated.sharedWithId)

    return ShareResponse(
        id=updated.id,
        ownerId=updated.ownerId,
        ownerEmail=current_user.email,
        ownerName=current_user.displayName,
        sharedWithId=updated.sharedWithId,
        sharedWithEmail=updated.sharedWithEmail,
        sharedWithName=shared_user.displayName if shared_user else None,
        role=updated.role,
        permissions=updated.permissions,
        createdAt=updated.createdAt,
        isActive=updated.isActive
    )


@router.get("/can-view/{user_id}")
async def can_view_user(user_id: str, current_user=Depends(get_current_user)):
    """
    Check if current user can view another user's data.
    Returns the share details if access is granted.
    """
    # Check if viewing own data
    if user_id == current_user.id:
        return {
            "canView": True,
            "isOwner": True,
            "permissions": [p.value for p in SharePermission]
        }

    # Check for share
    share = await sharing_repo.get_share(user_id, current_user.id)
    if share and share.isActive:
        return {
            "canView": True,
            "isOwner": False,
            "role": share.role,
            "permissions": share.permissions
        }

    return {
        "canView": False,
        "isOwner": False,
        "permissions": []
    }
