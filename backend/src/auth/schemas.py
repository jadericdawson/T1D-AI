"""
Authentication schemas for request/response models.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    displayName: Optional[str] = Field(None, description="Display name")


class LoginRequest(BaseModel):
    """Email/password login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserAuthResponse(BaseModel):
    """User info returned after authentication."""
    id: str
    email: str
    displayName: Optional[str]
    createdAt: datetime
    emailVerified: bool = True  # Default True for backwards compatibility
    onboardingCompleted: bool = False
    isAdmin: bool = False


class AuthResponse(BaseModel):
    """Full authentication response with user and tokens."""
    user: UserAuthResponse
    tokens: TokenResponse


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation with token."""
    token: str
    new_password: str = Field(..., min_length=8)


class MicrosoftAuthRequest(BaseModel):
    """Microsoft OAuth callback data."""
    code: str
    state: Optional[str] = None
