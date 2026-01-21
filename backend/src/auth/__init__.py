"""
Authentication module for T1D-AI.
Provides email/password and Microsoft OAuth authentication.
"""
from auth.routes import router as auth_router
from auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
    create_refresh_token,
)
from auth.routes import get_current_user

__all__ = [
    "auth_router",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_access_token",
    "create_refresh_token",
    "get_current_user",
]
