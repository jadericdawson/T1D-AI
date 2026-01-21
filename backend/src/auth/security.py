"""
Security utilities for authentication.
Password hashing and JWT token handling.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from passlib.context import CryptContext
from jose import JWTError, jwt

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_access_token_expire_minutes


def _truncate_password(password: str) -> str:
    """Truncate password to 72 bytes (bcrypt limit)."""
    # Encode to bytes and truncate if needed
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate at byte boundary, then decode back
        password_bytes = password_bytes[:72]
        # Decode safely (may truncate mid-character)
        password = password_bytes.decode('utf-8', errors='ignore')
    return password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(_truncate_password(plain_password), hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password (truncates to 72 bytes for bcrypt compatibility)."""
    return pwd_context.hash(_truncate_password(password))


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


def create_refresh_token(user_id: str) -> str:
    """Create a refresh token with longer expiration."""
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=ALGORITHM)
