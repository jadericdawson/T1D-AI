"""
Encryption Utilities for T1D-AI
Provides symmetric encryption for storing sensitive data like API secrets.
Uses Fernet (AES-128-CBC with HMAC-SHA256).
"""
import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from config import get_settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting sensitive data."""

    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryption service.

        Args:
            key: Fernet-compatible encryption key (32 bytes, base64 encoded).
                 If not provided, uses ENCRYPTION_MASTER_KEY from settings.
        """
        settings = get_settings()
        encryption_key = key or settings.encryption_master_key

        if not encryption_key:
            # Generate a key from JWT secret as fallback (not recommended for production)
            logger.warning(
                "ENCRYPTION_MASTER_KEY not set. Using derived key from JWT_SECRET_KEY. "
                "This is not recommended for production - set ENCRYPTION_MASTER_KEY."
            )
            # Derive a Fernet-compatible key from JWT secret
            derived = hashlib.sha256(settings.jwt_secret_key.encode()).digest()
            encryption_key = base64.urlsafe_b64encode(derived).decode()

        try:
            self._fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise ValueError(
                "Invalid encryption key. Generate a valid Fernet key with: "
                "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty string")

        encrypted = self._fernet.encrypt(plaintext.encode('utf-8'))
        return encrypted.decode('utf-8')

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string.

        Args:
            ciphertext: Base64-encoded encrypted data

        Returns:
            Original plaintext string

        Raises:
            ValueError: If decryption fails (invalid key or corrupted data)
        """
        if not ciphertext:
            raise ValueError("Cannot decrypt empty string")

        try:
            decrypted = self._fernet.decrypt(ciphertext.encode('utf-8'))
            return decrypted.decode('utf-8')
        except InvalidToken:
            raise ValueError("Decryption failed - invalid key or corrupted data")


# Singleton instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get or create the encryption service singleton."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def encrypt_secret(plaintext: str) -> str:
    """Convenience function to encrypt a secret."""
    return get_encryption_service().encrypt(plaintext)


def decrypt_secret(ciphertext: str) -> str:
    """Convenience function to decrypt a secret."""
    return get_encryption_service().decrypt(ciphertext)
