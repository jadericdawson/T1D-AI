#!/usr/bin/env python3
"""
Set up the emrys datasource in CosmosDB for ongoing Gluroo sync.
"""
import base64
import hashlib
import logging
import os
from datetime import datetime, timezone

from azure.cosmos import CosmosClient
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gluroo credentials for emrys
GLUROO_URL = "https://81ca.ns.gluroo.com"
GLUROO_API_SECRET = "81ca21b8-9002-4d24-8492-d7546ff6fade"


def get_fernet() -> Fernet:
    """Get Fernet instance for encryption."""
    key = os.environ.get("ENCRYPTION_MASTER_KEY")
    if not key:
        jwt_secret = os.environ.get("JWT_SECRET_KEY", "default-secret")
        derived = hashlib.sha256(jwt_secret.encode()).digest()
        key = base64.urlsafe_b64encode(derived).decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt an API secret."""
    fernet = get_fernet()
    encrypted = fernet.encrypt(plaintext.encode('utf-8'))
    return encrypted.decode('utf-8')


def main():
    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")

    if not cosmos_endpoint or not cosmos_key:
        logger.error("COSMOS_ENDPOINT and COSMOS_KEY required")
        return

    client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
    database = client.get_database_client("T1D-AI-DB")
    container = database.get_container_client("datasources")

    # Encrypt the API secret
    encrypted_secret = encrypt_secret(GLUROO_API_SECRET)

    # Create datasource document
    datasource = {
        "id": "emrys_gluroo",
        "userId": "emrys",
        "sourceType": "gluroo",
        "status": "connected",
        "credentials": {
            "url": GLUROO_URL,
            "apiSecretEncrypted": encrypted_secret
        },
        "lastSyncAt": datetime.now(timezone.utc).isoformat(),
        "recordsSynced": 0,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat()
    }

    # Upsert to CosmosDB
    container.upsert_item(datasource)
    logger.info(f"Created/updated datasource for emrys: {datasource['id']}")
    logger.info("Ongoing sync is now configured!")


if __name__ == "__main__":
    main()
