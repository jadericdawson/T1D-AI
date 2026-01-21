"""Test Gluroo Push - Verify we can push treatments to Gluroo via Nightscout API"""
import asyncio
import os
import sys
import subprocess
import json
import hashlib
import base64
from datetime import datetime, timezone, timedelta

# Get secrets from Azure - MUST be done before any imports that use config
def get_azure_setting(name):
    result = subprocess.run(
        ["az", "webapp", "config", "appsettings", "list", "--name", "t1d-ai",
         "--resource-group", "rg-knowledge2ai-eastus", "--query", f"[?name=='{name}'].value", "-o", "tsv"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Set environment for the app BEFORE importing any modules
os.environ["COSMOS_ENDPOINT"] = get_azure_setting("COSMOS_ENDPOINT")
os.environ["COSMOS_KEY"] = get_azure_setting("COSMOS_KEY")
os.environ["COSMOS_DATABASE"] = get_azure_setting("COSMOS_DATABASE") or "T1D-AI-DB"
os.environ["GPT41_ENDPOINT"] = get_azure_setting("GPT41_ENDPOINT") or "https://placeholder.openai.azure.com"
os.environ["AZURE_OPENAI_KEY"] = get_azure_setting("AZURE_OPENAI_KEY") or "placeholder"
os.environ["STORAGE_ACCOUNT_URL"] = get_azure_setting("STORAGE_ACCOUNT_URL") or "https://placeholder.blob.core.windows.net"
os.environ["STORAGE_CONNECTION_STRING"] = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING") or "DefaultEndpointsProtocol=https;AccountName=placeholder"
jwt_secret = get_azure_setting("JWT_SECRET_KEY")
os.environ["JWT_SECRET_KEY"] = jwt_secret if jwt_secret else "placeholder-jwt-secret-key-for-testing"
print(f"JWT Secret loaded: {'Yes (' + str(len(jwt_secret)) + ' chars)' if jwt_secret else 'No'}")

# Create Fernet key from JWT secret (like encryption.py does)
from cryptography.fernet import Fernet
derived = hashlib.sha256(os.environ["JWT_SECRET_KEY"].encode()).digest()
encryption_key = base64.urlsafe_b64encode(derived)
fernet = Fernet(encryption_key)

def decrypt_api_secret(ciphertext: str) -> str:
    """Decrypt API secret using derived key."""
    return fernet.decrypt(ciphertext.encode('utf-8')).decode('utf-8')

# Now import the modules
sys.path.insert(0, "src")

from services.gluroo_service import GlurooService
from database.cosmos_client import get_cosmos_manager

async def test_gluroo_push():
    """Test pushing a treatment to Gluroo"""
    try:
        # Get CosmosDB manager
        manager = get_cosmos_manager()
        await manager.initialize_containers()

        # Query all datasources directly
        print("Looking for Gluroo datasource...")
        container = manager.get_container('datasources')
        query = "SELECT * FROM c WHERE c.type = 'gluroo'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            print("No Gluroo datasource found!")
            return

        # Use the first Gluroo datasource
        ds = items[0]
        print(f"Found Gluroo datasource:")
        print(f"  - userId: {ds.get('userId')}")
        print(f"  - type: {ds.get('type')}")

        # Get credentials - handle both structures (credentials or config)
        creds = ds.get('credentials') or ds.get('config') or {}
        url = creds.get('url')
        api_secret_encrypted = creds.get('apiSecretEncrypted') or creds.get('apiSecret')

        print(f"  - URL: {url}")

        if not url:
            print("ERROR: No URL found in datasource!")
            return

        if not api_secret_encrypted:
            print("ERROR: No API secret found in datasource!")
            return

        # Decrypt the API secret
        try:
            api_secret = decrypt_api_secret(api_secret_encrypted)
            print("  - API Secret: [decrypted successfully]")
        except Exception as e:
            # Try using it as plain text (might not be encrypted)
            api_secret = api_secret_encrypted
            print(f"  - API Secret: [using as plain text, decrypt failed: {e}]")

        # Create Gluroo service
        print(f"\nConnecting to Gluroo: {url}")
        service = GlurooService(
            base_url=url,
            api_secret=api_secret
        )

        # Test connection first
        success, message = await service.test_connection()
        print(f"Connection test: {message}")

        if not success:
            print("Cannot connect to Gluroo, aborting push test")
            return

        # Try to push a small test carb entry (1g carb, backdated by 1 hour so it doesn't affect current data)
        test_time = datetime.now(timezone.utc) - timedelta(hours=1)

        print(f"\n--- Testing CARBS push ---")
        print(f"Pushing: 1g carbs at {test_time.isoformat()} with note 'T1D-AI Test - please delete'")

        success, message, result = await service.push_treatment(
            treatment_type="carbs",
            value=1,  # 1 gram - minimal test
            timestamp=test_time,
            notes="T1D-AI Test - please delete"
        )

        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {message}")
        if result:
            print(f"Response: {json.dumps(result, indent=2, default=str)}")

        if success:
            print("\n✅ Gluroo push WORKS! Treatments can be synced to Gluroo.")
        else:
            print("\n❌ Gluroo push FAILED. The API may be read-only or require different permissions.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gluroo_push())
