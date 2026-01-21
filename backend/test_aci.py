"""Test ACI Training Service"""
import os
import sys
import asyncio
import subprocess

# Set environment variables from Azure
os.environ["AZURE_SUBSCRIPTION_ID"] = "4151f5f9-5d87-40e7-8329-4921512a08ee"
os.environ["AZURE_RESOURCE_GROUP"] = "rg-knowledge2ai-eastus"
os.environ["ACR_SERVER"] = "knowledge2aiacr.azurecr.io"
os.environ["ACR_USERNAME"] = "knowledge2aiacr"

def get_azure_setting(name):
    """Get app setting from Azure"""
    result = subprocess.run(
        ["az", "webapp", "config", "appsettings", "list", "--name", "t1d-ai", 
         "--resource-group", "rg-knowledge2ai-eastus", "--query", f"[?name=='{name}'].value", "-o", "tsv"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Get all required secrets from Azure
print("Fetching secrets from Azure...")
os.environ["ACR_PASSWORD"] = subprocess.run(
    ["az", "acr", "credential", "show", "--name", "knowledge2aiacr", "--query", "passwords[0].value", "-o", "tsv"],
    capture_output=True, text=True
).stdout.strip()

os.environ["COSMOS_KEY"] = get_azure_setting("COSMOS_KEY")
os.environ["COSMOS_ENDPOINT"] = get_azure_setting("COSMOS_ENDPOINT")
os.environ["COSMOS_DATABASE"] = get_azure_setting("COSMOS_DATABASE") or "T1D-AI-DB"
os.environ["STORAGE_ACCOUNT_URL"] = get_azure_setting("STORAGE_ACCOUNT_URL")
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING")
os.environ["GPT41_ENDPOINT"] = get_azure_setting("GPT41_ENDPOINT")
os.environ["AZURE_OPENAI_KEY"] = get_azure_setting("AZURE_OPENAI_KEY")
os.environ["JWT_SECRET_KEY"] = get_azure_setting("JWT_SECRET_KEY")

print(f"  COSMOS_ENDPOINT: {os.environ.get('COSMOS_ENDPOINT', '')[:50]}...")
print(f"  COSMOS_KEY: {'set' if os.environ.get('COSMOS_KEY') else 'NOT SET'} ({len(os.environ.get('COSMOS_KEY', ''))} chars)")
print(f"  AZURE_STORAGE_CONNECTION_STRING: {'set' if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') else 'NOT SET'}")
print(f"  AZURE_OPENAI_KEY: {'set' if os.environ.get('AZURE_OPENAI_KEY') else 'NOT SET'}")

sys.path.insert(0, "src")

from services.aci_training_service import ACITrainingService

async def test_aci():
    service = ACITrainingService()
    test_user_id = "jaderic_gluroo"
    
    print(f"\nStarting ACI training test for user: {test_user_id}")
    print(f"GPU will be tried first, with CPU fallback")
    
    try:
        result = await service.start_training(
            user_id=test_user_id,
            model_type="tft",
            job_id="test-job-003",
            use_gpu=True
        )
        print(f"\n✅ ACI container started successfully!")
        print(f"Container group: {result['container_group_name']}")
        print(f"GPU: {result.get('use_gpu', 'unknown')}")
        return result
    except Exception as e:
        print(f"\n❌ Failed to start ACI container: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_aci())
