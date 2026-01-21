import os
import subprocess

def get_azure_setting(name):
    result = subprocess.run(
        ["az", "webapp", "config", "appsettings", "list", "--name", "t1d-ai", 
         "--resource-group", "rg-knowledge2ai-eastus", "--query", f"[?name=='{name}'].value", "-o", "tsv"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# Set all env vars
os.environ["COSMOS_ENDPOINT"] = get_azure_setting("COSMOS_ENDPOINT")
os.environ["COSMOS_KEY"] = get_azure_setting("COSMOS_KEY")
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING")
os.environ["GPT41_ENDPOINT"] = get_azure_setting("GPT41_ENDPOINT") 
os.environ["GPT41_API_KEY"] = get_azure_setting("GPT41_API_KEY")
os.environ["JWT_SECRET_KEY"] = get_azure_setting("JWT_SECRET_KEY")

print("Environment check:")
print(f"  COSMOS_ENDPOINT: {'set' if os.getenv('COSMOS_ENDPOINT') else 'NOT SET'} ({len(os.getenv('COSMOS_ENDPOINT', ''))} chars)")
print(f"  COSMOS_KEY: {'set' if os.getenv('COSMOS_KEY') else 'NOT SET'} ({len(os.getenv('COSMOS_KEY', ''))} chars)")
print(f"  AZURE_STORAGE_CONNECTION_STRING: {'set' if os.getenv('AZURE_STORAGE_CONNECTION_STRING') else 'NOT SET'}")
print(f"  GPT41_ENDPOINT: {'set' if os.getenv('GPT41_ENDPOINT') else 'NOT SET'}")
print(f"  GPT41_API_KEY: {'set' if os.getenv('GPT41_API_KEY') else 'NOT SET'}")
print(f"  JWT_SECRET_KEY: {'set' if os.getenv('JWT_SECRET_KEY') else 'NOT SET'}")
