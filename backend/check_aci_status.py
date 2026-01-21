import os
import subprocess
import asyncio
import sys

os.environ["AZURE_SUBSCRIPTION_ID"] = "4151f5f9-5d87-40e7-8329-4921512a08ee"
os.environ["AZURE_RESOURCE_GROUP"] = "rg-knowledge2ai-eastus"

sys.path.insert(0, "src")
from services.aci_training_service import ACITrainingService

async def check_status():
    service = ACITrainingService()
    container_name = "t1d-train-jaderic_-20260109161951-cpu"
    
    status = await service.get_training_status(container_name)
    print(f"Container: {status['container_group_name']}")
    print(f"State: {status.get('state', 'Unknown')}")
    print(f"Provisioning: {status.get('provisioning_state', 'Unknown')}")
    if status.get('start_time'):
        print(f"Started: {status['start_time']}")
    if status.get('exit_code') is not None:
        print(f"Exit code: {status['exit_code']}")
    if status.get('error'):
        print(f"Error: {status['error']}")

asyncio.run(check_status())
