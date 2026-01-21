#!/usr/bin/env python3
"""
Azure ML Workspace Setup Script for T1D-AI

Creates and configures an Azure ML Workspace for MLflow tracking.
Estimated cost: ~$5-10/month for tracking only (storage + container registry).

Prerequisites:
- Azure CLI installed and logged in (`az login`)
- Azure subscription with contributor access

Usage:
    python scripts/setup_azure_ml.py

After running, add these to your .env or Azure App Service settings:
    AZURE_ML_WORKSPACE_NAME=t1d-ai-ml
    AZURE_ML_RESOURCE_GROUP=rg-knowledge2ai-eastus
    AZURE_ML_SUBSCRIPTION_ID=<your-subscription-id>
    AZURE_ML_REGION=eastus
"""
import subprocess
import sys
import json
import os

# Configuration
WORKSPACE_NAME = "t1d-ai-ml"
RESOURCE_GROUP = "rg-knowledge2ai-eastus"
LOCATION = "eastus"
SKU = "Basic"  # Basic is cheapest, sufficient for MLflow tracking


def run_az_command(args: list, capture_output: bool = True) -> dict | str | None:
    """Run an Azure CLI command and return the result."""
    cmd = ["az"] + args
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        if capture_output and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout.strip()
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr if e.stderr else e}")
        return None


def check_az_cli():
    """Check if Azure CLI is installed and logged in."""
    print("\n=== Checking Azure CLI ===")

    # Check if az is installed
    try:
        result = subprocess.run(["az", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Azure CLI is not installed. Please install it first:")
            print("  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
            return False
    except FileNotFoundError:
        print("Azure CLI is not installed. Please install it first:")
        print("  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        return False

    # Check if logged in
    account = run_az_command(["account", "show"])
    if not account:
        print("Not logged in to Azure. Please run: az login")
        return False

    print(f"Logged in as: {account.get('user', {}).get('name', 'Unknown')}")
    print(f"Subscription: {account.get('name', 'Unknown')} ({account.get('id', 'Unknown')})")

    return account


def check_resource_group():
    """Check if resource group exists."""
    print(f"\n=== Checking Resource Group: {RESOURCE_GROUP} ===")

    result = run_az_command([
        "group", "show",
        "--name", RESOURCE_GROUP
    ])

    if result:
        print(f"Resource group exists in {result.get('location', 'unknown')}")
        return True
    else:
        print(f"Resource group {RESOURCE_GROUP} not found.")
        return False


def check_workspace_exists():
    """Check if ML workspace already exists."""
    print(f"\n=== Checking ML Workspace: {WORKSPACE_NAME} ===")

    result = run_az_command([
        "ml", "workspace", "show",
        "--name", WORKSPACE_NAME,
        "--resource-group", RESOURCE_GROUP
    ])

    if result:
        print(f"Workspace already exists!")
        return result
    return None


def create_workspace():
    """Create the Azure ML Workspace."""
    print(f"\n=== Creating ML Workspace: {WORKSPACE_NAME} ===")

    result = run_az_command([
        "ml", "workspace", "create",
        "--name", WORKSPACE_NAME,
        "--resource-group", RESOURCE_GROUP,
        "--location", LOCATION,
        "--sku", SKU
    ])

    if result:
        print("Workspace created successfully!")
        return result
    else:
        print("Failed to create workspace.")
        return None


def get_mlflow_tracking_uri(subscription_id: str):
    """Generate the MLflow tracking URI for Azure ML."""
    return (
        f"azureml://{LOCATION}.api.azureml.ms/mlflow/v1.0/"
        f"subscriptions/{subscription_id}/"
        f"resourceGroups/{RESOURCE_GROUP}/"
        f"providers/Microsoft.MachineLearningServices/workspaces/"
        f"{WORKSPACE_NAME}"
    )


def print_env_config(subscription_id: str):
    """Print environment configuration for the app."""
    mlflow_uri = get_mlflow_tracking_uri(subscription_id)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)

    print("\nAdd these to your .env file or Azure App Service settings:")
    print("-" * 60)
    print(f"AZURE_ML_WORKSPACE_NAME={WORKSPACE_NAME}")
    print(f"AZURE_ML_RESOURCE_GROUP={RESOURCE_GROUP}")
    print(f"AZURE_ML_SUBSCRIPTION_ID={subscription_id}")
    print(f"AZURE_ML_REGION={LOCATION}")
    print("-" * 60)

    print("\nOr set the full tracking URI directly:")
    print("-" * 60)
    print(f"MLFLOW_TRACKING_URI={mlflow_uri}")
    print("-" * 60)

    print("\nExpected monthly cost: ~$5-10")
    print("- Container Registry (Basic): ~$5/month")
    print("- Storage: ~$0.02/GB (minimal for tracking)")
    print("- Key Vault operations: ~$0.03/10K ops")
    print("- Application Insights: Free tier (first 5GB)")

    print("\nTo verify setup, run:")
    print(f"  az ml workspace show --name {WORKSPACE_NAME} --resource-group {RESOURCE_GROUP}")


def main():
    print("=" * 60)
    print("Azure ML Workspace Setup for T1D-AI")
    print("=" * 60)

    # Check prerequisites
    account = check_az_cli()
    if not account:
        sys.exit(1)

    subscription_id = account.get("id")

    # Check resource group
    if not check_resource_group():
        print(f"\nPlease create the resource group first:")
        print(f"  az group create --name {RESOURCE_GROUP} --location {LOCATION}")
        sys.exit(1)

    # Check if workspace exists
    existing = check_workspace_exists()
    if existing:
        print("\nWorkspace already exists. Printing configuration...")
        print_env_config(subscription_id)
        return

    # Create workspace
    print("\nWorkspace not found. Creating new workspace...")
    print(f"  Name: {WORKSPACE_NAME}")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Location: {LOCATION}")
    print(f"  SKU: {SKU}")

    response = input("\nProceed with creation? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    workspace = create_workspace()
    if workspace:
        print_env_config(subscription_id)
    else:
        print("\nFailed to create workspace. Check the error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
