#!/usr/bin/env python3
"""
Trigger verification email from production Azure app.
Calls the deployed API which has email service configured.
"""
import asyncio
import httpx

async def main():
    # Production API base URL
    base_url = "https://t1d-ai.azurewebsites.net"

    print("Triggering verification email from production...")
    print("Target: dadawson62@gmail.com (Denise)")
    print("Copy to: jadericdawson@gmail.com (for testing)")

    # We need to call the backend directly with admin privileges
    # Since there's no public resend endpoint, let's simulate registration email sending
    # by accessing the email service through a test endpoint

    # Actually, let me call the production environment to execute our Python script
    import subprocess

    script = """
import asyncio
from services.email_service import get_email_service

async def send_emails():
    email_service = get_email_service()

    # Denise's verification token from database
    token = "RokwFBe_E6DxwAqigAI_TbIgrhWIhWlQR6XoavyYJxk"

    try:
        # Send to Denise
        await email_service.send_verification_email(
            to_email="dadawson62@gmail.com",
            display_name="Denise",
            verification_token=token
        )
        print("✅ Sent to dadawson62@gmail.com")
    except Exception as e:
        print(f"❌ Failed to send to Denise: {e}")

    try:
        # Send copy to Jaderic
        await email_service.send_verification_email(
            to_email="jadericdawson@gmail.com",
            display_name="Denise (Test Copy)",
            verification_token=token
        )
        print("✅ Sent to jadericdawson@gmail.com")
    except Exception as e:
        print(f"❌ Failed to send to Jaderic: {e}")

asyncio.run(send_emails())
"""

    print("\n⚠️  Need to run this script in Azure environment where email service is configured")
    print(f"\nScript content:\n{script}")

if __name__ == "__main__":
    asyncio.run(main())
