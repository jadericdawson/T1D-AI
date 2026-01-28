#!/usr/bin/env python3
"""Send verification email to Denise and copy to Jaderic for testing."""
import asyncio
import sys
import os
import secrets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from database.repositories import UserRepository
    from services.email_service import get_email_service

    user_repo = UserRepository()
    email_service = get_email_service()

    # Find Denise
    container = user_repo.container
    query = "SELECT * FROM c WHERE c.email = 'dadawson62@gmail.com'"
    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    if not results:
        print("❌ Denise not found")
        return

    user_doc = results[0]
    print(f"✓ Found: {user_doc['email']}")
    print(f"  Email verified: {user_doc.get('emailVerified', False)}")

    # Generate or use existing verification token
    verification_token = user_doc.get('emailVerificationToken')
    if not verification_token:
        verification_token = secrets.token_urlsafe(32)
        user_doc['emailVerificationToken'] = verification_token
        container.upsert_item(user_doc)
        print(f"  Generated new token")

    print(f"\n📧 Sending verification emails...")

    # Send to Denise
    try:
        await email_service.send_verification_email(
            to_email="dadawson62@gmail.com",
            display_name="Denise",
            verification_token=verification_token
        )
        print(f"✅ Sent to dadawson62@gmail.com")
    except Exception as e:
        print(f"❌ Failed to send to Denise: {e}")

    # Send copy to Jaderic for testing
    try:
        await email_service.send_verification_email(
            to_email="jadericdawson@gmail.com",
            display_name="Denise (copy for testing)",
            verification_token=verification_token
        )
        print(f"✅ Sent copy to jadericdawson@gmail.com")
    except Exception as e:
        print(f"❌ Failed to send to Jaderic: {e}")

    print(f"\n📋 Verification link:")
    from config import get_settings
    settings = get_settings()
    print(f"{settings.frontend_url}/verify-email?token={verification_token}")

if __name__ == "__main__":
    asyncio.run(main())
