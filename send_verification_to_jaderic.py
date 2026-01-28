#!/usr/bin/env python3
"""Find Denise and send verification email to jadericdawson@gmail.com for testing."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from database.repositories import UserRepository
    from services.email_service import get_email_service

    user_repo = UserRepository()
    email_service = get_email_service()

    print("Searching for users with 'denise' in email...")

    try:
        container = user_repo.container
        query = "SELECT * FROM c WHERE CONTAINS(LOWER(c.email), 'denise')"
        results = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        if not results:
            print("❌ No user found with 'denise' in email")
            return

        user_doc = results[0]
        print(f"\n✓ Found user: {user_doc['email']}")
        print(f"  ID: {user_doc['id']}")
        print(f"  Email verified: {user_doc.get('emailVerified', False)}")

        # Get or create verification token
        verification_token = user_doc.get('emailVerificationToken')
        if not verification_token:
            import secrets
            verification_token = secrets.token_urlsafe(32)
            print(f"  Generated new token: {verification_token[:20]}...")

        # Send verification email to jadericdawson@gmail.com for testing
        test_email = "jadericdawson@gmail.com"
        print(f"\n📧 Sending verification email to: {test_email}")

        try:
            result = await email_service.send_verification_email(
                to_email=test_email,
                display_name=user_doc.get('displayName') or user_doc['email'].split('@')[0],
                verification_token=verification_token
            )
            print(f"✅ Email sent successfully!")
            print(f"   Original user: {user_doc['email']}")
            print(f"   Sent to: {test_email}")
            print(f"   Verification token: {verification_token[:20]}...")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
