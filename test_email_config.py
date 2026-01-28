#!/usr/bin/env python3
"""Test email configuration and send test email."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from services.email_service import get_email_service
    from config import get_settings

    settings = get_settings()
    email_service = get_email_service()

    print("=" * 80)
    print("EMAIL SERVICE CONFIGURATION TEST")
    print("=" * 80)

    print(f"\n1. Configuration:")
    print(f"   Connection string: {'SET' if settings.azure_communication_connection_string else 'NOT SET'}")
    print(f"   Sender address: {settings.email_sender_address or 'NOT SET'}")
    print(f"   Frontend URL: {settings.frontend_url}")
    print(f"   Is configured: {email_service.is_configured}")

    if email_service.is_configured:
        print(f"\n2. Testing email send to jadericdawson@gmail.com...")
        try:
            success = await email_service.send_email(
                to_email="jadericdawson@gmail.com",
                subject="T1D-AI Email Test",
                html_content="<h1>Email Service Works!</h1><p>This is a test email from T1D-AI.</p>",
                plain_text="Email Service Works! This is a test email from T1D-AI."
            )

            if success:
                print(f"   ✅ Email sent successfully!")
            else:
                print(f"   ❌ Email send returned False")

        except Exception as e:
            print(f"   ❌ Exception during send: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Email service not configured - cannot send emails")
        print(f"   Required environment variables:")
        print(f"   - AZURE_COMMUNICATION_CONNECTION_STRING")
        print(f"   - EMAIL_SENDER_ADDRESS")

if __name__ == "__main__":
    asyncio.run(main())
