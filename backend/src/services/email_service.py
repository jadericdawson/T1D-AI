"""
Email Service using Azure Communication Services
Handles verification emails, welcome emails, and other transactional emails.
"""
import logging
from typing import Optional
from azure.communication.email import EmailClient
from config import get_settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via Azure Communication Services."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[EmailClient] = None

    @property
    def client(self) -> Optional[EmailClient]:
        """Lazy initialization of email client."""
        if self._client is None and self.settings.azure_communication_connection_string:
            try:
                self._client = EmailClient.from_connection_string(
                    self.settings.azure_communication_connection_string
                )
            except Exception as e:
                logger.error(f"Failed to initialize email client: {e}")
        return self._client

    @property
    def is_configured(self) -> bool:
        """Check if email service is properly configured."""
        return bool(
            self.settings.azure_communication_connection_string
            and self.settings.email_sender_address
        )

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        plain_text: Optional[str] = None
    ) -> bool:
        """
        Send an email using Azure Communication Services.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML body content
            plain_text: Optional plain text body (for email clients that don't support HTML)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.is_configured:
            logger.warning("Email service not configured - skipping email send")
            return False

        if not self.client:
            logger.error("Email client not initialized")
            return False

        try:
            message = {
                "senderAddress": self.settings.email_sender_address,
                "recipients": {
                    "to": [{"address": to_email}]
                },
                "content": {
                    "subject": subject,
                    "html": html_content,
                }
            }

            # Add plain text if provided
            if plain_text:
                message["content"]["plainText"] = plain_text

            # Send email (synchronous operation)
            poller = self.client.begin_send(message)
            result = poller.result()

            if result["status"] == "Succeeded":
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Email send failed: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    async def send_verification_email(
        self,
        to_email: str,
        display_name: str,
        verification_token: str
    ) -> bool:
        """
        Send email verification email with verification link.

        Args:
            to_email: Recipient email address
            display_name: User's display name for personalization
            verification_token: Token to include in verification link
        """
        verification_url = f"{self.settings.frontend_url}/verify-email?token={verification_token}"

        subject = "Verify your T1D-AI account"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 20px 0; }}
                .logo {{ font-size: 32px; font-weight: bold; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .button {{ display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
                .button:hover {{ opacity: 0.9; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 14px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">T1D-AI</div>
                </div>

                <p>Hi {display_name or 'there'},</p>

                <p>Welcome to T1D-AI! Please verify your email address by clicking the button below:</p>

                <p style="text-align: center;">
                    <a href="{verification_url}" class="button">Verify Email Address</a>
                </p>

                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; background: #f3f4f6; padding: 10px; border-radius: 4px; font-size: 14px;">
                    {verification_url}
                </p>

                <p>This link expires in 24 hours.</p>

                <p>If you didn't create this account, you can safely ignore this email.</p>

                <div class="footer">
                    <p>- The T1D-AI Team</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        plain_text = f"""
Hi {display_name or 'there'},

Welcome to T1D-AI! Please verify your email address by clicking the link below:

{verification_url}

This link expires in 24 hours.

If you didn't create this account, you can safely ignore this email.

- The T1D-AI Team
        """

        return await self.send_email(to_email, subject, html_content, plain_text)

    async def send_welcome_email(
        self,
        to_email: str,
        display_name: str
    ) -> bool:
        """
        Send welcome email after successful verification.

        Args:
            to_email: Recipient email address
            display_name: User's display name for personalization
        """
        dashboard_url = f"{self.settings.frontend_url}/dashboard"

        subject = "Welcome to T1D-AI - Let's get started!"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 20px 0; }}
                .logo {{ font-size: 32px; font-weight: bold; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .button {{ display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
                .steps {{ background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .step {{ display: flex; align-items: center; margin: 10px 0; }}
                .step-number {{ background: #06b6d4; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px; flex-shrink: 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 14px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">T1D-AI</div>
                </div>

                <p>Hi {display_name or 'there'},</p>

                <p>Your email is verified! Welcome to T1D-AI - your AI-powered Type 1 Diabetes companion.</p>

                <div class="steps">
                    <h3 style="margin-top: 0;">Here's what's next:</h3>
                    <div class="step">
                        <span class="step-number">1</span>
                        <span>Connect your CGM (Gluroo, Dexcom, etc.)</span>
                    </div>
                    <div class="step">
                        <span class="step-number">2</span>
                        <span>Set up your insulin parameters</span>
                    </div>
                    <div class="step">
                        <span class="step-number">3</span>
                        <span>Let the AI learn your patterns</span>
                    </div>
                </div>

                <p style="text-align: center;">
                    <a href="{dashboard_url}" class="button">Go to Dashboard</a>
                </p>

                <p>Need help? Check out our setup guide or reply to this email.</p>

                <div class="footer">
                    <p>- The T1D-AI Team</p>
                </div>
            </div>
        </body>
        </html>
        """

        plain_text = f"""
Hi {display_name or 'there'},

Your email is verified! Welcome to T1D-AI - your AI-powered Type 1 Diabetes companion.

Here's what's next:

1. Connect your CGM (Gluroo, Dexcom, etc.)
2. Set up your insulin parameters
3. Let the AI learn your patterns

Go to Dashboard: {dashboard_url}

Need help? Check out our setup guide or reply to this email.

- The T1D-AI Team
        """

        return await self.send_email(to_email, subject, html_content, plain_text)

    async def send_password_reset_email(
        self,
        to_email: str,
        display_name: str,
        reset_token: str
    ) -> bool:
        """
        Send password reset email.

        Args:
            to_email: Recipient email address
            display_name: User's display name for personalization
            reset_token: Token for password reset link
        """
        reset_url = f"{self.settings.frontend_url}/reset-password?token={reset_token}"

        subject = "Reset your T1D-AI password"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 20px 0; }}
                .logo {{ font-size: 32px; font-weight: bold; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .button {{ display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 14px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">T1D-AI</div>
                </div>

                <p>Hi {display_name or 'there'},</p>

                <p>We received a request to reset your password. Click the button below to create a new password:</p>

                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Reset Password</a>
                </p>

                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; background: #f3f4f6; padding: 10px; border-radius: 4px; font-size: 14px;">
                    {reset_url}
                </p>

                <p>This link expires in 1 hour.</p>

                <p>If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>

                <div class="footer">
                    <p>- The T1D-AI Team</p>
                </div>
            </div>
        </body>
        </html>
        """

        plain_text = f"""
Hi {display_name or 'there'},

We received a request to reset your password. Click the link below to create a new password:

{reset_url}

This link expires in 1 hour.

If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.

- The T1D-AI Team
        """

        return await self.send_email(to_email, subject, html_content, plain_text)

    async def send_sharing_invitation(
        self,
        to_email: str,
        owner_name: str,
        owner_email: str,
        role: str,
        invitation_token: str
    ) -> bool:
        """
        Send invitation email to share glucose data access.

        Args:
            to_email: Recipient email address
            owner_name: Name of the person sharing their data
            owner_email: Email of the person sharing their data
            role: Role being granted (viewer, caregiver, admin)
            invitation_token: Token to include in accept link
        """
        accept_url = f"{self.settings.frontend_url}/accept-invite?token={invitation_token}"

        role_descriptions = {
            "viewer": "view glucose readings, treatments, predictions, and insights (read-only)",
            "caregiver": "view all data plus log treatments and receive alerts",
            "admin": "full access including settings and all permissions"
        }
        role_description = role_descriptions.get(role.lower(), "view their data")

        subject = f"{owner_name or owner_email} wants to share their T1D-AI data with you"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 20px 0; }}
                .logo {{ font-size: 32px; font-weight: bold; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .button {{ display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
                .button:hover {{ opacity: 0.9; }}
                .info-box {{ background: #f8fafc; padding: 16px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #06b6d4; }}
                .role-badge {{ display: inline-block; padding: 4px 12px; background: linear-gradient(135deg, #06b6d4, #8b5cf6); color: white; border-radius: 4px; font-size: 14px; font-weight: 600; text-transform: capitalize; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 14px; color: #6b7280; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">T1D-AI</div>
                </div>

                <p>Hi there,</p>

                <p><strong>{owner_name or owner_email}</strong> has invited you to access their glucose data on T1D-AI.</p>

                <div class="info-box">
                    <p style="margin: 0;"><strong>Access Level:</strong> <span class="role-badge">{role}</span></p>
                    <p style="margin: 8px 0 0 0;">You'll be able to {role_description}.</p>
                </div>

                <p style="text-align: center;">
                    <a href="{accept_url}" class="button">Accept Invitation</a>
                </p>

                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; background: #f3f4f6; padding: 10px; border-radius: 4px; font-size: 14px;">
                    {accept_url}
                </p>

                <p>This invitation expires in 7 days.</p>

                <p>If you don't have a T1D-AI account yet, you'll be prompted to create one when you accept the invitation.</p>

                <p>If you weren't expecting this invitation, you can safely ignore this email.</p>

                <div class="footer">
                    <p>- The T1D-AI Team</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        plain_text = f"""
Hi there,

{owner_name or owner_email} has invited you to access their glucose data on T1D-AI.

Access Level: {role.capitalize()}
You'll be able to {role_description}.

Accept the invitation here:
{accept_url}

This invitation expires in 7 days.

If you don't have a T1D-AI account yet, you'll be prompted to create one when you accept the invitation.

If you weren't expecting this invitation, you can safely ignore this email.

- The T1D-AI Team
        """

        return await self.send_email(to_email, subject, html_content, plain_text)


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get singleton email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
