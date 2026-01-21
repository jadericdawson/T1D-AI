"""
Authentication API Routes.
Handles email/password and Microsoft OAuth authentication.
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

from auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    AuthResponse,
    UserAuthResponse,
    PasswordChangeRequest,
)
from auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from database.repositories import UserRepository
from services.email_service import get_email_service
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["authentication"])
security = HTTPBearer(auto_error=False)
settings = get_settings()


# ==================== Email Verification Schemas ====================

class VerifyEmailRequest(BaseModel):
    """Request to verify email with token."""
    token: str


class ResendVerificationRequest(BaseModel):
    """Request to resend verification email."""
    email: EmailStr

# Repository
user_repo = UserRepository()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Dependency to get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """
    Register a new user with email and password.
    Sends verification email - user must verify before full access.
    """
    try:
        # Check if user already exists
        existing = await user_repo.get_by_email(request.email)
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")

        # Generate verification token
        verification_token = secrets.token_urlsafe(32)
        verification_expires = datetime.utcnow() + timedelta(hours=24)

        # Create user with hashed password and verification token
        password_hash = get_password_hash(request.password)
        user = await user_repo.create(
            email=request.email,
            display_name=request.displayName,
            password_hash=password_hash,
            auth_provider="email",
            email_verified=False,
            email_verification_token=verification_token,
            email_verification_expires=verification_expires
        )

        # Send verification email (non-blocking - don't fail registration if email fails)
        email_service = get_email_service()
        try:
            await email_service.send_verification_email(
                to_email=user.email,
                display_name=user.displayName or user.email.split('@')[0],
                verification_token=verification_token
            )
            logger.info(f"Verification email sent to {user.email}")
        except Exception as email_error:
            logger.error(f"Failed to send verification email: {email_error}")
            # Continue with registration even if email fails

        # Generate tokens (user can still log in, but emailVerified will be false)
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = create_refresh_token(user.id)

        return AuthResponse(
            user=UserAuthResponse(
                id=user.id,
                email=user.email,
                displayName=user.displayName,
                createdAt=user.createdAt,
                emailVerified=user.emailVerified,
                onboardingCompleted=user.onboardingCompleted,
                isAdmin=getattr(user, 'isAdmin', False)
            ),
            tokens=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.
    """
    try:
        # Find user
        user = await user_repo.get_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Check password
        if not user.passwordHash:
            raise HTTPException(
                status_code=401,
                detail="This account uses Microsoft login. Please sign in with Microsoft."
            )

        if not verify_password(request.password, user.passwordHash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Update last login timestamp
        try:
            await user_repo.update(user.id, {"lastLoginAt": datetime.utcnow()})
        except Exception as e:
            logger.warning(f"Failed to update lastLoginAt: {e}")

        # Generate tokens
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = create_refresh_token(user.id)

        return AuthResponse(
            user=UserAuthResponse(
                id=user.id,
                email=user.email,
                displayName=user.displayName,
                createdAt=user.createdAt,
                emailVerified=getattr(user, 'emailVerified', True),  # Default True for existing users
                onboardingCompleted=getattr(user, 'onboardingCompleted', False),
                isAdmin=getattr(user, 'isAdmin', False)
            ),
            tokens=TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    payload = decode_access_token(request.refresh_token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    user = await user_repo.get_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Generate new tokens
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    new_refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=UserAuthResponse)
async def get_me(current_user=Depends(get_current_user)):
    """
    Get current authenticated user info.
    """
    return UserAuthResponse(
        id=current_user.id,
        email=current_user.email,
        displayName=current_user.displayName,
        createdAt=current_user.createdAt
    )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user=Depends(get_current_user)
):
    """
    Change password for authenticated user.
    """
    if not current_user.passwordHash:
        raise HTTPException(
            status_code=400,
            detail="Cannot change password for OAuth accounts"
        )

    if not verify_password(request.current_password, current_user.passwordHash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    new_hash = get_password_hash(request.new_password)
    await user_repo.update(current_user.id, {"passwordHash": new_hash})

    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(current_user=Depends(get_current_user)):
    """
    Logout - client should discard tokens.
    In a more complete implementation, we'd blacklist the token.
    """
    return {"message": "Logged out successfully"}


# ==================== Microsoft OAuth Routes ====================

@router.get("/microsoft/login-url")
async def get_microsoft_login_url():
    """
    Get the Microsoft OAuth login URL.
    Redirect user to this URL to initiate Microsoft login.
    """
    import msal

    if not settings.microsoft_client_id:
        raise HTTPException(
            status_code=503,
            detail="Microsoft login is not configured"
        )

    # Use MSAL to generate the authorization URL
    # This ensures scopes are consistent between auth and token exchange
    authority = f"https://login.microsoftonline.com/{settings.microsoft_tenant_id}"
    app = msal.ConfidentialClientApplication(
        settings.microsoft_client_id,
        authority=authority,
        client_credential=settings.microsoft_client_secret,
    )

    # MSAL automatically adds openid, profile, offline_access
    # Just specify the resource scope we need
    auth_url = app.get_authorization_request_url(
        scopes=["User.Read"],
        redirect_uri=settings.microsoft_redirect_uri
    )

    return {"login_url": auth_url}


@router.get("/microsoft/callback")
async def microsoft_callback(code: str, state: Optional[str] = None, error: Optional[str] = None, error_description: Optional[str] = None):
    """
    Handle Microsoft OAuth callback.
    Exchange authorization code for tokens and create/login user.
    """
    from fastapi.responses import HTMLResponse

    # Handle OAuth errors
    if error:
        logger.error(f"Microsoft OAuth error: {error} - {error_description}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Login Failed</title></head>
        <body>
        <script>
            if (window.opener) {{
                window.opener.postMessage({{ type: 'auth-error', error: '{error}' }}, '*');
                window.close();
            }} else {{
                window.location.href = '/login?error={error}';
            }}
        </script>
        <p>Login failed: {error_description or error}</p>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)

    try:
        import msal

        if not settings.microsoft_client_id:
            raise HTTPException(status_code=503, detail="Microsoft login not configured")

        # Create MSAL app for standard Azure AD
        authority = f"https://login.microsoftonline.com/{settings.microsoft_tenant_id}"
        app = msal.ConfidentialClientApplication(
            settings.microsoft_client_id,
            authority=authority,
            client_credential=settings.microsoft_client_secret,
        )

        # Exchange code for token
        # Use same scope format as in get_authorization_request_url
        # MSAL automatically handles OIDC scopes
        result = app.acquire_token_by_authorization_code(
            code,
            scopes=["User.Read"],
            redirect_uri=settings.microsoft_redirect_uri
        )

        if "error" in result:
            logger.error(f"Microsoft auth error: {result}")
            raise HTTPException(status_code=401, detail=f"Microsoft authentication failed: {result.get('error_description', result.get('error'))}")

        # Get user info from id_token claims
        id_token_claims = result.get("id_token_claims", {})
        microsoft_id = id_token_claims.get("oid") or id_token_claims.get("sub")
        email = id_token_claims.get("preferred_username") or id_token_claims.get("email")
        name = id_token_claims.get("name")

        if not email:
            # Try to get email from Microsoft Graph API
            access_token_ms = result.get("access_token")
            if access_token_ms:
                import httpx
                async with httpx.AsyncClient() as client:
                    graph_response = await client.get(
                        "https://graph.microsoft.com/v1.0/me",
                        headers={"Authorization": f"Bearer {access_token_ms}"}
                    )
                    if graph_response.status_code == 200:
                        graph_data = graph_response.json()
                        email = graph_data.get("mail") or graph_data.get("userPrincipalName")
                        name = name or graph_data.get("displayName")

        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Microsoft")

        # Find or create user
        user = await user_repo.get_by_email(email)

        if not user:
            # Create new user
            user = await user_repo.create(
                email=email,
                display_name=name,
                auth_provider="microsoft",
                microsoft_id=microsoft_id
            )
        elif user.microsoftId != microsoft_id:
            # Link/update Microsoft account
            await user_repo.update(user.id, {
                "microsoftId": microsoft_id,
            })
            user = await user_repo.get_by_id(user.id)

        # Generate our own tokens
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = create_refresh_token(user.id)

        # Calculate actual token expiration in seconds
        token_expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60

        # Return HTML that sends tokens to the frontend
        # Store in Zustand's expected format for persistence
        display_name_js = (user.displayName or '').replace('"', '\\"').replace("'", "\\'")
        is_admin = "true" if getattr(user, 'isAdmin', False) else "false"
        # Microsoft OAuth users are always verified (Microsoft verified their email)
        email_verified = "true"
        onboarding_completed = "true" if getattr(user, 'onboardingCompleted', False) else "false"

        # Also update the database to mark Microsoft users as verified
        if not getattr(user, 'emailVerified', False):
            try:
                await user_repo.update(user.id, {"emailVerified": True})
            except Exception as e:
                logger.warning(f"Failed to update emailVerified for Microsoft user: {e}")

        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Login Successful</title></head>
        <body>
        <script>
            const authData = {{
                access_token: "{access_token}",
                refresh_token: "{refresh_token}",
                user: {{
                    id: "{user.id}",
                    email: "{user.email}",
                    displayName: "{display_name_js}",
                    isAdmin: {is_admin},
                    emailVerified: {email_verified},
                    onboardingCompleted: {onboarding_completed}
                }}
            }};

            // Format for Zustand persist storage
            const zustandData = {{
                state: {{
                    user: authData.user,
                    tokens: {{
                        accessToken: authData.access_token,
                        refreshToken: authData.refresh_token,
                        expiresIn: {token_expires_in}
                    }},
                    isAuthenticated: true
                }},
                version: 0
            }};

            // Try postMessage first (for popup flow)
            if (window.opener) {{
                try {{
                    window.opener.postMessage({{ type: 'auth-success', data: authData }}, '*');
                    window.close();
                }} catch (e) {{
                    console.error('postMessage failed:', e);
                }}
            }}

            // Always set localStorage as fallback and redirect to dashboard
            localStorage.setItem('t1d-ai-auth', JSON.stringify(zustandData));
            window.location.href = '/dashboard';
        </script>
        <p>Login successful! Redirecting...</p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Microsoft callback error: {e}")
        raise HTTPException(status_code=500, detail="Microsoft login failed")


# ==================== Email Verification Routes ====================

@router.post("/verify-email")
async def verify_email(request: VerifyEmailRequest):
    """
    Verify email address using the token sent via email.
    """
    try:
        # Find user by verification token
        user = await user_repo.get_by_verification_token(request.token)

        if not user:
            raise HTTPException(status_code=400, detail="Invalid verification token")

        # Check if token has expired
        if user.emailVerificationExpires and datetime.utcnow() > user.emailVerificationExpires:
            raise HTTPException(status_code=400, detail="Verification token has expired")

        # Mark email as verified
        await user_repo.update(user.id, {
            "emailVerified": True,
            "emailVerificationToken": None,
            "emailVerificationExpires": None
        })

        # Send welcome email
        email_service = get_email_service()
        try:
            await email_service.send_welcome_email(
                to_email=user.email,
                display_name=user.displayName or user.email.split('@')[0]
            )
        except Exception as email_error:
            logger.error(f"Failed to send welcome email: {email_error}")

        return {
            "message": "Email verified successfully",
            "email": user.email
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification error: {e}")
        raise HTTPException(status_code=500, detail="Email verification failed")


@router.post("/resend-verification")
async def resend_verification(request: ResendVerificationRequest):
    """
    Resend verification email to user.
    """
    try:
        user = await user_repo.get_by_email(request.email)

        if not user:
            # Don't reveal if email exists
            return {"message": "If this email is registered, a verification link has been sent"}

        if user.emailVerified:
            return {"message": "Email is already verified"}

        # Generate new verification token
        verification_token = secrets.token_urlsafe(32)
        verification_expires = datetime.utcnow() + timedelta(hours=24)

        await user_repo.update(user.id, {
            "emailVerificationToken": verification_token,
            "emailVerificationExpires": verification_expires
        })

        # Send verification email
        email_service = get_email_service()
        await email_service.send_verification_email(
            to_email=user.email,
            display_name=user.displayName or user.email.split('@')[0],
            verification_token=verification_token
        )

        return {"message": "Verification email sent"}

    except Exception as e:
        logger.error(f"Resend verification error: {e}")
        # Don't reveal errors to prevent email enumeration
        return {"message": "If this email is registered, a verification link has been sent"}
