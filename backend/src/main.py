"""
T1D-AI Backend - FastAPI Application
Cloud-hosted Type 1 Diabetes management platform.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"

        # HSTS for production (commented for local dev)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/"]:
            return await call_next(request)

        # Clean old requests (older than 1 minute)
        current_time = time.time()
        self.request_counts[client_ip] = [
            t for t in self.request_counts[client_ip]
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."}
            )

        # Record this request
        self.request_counts[client_ip].append(current_time)

        return await call_next(request)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"ML Device: {settings.model_device}")

    # Initialize services on startup
    try:
        from database.cosmos_client import get_cosmos_manager
        cosmos = get_cosmos_manager()
        await cosmos.initialize_containers()
        logger.info("CosmosDB containers initialized")
    except Exception as e:
        logger.warning(f"CosmosDB initialization skipped: {e}")

    # Initialize ML prediction service
    try:
        from pathlib import Path
        from services.prediction_service import get_prediction_service

        # Check multiple locations for models
        models_dir = None
        possible_paths = [
            Path(settings.models_dir) if hasattr(settings, 'models_dir') else None,
            Path("/app/models"),
            Path("./models"),
            Path("./data/models"),
        ]
        for path in possible_paths:
            if path and path.exists():
                models_dir = path
                break

        pred_service = get_prediction_service(models_dir, settings.model_device)
        logger.info(
            f"Prediction service initialized - "
            f"LSTM: {pred_service.lstm_available}, "
            f"ISF: {pred_service.isf_available}"
        )
    except Exception as e:
        logger.warning(f"ML service initialization skipped: {e}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down T1D-AI...")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Cloud-hosted Type 1 Diabetes management platform with ML predictions",
    lifespan=lifespan
)

# Security middleware (order matters - last added runs first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

# CORS middleware - use configured origins
cors_origins = settings.cors_origins_list
if settings.debug:
    cors_origins.append("*")  # Allow all in debug mode

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Azure App Service."""
    return {"status": "healthy", "version": settings.app_version}


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies all dependencies are available."""
    checks = {"database": False, "ml_service": False}

    try:
        from database.cosmos_client import get_cosmos_manager
        cosmos = get_cosmos_manager()
        checks["database"] = cosmos._initialized
    except Exception:
        pass

    try:
        from services.prediction_service import get_prediction_service
        pred_service = get_prediction_service()
        checks["ml_service"] = pred_service.lstm_available or True  # Linear always available
    except Exception:
        pass

    all_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={"ready": all_ready, "checks": checks}
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs"
    }


# Import and register API routers
from api.v1 import glucose, treatments, datasources, predictions, calculations, users, insights, websocket

# REST API routers
app.include_router(glucose.router, prefix="/api/v1/glucose", tags=["Glucose"])
app.include_router(treatments.router, prefix="/api/v1/treatments", tags=["Treatments"])
app.include_router(datasources.router, prefix="/api/v1/datasources", tags=["Data Sources"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(calculations.router, prefix="/api/v1", tags=["Calculations"])
app.include_router(users.router, prefix="/api/v1", tags=["Users"])
app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])

# WebSocket router
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
