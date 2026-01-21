"""
T1D-AI Configuration
Loads settings from environment variables with validation.
Supports pydantic-settings v2 syntax.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # App
    app_name: str = "T1D-AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Azure CosmosDB (reusing existing knowledge2ai-cosmos-serverless)
    cosmos_endpoint: str = Field(...)
    cosmos_key: str = Field(...)
    cosmos_database: str = Field(default="T1D-AI-DB")

    # Azure Blob Storage (reusing existing knowledge2aistorage)
    storage_account_url: str = Field(...)
    storage_connection_string: str = Field(
        ...,
        validation_alias=AliasChoices("storage_connection_string", "azure_storage_connection_string")
    )
    models_container: str = Field(default="t1d-ai-models")
    data_container: str = Field(default="t1d-ai-data")

    # Azure OpenAI (reusing existing jadericdawson-4245-resource)
    azure_openai_endpoint: str = Field(
        ...,
        validation_alias=AliasChoices("azure_openai_endpoint", "gpt41_endpoint")
    )
    azure_openai_key: str = Field(
        ...,
        validation_alias=AliasChoices("azure_openai_key", "gpt41_api_key")
    )
    azure_openai_deployment: str = Field(
        default="H4D_Assistant_gpt-4.1",
        validation_alias=AliasChoices("azure_openai_deployment", "gpt41_deployment")
    )
    azure_openai_api_version: str = Field(default="2024-12-01-preview")

    # Microsoft OAuth (Azure AD)
    microsoft_client_id: str = Field(default="")
    microsoft_client_secret: str = Field(default="")
    microsoft_tenant_id: str = Field(default="common")  # "common" allows personal + work accounts
    microsoft_redirect_uri: str = Field(default="http://localhost:5173/api/auth/microsoft/callback")

    # JWT Settings
    jwt_secret_key: str = Field(...)
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=1440)

    # ML Settings
    model_device: str = Field(default="cpu")  # cpu or cuda
    bg_model_path: str = Field(default="bg_predictor_3step_v2.pth")
    isf_model_path: str = Field(default="best_isf_net.pth")

    # MLflow / Azure ML Settings
    mlflow_tracking_uri: str = Field(
        default="",
        description="MLflow tracking URI. Leave empty for local file store, or set to Azure ML URI"
    )
    mlflow_experiment_prefix: str = Field(default="T1D-AI")
    azure_ml_workspace_name: str = Field(
        default="",
        description="Azure ML Workspace name for MLflow tracking"
    )
    azure_ml_resource_group: str = Field(
        default="",
        description="Azure resource group containing the ML workspace"
    )
    azure_ml_subscription_id: str = Field(
        default="",
        description="Azure subscription ID for ML workspace"
    )
    azure_ml_region: str = Field(
        default="eastus",
        description="Azure region for ML workspace"
    )

    # Gluroo API (default Nightscout-compatible endpoint)
    default_gluroo_url: str = Field(default="https://share.gluroo.com")

    # Insulin/Diabetes Settings
    # NOTE: This child's insulin acts 30% faster than adult formula!
    # Learned from actual BG drop data: 54 min half-life vs 81 min adult standard
    insulin_action_duration_minutes: int = Field(default=300)  # 5 hours - capture full insulin tail
    insulin_half_life_minutes: float = Field(default=54.0)  # PERSONALIZED: 54 min (not 81 adult)
    carb_absorption_duration_minutes: int = Field(default=180)
    carb_half_life_minutes: float = Field(default=45.0)
    carb_bg_factor: float = Field(default=4.0)  # mg/dL per gram
    target_bg: int = Field(default=100)

    # Alert Settings
    high_bg_threshold: int = Field(default=180)
    low_bg_threshold: int = Field(default=70)
    critical_high_threshold: int = Field(default=250)
    critical_low_threshold: int = Field(default=54)

    # Weather API Settings (OpenWeatherMap)
    openweathermap_api_key: str = Field(
        default="",
        description="OpenWeatherMap API key for weather data integration"
    )
    weather_cache_minutes: int = Field(default=15)
    default_latitude: float = Field(default=0.0)
    default_longitude: float = Field(default=0.0)

    # Security Settings
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")
    rate_limit_per_minute: int = Field(default=60)

    # Encryption Settings (for storing API secrets)
    encryption_master_key: str = Field(
        default="",
        description="Fernet encryption key for securing API secrets. Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
    )

    # Azure Communication Services (Email)
    azure_communication_connection_string: str = Field(
        default="",
        description="Azure Communication Services connection string for sending emails"
    )
    email_sender_address: str = Field(
        default="",
        description="Email sender address (e.g., DoNotReply@xxx.azurecomm.net)"
    )
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Frontend URL for email verification links"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
