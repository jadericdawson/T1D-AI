"""
T1D-AI Configuration
Loads settings from environment variables with validation.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "T1D-AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Azure CosmosDB (reusing existing knowledge2ai-cosmos-serverless)
    cosmos_endpoint: str = Field(..., env="COSMOS_ENDPOINT")
    cosmos_key: str = Field(..., env="COSMOS_KEY")
    cosmos_database: str = Field(default="T1D-AI-DB", env="COSMOS_DATABASE")

    # Azure Blob Storage (reusing existing knowledge2aistorage)
    storage_account_url: str = Field(..., env="STORAGE_ACCOUNT_URL")
    storage_connection_string: str = Field(..., env="AZURE_STORAGE_CONNECTION_STRING")
    models_container: str = Field(default="t1d-ai-models", env="MODELS_CONTAINER")
    data_container: str = Field(default="t1d-ai-data", env="DATA_CONTAINER")

    # Azure OpenAI (reusing existing jadericdawson-4245-resource)
    azure_openai_endpoint: str = Field(..., env="GPT41_ENDPOINT")
    azure_openai_key: str = Field(..., env="AZURE_OPENAI_KEY")
    azure_openai_deployment: str = Field(default="H4D_Assistant_gpt-4.1", env="GPT41_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", env="AZURE_OPENAI_API_VERSION")

    # Azure AD B2C Auth
    b2c_tenant_name: str = Field(default="", env="B2C_TENANT_NAME")
    b2c_client_id: str = Field(default="", env="B2C_CLIENT_ID")
    b2c_client_secret: str = Field(default="", env="B2C_CLIENT_SECRET")
    b2c_policy_name: str = Field(default="B2C_1_signupsignin", env="B2C_POLICY_NAME")

    # JWT Settings
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=1440, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # ML Settings
    model_device: str = Field(default="cpu", env="MODEL_DEVICE")  # cpu or cuda
    bg_model_path: str = Field(default="bg_predictor_3step_v2.pth", env="BG_MODEL_PATH")
    isf_model_path: str = Field(default="best_isf_net.pth", env="ISF_MODEL_PATH")

    # Gluroo API (default Nightscout-compatible endpoint)
    default_gluroo_url: str = Field(default="https://share.gluroo.com", env="DEFAULT_GLUROO_URL")

    # Insulin/Diabetes Settings
    insulin_action_duration_minutes: int = Field(default=180, env="INSULIN_ACTION_DURATION")
    insulin_half_life_minutes: float = Field(default=81.0, env="INSULIN_HALF_LIFE")
    carb_absorption_duration_minutes: int = Field(default=180, env="CARB_ABSORPTION_DURATION")
    carb_half_life_minutes: float = Field(default=45.0, env="CARB_HALF_LIFE")
    carb_bg_factor: float = Field(default=4.0, env="CARB_BG_FACTOR")  # mg/dL per gram
    target_bg: int = Field(default=100, env="TARGET_BG")

    # Alert Settings
    high_bg_threshold: int = Field(default=180, env="HIGH_BG_THRESHOLD")
    low_bg_threshold: int = Field(default=70, env="LOW_BG_THRESHOLD")
    critical_high_threshold: int = Field(default=250, env="CRITICAL_HIGH_THRESHOLD")
    critical_low_threshold: int = Field(default=54, env="CRITICAL_LOW_THRESHOLD")

    # Security Settings
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS"
    )
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
