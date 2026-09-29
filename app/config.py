
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Data Analyst Team Agent"
    app_env: str = "development"
    debug: bool = False

    # Application-owned database only.
    database_url: SecretStr | None = None

    # Groq is introduced in a later phase.
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    # Authentication
    jwt_secret_key: SecretStr | None = None
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
    )

    db_connect_timeout: int = Field(default=5, ge=1, le=30)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()