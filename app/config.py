"""
config.py — Application settings loaded from .env via pydantic-settings.

API keys are no longer stored here — they live in the SQLite database.
Only app-level configuration remains.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App identity
    api_title: str = "Churn Prediction API"
    api_version: str = "3.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Model artefacts directory
    model_dir: str = "models"

    # SQLite database path (can be overridden by DB_PATH env var)
    db_path: str = "churn_api.db"

    # CORS — comma-separated origins
    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is read once per process."""
    return Settings()