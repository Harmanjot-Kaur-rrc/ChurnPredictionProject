"""
app/config.py
─────────────
Loads all configuration from environment variables / .env file.
No secrets are ever hardcoded here.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    api_title: str = "Churn Prediction API"
    api_version: str = "2.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    model_dir: str = "models"

    # CORS — comma-separated origins allowed to call the API
    # e.g. "http://localhost:8501,https://yourdomain.com"
    cors_origins: str = "http://localhost:8501"

    # API Keys loaded from environment (see .env.example for format)
    # Each variable is: ROLE:model1,model2,...
    api_key_admin: str = ""
    api_key_analyst: str = ""
    api_key_guest: str = ""

    model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",          
)

    def get_api_keys(self) -> dict[str, dict]:
        """
        Parse the three env vars into the same dict structure
        the rest of the app expects:
          { "<key>": { "role": "...", "allowed_models": [...] } }
        """
        raw_entries = {
            "admin":   self.api_key_admin,
            "analyst": self.api_key_analyst,
            "guest":   self.api_key_guest,
        }
        result = {}
        for role, raw in raw_entries.items():
            if not raw:
                continue
            # format expected: "actual-key-value:role:model1,model2"
            parts = raw.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed API key env var for role '{role}'. "
                    f"Expected format: key:role:model1,model2  — got: '{raw}'"
                )
            key_value, key_role, models_str = parts
            result[key_value] = {
                "role": key_role,
                "allowed_models": [m.strip() for m in models_str.split(",")],
            }
        return result

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is only read once."""
    return Settings()