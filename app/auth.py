"""
app/auth.py
───────────
API key authentication + role-based authorization.
Keys are loaded from environment variables — never hardcoded.
"""
from fastapi import Security, HTTPException, Request
from fastapi.security import APIKeyHeader

from app.config import get_settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def _get_api_keys() -> dict:
    """Load and parse API keys from settings (reads .env once via lru_cache)."""
    return get_settings().get_api_keys()


def authenticate(
    request: Request,
    api_key: str = Security(api_key_header),
) -> dict:
    """
    Validates the x-api-key header.
    Returns the user dict: { "role": "...", "allowed_models": [...] }
    Raises 401 if missing or invalid.
    """
    request_id = request.headers.get("x-request-id", "n/a")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "MISSING_API_KEY",
                "message": "Provide your API key in the 'x-api-key' request header.",
                "request_id": request_id,
            },
        )

    api_keys = _get_api_keys()
    user = api_keys.get(api_key)

    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_API_KEY",
                "message": "The API key provided is not recognized.",
                "request_id": request_id,
            },
        )

    return user