"""
auth.py — DB-backed authentication.

  1. API key extracted from x-api-key header
  2. SHA-256 hashed → looked up in api_keys table (parameterised query — no SQL injection)
  3. Key checked: not revoked, not expired
  4. Allowed models fetched from role_model_access table for the user's role
  5. Returns user dict: { "username", "role", "allowed_models" }
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.database import ApiKey, RoleModelAccess, User, get_db
from app.security import hash_key

logger = logging.getLogger("churn_api")

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def authenticate(
    request: Request,
    raw_key: str = Security(api_key_header),
    db: Session = Depends(get_db),
) -> dict:
    """
    FastAPI dependency.  Validates the x-api-key header against the DB.

    Security measures:
      - Raw key is NEVER stored; only its SHA-256 hash is looked up
      - ORM parameterised queries prevent SQL injection
      - Revoked + expired keys are rejected with distinct error messages
    """
    request_id = getattr(request.state, "request_id", "n/a")

    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "MISSING_API_KEY",
                "message": "Provide your API key in the 'x-api-key' header.",
                "request_id": request_id,
            },
        )

    key_hash = hash_key(raw_key)

    # ── Parameterised ORM query (safe from SQL injection) ──────────────────
    record: ApiKey | None = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == key_hash)
        .first()
    )

    if record is None:
        logger.warning("Auth failed: unknown key", extra={"request_id": request_id})
        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_API_KEY",
                "message": "The API key provided is not recognised.",
                "request_id": request_id,
            },
        )

    if record.revoked:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "REVOKED_API_KEY",
                "message": "This API key has been revoked.",
                "request_id": request_id,
            },
        )

    if datetime.utcnow() > record.expires_at:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "EXPIRED_API_KEY",
                "message": (
                    f"This API key expired on {record.expires_at.isoformat()}. "
                    "Please generate a new key via POST /v1/auth/keys."
                ),
                "request_id": request_id,
            },
        )

    user: User = record.user
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "ACCOUNT_DISABLED",
                "message": "Your account has been disabled.",
                "request_id": request_id,
            },
        )

    # ── Role: allowed models from DB ──────────────────────────────────────
    allowed_models = [
        row.model_id
        for row in db.query(RoleModelAccess).filter_by(role=user.role).all()
    ]

    return {
        "username": user.username,
        "role": user.role,
        "allowed_models": allowed_models,
        "key_expires_at": record.expires_at.isoformat(),
    }