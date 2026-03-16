"""
security.py — Cryptographic helpers.

  - Passwords : bcrypt via passlib
  - API keys  : secrets.token_urlsafe → SHA-256 stored in DB
  - Key expiry: configurable TTL per role (default 30 days)
"""
from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Tuple

from passlib.context import CryptContext

# ── Password hashing (bcrypt) ──────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── API key generation / hashing ───────────────────────────────────────────
KEY_PREFIX = "sk-"   # cosmetic prefix so users recognise their keys


def generate_api_key() -> Tuple[str, str]:
    """
    Returns (raw_key, key_hash).
    raw_key  — shown to user ONCE (e.g. "sk-abc123…")
    key_hash — SHA-256 hex stored in DB, never the raw value
    """
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return raw, digest


def hash_key(raw_key: str) -> str:
    """Hash an incoming key for DB lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ── Dynamic TTL per role ───────────────────────────────────────────────────
# Override via env vars: API_KEY_TTL_ADMIN=7  (days)
_DEFAULT_TTL: dict[str, int] = {
    "admin":   int(os.getenv("API_KEY_TTL_ADMIN",   "365")),
    "analyst": int(os.getenv("API_KEY_TTL_ANALYST",  "90")),
    "guest":   int(os.getenv("API_KEY_TTL_GUEST",    "30")),
}


def key_ttl_for_role(role: str) -> timedelta:
    days = _DEFAULT_TTL.get(role, 30)
    return timedelta(days=days)


def key_expiry_for_role(role: str) -> datetime:
    return datetime.utcnow() + key_ttl_for_role(role)