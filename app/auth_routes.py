"""
auth_routes.py — User-facing auth endpoints.

  POST /v1/auth/signup          register a new account (role defaults to 'guest')
  POST /v1/auth/login           verify password, return active API keys summary
  POST /v1/auth/keys            generate a new time-limited API key
  GET  /v1/auth/keys            list your active keys (prefix + expiry, never raw)
  DELETE /v1/auth/keys/{prefix} revoke a key by its prefix
  
  Admin only:
  PATCH /v1/auth/users/{username}/role   change a user's role
  GET   /v1/auth/users                   list all users
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import ApiKey, RoleModelAccess, User, get_db
from app.security import (
    generate_api_key,
    hash_password,
    key_expiry_for_role,
    verify_password,
)

logger = logging.getLogger("churn_api")
router = APIRouter(prefix="/v1/auth", tags=["Auth"])

VALID_ROLES = {"admin", "analyst", "guest"}


# ─────────────────────────────────────────────────────────────
# Pydantic schemas (auth-specific)
# ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, examples=["alice"])
    password: str = Field(..., min_length=8, max_length=128, examples=["Secret@99"])

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class KeyInfo(BaseModel):
    prefix: str
    expires_at: str
    created_at: str
    revoked: bool


class LoginResponse(BaseModel):
    message: str
    username: str
    role: str
    active_keys: List[KeyInfo]


class NewKeyResponse(BaseModel):
    message: str
    raw_key: str = Field(..., description="Store this - shown ONCE.")
    prefix: str
    expires_at: str


class RoleChangeRequest(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"Role must be one of {VALID_ROLES}")
        return v


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _active_keys(user: User) -> List[KeyInfo]:
    now = datetime.utcnow()
    return [
        KeyInfo(
            prefix=k.key_prefix,
            expires_at=k.expires_at.isoformat(),
            created_at=k.created_at.isoformat(),
            revoked=k.revoked,
        )
        for k in user.api_keys
        if not k.revoked and k.expires_at > now
    ]


def _require_admin(db: Session, username: str) -> User:
    user = db.query(User).filter_by(username=username).first()
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.post("/signup", status_code=201, summary="Register a new account")
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    """
    Creates a new user with role='guest'.
    Passwords are bcrypt-hashed before storage, plaintext is never persisted.
    """
    if db.query(User).filter_by(username=body.username).first():
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        role="guest",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("New user registered", extra={"username": user.username})
    return {"message": "Account created.", "username": user.username, "role": user.role}


@router.post("/login", response_model=LoginResponse, summary="Login and view your API keys")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """
    Verifies password (bcrypt). Returns the user's active API key summary.
    Passwords are compared via constant-time bcrypt verify — not plain equality.
    """
    user = db.query(User).filter_by(username=body.username).first()

    # Constant-time comparison even on missing user (prevent username enumeration)
    dummy_hash = "$2b$12$notarealhashatallxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    stored = user.hashed_password if user else dummy_hash

    if not verify_password(body.password, stored) or not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled.")

    return LoginResponse(
        message="Login successful.",
        username=user.username,
        role=user.role,
        active_keys=_active_keys(user),
    )


@router.post("/keys", response_model=NewKeyResponse, summary="Generate a new API key")
def create_key(body: LoginRequest, db: Session = Depends(get_db)):
    """
    Password-gated key issuance. TTL is set dynamically based on the user's role:
      - admin   : 365 days (configurable via API_KEY_TTL_ADMIN env var)
      - analyst : 90 days
      - guest   : 30 days

    The raw key is returned **once** — it is not stored. Only a SHA-256 hash
    is persisted so a DB breach cannot expose live keys.
    """
    user = db.query(User).filter_by(username=body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    raw_key, key_hash = generate_api_key()
    expires_at = key_expiry_for_role(user.role)

    api_key = ApiKey(
        key_hash=key_hash,
        key_prefix=raw_key[:7],
        user_id=user.id,
        expires_at=expires_at,
    )
    db.add(api_key)
    db.commit()

    logger.info(
        "API key issued",
        extra={"username": user.username, "role": user.role, "expires_at": expires_at.isoformat()},
    )

    return NewKeyResponse(
        message="Key created. Store it securely — it will not be shown again.",
        raw_key=raw_key,
        prefix=raw_key[:7],
        expires_at=expires_at.isoformat(),
    )


@router.get("/keys", summary="List your active API keys")
def list_keys(body: LoginRequest, db: Session = Depends(get_db)):
    """Returns prefixes and expiry timestamps — never the raw keys."""
    user = db.query(User).filter_by(username=body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    return {"username": user.username, "role": user.role, "keys": _active_keys(user)}


@router.delete("/keys/{prefix}", summary="Revoke an API key by its prefix")
def revoke_key(prefix: str, body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    key = db.query(ApiKey).filter_by(user_id=user.id, key_prefix=prefix).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found.")

    key.revoked = True
    db.commit()
    return {"message": f"Key '{prefix}' revoked."}


# ─────────────────────────────────────────────────────────────
# Admin-only routes
# ─────────────────────────────────────────────────────────────

@router.get("/users", summary="[Admin] List all users")
def list_users(admin_username: str, admin_password: str, db: Session = Depends(get_db)):
    admin = db.query(User).filter_by(username=admin_username).first()
    if not admin or not verify_password(admin_password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    _require_admin(db, admin_username)

    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.patch("/users/{username}/role", summary="[Admin] Change a user's role")
def set_user_role(
    username: str,
    body: RoleChangeRequest,
    admin_username: str,
    admin_password: str,
    db: Session = Depends(get_db),
):
    """
    Role injection is guarded:
      1. Only admins can call this endpoint.
      2. Role value is validated against a whitelist, arbitrary strings rejected.
      3. After role change, existing keys retain old expiry; new keys use new role TTL.
    """
    admin = db.query(User).filter_by(username=admin_username).first()
    if not admin or not verify_password(admin_password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")
    _require_admin(db, admin_username)

    target = db.query(User).filter_by(username=username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    old_role = target.role
    target.role = body.role
    db.commit()

    logger.info(
        "Role changed",
        extra={"target": username, "old_role": old_role, "new_role": body.role, "by": admin_username},
    )
    return {"message": f"Role updated.", "username": username, "role": body.role}


@router.patch("/users/{username}/activate", summary="[Admin] Enable or disable a user")
def toggle_user(
    username: str,
    active: bool,
    admin_username: str,
    admin_password: str,
    db: Session = Depends(get_db),
):
    admin = db.query(User).filter_by(username=admin_username).first()
    if not admin or not verify_password(admin_password, admin.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")
    _require_admin(db, admin_username)

    target = db.query(User).filter_by(username=username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found.")

    target.is_active = active
    db.commit()
    return {"message": f"User '{username}' {'activated' if active else 'deactivated'}."}


@router.get("/roles", summary="List roles and their model access")
def list_role_access(db: Session = Depends(get_db)):
    rows = db.query(RoleModelAccess).all()
    result: dict[str, list] = {}
    for row in rows:
        result.setdefault(row.role, []).append(row.model_id)
    return result