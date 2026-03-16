"""
database.py — SQLite database setup via SQLAlchemy.

Tables:
  users       — username / hashed_password / role / created_at
  api_keys    — key_hash / user_id / expires_at / created_at / revoked
  role_models — role / model_id  (many-to-many)
  retrain_jobs — job_id / model_id / status / created_at / finished_at / error
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DB_PATH = os.getenv("DB_PATH", "churn_api.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="guest")  # admin | analyst | guest
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(256), unique=True, nullable=False, index=True)
    # Store a short prefix so users can identify their key (e.g. "sk-abcd")
    key_prefix = Column(String(16), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="api_keys")


class RoleModelAccess(Base):
    """Which model_ids each role may access."""
    __tablename__ = "role_model_access"

    id = Column(Integer, primary_key=True)
    role = Column(String(32), nullable=False, index=True)
    model_id = Column(String(32), nullable=False)


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    model_id = Column(String(32), nullable=False)
    status = Column(String(16), default="queued")  # queued | running | done | failed
    rows_received = Column(Integer, default=0)
    metrics = Column(Text, nullable=True)           # JSON string of eval metrics
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    triggered_by = Column(String(64), nullable=True)  # username


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency — yields a DB session and closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables + seed default roles and an admin user if missing."""
    Base.metadata.create_all(bind=engine)

    from app.security import hash_password, generate_api_key, hash_key
    from datetime import timedelta

    db = SessionLocal()
    try:
        # ── Seed role → model access ──────────────────────────────
        default_access = {
            "admin":   ["logreg", "rf", "gb", "xgb", "mlp"],
            "analyst": ["rf", "gb", "xgb"],
            "guest":   ["logreg"],
        }
        for role, models in default_access.items():
            existing = db.query(RoleModelAccess).filter_by(role=role).all()
            existing_ids = {r.model_id for r in existing}
            for mid in models:
                if mid not in existing_ids:
                    db.add(RoleModelAccess(role=role, model_id=mid))

        # ── Seed admin user if none exists ────────────────────────
        admin = db.query(User).filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                hashed_password=hash_password("Admin@1234"),
                role="admin",
            )
            db.add(admin)
            db.flush()  # get admin.id

            raw_key, key_hash = generate_api_key()
            db.add(ApiKey(
                key_hash=key_hash,
                key_prefix=raw_key[:7],
                user_id=admin.id,
                expires_at=datetime.utcnow() + timedelta(days=365),
            ))
            print(f"\n[SEED] Admin API key (save this — shown once): {raw_key}\n")

        db.commit()
    finally:
        db.close()