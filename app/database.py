"""
database.py — SQLite via SQLAlchemy.

Tables (original):  users, api_keys, role_model_access, retrain_jobs
Tables (new):       model_versions, batch_jobs
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DB_PATH      = os.getenv("DB_PATH", "churn_api.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Original tables ────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role            = Column(String(32), nullable=False, default="guest")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    api_keys        = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"
    id         = Column(Integer, primary_key=True, index=True)
    key_hash   = Column(String(256), unique=True, nullable=False, index=True)
    key_prefix = Column(String(16), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked    = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User", back_populates="api_keys")


class RoleModelAccess(Base):
    __tablename__ = "role_model_access"
    id       = Column(Integer, primary_key=True)
    role     = Column(String(32), nullable=False, index=True)
    model_id = Column(String(32), nullable=False)


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"
    id            = Column(Integer, primary_key=True, index=True)
    job_id        = Column(String(64), unique=True, nullable=False, index=True)
    model_id      = Column(String(32), nullable=False)
    status        = Column(String(16), default="queued")
    rows_received = Column(Integer, default=0)
    metrics       = Column(Text, nullable=True)
    error         = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    finished_at   = Column(DateTime, nullable=True)
    triggered_by  = Column(String(64), nullable=True)


# ── New: model_versions ────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    One row per trained / retrained version of each model_id.
    version_number increments per model_id (1, 2, 3…).
    is_active=True  → this version is currently serving predictions.
    artifact_path   → versioned .pkl path, e.g. models/rf_v3.pkl
    """
    __tablename__ = "model_versions"
    id             = Column(Integer, primary_key=True, index=True)
    model_id       = Column(String(32), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    artifact_path  = Column(String(512), nullable=False)
    is_active      = Column(Boolean, default=False, nullable=False)
    metrics        = Column(Text, nullable=True)     # JSON string
    trained_by     = Column(String(64), nullable=True)
    train_rows     = Column(Integer, nullable=True)
    notes          = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


# ── New: batch_jobs ────────────────────────────────────────────────────────

class BatchJob(Base):
    """
    One row per batch prediction request.
    result_path → CSV written when status=done.
    """
    __tablename__ = "batch_jobs"
    id             = Column(Integer, primary_key=True, index=True)
    job_id         = Column(String(64), unique=True, nullable=False, index=True)
    model_id       = Column(String(32), nullable=False)
    model_version  = Column(Integer, nullable=True)
    status         = Column(String(16), default="queued")
    total_rows     = Column(Integer, default=0)
    processed_rows = Column(Integer, default=0)
    result_path    = Column(String(512), nullable=True)
    error          = Column(Text, nullable=True)
    triggered_by   = Column(String(64), nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    finished_at    = Column(DateTime, nullable=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    from app.security import hash_password, generate_api_key
    from datetime import timedelta

    db = SessionLocal()
    try:
        # Role → model access
        defaults = {
            "admin":   ["logreg", "rf", "gb", "xgb", "mlp"],
            "analyst": ["rf", "gb", "xgb"],
            "guest":   ["logreg"],
        }
        for role, models in defaults.items():
            existing = {r.model_id for r in db.query(RoleModelAccess).filter_by(role=role).all()}
            for mid in models:
                if mid not in existing:
                    db.add(RoleModelAccess(role=role, model_id=mid))

        # Seed admin
        if not db.query(User).filter_by(username="admin").first():
            admin = User(
                username="admin",
                hashed_password=hash_password("Admin@1234"),
                role="admin",
            )
            db.add(admin)
            db.flush()
            raw_key, key_hash = generate_api_key()
            db.add(ApiKey(
                key_hash=key_hash,
                key_prefix=raw_key[:7],
                user_id=admin.id,
                expires_at=datetime.utcnow() + timedelta(days=365),
            ))
            print(f"\n[SEED] Admin API key (save this — shown once): {raw_key}\n")

        # Register existing .pkl files as version 1 if not already in DB
        _seed_existing_versions(db)

        db.commit()
    finally:
        db.close()


def _seed_existing_versions(db: Session) -> None:
    """
    If model_versions is empty but original .pkl files exist on disk,
    register them as version 1 (active) so versioning works immediately.
    """
    from app.config import get_settings
    model_dir = get_settings().model_dir

    for mid in ["logreg", "rf", "gb", "xgb", "mlp"]:
        if db.query(ModelVersion).filter_by(model_id=mid).first():
            continue
        path = os.path.join(model_dir, f"{mid}.pkl")
        if not os.path.exists(path):
            continue
        db.add(ModelVersion(
            model_id=mid,
            version_number=1,
            artifact_path=path,
            is_active=True,
            trained_by="pipeline",
            notes="Initial version — registered automatically from train_pipeline.py output.",
        ))