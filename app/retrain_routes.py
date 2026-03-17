"""
retrain_routes.py — Retraining + model versioning endpoints.

  POST /v1/retrain                     Submit a retraining job (saves a new version)
  GET  /v1/retrain/{job_id}            Poll job status + metrics
  GET  /v1/retrain                     List all jobs (admin only)

  GET  /v1/models/{model_id}/versions  List all versions of a model
  POST /v1/models/{model_id}/promote   Promote a version to active (admin only)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy.orm import Session

from app.auth import authenticate
from app.config import get_settings
from app.database import BatchJob, ModelVersion, RetrainJob, get_db
from app.model_loader import MODEL_REGISTRY, _model_cache, get_active_version, promote_version

logger = logging.getLogger("churn_api")
router = APIRouter(tags=["Retrain & Versioning"])

_retrain_locks: Dict[str, threading.Lock] = {}


def _get_lock(model_id: str) -> threading.Lock:
    if model_id not in _retrain_locks:
        _retrain_locks[model_id] = threading.Lock()
    return _retrain_locks[model_id]


# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    model_id: str = Field(..., examples=["rf"])
    data: List[Dict[str, Any]] = Field(..., min_length=10)
    labels: List[int] = Field(..., min_length=10)
    notes: Optional[str] = Field(None, description="Optional description for this version.")


class RetrainJobStatus(BaseModel):
    job_id: str
    model_id: str
    status: str
    rows_received: int
    new_version: Optional[int]
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    finished_at: Optional[str]
    triggered_by: Optional[str]


class VersionInfo(BaseModel):
    version_number: int
    artifact_path: str
    is_active: bool
    metrics: Optional[Dict[str, Any]]
    trained_by: Optional[str]
    train_rows: Optional[int]
    notes: Optional[str]
    created_at: str


class PromoteRequest(BaseModel):
    version_number: int = Field(..., description="Version number to promote to active.")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _next_version_number(model_id: str, db: Session) -> int:
    from sqlalchemy import func
    result = db.query(func.max(ModelVersion.version_number)).filter_by(model_id=model_id).scalar()
    return (result or 0) + 1


def _versioned_path(model_id: str, version: int) -> str:
    model_dir = get_settings().model_dir
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{model_id}_v{version}.pkl")


# ─────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────

def _run_retrain(
    job_id: str,
    model_id: str,
    df: pd.DataFrame,
    labels: list,
    notes: str | None,
    triggered_by: str,
) -> None:
    from app.database import SessionLocal

    db  = SessionLocal()
    lock = _get_lock(model_id)

    try:
        job = db.query(RetrainJob).filter_by(job_id=job_id).first()
        job.status = "running"
        db.commit()

        # ── Load the currently active pipeline ─────────────────
        active_ver = db.query(ModelVersion).filter_by(model_id=model_id, is_active=True).first()
        if not active_ver:
            raise RuntimeError(f"No active version found for model '{model_id}'.")

        pipeline     = joblib.load(active_ver.artifact_path)
        preprocessor = pipeline.named_steps["preprocessor"]
        model_step   = pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(df)
        y             = np.array(labels)

        split     = max(1, int(len(y) * 0.2))
        X_train_r = X_transformed[:-split]
        X_eval    = X_transformed[-split:]
        y_train_r = y[:-split]
        y_eval    = y[-split:]

        with lock:
            model_step.fit(X_train_r, y_train_r)

            metrics: dict[str, Any] = {}
            if len(set(y_eval)) > 1:
                y_pred  = model_step.predict(X_eval)
                y_proba = model_step.predict_proba(X_eval)[:, 1]
                metrics = {
                    "accuracy":   round(accuracy_score(y_eval, y_pred), 4),
                    "f1":         round(f1_score(y_eval, y_pred, zero_division=0), 4),
                    "roc_auc":    round(roc_auc_score(y_eval, y_proba), 4),
                    "train_rows": int(len(y_train_r)),
                    "eval_rows":  int(split),
                }
            else:
                metrics = {"warning": "Not enough class diversity in eval split."}

            # ── Save as a new versioned .pkl ────────────────────
            new_ver_num  = _next_version_number(model_id, db)
            new_ver_path = _versioned_path(model_id, new_ver_num)
            joblib.dump(pipeline, new_ver_path)

            # ── Deactivate old version, register new one ────────
            db.query(ModelVersion).filter_by(model_id=model_id).update({"is_active": False})
            new_version = ModelVersion(
                model_id=model_id,
                version_number=new_ver_num,
                artifact_path=new_ver_path,
                is_active=True,
                metrics=json.dumps(metrics),
                trained_by=triggered_by,
                train_rows=int(len(y_train_r)),
                notes=notes,
            )
            db.add(new_version)

            # ── Hot-swap in-memory cache ────────────────────────
            _model_cache[model_id] = pipeline

        # Update retrain job record
        job.status      = "done"
        job.metrics     = json.dumps({**metrics, "new_version": new_ver_num})
        job.finished_at = datetime.utcnow()
        db.commit()

        logger.info(
            "Retrain done",
            extra={"job_id": job_id, "model_id": model_id,
                   "new_version": new_ver_num, "metrics": metrics},
        )

    except Exception as exc:
        logger.exception("Retrain failed", extra={"job_id": job_id})
        job = db.query(RetrainJob).filter_by(job_id=job_id).first()
        if job:
            job.status      = "failed"
            job.error       = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# Retrain routes
# ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/retrain",
    status_code=202,
    response_model=RetrainJobStatus,
    summary="Retrain a model — saves a new versioned artifact",
)
def submit_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """
    Retrains the model's *model step* on new data. The original preprocessor
    (scaling + encoding) is reused so feature transformations stay consistent.

    Each successful retrain creates a new versioned .pkl (e.g. `rf_v3.pkl`)
    and automatically promotes it to active. You can roll back via
    `POST /v1/models/{model_id}/promote`.
    """
    request_id = getattr(request.state, "request_id", "n/a")

    if user["role"] not in {"admin", "analyst"}:
        raise HTTPException(status_code=403, detail="Only admin and analyst may retrain.")
    if body.model_id not in user["allowed_models"]:
        raise HTTPException(status_code=403, detail=f"Not authorized for model '{body.model_id}'.")
    if body.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {body.model_id}")
    if len(body.data) != len(body.labels):
        raise HTTPException(status_code=422, detail="`data` and `labels` lengths must match.")
    if not all(lbl in (0, 1) for lbl in body.labels):
        raise HTTPException(status_code=422, detail="`labels` must only contain 0 or 1.")

    job_id = str(uuid.uuid4())
    job    = RetrainJob(
        job_id=job_id,
        model_id=body.model_id,
        status="queued",
        rows_received=len(body.data),
        triggered_by=user["username"],
    )
    db.add(job)
    db.commit()

    df = pd.DataFrame(body.data)
    background_tasks.add_task(
        _run_retrain, job_id, body.model_id, df, body.labels,
        body.notes, user["username"],
    )

    logger.info("Retrain queued", extra={"job_id": job_id, "model_id": body.model_id})
    return _job_to_schema(job)


@router.get("/v1/retrain/{job_id}", response_model=RetrainJobStatus, summary="Poll a retrain job")
def get_retrain_job(
    job_id: str,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    job = db.query(RetrainJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_schema(job)


@router.get("/v1/retrain", summary="[Admin] List all retrain jobs")
def list_retrain_jobs(
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    jobs = db.query(RetrainJob).order_by(RetrainJob.created_at.desc()).limit(100).all()
    return [_job_to_schema(j) for j in jobs]


# ─────────────────────────────────────────────────────────────
# Versioning routes
# ─────────────────────────────────────────────────────────────

@router.get(
    "/v1/models/{model_id}/versions",
    response_model=List[VersionInfo],
    summary="List all saved versions of a model",
)
def list_versions(
    model_id: str,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """
    Returns all versions for a model, newest first.
    The version with `is_active=true` is the one currently serving predictions.
    """
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    versions = (
        db.query(ModelVersion)
        .filter_by(model_id=model_id)
        .order_by(ModelVersion.version_number.desc())
        .all()
    )
    return [_ver_to_schema(v) for v in versions]


@router.post(
    "/v1/models/{model_id}/promote",
    summary="[Admin] Promote a model version to active",
)
def promote_model_version(
    model_id: str,
    body: PromoteRequest,
    user: dict = Depends(authenticate),
):
    """
    Sets the specified version as active and immediately hot-swaps the
    in-memory model cache. All subsequent predictions use the new version.
    No restart required.

    Use this to roll back to a previous version if a retrain degraded metrics.
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    try:
        promote_version(model_id, body.version_number)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "message":        f"Model '{model_id}' promoted to version {body.version_number}.",
        "model_id":       model_id,
        "active_version": body.version_number,
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _job_to_schema(job: RetrainJob) -> RetrainJobStatus:
    metrics_raw = json.loads(job.metrics) if job.metrics else None
    new_version = metrics_raw.pop("new_version", None) if metrics_raw else None
    return RetrainJobStatus(
        job_id=job.job_id,
        model_id=job.model_id,
        status=job.status,
        rows_received=job.rows_received,
        new_version=new_version,
        metrics=metrics_raw,
        error=job.error,
        created_at=job.created_at.isoformat(),
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        triggered_by=job.triggered_by,
    )


def _ver_to_schema(v: ModelVersion) -> VersionInfo:
    return VersionInfo(
        version_number=v.version_number,
        artifact_path=v.artifact_path,
        is_active=v.is_active,
        metrics=json.loads(v.metrics) if v.metrics else None,
        trained_by=v.trained_by,
        train_rows=v.train_rows,
        notes=v.notes,
        created_at=v.created_at.isoformat(),
    )