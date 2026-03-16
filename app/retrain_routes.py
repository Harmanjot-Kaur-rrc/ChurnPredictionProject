"""
retrain_routes.py — Fine-tuning / model retraining API.

  POST /v1/retrain              Submit a retraining job
  GET  /v1/retrain/{job_id}     Poll job status + metrics
  GET  /v1/retrain              List all jobs (admin only)

Flow:
  1. Client sends { model_id, data: [{feature_cols}, ...], labels: [0/1, ...] }
  2. A RetrainJob row is created (status=queued)
  3. A background thread loads the existing pipeline, clones the preprocessor,
     retrains the model step on the new data, evaluates, saves updated .pkl
  4. Job status transitions: queued → running → done | failed
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy.orm import Session

from app.auth import authenticate
from app.database import RetrainJob, get_db
from app.model_loader import MODEL_REGISTRY, _model_cache

logger = logging.getLogger("churn_api")
router = APIRouter(prefix="/v1/retrain", tags=["Retrain"])

# Mutex so two simultaneous retrains on the same model_id don't clobber each other
_retrain_locks: Dict[str, threading.Lock] = {}


def _get_lock(model_id: str) -> threading.Lock:
    if model_id not in _retrain_locks:
        _retrain_locks[model_id] = threading.Lock()
    return _retrain_locks[model_id]


# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="Which model to retrain. Must match a registered model_id.",
        examples=["rf"],
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of customer feature dicts. Keys must match training columns: "
            "Age, Gender, Tenure, 'Usage Frequency', 'Support Calls', "
            "'Payment Delay', 'Subscription Type', 'Contract Length', "
            "'Total Spend', 'Last Interaction'."
        ),
        min_length=10,
    )
    labels: List[int] = Field(
        ...,
        description="Churn labels (0 or 1) aligned with `data`.",
        min_length=10,
    )

    model_config = {"json_schema_extra": {
        "example": {
            "model_id": "rf",
            "data": [
                {
                    "Age": 34, "Gender": "Female", "Tenure": 12,
                    "Usage Frequency": 8, "Support Calls": 1,
                    "Payment Delay": 3, "Subscription Type": "Basic",
                    "Contract Length": "Monthly", "Total Spend": 200.0,
                    "Last Interaction": 15
                }
            ],
            "labels": [1]
        }
    }}


class RetrainJobStatus(BaseModel):
    job_id: str
    model_id: str
    status: str
    rows_received: int
    metrics: Dict[str, Any] | None
    error: str | None
    created_at: str
    finished_at: str | None
    triggered_by: str | None


# ─────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────

def _run_retrain(job_id: str, model_id: str, df: pd.DataFrame, labels: list) -> None:
    """
    Runs in a daemon thread. Retrains the *model step* of an existing sklearn Pipeline
    using the new data, evaluates on a 20% hold-out, saves the updated .pkl.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    lock = _get_lock(model_id)

    try:
        job = db.query(RetrainJob).filter_by(job_id=job_id).first()
        job.status = "running"
        db.commit()

        # ── Load the existing pipeline ──────────────────────────
        meta = MODEL_REGISTRY.get(model_id)
        if not meta:
            raise ValueError(f"Unknown model_id: {model_id}")

        pipeline = joblib.load(meta["path"])

        # ── Transform features using the *existing* preprocessor ─
        preprocessor = pipeline.named_steps["preprocessor"]
        model_step    = pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(df)
        y = np.array(labels)

        # Hold-out split for evaluation
        split = max(1, int(len(y) * 0.2))
        X_train_r, X_eval = X_transformed[:-split], X_transformed[-split:]
        y_train_r, y_eval = y[:-split], y[-split:]

        # ── Retrain model step only ──────────────────────────────
        with lock:
            model_step.fit(X_train_r, y_train_r)

            # ── Evaluate ─────────────────────────────────────────
            metrics: dict[str, Any] = {}
            if len(set(y_eval)) > 1:
                y_pred  = model_step.predict(X_eval)
                y_proba = model_step.predict_proba(X_eval)[:, 1]
                metrics = {
                    "accuracy":  round(accuracy_score(y_eval, y_pred), 4),
                    "f1":        round(f1_score(y_eval, y_pred, zero_division=0), 4),
                    "roc_auc":   round(roc_auc_score(y_eval, y_proba), 4),
                    "eval_rows": int(split),
                    "train_rows": int(len(y_train_r)),
                }
            else:
                metrics = {"warning": "Not enough class diversity in eval split."}

            # ── Persist updated pipeline ─────────────────────────
            joblib.dump(pipeline, meta["path"])

            # Update in-memory cache
            _model_cache[model_id] = pipeline

        job.status     = "done"
        job.metrics    = json.dumps(metrics)
        job.finished_at = datetime.utcnow()
        db.commit()

        logger.info("Retrain done", extra={"job_id": job_id, "model_id": model_id, "metrics": metrics})

    except Exception as exc:
        logger.exception("Retrain failed", extra={"job_id": job_id})
        job = db.query(RetrainJob).filter_by(job_id=job_id).first()
        if job:
            job.status     = "failed"
            job.error      = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.post("", status_code=202, response_model=RetrainJobStatus, summary="Submit a retraining job")
def submit_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """
    **Only admin and analyst roles may retrain models.**

    The endpoint accepts raw customer data + binary labels, validates them,
    then kicks off a background retrain so the response is immediate (HTTP 202).
    Poll `GET /v1/retrain/{job_id}` to check progress and final metrics.
    """
    request_id = getattr(request.state, "request_id", "n/a")

    if user["role"] not in {"admin", "analyst"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "FORBIDDEN",
                "message": "Only admin and analyst roles may retrain models.",
                "request_id": request_id,
            },
        )

    if body.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "NOT_AUTHORIZED",
                "message": f"Your role cannot use model '{body.model_id}'.",
                "request_id": request_id,
            },
        )

    if body.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {body.model_id}")

    if len(body.data) != len(body.labels):
        raise HTTPException(status_code=422, detail="`data` and `labels` must have the same length.")

    if not all(lbl in (0, 1) for lbl in body.labels):
        raise HTTPException(status_code=422, detail="`labels` must contain only 0 or 1.")

    # ── Create job record ──────────────────────────────────────
    job_id = str(uuid.uuid4())
    job = RetrainJob(
        job_id=job_id,
        model_id=body.model_id,
        status="queued",
        rows_received=len(body.data),
        triggered_by=user["username"],
    )
    db.add(job)
    db.commit()

    # Convert to DataFrame now (in request thread — fast)
    df = pd.DataFrame(body.data)

    # ── Kick off background thread ─────────────────────────────
    background_tasks.add_task(_run_retrain, job_id, body.model_id, df, body.labels)

    logger.info(
        "Retrain job queued",
        extra={"job_id": job_id, "model_id": body.model_id, "rows": len(body.data)},
    )

    return _job_to_schema(job)


@router.get("/{job_id}", response_model=RetrainJobStatus, summary="Poll a retrain job")
def get_job(
    job_id: str,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    job = db.query(RetrainJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_schema(job)


@router.get("", summary="[Admin] List all retrain jobs")
def list_jobs(
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    jobs = db.query(RetrainJob).order_by(RetrainJob.created_at.desc()).limit(100).all()
    return [_job_to_schema(j) for j in jobs]


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _job_to_schema(job: RetrainJob) -> RetrainJobStatus:
    return RetrainJobStatus(
        job_id=job.job_id,
        model_id=job.model_id,
        status=job.status,
        rows_received=job.rows_received,
        metrics=json.loads(job.metrics) if job.metrics else None,
        error=job.error,
        created_at=job.created_at.isoformat(),
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        triggered_by=job.triggered_by,
    )