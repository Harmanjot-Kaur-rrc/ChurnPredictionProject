"""
batch_routes.py — Batch prediction endpoint.

  POST /v1/predict/batch           Submit a batch job (list of rows, up to 1000)
  GET  /v1/predict/batch/{job_id}  Poll status + get result summary
  GET  /v1/predict/batch/{job_id}/download  Download result CSV

Results are written to models/batch_results/{job_id}.csv
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import authenticate
from app.config import get_settings
from app.database import BatchJob, get_db
from app.model_loader import MODEL_REGISTRY, get_active_version, get_model

logger = logging.getLogger("churn_api")
router = APIRouter(tags=["Batch Prediction"])

BATCH_RESULTS_DIR = os.path.join(get_settings().model_dir, "batch_results")
MAX_BATCH_ROWS    = 1000


# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class BatchRequest(BaseModel):
    model_id: str = Field(..., examples=["rf"])
    data: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ROWS,
        description=f"List of customer feature dicts. Max {MAX_BATCH_ROWS} rows per request.",
    )

    model_config = {"json_schema_extra": {"example": {
        "model_id": "rf",
        "data": [
            {
                "Age": 34, "Gender": "Female", "Tenure": 12,
                "Usage Frequency": 8, "Support Calls": 1, "Payment Delay": 3,
                "Subscription Type": "Basic", "Contract Length": "Monthly",
                "Total Spend": 200.0, "Last Interaction": 15,
            }
        ],
    }}}


class BatchJobStatus(BaseModel):
    job_id: str
    model_id: str
    model_version: Optional[int]
    status: str
    total_rows: int
    processed_rows: int
    churn_rate: Optional[float]     # fraction of rows predicted as churn=1
    avg_probability: Optional[float]
    error: Optional[str]
    created_at: str
    finished_at: Optional[str]
    triggered_by: Optional[str]
    download_url: Optional[str]


# ─────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────

def _run_batch(job_id: str, model_id: str, records: list) -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)

    try:
        job        = db.query(BatchJob).filter_by(job_id=job_id).first()
        job.status = "running"
        db.commit()

        model  = get_model(model_id)
        df     = pd.DataFrame(records)
        preds  = model.predict(df).tolist()
        probas = model.predict_proba(df)[:, 1].tolist()

        # Write CSV result
        result_path = os.path.join(BATCH_RESULTS_DIR, f"{job_id}.csv")
        df_out      = df.copy()
        df_out["churn_prediction"]  = preds
        df_out["churn_probability"] = [round(p, 4) for p in probas]
        df_out.to_csv(result_path, index=False)

        # Summary stats
        churn_rate      = round(sum(preds) / len(preds), 4)
        avg_probability = round(sum(probas) / len(probas), 4)

        job.status         = "done"
        job.processed_rows = len(records)
        job.result_path    = result_path
        job.finished_at    = datetime.utcnow()
        # Store summary in error field temporarily as JSON (reuse existing column)
        job.error = json.dumps({
            "churn_rate":      churn_rate,
            "avg_probability": avg_probability,
        })
        db.commit()

        logger.info("Batch done", extra={"job_id": job_id, "rows": len(records)})

    except Exception as exc:
        logger.exception("Batch failed", extra={"job_id": job_id})
        job = db.query(BatchJob).filter_by(job_id=job_id).first()
        if job:
            job.status      = "failed"
            job.error       = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/predict/batch",
    status_code=202,
    response_model=BatchJobStatus,
    summary="Submit a batch prediction job",
)
def submit_batch(
    body: BatchRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """
    Accepts a list of customer rows (up to 1,000) and runs predictions
    asynchronously. Returns a `job_id` — poll `GET /v1/predict/batch/{job_id}`
    for status. Download the full CSV from `.../download` when status is `done`.
    """
    request_id = getattr(request.state, "request_id", "n/a")

    if body.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error":      "NOT_AUTHORIZED",
                "request_id": request_id,
                "message":    f"Your role cannot use model '{body.model_id}'.",
            },
        )
    if body.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {body.model_id}")

    # Snapshot which version is active at submission time
    active = get_active_version(body.model_id)
    ver_num = active["version_number"] if active else None

    job_id = str(uuid.uuid4())
    job    = BatchJob(
        job_id=job_id,
        model_id=body.model_id,
        model_version=ver_num,
        status="queued",
        total_rows=len(body.data),
        triggered_by=user["username"],
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(_run_batch, job_id, body.model_id, body.data)

    logger.info("Batch queued", extra={"job_id": job_id, "rows": len(body.data)})
    return _job_to_schema(job, request)


@router.get(
    "/v1/predict/batch/{job_id}",
    response_model=BatchJobStatus,
    summary="Poll a batch job",
)
def get_batch_job(
    job_id: str,
    request: Request,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    job = db.query(BatchJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return _job_to_schema(job, request)


@router.get(
    "/v1/predict/batch/{job_id}/download",
    summary="Download batch prediction results as CSV",
)
def download_batch_results(
    job_id: str,
    user: dict = Depends(authenticate),
    db: Session = Depends(get_db),
):
    """
    Streams the result CSV. Only available when status is `done`.
    The CSV contains all original input columns plus:
      - `churn_prediction`  (0 or 1)
      - `churn_probability` (0.0 – 1.0)
    """
    job = db.query(BatchJob).filter_by(job_id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done yet (status: {job.status}).")
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Result file not found on disk.")

    def _stream():
        with open(job.result_path, "rb") as f:
            yield from f

    return StreamingResponse(
        _stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch_{job_id}.csv"'},
    )


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _job_to_schema(job: BatchJob, request: Request | None = None) -> BatchJobStatus:
    summary      = {}
    error_str    = None
    if job.error:
        try:
            summary   = json.loads(job.error)
        except Exception:
            error_str = job.error

    download_url = None
    if job.status == "done" and request:
        base = str(request.base_url).rstrip("/")
        download_url = f"{base}/v1/predict/batch/{job.job_id}/download"

    return BatchJobStatus(
        job_id=job.job_id,
        model_id=job.model_id,
        model_version=job.model_version,
        status=job.status,
        total_rows=job.total_rows,
        processed_rows=job.processed_rows,
        churn_rate=summary.get("churn_rate"),
        avg_probability=summary.get("avg_probability"),
        error=error_str,
        created_at=job.created_at.isoformat(),
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        triggered_by=job.triggered_by,
        download_url=download_url,
    )