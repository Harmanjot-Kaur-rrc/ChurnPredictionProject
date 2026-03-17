"""
predict_routes.py — All prediction endpoints.

  POST /v1/predict              Single prediction (rate limited)
  POST /v1/predict/explain      Single prediction + SHAP top-N drivers (rate limited)
  POST /v1/predict/batch        Upload CSV → scored CSV download (rate limited)

Every request is written to the prediction_log audit table regardless of outcome.
"""
from __future__ import annotations

import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import authenticate
from app.database import PredictionLog, get_db
from app.limiter import _limit_for_request, limiter
from app.model_loader import get_model
from app.schemas import ErrorResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger("churn_api")
router = APIRouter(tags=["Predict"])

# ─────────────────────────────────────────────────────────────
# SHAP explainer cache — built once per model, reused
# ─────────────────────────────────────────────────────────────
_explainer_cache: Dict[str, Any] = {}

_DUMMY_ROW = {
    "Age": 35, "Gender": "Male", "Tenure": 24,
    "Usage Frequency": 10, "Support Calls": 2,
    "Payment Delay": 5, "Subscription Type": "Standard",
    "Contract Length": "Annual", "Total Spend": 500.0,
    "Last Interaction": 30,
}

REQUIRED_COLS = {
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Last Interaction",
}


def _get_explainer(model_id: str, pipeline) -> Any:
    """
    Build a SHAP explainer for the model step. Cached per model_id.
    - Tree models (RF, GB, XGB) → fast TreeExplainer
    - Linear/neural (LogReg, MLP) → KernelExplainer (slower, ~2s extra)
    """
    if model_id in _explainer_cache:
        return _explainer_cache[model_id]

    model_step = pipeline.named_steps["model"]
    model_type = type(model_step).__name__

    if model_type in {"RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"}:
        explainer = shap.TreeExplainer(model_step)
    else:
        preprocessor = pipeline.named_steps["preprocessor"]
        n_features   = preprocessor.transform(pd.DataFrame([_DUMMY_ROW])).shape[1]
        background   = np.zeros((1, n_features))
        explainer    = shap.KernelExplainer(
            lambda x: model_step.predict_proba(x)[:, 1],
            background,
        )

    _explainer_cache[model_id] = explainer
    return explainer


def _top_shap_drivers(
    pipeline,
    model_id: str,
    df: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns top-N SHAP drivers for a single-row prediction.
    Each entry: { feature, shap_value, direction, raw_value }
    """
    preprocessor  = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    X_transformed = preprocessor.transform(df)

    explainer = _get_explainer(model_id, pipeline)

    try:
        sv = explainer.shap_values(X_transformed)
        # TreeExplainer on classifiers returns [class0_shap, class1_shap]
        shap_row = sv[1][0] if isinstance(sv, list) else sv[0]
    except Exception as e:
        logger.warning(f"SHAP failed for {model_id}: {e}")
        return []

    pairs    = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    raw_vals = X_transformed[0]
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    return [
        {
            "feature":    feat,
            "shap_value": round(float(sv_val), 4),
            "direction":  "increases_churn" if sv_val > 0 else "decreases_churn",
            "raw_value":  round(float(raw_vals[feat_idx[feat]]), 4) if feat in feat_idx else None,
        }
        for feat, sv_val in pairs
    ]


# ─────────────────────────────────────────────────────────────
# Audit helper
# ─────────────────────────────────────────────────────────────

def _log_prediction(
    db: Session,
    *,
    request_id: str | None,
    user: dict,
    model_id: str,
    endpoint: str,
    latency_ms: int,
    churn_prediction: int | None = None,
    churn_probability: float | None = None,
    batch_rows: int | None = None,
    batch_churners: int | None = None,
    input_snapshot: dict | None = None,
    shap_snapshot: list | None = None,
) -> None:
    """
    Write one immutable row to prediction_log.
    Failures here should never crash a prediction — logged and swallowed.
    """
    try:
        db.add(PredictionLog(
            request_id        = request_id,
            username          = user["username"],
            role              = user["role"],
            model_id          = model_id,
            endpoint          = endpoint,
            churn_prediction  = churn_prediction,
            churn_probability = churn_probability,
            batch_rows        = batch_rows,
            batch_churners    = batch_churners,
            input_snapshot    = json.dumps(input_snapshot)  if input_snapshot else None,
            shap_snapshot     = json.dumps(shap_snapshot)   if shap_snapshot  else None,
            latency_ms        = latency_ms,
        ))
        db.commit()
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")
        db.rollback()


# ─────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────

class ShapDriver(BaseModel):
    feature:    str
    shap_value: float = Field(..., description="Magnitude = impact, sign = direction")
    direction:  str   = Field(..., description="'increases_churn' or 'decreases_churn'")
    raw_value:  Optional[float] = None


class ExplainResponse(BaseModel):
    model_id:          str
    churn_prediction:  int
    churn_probability: float
    top_drivers:       List[ShapDriver]
    explanation_note:  str = (
        "SHAP values show each feature's contribution to the churn probability. "
        "Positive = pushes toward churn, negative = pushes toward staying."
    )


# ─────────────────────────────────────────────────────────────
# POST /v1/predict/explain
# ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/predict/explain",
    response_model=ExplainResponse,
    summary="Predict churn + SHAP explanation of top drivers",
    description=(
        "Same inputs as `/v1/predict` but also returns the top features driving "
        "this specific prediction with their SHAP impact values.\n\n"
        "**Fast models** (rf, gb, xgb): TreeExplainer — adds ~50ms.\n"
        "**Slow models** (logreg, mlp): KernelExplainer — adds ~2s on first call "
        "(cached after that).\n\n"
        "Use `top_n` query param (1–15) to control how many drivers are returned."
    ),
    responses={429: {"description": "Rate limit exceeded"}},
)
@limiter.limit(_limit_for_request)
def predict_explain(
    data:    PredictionRequest,
    request: Request,
    top_n:   int  = 5,
    user:    dict = Depends(authenticate),
    db:      Session = Depends(get_db),
):
    request_id = getattr(request.state, "request_id", "n/a")
    t0         = time.perf_counter()

    if data.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error":   "NOT_AUTHORIZED",
                "request_id": request_id,
                "message": f"Role '{user['role']}' cannot use '{data.model_id}'.",
            },
        )

    try:
        pipeline = get_model(data.model_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=400 if isinstance(e, ValueError) else 503,
            detail={"error": "MODEL_ERROR", "message": str(e)},
        )

    df          = pd.DataFrame([data.model_dump(exclude={"model_id"}, by_alias=True)])
    prediction  = int(pipeline.predict(df)[0])
    probability = round(float(pipeline.predict_proba(df)[0][1]), 4)

    top_n   = max(1, min(top_n, 15))
    drivers = _top_shap_drivers(pipeline, data.model_id, df, top_n=top_n)

    latency = int((time.perf_counter() - t0) * 1000)

    _log_prediction(
        db,
        request_id        = request_id,
        user              = user,
        model_id          = data.model_id,
        endpoint          = "explain",
        latency_ms        = latency,
        churn_prediction  = prediction,
        churn_probability = probability,
        input_snapshot    = data.model_dump(exclude={"model_id"}, by_alias=True),
        shap_snapshot     = drivers,
    )

    logger.info("Explain prediction", extra={
        "request_id": request_id, "model_id": data.model_id,
        "username": user["username"], "prediction": prediction,
        "probability": probability, "latency_ms": latency,
    })

    return ExplainResponse(
        model_id          = data.model_id,
        churn_prediction  = prediction,
        churn_probability = probability,
        top_drivers       = drivers,
    )


# ─────────────────────────────────────────────────────────────
# POST /v1/predict/batch
# ─────────────────────────────────────────────────────────────

@router.post(
    "/v1/predict/batch",
    summary="Batch predict — upload CSV, download scored CSV",
    description=(
        "Upload a CSV with the same feature columns as single predict.\n"
        "Returns a CSV with two new columns appended:\n"
        "- `churn_prediction` (0 or 1)\n"
        "- `churn_probability` (0.0 – 1.0)\n\n"
        "Response headers include summary stats:\n"
        "`X-Total-Rows`, `X-Churners`, `X-Non-Churners`, `X-Avg-Probability`, `X-Processing-Ms`\n\n"
        "**Max 50,000 rows per request.**"
    ),
    responses={
        200: {"content": {"text/csv": {}}, "description": "Scored CSV"},
        422: {"description": "Missing columns or validation error", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(_limit_for_request)
async def predict_batch(
    request:  Request,
    file:     UploadFile = File(..., description="CSV with feature columns"),
    model_id: str        = Form(..., description="Model ID, e.g. 'rf'"),
    user:     dict       = Depends(authenticate),
    db:       Session    = Depends(get_db),
):
    request_id = getattr(request.state, "request_id", "n/a")
    t0         = time.perf_counter()

    # ── Auth ──────────────────────────────────────────────────
    if model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error":      "NOT_AUTHORIZED",
                "request_id": request_id,
                "message":    f"Role '{user['role']}' cannot use '{model_id}'.",
            },
        )

    # ── Load model ────────────────────────────────────────────
    try:
        pipeline = get_model(model_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=400 if isinstance(e, ValueError) else 503,
            detail={"error": "MODEL_ERROR", "message": str(e)},
        )

    # ── Validate file ─────────────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV is empty.")
    if len(df) > 50_000:
        raise HTTPException(status_code=400, detail="Maximum 50,000 rows per batch.")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(REQUIRED_COLS)}"
            ),
        )

    # ── Score ─────────────────────────────────────────────────
    try:
        feature_df    = df[list(REQUIRED_COLS)]
        predictions   = pipeline.predict(feature_df)
        probabilities = pipeline.predict_proba(feature_df)[:, 1]
    except Exception as e:
        logger.exception("Batch prediction failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms   = int((time.perf_counter() - t0) * 1000)
    churners     = int((predictions == 1).sum())
    non_churners = int((predictions == 0).sum())
    avg_prob     = round(float(probabilities.mean()), 4)

    df["churn_prediction"]  = predictions.astype(int)
    df["churn_probability"] = probabilities.round(4)

    # ── Audit log ─────────────────────────────────────────────
    _log_prediction(
        db,
        request_id    = request_id,
        user          = user,
        model_id      = model_id,
        endpoint      = "batch",
        latency_ms    = latency_ms,
        batch_rows    = len(df),
        batch_churners = churners,
    )

    logger.info("Batch prediction complete", extra={
        "request_id": request_id, "model_id": model_id,
        "username": user["username"], "rows": len(df),
        "churners": churners, "elapsed_ms": latency_ms,
    })

    # ── Stream CSV response ───────────────────────────────────
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = f"churn_scored_{model_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Total-Rows":        str(len(df)),
            "X-Churners":          str(churners),
            "X-Non-Churners":      str(non_churners),
            "X-Avg-Probability":   str(avg_prob),
            "X-Model-Id":          model_id,
            "X-Processing-Ms":     str(latency_ms),
        },
    )