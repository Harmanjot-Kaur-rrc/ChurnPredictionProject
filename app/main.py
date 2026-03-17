"""
main.py — FastAPI application entry point (v4).

New in this version:
  - POST /v1/predict          now accepts ?explain=true for SHAP explanations
  - POST /v1/predict/batch    batch prediction (async, CSV download)
  - GET/POST /v1/models/{id}/versions  and /promote  — model versioning
  - All predictions log which model version was used
"""
from __future__ import annotations

import logging
import logging.config
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import authenticate
from app.auth_routes import router as auth_router
from app.batch_routes import router as batch_router
from app.config import get_settings
from app.database import init_db
from app.middleware import RequestLoggingMiddleware
from app.model_loader import MODEL_REGISTRY, _model_cache, get_active_version, get_model, load_all_models
from app.retrain_routes import router as retrain_router
from app.schemas import (
    ErrorDetail, ErrorResponse, ModelInfo, ModelsResponse,
    PredictionRequest, PredictionResponse,
)

warnings.filterwarnings("ignore", category=UserWarning)

settings = get_settings()

logging.config.dictConfig({
    "version": 1,
    "formatters": {"json": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers":   {"console": {"class": "logging.StreamHandler", "formatter": "json"}},
    "root":       {"level": settings.log_level, "handlers": ["console"]},
})
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising DB and loading models…")
    init_db()
    load_all_models()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.api_title,
    description=(
        "**Churn Prediction API**\n\n"
        "1. `POST /v1/auth/signup` — create account\n"
        "2. `POST /v1/auth/keys` — generate time-limited API key\n"
        "3. `GET /v1/models` — list models you can use\n"
        "4. `POST /v1/predict` — single prediction (add `?explain=true` for SHAP)\n"
        "5. `POST /v1/predict/batch` — batch predictions (async, up to 1,000 rows)\n"
        "6. `POST /v1/retrain` — retrain a model (saves a new versioned artifact)\n"
        "7. `GET /v1/models/{model_id}/versions` — view version history\n"
        "8. `POST /v1/models/{model_id}/promote` — roll back or forward to any version\n"
    ),
    version="4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["x-api-key", "x-request-id", "content-type"],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth_router)
app.include_router(retrain_router)
app.include_router(batch_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    details = []
    for err in exc.errors():
        field   = str(err["loc"][-1]) if err["loc"] else "unknown"
        message = err["msg"]
        if err["type"] == "literal_error":
            message = f"Invalid value. Accepted: {err.get('ctx', {}).get('expected', '')}"
        details.append(ErrorDetail(field=field, message=message))
    body = ErrorResponse(error="VALIDATION_ERROR", request_id=request_id, details=details)
    return JSONResponse(status_code=422, content=body.model_dump())


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def readiness_check():
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"status": "ready", "models_loaded": list(_model_cache.keys())}


@app.get("/v1/models", response_model=ModelsResponse, tags=["Models"])
def list_models(user: dict = Depends(authenticate)):
    all_models = [
        ModelInfo(model_id=mid, description=meta["description"])
        for mid, meta in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(available_models=all_models, your_allowed_models=user["allowed_models"])


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Predict"],
    summary="Single prediction — add ?explain=true for SHAP feature contributions",
    responses={
        422: {"description": "Validation error", "model": ErrorResponse},
        403: {"description": "Not authorized for this model"},
        503: {"description": "Model artifact missing"},
    },
)
def predict_churn(
    data: PredictionRequest,
    request: Request,
    user: dict = Depends(authenticate),
    explain: bool = Query(False, description="Return SHAP feature contributions alongside the prediction."),
):
    request_id = getattr(request.state, "request_id", None)

    if data.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error":      "NOT_AUTHORIZED",
                "request_id": request_id,
                "message":    f"Role '{user['role']}' cannot use '{data.model_id}'. Allowed: {user['allowed_models']}",
            },
        )

    try:
        model = get_model(data.model_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=400 if isinstance(e, ValueError) else 503,
            detail={"error": "MODEL_ERROR", "request_id": request_id, "message": str(e)},
        )

    df          = pd.DataFrame([data.model_dump(exclude={"model_id"}, by_alias=True)])
    prediction  = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)

    # ── Active version snapshot ────────────────────────────────
    active = get_active_version(data.model_id)
    version_used = active["version_number"] if active else None

    # ── SHAP explanations (optional) ──────────────────────────
    explanation: list | None = None
    if explain:
        explanation = _compute_shap(model, df, data.model_id)

    logger.info(
        "Prediction made",
        extra={
            "request_id":     request_id,
            "model_id":       data.model_id,
            "version":        version_used,
            "username":       user["username"],
            "prediction":     prediction,
            "probability":    probability,
            "explain":        explain,
        },
    )

    return PredictionResponse(
        model_id=data.model_id,
        model_version=version_used,
        churn_prediction=prediction,
        churn_probability=probability,
        explanation=explanation,
    )


def _compute_shap(model, df: pd.DataFrame, model_id: str) -> list:
    """
    Compute SHAP values using the fastest available explainer for each model type:
      - TreeExplainer  : XGBoost, RandomForest, GradientBoosting  (milliseconds)
      - LinearExplainer: LogisticRegression                        (milliseconds)
      - KernelExplainer: MLP fallback                              (slow ~5s)

    Returns a list of dicts sorted by |shap_value| descending, top 10 only.
    """
    try:
        import shap
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        preprocessor  = model.named_steps["preprocessor"]
        clf           = model.named_steps["model"]
        X_transformed = preprocessor.transform(df)
        feature_names = [
            n.replace("num__", "").replace("cat__", "")
            for n in preprocessor.get_feature_names_out()
        ]

        # ── Pick the right explainer ───────────────────────────
        if isinstance(clf, (XGBClassifier, RandomForestClassifier, GradientBoostingClassifier)):
            # TreeExplainer is native to tree models — fast and exact
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_transformed)
            # RF/GB return list[class0_array, class1_array]; XGB returns single array
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]   # class 1 = churn
            else:
                sv = shap_vals[0]

        elif isinstance(clf, LogisticRegression):
            # LinearExplainer uses the model coefficients directly — instant
            explainer = shap.LinearExplainer(clf, X_transformed, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_transformed)
            # LinearExplainer for binary returns single array
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

        else:
            # MLP fallback — KernelExplainer with small sample for speed
            background = shap.kmeans(X_transformed, min(10, len(X_transformed)))
            explainer  = shap.KernelExplainer(clf.predict_proba, background)
            shap_vals  = explainer.shap_values(X_transformed, nsamples=50, silent=True)
            sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        # ── Build result list ──────────────────────────────────
        raw_values = X_transformed[0]
        results    = []
        for fname, sval, fval in zip(feature_names, sv, raw_values):
            results.append({
                "feature":    fname,
                "value":      round(float(fval), 4),
                "shap_value": round(float(sval), 4),
                "direction":  "increases churn" if sval > 0 else "decreases churn",
            })

        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:10]

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}", exc_info=True)
        return [{"feature": "explanation_error", "value": 0, "shap_value": 0,
                 "direction": f"SHAP failed: {str(e)}"}]