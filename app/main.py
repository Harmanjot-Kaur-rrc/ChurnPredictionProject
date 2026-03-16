"""
app/main.py
───────────
FastAPI application with:
  - Lifespan-based model cache (load once at startup)
  - CORS middleware (outermost layer)
  - Request logging middleware with X-Request-ID tracing
  - Custom exception handlers for Pydantic validation errors
    (returns clean, consistent JSON instead of FastAPI's raw 422)
  - Two-step prediction flow: GET /models → POST /predict

main.py — FastAPI application entry point.

New in v3:
  - DB-backed auth (SQLite via SQLAlchemy)
  - /v1/auth/* routes (signup, login, key management)
  - /v1/retrain routes (background retraining pipeline)
  - init_db() called at startup to create tables + seed admin
"""
from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import authenticate
from app.auth_routes import router as auth_router
from app.config import get_settings
from app.database import init_db
from app.middleware import RequestLoggingMiddleware
from app.model_loader import MODEL_REGISTRY, _model_cache, get_model, load_all_models
from app.retrain_routes import router as retrain_router
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
settings = get_settings()

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {"json": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "json"}},
        "root": {"level": settings.log_level, "handlers": ["console"]},
    }
)
logger = logging.getLogger("churn_api")


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising DB and loading models...")
    init_db()           # create tables + seed default data
    load_all_models()   # warm the model cache
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.api_title,
    description=(
        "**Churn Prediction API v3**\n\n"
        "**Quickstart:**\n"
        "1. `POST /v1/auth/signup` — create an account\n"
        "2. `POST /v1/auth/keys` — generate a time-limited API key\n"
        "3. `GET /v1/models` — discover available models\n"
        "4. `POST /v1/predict` — predict churn\n"
        "5. `POST /v1/retrain` — retrain a model with new data *(analyst/admin only)*\n\n"
        "All prediction/retrain endpoints require `x-api-key` header."
    ),
    version="3.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["x-api-key", "x-request-id", "content-type"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(retrain_router)


# ─────────────────────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    details = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err["loc"] else "unknown"
        message = err["msg"]
        if err["type"] == "literal_error":
            ctx = err.get("ctx", {})
            message = f"Invalid value. Accepted values are: {ctx.get('expected', '')}"
        details.append(ErrorDetail(field=field, message=message))

    body = ErrorResponse(error="VALIDATION_ERROR", request_id=request_id, details=details)
    return JSONResponse(status_code=422, content=body.model_dump())


# ─────────────────────────────────────────────────────────────
# Core routes
# ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def readiness_check():
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"status": "ready", "models_loaded": list(_model_cache.keys())}


@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    tags=["Step 1 — Discover Models"],
    summary="List available models",
)
def list_models(user: dict = Depends(authenticate)):
    all_models = [
        ModelInfo(model_id=mid, description=meta["description"])
        for mid, meta in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(available_models=all_models, your_allowed_models=user["allowed_models"])


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Step 2 — Predict"],
    summary="Predict customer churn",
    responses={
        422: {"description": "Validation error", "model": ErrorResponse},
        403: {"description": "Not authorized for this model"},
        503: {"description": "Model artifact missing"},
    },
)
def predict_churn(data: PredictionRequest, request: Request, user: dict = Depends(authenticate)):
    """
    Submit customer data with a `model_id` from `GET /v1/models`.
    Requires a valid, non-expired `x-api-key`.
    """
    request_id = getattr(request.state, "request_id", None)

    if data.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "NOT_AUTHORIZED",
                "request_id": request_id,
                "message": (
                    f"Your role '{user['role']}' cannot use model '{data.model_id}'. "
                    f"Allowed models: {user['allowed_models']}"
                ),
            },
        )

    try:
        model = get_model(data.model_id)
    except (ValueError, RuntimeError) as e:
        status = 400 if isinstance(e, ValueError) else 503
        raise HTTPException(
            status_code=status,
            detail={"error": "MODEL_ERROR", "request_id": request_id, "message": str(e)},
        )

    df = pd.DataFrame([data.model_dump(exclude={"model_id"}, by_alias=True)])
    prediction = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)

    logger.info(
        "Prediction made",
        extra={
            "request_id": request_id,
            "model_id": data.model_id,
            "username": user["username"],
            "role": user["role"],
            "prediction": prediction,
            "probability": probability,
        },
    )

    return PredictionResponse(
        model_id=data.model_id,
        churn_prediction=prediction,
        churn_probability=probability,
    )