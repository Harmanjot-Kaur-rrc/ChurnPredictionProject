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
from app.config import get_settings
from app.middleware import RequestLoggingMiddleware
from app.model_loader import MODEL_REGISTRY, get_model, load_all_models
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────
settings = get_settings()

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "json": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {"level": settings.log_level, "handlers": ["console"]},
    }
)

logger = logging.getLogger("churn_api")


# ─────────────────────────────────────────────────────────────
# Lifespan — runs setup/teardown around the server's lifetime
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models into cache...")
    load_all_models()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.api_title,
    description=(
        "**Two-step churn prediction flow:**\n\n"
        "1. `GET /v1/models` — authenticate and browse available models\n"
        "2. `POST /v1/predict` — send your chosen `model_id` with customer data\n\n"
        "All endpoints require an `x-api-key` header."
    ),
    version=settings.api_version,
    lifespan=lifespan,
)

# ── Middleware (outermost first) ───────────────────────────────────────────
# CORS must be the outermost middleware so preflight OPTIONS requests
# are handled before any auth or logging runs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["x-api-key", "x-request-id", "content-type"],
)

# Request logging sits inside CORS — logs real requests, not preflight noise.
app.add_middleware(RequestLoggingMiddleware)


# ─────────────────────────────────────────────────────────────
# Custom exception handlers
# ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Replaces FastAPI's default 422 response with a clean, consistent shape.

    Default FastAPI 422 exposes internal Pydantic field paths and is noisy.
    This handler:
      - Returns HTTP 422 with error code VALIDATION_ERROR
      - Lists each invalid field with a plain-English message
      - Echoes the X-Request-ID so the caller can correlate with logs

    Example response:
    {
      "error": "VALIDATION_ERROR",
      "request_id": "a1b2c3d4",
      "details": [
        { "field": "Gender", "message": "Input should be 'Male' or 'Female'" },
        { "field": "Age",    "message": "Input should be >= 18" }
      ]
    }
    """
    request_id = getattr(request.state, "request_id", None)

    details = []
    for err in exc.errors():
        # loc is a tuple like ("body", "Gender") — take the last element
        field = str(err["loc"][-1]) if err["loc"] else "unknown"
        message = err["msg"]

        # Make the message friendlier for enum/literal validation failures
        if err["type"] == "literal_error":
            ctx = err.get("ctx", {})
            expected = ctx.get("expected", "")
            message = f"Invalid value. Accepted values are: {expected}"

        details.append(ErrorDetail(field=field, message=message))

    body = ErrorResponse(
        error="VALIDATION_ERROR",
        request_id=request_id,
        details=details,
    )

    logger.warning(
        "Validation error",
        extra={"request_id": request_id, "errors": len(details)},
    )

    return JSONResponse(status_code=422, content=body.model_dump())


# ─────────────────────────────────────────────────────────────
# Routes — all versioned under /v1
# ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe — is the process running?"""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def readiness_check():
    """
    Readiness probe — are models loaded and ready to serve?
    Returns 503 if no models are available yet.
    """
    from app.model_loader import _model_cache
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {"status": "ready", "models_loaded": list(_model_cache.keys())}


@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    tags=["Step 1 — Discover Models"],
    summary="List available models and see which ones you can use",
)
def list_models(user: dict = Depends(authenticate)):
    """
    **Start here.** Returns all registered models and highlights
    which ones your API key is authorized to call.

    Use the `model_id` values from `your_allowed_models` when calling `POST /v1/predict`.
    """
    all_models = [
        ModelInfo(model_id=mid, description=meta["description"])
        for mid, meta in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(
        available_models=all_models,
        your_allowed_models=user["allowed_models"],
    )


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Step 2 — Predict"],
    summary="Predict customer churn using a chosen model",
    responses={
        422: {
            "description": "Validation error — invalid field values",
            "model": ErrorResponse,
        },
        403: {"description": "You are not authorized to use this model"},
        503: {"description": "Model artifact missing — run train_pipeline.py"},
    },
)
def predict_churn(
    data: PredictionRequest,
    request: Request,
    user: dict = Depends(authenticate),
):
    """
    **Step 2.** Submit customer data with a `model_id` from `GET /v1/models`.

    **Categorical field accepted values:**
    - `Gender`: `Male`, `Female`
    - `Subscription Type`: `Basic`, `Standard`, `Premium`
    - `Contract Length`: `Monthly`, `Quarterly`, `Annual`

    **Numeric field ranges:**
    - `Age`: 18 – 100
    - `Tenure`: 0 – 120 months
    - `Usage Frequency`: 0 – 30
    - `Support Calls`: 0 – 20
    - `Payment Delay`: 0 – 30 days
    - `Total Spend`: 0 – 10,000 USD
    - `Last Interaction`: 0 – 365 days
    """
    request_id = getattr(request.state, "request_id", None)

    # Authorization check
    if data.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "NOT_AUTHORIZED",
                "request_id": request_id,
                "message": (
                    f"Your role '{user['role']}' cannot use model '{data.model_id}'. "
                    f"Allowed models for your key: {user['allowed_models']}"
                ),
            },
        )

    # Model retrieval (from cache — fast)
    try:
        model = get_model(data.model_id)
    except (ValueError, RuntimeError) as e:
        status = 400 if isinstance(e, ValueError) else 503
        raise HTTPException(
            status_code=status,
            detail={"error": "MODEL_ERROR", "request_id": request_id, "message": str(e)},
        )

    # Build DataFrame using aliases so column names match training data
    df = pd.DataFrame([data.model_dump(exclude={"model_id"}, by_alias=True)])

    prediction = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)

    logger.info(
        "Prediction made",
        extra={
            "request_id": request_id,
            "model_id": data.model_id,
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