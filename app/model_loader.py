"""
app/model_loader.py
────────────────────
Model registry + startup cache.
Models are loaded from disk ONCE at startup — not on every request.
"""
from __future__ import annotations

import logging
import os

import joblib

from app.config import get_settings

logger = logging.getLogger("churn_api")

# ── Registry: model_id → metadata ──────────────────────────────────────────
def _build_registry() -> dict[str, dict]:
    model_dir = get_settings().model_dir
    return {
        "logreg": {"path": f"{model_dir}/logreg.pkl", "description": "Logistic Regression"},
        "rf":     {"path": f"{model_dir}/rf.pkl",     "description": "Random Forest"},
        "gb":     {"path": f"{model_dir}/gb.pkl",     "description": "Gradient Boosting"},
        "xgb":    {"path": f"{model_dir}/xgb.pkl",    "description": "XGBoost"},
        "mlp":    {"path": f"{model_dir}/mlp.pkl",    "description": "Neural Network (MLP)"},
    }


MODEL_REGISTRY = _build_registry()

# ── In-memory cache populated at startup ───────────────────────────────────
_model_cache: dict[str, object] = {}


def load_all_models() -> None:
    """
    Called once during app startup (via lifespan).
    Loads every model into memory so predictions are fast.
    """
    for model_id, meta in MODEL_REGISTRY.items():
        path = meta["path"]
        if not os.path.exists(path):
            logger.warning("Model artifact missing — skipping", extra={"model_id": model_id, "path": path})
            continue
        _model_cache[model_id] = joblib.load(path)
        logger.info("Model loaded", extra={"model_id": model_id, "path": path})

    logger.info("Model cache ready", extra={"loaded": list(_model_cache.keys())})


def get_model(model_id: str):
    """
    Retrieve a model from the in-memory cache.
    Raises ValueError for unknown IDs, RuntimeError if not loaded.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )
    if model_id not in _model_cache:
        raise RuntimeError(
            f"Model '{model_id}' is registered but its artifact file was not found. "
            f"Run train_pipeline.py to generate model files."
        )
    return _model_cache[model_id]