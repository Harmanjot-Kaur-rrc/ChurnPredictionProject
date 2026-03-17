"""
model_loader.py — Model cache backed by the model_versions table.

Changes from original:
  - get_model() now loads the ACTIVE version for a model_id from the DB.
  - load_all_models() reads model_versions.is_active to find the right .pkl.
  - get_active_version() returns the ModelVersion row for a model_id.
  - promote_version() atomically swaps is_active and reloads the cache.
"""
from __future__ import annotations

import logging
import os

import joblib

from app.config import get_settings

logger = logging.getLogger("churn_api")

# ── Registry: model_id → metadata (static, just descriptions) ─────────────
MODEL_REGISTRY: dict[str, dict] = {
    "logreg": {"description": "Logistic Regression"},
    "rf":     {"description": "Random Forest"},
    "gb":     {"description": "Gradient Boosting"},
    "xgb":    {"description": "XGBoost"},
    "mlp":    {"description": "Neural Network (MLP)"},
}

# ── In-memory cache: model_id → loaded pipeline object ────────────────────
_model_cache: dict[str, object] = {}


def load_all_models() -> None:
    """
    Called at startup. Loads the ACTIVE version of each model into memory.
    Falls back to the legacy {model_id}.pkl if no version row exists yet.
    """
    from app.database import ModelVersion, SessionLocal

    db = SessionLocal()
    try:
        for model_id in MODEL_REGISTRY:
            active: ModelVersion | None = (
                db.query(ModelVersion)
                .filter_by(model_id=model_id, is_active=True)
                .first()
            )
            if active:
                path = active.artifact_path
            else:
                # Legacy fallback (before versioning was introduced)
                path = os.path.join(get_settings().model_dir, f"{model_id}.pkl")

            if not os.path.exists(path):
                logger.warning("Model artifact missing — skipping",
                               extra={"model_id": model_id, "path": path})
                continue

            _model_cache[model_id] = joblib.load(path)
            logger.info("Model loaded", extra={"model_id": model_id, "path": path})
    finally:
        db.close()

    logger.info("Model cache ready", extra={"loaded": list(_model_cache.keys())})


def get_model(model_id: str):
    """Retrieve from in-memory cache. Raises on unknown / missing models."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Valid: {list(MODEL_REGISTRY.keys())}"
        )
    if model_id not in _model_cache:
        raise RuntimeError(
            f"Model '{model_id}' artifact not found. Run train_pipeline.py first."
        )
    return _model_cache[model_id]


def get_active_version(model_id: str) -> dict | None:
    """
    Returns a dict describing the active version of model_id, or None.
    Used by predict and batch endpoints to snapshot which version was used.
    """
    from app.database import ModelVersion, SessionLocal

    db = SessionLocal()
    try:
        row: ModelVersion | None = (
            db.query(ModelVersion)
            .filter_by(model_id=model_id, is_active=True)
            .first()
        )
        if not row:
            return None
        return {
            "model_id":       row.model_id,
            "version_number": row.version_number,
            "artifact_path":  row.artifact_path,
            "trained_by":     row.trained_by,
            "train_rows":     row.train_rows,
            "notes":          row.notes,
            "created_at":     row.created_at.isoformat(),
        }
    finally:
        db.close()


def promote_version(model_id: str, version_number: int) -> None:
    """
    Atomically sets is_active=True for the given version and False for all
    others of the same model_id, then reloads the in-memory cache.
    Raises ValueError if the version doesn't exist or the artifact is missing.
    """
    from app.database import ModelVersion, SessionLocal

    db = SessionLocal()
    try:
        target: ModelVersion | None = (
            db.query(ModelVersion)
            .filter_by(model_id=model_id, version_number=version_number)
            .first()
        )
        if not target:
            raise ValueError(
                f"Version {version_number} of model '{model_id}' not found."
            )
        if not os.path.exists(target.artifact_path):
            raise ValueError(
                f"Artifact file missing for version {version_number}: {target.artifact_path}"
            )

        # Deactivate all other versions of this model
        db.query(ModelVersion).filter(
            ModelVersion.model_id == model_id,
            ModelVersion.version_number != version_number,
        ).update({"is_active": False})

        target.is_active = True
        db.commit()

        # Hot-swap in-memory cache
        _model_cache[model_id] = joblib.load(target.artifact_path)
        logger.info(
            "Model promoted",
            extra={"model_id": model_id, "version": version_number, "path": target.artifact_path},
        )
    finally:
        db.close()