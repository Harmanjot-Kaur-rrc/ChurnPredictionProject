from fastapi import FastAPI, HTTPException, Depends
import pandas as pd

from app.auth import authenticate
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelsResponse,
    ModelInfo,
)
from app.model_loader import MODEL_REGISTRY, load_model

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Two-step churn prediction:\n"
        "1. `GET /models` — browse available models\n"
        "2. `POST /predict` — send chosen model_id with customer data"
    ),
    version="2.0",
)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "API is running"}


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
def list_models(user=Depends(authenticate)):
    """
    Step 1 — List all models in the registry.
    Returns all models plus the subset the caller is authorized to use.
    """
    all_models = [
        ModelInfo(model_id=mid, description=meta["description"])
        for mid, meta in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(
        available_models=all_models,
        your_allowed_models=user["allowed_models"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(data: PredictionRequest, user=Depends(authenticate)):
    """
    Step 2 — Submit customer data with a chosen model_id to get a churn prediction.
    Only models listed in your_allowed_models (from /models) are accessible.
    """
    if data.model_id not in user["allowed_models"]:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Your role '{user['role']}' is not authorized to use model "
                f"'{data.model_id}'. Allowed: {user['allowed_models']}"
            ),
        )

    try:
        model = load_model(data.model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = pd.DataFrame([data.model_dump(exclude={"model_id"}, by_alias=True)])

    prediction = int(model.predict(df)[0])
    probability = round(float(model.predict_proba(df)[0][1]), 4)

    return PredictionResponse(
        model_id=data.model_id,
        churn_prediction=prediction,
        churn_probability=probability,
    )

