from fastapi import FastAPI, HTTPException
import pandas as pd

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import load_model, MODEL_REGISTRY

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using ML",
    version="2.0"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.get("/models")
def list_models():
    return {
        "available_models": list(MODEL_REGISTRY.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: PredictionRequest):

    try:
        model = load_model(data.model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = pd.DataFrame([data.dict(exclude={"model_id"}, by_alias=True)])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
