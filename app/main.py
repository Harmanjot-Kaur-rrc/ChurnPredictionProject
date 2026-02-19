from fastapi import FastAPI
import pandas as pd

from app.schemas import CustomerData, PredictionResponse
from app.model_loader import load_model

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using ML",
    version="1.0"
)

model = load_model()

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict(by_alias=True)])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
