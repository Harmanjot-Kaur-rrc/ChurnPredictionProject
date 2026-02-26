import joblib

MODEL_REGISTRY = {
    "logreg": "models/logreg.pkl",
    "rf": "models/rf.pkl",
    "gb": "models/gb.pkl",
    "xgb": "models/xgb.pkl",
    "mlp": "models/mlp.pkl",
}

def load_model(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model_id: {model_id}")

    return joblib.load(MODEL_REGISTRY[model_id])
