import joblib
import os

MODEL_REGISTRY = {
    "logreg": {"path": "models/logreg.pkl", "description": "Logistic Regression"},
    "rf":     {"path": "models/rf.pkl",     "description": "Random Forest"},
    "gb":     {"path": "models/gb.pkl",     "description": "Gradient Boosting"},
    "xgb":    {"path": "models/xgb.pkl",    "description": "XGBoost"},
    "mlp":    {"path": "models/mlp.pkl",    "description": "Neural Network (MLP)"},
}


def load_model(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model_id '{model_id}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    path = MODEL_REGISTRY[model_id]["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at '{path}'. "
                                f"Run train_pipeline.py first.")
    return joblib.load(path)
