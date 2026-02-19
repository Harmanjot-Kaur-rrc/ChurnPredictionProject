import joblib

MODEL_PATH = "models/best_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)
