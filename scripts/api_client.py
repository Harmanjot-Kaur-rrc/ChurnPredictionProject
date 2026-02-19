import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict_churn(payload):
    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Prediction failed:", e)
        return None
