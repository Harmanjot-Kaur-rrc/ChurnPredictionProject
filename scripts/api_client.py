"""
scripts/api_client.py
─────────────────────
Simple reusable client for the Churn Prediction API.
Usage: import and call predict_churn() from other scripts.
"""
import requests

API_URL = "http://127.0.0.1:8000"


def list_models(api_key: str) -> dict:
    """Step 1 — fetch available models for this API key."""
    response = requests.get(
        f"{API_URL}/v1/models",
        headers={"x-api-key": api_key},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def predict_churn(payload: dict, api_key: str) -> dict | None:
    """
    Step 2 — predict churn for a single customer.

    payload must include model_id plus all customer fields.
    Example:
        predict_churn({
            "model_id": "rf",
            "Age": 35,
            "Gender": "Male",
            "Tenure": 24,
            "Usage Frequency": 10,
            "Support Calls": 2,
            "Payment Delay": 5,
            "Subscription Type": "Standard",
            "Contract Length": "Annual",
            "Total Spend": 500.0,
            "Last Interaction": 30,
        }, api_key="analyst-key-456")
    """
    try:
        response = requests.post(
            f"{API_URL}/v1/predict",
            json=payload,
            headers={"x-api-key": api_key},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error {e.response.status_code}: {e.response.json()}")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot reach API. Is uvicorn running on localhost:8000?")
        return None
