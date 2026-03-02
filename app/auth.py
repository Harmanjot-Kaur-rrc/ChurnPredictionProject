from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

API_KEYS = {
    "admin-key-123": {
        "role": "admin",
        "allowed_models": ["logreg", "rf", "gb", "xgb", "mlp"]
    },
    "analyst-key-456": {
        "role": "analyst",
        "allowed_models": ["logreg", "rf", "xgb"]
    },
    "guest-key-789": {
        "role": "guest",
        "allowed_models": ["logreg"]
    }
}


def authenticate(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[api_key]