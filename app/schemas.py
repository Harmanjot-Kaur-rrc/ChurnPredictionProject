"""
app/schemas.py
──────────────
All request / response models with:
  - Field-level validation (ranges, allowed values)
  - Swagger examples so users know exactly what to send
  - Consistent error response shape
"""
from __future__ import annotations
from typing import Literal, List
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ─────────────────────────────────────────────────────────────
# Allowed categorical values — single source of truth
# ─────────────────────────────────────────────────────────────
GENDER_VALUES        = Literal["Male", "Female"]
SUBSCRIPTION_VALUES  = Literal["Basic", "Standard", "Premium"]
CONTRACT_VALUES      = Literal["Monthly", "Quarterly", "Annual"]


# ─────────────────────────────────────────────────────────────
# Customer data with full field validation
# ─────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    """
    All fields are validated. Categorical fields only accept the
    exact values listed. Numeric fields are bounded to realistic ranges.
    """

    Age: int = Field(
        ...,
        ge=18, le=100,
        description="Customer age in years. Must be between 18 and 100.",
        examples=[35],
    )
    Gender: GENDER_VALUES = Field(
        ...,
        description="Customer gender. Accepted values: 'Male', 'Female'.",
        examples=["Male"],
    )
    Tenure: int = Field(
        ...,
        ge=0, le=120,
        description="Number of months the customer has been with the company (0–120).",
        examples=[24],
    )
    Usage_Frequency: int = Field(
        ...,
        alias="Usage Frequency",
        ge=0, le=30,
        description="Number of times the customer used the service in the last month (0–30).",
        examples=[10],
    )
    Support_Calls: int = Field(
        ...,
        alias="Support Calls",
        ge=0, le=20,
        description="Number of support calls made in the last month (0–20).",
        examples=[2],
    )
    Payment_Delay: int = Field(
        ...,
        alias="Payment Delay",
        ge=0, le=30,
        description="Number of days payment was delayed (0–30).",
        examples=[5],
    )
    Subscription_Type: SUBSCRIPTION_VALUES = Field(
        ...,
        alias="Subscription Type",
        description="Customer subscription tier. Accepted values: 'Basic', 'Standard', 'Premium'.",
        examples=["Standard"],
    )
    Contract_Length: CONTRACT_VALUES = Field(
        ...,
        alias="Contract Length",
        description="Contract duration. Accepted values: 'Monthly', 'Quarterly', 'Annual'.",
        examples=["Annual"],
    )
    Total_Spend: float = Field(
        ...,
        alias="Total Spend",
        ge=0.0, le=10000.0,
        description="Total amount spent by the customer in USD (0–10,000).",
        examples=[500.0],
    )
    Last_Interaction: int = Field(
        ...,
        alias="Last Interaction",
        ge=0, le=365,
        description="Number of days since the customer last interacted with the service (0–365).",
        examples=[30],
    )

    model_config = ConfigDict(
        validate_by_name=True,
        json_schema_extra={
            "example": {
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
            }
        },
    )


# ─────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────
class PredictionRequest(CustomerData):
    model_id: str = Field(
        ...,
        description=(
            "ID of the model to use for prediction. "
            "Call GET /models first to see which model IDs you are authorized to use. "
            "Accepted values: 'logreg', 'rf', 'gb', 'xgb', 'mlp'."
        ),
        examples=["rf"],
    )

    model_config = ConfigDict(
        validate_by_name=True,
        json_schema_extra={
            "example": {
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
            }
        },
    )


# ─────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Short model identifier.", examples=["rf"])
    description: str = Field(..., description="Human-readable model name.", examples=["Random Forest"])


class ModelsResponse(BaseModel):
    available_models: List[ModelInfo] = Field(
        ..., description="All models registered in the system."
    )
    your_allowed_models: List[str] = Field(
        ..., description="Model IDs your API key is authorized to use."
    )


class PredictionResponse(BaseModel):
    model_id: str = Field(..., description="Model that produced this prediction.")
    churn_prediction: int = Field(..., description="1 = will churn, 0 = will stay.")
    churn_probability: float = Field(
        ..., description="Probability of churn (0.0 – 1.0)."
    )


# ─────────────────────────────────────────────────────────────
# Consistent error response — used in exception handlers
# ─────────────────────────────────────────────────────────────
class ErrorDetail(BaseModel):
    field: str | None = Field(None, description="Field that caused the error, if applicable.")
    message: str = Field(..., description="Human-readable error message.")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Short error code, e.g. VALIDATION_ERROR.")
    request_id: str | None = Field(None, description="Echo of X-Request-ID header for tracing.")
    details: List[ErrorDetail] = Field(default_factory=list)