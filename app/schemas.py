from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

GENDER_VALUES       = Literal["Male", "Female"]
SUBSCRIPTION_VALUES = Literal["Basic", "Standard", "Premium"]
CONTRACT_VALUES     = Literal["Monthly", "Quarterly", "Annual"]


class CustomerData(BaseModel):
    Age: int = Field(..., ge=18, le=100, examples=[35])
    Gender: GENDER_VALUES = Field(..., examples=["Male"])
    Tenure: int = Field(..., ge=0, le=120, examples=[24])
    Usage_Frequency: int = Field(..., alias="Usage Frequency", ge=0, le=30, examples=[10])
    Support_Calls: int = Field(..., alias="Support Calls", ge=0, le=20, examples=[2])
    Payment_Delay: int = Field(..., alias="Payment Delay", ge=0, le=30, examples=[5])
    Subscription_Type: SUBSCRIPTION_VALUES = Field(..., alias="Subscription Type", examples=["Standard"])
    Contract_Length: CONTRACT_VALUES = Field(..., alias="Contract Length", examples=["Annual"])
    Total_Spend: float = Field(..., alias="Total Spend", ge=0.0, le=10000.0, examples=[500.0])
    Last_Interaction: int = Field(..., alias="Last Interaction", ge=0, le=365, examples=[30])

    model_config = ConfigDict(
        validate_by_name=True,
        json_schema_extra={"example": {
            "Age": 35, "Gender": "Male", "Tenure": 24,
            "Usage Frequency": 10, "Support Calls": 2, "Payment Delay": 5,
            "Subscription Type": "Standard", "Contract Length": "Annual",
            "Total Spend": 500.0, "Last Interaction": 30,
        }},
    )


class PredictionRequest(CustomerData):
    model_id: str = Field(..., examples=["rf"])

    model_config = ConfigDict(
        validate_by_name=True,
        json_schema_extra={"example": {
            "model_id": "rf", "Age": 35, "Gender": "Male", "Tenure": 24,
            "Usage Frequency": 10, "Support Calls": 2, "Payment Delay": 5,
            "Subscription Type": "Standard", "Contract Length": "Annual",
            "Total Spend": 500.0, "Last Interaction": 30,
        }},
    )


class FeatureContribution(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str   # "increases churn" | "decreases churn"


class PredictionResponse(BaseModel):
    model_id:          str
    model_version:     Optional[int]   = Field(None, description="Version number used for this prediction.")
    churn_prediction:  int             = Field(..., description="1 = will churn, 0 = will stay.")
    churn_probability: float           = Field(..., description="Probability of churn (0.0 – 1.0).")
    explanation:       Optional[List[FeatureContribution]] = Field(
        None,
        description="SHAP feature contributions — only present when ?explain=true.",
    )


class ModelInfo(BaseModel):
    model_id:    str
    description: str


class ModelsResponse(BaseModel):
    available_models:    List[ModelInfo]
    your_allowed_models: List[str]


class ErrorDetail(BaseModel):
    field:   Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    error:      str
    request_id: Optional[str] = None
    details:    List[ErrorDetail] = Field(default_factory=list)