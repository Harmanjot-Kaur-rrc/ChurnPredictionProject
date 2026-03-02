from pydantic import BaseModel, Field, ConfigDict
from typing import List


class CustomerData(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    Usage_Frequency: float = Field(..., alias="Usage Frequency")
    Support_Calls: float = Field(..., alias="Support Calls")
    Payment_Delay: float = Field(..., alias="Payment Delay")
    Subscription_Type: str = Field(..., alias="Subscription Type")
    Contract_Length: str = Field(..., alias="Contract Length")
    Total_Spend: float = Field(..., alias="Total Spend")
    Last_Interaction: float = Field(..., alias="Last Interaction")

    model_config = ConfigDict(validate_by_name=True)


class ModelInfo(BaseModel):
    model_id: str
    description: str


class ModelsResponse(BaseModel):
    available_models: List[ModelInfo]
    your_allowed_models: List[str]


class PredictionRequest(CustomerData):
    model_id: str


class PredictionResponse(BaseModel):
    model_id: str
    churn_prediction: int
    churn_probability: float