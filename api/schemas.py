from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
	amount: float = Field(..., gt=0)
	timestamp: str
	customer_id: str | None = None
	merchant_id: str | None = None


class PredictResponse(BaseModel):
	fraud_score: float
	is_fraud: bool
	threshold: float

