import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_model
from api.schemas import PredictRequest, PredictResponse
from src.config import DEFAULT_THRESHOLD
from src.models.predict import predict_scores


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, model=Depends(get_model)) -> PredictResponse:
	try:
		raw = pd.DataFrame([payload.model_dump()])
		score = float(predict_scores(model=model, raw_df=raw).iloc[0])
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

	return PredictResponse(
		fraud_score=score,
		is_fraud=score >= DEFAULT_THRESHOLD,
		threshold=DEFAULT_THRESHOLD,
	)

