from pathlib import Path

import joblib
import pandas as pd

from src.features.pipeline import prepare_inference_features


def load_model(model_path: str | Path):
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Model not found: {path}")
	return joblib.load(path)


def predict_scores(model, raw_df: pd.DataFrame) -> pd.Series:
	features = prepare_inference_features(raw_df)
	proba = model.predict_proba(features)[:, 1]
	return pd.Series(proba, name="fraud_score")

