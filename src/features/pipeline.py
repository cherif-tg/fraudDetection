from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN
from src.features.temporal import build_temporal_features
from src.features.statistical import build_statistical_features


@dataclass
class FeatureOutput:
	X: pd.DataFrame
	y: pd.Series


FEATURE_COLUMNS = [
	"amount",
	"tx_hour",
	"tx_day_of_week",
	"tx_is_weekend",
	"tx_is_night",
	"amount_log1p",
	"amount_zscore",
]


def prepare_training_features(df: pd.DataFrame) -> FeatureOutput:
	enriched = build_temporal_features(df)
	enriched = build_statistical_features(enriched)
	X = enriched[FEATURE_COLUMNS].copy()
	y = enriched[TARGET_COLUMN].copy()
	return FeatureOutput(X=X, y=y)


def prepare_inference_features(df: pd.DataFrame) -> pd.DataFrame:
	enriched = build_temporal_features(df)
	enriched = build_statistical_features(enriched)
	return enriched[FEATURE_COLUMNS].copy()


def build_model_pipeline() -> Pipeline:
	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[("num", numeric_transformer, FEATURE_COLUMNS)],
		remainder="drop",
	)

	model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
	return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

