import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN
from src.data.loader import load_transactions_csv, validate_minimum_schema
from src.data.preprocessor import basic_clean
from src.features.pipeline import build_model_pipeline, prepare_training_features
from src.models.evaluate import compute_metrics


def train_model(input_csv: str | Path, output_model: str | Path) -> dict:
	df = load_transactions_csv(input_csv)
	validate_minimum_schema(df)
	df = basic_clean(df)

	feat = prepare_training_features(df)
	X_train, X_valid, y_train, y_valid = train_test_split(
		feat.X,
		feat.y,
		test_size=0.2,
		random_state=42,
		stratify=feat.y,
	)

	pipeline = build_model_pipeline()
	pipeline.fit(X_train, y_train)

	valid_scores = pipeline.predict_proba(X_valid)[:, 1]
	metrics = compute_metrics(y_true=y_valid.to_numpy(), y_score=valid_scores)

	output_path = Path(output_model)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, output_path)

	report = metrics.as_dict()
	report["target"] = TARGET_COLUMN
	report["model_path"] = str(output_path)
	return report


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train fraud model baseline pipeline")
	parser.add_argument("--input", required=True, help="Path to raw CSV")
	parser.add_argument("--output", required=True, help="Path to output model (.joblib)")
	return parser


if __name__ == "__main__":
	args = build_parser().parse_args()
	summary = train_model(input_csv=args.input, output_model=args.output)
	print("Training complete")
	print(summary)
