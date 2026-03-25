import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN
from src.data.loader import load_transactions_csv, validate_minimum_schema
from src.data.preprocessor import basic_clean
from src.features.pipeline import build_model_pipeline, prepare_training_features
from src.models.evaluate import compute_metrics, select_best_threshold


def split_train_valid(df, test_size: float, split_strategy: str):
	if split_strategy == "temporal":
		sorted_df = df.sort_values("timestamp").reset_index(drop=True)
		split_idx = int(len(sorted_df) * (1 - test_size))
		if split_idx <= 0 or split_idx >= len(sorted_df):
			raise ValueError("Invalid split index. Check test_size and dataset size.")
		train_df = sorted_df.iloc[:split_idx].copy()
		valid_df = sorted_df.iloc[split_idx:].copy()
		return train_df, valid_df

	train_df, valid_df = train_test_split(
		df,
		test_size=test_size,
		random_state=42,
		stratify=df[TARGET_COLUMN],
	)
	return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def train_model(
	input_csv: str | Path,
	output_model: str | Path,
	test_size: float = 0.2,
	split_strategy: str = "temporal",
	threshold: float | None = None,
) -> dict:
	df = load_transactions_csv(input_csv)
	validate_minimum_schema(df)
	df = basic_clean(df)
	train_df, valid_df = split_train_valid(df=df, test_size=test_size, split_strategy=split_strategy)

	train_feat = prepare_training_features(train_df)
	valid_feat = prepare_training_features(valid_df)
	X_train, y_train = train_feat.X, train_feat.y
	X_valid, y_valid = valid_feat.X, valid_feat.y

	pipeline = build_model_pipeline()
	pipeline.fit(X_train, y_train)

	valid_scores = pipeline.predict_proba(X_valid)[:, 1]
	y_valid_np = y_valid.to_numpy()
	selected_threshold = threshold if threshold is not None else select_best_threshold(y_true=y_valid_np, y_score=valid_scores)
	metrics = compute_metrics(y_true=y_valid_np, y_score=valid_scores, threshold=selected_threshold)

	output_path = Path(output_model)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, output_path)

	report = metrics.as_dict()
	report["target"] = TARGET_COLUMN
	report["model_path"] = str(output_path)
	report["split_strategy"] = split_strategy
	report["test_size"] = float(test_size)
	report["threshold"] = float(selected_threshold)
	return report


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train fraud model baseline pipeline")
	parser.add_argument("--input", required=True, help="Path to raw CSV")
	parser.add_argument("--output", required=True, help="Path to output model (.joblib)")
	parser.add_argument("--test-size", type=float, default=0.2, help="Validation ratio")
	parser.add_argument(
		"--split-strategy",
		choices=["temporal", "stratified"],
		default="temporal",
		help="Validation split strategy",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=None,
		help="Optional fixed decision threshold. If absent, threshold is optimized on validation set.",
	)
	return parser


if __name__ == "__main__":
	args = build_parser().parse_args()
	summary = train_model(
		input_csv=args.input,
		output_model=args.output,
		test_size=args.test_size,
		split_strategy=args.split_strategy,
		threshold=args.threshold,
	)
	print("Training complete")
	print(summary)
