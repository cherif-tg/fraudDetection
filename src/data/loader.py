from pathlib import Path
import pandas as pd

from src.config import TARGET_COLUMN


def load_transactions_csv(path: str | Path) -> pd.DataFrame:
	"""Load raw transaction data from CSV."""
	csv_path = Path(path)
	if not csv_path.exists():
		raise FileNotFoundError(f"Data file not found: {csv_path}")
	return pd.read_csv(csv_path)


def validate_minimum_schema(df: pd.DataFrame) -> None:
	"""Validate mandatory columns for baseline modeling."""
	required = {"amount", "timestamp", TARGET_COLUMN}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

