import pandas as pd

from src.config import TARGET_COLUMN


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply basic cleaning and type coercions used in all pipelines."""
	cleaned = df.copy()
	cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
	cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)
	cleaned = cleaned.dropna(subset=["amount", "timestamp"])

	if TARGET_COLUMN in cleaned.columns:
		cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

	return cleaned.reset_index(drop=True)

