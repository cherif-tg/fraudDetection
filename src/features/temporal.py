import pandas as pd


def build_temporal_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
	"""Create time-based features from a timestamp column."""
	out = df.copy()
	ts = pd.to_datetime(out[timestamp_col], errors="coerce", utc=True)
	out["tx_hour"] = ts.dt.hour
	out["tx_day_of_week"] = ts.dt.dayofweek
	out["tx_is_weekend"] = (out["tx_day_of_week"] >= 5).astype(int)
	out["tx_is_night"] = ((out["tx_hour"] <= 6) | (out["tx_hour"] >= 23)).astype(int)
	return out

