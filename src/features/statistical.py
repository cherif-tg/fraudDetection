import numpy as np
import pandas as pd


def build_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Create baseline statistical features useful for fraud classification."""
	out = df.copy()
	out["amount_log1p"] = np.log1p(out["amount"].clip(lower=0))

	global_std = out["amount"].std(ddof=0)
	if global_std and not np.isnan(global_std):
		out["amount_zscore"] = (out["amount"] - out["amount"].mean()) / global_std
	else:
		out["amount_zscore"] = 0.0

	return out

