import pandas as pd

from src.features.temporal import build_temporal_features
from src.features.statistical import build_statistical_features


def test_temporal_features_created() -> None:
	df = pd.DataFrame(
		{
			"amount": [100.0],
			"timestamp": ["2026-01-01T03:30:00Z"],
		}
	)

	out = build_temporal_features(df)
	assert "tx_hour" in out.columns
	assert "tx_is_night" in out.columns
	assert int(out.loc[0, "tx_is_night"]) == 1


def test_statistical_features_created() -> None:
	df = pd.DataFrame(
		{
			"amount": [10.0, 20.0, 50.0],
			"timestamp": [
				"2026-01-01T00:00:00Z",
				"2026-01-01T01:00:00Z",
				"2026-01-01T02:00:00Z",
			],
		}
	)
	out = build_statistical_features(df)
	assert "amount_log1p" in out.columns
	assert "amount_zscore" in out.columns

