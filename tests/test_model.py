import pandas as pd

from src.features.pipeline import build_model_pipeline


def test_model_pipeline_fit_predict() -> None:
	X = pd.DataFrame(
		{
			"amount": [10, 12, 500, 520],
			"tx_hour": [10, 12, 2, 3],
			"tx_day_of_week": [1, 1, 5, 6],
			"tx_is_weekend": [0, 0, 1, 1],
			"tx_is_night": [0, 0, 1, 1],
			"amount_log1p": [2.3, 2.5, 6.2, 6.3],
			"amount_zscore": [-0.7, -0.6, 1.3, 1.4],
		}
	)
	y = [0, 0, 1, 1]

	model = build_model_pipeline()
	model.fit(X, y)
	scores = model.predict_proba(X)[:, 1]

	assert len(scores) == 4
	assert (scores >= 0).all() and (scores <= 1).all()

