from typing import Any

import pandas as pd


def compute_local_explanation_note(model: Any, row: pd.DataFrame) -> dict:
	"""Placeholder for SHAP integration in the next iteration."""
	_ = model
	_ = row
	return {
		"status": "not_implemented",
		"message": "SHAP integration will be added after baseline model validation.",
	}
