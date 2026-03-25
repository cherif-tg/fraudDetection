from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


@dataclass
class EvalMetrics:
	auc_pr: float
	precision: float
	recall: float
	f1: float

	def as_dict(self) -> Dict[str, float]:
		return {
			"auc_pr": float(self.auc_pr),
			"precision": float(self.precision),
			"recall": float(self.recall),
			"f1": float(self.f1),
		}


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> EvalMetrics:
	y_pred = (y_score >= threshold).astype(int)
	return EvalMetrics(
		auc_pr=average_precision_score(y_true, y_score),
		precision=precision_score(y_true, y_pred, zero_division=0),
		recall=recall_score(y_true, y_pred, zero_division=0),
		f1=f1_score(y_true, y_pred, zero_division=0),
	)

