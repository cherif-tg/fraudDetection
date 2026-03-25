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


def select_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Pick threshold maximizing F1 on validation scores."""
	thresholds = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
	best_threshold = 0.5
	best_f1 = -1.0

	for th in thresholds:
		metrics = compute_metrics(y_true=y_true, y_score=y_score, threshold=float(th))
		if metrics.f1 > best_f1:
			best_f1 = metrics.f1
			best_threshold = float(th)

	return best_threshold

