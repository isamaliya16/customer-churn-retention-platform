"""Probability calibration (Platt scaling).

The classifiers are trained with `class_weight="balanced"`, which pushes predicted probabilities well above the
real churn frequency (e.g. a raw "0.80" customer churned ~60% of the time). `ProbabilityCalibrator` learns a
monotone logistic mapping raw -> calibrated probability from out-of-fold training predictions, so the number
shown to users means what it says: "of customers scored like this, about this share churn".
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ProbabilityCalibrator:
    """Platt scaling: calibrated = sigmoid(a * logit(raw) + b), fitted on out-of-fold predictions."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, raw_probabilities, y) -> "ProbabilityCalibrator":
        self._lr.fit(_logit(raw_probabilities), np.asarray(y))
        return self

    def transform(self, raw_probabilities) -> np.ndarray:
        return self._lr.predict_proba(_logit(raw_probabilities))[:, 1]
