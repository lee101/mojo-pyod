"""Shared PyOD-compatible detector behavior."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.special import erf
from scipy.stats import binom


class BaseDetector:
    def __init__(self, contamination=0.1):
        if not isinstance(contamination, (float, int)) or not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.contamination = contamination
        self._classes = 2

    def _process_decision_scores(self):
        self.threshold_ = np.percentile(
            self.decision_scores_, 100 * (1 - self.contamination)
        )
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int).ravel()
        self._mu = float(np.mean(self.decision_scores_))
        self._sigma = float(np.std(self.decision_scores_))
        return self

    def predict(self, X, return_confidence=False):
        prediction = (self.decision_function(X) > self.threshold_).astype(int).ravel()
        if return_confidence:
            return prediction, self.predict_confidence(X)
        return prediction

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict_proba(self, X, method="linear", return_confidence=False):
        scores = self.decision_function(X)
        probs = np.empty((len(scores), 2), dtype=np.float64)
        if method == "linear":
            low = float(np.min(self.decision_scores_))
            high = float(np.max(self.decision_scores_))
            if high == low:
                outlier = np.zeros_like(scores)
            else:
                outlier = np.clip((scores - low) / (high - low), 0.0, 1.0)
        elif method == "unify":
            if self._sigma == 0:
                outlier = np.zeros_like(scores)
            else:
                outlier = np.clip(
                    erf((scores - self._mu) / (self._sigma * np.sqrt(2))), 0.0, 1.0
                )
        else:
            raise ValueError(method, "is not a valid probability conversion method")
        probs[:, 1] = outlier
        probs[:, 0] = 1.0 - outlier
        if return_confidence:
            return probs, self.predict_confidence(X)
        return probs

    def predict_confidence(self, X):
        scores = self.decision_function(X)
        sorted_scores = np.sort(self.decision_scores_)
        ranks = np.searchsorted(sorted_scores, scores, side="right")
        posterior = (1.0 + ranks) / (2.0 + len(sorted_scores))
        confidence = 1.0 - binom.cdf(
            len(sorted_scores) - int(len(sorted_scores) * self.contamination),
            len(sorted_scores),
            posterior,
        )
        prediction = (scores > self.threshold_).astype(int).ravel()
        confidence[prediction == 0] = 1.0 - confidence[prediction == 0]
        return confidence

    def get_params(self, deep=True):
        params = {}
        for name in inspect.signature(self.__init__).parameters:
            if name != "self":
                params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"invalid parameter {name!r}")
            setattr(self, name, value)
        return self
