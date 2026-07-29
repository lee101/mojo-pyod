"""Histogram-based outlier detection."""

from __future__ import annotations

import numpy as np

from .._lib import addr, check_status, f64, i64, lib
from ._neighbors import check_2d
from .base import BaseDetector


def _optimal_n_bins(values):
    upper = int(np.sqrt(values.shape[0]))
    likelihood = np.zeros(max(upper - 1, 1))
    n = values.shape[0]
    for index, bins in enumerate(range(1, upper)):
        histogram, _ = np.histogram(values, bins=bins)
        likelihood[index] = np.sum(
            histogram * np.log(bins * histogram / n + 1)
            - (bins - 1 + np.log(bins) ** 2.5)
        )
    return int(np.argmax(likelihood) + 1)


class HBOS(BaseDetector):
    def __init__(self, n_bins=10, alpha=0.1, tol=0.5, contamination=0.1):
        super().__init__(contamination=contamination)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < tol < 1:
            raise ValueError("tol must be in (0, 1)")
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol

    def fit(self, X, y=None):
        X = check_2d(X)
        self.n_features_in_ = X.shape[1]
        if isinstance(self.n_bins, str):
            if self.n_bins.lower() != "auto":
                raise ValueError("n_bins must be an integer >= 2 or 'auto'")
            self.hist_ = []
            self.bin_edges_ = []
            for feature in range(X.shape[1]):
                bins = _optimal_n_bins(X[:, feature])
                hist, edges = np.histogram(X[:, feature], bins=bins, density=True)
                self.hist_.append(f64(hist))
                self.bin_edges_.append(f64(edges))
            self._bins = i64([len(hist) for hist in self.hist_])
            self._edge_offsets = i64(
                np.r_[0, np.cumsum([len(edges) for edges in self.bin_edges_])[:-1]]
            )
            self._hist_offsets = i64(
                np.r_[0, np.cumsum([len(hist) for hist in self.hist_])[:-1]]
            )
            self._edges = f64(np.concatenate(self.bin_edges_))
            self._log_hist = f64(
                np.log2(np.concatenate(self.hist_) + self.alpha)
            )
        else:
            bins = int(self.n_bins)
            if bins < 2:
                raise ValueError("n_bins must be at least 2")
            self.hist_ = np.zeros((bins, X.shape[1]), dtype=np.float64)
            self.bin_edges_ = np.zeros((bins + 1, X.shape[1]), dtype=np.float64)
            for feature in range(X.shape[1]):
                self.hist_[:, feature], self.bin_edges_[:, feature] = np.histogram(
                    X[:, feature], bins=bins, density=True
                )
            self._edges = f64(self.bin_edges_)
            self._log_hist = f64(np.log2(self.hist_ + self.alpha))
        self.decision_scores_ = self._score(X)
        self._process_decision_scores()
        return self

    def _score(self, X):
        scores = np.empty(X.shape[0], dtype=np.float64)
        if isinstance(self.n_bins, str):
            status = lib().mpy_hbos_score_auto(
                addr(X),
                addr(self._edges),
                addr(self._log_hist),
                addr(self._edge_offsets),
                addr(self._hist_offsets),
                addr(self._bins),
                addr(scores),
                X.shape[0],
                X.shape[1],
                float(self.alpha),
                float(self.tol),
            )
        else:
            status = lib().mpy_hbos_score(
                addr(X),
                addr(self._edges),
                addr(self._log_hist),
                addr(scores),
                X.shape[0],
                X.shape[1],
                int(self.n_bins),
                float(self.alpha),
                float(self.tol),
            )
        check_status(status, "HBOS kernel")
        return scores

    def decision_function(self, X):
        if not hasattr(self, "hist_"):
            raise RuntimeError("estimator is not fitted")
        X = check_2d(X, features=self.n_features_in_)
        return self._score(X)
