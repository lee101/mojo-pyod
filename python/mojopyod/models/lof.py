"""Local Outlier Factor with Mojo brute-force neighbor search."""

from __future__ import annotations

import warnings

import numpy as np

from ._neighbors import BruteNeighbors, check_2d
from .base import BaseDetector


class LOF(BaseDetector):
    def __init__(
        self,
        n_neighbors=20,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination=0.1,
        n_jobs=1,
        novelty=True,
    ):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.novelty = novelty

    def fit(self, X, y=None):
        if self.metric_params is not None:
            raise NotImplementedError("metric_params is not covered by the Mojo backend")
        X = check_2d(X)
        n = X.shape[0]
        if n < 2:
            raise ValueError("LOF requires at least two samples")
        if self.n_neighbors > n:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than the total "
                f"number of samples ({n}); using n_samples - 1",
                UserWarning,
            )
        self.n_neighbors_ = max(1, min(int(self.n_neighbors), n - 1))
        self.n_features_in_ = X.shape[1]
        self.neigh_ = BruteNeighbors(self.n_neighbors_, self.metric, self.p).fit(X)
        self._fit_X = self.neigh_._fit_X

        distances, neighbors = self.neigh_.kneighbors(
            n_neighbors=self.n_neighbors_, return_distance=True
        )
        self._distances_fit_X_ = distances
        neighbor_k_distance = distances[neighbors, self.n_neighbors_ - 1]
        reachability = np.maximum(distances, neighbor_k_distance)
        self._lrd = 1.0 / (np.mean(reachability, axis=1) + 1e-10)
        self.decision_scores_ = np.mean(
            self._lrd[neighbors] / self._lrd[:, np.newaxis], axis=1
        )
        self.negative_outlier_factor_ = -self.decision_scores_
        self.detector_ = _DetectorView(self)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if not hasattr(self, "decision_scores_"):
            raise RuntimeError("estimator is not fitted")
        if not self.novelty:
            raise AttributeError(
                "decision_function is unavailable when novelty=False"
            )
        X = check_2d(X, features=self.n_features_in_)
        distances, neighbors = self.neigh_.kneighbors(
            X, n_neighbors=self.n_neighbors_, return_distance=True
        )
        neighbor_k_distance = self._distances_fit_X_[
            neighbors, self.n_neighbors_ - 1
        ]
        query_lrd = 1.0 / (
            np.mean(np.maximum(distances, neighbor_k_distance), axis=1) + 1e-10
        )
        return np.mean(
            self._lrd[neighbors] / query_lrd[:, np.newaxis], axis=1
        )


class _DetectorView:
    def __init__(self, owner):
        self.negative_outlier_factor_ = owner.negative_outlier_factor_
        self.n_neighbors_ = owner.n_neighbors_
        self._distances_fit_X_ = owner._distances_fit_X_
