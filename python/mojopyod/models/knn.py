"""K-nearest-neighbor outlier detector."""

from __future__ import annotations

import numpy as np

from ._neighbors import BruteNeighbors, check_2d
from .base import BaseDetector


class KNN(BaseDetector):
    def __init__(
        self,
        contamination=0.1,
        n_neighbors=5,
        method="largest",
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=1,
    ):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.neigh_ = BruteNeighbors(n_neighbors, metric, p)

    def _get_dist_by_method(self, distances):
        if self.method == "largest":
            return distances[:, -1]
        if self.method == "mean":
            return np.mean(distances, axis=1)
        if self.method == "median":
            return np.median(distances, axis=1)
        raise ValueError("method must be one of 'largest', 'mean', or 'median'")

    def fit(self, X, y=None):
        if self.metric_params is not None:
            raise NotImplementedError("metric_params is not covered by the Mojo backend")
        X = check_2d(X)
        self.n_features_in_ = X.shape[1]
        self.neigh_.fit(X)
        self._X = self.neigh_._fit_X
        distances, _ = self.neigh_.kneighbors(
            n_neighbors=int(self.n_neighbors), return_distance=True
        )
        self.decision_scores_ = self._get_dist_by_method(distances).ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if not hasattr(self, "decision_scores_"):
            raise RuntimeError("estimator is not fitted")
        X = check_2d(X, features=self.n_features_in_)
        distances, _ = self.neigh_.kneighbors(
            X, n_neighbors=int(self.n_neighbors), return_distance=True
        )
        return self._get_dist_by_method(distances).ravel()
