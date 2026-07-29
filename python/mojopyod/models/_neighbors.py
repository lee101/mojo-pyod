"""Brute-force neighbor queries backed by Mojo."""

from __future__ import annotations

import numpy as np

from .._lib import addr, check_status, f64, lib


def metric_code(metric, p):
    if metric in ("euclidean", "l2") or (metric == "minkowski" and p == 2):
        return 2
    if metric in ("manhattan", "cityblock", "l1") or (
        metric == "minkowski" and p == 1
    ):
        return 1
    raise NotImplementedError(
        "the Mojo backend covers Euclidean and Manhattan distances "
        "(including Minkowski p=2 and p=1)"
    )


def check_2d(X, *, features=None):
    original = np.asarray(X)
    if original.dtype.kind not in "buif":
        raise TypeError("feature matrices must contain real numeric values")
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        raise TypeError("floating-point dtypes wider than float64 are not supported")
    array = f64(X)
    if array.ndim != 2:
        raise ValueError("expected a 2-dimensional feature matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("feature matrices must contain samples and features")
    if not np.isfinite(array).all():
        raise ValueError("input contains NaN or infinity")
    if features is not None and array.shape[1] != features:
        raise ValueError(
            f"X has {array.shape[1]} features, but the estimator expects {features}"
        )
    return array


class BruteNeighbors:
    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X):
        self._fit_X = check_2d(X)
        self.n_samples_fit_ = self._fit_X.shape[0]
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        k = self.n_neighbors if n_neighbors is None else int(n_neighbors)
        exclude_self = X is None
        query = self._fit_X if exclude_self else check_2d(
            X, features=self._fit_X.shape[1]
        )
        n, d = self._fit_X.shape
        if k < 1 or k > n - int(exclude_self):
            raise ValueError("n_neighbors exceeds the available reference samples")
        distances = np.empty((query.shape[0], k), dtype=np.float64)
        indices = np.empty((query.shape[0], k), dtype=np.int64)
        status = lib().mpy_knn_distances(
            addr(self._fit_X),
            addr(query),
            addr(distances),
            addr(indices),
            n,
            d,
            query.shape[0],
            k,
            metric_code(self.metric, self.p),
            int(exclude_self),
        )
        check_status(status, "KNN kernel")
        if return_distance:
            return distances, indices
        return indices
