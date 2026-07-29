"""Numerical and behavioral parity with PyOD on identical inputs."""

import inspect

import numpy as np
import pytest

from mojopyod.models.hbos import HBOS
from mojopyod._lib import lib
from mojopyod.models._neighbors import BruteNeighbors
from mojopyod.models.knn import KNN
from mojopyod.models.lof import LOF
from pyod.models.hbos import HBOS as PyODHBOS
from pyod.models.knn import KNN as PyODKNN
from pyod.models.lof import LOF as PyODLOF
from sklearn.neighbors import NearestNeighbors


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(180, 7))
    train[-12:] += rng.normal(5.0, 0.4, size=(12, 7))
    query = rng.normal(size=(31, 7))
    query[-3:] += 6.0
    return train, query


@pytest.mark.parametrize("method", ["largest", "mean", "median"])
def test_knn_training_score_parity(data, method):
    X, _ = data
    ours = KNN(n_neighbors=9, method=method, algorithm="brute").fit(X)
    upstream = PyODKNN(n_neighbors=9, method=method, algorithm="brute").fit(X)
    assert np.allclose(ours.decision_scores_, upstream.decision_scores_, atol=1e-12)


@pytest.mark.parametrize(
    ("metric", "p"),
    [("minkowski", 2), ("euclidean", 2), ("minkowski", 1), ("manhattan", 2)],
)
def test_knn_query_score_parity(data, metric, p):
    X, Q = data
    ours = KNN(
        n_neighbors=6, method="mean", metric=metric, p=p, algorithm="brute"
    ).fit(X)
    upstream = PyODKNN(
        n_neighbors=6, method="mean", metric=metric, p=p, algorithm="brute"
    ).fit(X)
    assert np.allclose(
        ours.decision_function(Q), upstream.decision_function(Q), atol=1e-12
    )


def test_knn_labels_threshold_and_predict_parity(data):
    X, Q = data
    ours = KNN(contamination=0.12, n_neighbors=7).fit(X)
    upstream = PyODKNN(contamination=0.12, n_neighbors=7).fit(X)
    assert ours.threshold_ == pytest.approx(upstream.threshold_, abs=1e-12)
    assert np.array_equal(ours.labels_, upstream.labels_)
    assert np.array_equal(ours.predict(Q), upstream.predict(Q))


def test_knn_neighbor_indices_and_distances(data):
    X, Q = data
    ours = KNN(n_neighbors=5).fit(X)
    upstream = PyODKNN(n_neighbors=5, algorithm="brute").fit(X)
    od, oi = ours.neigh_.kneighbors(Q)
    pd, pi = upstream.neigh_.kneighbors(Q)
    assert np.allclose(od, pd, atol=1e-12)
    assert np.array_equal(oi, pi)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_simd_tail(metric):
    rng = np.random.default_rng(81)
    X = rng.normal(size=(73, 11))
    Q = rng.normal(size=(13, 11))
    actual_distances, actual_indices = BruteNeighbors(
        5, metric=metric, p=2
    ).fit(X).kneighbors(Q)
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=5, metric=metric, algorithm="brute"
    ).fit(X).kneighbors(Q)
    assert np.allclose(actual_distances, expected_distances, atol=1e-12)
    assert np.array_equal(actual_indices, expected_indices)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_knn_parallel_query_path(metric):
    rng = np.random.default_rng(82)
    X = rng.normal(size=(7_500, 17))
    Q = rng.normal(size=(64, 17))
    actual_distances, actual_indices = BruteNeighbors(
        7, metric=metric, p=2
    ).fit(X).kneighbors(Q)
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=7, metric=metric, algorithm="brute"
    ).fit(X).kneighbors(Q)
    assert np.allclose(actual_distances, expected_distances, atol=1e-12)
    assert np.array_equal(actual_indices, expected_indices)


def test_knn_parallel_exclude_self_path():
    X = np.random.default_rng(83).normal(size=(600, 24))
    actual_distances, actual_indices = BruteNeighbors(
        6, metric="euclidean", p=2
    ).fit(X).kneighbors()
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=6, metric="euclidean", algorithm="brute"
    ).fit(X).kneighbors()
    assert np.allclose(actual_distances, expected_distances, atol=1e-12)
    assert np.array_equal(actual_indices, expected_indices)


def test_knn_constructor_signature_matches_upstream():
    assert inspect.signature(KNN) == inspect.signature(PyODKNN)


@pytest.mark.parametrize("method", ["linear", "unify"])
def test_predict_proba_parity(data, method):
    X, Q = data
    ours = KNN(n_neighbors=5).fit(X)
    upstream = PyODKNN(n_neighbors=5).fit(X)
    assert np.allclose(
        ours.predict_proba(Q, method=method),
        upstream.predict_proba(Q, method=method),
        atol=1e-12,
    )


def test_predict_and_probability_confidence_parity(data):
    X, Q = data
    ours = KNN(n_neighbors=5).fit(X)
    upstream = PyODKNN(n_neighbors=5).fit(X)
    ours_labels, ours_confidence = ours.predict(Q, return_confidence=True)
    upstream_labels, upstream_confidence = upstream.predict(
        Q, return_confidence=True
    )
    assert np.array_equal(ours_labels, upstream_labels)
    assert np.allclose(ours_confidence, upstream_confidence, atol=1e-12)
    ours_proba, ours_confidence = ours.predict_proba(
        Q, return_confidence=True
    )
    upstream_proba, upstream_confidence = upstream.predict_proba(
        Q, return_confidence=True
    )
    assert np.allclose(ours_proba, upstream_proba, atol=1e-12)
    assert np.allclose(ours_confidence, upstream_confidence, atol=1e-12)


@pytest.mark.parametrize("p", [1, 2])
def test_lof_training_score_parity(data, p):
    X, _ = data
    ours = LOF(n_neighbors=13, p=p, algorithm="brute").fit(X)
    upstream = PyODLOF(n_neighbors=13, p=p, algorithm="brute").fit(X)
    assert np.allclose(ours.decision_scores_, upstream.decision_scores_, atol=1e-12)
    assert np.allclose(
        ours.negative_outlier_factor_,
        upstream.detector_.negative_outlier_factor_,
        atol=1e-12,
    )


@pytest.mark.parametrize("p", [1, 2])
def test_lof_novelty_score_parity(data, p):
    X, Q = data
    ours = LOF(n_neighbors=11, p=p, algorithm="brute").fit(X)
    upstream = PyODLOF(n_neighbors=11, p=p, algorithm="brute").fit(X)
    assert np.allclose(
        ours.decision_function(Q), upstream.decision_function(Q), atol=1e-12
    )


def test_lof_neighbor_clamping_matches_upstream():
    X = np.arange(18, dtype=np.float64).reshape(6, 3)
    with pytest.warns(UserWarning):
        ours = LOF(n_neighbors=20).fit(X)
    with pytest.warns(UserWarning):
        upstream = PyODLOF(n_neighbors=20).fit(X)
    assert ours.n_neighbors_ == upstream.n_neighbors_ == 5
    assert np.allclose(ours.decision_scores_, upstream.decision_scores_)


def test_lof_constructor_signature_matches_upstream():
    assert inspect.signature(LOF) == inspect.signature(PyODLOF)


@pytest.mark.parametrize("bins", [4, 10, 17])
def test_hbos_static_training_score_parity(data, bins):
    X, _ = data
    ours = HBOS(n_bins=bins, alpha=0.15, tol=0.35).fit(X)
    upstream = PyODHBOS(n_bins=bins, alpha=0.15, tol=0.35).fit(X)
    assert np.array_equal(ours.hist_, upstream.hist_)
    assert np.array_equal(ours.bin_edges_, upstream.bin_edges_)
    assert np.allclose(ours.decision_scores_, upstream.decision_scores_, atol=1e-12)


def test_hbos_static_query_and_outside_bin_parity(data):
    X, Q = data
    edges = np.vstack([Q, np.full((1, X.shape[1]), -100), np.full((1, X.shape[1]), 100)])
    ours = HBOS(n_bins=8, alpha=0.1, tol=0.5).fit(X)
    upstream = PyODHBOS(n_bins=8, alpha=0.1, tol=0.5).fit(X)
    assert np.allclose(
        ours.decision_function(edges),
        upstream.decision_function(edges),
        atol=1e-12,
    )


def test_hbos_auto_fit_and_query_parity(data):
    X, Q = data
    ours = HBOS(n_bins="auto").fit(X)
    upstream = PyODHBOS(n_bins="auto").fit(X)
    assert [len(v) for v in ours.hist_] == [len(v) for v in upstream.hist_]
    assert np.allclose(ours.decision_scores_, upstream.decision_scores_, atol=1e-12)
    assert np.allclose(
        ours.decision_function(Q), upstream.decision_function(Q), atol=1e-12
    )


def test_hbos_labels_and_predict_parity(data):
    X, Q = data
    ours = HBOS(n_bins=9, contamination=0.15).fit(X)
    upstream = PyODHBOS(n_bins=9, contamination=0.15).fit(X)
    assert ours.threshold_ == pytest.approx(upstream.threshold_, abs=1e-12)
    assert np.array_equal(ours.labels_, upstream.labels_)
    assert np.array_equal(ours.predict(Q), upstream.predict(Q))


def test_hbos_constructor_signature_matches_upstream():
    assert inspect.signature(HBOS) == inspect.signature(PyODHBOS)


def test_unsupported_metric_is_explicit(data):
    X, _ = data
    with pytest.raises(NotImplementedError, match="Euclidean and Manhattan"):
        KNN(metric="cosine").fit(X)


def test_fit_predict_and_parameter_protocol(data):
    X, _ = data
    detector = HBOS(n_bins=6)
    labels = detector.fit_predict(X)
    assert np.array_equal(labels, detector.labels_)
    assert detector.get_params()["n_bins"] == 6
    assert detector.set_params(n_bins=7) is detector


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.ones((3, 2), dtype=np.complex128),
        np.ones((3, 2), dtype=np.longdouble),
    ],
)
def test_ffi_rejects_empty_or_narrowed_inputs(bad):
    with pytest.raises((TypeError, ValueError)):
        KNN(n_neighbors=1).fit(bad)


def test_ffi_rejects_null_buffers_before_pointer_construction():
    assert lib().mpy_knn_distances(0, 0, 0, 0, 1, 1, 1, 1, 2, 0) == -1
    assert lib().mpy_hbos_score(0, 0, 0, 0, 1, 1, 2, 0.1, 0.5) == -1
    assert (
        lib().mpy_hbos_score_auto(0, 0, 0, 0, 0, 0, 0, 1, 1, 0.1, 0.5)
        == -1
    )


def test_noncontiguous_and_float32_buffers_are_normalized(data):
    X, Q = data
    X = X[:, ::2].astype(np.float32)
    Q = Q[:, ::2].astype(np.float32)
    ours = KNN(n_neighbors=5).fit(X)
    upstream = PyODKNN(n_neighbors=5, algorithm="brute").fit(X)
    assert ours._X.dtype == np.float64
    assert ours._X.flags.c_contiguous
    assert np.allclose(
        ours.decision_function(Q), upstream.decision_function(Q), atol=1e-6
    )
