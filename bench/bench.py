"""Benchmark Mojo-backed detectors against PyOD on the same arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojopyod import HBOS, KNN, LOF  # noqa: E402
from pyod.models.hbos import HBOS as PyODHBOS  # noqa: E402
from pyod.models.knn import KNN as PyODKNN  # noqa: E402
from pyod.models.lof import LOF as PyODLOF  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def matrix(rows, features, seed):
    return np.ascontiguousarray(
        np.random.default_rng(seed).normal(size=(rows, features))
    )


def cases():
    train = matrix(6_000, 24, 1)
    query = matrix(750, 24, 2)
    mojo_euclidean = KNN(n_neighbors=10, algorithm="brute").fit(train)
    pyod_euclidean = PyODKNN(n_neighbors=10, algorithm="brute").fit(train)
    yield (
        "KNN decision, Euclidean (6k train, 750 query, 24d)",
        lambda: mojo_euclidean.decision_function(query),
        lambda: pyod_euclidean.decision_function(query),
    )
    mojo_manhattan = KNN(
        n_neighbors=10, metric="manhattan", algorithm="brute"
    ).fit(train)
    pyod_manhattan = PyODKNN(
        n_neighbors=10, metric="manhattan", algorithm="brute"
    ).fit(train)
    yield (
        "KNN decision, Manhattan (6k train, 750 query, 24d)",
        lambda: mojo_manhattan.decision_function(query),
        lambda: pyod_manhattan.decision_function(query),
    )

    fit_data = matrix(2_500, 24, 3)
    yield (
        "KNN fit (2.5k x 24, k=10)",
        lambda: KNN(n_neighbors=10, algorithm="brute").fit(fit_data),
        lambda: PyODKNN(n_neighbors=10, algorithm="brute").fit(fit_data),
    )
    yield (
        "LOF fit (2.5k x 24, k=20)",
        lambda: LOF(n_neighbors=20, algorithm="brute").fit(fit_data),
        lambda: PyODLOF(n_neighbors=20, algorithm="brute").fit(fit_data),
    )

    hbos_train = matrix(80_000, 16, 4)
    hbos_query = matrix(600_000, 16, 5)
    ours_hbos = HBOS(n_bins=10).fit(hbos_train)
    pyod_hbos = PyODHBOS(n_bins=10).fit(hbos_train)
    yield (
        "HBOS decision (600k x 16, 10 bins)",
        lambda: ours_hbos.decision_function(hbos_query),
        lambda: pyod_hbos.decision_function(hbos_query),
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Case | mojo-pyod | PyOD | PyOD / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_seconds = best_time(ours)
        pyod_seconds = best_time(upstream)
        ratio = pyod_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.1f} ms | "
            f"{pyod_seconds * 1e3:.1f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
