# mojo-pyod

`mojo-pyod` is a focused port of compute-heavy [PyOD](https://pyod.readthedocs.io/)
outlier detectors to Mojo. It keeps the familiar estimator workflow and
constructor signatures while moving neighbor search and histogram scoring into
a compiled shared library.

This is an independent open-source package named `mojopyod`; it does not shadow
the upstream `pyod` installation, so both implementations can be imported in
the same process for validation.

## Coverage

The covered estimators are:

- `mojopyod.models.knn.KNN`: `largest`, `mean`, and `median` scoring with
  Euclidean or Manhattan distance.
- `mojopyod.models.lof.LOF`: training and novelty scoring with Euclidean or
  Manhattan distance.
- `mojopyod.models.hbos.HBOS`: fixed histogram counts and PyOD's per-feature
  `n_bins="auto"` selection.
- Shared PyOD behavior including `decision_scores_`, `threshold_`, `labels_`,
  `predict`, confidence values, `fit_predict`, both `predict_proba` methods,
  `get_params`, and `set_params`.

The KNN and LOF constructors retain PyOD's full signatures. The Mojo backend is
always a bounded-memory brute-force scan, so `algorithm`, `leaf_size`, `n_jobs`,
and `metric_params` do not select another search implementation. Supported
metrics are `euclidean`/`l2`/Minkowski `p=2` and
`manhattan`/`cityblock`/`l1`/Minkowski `p=1`; other metrics raise
`NotImplementedError`.

PyOD's many other model families, sparse matrices, callable metrics,
precomputed distance matrices, PyThresh contamination objects, and estimator
methods not named above are not covered. Input matrices are converted to
C-contiguous `float64`; complex values and floating-point dtypes wider than
`float64` are rejected instead of silently narrowed. This package does not
wrap upstream estimators to create the appearance of broader coverage.

## Install

The repository pins the Mojo nightly used to build the shared library and
installs PyOD 3.6 for parity tests and benchmarks.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-pyod.so`. Set `MOJOPYOD_LIB` to an
already-built library at another location when deploying the Python package
separately from this checkout.

## Usage

```python
import numpy as np
from mojopyod.models.knn import KNN

rng = np.random.default_rng(7)
X = rng.normal(size=(500, 8))
X[-10:] += 6.0

detector = KNN(contamination=0.02, n_neighbors=10, method="largest")
labels = detector.fit_predict(X)

print(labels.sum())
print(detector.decision_scores_[labels == 1])
```

Run the example from the repository with
`pixi run python examples/quickstart.py`, or use the same imports in any
command launched through the Pixi environment.

## Benchmarks

Measured by running `pixi run bench` on this checkout on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86-64. Times are the best of three warmed runs.
A ratio above 1 means the Mojo-backed implementation was faster. PyOD 3.6.2
used scikit-learn's brute neighbor backend for the KNN and LOF comparisons.

| Case | mojo-pyod | PyOD | PyOD / Mojo |
|---|---:|---:|---:|
| KNN decision, Euclidean (6k train, 750 query, 24d) | 6.8 ms | 19.2 ms | 2.83x |
| KNN decision, Manhattan (6k train, 750 query, 24d) | 8.2 ms | 15.7 ms | 1.92x |
| KNN fit (2.5k x 24, k=10) | 8.8 ms | 44.4 ms | 5.05x |
| LOF fit (2.5k x 24, k=20) | 10.6 ms | 44.4 ms | 4.18x |
| HBOS decision (600k x 16, 10 bins) | 20.2 ms | 551.2 ms | 27.28x |

These measurements are specific to this CPU, array sizes, feature counts, and
PyOD/scikit-learn versions. The Mojo kernels win all five cases in this run.
Tree search can still be a better choice than this package's brute scan for
lower-dimensional or much larger reference sets.

There is no GPU path. Neighbor distance scans perform well below two FLOPs per
byte moved, while HBOS is a branchy histogram lookup with small cached tables.
Neither has enough arithmetic intensity to justify device transfers, so
execution remains on the CPU.

## How it works

All kernels live in one Mojo compilation unit. The build emits one C-ABI shared
library, and a small `ctypes` layer calls its exported functions. NumPy owns all
input, output, and scratch memory; buffers cross the ABI as integer addresses
and Mojo reconstructs typed pointers inside each exported function.

Feature matrices are C-contiguous row-major `float64`. Neighbor indices are
row-major `int64`, while distances and scores are `float64`. KNN uses the
machine's native SIMD width across features, including a scalar remainder,
and maintains a sorted `k`-element result without materializing an all-pairs
distance matrix. Euclidean search ranks squared distances and takes square
roots only for the final neighbors. Large independent query batches run in
coarse parallel chunks, while small batches stay serial. LOF reuses those
neighbors for reachability-density calculations. HBOS precomputes invariant
log densities and per-feature minima in Python, then performs SIMD bin lookup
and score reduction in Mojo. Large row batches use coarse parallel chunks;
small batches stay serial.
The Python boundary keeps every NumPy owner alive for the synchronous call,
passes only contiguous buffers, and checks shapes, dtypes, pointer addresses,
and the status returned by each kernel.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares numerical scores, fitted attributes, thresholds,
labels, predictions, probability conversion, and public signatures directly
against the installed upstream PyOD.
