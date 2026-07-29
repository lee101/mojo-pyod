"""Compute kernels exported to the Python bindings.

Python owns every buffer. Addresses cross the C ABI as Int values and are
rebuilt here with a concrete mutable origin.
"""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_WORK_THRESHOLD = 8_000_000
comptime PARALLEL_WORKERS = 16


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def manhattan_distance(a: FPtr, b: FPtr, d: Int) -> Float64:
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + 2 * W <= d:
        var delta0 = a.load[width=W](j) - b.load[width=W](j)
        var delta1 = a.load[width=W](j + W) - b.load[width=W](j + W)
        acc0 += max(delta0, -delta0)
        acc1 += max(delta1, -delta1)
        j += 2 * W
    var acc = acc0 + acc1
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        acc += max(delta, -delta)
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        total += delta if delta >= 0.0 else -delta
        j += 1
    return total


def squared_euclidean_distance(a: FPtr, b: FPtr, d: Int) -> Float64:
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + 2 * W <= d:
        var delta0 = a.load[width=W](j) - b.load[width=W](j)
        var delta1 = a.load[width=W](j + W) - b.load[width=W](j + W)
        acc0 += delta0 * delta0
        acc1 += delta1 * delta1
        j += 2 * W
    var acc = acc0 + acc1
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        total += delta * delta
        j += 1
    return total


def knn_distances(
    train: FPtr,
    query: FPtr,
    distances: FPtr,
    indices: IPtr,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
    metric: Int,
    exclude_self: Bool,
):
    @parameter
    def process_query[is_manhattan: Bool](q: Int):
        var base = q * k
        for s in range(k):
            distances[base + s] = 1.7976931348623157e308
            indices[base + s] = -1

        for r in range(n):
            if exclude_self and q == r:
                continue
            var dist: Float64
            comptime if is_manhattan:
                dist = manhattan_distance(
                    query + q * d, train + r * d, d
                )
            else:
                dist = squared_euclidean_distance(
                    query + q * d, train + r * d, d
                )
            if dist >= distances[base + k - 1]:
                continue
            var s = k - 1
            while s > 0 and distances[base + s - 1] > dist:
                distances[base + s] = distances[base + s - 1]
                indices[base + s] = indices[base + s - 1]
                s -= 1
            distances[base + s] = dist
            indices[base + s] = Int64(r)

        comptime if not is_manhattan:
            for s in range(k):
                distances[base + s] = sqrt(distances[base + s])

    @parameter
    def process_chunk[is_manhattan: Bool](chunk_index: Int):
        var chunk_count = min(m, PARALLEL_WORKERS)
        var chunk_size = (m + chunk_count - 1) // chunk_count
        var first = chunk_index * chunk_size
        var last = min(first + chunk_size, m)
        for q in range(first, last):
            process_query[is_manhattan](q)

    if metric == 1:
        if m * n * d >= PARALLEL_WORK_THRESHOLD:
            parallelize[process_chunk[True]](
                min(m, PARALLEL_WORKERS), min(m, PARALLEL_WORKERS)
            )
        else:
            for q in range(m):
                process_query[True](q)
    else:
        if m * n * d >= PARALLEL_WORK_THRESHOLD:
            parallelize[process_chunk[False]](
                min(m, PARALLEL_WORKERS), min(m, PARALLEL_WORKERS)
            )
        else:
            for q in range(m):
                process_query[False](q)


@export("mpy_knn_distances")
def mpy_knn_distances(
    train_addr: Int,
    query_addr: Int,
    distances_addr: Int,
    indices_addr: Int,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
    metric: Int,
    exclude_self: Int,
) abi("C") -> Int64:
    if (
        train_addr == 0 or query_addr == 0 or distances_addr == 0
        or indices_addr == 0 or n <= 0 or d <= 0 or m <= 0 or k <= 0
        or k > n - (1 if exclude_self != 0 else 0)
        or (metric != 1 and metric != 2)
    ):
        return -1
    knn_distances(
        fp(train_addr),
        fp(query_addr),
        fp(distances_addr),
        ip(indices_addr),
        n,
        d,
        m,
        k,
        metric,
        exclude_self != 0,
    )
    return 0


@export("mpy_hbos_score")
def mpy_hbos_score(
    x_addr: Int,
    edges_addr: Int,
    hist_addr: Int,
    scores_addr: Int,
    n: Int,
    d: Int,
    bins: Int,
    alpha: Float64,
    tolerance: Float64,
) abi("C") -> Int64:
    if (
        x_addr == 0 or edges_addr == 0 or hist_addr == 0 or scores_addr == 0
        or n <= 0 or d <= 0 or bins < 2
    ):
        return -1
    var x = fp(x_addr)
    var edges = fp(edges_addr)
    var hist = fp(hist_addr)
    var scores = fp(scores_addr)

    for row in range(n):
        var total = 0.0
        for feature in range(d):
            var value = x[row * d + feature]
            var bin_index = 0
            while bin_index < bins + 1 and value > edges[bin_index * d + feature]:
                bin_index += 1

            var minimum = hist[feature]
            for b in range(1, bins):
                var candidate = hist[b * d + feature]
                if candidate < minimum:
                    minimum = candidate

            var selected = minimum
            if bin_index == 0:
                var width = edges[d + feature] - edges[feature]
                if edges[feature] - value <= width * tolerance:
                    selected = hist[feature]
            elif bin_index == bins + 1:
                var last_edge = edges[bins * d + feature]
                var width = last_edge - edges[(bins - 1) * d + feature]
                if value - last_edge <= width * tolerance:
                    selected = hist[(bins - 1) * d + feature]
            else:
                selected = hist[(bin_index - 1) * d + feature]
            total += selected
        scores[row] = -total
    return 0


@export("mpy_hbos_score_auto")
def mpy_hbos_score_auto(
    x_addr: Int,
    edges_addr: Int,
    hist_addr: Int,
    edge_offsets_addr: Int,
    hist_offsets_addr: Int,
    bins_addr: Int,
    scores_addr: Int,
    n: Int,
    d: Int,
    alpha: Float64,
    tolerance: Float64,
) abi("C") -> Int64:
    if (
        x_addr == 0 or edges_addr == 0 or hist_addr == 0
        or edge_offsets_addr == 0 or hist_offsets_addr == 0 or bins_addr == 0
        or scores_addr == 0 or n <= 0 or d <= 0
    ):
        return -1
    var x = fp(x_addr)
    var edges = fp(edges_addr)
    var hist = fp(hist_addr)
    var edge_offsets = ip(edge_offsets_addr)
    var hist_offsets = ip(hist_offsets_addr)
    var bin_counts = ip(bins_addr)
    var scores = fp(scores_addr)

    for row in range(n):
        var total = 0.0
        for feature in range(d):
            var bins = Int(bin_counts[feature])
            var edge_start = Int(edge_offsets[feature])
            var hist_start = Int(hist_offsets[feature])
            var value = x[row * d + feature]
            var bin_index = 0
            while bin_index < bins + 1 and value > edges[edge_start + bin_index]:
                bin_index += 1

            var minimum = hist[hist_start]
            for b in range(1, bins):
                var candidate = hist[hist_start + b]
                if candidate < minimum:
                    minimum = candidate

            var selected = minimum
            if bin_index == 0:
                var width = edges[edge_start + 1] - edges[edge_start]
                if edges[edge_start] - value <= width * tolerance:
                    selected = hist[hist_start]
            elif bin_index == bins + 1:
                var last_edge = edges[edge_start + bins]
                var width = last_edge - edges[edge_start + bins - 1]
                if value - last_edge <= width * tolerance:
                    selected = hist[hist_start + bins - 1]
            else:
                selected = hist[hist_start + bin_index - 1]
            total += selected
        scores[row] = -total
    return 0
