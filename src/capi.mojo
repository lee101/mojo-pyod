"""Compute kernels exported to the Python bindings.

Python owns every buffer. Addresses cross the C ABI as Int values and are
rebuilt here with a concrete mutable origin.
"""

from max.algorithm import parallelize
from std.math import sqrt
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_WORK_THRESHOLD = 8_000_000
comptime HBOS_PARALLEL_WORK_THRESHOLD = 1_000_000
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
        var delta0 = a.unsafe_load[width=W](j) - b.unsafe_load[width=W](j)
        var delta1 = a.unsafe_load[width=W](j + W) - b.unsafe_load[width=W](
            j + W
        )
        acc0 += max(delta0, -delta0)
        acc1 += max(delta1, -delta1)
        j += 2 * W
    var acc = acc0 + acc1
    while j + W <= d:
        var delta = a.unsafe_load[width=W](j) - b.unsafe_load[width=W](j)
        acc += max(delta, -delta)
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[unsafe_offset=j] - b[unsafe_offset=j]
        total += delta if delta >= 0.0 else -delta
        j += 1
    return total


def squared_euclidean_distance(a: FPtr, b: FPtr, d: Int) -> Float64:
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + 2 * W <= d:
        var delta0 = a.unsafe_load[width=W](j) - b.unsafe_load[width=W](j)
        var delta1 = a.unsafe_load[width=W](j + W) - b.unsafe_load[width=W](
            j + W
        )
        acc0 += delta0 * delta0
        acc1 += delta1 * delta1
        j += 2 * W
    var acc = acc0 + acc1
    while j + W <= d:
        var delta = a.unsafe_load[width=W](j) - b.unsafe_load[width=W](j)
        acc += delta * delta
        j += W
    var total = acc.reduce_add()
    while j < d:
        var delta = a[unsafe_offset=j] - b[unsafe_offset=j]
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
    @__parameter
    def process_query[is_manhattan: Bool](q: Int):
        var base = q * k
        for s in range(k):
            distances[unsafe_offset=base + s] = 1.7976931348623157e308
            indices[unsafe_offset=base + s] = -1

        for r in range(n):
            if exclude_self and q == r:
                continue
            var dist: Float64
            comptime if is_manhattan:
                dist = manhattan_distance(
                    query.unsafe_offset(q * d), train.unsafe_offset(r * d), d
                )
            else:
                dist = squared_euclidean_distance(
                    query.unsafe_offset(q * d), train.unsafe_offset(r * d), d
                )
            if dist >= distances[unsafe_offset=base + k - 1]:
                continue
            var s = k - 1
            while s > 0 and distances[unsafe_offset=base + s - 1] > dist:
                distances[unsafe_offset=base + s] = distances[
                    unsafe_offset=base + s - 1
                ]
                indices[unsafe_offset=base + s] = indices[
                    unsafe_offset=base + s - 1
                ]
                s -= 1
            distances[unsafe_offset=base + s] = dist
            indices[unsafe_offset=base + s] = Int64(r)

        comptime if not is_manhattan:
            for s in range(k):
                distances[unsafe_offset=base + s] = sqrt(
                    distances[unsafe_offset=base + s]
                )

    @__parameter
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
        train_addr == 0
        or query_addr == 0
        or distances_addr == 0
        or indices_addr == 0
        or n <= 0
        or d <= 0
        or m <= 0
        or k <= 0
        or k > n - (1 if exclude_self != 0 else 0)
        or (metric != 1 and metric != 2)
    ):
        return -1
    initialize_runtime()
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


def hbos_score(
    x: FPtr,
    edges: FPtr,
    hist: FPtr,
    minima: FPtr,
    scores: FPtr,
    n: Int,
    d: Int,
    bins: Int,
    tolerance: Float64,
):
    @__parameter
    def process_row(row: Int):
        var vector_total = SIMD[DType.float64, W](0.0)
        var feature = 0
        while feature + W <= d:
            var values = x.unsafe_load[width=W](row * d + feature)
            var minimum = minima.unsafe_load[width=W](feature)
            var first_edge = edges.unsafe_load[width=W](feature)
            var first_hist = hist.unsafe_load[width=W](feature)
            var first_width = (
                edges.unsafe_load[width=W](d + feature) - first_edge
            )
            var near_first = values.le(first_edge) & (
                (first_edge - values).le(first_width * tolerance)
            )
            var selected = near_first.select(first_hist, minimum)
            for b in range(bins):
                var edge = edges.unsafe_load[width=W](b * d + feature)
                var density = hist.unsafe_load[width=W](b * d + feature)
                selected = values.gt(edge).select(density, selected)

            var last_edge = edges.unsafe_load[width=W](bins * d + feature)
            var last_hist = hist.unsafe_load[width=W](
                (bins - 1) * d + feature
            )
            var last_width = last_edge - edges.unsafe_load[width=W](
                (bins - 1) * d + feature
            )
            var above = values.gt(last_edge)
            var near_last = (values - last_edge).le(last_width * tolerance)
            selected = above.select(
                near_last.select(last_hist, minimum), selected
            )
            vector_total += selected
            feature += W

        var total = vector_total.reduce_add()
        while feature < d:
            var value = x[unsafe_offset=row * d + feature]
            var bin_index = 0
            while (
                bin_index < bins + 1
                and value > edges[unsafe_offset=bin_index * d + feature]
            ):
                bin_index += 1

            var selected = minima[unsafe_offset=feature]
            if bin_index == 0:
                var width = (
                    edges[unsafe_offset=d + feature]
                    - edges[unsafe_offset=feature]
                )
                if edges[unsafe_offset=feature] - value <= width * tolerance:
                    selected = hist[unsafe_offset=feature]
            elif bin_index == bins + 1:
                var last_edge = edges[unsafe_offset=bins * d + feature]
                var width = (
                    last_edge - edges[unsafe_offset=(bins - 1) * d + feature]
                )
                if value - last_edge <= width * tolerance:
                    selected = hist[unsafe_offset=(bins - 1) * d + feature]
            else:
                selected = hist[unsafe_offset=(bin_index - 1) * d + feature]
            total += selected
            feature += 1
        scores[unsafe_offset=row] = -total

    @__parameter
    def process_chunk(chunk_index: Int):
        var chunk_count = min(n, PARALLEL_WORKERS)
        var chunk_size = (n + chunk_count - 1) // chunk_count
        var first = chunk_index * chunk_size
        var last = min(first + chunk_size, n)
        for row in range(first, last):
            process_row(row)

    if n * d >= HBOS_PARALLEL_WORK_THRESHOLD:
        parallelize[process_chunk](
            min(n, PARALLEL_WORKERS), min(n, PARALLEL_WORKERS)
        )
    else:
        for row in range(n):
            process_row(row)


@export("mpy_hbos_score")
def mpy_hbos_score(
    x_addr: Int,
    edges_addr: Int,
    hist_addr: Int,
    minima_addr: Int,
    scores_addr: Int,
    n: Int,
    d: Int,
    bins: Int,
    alpha: Float64,
    tolerance: Float64,
) abi("C") -> Int64:
    if (
        x_addr == 0
        or edges_addr == 0
        or hist_addr == 0
        or minima_addr == 0
        or scores_addr == 0
        or n <= 0
        or d <= 0
        or bins < 2
    ):
        return -1
    initialize_runtime()
    hbos_score(
        fp(x_addr),
        fp(edges_addr),
        fp(hist_addr),
        fp(minima_addr),
        fp(scores_addr),
        n,
        d,
        bins,
        tolerance,
    )
    return 0


@export("mpy_hbos_score_auto")
def mpy_hbos_score_auto(
    x_addr: Int,
    edges_addr: Int,
    hist_addr: Int,
    minima_addr: Int,
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
        x_addr == 0
        or edges_addr == 0
        or hist_addr == 0
        or minima_addr == 0
        or edge_offsets_addr == 0
        or hist_offsets_addr == 0
        or bins_addr == 0
        or scores_addr == 0
        or n <= 0
        or d <= 0
    ):
        return -1
    var x = fp(x_addr)
    var edges = fp(edges_addr)
    var hist = fp(hist_addr)
    var minima = fp(minima_addr)
    var edge_offsets = ip(edge_offsets_addr)
    var hist_offsets = ip(hist_offsets_addr)
    var bin_counts = ip(bins_addr)
    var scores = fp(scores_addr)

    @__parameter
    def process_row(row: Int):
        var total = 0.0
        for feature in range(d):
            var bins = Int(bin_counts[unsafe_offset=feature])
            var edge_start = Int(edge_offsets[unsafe_offset=feature])
            var hist_start = Int(hist_offsets[unsafe_offset=feature])
            var value = x[unsafe_offset=row * d + feature]
            var bin_index = 0
            while (
                bin_index < bins + 1
                and value > edges[unsafe_offset=edge_start + bin_index]
            ):
                bin_index += 1

            var selected = minima[unsafe_offset=feature]
            if bin_index == 0:
                var width = (
                    edges[unsafe_offset=edge_start + 1]
                    - edges[unsafe_offset=edge_start]
                )
                if edges[unsafe_offset=edge_start] - value <= width * tolerance:
                    selected = hist[unsafe_offset=hist_start]
            elif bin_index == bins + 1:
                var last_edge = edges[unsafe_offset=edge_start + bins]
                var width = (
                    last_edge - edges[unsafe_offset=edge_start + bins - 1]
                )
                if value - last_edge <= width * tolerance:
                    selected = hist[unsafe_offset=hist_start + bins - 1]
            else:
                selected = hist[unsafe_offset=hist_start + bin_index - 1]
            total += selected
        scores[unsafe_offset=row] = -total

    @__parameter
    def process_chunk(chunk_index: Int):
        var chunk_count = min(n, PARALLEL_WORKERS)
        var chunk_size = (n + chunk_count - 1) // chunk_count
        var first = chunk_index * chunk_size
        var last = min(first + chunk_size, n)
        for row in range(first, last):
            process_row(row)

    initialize_runtime()
    if n * d >= HBOS_PARALLEL_WORK_THRESHOLD:
        parallelize[process_chunk](
            min(n, PARALLEL_WORKERS), min(n, PARALLEL_WORKERS)
        )
    else:
        for row in range(n):
            process_row(row)
    return 0
