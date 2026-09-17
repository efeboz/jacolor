"""Distance-2 coloring of a Jacobian sparsity pattern.

A set of columns may share one AD seed vector iff no two of them have a nonzero
in the same row. That condition is distance-1 coloring of the column-intersection
graph A = P^T P, whose edge (c, c') exists iff columns c and c' share a row. So
no distance-2 walk over the bipartite pattern is needed.

Rows are the same construction on P @ P^T, which is column coloring of P^T.
"""

import heapq
import warnings

import numpy as np

from . import _boolcsr as bc

__all__ = ["Coloring", "color_cols", "color_rows", "ORDERS"]

# Orderings tried by default. The one giving the fewest colors wins.
ORDERS = ("natural", "lf", "sl")

# Above this many columns the pure-Python greedy loop gets slow.
_SLOW_ABOVE = 10_000

# Refuse to build a column-intersection graph larger than this. At roughly 9
# bytes an entry that is about 1.8 GB. Measured patterns sit near a million.
_MAX_EDGES = 200_000_000


def _conflict(L, colors):
    # Some row of L holds two entries of one color, so decompression would read
    # a single compressed value back for two different Jacobian entries.
    if L.nnz == 0:
        return False
    rows = np.repeat(np.arange(L.shape[0], dtype=np.int64), bc.row_nnz(L))
    keys = rows * (int(colors.max()) + 1) + colors[L.indices]
    return np.unique(keys).size != keys.size


class Coloring:
    """A coloring together with the pattern it is valid for.

    The pattern is copied in and checked once here, so decompression can never
    pair a coloring with a pattern it does not fit.
    """

    __slots__ = ("colors", "n_colors", "lower_bound", "order", "axis", "pattern")

    def __init__(self, colors, n_colors, lower_bound, order, axis, pattern):
        if axis not in ("cols", "rows"):
            raise ValueError(f"axis must be 'cols' or 'rows', got {axis!r}")
        P = bc.check(pattern).copy()
        L = P if axis == "cols" else bc.transpose(P)  # colored lines are L's columns
        colors = np.asarray(colors)
        if not np.issubdtype(colors.dtype, np.integer):
            raise ValueError(f"colors must be integers, got dtype {colors.dtype}")
        if colors.ndim != 1:
            raise ValueError(f"colors must be one-dimensional, got shape {colors.shape}")
        colors = colors.astype(np.int64)
        if colors.size != L.shape[1]:
            raise ValueError(f"{colors.size} colors for {L.shape[1]} {axis}")
        # A label outside the range leaves its lines unseeded, so their entries
        # would never be written and would keep whatever the buffer held.
        if colors.size and (colors.min() < 0 or colors.max() >= n_colors):
            raise ValueError(
                f"colors must lie in 0 to {n_colors - 1}, got "
                f"{colors.min()} to {colors.max()}"
            )
        if _conflict(L, colors):
            other = "row" if axis == "cols" else "column"
            raise ValueError(f"coloring is invalid: two {axis} sharing a {other} have one color")
        colors = colors.copy()  # the caller keeps their own array
        colors.flags.writeable = False  # and cannot edit past the checks above
        self.colors = colors  # color index per column (or row), shape (n,)
        self.n_colors = n_colors
        self.lower_bound = lower_bound  # densest line of P: no coloring beats it
        self.order = order  # ordering that produced this result
        self.axis = axis  # "cols" for forward mode, "rows" for reverse
        self.pattern = P  # the pattern, in its original orientation

    def __repr__(self):
        return (
            f"Coloring(axis={self.axis!r}, n_colors={self.n_colors}, "
            f"lower_bound={self.lower_bound}, order={self.order!r})"
        )


def _greedy_py(indptr, indices, perm, n):
    # Greedy distance-1 coloring with a stamped forbidden array: stamp[k] == v
    # means color k is taken by an already-colored neighbour of v, which avoids
    # clearing the array per vertex. Self-loops are skipped because v is still
    # uncolored when its own edge is read.
    colors = np.full(n, -1, dtype=np.int64)
    stamp = np.full(n + 1, -1, dtype=np.int64)
    for i in range(n):
        v = perm[i]
        for j in range(indptr[v], indptr[v + 1]):
            cu = colors[indices[j]]
            if cu >= 0:
                stamp[cu] = v
        k = 0
        while stamp[k] == v:
            k += 1
        colors[v] = k
    return colors


try:  # jacolor[fast]
    from numba import njit

    _greedy = njit(cache=True)(_greedy_py)
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the no-numba install
    _greedy = _greedy_py
    HAS_NUMBA = False


def _degrees(A):
    # Neighbours excluding the self-loop, which A carries for every nonempty column.
    return np.maximum(bc.row_nnz(A).astype(np.int64) - 1, 0)


def _smallest_last(indptr, indices, deg):
    # Repeatedly strip a minimum-degree vertex. The order is the reverse of
    # removal. Lazy-deletion heap, so stale entries are dropped when popped.
    n = deg.size
    d = deg.copy()
    alive = np.ones(n, dtype=bool)
    heap = [(int(d[v]), int(v)) for v in range(n)]
    heapq.heapify(heap)
    out = np.empty(n, dtype=np.int64)
    k = n
    while k > 0:
        dv, v = heapq.heappop(heap)
        if not alive[v] or dv != d[v]:
            continue
        alive[v] = False
        k -= 1
        out[k] = v
        for j in range(indptr[v], indptr[v + 1]):
            u = indices[j]
            if alive[u]:
                d[u] -= 1
                heapq.heappush(heap, (int(d[u]), int(u)))
    return out


def _perm(name, A, deg):
    if name == "natural":
        return np.arange(deg.size, dtype=np.int64)
    if name == "lf":  # largest-first
        return np.argsort(-deg, kind="stable").astype(np.int64)
    if name == "sl":  # smallest-last
        return _smallest_last(A.indptr, A.indices, deg)
    raise ValueError(f"unknown ordering {name!r}, expected one of {ORDERS}")


def _best(P, orders):
    # Greedy over each ordering on A = P^T P, keeping the fewest colors.
    P = bc.check(P)
    n = P.shape[1]
    if n > _SLOW_ABOVE and not HAS_NUMBA:
        warnings.warn(
            f"coloring {n} columns with the pure-Python greedy loop. "
            "Install jacolor[fast] for the numba kernel",
            stacklevel=3,
        )
    # A row with k nonzeros contributes at most k * k pairs to A, so the cost is
    # known before the product is built. The bound is loose where columns share
    # many rows, which is exactly where the product is cheap anyway.
    rn = bc.row_nnz(P).astype(np.int64)
    # The graph cannot hold more than one entry per ordered pair of columns, so
    # the pair count is only a bound while the columns outnumber the pairs.
    edges = min(int((rn * rn).sum()), n * n)
    if edges > _MAX_EDGES:
        raise MemoryError(
            f"the column-intersection graph could hold up to {edges} entries, about "
            f"{edges * 9 / 1e9:.1f} GB. The densest row has {int(rn.max())} nonzeros, "
            f"so no coloring can use fewer than {int(rn.max())} colors here."
        )
    A = bc.matmul(bc.transpose(P), P)
    deg = _degrees(A)
    lb = int(bc.row_nnz(P).max()) if P.shape[0] else 0

    best = None
    for name in orders:
        colors = _greedy(A.indptr, A.indices, _perm(name, A, deg), n)
        k = int(colors.max()) + 1 if n else 0
        if best is None or k < best[1]:
            best = (colors, k, lb, name)
    return best


def color_cols(P, orders=ORDERS):
    """Color the columns of pattern P for forward-mode seeding."""
    return Coloring(*_best(P, orders), "cols", P)


def color_rows(P, orders=ORDERS):
    """Color the rows of pattern P for reverse-mode seeding."""
    return Coloring(*_best(bc.transpose(P), orders), "rows", P)
