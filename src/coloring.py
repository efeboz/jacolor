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


class Coloring:
    """Result of coloring one axis of a pattern."""

    __slots__ = ("colors", "n_colors", "lower_bound", "order", "axis")

    def __init__(self, colors, n_colors, lower_bound, order, axis):
        self.colors = colors  # color index per column (or row), shape (n,)
        self.n_colors = n_colors
        self.lower_bound = lower_bound  # densest line of P: no coloring beats it
        self.order = order  # ordering that produced this result
        self.axis = axis  # "cols" for forward mode, "rows" for reverse

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


def color_cols(P, orders=ORDERS):
    """Color the columns of pattern `P` for forward-mode seeding."""
    P = bc.check(P)
    n = P.shape[1]
    if n > _SLOW_ABOVE and not HAS_NUMBA:
        warnings.warn(
            f"coloring {n} columns with the pure-Python greedy loop. "
            "Install jacolor[fast] for the numba kernel",
            stacklevel=2,
        )
    A = bc.matmul(bc.transpose(P), P)
    deg = _degrees(A)
    lb = int(bc.row_nnz(P).max()) if P.shape[0] else 0

    best = None
    for name in orders:
        colors = _greedy(A.indptr, A.indices, _perm(name, A, deg), n)
        n_colors = int(colors.max()) + 1 if n else 0
        if best is None or n_colors < best.n_colors:
            best = Coloring(colors, n_colors, lb, name, "cols")
    return best


def color_rows(P, orders=ORDERS):
    """Color the rows of pattern `P` for reverse-mode seeding."""
    c = color_cols(bc.transpose(P), orders=orders)
    return Coloring(c.colors, c.n_colors, c.lower_bound, c.order, "rows")
