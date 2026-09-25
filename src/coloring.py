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

__all__ = ["Coloring", "color_cols", "color_rows", "refine", "graph_estimate",
           "MAX_EDGES", "ORDERS"]

# Orderings tried by default. The one giving the fewest colors wins. "iterated"
# starts from the best of those before it, so it goes last.
ORDERS = ("natural", "lf", "sl", "iterated")

# Rounds of iterated greedy, and how many may pass without a gain before it stops.
_ROUNDS = 30
_PATIENCE = 10

# Moves tabu search may spend trying to reach each lower count.
_TABU_MOVES = 20_000

# Memory tabu search may hold: two int32 tables, active lines by colors.
_TABU_BYTES = 1 << 28

# Above this many columns the pure-Python greedy loop gets slow.
_SLOW_ABOVE = 10_000

# Refuse to build a column-intersection graph larger than this. At roughly 9
# bytes an entry that is about 1.8 GB. Measured patterns sit near a million.
MAX_EDGES = 200_000_000


def graph_estimate(P):
    """Entries the column-intersection graph of P could hold, at most.

    A row with k entries contributes at most k squared pairs, and the graph
    cannot hold more than one entry per ordered pair of columns. Cheap enough to
    ask before building anything, which is what lets a caller pick a direction.
    """
    rn = bc.row_nnz(bc.check(P)).astype(np.int64)
    return min(int((rn * rn).sum()), P.shape[1] ** 2)


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


def _iterated(A, colors):
    # Culberson's iterated greedy. Greedy over an order that keeps each color
    # class together never needs more colors than there were classes, so
    # reordering the classes can only keep or lower the count. Reversed, smallest
    # first and shuffled with a fixed seed, in turn.
    if colors.size == 0:
        return colors
    rng = np.random.default_rng(0)
    best = cur = colors
    idle = 0
    for r in range(_ROUNDS):
        k = int(cur.max()) + 1
        order = (np.arange(k)[::-1], np.argsort(np.bincount(cur, minlength=k), kind="stable"),
                 rng.permutation(k))[r % 3]
        rank = np.empty(k, dtype=np.int64)
        rank[order] = np.arange(k)
        perm = np.argsort(rank[cur], kind="stable").astype(np.int64)
        cur = _greedy(A.indptr, A.indices, perm, colors.size)
        if cur.max() < best.max():
            best, idle = cur, 0
        else:
            idle += 1
            if idle >= _PATIENCE:
                break
    return best


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
    rn = bc.row_nnz(P).astype(np.int64)
    edges = graph_estimate(P)
    if edges > MAX_EDGES:
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
        if name == "iterated":
            start = best[0] if best else _greedy(A.indptr, A.indices, _perm("natural", A, deg), n)
            colors = _iterated(A, start)
        else:
            colors = _greedy(A.indptr, A.indices, _perm(name, A, deg), n)
        k = int(colors.max()) + 1 if n else 0
        if best is None or k < best[1]:
            best = (colors, k, lb, f"{best[3]}+iterated" if name == "iterated" and best else name)
        if best[1] <= lb:  # the bound is reached, so no ordering can do better
            break
    return best


def color_cols(P, orders=ORDERS):
    """Color the columns of pattern P for forward-mode seeding."""
    return Coloring(*_best(P, orders), "cols", P)


def color_rows(P, orders=ORDERS):
    """Color the rows of pattern P for reverse-mode seeding."""
    return Coloring(*_best(bc.transpose(P), orders), "rows", P)


def _tabu(A, colors, k, moves, seed):
    """Look for a coloring with k colors, starting from colors. None if not found.

    TabuCol: count the conflicting pairs, and move one conflicting vertex at a
    time to the color that lowers the count most, forbidding a vertex to return
    to a color it just left for a while so the search cannot circle. A move that
    beats the best count so far is allowed even if forbidden.
    """
    rng = np.random.default_rng(seed)
    n = colors.size
    ip, ix = A.indptr, A.indices
    rows = np.repeat(np.arange(n), np.diff(ip))
    edge = rows != ix  # the diagonal is every column meeting itself
    c = np.where(colors < k, colors, rng.integers(0, k, n)).astype(np.int64)
    # g[v, j] counts the neighbours of v colored j.
    g = np.zeros((n, k), dtype=np.int32)
    np.add.at(g, (rows[edge], c[ix[edge]]), 1)
    conf = int(g[np.arange(n), c].sum()) // 2
    until = np.zeros((n, k), dtype=np.int32)
    best = conf
    big = np.iinfo(np.int32).max // 4
    for it in range(moves):
        if conf == 0:
            return c
        bad = np.flatnonzero(g[np.arange(n), c] > 0)
        d = g[bad] - g[bad, c[bad]][:, None]
        d[np.arange(bad.size), c[bad]] = big
        d = np.where((until[bad] <= it) | (conf + d < best), d, big)
        low = d.min()
        if low >= big:
            continue
        i, j = np.argwhere(d == low)[rng.integers(int((d == low).sum()))]
        v, old = bad[i], c[bad[i]]
        nb = ix[ip[v]:ip[v + 1]]
        nb = nb[nb != v]
        g[nb, old] -= 1
        g[nb, j] += 1
        c[v] = j
        conf += int(low)
        until[v, old] = it + int(0.6 * bad.size) + int(rng.integers(0, 10))
        best = min(best, conf)
    return c if conf == 0 else None


def refine(coloring, moves=_TABU_MOVES):
    """Try to use fewer colors than greedy found, by tabu search.

    Greedy colors each line once and never looks back, which on a periodic grid
    can leave it well above what is possible. This searches for one color fewer
    at a time, until it reaches the lower bound or a count it cannot reach
    within moves. Each attempt that fails costs the full budget, so this is for
    a coloring that will be used many times.

    Returns a new Coloring, or the same one when nothing was gained, or when the
    search would not fit in memory: its graph under MAX_EDGES, and its tables
    under 256 MB. The result goes through the same checks as any coloring, so
    the search can cost time but never a wrong Jacobian.
    """
    if coloring.n_colors <= coloring.lower_bound:  # nothing below it to find
        return coloring
    P = coloring.pattern
    L = P if coloring.axis == "cols" else bc.transpose(P)
    if graph_estimate(L) > MAX_EDGES:
        return coloring
    A = bc.matmul(bc.transpose(L), L)
    # A line that shares no row with another never conflicts. It keeps color 0
    # and stays out of the search, and out of its tables.
    active = np.flatnonzero(_degrees(A) > 0)
    if 8 * active.size * (coloring.n_colors - 1) > _TABU_BYTES:
        return coloring
    A = A[active][:, active]
    colors, k = coloring.colors[active], coloring.n_colors
    while k - 1 >= max(coloring.lower_bound, 1):
        got = _tabu(A, colors, k - 1, moves, seed=k)
        if got is None:
            break
        colors, k = got, k - 1
    if k == coloring.n_colors:
        return coloring
    full = np.zeros(coloring.colors.size, dtype=np.int64)
    full[active] = colors
    return Coloring(full, k, coloring.lower_bound, f"{coloring.order}+tabu",
                    coloring.axis, P)
