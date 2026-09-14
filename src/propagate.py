"""Sparsity pattern propagation.

A tracked tensor of shape S carries P in bool^(N x n), where N = prod(S) and n
is the number of independent scalar inputs. Rows follow row-major order, the
same order as reshape(-1). P[i, j] set means d(out_i)/d(x_j) may be nonzero.

Soundness is one containment: every (i, j) whose derivative is nonzero anywhere
in the domain must be set in P. An extra entry costs a color. A missing one is a
wrong Jacobian.

Every rule below is a left-multiply by a boolean incidence matrix M, where
M[i, k] says output element i draws from input element k. Two shapes of M cover
the whole set. A row map has one entry per row and is exact. A coupling has
several and is conservative by construction, since it unions instead of
tracking which term actually contributes.
"""

import numpy as np

from . import _boolcsr as bc

# What each rule claims, in the sense the tests check. "exact": equals the true
# pattern at a generic point. "structural": exact unless terms cancel, as in
# x - x. "conservative": a superset by design, here because a coupling never
# reads which term is nonzero and mm never reads an untracked operand's zeros.
TIGHTNESS = {
    "gather": "exact",
    "reduce_sum": "exact",
    "pointwise": "structural",
    "couple": "conservative",
    "mm": "conservative",
}

__all__ = [
    "TIGHTNESS",
    "bcast_src",
    "reduce_src",
    "gather",
    "union",
    "couple",
    "pointwise",
    "reduce_sum",
    "mm",
]


def _n(shape):
    return int(np.prod(shape, dtype=np.int64)) if len(shape) else 1


# --- index maps --------------------------------------------------------------


def bcast_src(shape_in, shape_out):
    """Input element feeding each output element under broadcasting."""
    idx = np.arange(_n(shape_in), dtype=np.int64).reshape(shape_in)
    return np.broadcast_to(idx, shape_out).reshape(-1)


def reduce_src(shape, dims):
    """Output element each input element contributes to, reducing over `dims`."""
    dims = tuple(d % len(shape) for d in dims)
    kept = tuple(s for d, s in enumerate(shape) if d not in dims)
    out = np.arange(_n(kept), dtype=np.int64).reshape(kept)
    for d in sorted(dims):  # put the reduced axes back as size 1
        out = np.expand_dims(out, d)
    return np.broadcast_to(out, shape).reshape(-1)


# --- primitives --------------------------------------------------------------


def gather(P, src):
    """Row map. Output element i has exactly the pattern of input element src[i].

    Exact. Covers reshape, permute, transpose, expand, slicing and concrete
    indexing, since each output element is one input element with derivative 1.
    """
    return bc.gather(P, src)


def union(*Ps):
    """Elementwise union of patterns that already share a row space."""
    return bc.union(*Ps)


def couple(P, out_idx, in_idx, m_out):
    """Coupling. Output element out_idx[k] unions input element in_idx[k].

    Conservative: it records that the output may depend on every listed input,
    without asking which term is actually nonzero.
    """
    M = bc.from_pairs(out_idx, in_idx, (m_out, P.shape[0]))
    return bc.matmul(M, P)


# --- rules -------------------------------------------------------------------


def pointwise(out_shape, terms):
    """Pointwise op over broadcast operands.

    `terms` is a list of (P, shape) for the tracked operands. Untracked operands
    are left out: they contribute no derivative. Structural: exact unless the op
    cancels, as x - x does, or a term's derivative is identically zero.
    """
    n_out = _n(out_shape)
    parts = [gather(P, bcast_src(shape, out_shape)) for P, shape in terms]
    if not parts:
        raise ValueError("pointwise needs at least one tracked operand")
    out = union(*parts)
    assert out.shape[0] == n_out
    return out


def reduce_sum(P, shape, dims):
    """Sum over `dims`. Each output element unions the slice that fed it.

    Exact for sum: every derivative in the slice is 1. Any other reduction
    (mean, prod, amax) has the same incidence and is conservative at best.
    """
    src = reduce_src(shape, dims)
    n_out = int(src.max()) + 1 if src.size else 1
    return couple(P, src, np.arange(src.size, dtype=np.int64), n_out)


def mm(Pa, Pb, m, k, n):
    """Matrix product (m, k) @ (k, n), either operand tracked or both.

    out[i, l] sums a[i, j] * b[j, l] over j, so it may depend on all of a's row i
    and all of b's column l. Pass None for an untracked operand.

    The union runs over every j, including positions where the untracked operand
    happens to be zero at this point. Reading those zeros would prune entries
    that come back the moment a weight changes.
    """
    if Pa is None and Pb is None:
        raise ValueError("mm needs at least one tracked operand")
    i, l = np.divmod(np.arange(m * n, dtype=np.int64), n)
    j = np.arange(k, dtype=np.int64)
    out_idx = np.repeat(np.arange(m * n, dtype=np.int64), k)
    parts = []
    if Pa is not None:
        parts.append(couple(Pa, out_idx, (i[:, None] * k + j).reshape(-1), m * n))
    if Pb is not None:
        parts.append(couple(Pb, out_idx, (j[None, :] * n + l[:, None]).reshape(-1), m * n))
    return union(*parts)
