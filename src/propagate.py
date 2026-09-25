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
several and takes their union, so it is at best exact up to cancellation, and
conservative whenever it skips reading a value that could be zero.
"""

import numpy as np

from . import _boolcsr as bc

# What each rule claims about the pattern it propagates, given exact incoming
# patterns. "exact": stays exact, because every output row is one input row.
# "structural": exact unless terms cancel, as in x - x, because the rule takes a
# union. "conservative": a superset by design, because the rule never reads a
# value that could be zero.
TIGHTNESS = {
    "gather": "exact",
    "cat": "exact",
    "reduce_sum": "structural",
    "pointwise": "structural",
    "couple": "conservative",
    "mm": "conservative",
    "bmm": "conservative",
    "slice_couple": "conservative",
    "prefix": "structural",
    "conv2d": "conservative",
}

__all__ = [
    "TIGHTNESS",
    "bcast_src",
    "reduce_src",
    "gather",
    "union",
    "couple",
    "pointwise",
    "cat",
    "reduce_sum",
    "slice_couple",
    "prefix",
    "conv2d",
    "mm",
    "bmm",
]


def _n(shape):
    return int(np.prod(shape, dtype=np.int64)) if len(shape) else 1


# --- index maps --------------------------------------------------------------


def bcast_src(shape_in, shape_out):
    """Input element feeding each output element under broadcasting."""
    idx = np.arange(_n(shape_in), dtype=np.int64).reshape(shape_in)
    return np.broadcast_to(idx, shape_out).reshape(-1)


def _kept(shape, dims):
    dims = tuple(d % len(shape) for d in dims)
    return dims, tuple(v for d, v in enumerate(shape) if d not in dims)


def reduce_src(shape, dims):
    """Output element each input element contributes to, reducing over dims."""
    if len(shape) == 0:  # a scalar reduced over dim 0 or -1 is itself
        return np.zeros(1, dtype=np.int64)
    dims, kept = _kept(shape, dims)
    out = np.arange(_n(kept), dtype=np.int64).reshape(kept)
    for d in sorted(dims):  # put the reduced axes back as size 1
        out = np.expand_dims(out, d)
    return np.broadcast_to(out, shape).reshape(-1)


# --- primitives --------------------------------------------------------------


def gather(P, src):
    """Row map. Output element i has exactly the pattern of input element src[i].

    Exact. Covers reshape, permute, transpose, expand, slicing and concrete
    indexing, since each output element is one input element with derivative 1.

    A src of -1 means no input element, as for padding, and gives an empty row.
    Checked here, since numpy would read -1 as the last row.
    """
    src = np.asarray(src, dtype=np.int64)
    if src.size and src.min() < 0:
        P = bc.vstack([P, bc.from_pairs([], [], (1, P.shape[1]))])
        src = np.where(src < 0, P.shape[0] - 1, src)
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

    terms is a list of (P, shape) for the tracked operands. Untracked operands
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
    """Sum over dims. Each output element unions the slice that fed it.

    Structural. Its own Jacobian is all ones, but the slice it sums can cancel,
    as in stack([x, -x]).sum(0), and a union cannot see that.
    """
    # The output size comes from the shape, never from the contributions: an
    # empty slice contributes nothing and still owns its output rows.
    n_out = 1 if len(shape) == 0 else _n(_kept(shape, dims)[1])
    src = reduce_src(shape, dims)
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
    # Union each row of a and each column of b once, then hand every output its
    # pair. Pairing outputs with terms directly would cost m * n * k.
    parts = []
    if Pa is not None:
        parts.append(gather(reduce_sum(Pa, (m, k), (1,)), reduce_src((m, n), (1,))))
    if Pb is not None:
        parts.append(gather(reduce_sum(Pb, (k, n), (0,)), reduce_src((m, n), (0,))))
    return union(*parts)


def bmm(Pa, Pb, b, m, k, n):
    """Batched matrix product (b, m, k) @ (b, k, n), either operand tracked or both.

    mm within each batch, and batches never mix. Conservative for the same
    reason mm is: the untracked operand's zeros are never read.
    """
    if Pa is None and Pb is None:
        raise ValueError("bmm needs at least one tracked operand")
    parts = []
    if Pa is not None:
        parts.append(gather(reduce_sum(Pa, (b, m, k), (2,)), reduce_src((b, m, n), (2,))))
    if Pb is not None:
        parts.append(gather(reduce_sum(Pb, (b, k, n), (1,)), reduce_src((b, m, n), (1,))))
    return union(*parts)


def cat(Ps, shapes, dim=0):
    """Concatenate along dim. A row map, so exact."""
    # Stack every input's rows, then read them back in the joined order. The
    # trick is joining the index arrays: numpy works out the interleaving.
    offs = np.cumsum([0] + [_n(s) for s in shapes])
    arrs = [np.arange(_n(s), dtype=np.int64).reshape(s) + o for s, o in zip(shapes, offs)]
    return gather(bc.vstack(Ps), np.concatenate(arrs, axis=dim).reshape(-1))


def slice_couple(P, shape, dims, out_shape=None):
    """Every output element depends on the whole input slice along dims.

    Covers ops whose output is a value-dependent function of the slice it sits
    in: softmax, sort, layer_norm. Conservative, and for sort badly so: the true
    Jacobian is a permutation and this claims the whole block.

    out_shape differs from shape only along dims, as topk's does, and defaults
    to shape.
    """
    # Union each slice once, then hand every element its slice's row. Pairing
    # elements directly would cost the slice size squared.
    return gather(reduce_sum(P, shape, dims), reduce_src(out_shape or shape, dims))


def prefix(P, shape, dim):
    """Running union along dim: element i draws on elements 0 to i of its line.

    cumsum and cumprod. Structural, as a sum is, and cumprod also loses entries
    where a factor is zero, which a union cannot see. Built by doubling: after
    the step with shift s every element holds the union of the 2s before it,
    so log2 of the line length steps suffice rather than a quadratic pairing.
    """
    if len(shape) == 0:
        return P
    dim %= len(shape)
    L, stride = shape[dim], _n(shape[dim + 1:])
    at = np.arange(_n(shape), dtype=np.int64)
    coord = (at // stride) % L if L else at
    s = 1
    while s < L:
        P = union(P, gather(P, np.where(coord >= s, at - s * stride, -1)))
        s *= 2
    return P


def _pair(v):
    return (int(v), int(v)) if np.isscalar(v) else (int(v[0]), int(v[1]))


def conv2d(Px, Pw, x_shape, w_shape, stride=1, padding=0, dilation=1, groups=1):
    """2D convolution. Pass None for an untracked input or weight.

    x_shape is (N, Ci, H, W) and w_shape is (Co, Ci // groups, KH, KW).

    Output element (n, co, oh, ow) depends on the input taps in its receptive
    field, restricted to the channels of its group, and on every weight entry of
    output channel co within that group. Taps landing in the padding are
    dropped: padding is a structural zero, not a value that could change.

    Conservative both ways. With the input tracked the true derivative is the
    weight at that tap, and the weight is never read. With the weight tracked it
    is the input at that tap, and the input is never read either.
    """
    if Px is None and Pw is None:
        raise ValueError("conv2d needs at least one tracked operand")
    N, Ci, H, W = (int(v) for v in x_shape)
    Co, Cig, KH, KW = (int(v) for v in w_shape)
    if Cig * groups != Ci:
        raise ValueError(f"weight has {Cig} in-channels per group, x has {Ci} over {groups}")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(f"kernel does not fit: output would be {OH} by {OW}")

    # One axis per free index, so the incidence falls out of broadcasting.
    grid = (N, Co, OH, OW, Cig, KH, KW)
    ax = lambda k, size: np.arange(size, dtype=np.int64).reshape(
        tuple(size if d == k else 1 for d in range(7))
    )
    n_, co, oh, ow, cil, kh, kw = (ax(k, g) for k, g in enumerate(grid))

    ih = oh * sh - ph + kh * dh
    iw = ow * sw - pw + kw * dw
    live = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    keep = np.broadcast_to(live, grid).reshape(-1)

    ci = (co // (Co // groups)) * Cig + cil
    out_idx = np.broadcast_to((((n_ * Co + co) * OH + oh) * OW + ow), grid).reshape(-1)[keep]
    m_out = N * Co * OH * OW

    parts = []
    if Px is not None:
        src = np.broadcast_to((((n_ * Ci + ci) * H + ih) * W + iw), grid).reshape(-1)[keep]
        parts.append(couple(Px, out_idx, src, m_out))
    if Pw is not None:
        src = np.broadcast_to((((co * Cig + cil) * KH + kh) * KW + kw), grid).reshape(-1)[keep]
        parts.append(couple(Pw, out_idx, src, m_out))
    return union(*parts)
