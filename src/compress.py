"""Seeding and decompression for sparse AD.

Columns sharing a color never share a row, so one AD pass per color is enough.
The compressed column for color k holds the sum of J's columns of that color, and
at a pattern nonzero (i, j) only one of them is nonzero in row i. Recovering
J[i, j] is a lookup, not a solve. Rows are the mirror image, with reverse passes.

A pattern that over-approximates the true nonzeros still decompresses exactly and
only costs colors. A pattern that misses one loses that entry quietly, which is
why detection has to be conservative.
"""

import numpy as np
import torch

from . import _boolcsr as bc

__all__ = ["seeds", "decompress"]


def _block(coloring, lo, hi, dtype, device):
    # Seed columns for colors lo up to hi, as (n_lines, hi - lo). Lines whose
    # color falls outside the range contribute nothing to this block.
    n = coloring.colors.size
    kc = torch.as_tensor(coloring.colors).to(device)
    keep = (kc >= lo) & (kc < hi)
    S = torch.zeros(n, hi - lo, dtype=dtype, device=device)
    S[torch.arange(n, device=device)[keep], kc[keep] - lo] = 1
    return S


def _index(P, coloring, device):
    # Row, column and owning color of every pattern nonzero, in CSR order.
    ri = torch.from_numpy(np.repeat(np.arange(P.shape[0]), bc.row_nnz(P))).to(device)
    ci = torch.from_numpy(P.indices.astype(np.int64)).to(device)
    kc = torch.as_tensor(coloring.colors).to(device)
    return ri, ci, (kc[ci] if coloring.axis == "cols" else kc[ri])


def seeds(coloring, dtype=None, device=None):
    """Seed matrix for a coloring.

    Forward (axis "cols"): shape (n_cols, n_colors), column k is the tangent for
    color k. Reverse (axis "rows"): shape (n_colors, n_rows), row k is the
    cotangent for color k.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    S = _block(coloring, 0, coloring.n_colors, dtype, device)
    return S if coloring.axis == "cols" else S.T.contiguous()


def decompress(B, coloring):
    """Scatter compressed AD results B back onto the coloring's pattern.

    Returns a coalesced sparse COO tensor holding exactly the pattern's
    nonzeros. The pattern travels with the coloring, so the two always fit.
    """
    P = coloring.pattern
    m, n = P.shape
    want = (m, coloring.n_colors) if coloring.axis == "cols" else (coloring.n_colors, n)
    if tuple(B.shape) != want:
        raise ValueError(
            f"compressed result for axis={coloring.axis!r} should have shape "
            f"{want}, got {tuple(B.shape)}"
        )
    ri, ci, line = _index(P, coloring, B.device)
    vals = B[ri, line] if coloring.axis == "cols" else B[line, ci]
    # Canonical CSR has no repeats, so coalesce() cannot merge two entries here.
    return torch.sparse_coo_tensor(torch.stack([ri, ci]), vals, P.shape).coalesce()
