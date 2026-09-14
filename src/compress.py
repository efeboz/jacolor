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


def seeds(coloring, dtype=None, device=None):
    """Seed matrix for `coloring`.

    Forward (`axis="cols"`): shape (n_cols, n_colors), column k is the tangent
    for color k. Reverse (`axis="rows"`): shape (n_colors, n_rows), row k is the
    cotangent for color k.
    """
    n = coloring.colors.size
    k = coloring.n_colors
    line = torch.arange(n, device=device)  # one seed entry per colored column/row
    kc = torch.as_tensor(coloring.colors).to(device)  # color of each line
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if coloring.axis == "cols":
        S = torch.zeros(n, k, dtype=dtype, device=device)
        S[line, kc] = 1
    else:
        S = torch.zeros(k, n, dtype=dtype, device=device)
        S[kc, line] = 1
    return S


def decompress(B, P, coloring):
    """Scatter compressed AD results `B` back onto pattern `P`.

    Returns a coalesced sparse COO tensor holding exactly `P`'s nonzeros.
    """
    bc.check(P)
    _check_fit(B, P, coloring)
    # Row and column index of every pattern nonzero, in CSR order.
    ri = torch.from_numpy(np.repeat(np.arange(P.shape[0]), bc.row_nnz(P))).to(B.device)
    ci = torch.from_numpy(P.indices.astype(np.int64)).to(B.device)
    kc = torch.as_tensor(coloring.colors).to(B.device)
    vals = B[ri, kc[ci]] if coloring.axis == "cols" else B[kc[ri], ci]
    # Canonical CSR has no repeats, so coalesce() cannot merge two entries here.
    return torch.sparse_coo_tensor(torch.stack([ri, ci]), vals, P.shape).coalesce()


def _check_fit(B, P, coloring):
    m, n = P.shape
    if coloring.axis == "cols":
        n_lines, want = n, (m, coloring.n_colors)
    else:
        n_lines, want = m, (coloring.n_colors, n)
    if coloring.colors.size != n_lines:
        raise ValueError(
            f"coloring covers {coloring.colors.size} {coloring.axis}, "
            f"pattern has {n_lines}"
        )
    if tuple(B.shape) != want:
        raise ValueError(
            f"compressed result for axis={coloring.axis!r} should have shape "
            f"{want}, got {tuple(B.shape)}"
        )
