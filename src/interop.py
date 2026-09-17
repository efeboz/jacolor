"""Handing patterns in and results out.

A user who already knows the structure of their Jacobian should be able to say
so without reaching into the internals, and a result is most useful to the rest
of scientific Python as a scipy matrix.
"""

import numpy as np
import scipy.sparse as sp
import torch

from . import _boolcsr as bc

__all__ = ["pattern", "pattern_from_pairs", "to_scipy"]


def pattern(obj):
    """A sparsity pattern from whatever the caller already has.

    Accepts a dense array or torch tensor, where anything nonzero counts as an
    entry, or any scipy sparse matrix. The result is the boolean CSR the rest of
    jacolor works in.
    """
    if sp.issparse(obj):
        return bc.check(sp.csr_array(obj.astype(bool)))
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    return bc.from_dense(np.asarray(obj) != 0)


def pattern_from_pairs(rows, cols, shape):
    """A sparsity pattern from the coordinates of its entries.

    Repeated coordinates are fine. Nothing is assumed about their order.
    """
    return bc.from_pairs(rows, cols, shape)


def to_scipy(J):
    """A sparse Jacobian as a scipy CSR matrix.

    The data is copied to the CPU and detached, because scipy holds plain numpy
    arrays. Nothing about autograd survives the trip.
    """
    if not J.is_sparse:
        raise TypeError(f"expected a sparse Jacobian, got a {J.layout} tensor")
    J = J if J.is_coalesced() else J.coalesce()
    idx = J.indices().detach().cpu().numpy()
    val = J.values().detach().cpu().numpy()
    return sp.csr_array((val, (idx[0], idx[1])), shape=tuple(J.shape))
