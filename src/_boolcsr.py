"""Boolean CSR pattern storage.

Every sparsity pattern in jacolor lives here, and every pattern is dtype=bool.
SciPy's sparsetools accumulate bool with logical OR. Integer dtypes wrap instead,
so 256 coincident contributions to one entry sum to 0 and the entry leaves the
pattern. A missing entry is an unsound Jacobian, so bool is enforced here rather
than assumed.
"""

import numpy as np
import scipy.sparse as sp

__all__ = ["check", "from_pairs", "from_dense", "transpose", "matmul", "row_nnz"]


def check(M):
    # Gate on every pattern entering or leaving this module.
    if not sp.issparse(M) or M.format != "csr":
        raise TypeError(f"expected a CSR pattern, got {type(M).__name__}")
    if M.dtype != np.bool_:
        raise TypeError(f"pattern dtype must be bool, got {M.dtype}")
    return M


def from_pairs(rows, cols, shape):
    # Coordinate pairs may repeat. Duplicates OR together on the COO->CSR sum.
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.ones(rows.size, dtype=np.bool_)
    return check(sp.csr_array(sp.coo_array((vals, (rows, cols)), shape=shape)))


def from_dense(a):
    return check(sp.csr_array(np.asarray(a, dtype=np.bool_)))


def transpose(M):
    return check(sp.csr_array(check(M).T))


def matmul(A, B):
    return check(check(A) @ check(B))


def row_nnz(M):
    # From indptr, never M.sum(axis=1): summing bool promotes to an int dtype.
    return np.diff(check(M).indptr)
