import numpy as np
import pytest
import scipy.sparse as sp

from src import _boolcsr as bc


def test_canary_300_way_accumulation():
    # 300 contributions to one entry: bool ORs, an integer dtype would wrap.
    # A dropped entry here is a silently wrong Jacobian.
    n = 300
    P = bc.from_pairs(np.zeros(n, dtype=np.int64), np.arange(n), shape=(1, n))
    A = bc.matmul(bc.transpose(P), P)
    assert A.dtype == np.bool_
    assert A.nnz == n * n
    assert A.toarray().all()


def test_canary_256_coincident_entries():
    # 256 is the exact int8/uint8 wrap point; bool must still hold the entry.
    k = 256
    P = bc.from_pairs(np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64), shape=(1, 1))
    assert P.toarray().tolist() == [[True]]


def test_row_nnz_matches_dense():
    a = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    assert bc.row_nnz(bc.from_dense(a)).tolist() == a.sum(axis=1).tolist()


def test_check_rejects_non_bool():
    M = sp.csr_array(np.eye(3, dtype=np.int8))
    with pytest.raises(TypeError, match="dtype must be bool"):
        bc.check(M)


def test_check_rejects_dense():
    with pytest.raises(TypeError, match="expected a CSR pattern"):
        bc.check(np.eye(3, dtype=bool))


def test_transpose_and_matmul_keep_bool():
    P = bc.from_dense([[1, 1, 0], [0, 1, 1]])
    assert bc.transpose(P).dtype == np.bool_
    assert bc.matmul(P, bc.transpose(P)).dtype == np.bool_
