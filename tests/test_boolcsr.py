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
    # 256 is the exact int8/uint8 wrap point. bool must still hold the entry.
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


def test_check_canonicalizes_repeated_indices_with_or():
    # A hand-built CSR can repeat a column index. Two repeats must become one
    # entry, never a count, and row_nnz must see one.
    M = sp.csr_array(
        (np.ones(4, dtype=np.bool_), np.array([1, 1, 0, 0]), np.array([0, 4])), shape=(1, 3)
    )
    assert not M.has_canonical_format
    M = bc.check(M)
    assert M.has_canonical_format
    assert M.indices.tolist() == [0, 1] and bc.row_nnz(M).tolist() == [2]


def test_every_op_returns_canonical_csr():
    # matmul output is unsorted from scipy; check() must have fixed it, since
    # decompress emits one COO entry per stored index and coalesce() sums.
    rng = np.random.default_rng(0)
    for _ in range(50):
        m, n = rng.integers(1, 8, size=2)
        P = bc.from_pairs(rng.integers(0, m, 20), rng.integers(0, n, 20), (m, n))
        outs = [P, bc.gather(P, rng.integers(0, m, 7)), bc.matmul(bc.transpose(P), P),
                bc.union(P, bc.gather(P, np.arange(m))), bc.transpose(P), bc.eye(n)]
        for Q in outs:
            assert Q.has_canonical_format
            for r in range(Q.shape[0]):
                row = Q.indices[Q.indptr[r]:Q.indptr[r + 1]]
                assert np.all(np.diff(row) > 0)
