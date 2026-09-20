import numpy as np
import pytest

from src import _boolcsr as bc, coloring as cl


def rand_pattern(rng, m, n, density):
    return bc.from_dense(rng.random((m, n)) < density)


def assert_valid(P, colors):
    # Two columns may share a color only if they share no row.
    A = bc.matmul(bc.transpose(P), P).tocoo()
    for c, d in zip(A.coords[0], A.coords[1]):
        if c != d:
            assert colors[c] != colors[d], f"columns {c},{d} share a row and a color"


@pytest.mark.parametrize("order", cl.ORDERS)
def test_coloring_is_valid(order):
    rng = np.random.default_rng(0)
    for _ in range(100):
        m, n = rng.integers(2, 20, size=2)
        P = rand_pattern(rng, m, n, rng.uniform(0.05, 0.6))
        assert_valid(P, cl.color_cols(P, orders=(order,)).colors)


def test_diagonal_needs_one_color():
    P = bc.from_dense(np.eye(8, dtype=bool))
    c = cl.color_cols(P)
    assert (c.n_colors, c.lower_bound) == (1, 1)


def test_dense_needs_one_color_per_column():
    P = bc.from_dense(np.ones((4, 6), dtype=bool))
    c = cl.color_cols(P)
    assert (c.n_colors, c.lower_bound) == (6, 6)


def test_lower_bound_is_densest_row_and_is_respected():
    rng = np.random.default_rng(1)
    for _ in range(50):
        P = rand_pattern(rng, 12, 12, rng.uniform(0.05, 0.7))
        c = cl.color_cols(P)
        assert c.lower_bound == int(bc.row_nnz(P).max())
        assert c.n_colors >= c.lower_bound


def test_empty_columns_share_a_color():
    P = bc.from_dense([[1, 0, 0, 1], [0, 0, 0, 1]])
    c = cl.color_cols(P)
    assert c.colors[1] == c.colors[2]  # neither column touches any row


def test_row_coloring_is_column_coloring_of_the_transpose():
    rng = np.random.default_rng(2)
    P = rand_pattern(rng, 9, 13, 0.3)
    assert cl.color_rows(P).n_colors == cl.color_cols(bc.transpose(P)).n_colors


@pytest.mark.skipif(not cl.HAS_NUMBA, reason="numba not installed")
def test_numba_and_python_kernels_agree():
    rng = np.random.default_rng(3)
    for _ in range(20):
        P = rand_pattern(rng, 15, 15, rng.uniform(0.1, 0.5))
        A = bc.matmul(bc.transpose(P), P)
        perm = np.arange(15, dtype=np.int64)
        jit = cl._greedy(A.indptr, A.indices, perm, 15)
        pure = cl._greedy_py(A.indptr, A.indices, perm, 15)
        assert np.array_equal(jit, pure)


def test_warns_on_large_pattern_without_numba(monkeypatch):
    monkeypatch.setattr(cl, "HAS_NUMBA", False)
    monkeypatch.setattr(cl, "_SLOW_ABOVE", 4)
    P = bc.from_dense(np.eye(6, dtype=bool))
    with pytest.warns(UserWarning, match="jacolor\\[fast\\]"):
        cl.color_cols(P)


def test_unknown_ordering_rejected():
    P = bc.from_dense(np.eye(3, dtype=bool))
    with pytest.raises(ValueError, match="unknown ordering"):
        cl.color_cols(P, orders=("dsatur",))


class TestAgainstNetworkx:
    """The kernel is small enough to check against a reference."""

    @staticmethod
    def graph(P):
        nx = pytest.importorskip("networkx")
        A = bc.matmul(bc.transpose(P), P).toarray()
        np.fill_diagonal(A, False)
        G = nx.Graph()
        G.add_nodes_from(range(P.shape[1]))
        G.add_edges_from(zip(*np.nonzero(np.triu(A))))
        return G

    def test_largest_first_matches(self):
        nx = pytest.importorskip("networkx")
        rng = np.random.default_rng(4)
        for _ in range(100):
            m, n = rng.integers(2, 12, size=2)
            P = rand_pattern(rng, m, n, rng.uniform(0.1, 0.6))
            ref = len(set(nx.greedy_color(self.graph(P), strategy="largest_first").values()))
            assert cl.color_cols(P, orders=("lf",)).n_colors == ref

    def test_best_of_orders_never_worse_than_largest_first(self):
        nx = pytest.importorskip("networkx")
        rng = np.random.default_rng(5)
        for _ in range(100):
            m, n = rng.integers(2, 25, size=2)
            P = rand_pattern(rng, m, n, rng.uniform(0.05, 0.5))
            ref = len(set(nx.greedy_color(self.graph(P), strategy="largest_first").values()))
            assert cl.color_cols(P).n_colors <= ref


class TestGraphEstimate:
    """The cost of a direction is knowable before anything is built."""

    def test_it_bounds_the_real_graph(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            m, n = rng.integers(2, 12, size=2)
            P = rand_pattern(rng, m, n, rng.uniform(0.1, 0.8))
            actual = bc.matmul(bc.transpose(P), P).nnz
            assert cl.graph_estimate(P) >= actual

    def test_it_is_clamped_by_the_column_count(self):
        # 20001 rows of 100 entries bound the pairs at 2e8, but the graph cannot
        # hold more than 100 by 100 entries.
        P = bc.from_dense(np.ones((20001, 100), dtype=bool))
        assert cl.graph_estimate(P) == 100 * 100

    def test_a_dense_row_is_exactly_its_square(self):
        P = bc.from_pairs(np.zeros(50, dtype=np.int64), np.arange(50), (1, 50))
        assert cl.graph_estimate(P) == 50 * 50


class TestStopsAtTheBound:
    """Trying more orderings after reaching the bound cannot improve anything."""

    def _count_greedy(self, monkeypatch):
        calls = []
        real = cl._greedy
        monkeypatch.setattr(cl, "_greedy", lambda *a: (calls.append(1), real(*a))[1])
        return calls

    def test_one_ordering_when_the_bound_is_reached(self, monkeypatch):
        calls = self._count_greedy(monkeypatch)
        P = bc.from_dense(np.eye(8, dtype=bool))
        c = cl.color_cols(P)
        assert c.n_colors == c.lower_bound == 1
        assert len(calls) == 1

    def test_every_ordering_when_it_is_not(self, monkeypatch):
        calls = self._count_greedy(monkeypatch)
        # A three cycle needs three colors although its densest line holds two.
        P = bc.from_dense(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool))
        c = cl.color_cols(P)
        assert c.n_colors == 3 and c.lower_bound == 2
        assert len(calls) == len(cl.ORDERS)

    def test_stopping_early_never_costs_a_color(self):
        # Whatever it stops at must still be the best any single ordering gives.
        rng = np.random.default_rng(1)
        for _ in range(40):
            m, n = rng.integers(2, 14, size=2)
            P = rand_pattern(rng, m, n, rng.uniform(0.1, 0.6))
            each = [cl.color_cols(P, orders=(o,)).n_colors for o in cl.ORDERS]
            assert cl.color_cols(P).n_colors == min(each)
