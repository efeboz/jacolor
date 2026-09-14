import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl, propagate as pr
from src.compress import decompress, seeds

F64 = torch.float64


def dense_jac(f, x):
    # (prod(out_shape), prod(in_shape)), row-major on both sides.
    return torch.func.jacrev(f)(x).reshape(-1, x.numel())


def assert_sound(P, f, n_in, points=8, seed=0):
    # Containment is the whole contract: every derivative that is nonzero
    # anywhere must be in the pattern. Checked at several random points, since
    # one point can hide an entry that vanishes only there.
    g = torch.Generator().manual_seed(seed)
    Pd = P.toarray()
    for _ in range(points):
        x = torch.randn(n_in, generator=g, dtype=F64)
        nz = (dense_jac(f, x) != 0).numpy()
        missed = np.argwhere(nz & ~Pd)
        assert not missed.size, f"pattern misses {missed[:5].tolist()}"


def assert_exact(P, f, n_in, seed=0):
    x = torch.randn(n_in, generator=torch.Generator().manual_seed(seed), dtype=F64)
    nz = (dense_jac(f, x) != 0).numpy()
    extra = np.argwhere(P.toarray() & ~nz)
    assert not extra.size, f"pattern is loose at {extra[:5].tolist()}"


class TestIndexMaps:
    """The two index maps carry all the shape reasoning, so pin them directly."""

    def test_bcast_src(self):
        assert pr.bcast_src((3,), (2, 3)).tolist() == [0, 1, 2, 0, 1, 2]
        assert pr.bcast_src((2, 1), (2, 3)).tolist() == [0, 0, 0, 1, 1, 1]
        assert pr.bcast_src((), (4,)).tolist() == [0, 0, 0, 0]

    def test_reduce_src(self):
        assert pr.reduce_src((2, 3), (1,)).tolist() == [0, 0, 0, 1, 1, 1]
        assert pr.reduce_src((2, 3), (0,)).tolist() == [0, 1, 2, 0, 1, 2]
        assert pr.reduce_src((2, 3), (0, 1)).tolist() == [0] * 6

    def test_reduce_src_negative_dim(self):
        assert (pr.reduce_src((2, 3), (-1,)) == pr.reduce_src((2, 3), (1,))).all()


class TestGather:
    """Row maps are exact: one output element is one input element."""

    @pytest.mark.parametrize(
        "f,src",
        [
            (lambda x: x.reshape(2, 3).T.reshape(-1), [0, 3, 1, 4, 2, 5]),
            (lambda x: x[::2], [0, 2, 4]),
            (lambda x: x[[4, 1, 1]], [4, 1, 1]),
            (lambda x: x.flip(0), [5, 4, 3, 2, 1, 0]),
        ],
    )
    def test_exact(self, f, src):
        P = pr.gather(bc.eye(6), src)
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)


class TestPointwise:
    def test_unary_is_identity(self):
        P = pr.pointwise((6,), [(bc.eye(6), (6,))])
        assert_sound(P, torch.tanh, 6)
        assert_exact(P, torch.tanh, 6)

    def test_binary_broadcast(self):
        # x.reshape(2,3) * x.reshape(2,3).sum(1, keepdim=True) is not this rule's
        # job; here both operands are plain broadcasts of x.
        f = lambda x: (x.reshape(2, 3) + x[:2, None]).reshape(-1)
        P = pr.pointwise((2, 3), [(bc.eye(6), (2, 3)), (pr.gather(bc.eye(6), [0, 1]), (2, 1))])
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)

    def test_needs_a_tracked_operand(self):
        with pytest.raises(ValueError, match="at least one tracked operand"):
            pr.pointwise((3,), [])


class TestReduceSum:
    @pytest.mark.parametrize(
        "dims,f",
        [
            ((1,), lambda x: x.reshape(2, 3).sum(1)),
            ((0,), lambda x: x.reshape(2, 3).sum(0)),
            ((0, 1), lambda x: x.reshape(2, 3).sum((0, 1)).reshape(1)),
        ],
    )
    def test_matches_true_jacobian(self, dims, f):
        P = pr.reduce_sum(bc.eye(6), (2, 3), dims)
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)  # d(sum)/dx is 1, never zero


class TestMatmul:
    def test_tracked_vector_constant_matrix(self):
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=F64)
        f = lambda x: A @ x
        P = pr.mm(None, bc.eye(3), 2, 3, 1)
        assert_sound(P, f, 3)

    def test_constant_matrix_zeros_are_not_pruned(self):
        # A[1,0] and A[1,1] are zero at this point. Pruning on them would be
        # unsound the moment A changes, so the pattern stays dense.
        A = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=F64)
        P = pr.mm(None, bc.eye(3), 2, 3, 1)
        assert P.toarray().all(), "pattern should be dense, not A's own pattern"
        J = dense_jac(lambda x: A @ x, torch.ones(3, dtype=F64))
        assert not (J != 0).numpy().all(), "the true Jacobian really does have zeros"

    def test_tracked_matrix_constant_vector(self):
        v = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
        f = lambda x: x.reshape(2, 3) @ v
        P = pr.mm(bc.eye(6), None, 2, 3, 1)
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)  # out[i] touches exactly row i of x

    def test_both_tracked(self):
        f = lambda x: (x[:6].reshape(2, 3) @ x[6:].reshape(3, 2)).reshape(-1)
        Pa = pr.gather(bc.eye(12), np.arange(6))
        Pb = pr.gather(bc.eye(12), np.arange(6, 12))
        P = pr.mm(Pa, Pb, 2, 3, 2)
        assert_sound(P, f, 12)

    def test_needs_a_tracked_operand(self):
        with pytest.raises(ValueError, match="at least one tracked operand"):
            pr.mm(None, None, 1, 1, 1)


def composed(x):
    # Four rules in sequence. Jacobian is block diagonal, 2 blocks of 3.
    a = torch.sin(x) * x
    b = a.reshape(2, 3)
    c = b.sum(dim=1)
    return (b * c[:, None]).reshape(-1)


def propagate_composed():
    P0 = bc.eye(6)
    Pa = pr.pointwise((6,), [(P0, (6,)), (P0, (6,))])  # sin(x) * x
    Pb = Pa  # reshape does not move rows
    Pc = pr.reduce_sum(Pb, (2, 3), (1,))
    return pr.pointwise((2, 3), [(Pb, (2, 3)), (Pc, (2, 1))])


class TestComposition:
    """The rules have to compose, not just hold one at a time."""

    def test_sound_and_exact(self):
        P = propagate_composed()
        assert_sound(P, composed, 6)
        assert_exact(P, composed, 6)

    def test_structure_is_block_diagonal(self):
        expected = np.zeros((6, 6), dtype=bool)
        expected[:3, :3] = True
        expected[3:, 3:] = True
        assert (propagate_composed().toarray() == expected).all()


class TestEndToEnd:
    """Propagated pattern, colored, seeded, one jvp per color, decompressed."""

    def test_recovers_the_dense_jacobian(self):
        P = propagate_composed()
        c = cl.color_cols(P)
        assert c.n_colors == 3 and c.lower_bound == 3  # 3 passes, not 6

        x = torch.randn(6, generator=torch.Generator().manual_seed(7), dtype=F64)
        S = seeds(c, dtype=F64)
        B = torch.stack(
            [torch.func.jvp(composed, (x,), (S[:, k],))[1] for k in range(c.n_colors)],
            dim=1,
        )
        got = decompress(B, P, c)
        torch.testing.assert_close(got.to_dense(), dense_jac(composed, x), rtol=1e-12, atol=1e-14)


def brute_bcast(shape_in, shape_out):
    # Reference: walk output indices, drop the broadcast axes, re-flatten.
    pad = (1,) * (len(shape_out) - len(shape_in)) + tuple(shape_in)
    src = []
    for flat in range(int(np.prod(shape_out))):
        oidx = np.unravel_index(flat, shape_out)
        iidx = tuple(0 if pad[d] == 1 else oidx[d] for d in range(len(shape_out)))
        src.append(int(np.ravel_multi_index(iidx, pad)))
    return np.array(src, dtype=np.int64)


def brute_reduce(shape, dims):
    kept = tuple(s for d, s in enumerate(shape) if d not in dims)
    out = []
    for flat in range(int(np.prod(shape))):
        idx = np.unravel_index(flat, shape)
        k = tuple(v for d, v in enumerate(idx) if d not in dims)
        out.append(int(np.ravel_multi_index(k, kept)) if kept else 0)
    return np.array(out, dtype=np.int64)


class TestIndexMapsAgainstBruteForce:
    """The index maps are where an off-by-one would silently drop entries."""

    @pytest.mark.parametrize("seed", range(40))
    def test_bcast(self, seed):
        rng = np.random.default_rng(seed)
        nd = int(rng.integers(1, 4))
        shape_out = tuple(int(rng.integers(1, 4)) for _ in range(nd))
        keep = rng.random(nd) < 0.6
        shape_in = tuple(s if k else 1 for s, k in zip(shape_out, keep))
        drop = int(rng.integers(0, nd))  # also test fewer input dims
        shape_in = shape_in[drop:]
        assert (pr.bcast_src(shape_in, shape_out) == brute_bcast(shape_in, shape_out)).all()

    @pytest.mark.parametrize("seed", range(40))
    def test_reduce(self, seed):
        rng = np.random.default_rng(seed)
        nd = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 4)) for _ in range(nd))
        dims = tuple(d for d in range(nd) if rng.random() < 0.5) or (0,)
        assert (pr.reduce_src(shape, dims) == brute_reduce(shape, dims)).all()


class TestMatmulIncidence:
    """mm builds its incidence by arithmetic, so check it against nested loops."""

    @pytest.mark.parametrize("seed", range(15))
    def test_against_loops(self, seed):
        rng = np.random.default_rng(seed)
        m, k, n = (int(rng.integers(1, 4)) for _ in range(3))
        want_a = np.zeros((m * n, m * k), dtype=bool)
        want_b = np.zeros((m * n, k * n), dtype=bool)
        for i in range(m):
            for l in range(n):
                for j in range(k):
                    want_a[i * n + l, i * k + j] = True
                    want_b[i * n + l, j * n + l] = True
        assert (pr.mm(bc.eye(m * k), None, m, k, n).toarray() == want_a).all()
        assert (pr.mm(None, bc.eye(k * n), m, k, n).toarray() == want_b).all()


class TestDegenerateDerivatives:
    """Entries that are zero at a point but not identically zero.

    These are exactly what a pattern read off one Jacobian sample would miss,
    and what soundness has to cover.
    """

    def test_dead_relu_stays_in_the_pattern(self):
        P = pr.pointwise((4,), [(bc.eye(4), (4,))])
        x = torch.full((4,), -1.0, dtype=F64)  # every derivative is 0 here
        assert not (dense_jac(torch.relu, x) != 0).any()
        assert P.toarray().diagonal().all()  # pattern still claims the diagonal
        assert_sound(P, torch.relu, 4)  # and it holds at points where it is alive

    def test_product_at_the_origin(self):
        f = lambda x: x[:2] * x[2:]
        P = pr.pointwise((2,), [(pr.gather(bc.eye(4), [0, 1]), (2,)),
                                (pr.gather(bc.eye(4), [2, 3]), (2,))])
        assert not (dense_jac(f, torch.zeros(4, dtype=F64)) != 0).any()
        assert_sound(P, f, 4)

    def test_cancellation_is_over_approximated(self):
        # x - x has a zero Jacobian everywhere. The union rule cannot see that,
        # and over-approximating is the correct failure direction.
        f = lambda x: x - x
        P = pr.pointwise((3,), [(bc.eye(3), (3,)), (bc.eye(3), (3,))])
        assert not (dense_jac(f, torch.randn(3, dtype=F64)) != 0).any()
        assert_sound(P, f, 3)
        assert P.nnz == 3  # loose, deliberately


class TestTightnessClaims:
    """Every rule states a tightness, and exact rules are held to it above."""

    def test_every_rule_is_labelled(self):
        rules = {"gather", "reduce_sum", "pointwise", "couple", "mm"}
        assert set(pr.TIGHTNESS) == rules
        assert set(pr.TIGHTNESS.values()) <= {"exact", "structural", "conservative"}

    def test_mm_with_a_constant_is_only_conservative(self):
        # Labelled conservative because it will not read the constant's zeros.
        assert pr.TIGHTNESS["mm"] == "conservative"
