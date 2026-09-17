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
        # Without this a one-row pattern broadcasts against a many-row Jacobian.
        assert nz.shape == Pd.shape, f"pattern {Pd.shape} vs Jacobian {nz.shape}"
        missed = np.argwhere(nz & ~Pd)
        assert not missed.size, f"pattern misses {missed[:5].tolist()}"


def assert_exact(P, f, n_in, seed=0):
    x = torch.randn(n_in, generator=torch.Generator().manual_seed(seed), dtype=F64)
    nz = (dense_jac(f, x) != 0).numpy()
    assert nz.shape == P.shape, f"pattern {P.shape} vs Jacobian {nz.shape}"
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
        got = decompress(B, c)
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

    # Not derivative claims: two return index arrays, one is a set operation.
    HELPERS = {"TIGHTNESS", "bcast_src", "reduce_src", "union"}

    def test_every_rule_is_labelled(self):
        # Derived from __all__, so adding a rule without a tightness fails here.
        assert set(pr.TIGHTNESS) == set(pr.__all__) - self.HELPERS
        assert set(pr.TIGHTNESS.values()) <= {"exact", "structural", "conservative"}

    def test_mm_with_a_constant_is_only_conservative(self):
        # Labelled conservative because it will not read the constant's zeros.
        assert pr.TIGHTNESS["mm"] == "conservative"


class TestJoin:
    """cat and stack are row maps over stacked rows, so exact."""

    def test_cat_dim0_reassembles_the_identity(self):
        P1, P2 = pr.gather(bc.eye(4), [0, 1]), pr.gather(bc.eye(4), [2, 3])
        assert (pr.cat([P1, P2], [(2,), (2,)], 0).toarray() == np.eye(4, dtype=bool)).all()

    def test_cat_dim0(self):
        f = lambda x: torch.cat([torch.sin(x[:2]), x[2:] * x[2:]])
        P = pr.cat([pr.gather(bc.eye(4), [0, 1]), pr.gather(bc.eye(4), [2, 3])],
                   [(2,), (2,)], 0)
        assert_sound(P, f, 4)
        assert_exact(P, f, 4)

    def test_cat_dim1_interleaves_rows(self):
        # Row order is not "first block then second", which is the whole point.
        f = lambda x: torch.cat(
            [x[:4].reshape(2, 2), torch.sin(x[4:]).reshape(2, 1)], dim=1
        ).reshape(-1)
        Pa = pr.gather(bc.eye(6), [0, 1, 2, 3])
        Pb = pr.gather(bc.eye(6), [4, 5])
        P = pr.cat([Pa, Pb], [(2, 2), (2, 1)], 1)
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)

    def test_stack(self):
        f = lambda x: torch.stack([torch.sin(x[:3]), x[3:]], dim=0).reshape(-1)
        P = pr.stack([pr.gather(bc.eye(6), [0, 1, 2]), pr.gather(bc.eye(6), [3, 4, 5])],
                     [(3,), (3,)], 0)
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)


class TestSliceCouple:
    def test_softmax_is_dense_within_its_slice(self):
        f = lambda x: torch.softmax(x.reshape(2, 3), dim=1).reshape(-1)
        P = pr.slice_couple(bc.eye(6), (2, 3), (1,))
        assert_sound(P, f, 6)
        assert_exact(P, f, 6)  # softmax really does couple the whole slice

    def test_sort_is_sound_but_loose(self):
        # sort's true Jacobian is a permutation, one nonzero per row. The rule
        # cannot know which, so it claims the block. Loose, never wrong.
        f = lambda x: x.reshape(2, 3).sort(dim=1).values.reshape(-1)
        P = pr.slice_couple(bc.eye(6), (2, 3), (1,))
        assert_sound(P, f, 6)
        J = dense_jac(f, torch.randn(6, generator=torch.Generator().manual_seed(1), dtype=F64))
        assert (J != 0).sum() == 6 and P.nnz == 18  # 6 true entries, 18 claimed

    def test_cummax_over_a_dim(self):
        f = lambda x: x.reshape(2, 3).cummax(dim=1).values.reshape(-1)
        assert_sound(pr.slice_couple(bc.eye(6), (2, 3), (1,)), f, 6)

    def test_all_dims_couples_everything(self):
        P = pr.slice_couple(bc.eye(6), (2, 3), (0, 1))
        assert P.toarray().all()


class TestIndexCouple:
    def test_tracked_index_gather(self):
        # The index comes from the data, so any input element could feed any
        # output element. The true Jacobian is two one-hot rows.
        f = lambda x: x[torch.argsort(x)[:2]]
        P = pr.index_couple(bc.eye(6), (6,), 0, 2)
        assert_sound(P, f, 6)
        assert P.toarray().all()
        J = dense_jac(f, torch.randn(6, generator=torch.Generator().manual_seed(2), dtype=F64))
        assert (J != 0).sum() == 2

    def test_argmax_scalar_index(self):
        f = lambda x: (x[x.argmax()] * x)[:1]
        P = pr.index_couple(bc.eye(5), (5,), 0, 1)
        assert_sound(P, f, 5)

    def test_keeps_other_dims_separate(self):
        # Indexing along dim 1 must not couple row 0 to row 1.
        f = lambda x: torch.take_along_dim(
            x.reshape(2, 3), x.reshape(2, 3).argsort(dim=1)[:, :2], dim=1
        ).reshape(-1)
        P = pr.index_couple(bc.eye(6), (2, 3), 1, 2)
        assert_sound(P, f, 6)
        assert P.nnz == 12  # 2 outputs x 3 inputs x 2 rows, not 4 x 6


def brute_group(shape, dims, out_shape=None, skip=None):
    # (a, b) set iff output element a and input element b agree on every axis
    # that is not coupled.
    out_shape = out_shape or shape
    free = [d for d in range(len(shape)) if d not in (skip if skip is not None else dims)]
    M = np.zeros((int(np.prod(out_shape)), int(np.prod(shape))), dtype=bool)
    for a in range(M.shape[0]):
        ia = np.unravel_index(a, out_shape)
        for b in range(M.shape[1]):
            ib = np.unravel_index(b, shape)
            M[a, b] = all(ia[d] == ib[d] for d in free)
    return M


class TestCouplingIncidence:
    """Both couplings are built by argsort arithmetic. Check against loops."""

    @pytest.mark.parametrize("seed", range(25))
    def test_slice_couple(self, seed):
        rng = np.random.default_rng(seed)
        nd = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 4)) for _ in range(nd))
        dims = tuple(d for d in range(nd) if rng.random() < 0.5) or (0,)
        got = pr.slice_couple(bc.eye(int(np.prod(shape))), shape, dims).toarray()
        assert (got == brute_group(shape, dims)).all()

    @pytest.mark.parametrize("seed", range(25))
    def test_index_couple(self, seed):
        rng = np.random.default_rng(seed)
        nd = int(rng.integers(1, 4))
        shape = tuple(int(rng.integers(1, 4)) for _ in range(nd))
        dim = int(rng.integers(0, nd))
        n_out = int(rng.integers(1, 4))
        out_shape = shape[:dim] + (n_out,) + shape[dim + 1:]
        got = pr.index_couple(bc.eye(int(np.prod(shape))), shape, dim, n_out).toarray()
        assert (got == brute_group(shape, (dim,), out_shape, skip=(dim,))).all()


# --- randomized soundness sweep ----------------------------------------------
# Random chains of rules against real Jacobians. A single-rule test cannot catch
# a composition that drops entries, and containment is the one property that
# must never fail.


def _candidates(shape):
    """Every step that fits shape, as (torch op, pattern rule, new shape)."""
    N = int(np.prod(shape))
    out = [(lambda v: torch.tanh(v) * v,
            lambda P, s: pr.pointwise(s, [(P, s), (P, s)]), shape)]
    if len(shape) == 1:
        if N % 2 == 0 and N > 2:
            h = N // 2
            out.append((lambda v: v.reshape(2, h), lambda P, s: P, (2, h)))
            out.append((lambda v, h=h: v[:h] * v[h:],
                        lambda P, s, h=h: pr.pointwise(
                            (h,), [(pr.gather(P, np.arange(h)), (h,)),
                                   (pr.gather(P, np.arange(h, 2 * h)), (h,))]), (h,)))
        if N > 2:
            out.append((lambda v: v[1:],
                        lambda P, s, N=N: pr.gather(P, np.arange(1, N)), (N - 1,)))
            k = max(1, N // 2)
            out.append((lambda v, k=k: v[torch.argsort(v)[:k]],
                        lambda P, s, k=k: pr.index_couple(P, s, 0, k), (k,)))
        if N <= 8:
            # The appended tail must NOT share the head's pattern, or a rule
            # that reorders rows would go unnoticed.
            out.append((lambda v: torch.cat([v, v.sum().reshape(1)]),
                        lambda P, s: pr.cat([P, pr.reduce_sum(P, s, (0,))], [s, (1,)], 0),
                        (N + 1,)))
    if len(shape) == 2:
        r, c = shape
        out.append((lambda v: v.T,
                    lambda P, s: pr.gather(P, np.arange(N).reshape(s).T.reshape(-1)),
                    (c, r)))
        out.append((lambda v: v * v.sum(dim=1, keepdim=True),
                    lambda P, s: pr.pointwise(
                        s, [(P, s), (pr.reduce_sum(P, s, (1,)), (s[0], 1))]), shape))
        out.append((lambda v: torch.softmax(v, dim=1),
                    lambda P, s: pr.slice_couple(P, s, (1,)), shape))
        out.append((lambda v: v.sort(dim=1).values,
                    lambda P, s: pr.slice_couple(P, s, (1,)), shape))
        if N > 2:
            for d in (0, 1):
                out.append((lambda v, d=d: v.sum(dim=d),
                            lambda P, s, d=d: pr.reduce_sum(P, s, (d,)), (shape[1 - d],)))
        if N <= 8:  # joining along dim 1 interleaves rows, unlike dim 0
            out.append((lambda v: torch.cat([v, v.sum(dim=1, keepdim=True)], dim=1),
                        lambda P, s: pr.cat(
                            [P, pr.reduce_sum(P, s, (1,))], [s, (s[0], 1)], 1),
                        (r, c + 1)))
        A = torch.zeros(c, 2, dtype=F64)
        A[0, 0] = A[c - 1, 1] = 1.0  # deliberately full of zeros
        out.append((lambda v, A=A: v @ A,
                    lambda P, s, c=c: pr.mm(P, None, s[0], c, 2), (r, 2)))
    return out


def _chain(rng, n_in, depth):
    shape, steps = (n_in,), []
    for _ in range(depth):
        cands = _candidates(shape)
        fo, fp, shape = cands[int(rng.integers(len(cands)))]
        steps.append((fo, fp))
    return steps


def _build(steps, n_in):
    def f(x):
        v = x
        for fo, _ in steps:
            v = fo(v)
        return v.reshape(-1)

    P, shape = bc.eye(n_in), (n_in,)
    for (fo, fp), _ in zip(steps, steps):
        P = fp(P, shape)
        shape = tuple(fo(torch.zeros(shape, dtype=F64)).shape)
    return f, P


@pytest.mark.parametrize("seed", range(120))
def test_random_chains_contain_the_true_jacobian(seed):
    rng = np.random.default_rng(seed)
    n_in = int(rng.choice([4, 6, 8]))
    f, P = _build(_chain(rng, n_in, int(rng.integers(1, 5))), n_in)
    Pd = P.toarray()
    g = torch.Generator().manual_seed(seed)
    for _ in range(3):
        J = dense_jac(f, torch.randn(n_in, generator=g, dtype=F64))
        assert Pd.shape == tuple(J.shape), f"shape drift: {Pd.shape} vs {tuple(J.shape)}"
        missed = np.argwhere((J != 0).numpy() & ~Pd)
        assert not missed.size, f"pattern misses {missed[:5].tolist()}"


# --- convolution -------------------------------------------------------------

CONV = torch.nn.functional.conv2d


def conv_cases():
    # (x_shape, w_shape, kwargs)
    return [
        ((1, 2, 5, 5), (3, 2, 3, 3), {}),
        ((1, 2, 5, 5), (3, 2, 3, 3), {"padding": 1}),
        ((1, 2, 6, 6), (2, 2, 3, 3), {"stride": 2}),
        ((1, 1, 7, 7), (2, 1, 3, 3), {"dilation": 2}),
        ((1, 4, 5, 5), (4, 2, 3, 3), {"groups": 2}),
        ((2, 2, 4, 4), (2, 2, 2, 2), {"padding": 1, "stride": 2}),
        ((1, 2, 5, 4), (3, 2, 3, 2), {"padding": (1, 0)}),
    ]


def brute_conv(x_shape, w_shape, stride=1, padding=0, dilation=1, groups=1):
    # Nested loops over exactly the sum that defines a convolution.
    N, Ci, H, W = x_shape
    Co, Cig, KH, KW = w_shape
    sh, sw = pr._pair(stride)
    ph, pw = pr._pair(padding)
    dh, dw = pr._pair(dilation)
    OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    Min = np.zeros((N * Co * OH * OW, N * Ci * H * W), dtype=bool)
    Mw = np.zeros((N * Co * OH * OW, Co * Cig * KH * KW), dtype=bool)
    for n in range(N):
        for co in range(Co):
            g = co // (Co // groups)
            for oh in range(OH):
                for ow in range(OW):
                    o = ((n * Co + co) * OH + oh) * OW + ow
                    for cil in range(Cig):
                        ci = g * Cig + cil
                        for kh in range(KH):
                            for kw in range(KW):
                                ih = oh * sh - ph + kh * dh
                                iw = ow * sw - pw + kw * dw
                                if 0 <= ih < H and 0 <= iw < W:
                                    Min[o, ((n * Ci + ci) * H + ih) * W + iw] = True
                                    Mw[o, ((co * Cig + cil) * KH + kh) * KW + kw] = True
    return Min, Mw


class TestConv2d:
    @pytest.mark.parametrize("xs,ws,kw", conv_cases())
    def test_incidence_against_loops(self, xs, ws, kw):
        nx, nw = int(np.prod(xs)), int(np.prod(ws))
        Min, Mw = brute_conv(xs, ws, **kw)
        assert (pr.conv2d(bc.eye(nx), None, xs, ws, **kw).toarray() == Min).all()
        assert (pr.conv2d(None, bc.eye(nw), xs, ws, **kw).toarray() == Mw).all()

    @pytest.mark.parametrize("xs,ws,kw", conv_cases())
    def test_tracked_input(self, xs, ws, kw):
        nx = int(np.prod(xs))
        w = torch.randn(ws, generator=torch.Generator().manual_seed(0), dtype=F64)
        f = lambda z: CONV(z.reshape(xs), w, **kw).reshape(-1)
        P = pr.conv2d(bc.eye(nx), None, xs, ws, **kw)
        assert_sound(P, f, nx, points=3)
        assert_exact(P, f, nx)  # random weights, so every tap is live

    @pytest.mark.parametrize("xs,ws,kw", conv_cases())
    def test_tracked_weight(self, xs, ws, kw):
        nw = int(np.prod(ws))
        x = torch.randn(xs, generator=torch.Generator().manual_seed(1), dtype=F64)
        f = lambda z: CONV(x, z.reshape(ws), **kw).reshape(-1)
        P = pr.conv2d(None, bc.eye(nw), xs, ws, **kw)
        assert_sound(P, f, nw, points=3)

    def test_both_tracked(self):
        xs, ws = (1, 2, 5, 5), (3, 2, 3, 3)
        nx, nw = int(np.prod(xs)), int(np.prod(ws))
        f = lambda z: CONV(z[:nx].reshape(xs), z[nx:].reshape(ws)).reshape(-1)
        P = pr.conv2d(
            pr.gather(bc.eye(nx + nw), np.arange(nx)),
            pr.gather(bc.eye(nx + nw), np.arange(nx, nx + nw)),
            xs, ws,
        )
        assert_sound(P, f, nx + nw, points=3)

    def test_groups_do_not_mix_channels(self):
        xs, ws = (1, 4, 5, 5), (4, 2, 3, 3)
        P = pr.conv2d(bc.eye(int(np.prod(xs))), None, xs, ws, groups=2).toarray()
        per_ch = 5 * 5
        # Output channels 0,1 are group 0 and may only touch input channels 0,1.
        assert not P[: 2 * 9, 2 * per_ch:].any()
        assert not P[2 * 9:, : 2 * per_ch].any()

    def test_padding_taps_are_dropped(self):
        # A 3x3 kernel with padding 1 at the top-left corner touches 4 real taps,
        # not 9. Keeping the padded ones would be sound but needlessly loose.
        xs, ws = (1, 1, 4, 4), (1, 1, 3, 3)
        P = pr.conv2d(bc.eye(16), None, xs, ws, padding=1)
        assert bc.row_nnz(P)[0] == 4
        assert bc.row_nnz(P)[5] == 9  # an interior output sees the full kernel

    def test_rejects_bad_group_split(self):
        with pytest.raises(ValueError, match="in-channels per group"):
            pr.conv2d(bc.eye(50), None, (1, 2, 5, 5), (3, 3, 3, 3))

    def test_rejects_oversized_kernel(self):
        with pytest.raises(ValueError, match="does not fit"):
            pr.conv2d(bc.eye(8), None, (1, 1, 2, 4), (1, 1, 3, 3))

    def test_needs_a_tracked_operand(self):
        with pytest.raises(ValueError, match="at least one tracked operand"):
            pr.conv2d(None, None, (1, 1, 3, 3), (1, 1, 2, 2))

    def test_end_to_end_recovers_a_conv_jacobian(self):
        # The case the whole library exists for: a conv Jacobian is sparse, and
        # the colors equal the lower bound, in-channels times kernel area.
        xs, ws = (1, 2, 8, 8), (4, 2, 3, 3)
        nx = int(np.prod(xs))
        w = torch.randn(ws, generator=torch.Generator().manual_seed(3), dtype=F64)
        f = lambda z: CONV(z.reshape(xs), w, padding=1).reshape(-1)
        P = pr.conv2d(bc.eye(nx), None, xs, ws, padding=1)
        c = cl.color_cols(P)
        assert c.n_colors == c.lower_bound == 2 * 3 * 3

        x = torch.randn(nx, generator=torch.Generator().manual_seed(4), dtype=F64)
        S = seeds(c, dtype=F64)
        B = torch.stack(
            [torch.func.jvp(f, (x,), (S[:, k],))[1] for k in range(c.n_colors)], dim=1
        )
        torch.testing.assert_close(
            decompress(B, c).to_dense(), dense_jac(f, x), rtol=1e-12, atol=1e-14
        )



# --- review findings ---------------------------------------------------------


class TestCancellationThroughReduce:
    def test_stack_negate_sum_is_loose_not_wrong(self):
        # The incoming pattern is exact and sum's own Jacobian is all ones, yet
        # x + (-x) has a zero Jacobian. This is why reduce_sum is structural.
        f = lambda x: torch.stack([x, -x]).sum(0)
        P = pr.reduce_sum(pr.stack([bc.eye(3), bc.eye(3)], [(3,), (3,)], 0), (2, 3), (0,))
        assert_sound(P, f, 3)
        assert not (dense_jac(f, torch.randn(3, dtype=F64)) != 0).any()
        assert P.nnz == 3
        assert pr.TIGHTNESS["reduce_sum"] == "structural"


def empty_rows(n_rows, n_cols=5):
    return bc.from_pairs([], [], (n_rows, n_cols))


def numel(shape):
    return int(np.prod(shape)) if len(shape) else 1


class TestDegenerateShapes:
    """Zero-sized axes and scalars. Row counts are checked against torch itself."""

    @pytest.mark.parametrize(
        "shape,dims",
        [((0, 3), (0,)), ((0, 3), (1,)), ((2, 0, 4), (1,)), ((2, 0, 4), (0, 2)),
         ((3,), (0,)), ((), (0,)), ((), (-1,))],
    )
    def test_reduce_sum_rows(self, shape, dims):
        want = torch.zeros(shape).sum(dim=dims).numel()
        assert pr.reduce_sum(empty_rows(numel(shape)), shape, dims).shape[0] == want

    @pytest.mark.parametrize("shape,dims", [((0, 3), (0,)), ((0, 3), (1,)), ((2, 0), (1,)), ((), (0,))])
    def test_slice_couple_rows(self, shape, dims):
        assert pr.slice_couple(empty_rows(numel(shape)), shape, dims).shape[0] == numel(shape)

    @pytest.mark.parametrize(
        "shape,dim,n_out", [((6,), 0, 0), ((2, 3), 1, 0), ((0, 3), 1, 2), ((2, 3), 0, 4)]
    )
    def test_index_couple_rows(self, shape, dim, n_out):
        want = torch.zeros(shape).index_select(dim, torch.zeros(n_out, dtype=torch.long)).numel()
        assert pr.index_couple(empty_rows(numel(shape)), shape, dim, n_out).shape[0] == want

    def test_index_couple_on_a_scalar(self):
        # A scalar has one line along its only axis, so every output draws on it.
        P = pr.index_couple(bc.eye(1), (), 0, 3)
        assert P.shape == (3, 1) and P.toarray().all()

    def test_slice_couple_on_a_scalar(self):
        assert pr.slice_couple(bc.eye(1), (), (0,)).shape == (1, 1)

    def test_scalar_sum_is_sound(self):
        f = lambda x: x.reshape(()).sum(0).reshape(1)
        assert_sound(pr.reduce_sum(bc.eye(1), (), (0,)), f, 1)


class TestCouplingCost:
    """Reproduced by review: a 1000-element slice on one scalar built a million
    incidence pairs to produce 1000 pattern entries."""

    @pytest.fixture
    def pairs(self, monkeypatch):
        sizes = []
        real = bc.from_pairs

        def spy(rows, cols, shape):
            sizes.append(len(rows))
            return real(rows, cols, shape)

        monkeypatch.setattr(bc, "from_pairs", spy)
        return sizes

    def test_slice_couple_is_linear(self, pairs):
        P = pr.gather(bc.eye(1), np.zeros(1000, dtype=np.int64))
        assert pr.slice_couple(P, (1000,), (0,)).nnz == 1000
        assert max(pairs) <= 1000

    def test_index_couple_is_linear(self, pairs):
        P = pr.gather(bc.eye(1), np.zeros(1000, dtype=np.int64))
        assert pr.index_couple(P, (1000,), 0, 1000).nnz == 1000
        assert max(pairs) <= 1000
