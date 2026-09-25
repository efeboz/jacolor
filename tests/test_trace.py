import warnings

import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl
from src.compress import decompress, seeds
from src.trace import (CustomBackward, TraceMismatch, TraceUnchecked, UnsupportedOp,
                       sparsity)

F64 = torch.float64
CONV = torch.nn.functional.conv2d
F_ = torch.nn.functional


def dense_jac(f, x):
    return torch.func.jacrev(f)(x).reshape(-1, x.numel())


def assert_contains(P, f, x0, points=4, seed=0):
    # The contract: every nonzero of the true Jacobian, at several points, is in P.
    g = torch.Generator().manual_seed(seed)
    Pd = P.toarray()
    for _ in range(points):
        nz = (dense_jac(f, torch.randn(x0.shape, generator=g, dtype=F64)) != 0).numpy()
        assert nz.shape == Pd.shape, f"pattern {Pd.shape} vs Jacobian {nz.shape}"
        missed = np.argwhere(nz & ~Pd)
        assert not missed.size, f"pattern misses {missed[:5].tolist()}"


def assert_exact(P, f, x0, seed=1):
    x = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed), dtype=F64)
    nz = (dense_jac(f, x) != 0).numpy()
    assert nz.shape == P.shape, f"pattern {P.shape} vs Jacobian {nz.shape}"
    extra = np.argwhere(P.toarray() & ~nz)
    assert not extra.size, f"pattern is loose at {extra[:5].tolist()}"


def jvp_jacobian(f, x, P):
    # The whole workflow: color, seed, one jvp per color, decompress.
    c = cl.color_cols(P)
    S = seeds(c, dtype=F64)
    B = torch.stack([torch.func.jvp(f, (x,), (S[:, k],))[1] for k in range(c.n_colors)], dim=1)
    return c, decompress(B, c).to_dense()


def band(x):
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])


class TestBanded:
    """The README example, with the pattern read off the function itself."""

    def test_matches_the_hand_built_pattern(self):
        n = 12
        rows = np.repeat(np.arange(n - 2), 3)
        cols = (np.arange(n - 2)[:, None] + np.arange(3)).ravel()
        want = bc.from_pairs(rows, cols, (n - 2, n)).toarray()
        assert (sparsity(band, torch.randn(n, dtype=F64)).toarray() == want).all()

    def test_end_to_end(self):
        x = torch.randn(12, dtype=F64)
        c, J = jvp_jacobian(band, x, sparsity(band, x))
        assert c.n_colors == 3
        torch.testing.assert_close(J, dense_jac(band, x), rtol=1e-12, atol=1e-14)


class TestConv:
    def test_closure_weight_end_to_end(self):
        w = torch.randn(4, 2, 3, 3, generator=torch.Generator().manual_seed(0), dtype=F64)
        f = lambda z: CONV(z.reshape(1, 2, 8, 8), w, padding=1).reshape(-1)
        x = torch.randn(128, dtype=F64)
        P = sparsity(f, x)
        assert_exact(P, f, x)
        c, J = jvp_jacobian(f, x, P)
        assert c.n_colors == c.lower_bound == 2 * 3 * 3
        torch.testing.assert_close(J, dense_jac(f, x), rtol=1e-12, atol=1e-14)

    def test_module_with_bias_relu_and_sum(self):
        torch.manual_seed(0)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = torch.nn.Conv2d(2, 3, 3, padding=1, dtype=F64)

            def forward(self, z):
                return torch.relu(self.c(z.reshape(1, 2, 5, 5))).sum((2, 3)).reshape(-1)

        net = Net()
        x = torch.randn(50, dtype=F64)
        assert_contains(sparsity(net, x), net, x)

    def test_transposed_is_refused(self):
        w = torch.randn(2, 1, 3, 3, dtype=F64)
        f = lambda z: torch.nn.functional.conv_transpose2d(z.reshape(1, 2, 4, 4), w).reshape(-1)
        with pytest.raises(UnsupportedOp, match="transposed"):
            sparsity(f, torch.randn(32, dtype=F64))


ROW_MAPS = [
    ("view and permute", lambda z: z.reshape(3, 4).T.reshape(-1)),
    ("slice with step", lambda z: z[1:-1:2]),
    ("slice on dim 1", lambda z: z.reshape(3, 4)[:, 1:3].reshape(-1)),
    ("expand", lambda z: z[:4].reshape(4, 1).expand(4, 3).reshape(-1)),
    ("unsqueeze", lambda z: z.unsqueeze(0).reshape(-1)),
    ("select", lambda z: z.reshape(3, 4)[:, 2]),
]


@pytest.mark.parametrize("name,f", ROW_MAPS, ids=[r[0] for r in ROW_MAPS])
def test_row_maps_are_exact(name, f):
    x = torch.randn(12, dtype=F64)
    P = sparsity(f, x)
    assert_contains(P, f, x)
    assert_exact(P, f, x)


POINTWISE = [
    ("add with broadcast", lambda z: (z.reshape(3, 4) + z[:4]).reshape(-1)),
    ("sub", lambda z: z[:6] - z[6:]),
    ("mul by a scalar", lambda z: z * 2.5),
    ("div", lambda z: z[:6] / (z[6:] + 3.0)),
    ("neg and exp", lambda z: torch.exp(-z)),
    ("tanh sin cos", lambda z: torch.tanh(z) + torch.sin(z) * torch.cos(z)),
    ("relu sigmoid", lambda z: torch.relu(z) * torch.sigmoid(z)),
]


@pytest.mark.parametrize("name,f", POINTWISE, ids=[p[0] for p in POINTWISE])
def test_pointwise_contains_the_jacobian(name, f):
    x = torch.randn(12, dtype=F64)
    assert_contains(sparsity(f, x), f, x)


IDX = torch.tensor([0, 3, 3, 4])
PAD = torch.nn.functional.pad


def unbound(z):
    a, b, c = z.reshape(3, 4)
    return a * b - c


def slice_write(z):
    r = torch.zeros_like(z)
    r[1:-1] = z[1:-1] * 2
    return r


def repeated_put(z):
    r = z.clone()
    r[torch.tensor([5, 5])] = z[:2] * 3
    return r


def row_write(z):
    r = torch.zeros(3, 4, dtype=F64)
    r[1] = z[:4]
    return r.reshape(-1)


def boundary_writes(z):
    r = torch.zeros(12, dtype=F64)
    r[0] = z[0] - 1
    r[1:-1] = z[:-2] - 2 * z[1:-1] + z[2:]
    r[-1] = z[-1] ** 2
    return r

# Every output element is one input element or none, so these are held to exact.
MOVES = [
    ("roll", lambda z: torch.roll(z, 2)),
    ("flip", lambda z: z.flip(0)),
    ("constant pad and crop", lambda z: PAD(z, (1, 2)) + PAD(z, (-1, 4))),
    ("2d pad with a value", lambda z: PAD(z.reshape(3, 4), (1, 1, 1, 1), value=2.0).reshape(-1)),
    ("circular pad", lambda z: PAD(z.reshape(1, 3, 4), (1, 1), mode="circular").reshape(-1)),
    ("reflect pad", lambda z: PAD(z.reshape(1, 3, 4), (1, 1), mode="reflect").reshape(-1)),
    ("x[idx]", lambda z: z[IDX]),
    ("x[rows, cols]", lambda z: z.reshape(3, 4)[torch.tensor([0, 2]), torch.tensor([1, 3])]),
    ("x[:, idx]", lambda z: z.reshape(3, 4)[:, torch.tensor([0, 2])].reshape(-1)),
    ("x[constant mask]", lambda z: z[torch.arange(12) % 3 == 0]),
    ("index_select", lambda z: z.index_select(0, IDX)),
    ("gather", lambda z: z.reshape(3, 4).gather(1, torch.tensor([[0, 1], [2, 2], [3, 0]])).reshape(-1)),
    ("repeat and tile", lambda z: torch.cat([z.repeat(2), z.reshape(3, 4).tile(1, 2).reshape(-1)])),
    ("unfold", lambda z: z.unfold(0, 3, 2).reshape(-1)),
    ("diagonal", lambda z: torch.diagonal(z.reshape(3, 4), 1)),
    ("split and chunk", lambda z: torch.cat(z.split([5, 7])[::-1]) * torch.cat(z.chunk(3)[::-1])),
    ("unbind", unbound),
    ("slice write into a buffer", slice_write),
    # Export writes one row as a where over a mask built from literals, which is
    # read, since nothing can change it between evaluations.
    ("row write into a buffer", row_write),
    ("boundary writes", boundary_writes),
    ("tril", lambda z: torch.tril(z.reshape(3, 4)).reshape(-1)),
    ("masked_fill with a constant mask", lambda z: z.masked_fill(torch.arange(12) % 2 == 0, 0.0)),
    ("detach", lambda z: z * z.detach() + z.detach().sum()),
]


@pytest.mark.parametrize("name,f", MOVES, ids=[m[0] for m in MOVES])
def test_moves_are_exact(name, f):
    x = torch.randn(12, dtype=F64)
    P = sparsity(f, x)
    assert_contains(P, f, x)
    assert_exact(P, f, x)


# A derivative that depends on values, or rules that are conservative on
# purpose. Held to containment.
VALUED = [
    ("comparisons into where", lambda z: torch.where((z > 0) & (z <= 1) | (z == 0.5), z, -z)),
    ("where on two tracked", lambda z: torch.where(z[:6] > z[6:], z[:6], z[6:])),
    ("sign and rounding", lambda z: z.sign() * z + z.floor() + z.round()),
    ("argmax as a value", lambda z: z.argmax().reshape(1).double() + z[:1]),
    ("mask from a comparison", lambda z: z * (z > 0).double() + z.masked_fill(z > 0.5, 0.0)),
    ("clamp", lambda z: z.clamp(-1, 1) + z.clamp_min(0) + z.clamp(min=torch.zeros(12, dtype=F64))),
    ("maximum and minimum", lambda z: torch.maximum(z, z.flip(0)) + torch.minimum(z, torch.zeros(12, dtype=F64))),
    ("elementary", lambda z: (z / 4).log1p() + z.expm1() + z.tan() + (z / 4).asin() + z.cosh()
                             + z.erf() + z.abs().log2() + (z / 4).atanh()),
    ("atan2 and pow", lambda z: torch.atan2(z, z.flip(0) + 2) + z.abs().pow(z) + 2.0 ** z),
    ("activations", lambda z: F_.gelu(z) + F_.silu(z) + F_.softplus(z) + F_.elu(z) + F_.leaky_relu(z)),
    ("bmm", lambda z: torch.bmm(z.reshape(2, 2, 3), z.reshape(2, 3, 2)).reshape(-1)),
    ("einsum", lambda z: torch.einsum("bij,bj->bi", z.reshape(2, 2, 3), z[:6].reshape(2, 3)).reshape(-1)),
    # Along different dims of one shape, so neither prefix covers the other's.
    ("cumsum", lambda z: (z.reshape(3, 4).cumsum(1) * z.reshape(3, 4).cumsum(0)).reshape(-1)),
    ("cumprod", lambda z: z.reshape(3, 4).cumprod(0).reshape(-1)),
    ("prod amax logsumexp", lambda z: z.reshape(3, 4).prod(1) + z.reshape(3, 4).amax(1)
                                      + z.reshape(3, 4).logsumexp(1)),
    ("max over a dim", lambda z: z.reshape(3, 4).max(1).values + z.reshape(3, 4).min(0).values[:3]),
    ("var and norm", lambda z: z.reshape(3, 4).std(1) + torch.linalg.vector_norm(z.reshape(3, 4), dim=1)),
    ("cross", lambda z: torch.linalg.cross(z[:3], z[3:6])),
    ("index_add", lambda z: z[:5].index_add(0, IDX, z[4:8])),
    ("scatter_add", lambda z: z.reshape(3, 4).scatter_add(
        1, torch.tensor([[0, 0], [3, 1], [2, 2]]), z[:6].reshape(3, 2)).reshape(-1)),
    ("scatter", lambda z: z[:6].scatter(0, torch.tensor([1, 4]), z[6:8])),
    ("put with a repeated index", repeated_put),
    ("sort", lambda z: z.reshape(3, 4).sort(1).values.reshape(-1)),
    ("stable descending sort", lambda z: torch.sort(z, stable=True, descending=True).values),
    ("topk", lambda z: z.reshape(3, 4).topk(2, dim=0).values.reshape(-1)),
    ("kthvalue and median", lambda z: z.reshape(3, 4).kthvalue(2).values + z.reshape(3, 4).median(1).values),
    ("cummax and cummin", lambda z: (z.reshape(3, 4).cummax(1).values
                                     * z.reshape(3, 4).cummin(0).values).reshape(-1)),
    ("logcumsumexp", lambda z: torch.logcumsumexp(z.reshape(3, 4), 0).reshape(-1)),
    ("layer_norm", lambda z: F_.layer_norm(z.reshape(1, 3, 4), (3, 4)).reshape(-1)),
    ("layer_norm with a tracked weight", lambda z: F_.layer_norm(z[:8].reshape(2, 4), (4,), z[8:]).reshape(-1)),
    ("scatter_reduce", lambda z: z[4:7].scatter_reduce(0, torch.tensor([0, 1, 1, 2]), z[:4], "prod",
                                                      include_self=False)),
    ("index_reduce", lambda z: z[:6].reshape(2, 3).index_reduce(
        1, torch.tensor([2, 0]), z[6:10].reshape(2, 2), "amax").reshape(-1)),
]


@pytest.mark.parametrize("name,f", VALUED, ids=[v[0] for v in VALUED])
def test_value_dependent_ops_contain_the_jacobian(name, f):
    x = torch.randn(12, dtype=F64)
    assert_contains(sparsity(f, x), f, x)


# An index read off x moves with x, so a pattern taken at one point goes stale.
TRACKED_INDEX = [
    ("mask", lambda z: z[z > 0]),
    ("gather", lambda z: z.gather(0, z.argmax().reshape(1))),
    ("put", lambda z: torch.zeros(12, dtype=F64).index_put((z.argmax().reshape(1),), z[:1])),
    ("scatter", lambda z: torch.zeros(12, dtype=F64).scatter_add(0, z.argmin().reshape(1), z[:1])),
]


def test_agreeing_on_nan_is_not_a_check():
    # Traced, f is sqrt(z), which is diagonal. Run, it is sqrt(z + z.sum()),
    # which is dense. At -1 both give nan everywhere, values and derivative, so
    # they agree without the check having seen anything.
    def f(z):
        if torch.compiler.is_compiling():
            return z.sqrt()
        return (z + z.sum()).sqrt()

    with pytest.warns(TraceUnchecked, match="not finite"):
        sparsity(f, -torch.ones(3, dtype=F64))


def test_a_captured_mask_is_not_read():
    # Only a mask built from literals may be read. A captured tensor can be
    # changed between evaluations, so where keeps both branches.
    # Through an op, since a captured tensor on its own is an input to the graph.
    mask = torch.arange(6) % 2 == 0
    P = sparsity(lambda z: torch.where(~mask, z[:6], z[6:]), torch.randn(12, dtype=F64))
    assert (P.toarray().sum(1) == 2).all()


@pytest.mark.parametrize("name,f", TRACKED_INDEX, ids=[t[0] for t in TRACKED_INDEX])
def test_an_index_computed_from_x_is_refused(name, f):
    with pytest.raises(UnsupportedOp, match="index computed from x"):
        sparsity(f, torch.randn(12, dtype=F64))


class TestCouplings:
    def test_softmax(self):
        f = lambda z: torch.softmax(z.reshape(2, 3), dim=1).reshape(-1)
        x = torch.randn(6, dtype=F64)
        P = sparsity(f, x)
        assert_contains(P, f, x)
        assert_exact(P, f, x)

    # x.sum() and torch.sum(x) export as sum(x, []), where an empty list means
    # every dim. Read as "no dims" it gave one row per element for a scalar.
    # dim=None exports as sum(x, None) and means the same.
    @pytest.mark.parametrize("f", [lambda z: z.reshape(2, 3).sum(1),
                                   lambda z: z.reshape(2, 3).sum().reshape(1),
                                   lambda z: torch.sum(z).reshape(1),
                                   lambda z: z.sum(dim=None, keepdim=True)])
    def test_sum(self, f):
        x = torch.randn(6, dtype=F64)
        P = sparsity(f, x)
        assert_contains(P, f, x)
        assert_exact(P, f, x)

    @pytest.mark.parametrize("which", ["tracked @ constant", "constant @ tracked", "both tracked"])
    def test_mm(self, which):
        A = torch.randn(3, 2, generator=torch.Generator().manual_seed(2), dtype=F64)
        f = {
            "tracked @ constant": lambda z: (z[:6].reshape(2, 3) @ A).reshape(-1),
            "constant @ tracked": lambda z: (A @ z[:6].reshape(2, 3)).reshape(-1),
            "both tracked": lambda z: (z[:6].reshape(2, 3) @ z[6:].reshape(3, 2)).reshape(-1),
        }[which]
        x = torch.randn(12, dtype=F64)
        assert_contains(sparsity(f, x), f, x)

    def test_cat_with_an_untracked_part(self):
        f = lambda z: torch.cat([z.reshape(2, 3), torch.ones(2, 1, dtype=F64)], dim=1).reshape(-1)
        x = torch.randn(6, dtype=F64)
        P = sparsity(f, x)
        assert_contains(P, f, x)
        assert_exact(P, f, x)


class TestInputsAndOutputs:
    def test_two_dimensional_input(self):
        f = lambda z: torch.softmax(z, dim=0).sum(1)
        x = torch.randn(3, 4, dtype=F64)
        assert_contains(sparsity(f, x), f, x)

    def test_output_that_ignores_the_input(self):
        P = sparsity(lambda z: torch.ones(3, dtype=F64), torch.randn(5, dtype=F64))
        assert P.shape == (3, 5) and P.nnz == 0

    def test_unsupported_op_on_untracked_data_just_runs(self):
        f = lambda z: z * torch.cumsum(torch.ones(4, dtype=F64), 0)
        x = torch.randn(4, dtype=F64)
        assert_exact(sparsity(f, x), f, x)


class Widen(torch.autograd.Function):
    # The forward reads only z[0]. The backward sends gradient to every element.
    # A pattern built from the traced forward would miss every column but one.
    @staticmethod
    def forward(ctx, z):
        return z[0].expand(z.shape[0]).clone()

    @staticmethod
    def backward(ctx, g):
        return g


class TestRefusals:
    def test_complex_input_and_output(self):
        with pytest.raises(TypeError, match="complex input"):
            sparsity(lambda z: z * 2, torch.ones(3, dtype=torch.complex128))
        with pytest.raises(TypeError, match="complex output"):
            sparsity(lambda z: z * (1 + 2j), torch.ones(3, dtype=F64))

    def test_unsupported_op_names_the_op_and_the_line(self):
        def f(z):
            return torch.lgamma(z)

        with pytest.raises(UnsupportedOp) as e:
            sparsity(f, torch.randn(4, dtype=F64))
        msg = str(e.value)
        assert "aten.lgamma" in msg
        # The exact line comes from a private torch hook. Without it the message
        # still names the op, it just cannot point at the caller.
        import torch.fx.proxy as fxp

        if not hasattr(fxp, "_STACK_TRACE_ANCHORS"):
            pytest.skip("this torch has no stack trace anchor registry")
        assert "test_trace.py" in msg and "torch.lgamma(z)" in msg

    def test_a_cast_goes_through(self):
        # Export puts an assertion ahead of the cast, which has to be ignored
        # rather than refused for the cast to reach its own rule.
        f = lambda z: z.float().double() * z
        x = torch.randn(4, dtype=F64)
        P = sparsity(f, x)
        assert_contains(P, f, x)
        assert_exact(P, f, x)

    def test_anchor_registry_is_left_as_found(self):
        # The exact-line trace uses a private torch registry. Tracing, refused or
        # not, must not leave user functions behind in it.
        import torch.fx.proxy as fxp

        if not hasattr(fxp, "_STACK_TRACE_ANCHORS"):
            pytest.skip("this torch has no stack trace anchor registry")
        before = set(fxp._STACK_TRACE_ANCHORS)

        def g(z):
            return torch.lgamma(z)

        with pytest.raises(UnsupportedOp):
            sparsity(g, torch.randn(4, dtype=F64))
        sparsity(band, torch.randn(6, dtype=F64))
        assert fxp._STACK_TRACE_ANCHORS == before

    def test_custom_backward_is_refused(self):
        with pytest.raises(CustomBackward, match="Widen"):
            sparsity(lambda z: Widen.apply(z) * 2.0, torch.randn(4, dtype=F64))

    def test_a_refusal_leaves_tracing_usable(self):
        # Detection no longer patches Function.apply, but a refusal still has to
        # leave no global state behind.
        with pytest.raises(CustomBackward):
            sparsity(lambda z: Widen.apply(z), torch.randn(4, dtype=F64))
        assert sparsity(torch.sin, torch.randn(3, dtype=F64)).nnz == 3


# --- randomized sweep through the tracer ---------------------------------------
# Same idea as the propagate sweep, but the pattern comes from sparsity(f, x), so
# the interpreter's plumbing is in the loop: argument mapping, untracked
# branches, shapes read off real values.


def _candidates(shape):
    N = int(np.prod(shape))
    c = [lambda v: torch.tanh(v) * v, lambda v: torch.exp(-v) + torch.sin(v)]
    if len(shape) == 1:
        if N % 2 == 0 and N > 2:
            h = N // 2
            c += [lambda v, h=h: v.reshape(2, h), lambda v, h=h: v[:h] * v[h:]]
        if N > 2:
            c += [lambda v: v[1:], lambda v: v[::2]]
        if N <= 8:
            c.append(lambda v: torch.cat([v, v.sum().reshape(1)]))
            # Untracked part first: a wrong row count for it shifts every tracked row.
            c.append(lambda v: torch.cat([torch.ones(2, dtype=F64), v]))
    if len(shape) == 2:
        k = shape[1]
        c += [lambda v: v.T, lambda v: torch.softmax(v, dim=1),
              lambda v: v * v.sum(dim=1, keepdim=True), lambda v: v.sum(dim=0),
              lambda v: v.reshape(-1), lambda v, k=k: v @ torch.eye(k, 2, dtype=F64)]
        if N <= 8:
            c.append(lambda v: torch.cat([v, torch.relu(v)], dim=1))
    return c


@pytest.mark.parametrize("seed", range(30))
def test_random_traced_chains_contain_the_true_jacobian(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([4, 6, 8]))
    steps, shape = [], (n,)
    for _ in range(int(rng.integers(1, 5))):
        cands = _candidates(shape)
        op = cands[int(rng.integers(len(cands)))]
        steps.append(op)
        shape = tuple(op(torch.zeros(shape, dtype=F64)).shape)

    def f(z):
        for op in steps:
            z = op(z)
        return z.reshape(-1)

    x = torch.randn(n, dtype=F64)
    assert_contains(sparsity(f, x), f, x, points=3, seed=seed)


class Cached(torch.autograd.Function):
    """Forward reads only z[0], backward sends gradient everywhere."""

    generate_vmap_rule = True

    @staticmethod
    def forward(z):
        return z[0].expand(z.shape[0]).clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, g):
        return g

    @staticmethod
    def jvp(ctx, gz):
        return gz


CACHED_APPLY = Cached.apply  # captured before any tracing, as a user might

_W = torch.randn(3, 2, 3, 3, generator=torch.Generator().manual_seed(15), dtype=F64)


class TestTraceFidelity:
    """The trace has to stand for the function autograd will differentiate."""

    def test_export_specialized_branch_is_refused(self):
        # Export takes the is_compiling branch, so the traced program computes z
        # while f computes z + z.sum(). The pattern would be the wrong shape of
        # dependency entirely, and the values silently wrong with it.
        f = lambda z: z if torch.compiler.is_compiling() else z + z.sum()
        with pytest.raises(TraceMismatch, match="does not agree with f"):
            sparsity(f, torch.randn(3, dtype=F64))

    @pytest.mark.parametrize("dt", [F64, torch.float32, torch.float16, torch.bfloat16],
                             ids=["float64", "float32", "float16", "bfloat16"])
    def test_a_specialized_branch_is_refused_in_every_dtype(self, dt):
        # The tolerance follows the dtype, so it must not follow it so far that
        # a coarse one accepts a different program.
        f = lambda z: z if torch.compiler.is_compiling() else z + z.sum()
        with pytest.raises(TraceMismatch, match="does not agree with f"):
            sparsity(f, torch.linspace(-1, 1, 6, dtype=dt))

    @pytest.mark.parametrize("dt", [F64, torch.float32], ids=["float64", "float32"])
    def test_arithmetic_that_rounds_differently_is_not_a_mismatch(self, dt):
        # A matmul and its traced form sum in a different order, which in float32
        # shows in the last bits. That is f's arithmetic, not a different f.
        g = torch.Generator().manual_seed(16)
        W = torch.randn(24, 24, generator=g, dtype=dt)
        k = torch.randn(1, 1, 3, 3, generator=g, dtype=dt)
        for f, n in ((lambda z: W @ z, 24),
                     (lambda z: CONV(z.reshape(1, 1, 6, 6), k, padding=1).reshape(-1), 36)):
            x = torch.randn(n, generator=g, dtype=dt)
            # How far the two land apart is the platform's business. That it is
            # not held to be a different program is this test's.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                P = sparsity(f, x)  # refused before the tolerance followed the dtype
            nz = (torch.func.jacrev(f)(x).reshape(-1, n) != 0).numpy()
            assert not (nz & ~P.toarray()).any()

    # Whether a given matmul cancels far enough to need the scale of the output
    # depends on the kernels a platform picks, so the comparison itself is put
    # the question here rather than a function that may or may not ask it.
    @pytest.mark.parametrize("dt", [F64, torch.float32], ids=["float64", "float32"])
    def test_an_entry_below_the_scale_is_allowed_and_said_to_be(self, dt):
        # An entry that cancels has nothing of its own left to measure against,
        # so it is held to the scale of the largest entry instead. That leaves
        # room for a real difference to hide in it, which is worth saying.
        from src.trace import _agree

        want = torch.tensor([1.0, 1e-12], dtype=dt)
        got = want.clone()
        got[1] = 4e-12  # four times over, and a trillionth of what it is held to
        agree, why = _agree(got, want, 1e-9)
        assert agree and "against the scale of the largest" in why

    @pytest.mark.parametrize("dt", [F64, torch.float32], ids=["float64", "float32"])
    def test_an_entry_of_its_own_size_is_resolved(self, dt):
        from src.trace import _agree

        want = torch.tensor([1.0, 0.5], dtype=dt)
        got = torch.tensor([1.0, 0.5 + 1e-11], dtype=dt)
        assert _agree(got, want, 1e-9) == (True, None)

    def test_an_ordinary_trace_says_nothing(self):
        # Pointwise all the way through, so the traced program lands on the
        # same values whatever the platform, and there is nothing to report.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for dt in (F64, torch.float32):
                sparsity(band, torch.linspace(-1, 1, 12, dtype=dt))
        assert not [m for m in w if isinstance(m.message, TraceUnchecked)]

    SMALL = staticmethod(
        lambda z: z if torch.compiler.is_compiling() else z + 0.01 * z.sum())

    @pytest.mark.parametrize("dt", [F64, torch.float32], ids=["float64", "float32"])
    def test_a_small_difference_in_program_is_still_refused(self, dt):
        # One percent apart, which is under the tolerance a coarse dtype needs.
        with pytest.raises(TraceMismatch):
            sparsity(self.SMALL, torch.zeros(3, dtype=dt))

    def test_a_dtype_coarse_on_one_side_only_still_says_so(self):
        # float16 in through a float64 constant, so the values compare in
        # float64 and resolve while the derivative compares in float16 and
        # cannot. Reading the dtype off the output alone misses this one.
        k = torch.ones(3, dtype=F64)
        f = lambda z: z * k if torch.compiler.is_compiling() else (z + 0.01 * z.sum()) * k
        with pytest.warns(TraceUnchecked, match="float16"):
            P = sparsity(f, torch.zeros(3, dtype=torch.float16))
        assert P.shape == (3, 3)

    def test_a_dtype_coarse_on_the_output_alone_is_refused(self):
        # The mirror of it. The derivative compares in float64 and resolves, so
        # the program one percent away is refused rather than reported.
        k = torch.ones(3, dtype=torch.float16)
        f = lambda z: z * k if torch.compiler.is_compiling() else (z + 0.01 * z.sum()) * k
        with pytest.raises(TraceMismatch):
            sparsity(f, torch.zeros(3, dtype=F64))

    @pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16],
                             ids=["float16", "bfloat16"])
    def test_a_dtype_too_coarse_to_check_says_so(self, dt):
        # The same one percent, where the comparison allows five. The pattern is
        # built, because its structure rarely turns on the dtype, but a trace
        # nothing can stand behind must not pass for a checked one.
        with pytest.warns(TraceUnchecked, match="more than two different programs"):
            P = sparsity(self.SMALL, torch.zeros(3, dtype=dt))
        assert P.shape == (3, 3)

    @pytest.mark.parametrize("k", [1.0, 1e100, 1e-100, 1e-300],
                             ids=["one", "1e100", "1e-100", "1e-300"])
    def test_the_scale_is_measured_where_the_values_live(self, k):
        # Narrowing to float32 to measure it turns 1e100 into an infinity and
        # 1e-100 into zero, and either way the tolerance goes with it.
        from src.trace import _agree

        want = torch.tensor([1.0, 1.0, 0.0], dtype=F64) * k
        got = want.clone()
        got[2] = 1e-16 * k  # an entry that cancels, off by a rounding of the scale
        assert _agree(got, want, 1e-9)[0]

    def test_a_traced_shape_that_differs_is_caught(self):
        # Nothing compares across shapes, so this has to be seen before the
        # values are, or it raises from inside the comparison instead.
        f = lambda z: z if torch.compiler.is_compiling() else torch.cat([z, z])
        with pytest.raises(TraceMismatch, match="does not agree with f"):
            sparsity(f, torch.randn(3, dtype=F64))

    def test_a_difference_in_value_alone_is_caught(self):
        # Same derivative, different function. Only the value check sees this
        # one, which is why it is not the derivative check on its own.
        f = lambda z: z + (0.0 if torch.compiler.is_compiling() else 1.0)
        with pytest.raises(TraceMismatch, match="does not agree with f"):
            sparsity(f, torch.randn(3, dtype=F64))

    def test_apply_captured_before_tracing_is_caught(self):
        # Replacing Function.apply cannot see this call. The autograd graph can.
        with pytest.raises(CustomBackward, match="Cached"):
            sparsity(lambda z: CACHED_APPLY(z), torch.randn(4, dtype=F64))

    def test_apply_through_the_class_is_caught(self):
        with pytest.raises(CustomBackward, match="Cached"):
            sparsity(lambda z: Cached.apply(z), torch.randn(4, dtype=F64))

    def test_a_custom_function_off_the_derivative_path_is_allowed(self):
        # It runs on a constant, so it cannot affect the Jacobian.
        k = torch.ones(4, dtype=F64)
        f = lambda z: z * CACHED_APPLY(k).sum()
        assert sparsity(f, torch.randn(4, dtype=F64)).nnz == 4

    def test_a_random_op_on_the_input_is_refused(self):
        # randn_like takes the tracked input, so the missing-rule guard gets it
        # first. Refused either way, which is the point.
        with pytest.raises(UnsupportedOp, match="randn_like"):
            sparsity(lambda z: z * torch.randn_like(z), torch.randn(4, dtype=F64))

    def test_a_random_constant_makes_the_trace_disagree(self):
        # The constant has no tracked input, so it runs as data and every call
        # draws a different one. No single pattern describes f, and a traced
        # sample would look authoritative.
        with pytest.raises(TraceMismatch):
            sparsity(lambda z: z * torch.randn(4, dtype=F64), torch.randn(4, dtype=F64))

    @pytest.mark.parametrize("name,f,x", [
        ("banded", band, torch.randn(12, dtype=F64)),
        ("conv", lambda z: CONV(z.reshape(1, 2, 5, 5), _W, padding=1).reshape(-1),
         torch.randn(50, dtype=F64)),
        ("softmax", lambda z: torch.softmax(z.reshape(2, 3), dim=1).reshape(-1),
         torch.randn(6, dtype=F64)),
    ])
    def test_no_false_positives_on_ordinary_code(self, name, f, x):
        assert sparsity(f, x).nnz > 0


class TestDetectionContexts:
    """Detection has to hold in the ambient contexts a caller might be in."""

    def test_custom_backward_is_caught_under_no_grad(self):
        # An ambient no_grad leaves no autograd graph to inspect unless the
        # inspection turns recording back on for itself.
        with torch.no_grad():
            with pytest.raises(CustomBackward, match="Cached"):
                sparsity(lambda z: CACHED_APPLY(z), torch.randn(4, dtype=F64))

    def test_ordinary_tracing_works_under_no_grad(self):
        with torch.no_grad():
            assert sparsity(band, torch.randn(12, dtype=F64)).nnz == 30

    def test_custom_backward_is_caught_under_inference_mode(self):
        with torch.inference_mode():
            with pytest.raises(CustomBackward, match="Cached"):
                sparsity(lambda z: CACHED_APPLY(z), torch.randn(4, dtype=F64))


class TestDerivativeFidelity:
    """Equal values at a point do not make two programs the same function."""

    def test_values_can_agree_while_derivatives_differ(self):
        # x.sum() is zero here, so eager and traced return identical values. The
        # pattern comes from the derivative, and the derivatives differ.
        f = lambda z: z if torch.compiler.is_compiling() else z + z.sum()
        x = torch.tensor([1.0, -1.0, 0.0], dtype=F64)
        assert torch.equal(f(x), x)  # a value-only check would pass here
        with pytest.raises(TraceMismatch):
            sparsity(f, x)

    def test_the_value_check_still_catches_its_own_case(self):
        f = lambda z: z if torch.compiler.is_compiling() else z + z.sum()
        with pytest.raises(TraceMismatch):
            sparsity(f, torch.randn(3, dtype=F64))


class TestLinearCoverage:
    """nn.Linear becomes addmm, which is what an ordinary network needs."""

    def test_linear_layer(self):
        torch.manual_seed(0)
        lin = torch.nn.Linear(4, 3, dtype=F64)
        f = lambda z: lin(z)
        x = torch.randn(4, dtype=F64)
        P = sparsity(f, x)
        assert (P.toarray() == (dense_jac(f, x) != 0).numpy()).all()

    def test_linear_without_bias(self):
        torch.manual_seed(1)
        lin = torch.nn.Linear(4, 3, bias=False, dtype=F64)
        f = lambda z: lin(z)
        x = torch.randn(4, dtype=F64)
        assert (sparsity(f, x).toarray() == (dense_jac(f, x) != 0).numpy()).all()

    def test_a_small_mlp(self):
        torch.manual_seed(2)

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 5, dtype=F64)
                self.b = torch.nn.Linear(5, 2, dtype=F64)

            def forward(self, z):
                return self.b(torch.tanh(self.a(z.reshape(1, 4)))).reshape(-1)

        m = MLP()
        x = torch.randn(4, dtype=F64)
        assert (sparsity(m, x).toarray() == (dense_jac(m, x) != 0).numpy()).all()

    def test_matmul_against_a_constant(self):
        A = torch.randn(4, 3, generator=torch.Generator().manual_seed(3), dtype=F64)
        f = lambda z: z @ A
        x = torch.randn(4, dtype=F64)
        assert_contains(sparsity(f, x), f, x)


# A mask built from the tracked input would be a comparison on a tracked value,
# which has no rule and is refused. A constant mask is the supported case.
_MASK = torch.tensor([[True, False, True, False]] * 3)

NEW_OPS = [
    ("pow and sqrt", lambda z: (z**2 + 3.0).sqrt()),
    ("log", lambda z: torch.log(z**2 + 2.0)),
    ("rsqrt", lambda z: torch.rsqrt(z**2 + 1.0)),
    ("reciprocal", lambda z: torch.reciprocal(z**2 + 2.0)),
    ("abs", lambda z: torch.abs(z) * 2.0),
    ("mean over a dim", lambda z: z.reshape(3, 4).mean(1)),
    ("mean over all", lambda z: z.mean().reshape(1)),
    ("log_softmax", lambda z: torch.log_softmax(z.reshape(3, 4), 1).reshape(-1)),
    ("where on a constant mask", lambda z: torch.where(_MASK, z.reshape(3, 4), z.reshape(3, 4) * 2).reshape(-1)),
    ("squeeze and unsqueeze", lambda z: z.reshape(3, 4).unsqueeze(0).squeeze(0).reshape(-1)),
    ("transpose then flatten", lambda z: z.reshape(3, 4).t().flatten()),
]


@pytest.mark.parametrize("name,f", NEW_OPS, ids=[n for n, _ in NEW_OPS])
def test_new_operator_coverage(name, f):
    x = torch.randn(12, generator=torch.Generator().manual_seed(4), dtype=F64)
    assert_contains(sparsity(f, x), f, x)


def test_supported_ops_is_a_sorted_table():
    from src.trace import supported_ops

    table = supported_ops()
    assert table == sorted(table)
    names = [n for n, _ in table]
    assert len(names) == len(set(names))
    assert "addmm.default" in names and "mm.default" in names
    assert set(k for _, k in table) <= {
        "row map", "pointwise", "zero derivative", "reduction", "scan",
        "slice coupling", "coupling", "scatter"
    }


def test_readme_table_matches_the_registry():
    # The README promises a set of operators. It has to be the set that exists.
    import collections
    import pathlib
    import re

    from src.trace import supported_ops

    want = collections.defaultdict(set)
    for name, kind in supported_ops():
        want[kind].add(name)
    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    got = collections.defaultdict(set)
    for kind, ops in re.findall(r"^\| (row map|pointwise|zero derivative|reduction|scan|slice coupling|coupling|scatter) \| (.+?) \|$",
                                readme.read_text(), re.M):
        got[kind] |= {o.strip() for o in ops.split(",")}
    assert got, (
        f"README.md lists no supported-operations table, but the registry has "
        f"{len(supported_ops())} ops. The table and the registry are committed "
        "together or this test fails."
    )
    missing = {k: sorted(want[k] - got.get(k, set())) for k in want if want[k] - got.get(k, set())}
    extra = {k: sorted(got[k] - want.get(k, set())) for k in got if got[k] - want.get(k, set())}
    assert not missing and not extra, f"README missing {missing}, README has extra {extra}"
