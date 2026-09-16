import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl
from src.compress import decompress, seeds
from src.trace import CustomBackward, TraceMismatch, UnsupportedOp, sparsity

F64 = torch.float64
CONV = torch.nn.functional.conv2d


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


class TestCouplings:
    def test_softmax(self):
        f = lambda z: torch.softmax(z.reshape(2, 3), dim=1).reshape(-1)
        x = torch.randn(6, dtype=F64)
        P = sparsity(f, x)
        assert_contains(P, f, x)
        assert_exact(P, f, x)

    # x.sum() and torch.sum(x) export as sum(x, []), where an empty list means
    # every dim. Read as "no dims" it gave one row per element for a scalar.
    @pytest.mark.parametrize("f", [lambda z: z.reshape(2, 3).sum(1),
                                   lambda z: z.reshape(2, 3).sum().reshape(1),
                                   lambda z: torch.sum(z).reshape(1)])
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
    def test_unsupported_op_names_the_op_and_the_line(self):
        def f(z):
            return torch.cumsum(z, 0)

        with pytest.raises(UnsupportedOp) as e:
            sparsity(f, torch.randn(4, dtype=F64))
        msg = str(e.value)
        assert "aten.cumsum" in msg
        assert "test_trace.py" in msg and "torch.cumsum(z, 0)" in msg

    def test_anchor_registry_is_left_as_found(self):
        # The exact-line trace uses a private torch registry. Tracing, refused or
        # not, must not leave user functions behind in it.
        import torch.fx.proxy as fxp

        if not hasattr(fxp, "_STACK_TRACE_ANCHORS"):
            pytest.skip("this torch has no stack trace anchor registry")
        before = set(fxp._STACK_TRACE_ANCHORS)

        def g(z):
            return torch.cumsum(z, 0)

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
