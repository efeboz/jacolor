import warnings

import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl
from src.evaluate import VerificationError, VerificationInconclusive, jacobian
from src.trace import sparsity

F64 = torch.float64
CONV = torch.nn.functional.conv2d


def dense_jac(f, x):
    return torch.func.jacrev(f)(x).reshape(-1, x.numel())


def band(x):
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])


def conv_fn():
    w = torch.randn(4, 2, 3, 3, generator=torch.Generator().manual_seed(0), dtype=F64)
    return lambda z: CONV(z.reshape(1, 2, 8, 8), w, padding=1).reshape(-1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = torch.nn.Conv2d(2, 3, 3, padding=1, dtype=F64)

    def forward(self, z):
        return torch.relu(self.c(z.reshape(1, 2, 5, 5))).sum((2, 3)).reshape(-1)


CASES = [
    ("banded", band, torch.randn(12, generator=torch.Generator().manual_seed(1), dtype=F64)),
    ("conv", conv_fn(), torch.randn(128, generator=torch.Generator().manual_seed(2), dtype=F64)),
    ("module", Net(), torch.randn(50, generator=torch.Generator().manual_seed(3), dtype=F64)),
    ("2d input", lambda z: torch.softmax(z, dim=0).sum(1),
     torch.randn(3, 4, generator=torch.Generator().manual_seed(4), dtype=F64)),
]
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("name,f,x", CASES, ids=IDS)
def test_matches_dense_jacrev(name, f, x):
    J = jacobian(f, x)
    assert J.is_sparse and J.is_coalesced()
    torch.testing.assert_close(J.to_dense(), dense_jac(f, x), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name,f,x", CASES, ids=IDS)
def test_reverse_mode_matches(name, f, x):
    c = cl.color_rows(sparsity(f, x))
    J = jacobian(f, x, coloring=c)
    torch.testing.assert_close(J.to_dense(), dense_jac(f, x), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name,f,x", CASES, ids=IDS)
def test_holds_exactly_the_pattern_nonzeros(name, f, x):
    P = sparsity(f, x)
    J = jacobian(f, x, coloring=cl.color_cols(P))
    assert J._nnz() == P.nnz


class TestChunking:
    """Every chunk size must give the same answer, one color at a time included."""

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_every_chunk_size_agrees(self, axis):
        f = conv_fn()
        x = torch.randn(128, generator=torch.Generator().manual_seed(5), dtype=F64)
        P = sparsity(f, x)
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        want = jacobian(f, x, coloring=c).to_dense()
        for chunk in (1, 2, 7, c.n_colors - 1, c.n_colors, c.n_colors + 5):
            got = jacobian(f, x, coloring=c, chunk=chunk).to_dense()
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)

    def test_chunk_bounds_the_block_width(self, monkeypatch):
        # Chunking is a memory bound, not a different answer, so a correctness
        # test cannot see it. Assert on the width of the seed blocks built.
        import src.evaluate as ev

        widths = []
        real = ev._block

        def spy(coloring, lo, hi, dtype, device):
            widths.append(hi - lo)
            return real(coloring, lo, hi, dtype, device)

        monkeypatch.setattr(ev, "_block", spy)
        f = conv_fn()
        x = torch.randn(128, generator=torch.Generator().manual_seed(10), dtype=F64)
        c = cl.color_cols(sparsity(f, x))
        ev.jacobian(f, x, coloring=c, chunk=4)
        assert widths and max(widths) <= 4
        widths.clear()
        ev.jacobian(f, x, coloring=c)
        assert max(widths) == c.n_colors

    def test_chunk_of_one_still_matches_dense(self):
        x = torch.randn(12, generator=torch.Generator().manual_seed(6), dtype=F64)
        J = jacobian(band, x, chunk=1)
        torch.testing.assert_close(J.to_dense(), dense_jac(band, x), rtol=1e-12, atol=1e-14)


class TestReuse:
    """The point of a coloring: trace once, evaluate many times."""

    def test_a_reused_coloring_works_at_other_points(self):
        x0 = torch.randn(12, generator=torch.Generator().manual_seed(7), dtype=F64)
        c = cl.color_cols(sparsity(band, x0))
        g = torch.Generator().manual_seed(8)
        for _ in range(4):
            x = torch.randn(12, generator=g, dtype=F64)
            torch.testing.assert_close(
                jacobian(band, x, coloring=c).to_dense(), dense_jac(band, x),
                rtol=1e-12, atol=1e-14,
            )

    def test_reuse_does_not_trace_again(self, monkeypatch):
        import src.evaluate as ev

        x = torch.randn(12, generator=torch.Generator().manual_seed(9), dtype=F64)
        c = cl.color_cols(sparsity(band, x))
        monkeypatch.setattr(ev, "sparsity", lambda *a, **k: pytest.fail("traced again"))
        ev.jacobian(band, x, coloring=c)


class TestColoringBudget:
    def test_refuses_a_graph_it_cannot_build(self):
        # One row of 20000 entries would need 4e8 intersection entries, about
        # 3.6 GB, to learn that the answer is 20000 colors.
        n = 20_000
        P = bc.from_pairs(np.zeros(n, dtype=np.int64), np.arange(n), (1, n))
        with pytest.raises(MemoryError, match="no coloring can use fewer than 20000"):
            cl.color_cols(P)

    def test_a_wide_dense_pattern_is_not_refused(self):
        # 20001 rows of 100 entries bound the pair count at 2e8, but the graph
        # cannot hold more than 100 by 100 entries, so the budget must see that.
        P = bc.from_dense(np.ones((20001, 100), dtype=bool))
        assert cl.color_cols(P).n_colors == 100

    def test_ordinary_patterns_are_unaffected(self):
        P = sparsity(band, torch.randn(12, dtype=F64))
        assert cl.color_cols(P).n_colors == 3


class Slicer(torch.nn.Module):
    """Its graph depends on an attribute, so a coloring can go stale."""

    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def forward(self, z):
        return z[self.start:self.start + 2]


class TestStaleStructure:
    """A coloring records a graph. It cannot know f still has that graph."""

    def test_a_changed_module_attribute_is_caught(self):
        m = Slicer(0)
        x = torch.randn(4, generator=torch.Generator().manual_seed(11), dtype=F64)
        c = cl.color_cols(sparsity(m, x))
        m.start = 1
        with pytest.raises(VerificationError, match="stale"):
            jacobian(m, x, coloring=c)

    def test_verify_off_returns_the_wrong_answer_quietly(self):
        # The escape hatch really does skip the check, which is why it is not
        # the default.
        m = Slicer(0)
        x = torch.randn(4, generator=torch.Generator().manual_seed(12), dtype=F64)
        c = cl.color_cols(sparsity(m, x))
        m.start = 1
        J = jacobian(m, x, coloring=c, verify=False)
        assert not torch.allclose(J.to_dense(), dense_jac(m, x))

    def test_wrong_input_size(self):
        f = lambda z: z * 2.0
        c = cl.color_cols(sparsity(f, torch.randn(4, dtype=F64)))
        with pytest.raises(ValueError, match="input of 4 elements, got 5"):
            jacobian(f, torch.randn(5, dtype=F64), coloring=c)

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_wrong_output_size_with_the_same_input_size(self, axis):
        f = lambda z: z.sum(dim=0)
        P = sparsity(f, torch.randn(3, 4, dtype=F64))
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with pytest.raises(ValueError, match="output of 4 elements, got 3"):
            jacobian(f, torch.randn(4, 3, dtype=F64), coloring=c)


class TestDegenerate:
    def test_empty_input(self):
        J = jacobian(lambda z: z * 2.0, torch.zeros(0, dtype=F64))
        assert tuple(J.shape) == (0, 0) and J._nnz() == 0

    def test_empty_output_in_reverse(self):
        f = lambda z: z[:0]
        x = torch.randn(3, dtype=F64)
        c = cl.color_rows(sparsity(f, x))
        J = jacobian(f, x, coloring=c)
        assert tuple(J.shape) == (0, 3) and J._nnz() == 0

    def test_dtype_follows_the_ad_pass_not_the_input(self):
        # Forward mode follows f's output, so a float32 input through a float64
        # constant gives a float64 Jacobian. jacrev follows the input instead and
        # returns float32, which is why the values are compared without dtype.
        k = torch.ones(3, dtype=F64)
        f = lambda z: z * k
        x = torch.randn(3, dtype=torch.float32)
        J = jacobian(f, x)
        assert J.dtype == torch.float64
        torch.testing.assert_close(J.to_dense(), dense_jac(f, x), check_dtype=False)

    def test_reverse_follows_the_input_dtype(self):
        k = torch.ones(3, dtype=F64)
        f = lambda z: z * k
        x = torch.randn(3, dtype=torch.float32)
        c = cl.color_rows(sparsity(f, x))
        assert jacobian(f, x, coloring=c).dtype == torch.float32

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "two"])
    def test_chunk_must_be_a_positive_whole_number(self, bad):
        x = torch.randn(12, generator=torch.Generator().manual_seed(13), dtype=F64)
        with pytest.raises(ValueError, match="chunk must be"):
            jacobian(band, x, chunk=bad)


class TestEvaluationCost:
    def test_reverse_builds_the_vjp_once_per_evaluation(self):
        calls = {"n": 0}

        def f(z):
            calls["n"] += 1
            return torch.stack([z[0] * z[1], z[2] * z[3], z[0] + z[3]])

        x = torch.randn(4, generator=torch.Generator().manual_seed(14), dtype=F64)
        c = cl.color_rows(sparsity(f, x))
        assert c.n_colors > 1
        calls["n"] = 0
        jacobian(f, x, coloring=c, chunk=1, verify=False)
        assert calls["n"] == 1


class TestEmptyPatternIsStillChecked:
    """An empty pattern claims every derivative is zero, which is a real claim."""

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_a_constant_coloring_reused_for_the_identity_is_caught(self, axis):
        const = lambda z: torch.zeros(3, dtype=F64)
        x = torch.randn(3, generator=torch.Generator().manual_seed(20), dtype=F64)
        P = sparsity(const, x)
        assert P.nnz == 0
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with pytest.raises(VerificationError):
            jacobian(lambda z: z, x, coloring=c)

    def test_a_genuinely_constant_function_still_passes(self):
        const = lambda z: torch.zeros(3, dtype=F64)
        x = torch.randn(3, generator=torch.Generator().manual_seed(21), dtype=F64)
        assert jacobian(const, x)._nnz() == 0


class Scaled(torch.nn.Module):
    """Slices two elements and scales them, so the derivative size is a knob."""

    def __init__(self, k, start=0):
        super().__init__()
        self.k = k
        self.start = start

    def forward(self, z):
        return z[self.start:self.start + 2] * self.k


SCALES = [(1.0, F64), (1e-8, F64), (1e8, F64), (1e-4, torch.float32), (1e4, torch.float32)]
SCALE_IDS = [f"{k:g}-{str(d).split('.')[-1]}" for k, d in SCALES]


class TestVerificationAcrossScales:
    """Tolerance has to follow the size of the derivative, not sit at a floor."""

    @pytest.mark.parametrize("k,dt", SCALES, ids=SCALE_IDS)
    def test_a_stale_coloring_is_caught_at_any_scale(self, k, dt):
        m = Scaled(k)
        x = torch.randn(4, generator=torch.Generator().manual_seed(22), dtype=dt)
        c = cl.color_cols(sparsity(m, x))
        m.start = 1
        with pytest.raises(VerificationError):
            jacobian(m, x, coloring=c)

    @pytest.mark.parametrize("k,dt", SCALES, ids=SCALE_IDS)
    def test_the_same_problem_passes_when_it_is_valid(self, k, dt):
        m = Scaled(k)
        x = torch.randn(4, generator=torch.Generator().manual_seed(23), dtype=dt)
        assert jacobian(m, x)._nnz() == 2


LOW = [torch.bfloat16, torch.float16, torch.float32, F64]
LOW_IDS = [str(d).split(".")[-1] for d in LOW]


class TestLowPrecisionDtypes:
    """Tolerance follows dtype precision, so a correct cheap result is accepted."""

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    @pytest.mark.parametrize("dt", LOW, ids=LOW_IDS)
    def test_verification_accepts_a_correct_result(self, dt, axis):
        x = torch.linspace(-1, 1, 12, dtype=dt)
        P = sparsity(band, x)
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerificationInconclusive)
            J = jacobian(band, x, coloring=c)  # raises if the tolerance is too tight
        tol = 8 * float(torch.finfo(dt).eps)
        want = dense_jac(band, x).to(F64)
        torch.testing.assert_close(J.to_dense().to(F64), want,
                                   rtol=tol, atol=tol * float(want.abs().max()))

    @pytest.mark.parametrize("dt", LOW, ids=LOW_IDS)
    def test_softmax_is_accepted_too(self, dt):
        f = lambda z: torch.softmax(z.reshape(2, 6), dim=1).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerificationInconclusive)
            jacobian(f, torch.linspace(-1, 1, 12, dtype=dt))


class TestVerificationOutcomes:
    """Agreeing inside a rounding allowance the size of the answer proves nothing."""

    # bfloat16 leaves room of about 0.55 of the terms summed, float16 about 0.07.
    @pytest.mark.parametrize("dt,expect", [
        (F64, False), (torch.float32, False), (torch.float16, False),
        (torch.bfloat16, True),
    ], ids=["float64", "float32", "float16", "bfloat16"])
    def test_only_the_coarsest_dtype_is_inconclusive(self, dt, expect):
        x = torch.linspace(-1, 1, 12, dtype=dt)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jacobian(band, x)
        got = any(isinstance(m.message, VerificationInconclusive) for m in w)
        assert got is expect

    def test_a_non_finite_derivative_is_inconclusive_not_a_pass(self):
        # sqrt has an infinite derivative at zero. Infinities cannot be compared
        # against a rounding allowance, so this must not be reported as a pass.
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=F64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J = jacobian(torch.sqrt, x)
        assert any(isinstance(m.message, VerificationInconclusive) for m in w)
        assert J._nnz() == 4  # the result is still returned, just unchecked

    def test_a_non_finite_input_with_a_finite_derivative_still_checks(self):
        # A nan in the input is not the trigger. z * 2 has a constant derivative,
        # so the comparison runs and means something.
        x = torch.tensor([float("nan"), 1.0, 2.0, 3.0], dtype=F64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jacobian(lambda z: z * 2.0, x)
        assert not any(isinstance(m.message, VerificationInconclusive) for m in w)

    def test_an_inconclusive_check_still_reports_a_real_disagreement(self):
        # A stale coloring in bfloat16 is wrong by far more than the rounding
        # allowance, so it is caught rather than excused.
        class Slicer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start = 0

            def forward(self, z):
                return z[self.start:self.start + 2]

        m = Slicer()
        x = torch.linspace(-1, 1, 4, dtype=torch.bfloat16)
        c = cl.color_cols(sparsity(m, x))
        m.start = 1
        with pytest.raises(VerificationError):
            jacobian(m, x, coloring=c)
