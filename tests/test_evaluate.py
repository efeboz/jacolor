import warnings

import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl
from src.evaluate import VerificationError, VerificationInconclusive, jacobian
from src.trace import TraceUnchecked, sparsity

F64 = torch.float64
CONV = torch.nn.functional.conv2d


def dense_jac(f, x):
    return torch.func.jacrev(f)(x).reshape(-1, x.numel())


# Dtypes whose own rounding is wider than a difference between two programs,
# so a trace in one cannot be checked and says so.
COARSE = (torch.float16, torch.bfloat16)


def traced(f, x):
    # The pattern, plus the fact that a coarse dtype leaves it unchecked.
    if x.dtype in COARSE:
        with pytest.warns(TraceUnchecked):
            return sparsity(f, x)
    return sparsity(f, x)


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

    def test_a_small_missing_entry_in_a_wide_row_is_caught(self):
        # The allowance follows the terms the line holds, not its width, or a
        # row of five thousand would excuse anything small.
        n = 5000
        f = lambda z: (z[0] + z[1] + 1e-12 * z[2]).reshape(1)
        x = torch.randn(n, generator=torch.Generator().manual_seed(17), dtype=F64)
        P = bc.from_pairs(np.zeros(2, int), np.array([0, 1]), (1, n))
        with pytest.raises(VerificationError, match=r"returns .* at entry \(0, 2\)"):
            jacobian(f, x, coloring=cl.color_cols(P))

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

    @pytest.mark.parametrize("bad", ["yes", 1, torch.int32])
    def test_verify_must_be_a_bool_or_a_floating_dtype(self, bad):
        x = torch.randn(12, generator=torch.Generator().manual_seed(13), dtype=F64)
        with pytest.raises(ValueError, match="verify must be"):
            jacobian(band, x, verify=bad)


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
        P = traced(band, x)
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
        x = torch.linspace(-1, 1, 12, dtype=dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerificationInconclusive)
            warnings.simplefilter("ignore", TraceUnchecked)
            jacobian(f, x)


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

    def test_sums_that_overflow_are_inconclusive_not_a_pass(self):
        # Entries of 1e307 are finite, but thirty of them summed are not. A stale
        # pattern then gets an infinite allowance, which must not read as a pass.
        from src.analysis import prepare

        x = torch.linspace(0.5, 1.0, 40, dtype=F64)
        p = prepare(lambda z: (1e307 * z[:30].sum()).reshape(1), x, mode="forward")
        p.f = lambda z: (1e307 * z[:31].sum()).reshape(1)
        with pytest.warns(VerificationInconclusive):
            p.jacobian(x)
        assert p.status == "inconclusive"

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
        c = cl.color_cols(traced(m, x))
        m.start = 1
        with pytest.raises(VerificationError):
            jacobian(m, x, coloring=c)


class TestCancellationCannotHideAWrongPattern:
    """Reconstructing exactly zero is not agreement, it is nothing to compare."""

    HEAD = staticmethod(lambda z: z[:128].sum().reshape(1))
    DIFF = staticmethod(lambda z: (z[128] - z[129]).reshape(1))

    def _stale(self, dtype, mode):
        # The pattern covers columns 0 to 127. The function it gets reused for
        # depends on 128 and 129, so every assembled value is zero.
        from src.analysis import prepare

        x = torch.randn(256, generator=torch.Generator().manual_seed(30), dtype=dtype)
        if dtype in COARSE:  # the trace cannot be checked there, which is its own test
            with pytest.warns(TraceUnchecked):
                p = prepare(self.HEAD, x, mode=mode)
        else:
            p = prepare(self.HEAD, x, mode=mode)
        p.f = self.DIFF
        return p, x

    def test_bfloat16_forward_is_inconclusive_not_ok(self):
        p, x = self._stale(torch.bfloat16, "forward")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J = p.jacobian(x)
        assert p.status == "inconclusive"
        assert any(isinstance(m.message, VerificationInconclusive) for m in w)
        assert int((J.to_dense() != 0).sum()) == 0  # and the answer really is wrong

    @pytest.mark.parametrize("mode", ["forward", "reverse"])
    def test_float64_catches_it_outright(self, mode):
        p, x = self._stale(F64, mode)
        with pytest.raises(VerificationError):
            p.jacobian(x)

    def test_bfloat16_reverse_still_catches_it(self):
        # Reverse sums one term per column, so the allowance stays small enough.
        p, x = self._stale(torch.bfloat16, "reverse")
        with pytest.raises(VerificationError):
            p.jacobian(x)


class TestMixedPrecisionTolerance:
    """The oracle's arithmetic ran at the input's precision, not the result's."""

    K = torch.ones(4, dtype=F64)

    @pytest.mark.parametrize("name,f", [
        ("z*z*k", lambda z: z * z * TestMixedPrecisionTolerance.K),
        ("z*k", lambda z: z * TestMixedPrecisionTolerance.K),
        ("tanh(z)*k", lambda z: torch.tanh(z) * TestMixedPrecisionTolerance.K),
    ], ids=["z*z*k", "z*k", "tanh"])
    def test_a_correct_float32_input_is_accepted(self, name, f):
        x = torch.randn(4, generator=torch.Generator().manual_seed(31), dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J = jacobian(f, x)
        # A tolerance that ignores the float32 arithmetic leaves this unchecked.
        assert not [m for m in w if isinstance(m.message, VerificationInconclusive)]
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(f)(x).to(F64),
                                   rtol=1e-6, atol=1e-6)

    # float64 in and out, float32 inside. Nothing at the boundary shows it.
    LOW = staticmethod(lambda z: (torch.tanh(z.float()) * 3.0).double())

    def test_hidden_float32_is_unchecked_and_says_what_to_pass(self):
        # Nothing is missing from the pattern, so this is not a wrong pattern.
        x = torch.linspace(0.5, 1.0, 6, dtype=F64)
        c = cl.color_cols(bc.from_dense(np.eye(6, dtype=bool)))
        with pytest.warns(VerificationInconclusive, match="pass that dtype as verify"):
            J = jacobian(self.LOW, x, coloring=c)
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(self.LOW)(x),
                                   rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_declaring_it_is_accepted(self, axis):
        x = torch.linspace(0.5, 1.0, 6, dtype=F64)
        P = bc.from_dense(np.eye(6, dtype=bool))
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J = jacobian(self.LOW, x, coloring=c, verify=torch.float32)
        # Declared, so the check resolves rather than giving up on it.
        assert not [m for m in w if isinstance(m.message, VerificationInconclusive)]
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(self.LOW)(x),
                                   rtol=1e-6, atol=1e-6)

    def test_declaring_it_still_catches_a_stale_pattern(self):
        x = torch.linspace(0.5, 1.0, 6, dtype=F64)
        c = cl.color_cols(bc.from_dense(np.roll(np.eye(6, dtype=bool), 1, axis=1)))
        with pytest.raises(VerificationError):
            jacobian(self.LOW, x, coloring=c, verify=torch.float32)

    def test_an_output_coarser_than_the_input_is_accepted(self):
        # float64 in, float32 out. The entries are float32 and the allowance has
        # to follow them, not the input.
        f = lambda z: (torch.tanh(z) * 3.0).float()
        x = torch.linspace(0.5, 1.0, 6, dtype=F64)
        c = cl.color_cols(bc.from_dense(np.eye(6, dtype=bool)))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            J = jacobian(f, x, coloring=c)
        assert not [m for m in w if isinstance(m.message, VerificationInconclusive)]
        torch.testing.assert_close(J.to_dense().to(F64), torch.func.jacrev(f)(x),
                                   rtol=1e-6, atol=1e-7)

    def test_a_stale_float32_pattern_is_still_caught(self):
        # The wider allowance must not blind the check to a real disagreement.
        from src.analysis import prepare

        class Slicer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start = 0

            def forward(self, z):
                return z[self.start:self.start + 2]

        m = Slicer()
        x = torch.randn(4, generator=torch.Generator().manual_seed(32), dtype=torch.float32)
        p = prepare(m, x, mode="forward")
        m.start = 1
        with pytest.raises(VerificationError):
            p.jacobian(x)


class TestZeroColorOutputSize:
    """No block runs, so the checks inside the loop never do either."""

    class Const(torch.nn.Module):
        def __init__(self, n=1):
            super().__init__()
            self.n = n

        def forward(self, z):
            return torch.zeros(self.n, dtype=F64)

    def test_an_output_that_grows_is_caught(self):
        from src.analysis import prepare

        m = self.Const(1)
        x = torch.zeros(0, dtype=F64)
        p = prepare(m, x)
        m.n = 3
        with pytest.raises(ValueError, match="output of 1 elements, got 3"):
            p.value_and_jacobian(x)

    def test_an_unchanged_constant_still_works(self):
        from src.analysis import prepare

        x = torch.zeros(0, dtype=F64)
        p = prepare(self.Const(1), x)
        y, J = p.value_and_jacobian(x)
        assert tuple(y.shape) == (1,) and tuple(J.shape) == (1, 0) and p.status == "ok"


class TestComplexIsRefused:
    """Forward mode gives the holomorphic derivative, reverse its conjugate."""

    P = np.eye(4, dtype=bool)

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_complex_input(self, axis):
        P = bc.from_dense(self.P)
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with pytest.raises(TypeError, match="complex input"):
            jacobian(lambda z: z * 2, torch.ones(4, dtype=torch.complex128), coloring=c)

    @pytest.mark.parametrize("axis", ["cols", "rows"])
    def test_complex_output(self, axis):
        P = bc.from_dense(self.P)
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with pytest.raises(TypeError, match="complex output"):
            jacobian(lambda z: z * (1 + 2j), torch.ones(4, dtype=F64), coloring=c)


class TestSaturationIsNotAWrongPattern:
    """A softmax at saturated logits computes its own derivative through
    cancellation, so the entries and the oracle disagree by far more than their
    magnitudes explain. The pattern is right, and saying it is wrong would be."""

    SM = staticmethod(lambda z: z.softmax(0))
    SPREAD = [10.0, 20.0, 30.0, 40.0]

    def x(self, k, dt):
        return torch.tensor([k, 0.0, 0.0, 0.0], dtype=dt)

    @pytest.mark.parametrize("dt", [F64, torch.float32], ids=["float64", "float32"])
    @pytest.mark.parametrize("axis", ["cols", "rows"])
    @pytest.mark.parametrize("k", SPREAD, ids=[f"x0={k:g}" for k in SPREAD])
    def test_a_correct_pattern_is_never_refused(self, k, axis, dt):
        x = self.x(k, dt)
        P = sparsity(self.SM, x)
        c = cl.color_cols(P) if axis == "cols" else cl.color_rows(P)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerificationInconclusive)
            J = jacobian(self.SM, x, coloring=c)
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(self.SM)(x))

    @pytest.mark.parametrize("k", SPREAD, ids=[f"x0={k:g}" for k in SPREAD])
    def test_the_unresolved_check_says_so_rather_than_passing(self, k):
        x = self.x(k, F64)
        c = cl.color_cols(sparsity(self.SM, x))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jacobian(self.SM, x, coloring=c)
        said = [m for m in w if isinstance(m.message, VerificationInconclusive)]
        assert said, "a disagreement beyond rounding must not read as a pass"
        assert "holds every entry of that line" in str(said[0].message)

    @pytest.mark.parametrize("k", SPREAD, ids=[f"x0={k:g}" for k in SPREAD])
    def test_a_missing_entry_is_still_caught_there(self, k):
        # The same saturation, with one column dropped from the pattern.
        x = self.x(k, F64)
        P = bc.from_dense(np.array([[1, 1, 1, 0]] * 4, dtype=bool))
        with pytest.raises(VerificationError, match=r"entry \(\d+, 3\)"):
            jacobian(self.SM, x, coloring=cl.color_cols(P))

    def test_the_error_names_what_is_missing(self):
        x = torch.linspace(-1, 1, 4, dtype=F64)
        P = bc.from_dense(np.array([[1, 1, 1, 0]] * 4, dtype=bool))
        with pytest.raises(VerificationError) as e:
            jacobian(self.SM, x, coloring=cl.color_cols(P))
        msg = str(e.value)
        assert "the pattern does not hold" in msg and "stale" in msg


class TestTheSearchStaysInItsMode:
    """Evidence about the entries has to come from the derivative the entries
    are. The other mode may not exist for f, and where both exist they can
    describe different derivatives."""

    GRID = torch.randn(1, 4, 4, 2, generator=torch.Generator().manual_seed(1), dtype=F64)

    def grid_sample(self, z):
        y = torch.nn.functional.grid_sample(z.reshape(1, 1, 4, 4), self.GRID,
                                            align_corners=True)
        return y.reshape(-1).softmax(0)

    def reverse_only(self):
        x = torch.randn(16, generator=torch.Generator().manual_seed(2), dtype=F64)
        try:
            torch.func.jvp(self.grid_sample, (x,), (torch.ones(16, dtype=F64),))
        except NotImplementedError:
            return x, torch.func.jacrev(self.grid_sample)(x)
        pytest.skip("this torch has forward AD for grid_sample")

    def test_a_reverse_only_operator_is_checked_without_it(self):
        x, truth = self.reverse_only()
        c = cl.color_rows(bc.from_dense(truth != 0))
        J = jacobian(self.grid_sample, x, coloring=c)  # raised NotImplementedError
        torch.testing.assert_close(J.to_dense(), truth)

    def test_a_reverse_only_operator_is_still_diagnosed(self):
        x, truth = self.reverse_only()
        c = cl.color_rows(bc.from_dense(torch.roll(truth, 1, 1) != 0))
        with pytest.raises(VerificationError, match="at entry"):
            jacobian(self.grid_sample, x, coloring=c)

    def test_a_mode_that_sees_more_does_not_invent_missing_entries(self):
        # no_grad hides the offset from reverse mode, not from forward. The
        # reverse Jacobian is block diagonal and right, and nothing is missing.
        def blocky(z):
            with torch.no_grad():
                off = 0.25 * z.sum()
            return z.reshape(2, 3).softmax(1).reshape(-1) + off

        x = torch.randn(6, generator=torch.Generator().manual_seed(3), dtype=F64)
        P = bc.from_dense(torch.func.jacrev(blocky)(x) != 0)
        assert P.nnz == 18  # block diagonal, not the 36 forward mode would give
        J = jacobian(blocky, x, coloring=cl.color_rows(P))
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(blocky)(x))

    @pytest.mark.parametrize("dt", [torch.bfloat16, F64], ids=["bfloat16", "float64"])
    def test_omitted_entries_that_cancel_in_a_sum_are_still_found(self, dt):
        # Three columns of one, a pattern holding only the first. Summed, the
        # two it leaves out can cancel, so the search reads them one at a time.
        n = 128
        f = lambda z: (z[0] + z[75] + z[125]).reshape(1)
        x = torch.randn(n, generator=torch.Generator().manual_seed(4), dtype=dt)
        P = bc.from_pairs(np.zeros(1, int), np.zeros(1, int), (1, n))
        with pytest.raises(VerificationError, match=r"at entry \(0, (75|125)\)"):
            jacobian(f, x, coloring=cl.color_cols(P))

    def counted(self, monkeypatch):
        import src.evaluate as ev

        spent = []
        real = ev._line
        monkeypatch.setattr(ev, "_line", lambda *a: (spent.append(1), real(*a))[1])
        return spent

    def test_a_search_that_runs_out_says_so_rather_than_concluding(self, monkeypatch):
        import src.evaluate as ev

        spent = self.counted(monkeypatch)
        monkeypatch.setattr(ev, "_PROBE_PASSES", 2)
        n = 128
        f = lambda z: (z[0] + z[75] + z[125]).reshape(1)
        x = torch.randn(n, generator=torch.Generator().manual_seed(4), dtype=F64)
        P = bc.from_pairs(np.zeros(1, int), np.zeros(1, int), (1, n))
        with pytest.warns(VerificationInconclusive, match="ran out of passes"):
            jacobian(f, x, coloring=cl.color_cols(P))
        assert len(spent) <= 2, "the budget is what stops it, not the line"

    def test_a_probe_that_overflows_resolves_nothing(self):
        # The line is finite, the probe of what it leaves out is not. An entry
        # read as an infinity has not been read, so it cannot name one either.
        def f(z):
            t = (z[1] - z[2]) * 1e308
            return (z[0] + 2 * t - t).reshape(1)

        x = torch.zeros(3, dtype=F64)
        P = bc.from_pairs(np.zeros(1, int), np.zeros(1, int), (1, 3))
        with pytest.warns(VerificationInconclusive, match="overflowed"):
            jacobian(f, x, coloring=cl.color_cols(P))

    def test_a_group_that_overflows_stops_the_search(self):
        # Every entry it leaves out is finite, while a group of them summed is
        # not. Reading the group settles nothing, so it cannot go on picking a
        # half from it, and naming the entry it landed on would be an accident.
        from src.analysis import prepare

        f = lambda z: (z[0] + (z[1] - z[2]) * 1.5e308).reshape(1)
        x = torch.zeros(6, dtype=F64)
        P = np.zeros((1, 6), dtype=bool)
        P[0, 0] = True
        p = prepare(f, x, mode="forward", pattern=P)
        with pytest.warns(VerificationInconclusive, match="overflowed"):
            p.jacobian(x)
        assert p.status == "inconclusive"

    def test_a_line_with_nothing_outside_it_stops_after_one_split(self, monkeypatch):
        # Every entry the pattern leaves out is zero, so there is nothing to
        # find and no reason to spend the budget looking.
        spent = self.counted(monkeypatch)
        f = lambda z: (torch.tanh(z.float()) * 3.0).double()
        x = torch.linspace(0.5, 1.0, 6, dtype=F64)
        c = cl.color_cols(bc.from_dense(np.eye(6, dtype=bool)))
        with pytest.warns(VerificationInconclusive):
            jacobian(f, x, coloring=c)
        assert len(spent) <= 2

    def test_an_omitted_entry_that_is_zero_is_not_blamed(self):
        # The line leaves out one entry and that entry is genuinely zero, so the
        # disagreement, which is f's float32 arithmetic, is not the pattern.
        f = lambda z: (torch.tanh(z.float()) * 3.0).double()
        x = torch.linspace(0.5, 1.0, 2, dtype=F64)
        c = cl.color_cols(bc.from_dense(np.eye(2, dtype=bool)))
        with pytest.warns(VerificationInconclusive, match="nothing it leaves out"):
            J = jacobian(f, x, coloring=c)
        torch.testing.assert_close(J.to_dense(), torch.func.jacrev(f)(x),
                                   rtol=1e-6, atol=1e-7)
