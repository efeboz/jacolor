import numpy as np
import pytest
import torch

from src import coloring as cl
from src.analysis import Prepared, prepare
from src.evaluate import VerificationError
from src.trace import sparsity

F64 = torch.float64


def band(x):
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])


def dense_jac(f, x):
    return torch.func.jacrev(f)(x).reshape(-1, x.numel())


def randn(n, seed, dtype=F64):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=dtype)


class TestReuse:
    """Trace and color once, evaluate many times. That is the whole point."""

    def test_the_same_analysis_serves_many_points(self):
        p = prepare(band, randn(12, 0))
        for seed in range(4):
            x = randn(12, 100 + seed)
            y, J = p.value_and_jacobian(x)
            torch.testing.assert_close(y, band(x))
            torch.testing.assert_close(J.to_dense(), dense_jac(band, x), rtol=1e-12, atol=1e-14)

    def test_it_traces_only_once(self, monkeypatch):
        p = prepare(band, randn(12, 1))
        import src.analysis as an

        monkeypatch.setattr(an, "sparsity", lambda *a, **k: pytest.fail("traced again"))
        p.jacobian(randn(12, 2))

    def test_the_decompression_indices_are_built_once(self, monkeypatch):
        import src.analysis as an

        p = prepare(band, randn(12, 3))
        calls = []
        real = an._index
        monkeypatch.setattr(an, "_index", lambda *a: (calls.append(1), real(*a))[1])
        for _ in range(3):
            p.jacobian(randn(12, 4))
        assert len(calls) == 1


class TestCompatibility:
    """An analysis knows the input it was built for and says so."""

    @pytest.mark.parametrize("bad,why", [
        (lambda: torch.randn(13, dtype=F64), "shape"),
        (lambda: torch.randn(12, dtype=torch.float32), "dtype"),
        (lambda: torch.randn(3, 4, dtype=F64), "shape"),
    ])
    def test_a_different_input_is_refused(self, bad, why):
        p = prepare(band, randn(12, 5))
        with pytest.raises(ValueError, match="prepared for"):
            p.jacobian(bad())

    def test_a_stale_function_is_still_caught(self):
        class Slicer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start = 0

            def forward(self, z):
                return z[self.start:self.start + 2]

        m = Slicer()
        p = prepare(m, randn(4, 6))
        m.start = 1
        with pytest.raises(VerificationError):
            p.jacobian(randn(4, 6))


class TestModes:
    def test_forward_and_reverse(self):
        x = randn(12, 7)
        for mode, axis in [("forward", "cols"), ("reverse", "rows")]:
            p = prepare(band, x, mode=mode)
            assert p.mode == mode and p.coloring.axis == axis
            torch.testing.assert_close(p.jacobian(x).to_dense(), dense_jac(band, x),
                                       rtol=1e-12, atol=1e-14)

    def test_auto_takes_the_smaller_color_count(self):
        x = randn(12, 8)
        P = sparsity(band, x)
        fewer = min(cl.color_cols(P).n_colors, cl.color_rows(P).n_colors)
        assert prepare(band, x, mode="auto").n_colors == fewer

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode must be"):
            prepare(band, randn(12, 9), mode="sideways")


class TestSummary:
    def test_reports_the_shape_of_the_problem(self):
        p = prepare(band, randn(12, 10))
        s = p.summary()
        assert s["rows"] == 10 and s["cols"] == 12 and s["nnz"] == 30
        assert s["density"] == pytest.approx(30 / 120)
        assert s["colors"] == 3 and s["lower_bound"] == 3
        assert s["optimal"] is True
        assert s["passes_saved"] == 12 - 3
        assert s["mode"] == "forward"

    def test_status_records_the_last_verification(self):
        p = prepare(band, randn(12, 11))
        assert p.status is None
        p.jacobian(randn(12, 12))
        assert p.status == "ok"

    def test_status_says_skipped_when_verification_is_off(self):
        p = prepare(band, randn(12, 13), verify=False)
        p.jacobian(randn(12, 14))
        assert p.status == "skipped"

    def test_optimal_is_false_when_the_bound_is_not_reached(self):
        # A pattern whose greedy coloring needs more colors than its densest row.
        P = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
        p = prepare(lambda z: z, torch.zeros(3, dtype=F64), pattern=P)
        assert p.n_colors == 3 and p.lower_bound == 2 and p.optimal is False

    def test_repr_names_the_mode_and_the_counts(self):
        r = repr(prepare(band, randn(12, 15)))
        assert "forward" in r and "10x12" in r and "3 colors" in r


class TestSettingsCarryThrough:
    def test_chunk_is_used_for_every_evaluation(self, monkeypatch):
        import src.evaluate as ev

        widths = []
        real = ev._block
        monkeypatch.setattr(ev, "_block", lambda c, lo, hi, d, dev: (widths.append(hi - lo),
                                                                     real(c, lo, hi, d, dev))[1])
        p = prepare(band, randn(12, 16), chunk=1)
        p.jacobian(randn(12, 17))
        assert widths and max(widths) == 1

    def test_verify_off_is_honoured(self):
        class Slicer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start = 0

            def forward(self, z):
                return z[self.start:self.start + 2]

        m = Slicer()
        p = prepare(m, randn(4, 18), verify=False)
        m.start = 1
        p.jacobian(randn(4, 18))  # would raise with verification on

    @pytest.mark.parametrize("chunk", [0, -1, 2.5, "two"])
    def test_a_bad_chunk_is_refused_before_tracing(self, chunk, monkeypatch):
        import src.analysis as an

        monkeypatch.setattr(an, "sparsity", lambda *a: pytest.fail("traced"))
        with pytest.raises(ValueError, match="chunk"):
            prepare(band, randn(12, 19), mode="hybrid", chunk=chunk)

    @pytest.mark.parametrize("verify", ["yes", 1, torch.int32])
    def test_a_bad_verify_is_refused(self, verify):
        with pytest.raises(ValueError, match="verify"):
            prepare(band, randn(12, 19), verify=verify)


def test_no_module_shares_a_name_with_an_exported_function():
    # A module named prepare.py exporting prepare() rebinds the package
    # attribute to the function, so import src.prepare returns the function.
    # That has cost two debugging sessions, so it is a test now.
    import pathlib

    import src as jacolor

    import types

    modules = {f.stem for f in pathlib.Path(jacolor.__file__).parent.glob("*.py")}
    modules.discard("__init__")
    # Exporting a module under its own name is fine, as propagate does. The bug
    # is a function taking the name, which rebinds the package attribute.
    clash = sorted(n for n in modules & set(jacolor.__all__)
                   if not isinstance(getattr(jacolor, n), types.ModuleType))
    assert not clash, f"these modules are shadowed by a function: {clash}"


def test_prepared_is_exported():
    assert isinstance(prepare(band, randn(12, 19)), Prepared)


class TestPlanner:
    """auto decides from the pattern before paying to color anything."""

    def test_it_skips_a_direction_that_cannot_be_built(self):
        # A sum over 15000 inputs: column coloring would need a 2.25e8 entry
        # graph, row coloring needs one color.
        f = lambda z: z.sum().reshape(1)
        p = prepare(f, torch.randn(15000, dtype=F64))
        assert p.mode == "reverse" and p.n_colors == 1
        assert "over the budget" in p.reason

    def test_it_skips_a_direction_whose_bound_cannot_win(self, monkeypatch):
        import src.analysis as an

        monkeypatch.setattr(an, "color_rows", lambda *a, **k: pytest.fail("colored rows"))
        p = prepare(band, randn(12, 20))
        assert p.mode == "forward" and p.n_colors == 3
        assert "cannot beat that" in p.reason

    def test_it_colors_both_when_neither_bound_settles_it(self):
        from src import _boolcsr as bc

        # A three cycle: every line holds two entries, so both bounds are 2,
        # while a greedy coloring needs three either way.
        P = bc.from_dense(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool))
        p = prepare(lambda z: z, torch.zeros(3, dtype=F64), pattern=P)
        assert p.n_colors == 3 and p.lower_bound == 2
        assert "against" in p.reason

    def test_neither_direction_buildable_names_both(self):
        from src import _boolcsr as bc

        n, k = 15_000, 120  # 15000 lines of 120 entries, both ways
        rows = np.repeat(np.arange(n), k)
        cols = (np.arange(n)[:, None] + np.arange(k)).ravel() % n
        P = bc.from_pairs(rows, cols, (n, n))
        with pytest.raises(MemoryError, match="neither direction fits"):
            prepare(lambda z: z, torch.zeros(n, dtype=F64), pattern=P)

    def test_an_explicit_mode_says_so(self):
        for mode in ("forward", "reverse"):
            assert prepare(band, randn(12, 21), mode=mode).reason.startswith(mode)

    def test_the_reason_is_in_the_summary(self):
        assert "reason" in prepare(band, randn(12, 22)).summary()


def arrow(z):
    """Row 0 depends on every input, row i on z[0] and z[i].

    A global constraint plus a shared parameter: one dense row and one dense
    column over a diagonal, which is the shape neither direction compresses.
    """
    return torch.cat([z.sum().reshape(1), z[0] * z[1:]])


class TestHybrid:
    def test_it_beats_both_plain_directions(self):
        x = randn(32, 40)
        assert prepare(arrow, x, mode="forward").n_colors == 32
        assert prepare(arrow, x, mode="reverse").n_colors == 32
        p = prepare(arrow, x, mode="hybrid")
        assert p.mode == "hybrid" and p.n_colors == 3
        assert p.summary()["mode"] == "hybrid"

    def test_the_result_is_exact(self):
        x = randn(32, 41)
        p = prepare(arrow, x, mode="hybrid")
        y, J = p.value_and_jacobian(x)
        torch.testing.assert_close(y, arrow(x))
        torch.testing.assert_close(J.to_dense(), dense_jac(arrow, x),
                                   rtol=1e-12, atol=1e-14)
        assert J._nnz() == p.pattern.nnz

    def test_the_structure_is_reusable(self):
        p = prepare(arrow, randn(32, 42), mode="hybrid")
        g = torch.Generator().manual_seed(43)
        for _ in range(3):
            x = torch.randn(32, generator=g, dtype=F64)
            torch.testing.assert_close(p.jacobian(x).to_dense(), dense_jac(arrow, x),
                                       rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [16, 32, 64])
    def test_it_follows_the_coupling_not_the_size(self, n):
        assert prepare(arrow, randn(n, 44), mode="hybrid").n_colors == 3

    def test_the_bound_is_the_sum_of_the_halves(self):
        p = prepare(arrow, randn(32, 45), mode="hybrid")
        assert p.lower_bound == p.split.rest.lower_bound + p.split.dense.lower_bound
        assert p.optimal is True

    def test_it_declines_when_it_would_not_pay(self):
        p = prepare(band, randn(12, 46), mode="hybrid")
        assert p.mode == "forward" and p.n_colors == 3
        assert "split" in p.reason

    def test_it_declines_when_the_split_would_cost_more(self):
        # Rows of 4, 3, 3, 3 entries: a split is possible, but the remainder
        # still needs four colors and the dense row adds one on top, so plain
        # forward wins. This is the cost comparison, not the equal-rows case.
        from src import _boolcsr as bc

        P = bc.from_dense(np.array([[1, 1, 1, 1], [1, 1, 1, 0],
                                    [0, 1, 1, 1], [1, 0, 1, 1]], dtype=bool))
        p = prepare(lambda z: z, torch.zeros(4, dtype=F64), pattern=P, mode="hybrid")
        assert p.mode == "forward" and p.n_colors == 4
        assert "would need 5 directions against 4" in p.reason

    def test_a_pattern_that_does_not_fit_is_still_caught(self):
        # Reused for a function whose rows depend on the last input, not the first.
        p = prepare(arrow, randn(32, 47), mode="hybrid")
        p.f = lambda z: torch.cat([z.sum().reshape(1), z[31] * z[1:]])
        with pytest.raises(VerificationError):
            p.jacobian(randn(32, 47))

    def test_chunk_is_used_in_both_halves(self, monkeypatch):
        # Two dense rows, so each half has two colors to chunk.
        import src.analysis as an

        widths = []
        real = an._block
        monkeypatch.setattr(an, "_block", lambda c, lo, hi, *a, **k: (widths.append(hi - lo),
                                                                      real(c, lo, hi, *a, **k))[1])
        f = lambda z: torch.cat([z.sum().reshape(1), (z * z).sum().reshape(1), z[0] * z[1:]])
        x = randn(32, 54)
        p = prepare(f, x, mode="hybrid", chunk=1)
        J = p.jacobian(x)
        assert p.split.rest.n_colors == 2 and p.split.dense.n_colors == 2
        assert widths == [1, 1, 1, 1]
        torch.testing.assert_close(J.to_dense(), dense_jac(f, x), rtol=1e-12, atol=1e-14)

    def test_a_plain_direction_its_bound_rules_out_is_never_colored(self, monkeypatch):
        # Both plain bounds are 32 against a split of 3, so neither needs coloring.
        import src.analysis as an

        full = []
        real = an.color_cols
        monkeypatch.setattr(an, "color_cols", lambda P, *a, **k: (full.append(P.nnz),
                                                                  real(P, *a, **k))[1])
        p = prepare(arrow, randn(32, 55), mode="hybrid")
        assert p.mode == "hybrid" and p.pattern.nnz not in full

    def test_it_splits_where_neither_plain_direction_fits(self):
        # The arrow at 15000: either plain coloring would build a 2.25e8 entry
        # graph, the split needs three directions.
        from src import _boolcsr as bc

        n = 15_000
        i = np.arange(1, n)
        P = bc.from_pairs(np.r_[np.zeros(n, int), i, i], np.r_[np.arange(n), 0 * i, i], (n, n))
        x = randn(n, 56)
        p = prepare(arrow, x, pattern=P, mode="hybrid")
        assert p.mode == "hybrid" and p.n_colors == 3
        assert "would not fit in memory" in p.reason
        J = p.jacobian(x)
        assert p.status == "ok" and J._nnz() == P.nnz

    def test_a_split_over_budget_falls_back_to_a_plain_direction_that_fits(self):
        # Three sums over 15000 inputs. Forward and the split's forward half are
        # both over the budget, reverse needs three colors.
        f = lambda z: torch.stack([z.sum(), z[1:].sum(), z[:-1].sum()])
        p = prepare(f, torch.randn(15000, dtype=F64), mode="hybrid")
        assert p.mode == "reverse" and p.n_colors == 3
        assert "would not fit in the coloring budget either" in p.reason

    def test_a_declared_precision_carries_into_the_split(self):
        from src import _boolcsr as bc

        low = lambda z: arrow(z.float()).double()
        x = randn(32, 59)
        P = bc.from_dense(dense_jac(arrow, x) != 0)
        with pytest.raises(VerificationError):
            prepare(low, x, pattern=P, mode="hybrid").jacobian(x)
        p = prepare(low, x, pattern=P, mode="hybrid", verify=torch.float32)
        p.jacobian(x)
        assert p.mode == "hybrid" and p.status == "ok"

    def test_it_declines_when_plain_reverse_is_cheaper(self):
        # The split, 5 forward and 1 reverse, beats plain forward at 7. Plain
        # reverse needs 4, so the split must lose to that instead.
        P = np.array([[1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1],
                      [1, 0, 0, 0, 0, 1, 1]], dtype=bool)
        p = prepare(lambda z: z[:4], torch.zeros(7, dtype=F64), pattern=P, mode="hybrid")
        assert p.mode == "reverse" and p.n_colors == 4
        assert "would need 6 directions against 4 for plain reverse" in p.reason

    def test_forward_and_reverse_must_agree(self):
        # no_grad hides the dense row from reverse mode but not from forward.
        def f(z):
            with torch.no_grad():
                head = z.sum().reshape(1)
            return torch.cat([head, z[0] * z[1:]])

        from src import _boolcsr as bc
        from src.trace import TraceMismatch

        x = randn(8, 57)
        with pytest.raises(TraceMismatch):
            prepare(f, x, mode="hybrid")
        p = prepare(f, x, pattern=bc.from_dense(dense_jac(arrow, x) != 0), mode="hybrid")
        with pytest.raises(VerificationError):
            p.jacobian(x)

    def test_complex_is_refused(self):
        from src import _boolcsr as bc

        P = bc.from_dense(np.eye(4, dtype=bool))
        with pytest.raises(TypeError, match="complex input"):
            prepare(lambda z: z, torch.zeros(4, dtype=torch.complex128), pattern=P,
                    mode="hybrid")
        x = randn(8, 58)
        p = prepare(lambda z: arrow(z) * (1 + 2j), x, mode="hybrid",
                    pattern=bc.from_dense(dense_jac(arrow, x) != 0))
        assert p.mode == "hybrid"
        with pytest.raises(TypeError, match="complex output"):
            p.jacobian(x)

    def test_an_unknown_mode_still_names_the_choices(self):
        with pytest.raises(ValueError, match="auto, forward, reverse or hybrid"):
            prepare(band, randn(12, 48), mode="sideways")


class TestExplain:
    def test_it_names_what_forces_the_bound(self):
        text = prepare(arrow, randn(32, 49)).explain()
        assert "32 by 32" in text
        assert "widest row holds 32" in text
        assert "widest column holds 32" in text

    def test_it_points_at_the_way_out(self):
        # The pattern that plain coloring cannot help should say what would.
        text = prepare(arrow, randn(32, 50)).explain()
        assert "hybrid" in text and "widest row holds 2" in text

    def test_hybrid_explains_its_own_split(self):
        text = prepare(arrow, randn(32, 51), mode="hybrid").explain()
        assert "chose hybrid" in text and "2 forward and 1 reverse" in text

    def test_it_says_when_the_coloring_cannot_be_improved(self):
        # Only within its own kind: forward at its bound says nothing of reverse.
        text = prepare(band, randn(12, 52)).explain()
        assert "as few as forward coloring of this pattern allows" in text
        text = prepare(arrow, randn(32, 52), mode="hybrid").explain()
        assert "as few as this split allows" in text

    def test_it_repeats_the_reason_for_the_direction(self):
        p = prepare(band, randn(12, 53), mode="reverse")
        assert p.reason in p.explain()
