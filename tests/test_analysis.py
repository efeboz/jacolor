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
