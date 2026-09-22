import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src import _boolcsr as bc
from src.evaluate import VerificationInconclusive, jacobian
from src.trace import TraceUnchecked
from src.interop import pattern, pattern_from_pairs, to_scipy
from src.analysis import prepare

F64 = torch.float64


def band(x):
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])


class TestPatternInput:
    """A caller who knows the structure should not need the internals."""

    WANT = [[True, False], [False, True]]

    def test_from_a_dense_array(self):
        assert pattern(np.array([[1.0, 0.0], [0.0, 2.0]])).toarray().tolist() == self.WANT

    def test_from_a_torch_tensor(self):
        assert pattern(torch.tensor([[1.0, 0.0], [0.0, 2.0]])).toarray().tolist() == self.WANT

    def test_from_a_scipy_matrix(self):
        assert pattern(sp.csr_matrix(np.eye(2))).toarray().tolist() == self.WANT

    def test_from_a_jacolor_pattern(self):
        assert pattern(bc.eye(2)).toarray().tolist() == self.WANT

    def test_from_pairs(self):
        P = pattern_from_pairs([0, 1, 0], [1, 0, 1], (2, 2))  # a repeat is fine
        assert P.toarray().tolist() == [[False, True], [True, False]]

    def test_nonzero_is_what_counts_not_true(self):
        # A value of -3 is an entry. Only an exact zero is absent.
        assert pattern(np.array([[-3.0, 0.0]])).toarray().tolist() == [[True, False]]

    def test_a_given_pattern_skips_tracing(self, monkeypatch):
        import src.analysis as pp

        monkeypatch.setattr(pp, "sparsity", lambda *a, **k: pytest.fail("traced"))
        x = torch.randn(12, generator=torch.Generator().manual_seed(0), dtype=F64)
        dense = (torch.func.jacrev(band)(x) != 0).numpy()
        p = pp.prepare(band, x, pattern=dense)
        torch.testing.assert_close(p.jacobian(x).to_dense(), torch.func.jacrev(band)(x))

    def test_a_pattern_of_the_wrong_width_is_refused(self):
        x = torch.randn(12, generator=torch.Generator().manual_seed(1), dtype=F64)
        with pytest.raises(ValueError, match="10 columns, x has 12"):
            prepare(band, x, pattern=np.ones((10, 10)))


class TestToScipy:
    def test_round_trips_the_values(self):
        x = torch.randn(12, generator=torch.Generator().manual_seed(2), dtype=F64)
        J = jacobian(band, x)
        S = to_scipy(J)
        assert isinstance(S, sp.csr_array) and S.shape == (10, 12)
        np.testing.assert_allclose(S.toarray(), J.to_dense().numpy())

    def test_keeps_only_the_pattern_entries(self):
        x = torch.randn(12, generator=torch.Generator().manual_seed(3), dtype=F64)
        assert to_scipy(jacobian(band, x)).nnz == 30

    def test_detaches_and_lands_on_the_cpu(self):
        x = torch.randn(12, generator=torch.Generator().manual_seed(4), dtype=F64)
        S = to_scipy(jacobian(band, x))
        assert isinstance(S.data, np.ndarray)

    def test_a_jacobian_that_requires_grad_converts(self):
        # A module's parameters require grad, so jacrev and jacolor both return
        # tensors carrying history. scipy holds plain arrays, so it has to detach.
        torch.manual_seed(0)
        lin = torch.nn.Linear(4, 3, dtype=F64)
        x = torch.randn(4, generator=torch.Generator().manual_seed(5), dtype=F64)
        J = jacobian(lin, x)
        S = to_scipy(J)
        want = torch.func.jacrev(lin)(x).reshape(3, 4).detach().numpy()
        np.testing.assert_allclose(S.toarray(), want)

    def test_a_dense_tensor_is_refused(self):
        with pytest.raises(TypeError, match="expected a sparse Jacobian"):
            to_scipy(torch.zeros(2, 2))

    def test_an_uncoalesced_tensor_is_handled(self):
        i = torch.tensor([[0, 0], [0, 0]])
        J = torch.sparse_coo_tensor(i, torch.tensor([1.0, 2.0]), (1, 1))
        assert to_scipy(J).toarray().tolist() == [[3.0]]


HALF = [torch.float16, torch.bfloat16]
HALF_IDS = [str(d).split(".")[-1] for d in HALF]


class TestLowPrecision:
    """numpy has no bfloat16 and scipy has no half precision at all."""

    @pytest.mark.parametrize("dt", HALF, ids=HALF_IDS)
    def test_pattern_from_a_half_precision_tensor(self, dt):
        t = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=dt)
        assert pattern(t).toarray().tolist() == [[True, False], [False, True]]

    @pytest.mark.parametrize("dt", HALF, ids=HALF_IDS)
    def test_to_scipy_promotes_half_precision(self, dt):
        x = torch.linspace(-1, 1, 12, dtype=dt)
        # Half precision cannot check the trace it came from, and bfloat16
        # cannot resolve the result check either. Both are the library saying
        # so, named here rather than filtered away.
        with pytest.warns((TraceUnchecked, VerificationInconclusive)):
            J = jacobian(band, x)
        S = to_scipy(J)
        assert S.dtype == np.float32 and S.nnz == 30
        # Promotion widens the type without moving a value.
        np.testing.assert_allclose(S.toarray(), J.to_dense().to(torch.float32).numpy())

    @pytest.mark.parametrize("dt,want", [(torch.float32, np.float32), (F64, np.float64)])
    def test_full_precision_is_left_alone(self, dt, want):
        x = torch.linspace(-1, 1, 12, dtype=dt)
        assert to_scipy(jacobian(band, x)).dtype == want
