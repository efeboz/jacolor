import numpy as np
import pytest
import torch

from src import _boolcsr as bc, coloring as cl
from src.compress import decompress, seeds

torch.manual_seed(0)
F64 = torch.float64


def pattern_of(J):
    # Exact pattern of a dense Jacobian.
    return bc.from_dense(J.detach().numpy() != 0)


def fwd_compress(f, x, S):
    # One jvp per color. Column k of the result is J @ S[:, k].
    return torch.stack(
        [torch.func.jvp(f, (x,), (S[:, k],))[1] for k in range(S.shape[1])], dim=1
    )


def rev_compress(f, x, S):
    # One vjp per color. Row k of the result is S[k] @ J.
    _, vjp = torch.func.vjp(f, x)
    return torch.stack([vjp(S[k])[0] for k in range(S.shape[0])], dim=0)


# Each output touches three consecutive inputs, so the Jacobian is banded.
def band(x):
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])


class TestSeeds:
    def test_forward_shape_and_content(self):
        P = bc.from_dense([[1, 1, 0], [0, 1, 1]])
        c = cl.color_cols(P)  # columns 0 and 2 share a color
        S = seeds(c, dtype=F64)
        assert S.shape == (3, c.n_colors)
        assert torch.equal(S.sum(dim=1), torch.ones(3, dtype=F64))  # one color each
        assert S[0, c.colors[0]] == 1 and S[2, c.colors[2]] == 1

    def test_reverse_is_transposed(self):
        P = bc.from_dense([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
        c = cl.color_rows(P)
        S = seeds(c, dtype=F64)
        assert S.shape == (c.n_colors, 3)
        assert torch.equal(S.sum(dim=0), torch.ones(3, dtype=F64))


class TestRoundTrip:
    """Compression is a matmul. Decompression must invert it exactly."""

    @pytest.mark.parametrize("seed", range(20))
    def test_forward(self, seed):
        rng = np.random.default_rng(seed)
        J = torch.tensor(rng.normal(size=(7, 9)) * (rng.random((7, 9)) < 0.25), dtype=F64)
        P = pattern_of(J)
        c = cl.color_cols(P)
        got = decompress(J @ seeds(c, dtype=F64), P, c)
        assert torch.equal(got.to_dense(), J)

    @pytest.mark.parametrize("seed", range(20))
    def test_reverse(self, seed):
        rng = np.random.default_rng(seed)
        J = torch.tensor(rng.normal(size=(9, 7)) * (rng.random((9, 7)) < 0.25), dtype=F64)
        P = pattern_of(J)
        c = cl.color_rows(P)
        got = decompress(seeds(c, dtype=F64) @ J, P, c)
        assert torch.equal(got.to_dense(), J)

    def test_overapproximate_pattern_is_still_exact(self):
        # A conservative pattern costs colors, never correctness.
        rng = np.random.default_rng(0)
        J = torch.tensor(rng.normal(size=(6, 8)) * (rng.random((6, 8)) < 0.3), dtype=F64)
        dense_P = bc.from_dense(np.ones((6, 8), dtype=bool))
        c = cl.color_cols(dense_P)
        assert c.n_colors == 8  # no grouping possible
        got = decompress(J @ seeds(c, dtype=F64), dense_P, c)
        assert torch.equal(got.to_dense(), J)


class TestAgainstAutograd:
    """The whole point: fewer AD passes must give the same Jacobian."""

    def test_forward_matches_jacrev(self):
        x = torch.randn(12, dtype=F64)
        J = torch.func.jacrev(band)(x)
        P = pattern_of(J)
        c = cl.color_cols(P)
        assert c.n_colors == 3 < 12  # banded: 3 passes instead of 12
        got = decompress(fwd_compress(band, x, seeds(c, dtype=F64)), P, c)
        torch.testing.assert_close(got.to_dense(), J, rtol=1e-12, atol=1e-14)

    def test_reverse_matches_jacrev(self):
        x = torch.randn(12, dtype=F64)
        J = torch.func.jacrev(band)(x)
        P = pattern_of(J)
        c = cl.color_rows(P)
        assert c.n_colors == 3  # banded: 3 passes instead of 10 rows
        got = decompress(rev_compress(band, x, seeds(c, dtype=F64)), P, c)
        torch.testing.assert_close(got.to_dense(), J, rtol=1e-12, atol=1e-14)


class TestRejects:
    def test_wrong_axis_length(self):
        P = bc.from_dense([[1, 1, 0], [0, 1, 1]])
        c = cl.color_cols(bc.from_dense([[1, 1]]))  # 2 columns, pattern has 3
        with pytest.raises(ValueError, match="coloring covers 2 cols, pattern has 3"):
            decompress(torch.zeros(2, c.n_colors, dtype=F64), P, c)

    def test_wrong_compressed_shape(self):
        P = bc.from_dense([[1, 1, 0], [0, 1, 1]])
        c = cl.color_cols(P)
        with pytest.raises(ValueError, match="should have shape"):
            decompress(torch.zeros(5, 5, dtype=F64), P, c)
