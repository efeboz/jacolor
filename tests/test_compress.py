import numpy as np
import pytest
import scipy.sparse as sp
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
        got = decompress(J @ seeds(c, dtype=F64), c)
        assert torch.equal(got.to_dense(), J)

    @pytest.mark.parametrize("seed", range(20))
    def test_reverse(self, seed):
        rng = np.random.default_rng(seed)
        J = torch.tensor(rng.normal(size=(9, 7)) * (rng.random((9, 7)) < 0.25), dtype=F64)
        P = pattern_of(J)
        c = cl.color_rows(P)
        got = decompress(seeds(c, dtype=F64) @ J, c)
        assert torch.equal(got.to_dense(), J)

    def test_overapproximate_pattern_is_still_exact(self):
        # A conservative pattern costs colors, never correctness.
        rng = np.random.default_rng(0)
        J = torch.tensor(rng.normal(size=(6, 8)) * (rng.random((6, 8)) < 0.3), dtype=F64)
        dense_P = bc.from_dense(np.ones((6, 8), dtype=bool))
        c = cl.color_cols(dense_P)
        assert c.n_colors == 8  # no grouping possible
        got = decompress(J @ seeds(c, dtype=F64), c)
        assert torch.equal(got.to_dense(), J)


class TestAgainstAutograd:
    """The whole point: fewer AD passes must give the same Jacobian."""

    def test_forward_matches_jacrev(self):
        x = torch.randn(12, dtype=F64)
        J = torch.func.jacrev(band)(x)
        P = pattern_of(J)
        c = cl.color_cols(P)
        assert c.n_colors == 3 < 12  # banded: 3 passes instead of 12
        got = decompress(fwd_compress(band, x, seeds(c, dtype=F64)), c)
        torch.testing.assert_close(got.to_dense(), J, rtol=1e-12, atol=1e-14)

    def test_reverse_matches_jacrev(self):
        x = torch.randn(12, dtype=F64)
        J = torch.func.jacrev(band)(x)
        P = pattern_of(J)
        c = cl.color_rows(P)
        assert c.n_colors == 3  # banded: 3 passes instead of 10 rows
        got = decompress(rev_compress(band, x, seeds(c, dtype=F64)), c)
        torch.testing.assert_close(got.to_dense(), J, rtol=1e-12, atol=1e-14)


class TestRejects:
    def test_wrong_compressed_shape(self):
        c = cl.color_cols(bc.from_dense([[1, 1, 0], [0, 1, 1]]))
        with pytest.raises(ValueError, match="should have shape"):
            decompress(torch.zeros(5, 5, dtype=F64), c)


def stored_false_identity():
    # Identity, but (0, 1) is explicitly stored as False. Valid bool CSR.
    return sp.csr_array(
        (np.array([True, False, True]), np.array([0, 1, 1]), np.array([0, 2, 3])),
        shape=(2, 2),
    )


class TestStoredFalse:
    """Reproduced by review: decompressed [[2, 2], [0, 3]] instead of [[2, 0], [0, 3]]."""

    J = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=F64)

    @staticmethod
    def f(x):
        return torch.stack([2 * x[0], 3 * x[1]])

    def test_forward(self):
        c = cl.color_cols(stored_false_identity())
        got = decompress(fwd_compress(self.f, torch.ones(2, dtype=F64), seeds(c, dtype=F64)), c)
        assert torch.equal(got.to_dense(), self.J)

    def test_reverse(self):
        c = cl.color_rows(stored_false_identity())
        got = decompress(rev_compress(self.f, torch.ones(2, dtype=F64), seeds(c, dtype=F64)), c)
        assert torch.equal(got.to_dense(), self.J)


class TestColoringFitsItsPattern:
    """Reproduced by review: an identity coloring on a triangular pattern gave
    [[6, 6], [0, 3]]. The pattern now travels with the coloring."""

    def test_triangular_pattern_decompresses_correctly(self):
        T = bc.from_dense([[1, 1], [0, 1]])
        g = lambda x: torch.stack([2 * x[0] + 4 * x[1], 3 * x[1]])
        c = cl.color_cols(T)
        assert c.n_colors == 2
        got = decompress(fwd_compress(g, torch.ones(2, dtype=F64), seeds(c, dtype=F64)), c)
        assert torch.equal(got.to_dense(), torch.tensor([[2.0, 4.0], [0.0, 3.0]], dtype=F64))

    def test_conflicting_column_coloring_is_rejected(self):
        T = bc.from_dense([[1, 1], [0, 1]])  # row 0 holds both columns
        with pytest.raises(ValueError, match="two cols sharing a row"):
            cl.Coloring(np.array([0, 0]), 1, 2, "natural", "cols", T)

    def test_conflicting_row_coloring_is_rejected(self):
        T = bc.from_dense([[1, 0], [1, 1]])  # column 0 holds both rows
        with pytest.raises(ValueError, match="two rows sharing a column"):
            cl.Coloring(np.array([0, 0]), 1, 2, "natural", "rows", T)

    def test_wrong_number_of_colors(self):
        with pytest.raises(ValueError, match="2 colors for 3 cols"):
            cl.Coloring(np.array([0, 1]), 2, 1, "natural", "cols", bc.eye(3))

    def test_bad_axis(self):
        with pytest.raises(ValueError, match="axis must be"):
            cl.Coloring(np.array([0]), 1, 1, "natural", "diag", bc.eye(1))

    @pytest.mark.parametrize("labels", [np.array([0, 2]), np.array([-1, 0])])
    def test_labels_outside_the_range_are_rejected(self, labels):
        # A line whose color is never seeded is never written, so its entries
        # would keep whatever the result buffer happened to hold.
        with pytest.raises(ValueError, match="colors must lie in"):
            cl.Coloring(labels, 2, 1, "natural", "cols", bc.eye(2))

    def test_float_labels_are_rejected(self):
        with pytest.raises(ValueError, match="colors must be integers"):
            cl.Coloring(np.array([0.0, 1.0]), 2, 1, "natural", "cols", bc.eye(2))

    def test_two_dimensional_labels_are_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            cl.Coloring(np.zeros((2, 1), dtype=np.int64), 1, 1, "natural", "cols", bc.eye(2))

    def test_the_stored_labels_are_a_copy(self):
        labels = np.array([0, 1])
        c = cl.Coloring(labels, 2, 1, "natural", "cols", bc.eye(2))
        labels[:] = 0
        assert c.colors.tolist() == [0, 1]

    def test_the_stored_labels_cannot_be_edited(self):
        # Validation happens once, so the validated state has to stay put.
        c = cl.Coloring(np.array([0, 1]), 2, 1, "natural", "cols", bc.eye(2))
        with pytest.raises(ValueError, match="read-only"):
            c.colors[0] = 1

    def test_pattern_is_copied(self):
        # Editing the caller's matrix afterwards must not move the pattern the
        # coloring decompresses onto.
        P = bc.from_dense([[1, 0], [0, 1]])
        c = cl.color_cols(P)
        P.data[:] = False
        # Not the stored count, which scipy keeps for entries it stores as False.
        assert c.pattern.toarray().tolist() == [[True, False], [False, True]]
