"""Prepare once, evaluate many times.

This module is named analysis rather than prepare so that the module and the
function it exports do not shadow each other on the package.

Tracing and coloring are the expensive part and they depend only on the shape of
the problem, not on the point. A prepared analysis holds the pattern, the
coloring and the decompression indices, so a solver loop pays for them once.

What it cannot do is notice that f has changed underneath it. It records the
signature of the input it was built for and checks that, and every evaluation is
still verified against autograd unless that is turned off.
"""

import torch

from .coloring import color_cols, color_rows
from .compress import _index
from .evaluate import _assemble
from .interop import pattern as as_pattern
from .trace import sparsity

__all__ = ["prepare", "Prepared"]


class Prepared:
    """A pattern, a coloring and the indices to scatter results back onto it."""

    __slots__ = ("f", "coloring", "chunk", "verify", "status", "_index", "_sig")

    def __init__(self, f, coloring, sig, chunk=None, verify=True):
        self.f = f
        self.coloring = coloring
        self.chunk = chunk
        self.verify = verify
        self.status = None  # outcome of the most recent verification
        self._sig = sig  # shape, dtype and device this was prepared for
        self._index = None  # built on first use, then reused

    # --- what was found -----------------------------------------------------

    @property
    def pattern(self):
        return self.coloring.pattern

    @property
    def mode(self):
        return "forward" if self.coloring.axis == "cols" else "reverse"

    @property
    def n_colors(self):
        return self.coloring.n_colors

    @property
    def lower_bound(self):
        return self.coloring.lower_bound

    @property
    def optimal(self):
        """True when the coloring provably cannot be improved for this pattern.

        The bound is the densest line, which any coloring must spend a color on.
        Reaching it settles the question. Falling short of it does not mean the
        heuristic did badly, because the bound itself can be loose.
        """
        return self.n_colors == self.lower_bound

    def summary(self):
        m, n = self.pattern.shape
        return {
            "mode": self.mode,
            "rows": m,
            "cols": n,
            "nnz": int(self.pattern.nnz),
            "density": self.pattern.nnz / (m * n) if m and n else 0.0,
            "colors": self.n_colors,
            "lower_bound": self.lower_bound,
            "optimal": self.optimal,
            "passes_saved": (n if self.mode == "forward" else m) - self.n_colors,
            "chunk": self.chunk,
            "verify": self.verify,
            "status": self.status,
        }

    def __repr__(self):
        s = self.summary()
        return (f"Prepared({s['mode']}, {s['rows']}x{s['cols']}, nnz {s['nnz']}, "
                f"{s['colors']} colors, lower bound {s['lower_bound']})")

    # --- using it -----------------------------------------------------------

    def _check(self, x):
        shape, dtype, device = self._sig
        got = (tuple(x.shape), x.dtype, x.device)
        if got != (shape, dtype, device):
            raise ValueError(
                f"this analysis was prepared for {shape} {dtype} on {device}, got "
                f"{got[0]} {got[1]} on {got[2]}. Prepare again for the new input."
            )
        if self._index is None:
            self._index = _index(self.pattern, self.coloring, x.device)

    def jacobian(self, x):
        """The sparse Jacobian of f at x."""
        return self.value_and_jacobian(x)[1]

    def value_and_jacobian(self, x):
        """f at x and its sparse Jacobian, which a nonlinear solver wants together."""
        self._check(x)
        J, y, self.status = _assemble(
            self.f, x, self.coloring, self.chunk, self.verify, self._index
        )
        return (self.f(x) if y is None else y), J


def prepare(f, x, mode="auto", pattern=None, chunk=None, verify=True):
    """Trace and color f once, for repeated evaluation at inputs shaped like x.

    mode picks the direction. "forward" colors columns and seeds tangents,
    "reverse" colors rows and seeds cotangents, and "auto" takes whichever needs
    fewer colors, which is a count and not a timing.

    pattern skips tracing and takes the structure as given: a dense array or
    tensor, a scipy matrix, or a jacolor pattern. Use it when the structure is
    known and an operator rule is missing. It is believed as given, so the
    containment it claims is the caller's to get right.

    chunk and verify carry through to every evaluation.
    """
    if mode not in ("auto", "forward", "reverse"):
        raise ValueError(f"mode must be auto, forward or reverse, got {mode!r}")
    P = sparsity(f, x) if pattern is None else as_pattern(pattern)
    if P.shape[1] != x.numel():
        raise ValueError(
            f"the pattern has {P.shape[1]} columns, x has {x.numel()} elements"
        )
    if mode == "auto":
        cols, rows = color_cols(P), color_rows(P)
        coloring = cols if cols.n_colors <= rows.n_colors else rows
    else:
        coloring = color_cols(P) if mode == "forward" else color_rows(P)
    sig = (tuple(x.shape), x.dtype, x.device)
    return Prepared(f, coloring, sig, chunk, verify)
