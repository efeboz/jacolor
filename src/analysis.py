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

import numpy as np
import torch

from . import _boolcsr as bc
from .coloring import MAX_EDGES, color_cols, color_rows, graph_estimate
from .compress import _block, _index
from .evaluate import _assemble, _check_chunk, _check_verify, _floor, _pull, _push, _verify
from .interop import pattern as as_pattern
from .trace import _real, sparsity

__all__ = ["prepare", "Prepared"]


class Prepared:
    """A pattern, a coloring and the indices to scatter results back onto it."""

    __slots__ = ("f", "coloring", "split", "chunk", "verify", "status", "reason",
                 "_index", "_sig")

    def __init__(self, f, coloring, sig, chunk=None, verify=True, reason="", split=None):
        self.f = f
        self.coloring = coloring
        self.split = split
        self.chunk = chunk
        self.verify = verify
        self.status = None  # outcome of the most recent verification
        self.reason = reason  # why this direction, in words
        self._sig = sig  # shape, dtype and device this was prepared for
        self._index = None  # built on first use, then reused

    # --- what was found -----------------------------------------------------

    @property
    def pattern(self):
        return self.split.full if self.split else self.coloring.pattern

    @property
    def mode(self):
        if self.split:
            return "hybrid"
        return "forward" if self.coloring.axis == "cols" else "reverse"

    @property
    def n_colors(self):
        return self.split.n_colors if self.split else self.coloring.n_colors

    @property
    def lower_bound(self):
        return self.split.lower_bound if self.split else self.coloring.lower_bound

    @property
    def optimal(self):
        """True when the coloring reaches its own lower bound.

        The bound belongs to one direction, or to one split. Reaching it proves
        nothing of the same kind can use fewer colors, and says nothing about
        the others. A pattern can sit at its forward bound and still be cheaper
        in reverse or split. Falling short does not mean the heuristic did
        badly either, because the bound itself can be loose.
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
            "reason": self.reason,
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
            self._index = (_split_index(self.split, x.device) if self.split
                           else _index(self.pattern, self.coloring, x.device))

    def jacobian(self, x):
        """The sparse Jacobian of f at x."""
        return self.value_and_jacobian(x)[1]

    def value_and_jacobian(self, x):
        """f at x and its sparse Jacobian, which a nonlinear solver wants together."""
        self._check(x)
        if self.split:
            J, y, self.status = _assemble_split(
                self.f, x, self.split, self._index, self.chunk, self.verify
            )
        else:
            J, y, self.status = _assemble(
                self.f, x, self.coloring, self.chunk, self.verify, self._index
            )
        return (self.f(x) if y is None else y), J

    def explain(self):
        """Where the cost comes from, in words.

        Everything here is read off the detected pattern. An entry means the
        derivative may be nonzero, not that it is, so a line called dense is
        dense as far as the rules can tell.
        """
        s = self.summary()
        scope = "this split" if self.split else f"{s['mode']} coloring of this pattern"
        rn = bc.row_nnz(self.pattern)
        cn = bc.row_nnz(bc.transpose(self.pattern))
        out = [f"{s['rows']} by {s['cols']}, {s['nnz']} nonzeros, "
               f"{100 * s['density']:.1f} percent dense",
               f"chose {s['mode']}: {self.reason}",
               f"{s['colors']} colors against a lower bound of {s['lower_bound']}"
               + (f", which is as few as {scope} allows" if s["optimal"] else "")]
        if rn.size:
            wide = int(rn.max())
            out.append(f"the widest row holds {wide} entries, and {int((rn == wide).sum())} "
                       "row(s) do, which is what forward mode cannot go below")
        if cn.size:
            tall = int(cn.max())
            out.append(f"the widest column holds {tall} entries, and "
                       f"{int((cn == tall).sum())} column(s) do, which is the same "
                       "limit for reverse mode")
        if not self.split and rn.size and int((rn == rn.max()).sum()) < rn.size:
            rest = int(rn[rn < rn.max()].max())
            out.append(f"setting those {int((rn == rn.max()).sum())} row(s) aside leaves a "
                       f"pattern whose widest row holds {rest}. mode='hybrid' recovers "
                       "them in reverse and the rest forward, and takes that only if it "
                       "beats both plain directions")
        return "\n".join(out)


class _Split:
    """A pattern cut in two: rows recovered forward, and rows recovered reverse.

    Some otherwise sparse problems carry a global constraint, which is one dense
    row, or a shared parameter, which is one dense column. Either alone forces
    ordinary coloring to spend a color per line. Setting the dense rows aside and
    recovering them in reverse leaves the rest to compress as it otherwise would.
    """

    __slots__ = ("full", "rows", "rest", "dense")

    def __init__(self, full, rows, rest, dense):
        self.full = full  # the whole pattern
        self.rows = rows  # row indices recovered in reverse
        self.rest = rest  # column coloring of the forward half
        self.dense = dense  # row coloring of the reverse half

    @property
    def n_colors(self):
        return self.rest.n_colors + self.dense.n_colors

    @property
    def lower_bound(self):
        # Each half keeps its own bound, and this split cannot go below the sum.
        return self.rest.lower_bound + self.dense.lower_bound


def _cut(P, rows):
    # Both halves keep the full shape, so a column keeps its index and each
    # coloring lines up with the whole pattern.
    ri = np.repeat(np.arange(P.shape[0]), bc.row_nnz(P))
    ci = P.indices.astype(np.int64)
    m = np.isin(ri, rows)
    return bc.from_pairs(ri[~m], ci[~m], P.shape), bc.from_pairs(ri[m], ci[m], P.shape)


def _split_plan(P):
    """Set aside the rows that force the forward bound, if that beats both plain modes.

    Returns (split, plain, reason). plain is set when a plain direction was
    colored for the comparison and won, so the caller need not color it again.
    A plain direction that cannot be built, or whose lower bound already reaches
    the split's count, cannot win and is never colored at all.
    """
    rn = bc.row_nnz(P)
    if P.shape[0] == 0 or P.nnz == 0:
        return None, None, "the pattern is empty, so there is nothing to split"
    rows = np.flatnonzero(rn == rn.max())
    if rows.size == P.shape[0]:
        return None, None, "every row is as dense as the densest, so a split changes nothing"
    rest_P, dense_P = _cut(P, rows)
    if max(graph_estimate(rest_P), graph_estimate(bc.transpose(dense_P))) > MAX_EDGES:
        return None, None, "a split would not fit in the coloring budget either"
    split = _Split(P, rows, color_cols(rest_P), color_rows(dense_P))

    said, best = [], None
    for name, Q, color in (("forward", P, color_cols), ("reverse", bc.transpose(P), color_rows)):
        lb = int(bc.row_nnz(Q).max()) if Q.shape[0] else 0
        if graph_estimate(Q) > MAX_EDGES:
            said.append(f"plain {name} would not fit in memory")
        elif lb >= split.n_colors:
            said.append(f"plain {name} needs at least {lb}")
        else:
            c = color(P)
            said.append(f"plain {name} needs {c.n_colors}")
            if c.n_colors < split.n_colors and (best is None or c.n_colors < best[1].n_colors):
                best = (name, c)
    if best is not None:
        name, c = best
        return None, c, (f"a split would need {split.n_colors} directions against "
                         f"{c.n_colors} for plain {name}, so it was not taken")
    return split, None, (f"hybrid, {split.rest.n_colors} forward and {split.dense.n_colors} "
                         f"reverse, where {' and '.join(said)}")


def _split_index(split, device):
    P = split.full
    ri = np.repeat(np.arange(P.shape[0]), bc.row_nnz(P))
    ci = P.indices.astype(np.int64)
    give = torch.from_numpy(np.isin(ri, split.rows)).to(device)
    return (torch.from_numpy(ri).to(device), torch.from_numpy(ci).to(device), give,
            torch.tensor(split.rest.colors[ci]).to(device),
            torch.tensor(split.dense.colors[ri]).to(device))


def _assemble_split(f, x, split, index, chunk, verify):
    """Each row belongs to one half, and each half is recovered its own way.

    chunk caps the colors in flight within each half, as it does for plain modes.
    """
    P = split.full
    ri, ci, give, line_f, line_r = index
    keep = ~give
    vals = primal = None

    kf = split.rest.n_colors
    step = chunk or max(kf, 1)
    for lo in range(0, kf, step):
        hi = min(lo + step, kf)
        primal, B = _push(f, x, _block(split.rest, lo, hi, x.dtype, x.device))
        if B.shape[0] != P.shape[0]:
            raise ValueError(
                f"the split is for an output of {P.shape[0]} elements, got {B.shape[0]}"
            )
        if vals is None:
            vals = torch.empty(P.nnz, dtype=B.dtype, device=B.device)
        sel = keep & (line_f >= lo) & (line_f < hi)
        vals[sel] = B[ri[sel], line_f[sel] - lo]

    _real(primal, "output")
    y, back = torch.func.vjp(f, x)  # built once, reused by every chunk
    kr = split.dense.n_colors
    step = chunk or max(kr, 1)
    for lo in range(0, kr, step):
        hi = min(lo + step, kr)
        Sr = _block(split.dense, lo, hi, x.dtype, x.device, lines=split.rows)
        Br = _pull(back, y.shape, Sr.T.contiguous())
        sel = give & (line_r >= lo) & (line_r < hi)
        vals[sel] = Br[line_r[sel] - lo, ci[sel]].to(vals.dtype)

    # Checked in forward mode, so the rows recovered in reverse are held to a
    # forward derivative. That is what catches the two modes disagreeing.
    status = (_verify(f, x, ri, ci, vals, P, "cols", primal, None, _floor(verify))
              if verify else "skipped")
    J = torch.sparse_coo_tensor(torch.stack([ri, ci]), vals, P.shape).coalesce()
    return J, primal, status


def _plan(P):
    """Choose a direction from what the pattern already says, before coloring.

    Both the memory a direction would need and the fewest colors it could
    possibly use are known from the pattern alone. That is enough to skip a
    direction that cannot be built, and to skip one that cannot win.
    """
    T = bc.transpose(P)
    lb_fwd = int(bc.row_nnz(P).max()) if P.shape[0] else 0
    lb_rev = int(bc.row_nnz(T).max()) if T.shape[0] else 0
    cand = [("forward", graph_estimate(P), lb_fwd, color_cols),
            ("reverse", graph_estimate(T), lb_rev, color_rows)]
    fits = [c for c in cand if c[1] <= MAX_EDGES]
    if not fits:
        raise MemoryError(
            "neither direction fits in the coloring budget: forward would need up "
            f"to {cand[0][1]} graph entries and reverse up to {cand[1][1]}, against "
            f"a budget of {MAX_EDGES}. Give a pattern with sparser rows or columns."
        )

    fits.sort(key=lambda c: c[2])  # the smaller lower bound is the better bet
    name, _, _, color = fits[0]
    best = color(P)
    if len(fits) == 1:
        other = next(c for c in cand if c[0] != name)
        return best, (f"{name}, {best.n_colors} colors. {other[0]} was skipped because "
                      f"its graph could reach {other[1]} entries, over the budget")

    alt_name, _, alt_lb, alt_color = fits[1]
    if best.n_colors <= alt_lb:
        return best, (f"{name}, {best.n_colors} colors. {alt_name} cannot beat that, "
                      f"since it needs at least {alt_lb}")
    alt = alt_color(P)
    if alt.n_colors < best.n_colors:
        return alt, f"{alt_name}, {alt.n_colors} colors against {best.n_colors} for {name}"
    return best, f"{name}, {best.n_colors} colors against {alt.n_colors} for {alt_name}"


def prepare(f, x, mode="auto", pattern=None, chunk=None, verify=True):
    """Trace and color f once, for repeated evaluation at inputs shaped like x.

    mode picks the direction. "forward" colors columns and seeds tangents,
    "reverse" colors rows and seeds cotangents, and "auto" plans: it skips a
    direction whose intersection graph would not fit, coloring only the other,
    and skips one whose lower bound already says it cannot win. The objective is
    a color count, not a timing. The choice is reported in reason.

    pattern skips tracing and takes the structure as given: a dense array or
    tensor, a scipy matrix, or a jacolor pattern. Use it when the structure is
    known and an operator rule is missing. It is believed as given, so the
    containment it claims is the caller's to get right.

    chunk and verify carry through to every evaluation. verify may be a dtype,
    the coarsest precision f computes in, when that is lower than its input and
    output show.

    "hybrid" recovers the rows that force the forward bound in reverse and the
    rest forward, and takes that only if it beats both plain directions. It
    assumes forward and reverse mode describe the same derivative. A no_grad
    block inside f can break that. sparsity refuses such an f through its own
    derivative check, and with pattern= the evaluation check is what catches it.
    Inputs and outputs must be real.
    """
    if mode not in ("auto", "forward", "reverse", "hybrid"):
        raise ValueError(
            f"mode must be auto, forward, reverse or hybrid, got {mode!r}"
        )
    _check_chunk(chunk)  # before tracing, so a bad value costs nothing
    _check_verify(verify)
    _real(x, "input")
    P = sparsity(f, x) if pattern is None else as_pattern(pattern)
    if P.shape[1] != x.numel():
        raise ValueError(
            f"the pattern has {P.shape[1]} columns, x has {x.numel()} elements"
        )
    if mode == "hybrid":
        split, plain, why = _split_plan(P)
        sig = (tuple(x.shape), x.dtype, x.device)
        if split is not None:
            return Prepared(f, split.rest, sig, chunk, verify, why, split)
        if plain is None:  # declined before any plain direction was colored
            plain, choice = _plan(P)
            why = f"{choice}. {why}"
        else:
            name = "forward" if plain.axis == "cols" else "reverse"
            why = f"{name}, {plain.n_colors} colors. {why}"
        return Prepared(f, plain, sig, chunk, verify, why)
    if mode == "auto":
        coloring, reason = _plan(P)
    else:
        coloring = color_cols(P) if mode == "forward" else color_rows(P)
        reason = f"{mode}, as asked, {coloring.n_colors} colors"
    sig = (tuple(x.shape), x.dtype, x.device)
    return Prepared(f, coloring, sig, chunk, verify, reason)
