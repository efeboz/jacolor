"""The sparse Jacobian in one call.

This module is named evaluate rather than jacobian so that the module and the
function it exports do not shadow each other on the package.

Detect the pattern, color it, run one batched AD pass per color, decompress.
The AD passes are batched with vmap rather than looped, which is where most of
the speed comes from once there is more than a handful of colors.

Structure is reusable. Pass a coloring computed once and every later evaluation
skips tracing and coloring entirely. A coloring says nothing about whether f
still has the graph it was built from, so the result is checked against one
directional derivative from autograd before it is returned.
"""

import math
import warnings

import torch

from .coloring import color_cols
from .compress import _block, _index
from .trace import _real, sparsity

__all__ = ["jacobian", "VerificationError", "VerificationInconclusive"]

# Below this, the rounding allowance is small enough against the terms being
# summed for the check to mean something. Measured worst case: 0.00 in float64
# and float32, 0.07 in float16, 0.55 in bfloat16, so bfloat16 lands outside.
_RESOLVE = 0.25

# AD passes the search for a missing entry may spend. Two per split and one to
# read the entry, so this covers a line of 2**15, and running out is reported
# rather than concluded.
_PROBE_PASSES = 32


class VerificationError(RuntimeError):
    """The assembled Jacobian disagrees with autograd on f."""


class VerificationInconclusive(UserWarning):
    """Verification could not tell agreement from rounding, so it proved nothing."""


def _eps(dtype):
    return float(torch.finfo(dtype).eps) if dtype.is_floating_point else 2.0**-52


def _push(f, x, S):
    # One jvp per seed column, batched. S is (n_cols, k), result is (m, k). jvp
    # returns the primal too, so the value of f comes free with the derivative.
    k = S.shape[1]
    primal, out = torch.func.vmap(
        lambda v: torch.func.jvp(f, (x,), (v.reshape(x.shape),))
    )(S.T.contiguous())
    return primal[0], out.reshape(k, -1).T


def _pull(back, shape, S):
    # One vjp per seed row, batched. S is (k, m), result is (k, n_cols). The
    # closure is built once per evaluation, not once per chunk.
    k = S.shape[0]
    out = torch.func.vmap(lambda g: back(g.reshape(shape))[0])(S)
    return out.reshape(k, -1)


def _verify(f, x, ri, ci, vals, P, axis, primal=None, back=None, floor=None):
    """Check the assembled entries against one directional derivative.

    A sound pattern gives the same answer whether the direction goes through
    autograd or through the entries. One point and one direction, so agreement
    is evidence and not a proof.

    A disagreement is not, by itself, evidence against the pattern. Both sides
    come from f's own derivative and carry its rounding, which cancellation
    inside f can make far larger than the entries suggest. So a disagreement is
    put to the structure, in _missing, before it is called an error.

    floor is the coarsest precision f computes in, when that is lower than its
    input and output show.

    Returns "ok" when the comparison could resolve the terms it was summing, and
    "inconclusive" when rounding or a non-finite value left it unable to tell.
    """
    seed = ((P.shape[0] * 1000003 + P.shape[1]) * 1000003 + P.nnz) % (2**31)
    g = torch.Generator().manual_seed(seed)
    wide = torch.float64  # the check must not contribute error of its own
    n_out = P.shape[0] if axis == "cols" else P.shape[1]
    acc = torch.zeros(n_out, dtype=wide, device=vals.device)
    mag = torch.zeros(n_out, dtype=wide, device=vals.device)
    cnt = torch.zeros(n_out, dtype=wide, device=vals.device)

    if axis == "cols":
        v = torch.randn(P.shape[1], generator=g).to(x.dtype).to(x.device)
        want = torch.func.jvp(f, (x,), (v.reshape(x.shape),))[1].reshape(-1)
        at, term = ri, vals.to(wide) * v[ci].to(wide)
    else:
        if back is None:  # reuse the caller's closure rather than run f again
            primal, back = torch.func.vjp(f, x)
        w = torch.randn(P.shape[0], generator=g).to(primal.dtype).to(x.device)
        want = back(w.reshape(primal.shape))[0].reshape(-1)
        at, term = ci, vals.to(wide) * w[ri].to(wide)
    acc.index_add_(0, at, term)
    mag.index_add_(0, at, term.abs())
    cnt.index_add_(0, at, torch.ones_like(term))
    want = want.to(wide)

    # Rounding room follows the size of the terms that were added and the number
    # of them, not an absolute floor. A Jacobian scaled down by 1e-8 stays just
    # as checkable, and a low-precision dtype is not held to float64 accuracy.
    # Casting the oracle to float64 does not make the arithmetic behind it
    # float64, so the input's precision counts as much as the result's, and so
    # does any lower one f declares it uses inside. The oracle carries the same
    # dtype as the entries, so it needs no term of its own.
    eps = max(_eps(vals.dtype), _eps(x.dtype),
              _eps(floor) if floor is not None else 0.0)
    scale = want.abs() + mag
    room = eps * (32.0 + cnt) * scale
    off = (acc - want).abs()

    # Every intermediate, not only the ends. Entries near 1e308 are finite but
    # their sums are not, and an infinite allowance would pass anything.
    if not all(bool(torch.isfinite(t).all()) for t in (want, acc, mag, room, off)):
        warnings.warn(VerificationInconclusive(
            "verification saw a non-finite value, so it could not compare the "
            "assembled Jacobian against autograd. The result is unchecked."
        ), stacklevel=3)
        return "inconclusive"
    if bool((off > room).any()):
        # Both sides come from f's own derivative, so numbers alone cannot say
        # whether the pattern is wrong or f rounded badly. Ask the structure.
        i = int((off - room).argmax())
        found, why = _missing(f, x, ri, ci, P, axis, i, primal, back, g)
        if found is not None:
            j, at_j = found
            r, c = (i, j) if axis == "cols" else (j, i)
            raise VerificationError(
                f"f's derivative returns {at_j:.3g} at entry ({r}, {c}), which the "
                "pattern does not hold, so a color carries it onto another entry and "
                "the assembled Jacobian is wrong. This is what a reused coloring gone "
                f"stale looks like. Line {i} disagrees with autograd by "
                f"{float(off[i]):.3g} of a scale of {float(scale[i]):.3g}."
            )
        warnings.warn(VerificationInconclusive(
            f"line {i} disagrees with autograd by {float(off[i]):.3g} of a scale of "
            f"{float(scale[i]):.3g}, which is more than rounding accounts for, and "
            f"{why}. Cancellation inside f's own derivative does this, and so does a "
            "hybrid whose two modes disagree. If f computes at lower precision inside "
            "than its input and output show, pass that dtype as verify. The result is "
            "unchecked."
        ), stacklevel=3)
        return "inconclusive"
    # Agreeing inside a rounding allowance as large as the answer says nothing.
    # The scale has to include the answer, not only the terms that were summed:
    # cancellation leaves no terms at all, and gating on those would let a
    # pattern that reconstructs exactly zero pass against a nonzero truth.
    live = scale > 0
    if bool((live & (room > _RESOLVE * scale)).any()):
        worst = float((room[live] / scale[live]).max()) if bool(live.any()) else 0.0
        warnings.warn(VerificationInconclusive(
            f"verification allowed rounding of up to {worst:.2f} of the terms it "
            f"summed, in {vals.dtype}, so agreement does not rule out a wrong "
            "pattern. The result is unchecked."
        ), stacklevel=3)
        return "inconclusive"
    return "ok"


def _line(f, x, axis, i, primal, back, cols, w):
    """Line i of f's derivative over cols, weighted by w, as one AD pass.

    Forward for entries that came from tangents and reverse for entries that
    came from adjoints, so the evidence is the derivative the entries are. The
    other mode may not exist for f, and where both exist they can disagree.
    """
    if axis == "cols":  # a row of J, from a tangent along those columns
        u = torch.zeros(x.numel(), dtype=x.dtype, device=x.device)
        u[cols] = w.to(u.dtype).to(u.device)
        return float(torch.func.jvp(f, (x,), (u.reshape(x.shape),))[1].reshape(-1)[i])
    u = torch.zeros(primal.numel(), dtype=primal.dtype, device=x.device)  # a column
    u[cols] = w.to(u.dtype).to(u.device)
    return float(back(u.reshape(primal.shape))[0].reshape(-1)[i])


def _missing(f, x, ri, ci, P, axis, i, primal, back, g):
    """Look for an entry of line i that f's derivative has and the pattern lacks.

    Bisection over what the pattern leaves out, one AD pass per step, within a
    budget. Returns ((index, value), "") for an entry that carries a derivative,
    or (None, why) when the search came up empty, which is not the same as the
    pattern being complete. Sums are what hide an entry, through cancellation,
    so the search goes on until one entry is left and reads that alone.
    """
    n = P.shape[1] if axis == "cols" else P.shape[0]
    held = ci[ri == i] if axis == "cols" else ri[ci == i]
    rest = torch.ones(n, dtype=torch.bool, device=x.device)
    rest[held] = False
    rest = torch.nonzero(rest).reshape(-1)
    if rest.numel() == 0:
        return None, "the pattern holds every entry of that line, so none is missing"

    left = _PROBE_PASSES
    while rest.numel() > 1:
        if left < 2:
            return None, "the search for an entry it leaves out ran out of passes"
        half = rest.numel() // 2
        parts = (rest[:half], rest[half:])
        seen = [abs(_line(f, x, axis, i, primal, back, part,
                          torch.randn(part.numel(), generator=g)))
                for part in parts]
        left -= 2
        if not all(math.isfinite(v) for v in seen):
            return None, "reading what it leaves out overflowed"
        if max(seen) == 0.0:
            return None, "nothing it leaves out moved that line"
        rest = parts[0] if seen[0] >= seen[1] else parts[1]
    if left < 1:
        return None, "the search for an entry it leaves out ran out of passes"
    at_j = _line(f, x, axis, i, primal, back, rest, torch.ones(1))
    if not math.isfinite(at_j):
        return None, "reading what it leaves out overflowed"
    if at_j == 0.0:
        return None, "nothing it leaves out moved that line"
    return (int(rest[0]), at_j), ""


def _check_verify(verify):
    if not (isinstance(verify, bool)
            or (isinstance(verify, torch.dtype) and verify.is_floating_point)):
        raise ValueError(f"verify must be True, False or a floating dtype, got {verify!r}")


def _floor(verify):
    # A dtype passed as verify means yes, and here is the coarsest precision f uses.
    return verify if isinstance(verify, torch.dtype) else None


def _check_chunk(chunk):
    if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
        raise ValueError(f"chunk must be a positive whole number, got {chunk!r}")


def _assemble(f, x, coloring, chunk, verify, index=None):
    """Run the AD passes, scatter the values, and check. Returns (J, y, status)."""
    P = coloring.pattern
    if x.numel() != P.shape[1]:
        raise ValueError(
            f"the coloring is for an input of {P.shape[1]} elements, got {x.numel()}"
        )
    _check_chunk(chunk)
    _check_verify(verify)
    _real(x, "input")

    ri, ci, line = _index(P, coloring, x.device) if index is None else index
    vals = torch.empty(P.nnz, dtype=x.dtype, device=x.device)
    back = shape = primal = None
    if coloring.axis == "rows" and coloring.n_colors:
        primal, back = torch.func.vjp(f, x)  # built once, reused by every chunk
        _real(primal, "output")  # before a pull, which fails on it less clearly
        shape = primal.shape
        if primal.numel() != P.shape[0]:
            raise ValueError(
                f"the coloring is for an output of {P.shape[0]} elements, "
                f"got {primal.numel()}"
            )

    if coloring.n_colors == 0:
        # No block ever runs, so the checks inside the loop never happen. The
        # output still has to be the size the pattern was built for.
        primal = f(x)
        if primal.numel() != P.shape[0]:
            raise ValueError(
                f"the coloring is for an output of {P.shape[0]} elements, "
                f"got {primal.numel()}"
            )

    step = chunk or max(coloring.n_colors, 1)  # a zero-color problem has no blocks
    for lo in range(0, coloring.n_colors, step):
        hi = min(lo + step, coloring.n_colors)
        S = _block(coloring, lo, hi, x.dtype, x.device)
        keep = (line >= lo) & (line < hi)
        if coloring.axis == "cols":
            primal, B = _push(f, x, S)
            if B.shape[0] != P.shape[0]:
                raise ValueError(
                    f"the coloring is for an output of {P.shape[0]} elements, "
                    f"got {B.shape[0]}"
                )
        else:
            B = _pull(back, shape, S.T.contiguous())
        if B.dtype != vals.dtype:  # a mixed-dtype f differentiates wider than x
            vals = vals.to(B.dtype)
        vals[keep] = B[ri[keep], line[keep] - lo] if coloring.axis == "cols" \
            else B[line[keep] - lo, ci[keep]]

    # An empty pattern claims every derivative is zero, which is as strong a
    # claim as any and just as worth checking.
    _real(primal, "output")
    status = (_verify(f, x, ri, ci, vals, P, coloring.axis, primal, back, _floor(verify))
              if verify else "skipped")
    J = torch.sparse_coo_tensor(torch.stack([ri, ci]), vals, P.shape).coalesce()
    return J, primal, status


def jacobian(f, x, coloring=None, chunk=None, verify=True):
    """Sparse Jacobian of f at x, as a coalesced sparse COO tensor.

    Without a coloring this traces f, colors the pattern for forward mode and
    evaluates. Pass a coloring to reuse structure across evaluations, or to work
    in reverse mode by giving one from color_rows. For repeated evaluation see
    prepare, which also keeps the decompression indices.

    chunk caps how many colors are in flight at once. Peak memory is then set by
    the chunk rather than by the color count, at the cost of more AD calls.

    verify checks the result against one directional derivative from autograd.
    It costs one extra AD pass and is what catches a coloring that no longer fits
    the function. Where rounding leaves the check unable to tell, it warns
    VerificationInconclusive rather than reporting a pass. Turn it off only in a
    loop you have already verified. If f computes at lower precision inside than
    its input and output show, pass that dtype instead, as verify=torch.float32,
    and the tolerance allows for it. A disagreement raises only where f's
    derivative holds something the pattern does not, and is otherwise reported
    as unchecked.

    Inputs and outputs must be real. For complex f the two AD modes give
    conjugate answers, and this release does not pick between them.

    The result takes the dtype of the AD pass, which is not always the dtype of
    x. Forward mode follows f's output and reverse mode follows f's input, so a
    float32 input through a float64 constant gives a float64 Jacobian forward and
    a float32 one in reverse, exactly as jvp and vjp do.
    """
    if coloring is None:
        coloring = color_cols(sparsity(f, x))
    return _assemble(f, x, coloring, chunk, verify)[0]
