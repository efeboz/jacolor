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

import warnings

import torch

from .coloring import color_cols
from .compress import _block, _index
from .trace import sparsity

__all__ = ["jacobian", "VerificationError", "VerificationInconclusive"]

# Below this, the rounding allowance is small enough against the terms being
# summed for the check to mean something. Measured worst case: 0.00 in float64
# and float32, 0.07 in float16, 0.55 in bfloat16, so bfloat16 lands outside.
_RESOLVE = 0.25


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


def _verify(f, x, ri, ci, vals, coloring, primal=None, back=None):
    """Check the assembled entries against one directional derivative.

    A sound pattern gives the same answer whether the direction goes through
    autograd or through the entries, so a disagreement means the structure does
    not belong to this function at this point. One point and one direction, so
    agreement is evidence and not a proof.

    Returns "ok" when the comparison could resolve the terms it was summing, and
    "inconclusive" when rounding or a non-finite value left it unable to tell.
    """
    P = coloring.pattern
    seed = ((P.shape[0] * 1000003 + P.shape[1]) * 1000003 + P.nnz) % (2**31)
    g = torch.Generator().manual_seed(seed)
    wide = torch.float64  # the check must not contribute error of its own
    n_out = P.shape[0] if coloring.axis == "cols" else P.shape[1]
    acc = torch.zeros(n_out, dtype=wide, device=vals.device)
    mag = torch.zeros(n_out, dtype=wide, device=vals.device)
    cnt = torch.zeros(n_out, dtype=wide, device=vals.device)

    if coloring.axis == "cols":
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
    eps = max(_eps(vals.dtype), _eps(want.dtype))
    room = eps * (32.0 + cnt) * (want.abs() + mag)
    off = (acc - want).abs()

    if not (bool(torch.isfinite(want).all()) and bool(torch.isfinite(acc).all())):
        warnings.warn(VerificationInconclusive(
            "verification saw a non-finite value, so it could not compare the "
            "assembled Jacobian against autograd. The result is unchecked."
        ), stacklevel=3)
        return "inconclusive"
    if bool((off > room).any()):
        i = int((off - room).argmax())
        raise VerificationError(
            "the assembled Jacobian disagrees with autograd on f. The pattern does "
            "not describe this function at this point, which happens when a reused "
            f"coloring is stale. Worst entry {i}: {float(acc[i])} against "
            f"{float(want[i])}, allowing {float(room[i])}."
        )
    # Agreeing inside a rounding allowance as large as the terms themselves says
    # nothing, so it is reported as such rather than as a pass.
    live = mag > 0
    if bool((live & (room > _RESOLVE * mag)).any()):
        worst = float((room[live] / mag[live]).max()) if bool(live.any()) else 0.0
        warnings.warn(VerificationInconclusive(
            f"verification allowed rounding of up to {worst:.2f} of the terms it "
            f"summed, in {vals.dtype}, so agreement does not rule out a wrong "
            "pattern. The result is unchecked."
        ), stacklevel=3)
        return "inconclusive"
    return "ok"


def _assemble(f, x, coloring, chunk, verify, index=None):
    """Run the AD passes, scatter the values, and check. Returns (J, y, status)."""
    P = coloring.pattern
    if x.numel() != P.shape[1]:
        raise ValueError(
            f"the coloring is for an input of {P.shape[1]} elements, got {x.numel()}"
        )
    if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
        raise ValueError(f"chunk must be a positive whole number, got {chunk!r}")

    ri, ci, line = _index(P, coloring, x.device) if index is None else index
    vals = torch.empty(P.nnz, dtype=x.dtype, device=x.device)
    back = shape = primal = None
    if coloring.axis == "rows" and coloring.n_colors:
        primal, back = torch.func.vjp(f, x)  # built once, reused by every chunk
        shape = primal.shape
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
    status = _verify(f, x, ri, ci, vals, coloring, primal, back) if verify else "skipped"
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
    loop you have already verified.

    The result takes the dtype of the AD pass, which is not always the dtype of
    x. Forward mode follows f's output and reverse mode follows f's input, so a
    float32 input through a float64 constant gives a float64 Jacobian forward and
    a float32 one in reverse, exactly as jvp and vjp do.
    """
    if coloring is None:
        coloring = color_cols(sparsity(f, x))
    return _assemble(f, x, coloring, chunk, verify)[0]
