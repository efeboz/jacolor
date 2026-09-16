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

import torch

from .coloring import color_cols
from .compress import _block, _index
from .trace import sparsity

__all__ = ["jacobian", "VerificationError"]


class VerificationError(RuntimeError):
    """The assembled Jacobian disagrees with autograd on f."""


def _push(f, x, S):
    # One jvp per seed column, batched. S is (n_cols, k), result is (m, k).
    k = S.shape[1]
    out = torch.func.vmap(
        lambda v: torch.func.jvp(f, (x,), (v.reshape(x.shape),))[1]
    )(S.T.contiguous())
    return out.reshape(k, -1).T


def _pull(back, shape, S):
    # One vjp per seed row, batched. S is (k, m), result is (k, n_cols). The
    # closure is built once per evaluation, not once per chunk.
    k = S.shape[0]
    out = torch.func.vmap(lambda g: back(g.reshape(shape))[0])(S)
    return out.reshape(k, -1)


def _verify(f, x, ri, ci, vals, coloring):
    # One directional derivative, from autograd and from the assembled entries.
    # A sound pattern gives the same answer either way, so a mismatch means the
    # structure does not belong to this function at this point.
    P = coloring.pattern
    seed = ((P.shape[0] * 1000003 + P.shape[1]) * 1000003 + P.nnz) % (2**31)
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(P.shape[0] if coloring.axis == "cols" else P.shape[1],
                      dtype=vals.dtype, device=vals.device)
    if coloring.axis == "cols":
        v = torch.randn(P.shape[1], generator=g, dtype=x.dtype).to(x.device)
        want = torch.func.jvp(f, (x,), (v.reshape(x.shape),))[1].reshape(-1)
        acc.index_add_(0, ri, vals * v[ci].to(vals.dtype))
    else:
        y, back = torch.func.vjp(f, x)
        w = torch.randn(P.shape[0], generator=g, dtype=y.dtype).to(x.device)
        want = back(w.reshape(y.shape))[0].reshape(-1)
        acc.index_add_(0, ci, vals * w[ri].to(vals.dtype))
    scale = float(want.abs().max()) if want.numel() else 0.0
    tol = 1e-3 if want.dtype in (torch.float16, torch.float32) else 1e-6
    if not torch.allclose(acc, want.to(acc.dtype), rtol=tol, atol=tol * max(scale, 1.0)):
        off = int((acc - want.to(acc.dtype)).abs().argmax())
        raise VerificationError(
            "the assembled Jacobian disagrees with autograd on f. The pattern does "
            "not describe this function at this point, which happens when a reused "
            f"coloring is stale. Largest disagreement at entry {off}: "
            f"{float(acc[off])} against {float(want[off])}."
        )


def jacobian(f, x, coloring=None, chunk=None, verify=True):
    """Sparse Jacobian of f at x, as a coalesced sparse COO tensor.

    Without a coloring this traces f, colors the pattern for forward mode and
    evaluates. Pass a coloring to reuse structure across evaluations, or to work
    in reverse mode by giving one from color_rows.

    chunk caps how many colors are in flight at once. Peak memory is then set by
    the chunk rather than by the color count, at the cost of more AD calls.

    verify checks the result against one directional derivative from autograd.
    It costs one extra AD pass and is what catches a coloring that no longer fits
    the function. Turn it off only in a loop you have already verified.

    The result takes the dtype of the AD pass, which is not always the dtype of
    x. Forward mode follows f's output and reverse mode follows f's input, so a
    float32 input through a float64 constant gives a float64 Jacobian forward and
    a float32 one in reverse, exactly as jvp and vjp do.
    """
    if coloring is None:
        coloring = color_cols(sparsity(f, x))
    P = coloring.pattern
    if x.numel() != P.shape[1]:
        raise ValueError(
            f"the coloring is for an input of {P.shape[1]} elements, got {x.numel()}"
        )
    if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
        raise ValueError(f"chunk must be a positive whole number, got {chunk!r}")

    ri, ci, line = _index(P, coloring, x.device)
    vals = torch.empty(P.nnz, dtype=x.dtype, device=x.device)
    back = shape = None
    if coloring.axis == "rows" and coloring.n_colors:
        y, back = torch.func.vjp(f, x)  # built once, reused by every chunk
        shape = y.shape
        if y.numel() != P.shape[0]:
            raise ValueError(
                f"the coloring is for an output of {P.shape[0]} elements, got {y.numel()}"
            )

    step = chunk or max(coloring.n_colors, 1)  # a zero-color problem has no blocks
    for lo in range(0, coloring.n_colors, step):
        hi = min(lo + step, coloring.n_colors)
        S = _block(coloring, lo, hi, x.dtype, x.device)
        keep = (line >= lo) & (line < hi)
        if coloring.axis == "cols":
            B = _push(f, x, S)
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

    if verify and P.nnz:
        _verify(f, x, ri, ci, vals, coloring)
    return torch.sparse_coo_tensor(torch.stack([ri, ci]), vals, P.shape).coalesce()
