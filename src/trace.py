"""Automatic sparsity detection: trace a function, then propagate patterns.

The function is exported to core ATen ops and the graph is walked once. Every
node is also evaluated on the real example value, which gives shapes and the
values of untracked branches. A node reachable from the input carries a pattern,
and each op maps to one rule in propagate.

Two things are refused rather than guessed, because the traced graph can then
disagree with the derivative autograd computes on f. An op on a tracked value
that has no rule, and a custom autograd.Function, whose backward the trace never
sees.
"""

import operator
import warnings

import numpy as np
import torch

from . import _boolcsr as bc
from . import propagate as pr

__all__ = ["sparsity", "supported_ops", "UnsupportedOp", "CustomBackward",
           "TraceMismatch", "TraceUnchecked"]

aten = torch.ops.aten


class UnsupportedOp(NotImplementedError):
    """An op on a tracked value with no sparsity rule."""


class CustomBackward(RuntimeError):
    """A custom autograd.Function takes part in the derivative of f."""


# Past this, a dtype's own rounding swamps the comparison, so the tolerance
# stops following it rather than letting the check accept anything.
_COARSE = 0.05

# A comparison looser than this cannot tell two programs apart, since it allows
# more than the difference between them need be.
_UNRESOLVED = 1e-3


class TraceMismatch(RuntimeError):
    """The traced program does not compute what f computes."""


class TraceUnchecked(UserWarning):
    """The trace could not be told from a different program, so it stands unchecked."""


def _real(t, what):
    # Forward mode gives the holomorphic derivative of a complex function and
    # reverse mode its conjugate. A result assembled from either, and the check
    # against them, would need a convention this release does not pick.
    if t.is_complex():
        raise TypeError(
            f"complex {what} is not supported. Forward and reverse mode disagree on "
            "complex derivatives, one gives the conjugate of the other"
        )


def _where(node):
    # Frames arrive as a "File ..." line followed by a code line. The wrapper in
    # this file is never the answer, so report the innermost frame outside it.
    frames = []
    for ln in (node.meta.get("stack_trace") or "").splitlines():
        ln = ln.strip()
        if ln.startswith("File "):
            frames.append([ln])
        elif frames and ln:
            frames[-1].append(ln)
    frames = [fr for fr in frames if __file__ not in fr[0]]
    return "\n".join(frames[-1]) if frames else "(no source location)"


def _refuse(node, why):
    return UnsupportedOp(f"{why}: {node.target}, at\n{_where(node)}")


# --- rules -------------------------------------------------------------------
# Each takes (node, args, kwargs, pats, out). args and kwargs hold real values,
# pats lines up with args and holds a pattern for tracked tensors, None otherwise.


def _flat(p):
    return [q for r in p for q in _flat(r)] if isinstance(p, list) else [p]


def _empty(n_rows, pats):
    # No derivative, in the column space of whatever is tracked.
    width = next(P for P in _flat(pats) if P is not None).shape[1]
    return bc.from_pairs([], [], (n_rows, width))


def _fixed_index(node, pats, at):
    # An index read off x itself would move with x, and a pattern taken from its
    # value at this point would go stale without a sound. It stays tracked, with
    # an empty pattern, precisely so that it lands here.
    if any(P is not None for i in at if i < len(pats) for P in _flat(pats[i])):
        raise _refuse(node, "an index computed from x has no rule")


def _run_on_index(node, args, kwargs, first=0):
    # The op itself, run on element numbers in place of values.
    x = args[0]
    idx = torch.arange(first, first + x.numel(), device=x.device).reshape(x.shape)
    return node.target(idx, *args[1:], **kwargs)


def _row_map(node, args, kwargs, pats, out):
    # Run the op on an index tensor. What comes out is the row map, with torch's
    # own handling of negative ends, -1 sizes and expansion. An op that returns
    # several tensors, as split does, gets one pattern for each.
    _fixed_index(node, pats, range(1, len(pats)))
    src = _run_on_index(node, args, kwargs)
    if isinstance(src, (list, tuple)):
        return [pr.gather(pats[0], t.reshape(-1).cpu().numpy()) for t in src]
    return pr.gather(pats[0], src.reshape(-1).cpu().numpy())


def _pad(node, args, kwargs, pats, out):
    # Numbered from 1 and padded with 0, so a padded position reads -1: no input
    # element, so an empty row. A negative pad crops and needs nothing extra.
    src = _run_on_index(node, (args[0], args[1], 0), {}, first=1)
    return pr.gather(pats[0], src.reshape(-1).cpu().numpy() - 1)


def _overwrite(node, args, kwargs, pats, out):
    # slice_scatter and select_scatter: every output element is one element of
    # base or one of src. Number both in one range and run the op on the numbers.
    base, src = args[0], args[1]
    nb = base.numel()
    ib = torch.arange(nb, device=base.device).reshape(base.shape)
    isrc = torch.arange(nb, nb + src.numel(), device=base.device).reshape(src.shape)
    at = node.target(ib, isrc, *args[2:], **kwargs).reshape(-1).cpu().numpy()
    parts = [P if P is not None else _empty(t.numel(), pats)
             for t, P in ((base, pats[0]), (src, pats[1]))]
    return pr.gather(bc.vstack(parts), at)


def _copy(node, args, kwargs, pats, out):
    # copy(self, src) is src broadcast into self's shape, so self contributes nothing.
    if pats[1] is None:
        return _empty(out.numel(), pats)
    return pr.gather(pats[1], pr.bcast_src(tuple(args[1].shape), tuple(out.shape)))


def _land(P_base, P_src, at, src, out):
    # Source element src[k] lands on output element at[k], on top of base. Base
    # is kept even where it is overwritten: a union of every writer is sound when
    # indices repeat, and a put with repeats has no defined winner in torch.
    n_out = out.numel()
    parts = []
    if P_base is not None:
        parts.append(P_base)
    if P_src is not None:
        parts.append(pr.couple(P_src, at, src, n_out))
    return pr.union(*parts) if parts else None


def _index_put(node, args, kwargs, pats, out):
    # index_put(self, indices, values, accumulate), which index_add also becomes.
    _fixed_index(node, pats, (1,))
    indices, values = args[1], args[2]
    pos = aten.index.Tensor(torch.arange(out.numel(), device=out.device).reshape(out.shape),
                            indices)
    src = pr.bcast_src(tuple(values.shape), tuple(pos.shape))
    P = _land(pats[0], pats[2], pos.reshape(-1).cpu().numpy(), src, out)
    return _empty(out.numel(), pats) if P is None else P


def _scatter(node, args, kwargs, pats, out):
    # scatter(self, dim, index, src) and scatter_add. src[k] lands where index
    # points along dim, and only the leading index-shaped block of src is read.
    _fixed_index(node, pats, (2,))
    dim, index, src = args[1], args[2], args[3]
    num = torch.arange(out.numel(), device=out.device).reshape(out.shape)
    at = num.gather(dim, index).reshape(-1).cpu().numpy()
    block = tuple(slice(0, k) for k in index.shape)
    read = torch.arange(src.numel()).reshape(src.shape)[block].reshape(-1).numpy()
    P = _land(pats[0], pats[3], at, read, out)
    return _empty(out.numel(), pats) if P is None else P


def _index_reduce(node, args, kwargs, pats, out):
    # index_reduce(self, dim, index, source, reduce): source slice k lands on
    # self's slice index[k] along dim.
    _fixed_index(node, pats, (2,))
    dim, index, source = args[1], args[2], args[3]
    num = torch.arange(out.numel(), device=out.device).reshape(out.shape)
    at = num.index_select(dim, index).reshape(-1).cpu().numpy()
    P = _land(pats[0], pats[3], at, np.arange(source.numel()), out)
    return _empty(out.numel(), pats) if P is None else P


def _where_fixed(node, args, kwargs, pats, out):
    # A condition built from literals in the graph alone, as export makes for
    # r[1] = v, is the same at every evaluation, so it may be read, and each
    # element takes its pattern from the one branch it comes from. A condition
    # that could change, from x, a parameter or a captured tensor, never
    # reaches here.
    cond, a, b = args
    shape, n = tuple(out.shape), out.numel()
    parts = [pr.gather(P, pr.bcast_src(tuple(t.shape), shape)) if P is not None
             else _empty(n, pats) for t, P in ((a, pats[1]), (b, pats[2]))]
    pick = torch.broadcast_to(cond, shape).reshape(-1).cpu().numpy()
    return pr.gather(bc.vstack(parts), np.where(pick, np.arange(n), n + np.arange(n)))


def _fixed(node, pats, fixed):
    # Computed from graph literals alone: no input, parameter, buffer or
    # captured tensor behind it, and nothing random.
    random = torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ())
    return (not random and not _tracked(pats)
            and all(a in fixed for a in node.all_input_nodes))


def _zero(node, args, kwargs, pats, out):
    # No derivative anywhere: comparisons, sign, rounding, argmax, a fresh tensor
    # shaped like x. The result stays tracked with an empty pattern rather than
    # becoming a constant, so that using it as an index is still refused.
    outs = out if isinstance(out, (list, tuple)) else [out]
    got = [_empty(t.numel(), pats) for t in outs]
    return got if isinstance(out, (list, tuple)) else got[0]


def _pointwise(node, args, kwargs, pats, out):
    terms = [(P, tuple(a.shape)) for a, P in zip(args, pats) if P is not None]
    return pr.pointwise(tuple(out.shape), terms)


# Where an op keeps its dim argument, when not second, and what an absent one
# means. None there means every dim.
_DIM_AT = {aten.linalg_vector_norm.default: (2, None), aten.kthvalue.default: (2, -1),
           aten.topk.default: (2, -1), aten.sort.default: (1, -1),
           aten.sort.stable: (None, -1)}


def _dims(node, args, kwargs, x):
    at, default = _DIM_AT.get(node.target, (1, None))
    dims = args[at] if at is not None and len(args) > at else kwargs.get("dim", default)
    # An empty or None dim list means every dim, as in ATen. x.sum() exports as
    # sum(x, []), and x.sum(dim=None) as sum(x, None).
    if isinstance(dims, int):
        return (dims,)
    return tuple(dims) if dims else tuple(range(x.dim()))


def _with_indices(P, out, pats):
    # Ops that return values and their indices. The indices carry no derivative.
    if isinstance(out, (list, tuple)):
        return [P, _empty(out[1].numel(), pats)]
    return P


def _reduce(node, args, kwargs, pats, out):
    # Every reduction shares one incidence: each output element unions the slice
    # that fed it. sum and mean are exact, their derivative being 1 and 1/n.
    # prod, amax, var, norm, median and kthvalue can lose an entry to a value,
    # which a union keeps.
    x = args[0]
    P = pr.reduce_sum(pats[0], tuple(x.shape), _dims(node, args, kwargs, x))
    return _with_indices(P, out, pats)


def _scan(node, args, kwargs, pats, out):
    # cumsum and logcumsumexp are structural. cummax and cummin pick one earlier
    # element, which one depending on values, so the prefix is conservative there.
    P = pr.prefix(pats[0], tuple(args[0].shape), args[1])
    return _with_indices(P, out, pats)


def _select(node, args, kwargs, pats, out):
    # sort and topk: each value is one element of its line, which one depending
    # on values, so it draws on the whole line. Conservative.
    x = args[0]
    (dim,) = _dims(node, args, kwargs, x)
    values = out[0] if isinstance(out, (list, tuple)) else out
    P = pr.slice_couple(pats[0], tuple(x.shape), (dim,), tuple(values.shape))
    return _with_indices(P, out, pats)


def _layer_norm(node, args, kwargs, pats, out):
    # native_layer_norm(x, normalized_shape, weight, bias, eps) returns the
    # output, the mean and the reciprocal deviation. Every output element draws
    # on the whole normalized slice, and on its own weight and bias entry.
    x, normalized, w, b = args[0], args[1], args[2], args[3]
    shape = tuple(x.shape)
    dims = tuple(range(x.dim() - len(normalized), x.dim()))
    parts = [pr.gather(P, pr.bcast_src(tuple(t.shape), shape))
             for t, P in ((w, pats[2]), (b, pats[3])) if P is not None]
    if pats[0] is not None:
        parts.append(pr.slice_couple(pats[0], shape, dims))
    stats = (pr.reduce_sum(pats[0], shape, dims) if pats[0] is not None
             else _empty(out[1].numel(), pats))
    return [pr.union(*parts), stats, stats]


def _softmax(node, args, kwargs, pats, out):
    return pr.slice_couple(pats[0], tuple(args[0].shape), (args[1],))


def _mm(node, args, kwargs, pats, out):
    (m, k), (_, n) = args[0].shape, args[1].shape
    return pr.mm(pats[0], pats[1], m, k, n)


def _addmm(node, args, kwargs, pats, out):
    # bias + mat1 @ mat2, which is what nn.Linear becomes. The scale factors are
    # values, so they are not read: a beta of zero still leaves the bias in.
    bias, a, b = args[0], args[1], args[2]
    (m, k), (_, n) = a.shape, b.shape
    parts = []
    if pats[1] is not None or pats[2] is not None:
        parts.append(pr.mm(pats[1], pats[2], m, k, n))
    if pats[0] is not None:
        parts.append(pr.gather(pats[0], pr.bcast_src(tuple(bias.shape), tuple(out.shape))))
    return pr.union(*parts)


def _bmm(node, args, kwargs, pats, out):
    (b, m, k), (_, _, n) = args[0].shape, args[1].shape
    return pr.bmm(pats[0], pats[1], b, m, k, n)


def _cat(node, args, kwargs, pats, out):
    xs, dim = args[0], (args[1] if len(args) > 1 else 0)
    width = next(P for P in pats[0] if P is not None).shape[1]
    parts = [
        P if P is not None else bc.from_pairs([], [], (t.numel(), width))
        for t, P in zip(xs, pats[0])
    ]
    return pr.cat(parts, [tuple(t.shape) for t in xs], dim)


def _conv(node, args, kwargs, pats, out):
    x, w, b, stride, padding, dilation, transposed, _, groups = args
    if transposed:
        raise _refuse(node, "no rule for transposed convolution")
    if x.dim() != 4:
        raise _refuse(node, "only 2D convolution has a rule")
    parts = []
    if pats[0] is not None or pats[1] is not None:
        parts.append(pr.conv2d(pats[0], pats[1], tuple(x.shape), tuple(w.shape),
                               stride, padding, dilation, groups))
    if pats[2] is not None:  # bias[co] reaches every output of channel co
        parts.append(pr.gather(pats[2], pr.bcast_src((1, b.shape[0], 1, 1), tuple(out.shape))))
    return pr.union(*parts)


# Ops whose output element is one input element moved. The row map comes from
# running the op itself on an index tensor, so torch defines the semantics.
# Indices, where an op takes them, must not be computed from x.
_ROW_MAPS = [aten.view.default, aten.slice.Tensor, aten.permute.default, aten.expand.default,
             aten.unsqueeze.default, aten.squeeze.default, aten.squeeze.dim,
             aten.squeeze.dims, aten.clone.default, aten.select.int, aten.alias.default,
             aten.flip.default, aten.repeat.default, aten.unfold.default,
             aten.diagonal.default, aten.index.Tensor, aten.index_select.default,
             aten.gather.default, aten.split.Tensor, aten.split_with_sizes.default,
             aten.unbind.int]
# Elementwise ops. where.self belongs here because its condition carries no
# derivative, being untracked or a comparison with an empty pattern, so the union
# comes from the branches. A cast is elementwise too.
_POINTWISE = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor,
              aten.neg.default, aten.tanh.default, aten.sin.default, aten.cos.default,
              aten.tan.default, aten.asin.default, aten.acos.default, aten.atan.default,
              aten.sinh.default, aten.cosh.default, aten.asinh.default, aten.acosh.default,
              aten.atanh.default, aten.atan2.default, aten.hypot.default,
              aten.exp.default, aten.exp2.default, aten.expm1.default,
              aten.log.default, aten.log2.default, aten.log10.default, aten.log1p.default,
              aten.sqrt.default, aten.rsqrt.default, aten.reciprocal.default,
              aten.abs.default, aten.erf.default,
              aten.pow.Tensor_Scalar, aten.pow.Tensor_Tensor, aten.pow.Scalar,
              aten.relu.default, aten.sigmoid.default, aten.gelu.default, aten.elu.default,
              aten.leaky_relu.default, aten.hardtanh.default,
              aten.clamp.default, aten.clamp.Tensor, aten.maximum.default,
              aten.minimum.default, aten.fmax.default, aten.fmin.default,
              aten.fmod.Scalar, aten.fmod.Tensor, aten.remainder.Scalar,
              aten.remainder.Tensor, aten.where.self, aten._to_copy.default]
# No derivative at all, as far as autograd is concerned.
_ZERO = [aten.gt.Scalar, aten.gt.Tensor, aten.ge.Scalar, aten.ge.Tensor,
         aten.lt.Scalar, aten.lt.Tensor, aten.le.Scalar, aten.le.Tensor,
         aten.eq.Scalar, aten.eq.Tensor, aten.ne.Scalar, aten.ne.Tensor,
         aten.bitwise_and.Tensor, aten.bitwise_or.Tensor, aten.bitwise_xor.Tensor,
         aten.bitwise_not.default, aten.logical_and.default, aten.logical_or.default,
         aten.logical_not.default, aten.sign.default, aten.floor.default,
         aten.ceil.default, aten.round.default, aten.trunc.default,
         aten.argmax.default, aten.argmin.default, aten.full_like.default,
         aten.empty_like.default, aten.detach.default]
_REDUCTIONS = [aten.sum.dim_IntList, aten.sum.default, aten.mean.dim, aten.mean.default,
               aten.prod.dim_int, aten.prod.default, aten.amax.default, aten.amin.default,
               aten.max.dim, aten.min.dim, aten.var.correction,
               aten.linalg_vector_norm.default, aten.median.dim, aten.median.default,
               aten.kthvalue.default]

RULES = {
    **{op: _row_map for op in _ROW_MAPS},
    **{op: _pointwise for op in _POINTWISE},
    **{op: _zero for op in _ZERO},
    **{op: _reduce for op in _REDUCTIONS},
    aten.cumsum.default: _scan,
    aten.cumprod.default: _scan,
    aten.logcumsumexp.default: _scan,
    aten.cummax.default: _scan,
    aten.cummin.default: _scan,
    aten.sort.default: _select,
    aten.sort.stable: _select,
    aten.topk.default: _select,
    aten.native_layer_norm.default: _layer_norm,
    aten.constant_pad_nd.default: _pad,
    aten.slice_scatter.default: _overwrite,
    aten.select_scatter.default: _overwrite,
    aten.copy.default: _copy,
    aten.index_put.default: _index_put,
    aten.scatter.src: _scatter,
    aten.scatter_add.default: _scatter,
    aten.scatter_reduce.two: _scatter,
    aten.index_reduce.default: _index_reduce,
    aten._softmax.default: _softmax,
    aten._log_softmax.default: _softmax,
    aten.mm.default: _mm,
    aten.bmm.default: _bmm,
    aten.addmm.default: _addmm,
    aten.cat.default: _cat,
    aten.convolution.default: _conv,
}

# Assertions export inserts, and reading a size, neither of which carries a
# derivative. An explicit list: any other op on a tracked value without a rule
# is refused.
_IGNORE = {aten._assert_tensor_metadata.default, aten.sym_size.int}

_KIND = {_row_map: "row map", _pad: "row map", _overwrite: "row map", _copy: "row map",
         _cat: "row map", _pointwise: "pointwise", _zero: "zero derivative",
         _reduce: "reduction", _scan: "scan", _softmax: "slice coupling",
         _mm: "coupling", _bmm: "coupling", _addmm: "coupling", _conv: "coupling",
         _select: "slice coupling", _layer_norm: "slice coupling",
         _index_put: "scatter", _scatter: "scatter", _index_reduce: "scatter"}


def supported_ops():
    """Every op with a rule, as (name, kind) pairs, sorted by name.

    The table in the README is checked against this, so the two cannot drift.
    """
    return sorted((str(op).replace("aten.", ""), _KIND[rule]) for op, rule in RULES.items())


# --- tracing -----------------------------------------------------------------


class _Wrap(torch.nn.Module):
    # torch.export takes a module, not a plain callable.
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def _anchor(f):
    # Export keeps a stack frame only if it is a forward or a registered anchor,
    # so a plain function's own lines would hide behind the wrapper. The registry
    # is private torch state: add f only if absent, and return the key so exactly
    # that entry can be removed again.
    reg = getattr(torch.fx.proxy, "_STACK_TRACE_ANCHORS", None)
    code = getattr(f, "__code__", None)
    if reg is None or code is None or (code.co_filename, code.co_name) in reg:
        return None
    key = (code.co_filename, code.co_name)
    reg.add(key)
    return key


def _eager(f, x):
    # One eager pass, used for two checks. Any custom autograd.Function that
    # takes part in the derivative leaves a BackwardCFunction node in the graph,
    # which is visible however apply was reached, including an apply captured
    # into a local name before tracing.
    with torch.enable_grad():  # an ambient no_grad would leave nothing to inspect
        xr = x.detach().clone().requires_grad_(True)
        y = f(xr)
    seen, stack, found = set(), [getattr(y, "grad_fn", None)], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, torch.autograd.function.BackwardCFunction):
            found.add(type(node).__name__)
        stack.extend(p for p, _ in getattr(node, "next_functions", ()))
    return y.detach(), sorted(found)


def _decompositions():
    # detach would decompose to alias, which autograd differentiates through, so
    # the traced program would have a different derivative from f. Kept as is.
    table = torch.export.default_decompositions()
    table.pop(aten.detach.default, None)
    return table


def _export(f, x):
    mod = f if isinstance(f, torch.nn.Module) else _Wrap(f)
    key = _anchor(f)
    try:
        return torch.export.export(mod, (x,), strict=False).run_decompositions(_decompositions())
    finally:
        if key is not None:
            torch.fx.proxy._STACK_TRACE_ANCHORS.discard(key)


def _vals(a, val):
    if isinstance(a, torch.fx.Node):
        return val[a]
    if isinstance(a, (list, tuple)):
        return type(a)(_vals(v, val) for v in a)
    if isinstance(a, dict):
        return {k: _vals(v, val) for k, v in a.items()}
    return a


def _pats(a, pat):
    if isinstance(a, torch.fx.Node):
        return pat.get(a)
    if isinstance(a, (list, tuple)):
        return [_pats(v, pat) for v in a]
    return None


def _tracked(p):
    return any(_tracked(q) for q in p) if isinstance(p, list) else p is not None


def _walk(ep, x):
    sig = ep.graph_signature
    lifted = {**sig.inputs_to_parameters, **sig.inputs_to_buffers,
              **sig.inputs_to_lifted_tensor_constants}
    (user,) = sig.user_inputs
    if len(sig.user_outputs) != 1:
        raise ValueError(f"f must return one tensor, it returns {len(sig.user_outputs)}")
    val, pat = {}, {}  # real value of every node, pattern of the tracked ones
    fixed = set()  # nodes built from graph literals alone, see _fixed

    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name == user:
                val[node], pat[node] = x, bc.eye(x.numel())
            else:
                fqn = lifted[node.name]
                val[node] = ep.state_dict[fqn] if fqn in ep.state_dict else ep.constants[fqn]
        elif node.op == "call_function":
            args, kwargs = _vals(node.args, val), _vals(node.kwargs, val)
            out = node.target(*args, **kwargs)
            pats = _pats(node.args, pat)
            if _tracked(_pats(list(node.kwargs.values()), pat)):
                raise _refuse(node, "tracked keyword argument")
            if node.target is operator.getitem and pats[0] is not None:
                # One piece of an op that returned several, as split does.
                pat[node] = pats[0][args[1]]
            elif _tracked(pats) and node.target not in _IGNORE:
                rule = RULES.get(node.target)
                if node.target is aten.where.self and node.args[0] in fixed:
                    rule = _where_fixed
                if rule is None:
                    raise _refuse(node, "no sparsity rule on a tracked input")
                P = rule(node, args, kwargs, pats, out)
                for Q, t in zip(_flat(P) if isinstance(P, list) else [P],
                                out if isinstance(out, (list, tuple)) else [out]):
                    assert Q.shape[0] == t.numel(), (
                        f"{node.target} rule gave {Q.shape[0]} rows for {t.numel()} elements"
                    )
                pat[node] = P
            elif _fixed(node, pats, fixed):
                fixed.add(node)
            val[node] = out
        elif node.op == "output":
            res = next(a for a in node.args[0]
                       if isinstance(a, torch.fx.Node) and a.name == sig.user_outputs[0])
        else:
            raise _refuse(node, f"graph node kind {node.op!r} is not handled")

    P = pat.get(res)
    if P is None:
        P = bc.from_pairs([], [], (val[res].numel(), x.numel()))
    return P, val[res]


def _tol(dtype, bound):
    """How far apart the same program may land in this dtype, relative to scale.

    A float32 matmul and its traced form differ in their last bits, and a fixed
    float64 bound refuses f for its arithmetic rather than for its program. The
    tolerance is capped, or a dtype coarse enough would turn the check off.
    """
    eps = float(torch.finfo(dtype).eps) if dtype.is_floating_point else 0.0
    return min(max(bound, 64.0 * eps), _COARSE)


def _agree(got, want, bound):
    """Two tensors to a tolerance that follows the dtype, not float64.

    Returns (agree, why). why is None when the comparison could resolve what it
    was looking at, and otherwise says what stopped it: a dtype whose rounding
    is wider than two programs need differ by, or an entry so far below the
    largest one that it is compared against that scale rather than itself.
    """
    if got.shape != want.shape:
        return False, None
    # A nan or an infinity carries no scale, and taking one would leave the
    # tolerance nan and refuse everything. allclose compares those itself.
    # float64 throughout, since narrowing turns a large value into an infinity
    # and a small one into zero, and either way the scale would be lost.
    finite = torch.nan_to_num(want.double(), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(finite.abs().max()) if want.numel() else 0.0
    tol = _tol(want.dtype, bound)
    agree = bool(torch.allclose(got, want, rtol=tol, atol=tol * scale, equal_nan=True))
    if tol > _UNRESOLVED:
        return agree, (f"comparing it with f in {want.dtype} allows {tol:.2g} of the "
                       "scale, which is more than two different programs need differ by")
    apart = (got.double() - want.double()).abs()
    if bool((apart > tol * want.double().abs()).any()):
        return agree, ("it agrees with f on an entry only against the scale of the "
                       "largest one, so that entry could be a different dependency "
                       "and not show")
    return agree, None


def _copy_differentiably(self, src, non_blocking=False):
    # What aten.copy computes, in ops autograd has a derivative for.
    return src.to(self.dtype).expand(self.shape).clone()


def _differentiable(ep):
    # Export writes into a buffer with the functional aten.copy, which autograd
    # has no derivative for, so the traced program is checked with it swapped
    # for the same computation spelled differently.
    gm = ep.module()
    for node in gm.graph.nodes:
        if node.target is aten.copy.default:
            node.target = _copy_differentiably
    gm.recompile()
    return gm


def _same_derivative(f, traced, x):
    # Reverse mode, because its operator coverage is the wider of the two. One
    # direction at one point, so this is evidence rather than proof.
    g = torch.Generator().manual_seed(1 + x.numel())
    with torch.enable_grad():
        y, back = torch.func.vjp(f, x)
        w = torch.randn(y.shape, generator=g).to(y.dtype).to(x.device)
        # Detached, since a module's parameters leave the cotangent requiring grad.
        want = back(w)[0].detach()
        got = torch.func.vjp(traced, x)[1](w)[0].detach()
    agree, why = _agree(got, want, 1e-6)
    if agree and not bool(torch.isfinite(want).all()):
        # Two different programs can both give nan or an infinity here, so
        # matching there is no evidence. The values are not held to this: the
        # derivative is what the pattern stands on, and a finite one still
        # speaks for it when the values overflow or carry a nan from x.
        why = ("f's derivative is not finite at this input, and agreeing on a nan "
               "or an infinity cannot tell two programs apart")
    return agree, why


def sparsity(f, x):
    """Jacobian sparsity pattern of f at x, as bool CSR of shape (f(x).numel(), x.numel()).

    Holds for every input of x's shape on which f takes the same path through its
    code. Control flow on the values of x is outside that, and export refuses it.

    Two things are checked against f itself at x: that the traced program returns
    the same values, and that it has the same derivative in one random direction.
    Both are samples, so they are evidence that the trace stands for f rather than
    a guarantee of it.
    """
    _real(x, "input")
    y, custom = _eager(f, x)
    _real(y, "output")
    if custom:
        raise CustomBackward(
            f"the derivative of f goes through a custom autograd.Function "
            f"({', '.join(custom)}). A trace keeps its forward only, so the pattern "
            "could miss what its backward does."
        )
    ep = _export(f, x)
    with torch.no_grad():
        P, traced = _walk(ep, x)
    # Export may specialize a branch, so the traced program can compute something
    # other than f. A pattern taken from it would then describe the wrong function.
    same, why = _agree(traced, y, 1e-9)
    if same:
        # Equal values are not enough. The two programs can agree at a point and
        # still have different derivatives, which is the only thing the pattern
        # is about, so compare a directional derivative as well. It runs in the
        # input's dtype, which need not be the output's, so it answers for
        # itself rather than for y.
        same, also = _same_derivative(f, _differentiable(ep), x)
        why = why or also
    if same and why is not None:
        # Agreement the comparison cannot resolve is not evidence. The pattern is
        # built anyway, since its structure rarely turns on the dtype, but
        # nothing here stands behind it and saying so is the least of it.
        warnings.warn(TraceUnchecked(
            f"this pattern is not checked against f, because {why}. Every evaluation "
            "is still checked, and tracing in float32 or float64 and passing the "
            "pattern to prepare is what checks the pattern itself."
        ), stacklevel=2)
    if not same:
        raise TraceMismatch(
            "the traced program does not agree with f on the example input, so the "
            "pattern would describe a different function. This happens when export "
            "specializes a branch, as with torch.compiler.is_compiling(), and when f "
            "is not deterministic."
        )
    return P
