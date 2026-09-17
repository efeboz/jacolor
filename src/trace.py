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

import torch

from . import _boolcsr as bc
from . import propagate as pr

__all__ = ["sparsity", "supported_ops", "UnsupportedOp", "CustomBackward",
           "TraceMismatch"]

aten = torch.ops.aten


class UnsupportedOp(NotImplementedError):
    """An op on a tracked value with no sparsity rule."""


class CustomBackward(RuntimeError):
    """A custom autograd.Function takes part in the derivative of f."""


class TraceMismatch(RuntimeError):
    """The traced program does not compute what f computes."""


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


def _row_map(node, args, kwargs, pats, out):
    # Run the op on an index tensor. What comes out is the row map, with torch's
    # own handling of negative ends, -1 sizes and expansion.
    idx = torch.arange(args[0].numel()).reshape(args[0].shape)
    src = node.target(idx, *args[1:], **kwargs).reshape(-1).numpy()
    return pr.gather(pats[0], src)


def _pointwise(node, args, kwargs, pats, out):
    terms = [(P, tuple(a.shape)) for a, P in zip(args, pats) if P is not None]
    return pr.pointwise(tuple(out.shape), terms)


def _reduce(node, args, kwargs, pats, out):
    # sum and mean share an incidence. Both have a nonzero derivative for every
    # element of the slice, 1 and 1/n, so both are exact.
    x = args[0]
    # An empty dim list means every dim, as in ATen. x.sum() exports as sum(x, []).
    dims = tuple(args[1]) if len(args) > 1 and len(args[1]) else tuple(range(x.dim()))
    return pr.reduce_sum(pats[0], tuple(x.shape), dims)


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
_ROW_MAPS = [aten.view.default, aten.slice.Tensor, aten.permute.default, aten.expand.default,
             aten.unsqueeze.default, aten.squeeze.dim, aten.squeeze.dims,
             aten.clone.default, aten.select.int]
# Elementwise ops. where.self belongs here because its condition is an untracked
# operand, so the union over the tracked branches already ignores it.
_POINTWISE = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor,
              aten.neg.default, aten.tanh.default, aten.sin.default, aten.cos.default,
              aten.exp.default, aten.relu.default, aten.sigmoid.default,
              aten.sqrt.default, aten.rsqrt.default, aten.log.default,
              aten.reciprocal.default, aten.abs.default, aten.pow.Tensor_Scalar,
              aten.where.self]

RULES = {
    **{op: _row_map for op in _ROW_MAPS},
    **{op: _pointwise for op in _POINTWISE},
    aten.sum.dim_IntList: _reduce,
    aten.sum.default: _reduce,
    aten.mean.dim: _reduce,
    aten.mean.default: _reduce,
    aten._softmax.default: _softmax,
    aten._log_softmax.default: _softmax,
    aten.mm.default: _mm,
    aten.addmm.default: _addmm,
    aten.cat.default: _cat,
    aten.convolution.default: _conv,
}

_KIND = {_row_map: "row map", _pointwise: "pointwise", _reduce: "reduction",
         _softmax: "slice coupling", _mm: "coupling", _addmm: "coupling",
         _cat: "row map", _conv: "coupling"}


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


def _export(f, x):
    mod = f if isinstance(f, torch.nn.Module) else _Wrap(f)
    key = _anchor(f)
    try:
        return torch.export.export(mod, (x,), strict=False).run_decompositions()
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
            if _tracked(pats):
                rule = RULES.get(node.target)
                if rule is None:
                    raise _refuse(node, "no sparsity rule on a tracked input")
                P = rule(node, args, kwargs, pats, out)
                assert P.shape[0] == out.numel(), (
                    f"{node.target} rule gave {P.shape[0]} rows for {out.numel()} elements"
                )
                pat[node] = P
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


def _same_derivative(f, traced, x):
    # Reverse mode, because its operator coverage is the wider of the two. One
    # direction at one point, so this is evidence rather than proof.
    g = torch.Generator().manual_seed(1 + x.numel())
    with torch.enable_grad():
        y, back = torch.func.vjp(f, x)
        w = torch.randn(y.shape, generator=g).to(y.dtype).to(x.device)
        want = back(w)[0]
        got = torch.func.vjp(traced, x)[1](w)[0]
    return torch.allclose(got, want, rtol=1e-6, atol=1e-9, equal_nan=True)


def sparsity(f, x):
    """Jacobian sparsity pattern of f at x, as bool CSR of shape (f(x).numel(), x.numel()).

    Holds for every input of x's shape on which f takes the same path through its
    code. Control flow on the values of x is outside that, and export refuses it.

    Two things are checked against f itself at x: that the traced program returns
    the same values, and that it has the same derivative in one random direction.
    Both are samples, so they are evidence that the trace stands for f rather than
    a guarantee of it.
    """
    y, custom = _eager(f, x)
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
    same = traced.shape == y.shape and torch.allclose(traced, y, rtol=1e-9, atol=1e-12,
                                                      equal_nan=True)
    if same:
        # Equal values are not enough. The two programs can agree at a point and
        # still have different derivatives, which is the only thing the pattern
        # is about, so compare a directional derivative as well.
        same = _same_derivative(f, ep.module(), x)
    if not same:
        raise TraceMismatch(
            "the traced program does not agree with f on the example input, so the "
            "pattern would describe a different function. This happens when export "
            "specializes a branch, as with torch.compiler.is_compiling(), and when f "
            "is not deterministic."
        )
    return P
