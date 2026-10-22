# jacolor

Jacobian sparsity detection and coloring for PyTorch.

When a Jacobian is sparse, columns that never share a row can be grouped into
one color and differentiated together, so the full matrix costs one AD pass per
color instead of one per column. jacolor detects the sparsity pattern from the
function, colors it, runs the batched passes and decompresses the result into a
sparse tensor. Julia provides this through SparseConnectivityTracer.jl and
SparseMatrixColorings.jl, and JAX through asdex; PyTorch has no built-in
equivalent.

**Status: pre-alpha.** The pipeline works end to end and every result is checked
against autograd. Only the operators listed below have rules, and a function
that uses anything else is refused.

## Example

```python
import torch
from jacolor import jacobian

def f(x):                                    # output i touches x[i], x[i+1], x[i+2]
    return torch.tanh(x[:-2] * x[1:-1]) + torch.sin(x[2:])

x = torch.randn(12, dtype=torch.float64)
J = jacobian(f, x)                           # sparse (10, 12), 3 AD passes not 12

torch.allclose(J.to_dense(), torch.func.jacrev(f)(x))   # True
```

In a solver loop, trace and color once, then evaluate many times:

```python
from jacolor import prepare, to_scipy

p = prepare(residual, x0)                    # traces and colors once
print(p)                                     # Prepared(forward, 400x400, ...)

for _ in range(steps):
    r, J = p.value_and_jacobian(x)
    x = x - solve(to_scipy(J), r)            # any scipy sparse solver
```

`prepare` accepts `mode` (`"forward"`, `"reverse"`, `"auto"` or `"hybrid"`),
`chunk` to cap how many colors are evaluated at once and so bound peak memory,
and `verify`. A known structure can be passed as `pattern=` (a dense array or
tensor, a scipy matrix or a jacolor pattern) to skip tracing. `refine=True`
searches longer for a smaller coloring at preparation time; on a 16 by 16
Brusselator it reduced the colors from 8 to 6 in under a second.

With `mode="auto"`, the memory each direction would need and the fewest colors
it could use are computed from the pattern before coloring. A direction that
does not fit or cannot beat the other is skipped, and the choice is reported in
`p.reason`.

A single dense row (a global constraint) or dense column (a shared parameter)
forces one color per line in both plain directions. `mode="hybrid"` recovers the
densest rows in reverse and the rest forward. On a 32 by 32 arrow pattern, plain
forward and reverse each need 32 directions and the split needs 3, a count that
stays at 3 at n = 64. The split is taken only when it uses fewer directions than
both plain modes. Fewer directions do not guarantee less time: at n = 256 on the
same pattern, dense `jacrev` (0.16 ms) beat both plain forward (0.29 ms) and the
split (0.31 ms), because each pass there is cheap. `p.explain()` describes the
pattern, the bound, the chosen direction and whether a split would help.

## Guarantees

The pattern is valid for inputs with the same shape, dtype and device on which
`f` follows the same code path. A prepared analysis rejects inputs that differ
in shape, dtype or device.

Each result is compared with one directional derivative from autograd. This
samples one point in one direction, so agreement is evidence rather than proof.
If the two disagree beyond rounding, jacolor searches, by bisection within a
budget of AD passes, for an entry of `f`'s derivative that the pattern omits. It
raises if it finds one, and otherwise reports the result as unchecked. The
outcome is stored in `p.status` as `"ok"`, `"inconclusive"`, `"skipped"` (when
`verify=False`) or `None` (when evaluation stopped early). An unchecked result
also warns, so a solver loop can stop on it:

```python
import warnings
from jacolor import TraceUnchecked, VerificationInconclusive

warnings.simplefilter("error", VerificationInconclusive)
warnings.simplefilter("error", TraceUnchecked)
```

The tolerance follows the dtypes of `x` and of the result. If `f` computes in a
lower precision internally, for example float32 inside a float64 function, pass
`verify=torch.float32`.

Detection errs towards extra entries, because a missing entry is worse than an
extra color: for a true row `[a, b]`, a pattern `[1, 0]` lets both columns share
a color, and recovery returns `[a + b, 0]`.

## Supported operations

| kind | ops |
|---|---|
| row map | alias.default, cat.default, clone.default, constant_pad_nd.default, copy.default, diagonal.default, expand.default, flip.default, gather.default, index.Tensor, index_select.default, permute.default, repeat.default, select.int, select_scatter.default, slice.Tensor, slice_scatter.default, split.Tensor, split_with_sizes.default, squeeze.default, squeeze.dim, squeeze.dims, unbind.int, unfold.default, unsqueeze.default, view.default |
| pointwise | _to_copy.default, abs.default, acos.default, acosh.default, add.Tensor, asin.default, asinh.default, atan.default, atan2.default, atanh.default, clamp.Tensor, clamp.default, cos.default, cosh.default, div.Tensor, elu.default, erf.default, exp.default, exp2.default, expm1.default, fmax.default, fmin.default, fmod.Scalar, fmod.Tensor, gelu.default, hardtanh.default, hypot.default, leaky_relu.default, log.default, log10.default, log1p.default, log2.default, maximum.default, minimum.default, mul.Tensor, neg.default, pow.Scalar, pow.Tensor_Scalar, pow.Tensor_Tensor, reciprocal.default, relu.default, remainder.Scalar, remainder.Tensor, rsqrt.default, sigmoid.default, sin.default, sinh.default, sqrt.default, sub.Tensor, tan.default, tanh.default, where.self |
| zero derivative | argmax.default, argmin.default, bitwise_and.Tensor, bitwise_not.default, bitwise_or.Tensor, bitwise_xor.Tensor, ceil.default, detach.default, empty_like.default, eq.Scalar, eq.Tensor, floor.default, full_like.default, ge.Scalar, ge.Tensor, gt.Scalar, gt.Tensor, le.Scalar, le.Tensor, logical_and.default, logical_not.default, logical_or.default, lt.Scalar, lt.Tensor, ne.Scalar, ne.Tensor, round.default, sign.default, trunc.default |
| reduction | amax.default, amin.default, kthvalue.default, linalg_vector_norm.default, max.dim, mean.default, mean.dim, median.default, median.dim, min.dim, prod.default, prod.dim_int, sum.default, sum.dim_IntList, var.correction |
| scan | cummax.default, cummin.default, cumprod.default, cumsum.default, logcumsumexp.default |
| slice coupling | _log_softmax.default, _softmax.default, native_layer_norm.default, sort.default, sort.stable, topk.default |
| coupling | addmm.default, bmm.default, convolution.default, mm.default |
| scatter | index_put.default, index_reduce.default, scatter.src, scatter_add.default, scatter_reduce.two |

`nn.Linear` lowers to `addmm`, so dense layers are covered. `supported_ops()`
returns the same table, and a test checks that the two agree.

## Limitations

- `f` must take one tensor and return one tensor; pytrees and multiple arguments
  are not supported.
- Inputs and outputs must be real, because the two AD modes disagree by a
  conjugate on complex functions.
- Refused: operators without a rule, custom `autograd.Function`s on the
  derivative path, traced programs that disagree with `f`, control flow on the
  values of `x`, and indices computed from `x` (as in `x[x.argmax()]`).
- `mode="hybrid"` handles dense rows only, not general bidirectional coloring.
  It assumes forward and reverse mode give the same derivative, which a
  `no_grad` block inside `f` breaks. Detection refuses such an `f`; with
  `pattern=` the result is reported as unchecked.
- `sort`, `topk`, `median`, `kthvalue`, `cummax` and `cummin` select elements by
  value, so their rules include the whole line any selection could come from.
  The pattern stays sound but can be much denser than the true one.
- Tested on CPU only, in float64, float32, float16 and bfloat16. In float16 and
  bfloat16 the trace check cannot resolve rounding and warns `TraceUnchecked`,
  and in bfloat16 the verification check is also inconclusive.
- CI covers Python 3.10 to 3.13 with the current torch, with and without numba,
  plus torch 2.6.0 with scipy 1.14.1 as the declared floor, and a smoke test of
  the built wheel.

## How it works

`f` is traced with `torch.export` and decomposed to core ATen operators, and each
operator's rule propagates the pattern by multiplying with a boolean incidence
matrix. Patterns are stored as boolean
CSR, because integer dtypes overflow when contributions accumulate and would
silently drop entries. Each rule declares in `propagate.TIGHTNESS` whether it is
exact or conservative, and tests check containment against real Jacobians at
random points, and equality where a rule claims to be exact. Rules never read
values that can change between evaluations, with one exception: masks built
only from literals in the graph.

Coloring is greedy over the column intersection graph, trying natural,
largest-first and smallest-last orderings, followed by iterated greedy. `refine`
continues with a tabu search until it reaches the lower bound (the densest line
of the pattern) or exhausts its move budget; on a 10 by 10 periodic grid it finds
5 colors where greedy needs 8. When a coloring reaches the lower bound,
`summary()` reports it as optimal for that direction.

## Install

```bash
pip install ".[fast]"          # fast adds the numba coloring kernel
```

Without numba, coloring above 10,000 columns warns and uses a pure-Python loop.
The wheel installs `src/` as the package `jacolor`, which rules out
`pip install -e .`. For development, install the extras and run the suite from
the repository root, where the tests import from `src/`:

```bash
pip install ".[dev,fast]"
pytest
```

## Measured performance

All measurements are on an Apple M5, single-threaded CPU, float64, torch 2.13.0
and numba 0.62.1. They cover two problems and should not be read as general
benchmarks.

[Case study 1](case_studies/cs1_sparse_jacobians/README.md) builds the full
Jacobian of the Broyden tridiagonal function for n from 64 to 100,000 and
compares it with `torch.func.jacfwd` and `jacrev`. jacolor found 3 colors at
every size, the minimum for this pattern, and every result matched the analytic
derivative. Dense AD is faster up to n = 256: at n = 64, `jacrev` takes 0.088 ms
against 0.403 ms for a prepared jacolor evaluation. The two cross between
n = 256 and n = 1024, and at n = 2048 jacolor takes 0.63 ms against 9.1 ms for
the faster dense method, a 14.6× speedup. Preparation costs 55 to 90 ms once,
which is recovered after about seven evaluations at n = 2048. At n = 100,000 an
evaluation takes 11.5 ms and the result occupies 6.9 MiB, where a dense matrix
would need 74.5 GiB; dense AD was not run above n = 2048. A handwritten
constructor that knows the formula is 3 to 34 times faster than jacolor, which
is the cost of detecting the structure and verifying each result.

A convolution layer (3 to 8 channels, 16 by 16, padding 1) gives a 2048 by 768
Jacobian that is 3.2% dense and needs 27 colors, equal to its lower bound.
Preparation takes about 57 ms, most of it in `torch.export`. A prepared
evaluation takes 1.77 ms with verification and 1.13 ms without, against 10.7 ms
for dense `jacrev`. The batched AD passes account for only 0.16 ms of that, so
assembly and verification dominate.

## License

MIT
