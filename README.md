# jacolor

Jacobian sparsity detection and coloring for PyTorch.

A sparse Jacobian rarely needs one AD pass per column. If you detect the
sparsity pattern first you can group the columns that never share a row into
colors, run one pass per color and decompress the result. Julia has this in
SparseConnectivityTracer.jl and SparseMatrixColorings.jl, and JAX has asdex,
while PyTorch has had no equivalent.

**Status: pre-alpha.** The pipeline works end to end and every
result is checked against autograd, though only the operators listed below have
rules and anything outside them is refused rather than guessed.

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

For a solver loop, trace and color once and evaluate many times:

```python
from jacolor import prepare, to_scipy

p = prepare(residual, x0)                    # traces and colors once
print(p)                                     # Prepared(forward, 400x400, ...)

for _ in range(steps):
    r, J = p.value_and_jacobian(x)           # both, which is what a solver wants
    x = x - solve(to_scipy(J), r)            # any scipy sparse solver
```

prepare takes mode ("forward", "reverse", "auto" or "hybrid"), chunk to cap how
many colors are in flight, and verify. If you already know the structure you can
pass it with pattern= and skip tracing altogether, and that accepts a dense
array or tensor, a scipy matrix, or a jacolor pattern.

refine=True spends longer on the coloring, once, to save passes on every
evaluation after. On a 2D Brusselator it took the colors from 8 to 6 with the
Laplacian written through F.pad, and from 10 to 7 with torch.roll, in under a
second at 16 by 16. refine() does the same to a coloring you hold yourself.

On "auto" it plans before it colors. The memory a direction would need and the
fewest colors it could possibly use both follow from the pattern alone, so it
skips a direction that cannot be built as well as one whose lower bound already
rules it out, and it reports the choice in reason.

### When neither direction compresses

A global constraint shows up as one dense row and a shared parameter as one
dense column, and either of them on its own forces a color per line, which
leaves forward and reverse equally stuck. Setting those rows aside and
recovering them in reverse lets the rest compress as usual:

```python
p = prepare(residual, x0, mode="hybrid")
print(p.explain())
```

On a 32 by 32 arrow, one dense row and one dense column over a diagonal, plain
forward and plain reverse each need 32 directions. The split needs 3, two
forward for the rest and one reverse for the dense row, and it stays at 3 when
the problem grows to 64, because the cost follows the coupling rather than the
size. prepare takes the split only when it beats both plain modes, and otherwise
says so and falls back to the cheaper of the two.

Those 3 directions count AD passes rather than time. The passes on this arrow
are batched and each one is trivial, so at n = 256 plain forward evaluates in
0.29 ms against 0.31 ms for the split, while dense jacrev beats both at 0.16 ms.
Trading passes for structure pays off once a pass is expensive, which is the
situation this is built for rather than the one measured here.

explain() gives the same facts in words, covering what the pattern costs, which
lines force the bound, why a direction was chosen and whether a split would
help. Everything it says comes off the detected pattern, where an entry means
the derivative may be nonzero rather than that it certainly is.

## What it promises

The pattern holds for inputs of the same shape, dtype and device on which f
takes the same path through its code, so it says nothing about how f behaves
elsewhere.

A prepared analysis is checked against the input it was built for, and every
result is checked against one directional derivative from autograd. That check
samples one point in one direction, so agreement counts as evidence rather than
proof, and where rounding or overflow leaves it unable to tell it says so
instead of reporting a pass.

When the two disagree by more than rounding accounts for, the check stops short
of blaming the pattern. Both sides come from f's own derivative and carry its
rounding, and a function like a saturated softmax computes its entries through
cancellation that leaves them far less exact than their size suggests. So it
asks a structural question instead, whether f's derivative holds anything on
that line which the pattern leaves out. It looks by bisecting what the pattern
leaves out, one AD pass per step and within a budget, and reads the last entry
on its own, since a sum of several missing ones can cancel. Finding one names it
and raises, while finding none, or running out of passes, leaves the result
unchecked, which falls short of showing that the pattern is complete.

The search stays in the mode the entries came from, tangents for forward and
adjoints for reverse. The other mode need not exist for f, grid_sample having no
forward rule, and where both exist they can describe different derivatives, as a
no_grad block does.

status records which of these happened, through p.status and
summary()["status"], and reads "ok", "inconclusive", "skipped" when verify is
off, or None when the evaluation never got that far. An unchecked result always
warns, so a solver loop can stop on one rather than read the status:

```python
import warnings
from jacolor import TraceUnchecked, VerificationInconclusive

warnings.simplefilter("error", VerificationInconclusive)
warnings.simplefilter("error", TraceUnchecked)   # and to refuse an unchecked pattern
```

The check allows for the precision it can see, which is that of x and of the
result. If f computes at lower precision inside than either of those shows, say
float32 inside a float64 function, pass that dtype as verify=torch.float32 and
the allowance follows it.

Detection leans towards holding too many entries rather than too few, because
the two mistakes cost very different amounts. An extra entry costs a color,
while a missing entry corrupts more than itself: for a true row of [a, b] a
pattern of [1, 0] lets both columns share a color and recovery returns
[a + b, 0], so b disappears and a comes back wrong along with it.

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

nn.Linear is addmm, so ordinary dense layers work. supported_ops() returns the
same table, and a test keeps this list from drifting away from it.

## Limitations

- One tensor in and one tensor out, with no pytrees and no multiple arguments.
- mode="hybrid" splits off the rows that force the bound, which covers dense
  rows rather than general bidirectional coloring. It assumes forward and
  reverse mode describe the same derivative, and a no_grad block inside f breaks
  that, since forward mode ignores it. Detection refuses such an f, while with
  pattern= nothing refuses it and the check reports the result unchecked,
  because the fault lies with f rather than with the pattern.
- Real inputs and outputs only. Complex is refused, since forward mode gives the
  holomorphic derivative and reverse mode its conjugate.
- Tested on CPU in float64, float32, float16 and bfloat16. In bfloat16 the
  verification check comes back inconclusive, its rounding allowance running to
  about half the size of the terms it sums, and in float16 and bfloat16 the
  trace cannot be checked either and warns TraceUnchecked rather than passing
  for checked. Both checks are sound in float64 and float32, and GPU is
  untested.
- An op with no rule, a custom autograd.Function on the derivative path, a
  traced program that disagrees with f, and control flow on the values of x are
  all refused. So is an index computed from x, as in x[x.argmax()] or x[x > 0],
  since the entries it picks move with x. Comparisons and argmax carry no
  derivative but stay tracked for exactly this reason.
- sort, topk, median, kthvalue, cummax and cummin pick elements by value, so
  their rules claim the whole line each pick could come from. That is sound,
  and loose by as much as the line is long.
- CI runs the suite on Python 3.10 to 3.13 against the current torch, with and
  without the numba extra, and once against the declared floor of torch 2.6.0
  with scipy 1.14.1. It also smoke tests the built wheel, hybrid included, from
  outside the source tree.

## Notes

- Patterns are always boolean CSR, since integer dtypes wrap on accumulation and
  drop entries without saying so.
- The coloring tries three orderings, natural, largest-first and smallest-last,
  then iterated greedy from the best of them, which colors again with each color
  class kept together and so can only keep or lower the count. Incidence-degree
  is left out because it is dynamic and does not fit the static-permutation
  kernel. refine goes further by tabu search, one color fewer at a time, until
  it reaches the lower bound or a budget of moves. On a 10 by 10 periodic grid
  greedy needs 8 where 5 is possible, and refine finds the 5.
- The lower bound is the densest line of the pattern, which any coloring has to
  spend a color on. Reaching it settles the question for that direction and
  summary() reports it as optimal, though it says nothing about the other
  direction or about a split. Falling short of it need not mean the heuristic
  did badly, because the bound itself can be loose. A hybrid split carries a
  bound of its own, the sum of its two halves, and gets under the
  single-direction bound by not being a single direction.
- Propagation is one construction throughout, a left-multiply by a boolean
  incidence matrix. Row maps (reshape, permute, slicing, broadcast, cat, stack,
  padding and indexing with a fixed index) are exact, while couplings take a
  union over a slice, where sums are structural and exact unless terms cancel,
  and matmul, conv and softmax are conservative on purpose. A put or scatter
  keeps what it overwrites, since a repeated index has no defined winner in
  torch and the union of every writer is the sound answer.
- Rules never read a value that could change between evaluations. The one
  exception is a mask built from literals in the graph alone, with no input,
  parameter, captured tensor or random op behind it, which is how export writes
  r[1] = v. A where over such a mask takes each element from its one branch, so
  writes into a buffer stay exact.
- detach is kept through export rather than lowered to alias, so the traced
  program has f's derivative and the pattern leaves the detached term out.
- The trace is checked against f in values and in one directional derivative, to
  a tolerance that follows the dtype. A float32 matmul and its traced form sum
  in a different order and differ in their last bits, so a float64 bound would
  refuse f for its arithmetic rather than for its program. Each of the two
  comparisons answers for the dtype it runs in, which for a derivative is the
  input's rather than the output's. Where one of them cannot resolve the two
  programs, either in a dtype whose rounding is wider than the difference
  between them or on an entry that cancels to far below the scale it is compared
  against, the pattern is still built and warns TraceUnchecked, which is as much
  as the check can honestly claim there.
- Every rule states its tightness in propagate.TIGHTNESS and is held to it, by
  containment against real Jacobians at several random points and by equality
  where the claim is exact.
- to_scipy copies to the CPU and detaches, because scipy holds plain numpy
  arrays, so nothing about autograd survives the trip.

## Install

```bash
pip install ".[fast]"          # fast adds the numba coloring kernel
```

Without numba, coloring above 10,000 columns warns and falls back to a
pure-Python loop.

Sources live flat in src/ and the wheel ships them as jacolor, so
pip install -e . does not work. To work on jacolor, install the extras for their
dependencies and run the suite from the repo root, where the tests import from
src/ rather than from the installed copy.

```bash
pip install ".[dev,fast]"
pytest
```

## Measured

These are two data points rather than a benchmark, taken on an Apple M5 with
Python 3.13.9, SciPy 1.16.3, numba 0.62.1 and torch 2.13.0, on one thread in
float64.

The first is a conv layer, 3 to 8 channels over 16x16 with padding 1. Its
Jacobian is 2048 by 768 and 3.2% dense, and the coloring finds 27 colors, equal
to the lower bound of in-channels times kernel area.

Setup is paid once and takes 54 ms to read the pattern off the function, almost
all of it torch.export, plus 2.6 ms to color the 768 columns, which stops at the
first ordering because that one already reaches the lower bound. Coloring the
2048 rows for reverse mode costs more, there being more of them.

Each evaluation from a prepared analysis then takes 1.77 ms against 10.7 ms for
dense jacrev, so 6.0 times faster with verification on, and preparation pays for
itself after about seven evaluations. Without verification an evaluation takes
1.13 ms.

Inside that, the AD passes account for 0.16 ms, against 1.70 ms if they are
looped rather than batched, so assembling the result and checking it is where
most of the time goes. chunk=8 brings an evaluation to 2.58 ms, which is what
bounding peak memory costs.

The second is a tridiagonal pattern with n = 100,000 and 299,998 nonzeros, which
takes 3 colors, again equal to the lower bound, in 0.27 s including the
transpose product.

## License

MIT
