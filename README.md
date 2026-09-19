# jacolor

Jacobian sparsity detection and coloring for PyTorch.

A sparse Jacobian does not need one AD pass per column. Detect the sparsity
pattern, group the columns that never share a row into colors, run one pass per
color, then decompress. Julia has this in SparseConnectivityTracer.jl and
SparseMatrixColorings.jl, and JAX has asdex. PyTorch does not.

**Status: pre-alpha, small op set.** The pipeline works end to end and every
result is checked against autograd, but only the operators listed below have
rules and anything else is refused rather than guessed.

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
many colors are in flight, and verify. If you already know the structure, pass it
and skip tracing: pattern= accepts a dense array or tensor, a scipy matrix, or a
jacolor pattern.

On "auto" it plans before it colors. Both the memory a direction would need and
the fewest colors it could possibly use follow from the pattern alone, so a
direction that cannot be built is skipped, and so is one whose lower bound
already says it cannot win. The choice is reported in reason.

### When neither direction compresses

A global constraint is one dense row and a shared parameter is one dense column.
Either one alone forces a color per line, so forward and reverse are equally
stuck. Setting those rows aside and recovering them in reverse leaves the rest
to compress as usual:

```python
p = prepare(residual, x0, mode="hybrid")
print(p.explain())
```

On a 32 by 32 arrow, one dense row and one dense column over a diagonal, plain
forward and plain reverse each need 32 directions. The split needs 3, two
forward for the rest and one reverse for the dense row, and it is still 3 when
the problem grows to 64, because the cost follows the coupling rather than the
size. If a split would not pay, prepare says so and stays with plain coloring.

That 3 is a count of AD passes, not a time. On this arrow the passes are batched
and each one is trivial, so at n = 256 plain forward evaluates in 0.29 ms
against 0.31 ms for the split, and dense jacrev beats both at 0.16 ms. Fewer
passes is worth having when a pass is expensive, which is the case this is meant
for and not the case measured here.

explain() reports the same facts in words: what the pattern costs, which lines
force the bound, why that direction was chosen, and whether a split would help.
Everything it says is read off the detected pattern, where an entry means the
derivative may be nonzero rather than that it is.

## What it promises

The pattern holds for inputs of the same shape, dtype and device on which f
takes the same path through its code. It is not a claim about f everywhere.

A prepared analysis is checked against the input it was built for, and every
result is checked against one directional derivative from autograd. That check
is a sample at one point in one direction, so agreement is evidence, not proof.
Where rounding leaves it unable to tell, it says so rather than reporting a pass.

Detection is conservative by design, and the asymmetry is not symmetric. An
extra entry costs a color. A missing entry corrupts more than itself: for a true
row of [a, b], a pattern of [1, 0] lets both columns share a color, and recovery
then returns [a + b, 0]. The b is not merely dropped, the a is wrong too.

## Supported operations

| kind | ops |
|---|---|
| row map | cat.default, clone.default, expand.default, permute.default, select.int, slice.Tensor, squeeze.dim, squeeze.dims, unsqueeze.default, view.default |
| pointwise | abs.default, add.Tensor, cos.default, div.Tensor, exp.default, log.default, mul.Tensor, neg.default, pow.Tensor_Scalar, reciprocal.default, relu.default, rsqrt.default, sigmoid.default, sin.default, sqrt.default, sub.Tensor, tanh.default, where.self |
| reduction | mean.default, mean.dim, sum.default, sum.dim_IntList |
| slice coupling | _log_softmax.default, _softmax.default |
| coupling | addmm.default, convolution.default, mm.default |

nn.Linear is addmm, so ordinary dense layers work. supported_ops() returns the
same table, and a test keeps this list from drifting from it.

## Limitations

- One tensor in and one tensor out. No pytrees, no multiple arguments.
- mode="hybrid" splits off the rows that force the bound and needs both AD
  directions to work for f. It is the restricted case, dense rows only, not
  general bidirectional coloring.
- Tested on CPU in float64, float32, float16 and bfloat16. In bfloat16 the
  verification check is reported inconclusive, because its rounding allowance is
  about half the size of the terms it sums. GPU is untested.
- Not supported and refused: an op with no rule, a custom autograd.Function on
  the derivative path, a traced program that disagrees with f, and control flow
  on the values of x.
- Indexing with a tensor index, batched matmul and padding have no rules yet.
- CI runs the suite on Python 3.10 to 3.13 against the current torch, with and
  without the numba extra, and smoke tests the built wheel from outside the
  source tree. The declared torch floor of 2.6 is still a claim, because nothing
  tests it. Locally the suite also runs on scipy 1.14.1 and 1.16.3.

## Notes

- Patterns are boolean CSR, always. Integer dtypes wrap on accumulation and drop
  entries without saying so.
- Coloring orderings: natural, largest-first and smallest-last, fewest colors
  wins. Incidence-degree is not implemented, since it is dynamic and does not fit
  the static-permutation kernel.
- The lower bound is the densest line of the pattern, which any coloring must
  spend a color on. Reaching it settles the question, and summary() reports that
  as optimal. Falling short of it does not mean the heuristic did badly, because
  the bound itself can be loose. A hybrid split has a bound of its own, the sum
  of its two halves, and beats the single-direction bound by not being one
  direction.
- Propagation is one construction throughout: a left-multiply by a boolean
  incidence matrix. Row maps (reshape, permute, slicing, broadcast, cat, stack)
  are exact. Couplings take a union over a slice: sums are structural, exact
  unless terms cancel, while matmul, conv, softmax, sort and tracked indices are
  conservative on purpose.
- Every rule states its tightness in propagate.TIGHTNESS and is held to it:
  containment against real Jacobians at several random points, and equality
  where the claim is exact.
- to_scipy copies to the CPU and detaches, because scipy holds plain numpy
  arrays. Nothing about autograd survives the trip.

## Install

```bash
pip install ".[fast]"          # fast adds the numba coloring kernel
```

Without numba, coloring above 10,000 columns warns and runs a pure-Python loop.

Sources live flat in src/ and the wheel ships them as jacolor, so
pip install -e . does not work. To work on jacolor, install the extras for their
dependencies and run the suite from the repo root. The tests import from src/,
not from the installed copy.

```bash
pip install ".[dev,fast]"
pytest
```

## Measured

Two data points, not a benchmark. Apple M5, Python 3.13.9, SciPy 1.16.3,
numba 0.62.1, torch 2.13.0, one thread, float64.

A conv layer, 3 to 8 channels over 16x16 with padding 1. The Jacobian is
2048 by 768 and 3.2% dense, and the coloring finds 27 colors, equal to the lower
bound of in-channels times kernel area.

Setup, paid once: 54 ms to read the pattern off the function, almost all of it
torch.export, and 2.6 ms to color the 768 columns, which stops at the first
ordering because that one already reaches the lower bound. Coloring the 2048
rows for reverse mode costs more, since there are more of them.

Each evaluation from a prepared analysis then takes 1.77 ms against 10.7 ms for
dense jacrev, so 6.0 times faster with verification on, and preparation pays for
itself after about seven evaluations. Without verification an evaluation is
1.13 ms.

Inside that the AD passes are 0.16 ms, against 1.70 ms if they are looped rather
than batched, so assembling the result and checking it is most of the time.
chunk=8 costs 2.58 ms, which is what bounding peak memory costs.

A tridiagonal pattern, n = 100,000 and 299,998 nonzeros: 3 colors, again equal
to the lower bound, in 0.27 s including the transpose product.

## License

MIT
