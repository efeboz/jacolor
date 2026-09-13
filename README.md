# jacolor

Jacobian sparsity detection and coloring for PyTorch.

Sparse automatic differentiation computes a sparse Jacobian in far fewer AD
passes than it has rows or columns: detect the sparsity pattern, group columns
that never share a row into colors, seed one AD pass per color, and decompress.
The Julia ecosystem has this (`SparseConnectivityTracer.jl`, `SparseMatrixColorings.jl`)
and JAX has `asdex`; PyTorch does not.

**Status: pre-alpha, not usable yet.** Only the pattern-storage and coloring core
is implemented. There is no tracing frontend and no operator rule set, so there
is no way to obtain a pattern from a PyTorch function yet. See
[dev_log.md](dev_log.md) for progress.

## What works today

```python
import numpy as np
from jacolor import color_cols
from jacolor import _boolcsr as bc

P = bc.from_dense([[1, 1, 0],
                   [0, 1, 1]])       # a Jacobian sparsity pattern
c = color_cols(P)
c.n_colors, c.lower_bound, c.colors  # (2, 2, array([0, 1, 0]))
```

Columns 0 and 2 share color 0 because no row contains both; the Jacobian of this
pattern needs 2 forward-mode passes instead of 3. `lower_bound` is the densest
row of `P`, which no coloring can beat.

`color_rows` is the reverse-mode equivalent.

## Design notes

- **Patterns are boolean CSR, always.** SciPy's sparsetools accumulate `bool`
  with logical OR; integer dtypes wrap, so 256 coincident contributions to one
  entry sum to 0 and the entry disappears. A missing entry is a wrong Jacobian,
  so `_boolcsr.py` gates dtype on every call.
- **Coloring is distance-1 on `A = Pᵀ P`.** `A[c, c']` is set iff columns `c`
  and `c'` share a row, which makes distance-2 coloring of the bipartite pattern
  an ordinary greedy coloring. Orderings tried: natural, largest-first,
  smallest-last; the fewest colors wins.
- Incidence-degree ordering is **not** implemented — it is dynamic
  (it depends on which neighbours are already colored) and does not fit the
  static-permutation kernel. Deferred, not dropped.

## Install

```bash
pip install -e ".[dev]"        # tests
pip install -e ".[dev,fast]"   # + numba kernel for the greedy loop
```

Without `numba`, coloring above 10,000 columns warns and runs a pure-Python loop.

## Measured

One data point, not a benchmark. Tridiagonal pattern, n = 100,000 columns,
299,998 nonzeros: **3 colors** (equal to the lower bound) in **0.27 s** for all
three orderings combined, including the `PᵀP` product.
Apple M5, macOS 26.6.2, Python 3.13.9, NumPy 2.2.6, SciPy 1.16.3, numba 0.62.1.

## License

MIT
