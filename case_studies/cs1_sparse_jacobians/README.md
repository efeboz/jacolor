# Case study 1: Broyden tridiagonal Jacobian

This study measures how long jacolor takes to build the full Jacobian of the Broyden
tridiagonal function, `f_i = (3 - x_i) x_i - x_(i-1) - 2 x_(i+1) + 1`, and compares it with
dense automatic differentiation through `torch.func.jacfwd` and `torch.func.jacrev` as the
number of variables grows from 64 to 100,000. Each output depends on three inputs, so the
Jacobian has `3n - 2` structural entries, and jacolor has to find that pattern from the
function itself. The design and measurement procedure are in [CASE_STUDY_1.md](../../CASE_STUDY_1.md).

```bash
python -m case_studies.cs1_sparse_jacobians.run --out /tmp/jacolor-cs1/results.json
python -m case_studies.cs1_sparse_jacobians.validate --out /tmp/jacolor-cs1/validation.json
python -m case_studies.cs1_sparse_jacobians.plot /tmp/jacolor-cs1/results.json --out /tmp/jacolor-cs1/figures
```

For a quick live demonstration, `python demo.py` from this folder shows detection, coloring,
correctness, speed, scaling and the safety check in about three seconds.

The runner prints a row for each size as it completes, saves all measurements and the
environment to JSON, and exits with a non-zero status if any correctness check fails. The
validation script compares the Jacobian with independent references and wrong patterns (below).
The plotting script reads the benchmark file and writes `scaling` (runtime and output storage
against `n`) and `sparsity` (the detected pattern of a 12-variable instance, coloured by colour
group); `--slides` sizes the figure for a 16:9 slide. A ten-minute talk built on these results,
with slides and speaker notes, is in [talk/](talk/).

## Results

The measurements below were taken on an Apple M5 running macOS 27.0.1 with Python 3.13.9,
torch 2.13.0 and numba 0.62.1, at repository revision c0644f9. Every run used the CPU,
float64, a single thread, seed 0 and the input `x = -ones(n)`. Times are medians, in
milliseconds, from `blocked_autorange(min_run_time=0.2)`. The speedup is the faster of the two
dense methods divided by jacolor's repeated evaluation.

|      n | prepare | jacolor | jacfwd | jacrev | analytic | speedup | sparse output | dense output |
|-------:|--------:|--------:|-------:|-------:|---------:|--------:|--------------:|-------------:|
|     64 |    54.5 |   0.403 |  0.208 |  0.088 |    0.012 |   0.22× |       4.5 KiB |       32 KiB |
|    256 |    56.8 |   0.425 |  0.316 |  0.201 |    0.018 |   0.47× |        18 KiB |      512 KiB |
|   1024 |    57.0 |   0.510 |   2.28 |   2.25 |    0.043 |   4.41× |        72 KiB |        8 MiB |
|   2048 |    57.5 |   0.626 |   9.11 |   9.77 |    0.076 |   14.6× |       144 KiB |       32 MiB |
|  10000 |    62.9 |    1.46 |      – |      – |    0.343 |       – |       703 KiB |   763 MiB \* |
| 100000 |    88.1 |    11.5 |      – |      – |     3.87 |       – |       6.9 MiB |  74.5 GiB \* |

At every size, jacolor detected the expected `3n - 2` entries and coloured them with 3
colours, the lowest number possible for this pattern. Its values matched the analytic
diagonals, and where dense AD was run, they agreed with both dense Jacobians to
`rtol=1e-10, atol=1e-12`. Its built-in verification reported `ok` at every size. The
interquartile range of each timing was no more than 1.4% of its median.

`validate.py` checks correctness against references that do not depend on jacolor or on
autograd. One analysis per size, prepared at `x = -1`, is reused at seven points, including
`x = 1.5`, where the diagonal is exactly zero, and five random points. Up to `n = 100000`, the
detected pattern equals the analytic one with no missing or extra entries, and for `n <= 64`
it equals the nonzeros of a Jacobian derived symbolically by SymPy. The values differ from the
analytic formula and from SymPy by at most 1.8e-15, and from `jacfwd` and `jacrev` by exactly
zero. Central finite differences agree to 6e-10 relative error, which is their own rounding
error (`f` is quadratic, so they have no truncation error). Above `n = 2048` the comparison is
`J v` against a difference along random `v`. When the upper diagonal is removed from the
pattern, verification raises `VerificationError` naming the missing entry; without
verification the same call returns a Jacobian that is off by up to 2. A pattern with two extra
diagonals needs 5 colours and still gives exact values.

Dense AD is faster for small problems. At `n = 64`, `jacrev` takes 0.088 ms, against 0.403 ms
for jacolor, because jacolor's fixed cost of sparse assembly and a verification pass is larger
than a 64-column dense Jacobian. The two approaches cross between `n = 256` and `n = 1024`.
After that, dense AD time rises roughly with `n²` (it quadruples from 1024 to 2048), while
jacolor stays near-linear because the number of colours does not grow. At `n = 2048`, jacolor
is 14.6 times faster than the faster dense method. Preparing the analysis once takes 55–90 ms.
At `n = 2048`, that cost is recovered after about seven evaluations; at `n ≤ 256`, where dense
AD is faster, it is never recovered.

Dense AD was not run for `n ≥ 10000`, because its output alone would exceed the 64 MiB limit.
Starred entries are calculated as `8n²` bytes, not measured. The sparse result stores 24 bytes
per entry (one value and two indices), so at `n = 100000` it needs 6.9 MiB, against 74.5 GiB
for a dense matrix. A handwritten COO constructor that knows the formula is 3 to 34 times
faster than jacolor across the sweep, which is the cost of discovering the structure
automatically and verifying each result.

Preparation time is a single wall-clock measurement in a process that was warmed up first,
so interpreter start-up and numba compilation are not included. jacolor's evaluation times
include sparse assembly and verification. Storage figures are for the returned Jacobian only,
not peak process memory.
