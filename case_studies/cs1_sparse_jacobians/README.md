# Case study 1: Broyden tridiagonal Jacobian

How does automatic sparse AD (jacolor) compare with dense AD (`torch.func.jacfwd`/`jacrev`)
as the problem grows, and where does its overhead win out? Design: [CASE_STUDY_1.md](../../CASE_STUDY_1.md).

```bash
python -m case_studies.cs1_sparse_jacobians.run --out /tmp/jacolor-cs1/results.json
python -m case_studies.cs1_sparse_jacobians.plot /tmp/jacolor-cs1/results.json --out /tmp/jacolor-cs1/figures
```

`run.py` prints one row per size and exits non-zero if any check fails. `plot.py` writes
`scaling` (time and output storage against n) and `sparsity` (detected pattern by color).

## Results

Apple M5, macOS 27.0.1, repo c0644f9, Python 3.13.9, torch 2.13.0, numba 0.62.1; CPU, float64, one thread,
seed 0, `x = -ones(n)`. Times in ms, median of `blocked_autorange(min_run_time=0.2)`.

|      n | colors | prepare | jacolor | jacfwd | jacrev | analytic ref | speedup | COO    | dense     |
|-------:|-------:|--------:|--------:|-------:|-------:|-------------:|--------:|-------:|----------:|
|     64 |      3 |    54.5 |   0.403 |  0.208 |  0.088 |        0.012 |   0.22× | 4.5 KiB | 32 KiB   |
|    256 |      3 |    56.8 |   0.425 |  0.316 |  0.201 |        0.018 |   0.47× | 18 KiB | 512 KiB   |
|   1024 |      3 |    57.0 |   0.510 |   2.28 |   2.25 |        0.043 |   4.41× | 72 KiB | 8 MiB     |
|   2048 |      3 |    57.5 |   0.626 |   9.11 |   9.77 |        0.076 |  14.6×  | 144 KiB | 32 MiB   |
|  10000 |      3 |    62.9 |    1.46 |      – |      – |        0.343 |       – | 703 KiB | 763 MiB* |
| 100000 |      3 |    88.1 |    11.5 |      – |      – |         3.87 |       – | 6.9 MiB | 74.5 GiB* |

Every size passed: pattern and values match the analytic diagonals, the dense Jacobians agree
(`rtol=1e-10, atol=1e-12`), and verification reported `ok`. Timing IQR was at most 1.4 % of the median.

- **Small n, dense AD wins.** At n ≤ 256, jacolor's assembly and verification pass cost more
  than a dense Jacobian of 64–256 columns. jacrev is about 4.6× faster at n = 64.
- **Crossover between 256 and 1024.** The color count stays at 3 (the lower bound), so
  jacolor grows roughly linearly in n while dense AD grows with n². By n = 2048 it is
  14.6× faster than the faster dense mode.
- **Preparation is a one-time ~55–90 ms.** It pays for itself against dense AD after about
  7 evaluations at n = 2048. It never does at n ≤ 256.
- **Storage.** COO holds 3n − 2 values and their indices (24 B per entry), against 8n² B for dense.
  Dense AD was not run above 64 MiB of output (n ≥ 10000). Those sizes appear only as calculated storage (*).
- **Against the handwritten reference** jacolor is 3–34× slower. That gap is the cost of
  generic AD passes, scatter and verification over a hand-built COO.

Notes: preparation is a single wall-clock measurement in a warmed process (interpreter start-up
and numba compilation excluded). jacolor timings include sparse assembly and verification.
Storage is the size of the returned Jacobian, not peak process memory.
