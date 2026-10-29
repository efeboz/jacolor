# Case study 2: trajectory reconstruction with a learned dynamics model

This study uses jacolor inside a complete inference task and asks whether it makes that task
faster. A small MLP is trained once on a nonlinear oscillator and then frozen. Given noisy,
incomplete position measurements of a held-out trajectory, `scipy.optimize.least_squares`
recovers every position and velocity. The residual combines the observations, the learned
transition `z_(t+1) ≈ g(z_t)` and a weak prior on `z_0`. The unknowns are the `2T` states and
the network weights stay fixed. The solver, tolerances, weights and starting point are the
same in every run. Only the source of the Jacobian changes:

| method | Jacobian | setup charged to it |
|---|---|---|
| jacolor | `prepare(residual, x0, mode="auto", verify=True)`, then `to_scipy(p.jacobian(x))` | `prepare` (tracing, detection, coloring) |
| jacfwd, jacrev | dense `torch.func`, only for `T <= 256` | building the transform |
| sparse FD | SciPy `jac="2-point"` with `jac_sparsity`, **the pattern given in advance** | building the pattern |
| local reference | each 2×2 transition Jacobian by `vmap(jacrev(g))`, put into the known layout | none |

jacolor finds the pattern on its own. Sparse FD and the local reference are given it. The
local reference is the structure-aware AD baseline, playing the role of CS1's handwritten
Jacobian, and every method's solution is checked against its solution. The design is in
[CASE_STUDY_2.md](../../CASE_STUDY_2.md).

## Running it

From the repository root, with the library's dependencies, its numba extra and Matplotlib
installed (the library is imported from `src`, as in CS1, so it needs no install):

```bash
python -m pip install "torch>=2.6" "numpy>=1.24" "scipy>=1.14" "numba>=0.59" matplotlib

python -m case_studies.cs2_neural_smoothing.train --out /tmp/jacolor-cs2/model.pt
# One short held-out trajectory first.
python -m case_studies.cs2_neural_smoothing.run --model /tmp/jacolor-cs2/model.pt --lengths 64 --seeds 0 --repeats 1 --out /tmp/jacolor-cs2/smoke.json
# Default sweep: T = 64, 256, 1024, 4096, three test trajectories, three repeats (about 40 s).
python -m case_studies.cs2_neural_smoothing.run --model /tmp/jacolor-cs2/model.pt --out /tmp/jacolor-cs2/results.json
python -m case_studies.cs2_neural_smoothing.plot /tmp/jacolor-cs2/results.json --out /tmp/jacolor-cs2/figures
```

To reproduce the snapshot below from the saved checkpoint, pass
`--model case_studies/cs2_neural_smoothing/res/model.pt`. `run.py` prints one row per trajectory. It exits non-zero if any
check fails, and it saves the JSON together with an NPZ of the truth, observations,
initialization and reconstructions. `plot.py` will not draw results whose validation failed.

| file | contents |
|---|---|
| `problem.py` | simulator, model, observation pattern, residual, initialization, expected pattern, local-block reference |
| `train.py` | one-step training, validation against baselines, checkpoint |
| `run.py` | correctness checks, solver comparisons, timings, saved results |
| `plot.py` | `reconstruction`, `runtime` and `storage` figures |
| `res/` | the snapshot: `model.pt` (+ `model.json` summary), `results.json`, `results.npz`, `figures/` |

## Snapshot

All numbers here come from `res/results.json`. That run was on an Apple M5 under macOS 27.2
with Python 3.13.9, torch 2.13.0, SciPy 1.16.3, NumPy 2.2.6 and numba 0.62.1. The
repository was at e89e9ee, with the case-study files still uncommitted
(`git_dirty: true`). It used CPU, float64 and one thread (torch, and `OMP`/`OPENBLAS`/`MKL`/
`VECLIB` set to 1). The checkpoint's sha256 begins `fa2b0060a595`.

### The learned model

Training used 64 trajectories, validation 16 different ones, and both used separate random
streams. Each trajectory has 256 states with `dt = 0.05`, from five RK4 steps per interval.
The `2 → 32 → 2` tanh MLP predicts the increment, `g(z) = z + dt·net(z)`, and its
normalization uses training statistics only. Training used Adam with batch size 256 and
learning rate 1e-3. It ran on 16,320 one-step pairs, took 3.7 s and kept the best validation
checkpoint, from update 19,825.

| validation next-state RMSE (4,080 pairs) | q | v |
|---|---:|---:|
| learned model | 2.8e-5 | 4.8e-5 |
| linear transition, least squares on training pairs | 8.7e-5 | 3.5e-3 |
| persistence `z_next = z` | 3.1e-2 | 3.4e-2 |

The plan's limit of 2,000 updates was not enough. At that point the model was worse than the
linear fit in `q` (8.5e-4) and no better in `v` (3.5e-3), and the validation curve was still
falling. The limit was raised once, to 20,000, after trying 10,000 and 20,000 on validation
data. No other setting was changed, and the settings were fixed before the test sweep.
Because both validation RMSEs are below the 1e-3 floor, `sigma_dyn = [1e-3, 1e-3]`.
Training again on the same machine reproduces the weights bit for bit. The file hash still
changes, because the checkpoint also records the training time.

### Problem size and detection

About half the positions are observed, one interior run of 8 samples is removed, and the
first and last are kept. The observation noise is `sigma_obs = 0.05`. Ranges cover the three
test trajectories at each length.

|    T | observed K | rows × cols | structural entries | colors | jacolor COO | SciPy CSR | dense (8·rows·cols) |
|-----:|-----------:|------------:|-------------------:|-------:|------------:|----------:|--------------------:|
|   64 |      28–31 |   156–159 × 128 |        408–411 | 4 |  9.6 KiB |  7.6 KiB |  157 KiB |
|  256 |    110–132 |   622–644 × 512 |      1642–1664 | 4 |   39 KiB |   31 KiB |  2.5 MiB |
| 1024 |    476–495 | 2524–2543 × 2048 |     6616–6635 | 4 |  155 KiB |  123 KiB | 39 MiB \* |
| 4096 |  2040–2090 | 10232–10282 × 8192 | 26612–26662 | 4 |  624 KiB |  496 KiB | 639 MiB \* |

In every case, jacolor found exactly `6(T−1) + K + 2` entries, the count the residual's
definition gives, with none missing and none extra. It chose forward mode with 4 colors.
The lower bound was 3, and reverse mode needs at least 4. Storage is for the first
trajectory at each length, and \* marks dense sizes that were calculated, not allocated.
None of these figures is peak process memory.

### Correctness

All of these checks passed on all 12 trajectories before any timing was interpreted:

- The vectorized residual agrees with a one-term-at-a-time loop to 2.3e-13 (checked for
  `T <= 256`).
- The jacolor Jacobian at the starting point matches the local-block reference in its
  coordinates exactly and in its values within `rtol=1e-10, atol=1e-12`, at every length.
  For `T <= 256` it also matches dense `jacfwd` (difference 0) and `jacrev` (at most
  1.1e-13, for entries of size up to 1,000).
- Along three random directions, `J v` matches central differences of the whole residual to
  a relative error of at most 2.6e-11. The tolerance is 1e-6.
- A single prepared analysis, reused at the starting point, at a perturbed point and at the
  recovered trajectory, matched the local reference every time with status `ok`. No
  `TraceUnchecked` or `VerificationInconclusive` occurred.
- Every solve with every method ended with SciPy status 2 (the `ftol` test). None reached
  the 100-evaluation limit, and each repeat gave a bit-identical objective.
- Against the local reference solve, the largest relative objective difference was 3.5e-8
  (limit 1e-4). The largest difference in position RMSE was 9.3e-7 (limit 1e-3). At one
  trajectory (`T = 64`, seed 1), the methods took different iteration counts (21–28),
  because rounding-level Jacobian differences changed the trust-region path. They still
  ended at the same solution.

### Reconstruction quality

These are jacolor's reconstructions. The other methods' agree with them to within the
differences above. Ranges cover the three trajectories at each length.

|    T | position RMSE | at missing timestamps | in the 8-sample gap | velocity RMSE | interpolation: position | interpolation: velocity |
|-----:|---------:|---------:|---------:|---------:|---------:|---------:|
|   64 | 0.011–0.019 | 0.012–0.018 | 0.011–0.018 | 0.015–0.019 | 0.037–0.049 | 0.39–0.58 |
|  256 | 0.002–0.007 | 0.002–0.007 | 0.001–0.004 | 0.003–0.009 | 0.041–0.049 | 0.46–0.56 |
| 1024 | 0.003–0.007 | 0.003–0.006 | 0.001–0.007 | 0.003–0.007 | 0.043–0.044 | 0.49–0.54 |
| 4096 | 0.005–0.006 | 0.005–0.006 | 0.002–0.012 | 0.005–0.006 | 0.044–0.046 | 0.52–0.55 |

The reconstruction reduces position error 2–19× relative to the interpolation it starts
from, to 0.002–0.019, below the 0.05 observation noise. It recovers the velocity, which is
never observed, to 0.003–0.019, where finite differences of the interpolation give about
0.5. Inside the gap, it follows the true trajectory where linear interpolation cuts
straight across ([figures/reconstruction](res/figures/reconstruction.png), `T = 256`). These
results concern the learned model and the inference task. jacolor contributes only
computation, and every method reaches the same answer.

### Runtime

Times are in ms. Each entry is the median over the three trajectories of each trajectory's
median over three complete reconstructions, each run from a fresh solver. A total includes
the method's own setup, the solve, NumPy/torch conversion in the callbacks and, for jacolor,
conversion to SciPy CSR. jacolor's preparation is repeated in every total. Results are in
[figures/runtime](res/figures/runtime.png).

|    T | jacolor prepare | jacolor solve | **jacolor total** | jacfwd | jacrev | sparse FD | local reference |
|-----:|------:|------:|------:|------:|------:|------:|------:|
|   64 |  86.9 |  34.1 | **127** |  29.9 |  21.3 |  21.5 |  17.2 |
|  256 |  90.3 |  99.8 | **190** |   304 |   277 |  67.4 |  46.6 |
| 1024 |  94.6 |   110 | **205** |     – |     – |  98.0 |  66.5 |
| 4096 |   113 |   250 | **363** |     – |     – |   268 |   152 |

Speedup is the baseline's time divided by jacolor's, so values above 1 mean jacolor is
faster. It is shown only where both methods passed the quality checks, as a median with the
range across trajectories in parentheses:

|    T | total vs jacfwd | vs jacrev | vs sparse FD | vs local | solve only vs sparse FD | solve only vs local |
|-----:|------:|------:|------:|------:|------:|------:|
|   64 | 0.24 (0.15–0.28) | 0.17 (0.10–0.23) | 0.17 (0.11–0.25) | 0.14 | 0.60 | 0.48 |
|  256 | **1.58** (1.21–1.62) | **1.46** (1.13–1.53) | 0.35 (0.28–0.36) | 0.24 | 0.67 | 0.47 |
| 1024 | – | – | 0.45 (0.45–0.48) | 0.32 | 0.89 | 0.60 |
| 4096 | – | – | 0.70 (0.67–0.77) | 0.43 | **1.10** (1.03–1.11) | 0.65 |

The table below gives repeated Jacobian callbacks at the saved starting point. Values are
medians from `blocked_autorange(min_run_time=0.2)` in ms, and jacolor's includes assembly,
verification and `to_scipy`:

|    T | jacolor | local reference | jacfwd | jacrev | one residual evaluation |
|-----:|------:|------:|------:|------:|------:|
|   64 | 1.79 | 0.22 | 1.19 | 0.43 | 0.033 |
|  256 | 1.98 | 0.28 | 7.53 | 6.47 | 0.072 |
| 1024 | 2.58 | 0.47 |    – |    – | 0.206 |
| 4096 | 5.10 | 1.22 |    – |    – | 0.715 |

## What the measurements show

**jacolor does not speed up this task against the sparse baselines.** Each reconstruction
prepares once and runs 7–31 solver iterations. With that setup, jacolor's total time is
slower than sparse finite differences at every length (0.17× to 0.70×). It is also slower
than the local-block reference at every length (0.14× to 0.43×). The only end-to-end win is
over dense AD at `T = 256`, 1.46–1.58×. Even there, sparse FD is 2.8× faster than jacolor.

**At small T, setup is the largest cost.** Preparation takes 87–113 ms at every length,
which is 69% of jacolor's total at `T = 64`, 47% at 256 and 31% at 4096. At `T = 64`, the
entire solve by any other method takes less time than jacolor's preparation. Each of
jacolor's Jacobian calls also has a fixed cost of about 1.8 ms for assembly and its
verification pass. This makes it slower per call than dense `jacrev` (0.43 ms) at `T = 64`.

**At large T, the solver's own work is the largest cost.** Excluding Jacobians, a solve at
`T = 4096` spends about 125–135 ms on residual evaluations, the trust-region step and LSMR. This
holds for both jacolor (250 − 114 ms in Jacobian callbacks) and the local reference
(152 − 28 ms). That shared part is the same for every method, so a faster Jacobian can
reduce only the remainder.

**Sparse FD wins because the residual is cheap.** SciPy used 7 residual evaluations per
iteration: 1 for the residual and 6 for column groups, compared with jacolor's 4 colors. One
evaluation costs only 0.03–0.7 ms, so the 6 extra evaluations cost less than jacolor's
per-call overhead. By `T = 4096` they cost about the same: about 4.3 ms of residual
evaluations against 5.1 ms for jacolor. Excluding preparation, jacolor's solve is then
1.10× faster than sparse FD (1.03–1.11 across trajectories), and the advantage has grown
with T (0.60×, 0.67×, 0.89×, 1.10×). It still does not cover a 113 ms preparation within one
reconstruction. One preparation saves about 18 ms per solve against sparse FD, so it would
have to serve about six solves with the same structure (same `T` and observation pattern) to
break even. That figure is arithmetic from this snapshot and was not measured. The plan
also says not to extrapolate beyond `T = 4096`.

**The structure-aware reference beats jacolor at every size.** It knows the residual layout
and does no verification, and its Jacobian is 4–8× faster than jacolor's (1.22 against
5.10 ms at `T = 4096`). jacolor's cost is the price of discovering the pattern from the code
and checking every result. In return, it needs no hand-written layout, and that is what the
local reference and sparse FD's `jac_sparsity` both rely on.

**Dense AD is the baseline jacolor clearly beats.** At `T = 256`, a dense Jacobian costs
6.5–7.5 ms against 2.0 ms, which makes jacolor's solve about 3× faster and its total about
1.5× faster despite preparation. Dense AD was not run above `T = 256`, as the plan requires.
The output limit alone would have allowed `T = 1024`, at 39 MiB.

## Limitations

- There was one machine, one CPU thread and float64. The model is small, and the residual
  costs at most 0.7 ms, which favors finite differences. A more expensive residual would
  change that balance. The study does not test one, and the model was deliberately not
  enlarged to produce a speedup.
- Preparation was measured in a warmed process. In a pilot, the first `prepare` in a cold
  process took about 0.8 s, and that figure is not part of the snapshot.
- Each case has three repeats, and single slow repeats push the interquartile range to as
  much as 24% of the median for some totals (the JSON has every value). Medians are robust
  to one such outlier, and variation across trajectories is reported separately (ranges
  above).
- `sigma_dyn` is set by the 1e-3 floor, not by the model's validation error, which is 20–35×
  lower. The dynamics term therefore carries about 50 times the weight of each observation.
- The local reference uses the same autograd as jacolor. It checks detection, coloring and
  assembly, not autograd itself. The independent checks are the directional central
  differences and, for `T <= 256`, the dense Jacobians.
- No jacolor failure was found, so no reproduction was needed. On this torch version,
  `torch.export` prints a `FutureWarning` from its own internals, which run.py ignores,
  as CS1 does.
