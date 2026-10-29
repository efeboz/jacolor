"""Figures for case study 2, from the JSON and NPZ that run.py saved.

    python -m case_studies.cs2_neural_smoothing.plot /tmp/jacolor-cs2/results.json --out /tmp/jacolor-cs2/figures

Writes reconstruction.{png,pdf} (truth, observations, interpolation and the
reconstruction for one trajectory, with the removed interval shaded),
runtime.{png,pdf} (total reconstruction time, and preparation against solve,
against sequence length) and storage.{png,pdf} (Jacobian storage against
length). Refuses results whose validation failed.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
STYLE = {"jacolor": ("jacolor (detects pattern)", BLUE, "-o"),
         "jacfwd": ("jacfwd (dense)", ORANGE, "-s"),
         "jacrev": ("jacrev (dense)", AQUA, "-^"),
         "sparse_fd": ("sparse FD (pattern given)", PURPLE, "-v"),
         "local": ("local-block AD reference", GRAY, ":D")}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def series(data, method, key):
    """(T, median, lower error, upper error) in ms: the median over trajectories of each
    trajectory's median over repeats, with bars to the fastest and slowest trajectory."""
    pts = [(int(T), a[method][key]) for T, a in data["aggregate"].items()
           if key in a.get(method, {})]
    T = np.array([t for t, _ in pts])
    med = np.array([s["median"] for _, s in pts]) * 1e3
    lo = med - np.array([s["min"] for _, s in pts]) * 1e3
    hi = np.array([s["max"] for _, s in pts]) * 1e3 - med
    return T, med, lo, hi


def reconstruction(data, npz, T, seed, out):
    k = f"T{T}_s{seed}_"
    if k + "truth" not in npz:
        sys.exit(f"no saved trajectory for T={T}, seed={seed}")
    Z, X0, R = npz[k + "truth"], npz[k + "init"], npz[k + "recon_jacolor"]
    obs, y, gap = npz[k + "obs"], npz[k + "y"], npz[k + "gap"]
    dt = data["model"]["config"]["dt"]
    t = np.arange(T) * dt
    fig, (ax_q, ax_v) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    for ax in (ax_q, ax_v):
        ax.axvspan(gap[0] * dt, (gap[1] - 1) * dt, color="#f3e9c8", lw=0, zorder=0)
    ax_q.plot(t, Z[:, 0], color=INK, lw=1.2, label="truth")
    ax_q.plot(t[obs], y, ".", color=MUTED, ms=3, label="noisy observation")
    ax_q.plot(t, X0[:, 0], color=ORANGE, lw=1, label="interpolation (initial guess)")
    ax_q.plot(t, R[:, 0], "--", color=BLUE, lw=1.4, label="reconstruction")
    ax_v.plot(t, Z[:, 1], color=INK, lw=1.2)
    ax_v.plot(t, X0[:, 1], color=ORANGE, lw=0.8, alpha=0.7)
    ax_v.plot(t, R[:, 1], "--", color=BLUE, lw=1.4)
    r = next(r for r in data["results"] if r["T"] == T and r["seed"] == seed)
    s, i = r["methods"]["jacolor"]["scores"], r["interpolation"]
    ax_q.set(ylabel="position q", title=f"Held-out trajectory, T = {T}, seed {seed}: position RMSE "
             f"{s['rmse_q']:.4f} (interpolation {i['rmse_q']:.4f}), "
             f"in the shaded gap {s['rmse_q_gap']:.4f} ({i['rmse_q_gap']:.4f})")
    ax_v.set(ylabel="velocity v (not observed)", xlabel="time",
             title=f"velocity RMSE {s['rmse_v']:.4f} (finite differences of the interpolation "
             f"{i['rmse_v']:.3f})")
    ax_v.set_ylim(Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5)
    fig.legend(*ax_q.get_legend_handles_labels(), loc="outside lower center", ncols=4)
    save(fig, out / "reconstruction")


def runtime(data, out):
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True, sharey=True)
    for m, (label, color, style) in STYLE.items():
        for ax, key in ((ax_t, "total_s"), (ax_s, "solve_s")):
            T, med, lo, hi = series(data, m, key)
            if len(T):
                ax.errorbar(T, med, yerr=[lo, hi], fmt=style, color=color, label=label,
                            capsize=2, elinewidth=1, markersize=4)
    T, med, lo, hi = series(data, "jacolor", "setup_s")
    ax_s.errorbar(T, med, yerr=[lo, hi], fmt="--o", color=BLUE, mfc="white",
                  label="jacolor prepare (in total)", capsize=2, elinewidth=1, markersize=4)
    T, med, _, _ = series(data, "jacolor", "jac_total_s")
    ax_s.plot(T, med, ":", color=BLUE, lw=1.2, label="jacolor Jacobian calls (in solve)")
    for ax, title in ((ax_t, "Total reconstruction (setup + solve)"),
                      (ax_s, "Solve after setup, and jacolor's preparation")):
        ax.set(xscale="log", yscale="log", xlabel="sequence length T", title=title)
        ax.set_xticks(sorted(int(t) for t in data["aggregate"]), labels=sorted(data["aggregate"], key=int))
        ax.minorticks_off()
    ax_t.set_ylabel("time [ms]")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    ax_s.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    env, st = data["environment"], data["settings"]
    fig.text(0.01, -0.12, f"{len(st['seeds'])} held-out trajectories per length, {st['repeats']} "
             "complete reconstructions each. Points are medians over trajectories of per-trajectory "
             "medians; bars span the fastest and slowest trajectory.\nSciPy least_squares trf/lsmr, "
             f"float64, CPU, 1 thread, {env['cpu']}, torch {env['torch']}, scipy {env['scipy']}. "
             f"Dense AD only for T <= {st['dense_max_t']}.", fontsize=7, color=MUTED)
    save(fig, out / "runtime")


def storage(data, out):
    rs = {}
    for r in data["results"]:  # one point per length: the first trajectory
        if "storage" in r:
            rs.setdefault(r["T"], r["storage"])
    T = np.array(sorted(rs))
    get = lambda k: np.array([rs[t][k] for t in T]) / 2 ** 20  # noqa: E731
    alloc = np.array([rs[t]["dense_allocated"] for t in T])
    fig, ax = plt.subplots(figsize=(4.6, 4), constrained_layout=True)
    ax.plot(T, get("coo_bytes"), "-o", color=BLUE, label="jacolor sparse COO")
    ax.plot(T, get("csr_bytes"), "-o", color=AQUA, label="SciPy CSR (to_scipy)")
    dense = get("dense_bytes")
    ax.plot(T, dense, "--", color=ORANGE, lw=1)
    ax.plot(T[alloc], dense[alloc], "s", color=ORANGE, label="dense, allocated")
    ax.plot(T[~alloc], dense[~alloc], "s", color=ORANGE, mfc="white", label="dense, calculated")
    ax.set(xscale="log", yscale="log", xlabel="sequence length T", ylabel="Jacobian output [MiB]",
           title="Jacobian storage (not peak memory)")
    ax.set_xticks(T, labels=[str(t) for t in T])
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    save(fig, out / "storage")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs2/figures"))
    ap.add_argument("--case", type=int, nargs=2, default=None, metavar=("T", "SEED"),
                    help="trajectory to draw (default: the second length, first seed)")
    args = ap.parse_args(argv)
    data = json.loads(args.results.read_text())
    if not data["validation"]["ok"]:
        sys.exit(f"refusing to plot: validation failed for (T, seed) "
                 f"{data['validation']['failed_cases']}; see the problems in {args.results}")
    npz = np.load(args.results.with_suffix(".npz"))
    lengths, seeds = data["settings"]["lengths"], data["settings"]["seeds"]
    T, seed = args.case or (lengths[min(1, len(lengths) - 1)], seeds[0])
    args.out.mkdir(parents=True, exist_ok=True)
    reconstruction(data, npz, T, seed, args.out)
    runtime(data, args.out)
    storage(data, args.out)


if __name__ == "__main__":
    main()
