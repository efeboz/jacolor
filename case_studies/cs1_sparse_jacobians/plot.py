"""Figures for case study 1, from the JSON that run.py saved.

    python -m case_studies.cs1_sparse_jacobians.plot /tmp/jacolor-cs1/results.json --out /tmp/jacolor-cs1/figures
    python plot.py /tmp/jacolor-cs1/results.json --out /tmp/jacolor-cs1/figures  # from this folder

Writes scaling.{png,pdf} (runtime and output storage against n) and
sparsity.{png,pdf} (the detected pattern on a small instance, by color).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def series(results, key):
    """(n, median, lower error, upper error) in ms, for sizes where key was measured."""
    rs = [r for r in results if key in r]
    n = np.array([r["n"] for r in rs])
    med = np.array([r[key]["median_s"] for r in rs]) * 1e3
    lo = med - np.array([r[key]["q1_s"] for r in rs]) * 1e3
    hi = np.array([r[key]["q3_s"] for r in rs]) * 1e3 - med
    return n, med, np.clip(lo, 0, None), np.clip(hi, 0, None)


def scaling(data, out, slides=False):
    res = data["results"]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.2, 3.9) if slides else (9, 4.2),
                                     constrained_layout=True)

    for key, label, color, style in (
            ("sparse", "jacolor, evaluation", BLUE, "-o"),
            ("jacfwd", "jacfwd (dense)", ORANGE, "-s"),
            ("jacrev", "jacrev (dense)", AQUA, "-^"),
            ("analytic", "analytic COO, by hand", GRAY, ":D")):
        n, med, lo, hi = series(res, key)
        ax_t.errorbar(n, med, yerr=[lo, hi], fmt=style, color=color, label=label,
                      capsize=2, elinewidth=1, markersize=4)
    prep = [(r["n"], r["prepare_s"] * 1e3) for r in res if "prepare_s" in r]
    if prep:
        n, t = zip(*prep)
        ax_t.plot(n, t, "--o", color=BLUE, mfc="white", label="jacolor, prepare (once)")
    ax_t.set(xscale="log", yscale="log", xlabel="n (inputs = outputs)", ylabel="time [ms]",
             title="Jacobian construction time")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)

    st = [(r["n"], r["storage"]) for r in res if "storage" in r]
    n = np.array([k for k, _ in st])
    coo = np.array([s["sparse_coo_bytes"] for _, s in st]) / 2 ** 20
    dense = np.array([s["dense_bytes"] for _, s in st]) / 2 ** 20
    alloc = np.array([s["dense_allocated"] for _, s in st])
    ax_m.plot(n, coo, "-o", color=BLUE, label="sparse COO")
    ax_m.plot(n, dense, "--", color=ORANGE, linewidth=1)
    ax_m.plot(n[alloc], dense[alloc], "s", color=ORANGE, label="dense, allocated")
    ax_m.plot(n[~alloc], dense[~alloc], "s", color=ORANGE, mfc="white",
              label="dense, calculated")
    limit = data["settings"]["dense_limit_mib"]
    ax_m.axhline(limit, color=MUTED, linewidth=0.8, linestyle=":")
    ax_m.annotate(f"{limit:g} MiB dense limit", (n.min(), limit), xytext=(0, 3),
                  textcoords="offset points", fontsize=7, color=MUTED)
    ax_m.set(xscale="log", yscale="log", xlabel="n (inputs = outputs)",
             ylabel="Jacobian output [MiB]", title="Output storage (not peak memory)")
    ax_m.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)

    env = data["environment"]
    if not slides:  # on a slide the frame says this
        fig.text(0.01, -0.06, f"Broyden tridiagonal, x = -1, float64, CPU, 1 thread. "
                 f"{env['cpu']}, torch {env['torch']}. Medians over blocked_autorange, "
                 "bars span the interquartile range. Dense AD omitted above the limit.",
                 fontsize=7, color=MUTED)
    save(fig, out / "scaling")


def sparsity(data, out):
    pv = data["validation"]["reuse_check"]
    n, k = pv["n"], pv["colors"]
    line = np.array(pv["cols"] if pv["axis"] == "cols" else pv["rows"])
    color = np.array(pv["color_of"])[line]
    palette = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(3.4, 3.8), constrained_layout=True)
    for c in range(k):
        m = color == c
        ax.scatter(np.array(pv["cols"])[m], np.array(pv["rows"])[m], marker="s", s=150,
                   color=palette[c % len(palette)], edgecolor="white", linewidth=1.5,
                   label=f"color {c}")
    ax.set(xlim=(-0.7, n - 0.3), ylim=(n - 0.3, -0.7), aspect="equal",
           xlabel="input j", ylabel="output i",
           title=f"Detected pattern, n = {n}: {len(pv['rows'])} entries, {k} colors")
    ax.grid(False)
    ax.set_xticks(range(0, n, 3))
    ax.set_yticks(range(0, n, 3))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=k,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, out / "sparsity")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs1/figures"))
    ap.add_argument("--slides", action="store_true",
                    help="larger text on a smaller canvas and no caption, for a 16:9 slide")
    args = ap.parse_args(argv)
    if args.slides:
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                             "legend.fontsize": 8.5})
    data = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    scaling(data, args.out, slides=args.slides)
    sparsity(data, args.out)


if __name__ == "__main__":
    main()
