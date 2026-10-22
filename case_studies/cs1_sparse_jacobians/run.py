"""Case study 1: Broyden tridiagonal Jacobian, jacolor against dense AD.

    python -m case_studies.cs1_sparse_jacobians.run --out /tmp/jacolor-cs1/results.json  # repo root
    python run.py --out /tmp/jacolor-cs1/results.json                                    # this folder

Prints one row per size as it finishes and writes every measurement, check and
the environment to a JSON file that plot.py reads. See CASE_STUDY_1.md.

Columns: prep is prepare() in a warmed process, 1st the first jacobian() (it
builds cached indices). jacolor, jacfwd, jacrev and ref (handwritten analytic
COO) are medians in ms. jacolor includes assembly and verification. speedup is
the faster dense method over jacolor. COO and dense are the size of the
Jacobian output, * where dense was calculated rather than allocated.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.utils.benchmark import Timer

# Run as a file (python run.py), only this directory is on the path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import TraceUnchecked, VerificationInconclusive, prepare  # noqa: E402
from src import __version__ as jacolor_version  # noqa: E402

DTYPE = torch.float64
SIZES = [64, 256, 1024, 2048, 10000, 100000]
RTOL, ATOL = 1e-10, 1e-12
MIB = 2 ** 20

# torch.export prints a FutureWarning from its own internals on some versions.
# It says nothing about this benchmark. The library's warnings become errors.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("error", category=TraceUnchecked)
warnings.filterwarnings("error", category=VerificationInconclusive)


def broyden(x):
    """f_i = (3 - x_i) x_i - x_(i-1) - 2 x_(i+1) + 1, with x_0 = x_(n+1) = 0."""
    z = x.new_zeros(1)
    left = torch.cat([z, x[:-1]])
    right = torch.cat([x[1:], z])
    return (3 - x) * x - left - 2 * right + 1


def analytic(x):
    """The handwritten reference: three diagonals as a coalesced sparse COO tensor."""
    n = x.numel()
    i = torch.arange(n)
    rows = torch.cat([i, i[1:], i[:-1]])
    cols = torch.cat([i, i[:-1], i[1:]])
    vals = torch.cat([3 - 2 * x, x.new_full((n - 1,), -1.0), x.new_full((n - 1,), -2.0)])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n),
                                   check_invariants=False).coalesce()


def check_sparse(J, x):
    """Coordinates equal and values close to the analytic diagonals, without densifying."""
    R = analytic(x)
    J = J.coalesce()
    if J.indices().shape != R.indices().shape or not torch.equal(J.indices(), R.indices()):
        return f"pattern differs: {J._nnz()} entries against {R._nnz()} expected"
    if not torch.allclose(J.values(), R.values(), rtol=RTOL, atol=ATOL):
        err = (J.values() - R.values()).abs().max().item()
        return f"values differ from analytic by up to {err:.3g}"
    return None


def coo_bytes(J):
    return J.values().nbytes + J.indices().nbytes


def timed(stmt, min_run_time, **env):
    m = Timer(stmt=stmt, globals=env, num_threads=1).blocked_autorange(min_run_time=min_run_time)
    return {"median_s": m.median, "iqr_s": m.iqr,
            "q1_s": float(np.percentile(m.times, 25)), "q3_s": float(np.percentile(m.times, 75)),
            "samples": len(m.times), "per_sample": m.number_per_run}


def environment(seed):
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True,
                                  cwd=Path(__file__).resolve().parents[2]).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    cpu = sh("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else None
    if cpu is None and Path("/proc/cpuinfo").exists():
        cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("model name")), None)
    try:
        import numba
        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
        "jacolor": jacolor_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "dtype": str(DTYPE),
        "device": "cpu",
        "seed": seed,
        "git_revision": sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain", "--", "src")),
    }


def reuse_check(seed, n=12):
    """One analysis evaluated at two different points, each checked against the reference.

    Also returns the pattern and its coloring, which plot.py draws as the preview.
    """
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, generator=g, dtype=DTYPE)
    x1 = torch.randn(n, generator=g, dtype=DTYPE)
    p = prepare(broyden, x0, mode="auto", verify=True)
    problems = []
    for name, x in (("x0", x0), ("x1", x1)):
        bad = check_sparse(p.jacobian(x), x)
        if bad or p.status != "ok":
            problems.append(f"{name}: {bad or 'verification ' + str(p.status)}")
    P = p.pattern
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr)).tolist()
    cols = P.indices.astype(int).tolist()
    return {"n": n, "ok": not problems, "problems": problems, "mode": p.mode,
            "colors": p.n_colors, "axis": p.coloring.axis,
            "color_of": p.coloring.colors.astype(int).tolist(),
            "rows": rows, "cols": cols}


def run_size(n, args):
    x = -torch.ones(n, dtype=DTYPE)
    r = {"n": n, "rows": n, "cols": n, "nnz_expected": 3 * n - 2, "problems": []}

    t = time.perf_counter()
    try:
        p = prepare(broyden, x, mode="auto", verify=True)
    except (TraceUnchecked, VerificationInconclusive) as e:
        r["problems"].append(f"prepare: {type(e).__name__}: {e}")
        return r
    r["prepare_s"] = time.perf_counter() - t

    t = time.perf_counter()
    J = p.jacobian(x)
    r["first_eval_s"] = time.perf_counter() - t

    s = p.summary()
    r.update(mode=s["mode"], nnz=s["nnz"], colors=s["colors"], lower_bound=s["lower_bound"],
             reason=s["reason"], status_first=p.status)
    bad = check_sparse(J, x)
    if bad:
        r["problems"].append(f"analytic: {bad}")

    try:
        r["sparse"] = timed("p.jacobian(x)", args.min_run_time, p=p, x=x)
    except (TraceUnchecked, VerificationInconclusive) as e:
        r["problems"].append(f"timed evaluation: {type(e).__name__}: {e}")
    r["status"] = p.status
    if p.status != "ok":
        r["problems"].append(f"verification status {p.status!r}")
    r["analytic"] = timed("analytic(x)", args.min_run_time, analytic=analytic, x=x)

    dense_bytes = n * n * x.element_size()
    r["storage"] = {"sparse_coo_bytes": coo_bytes(J), "dense_bytes": dense_bytes,
                    "dense_allocated": False}
    if dense_bytes > args.dense_limit_mib * MIB:
        r["dense_skipped"] = (f"output {dense_bytes / MIB:.0f} MiB exceeds the "
                              f"{args.dense_limit_mib} MiB limit")
    else:
        Jd = J.to_dense()
        for name, transform in (("jacfwd", torch.func.jacfwd), ("jacrev", torch.func.jacrev)):
            fn = transform(broyden)
            D = fn(x)
            if not torch.allclose(Jd, D, rtol=RTOL, atol=ATOL):
                r["problems"].append(f"{name}: differs from jacolor by "
                                     f"{(Jd - D).abs().max().item():.3g}")
            del D
            r[name] = timed("fn(x)", args.min_run_time, fn=fn, x=x)
        r["storage"]["dense_allocated"] = True
        best = min(r["jacfwd"]["median_s"], r["jacrev"]["median_s"])
        if "sparse" in r:
            r["speedup_vs_best_dense"] = best / r["sparse"]["median_s"]
    return r


def ms(s):
    if s is None:
        return "-"
    v = 1e3 * s
    return f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 100 else f"{v:.0f}"


def size(b):
    for unit, k in (("G", 2 ** 30), ("M", MIB), ("K", 2 ** 10)):
        if b >= k:
            return f"{b / k:.1f}{unit}"
    return f"{b}B"


HEAD = (f"{'n':>7} {'nnz':>7} {'col':>3} | {'prep':>6} {'1st':>6} | "
        f"{'jacolor':>7} {'IQR%':>5} {'jacfwd':>7} {'jacrev':>7} {'ref':>6} {'speedup':>7} | "
        f"{'COO':>6} {'dense':>7} | check")


def row(r):
    sp = r.get("sparse", {})
    fwd, rev = r.get("jacfwd", {}), r.get("jacrev", {})
    su = r.get("speedup_vs_best_dense")
    st = r.get("storage")
    iqr = f"{100 * sp['iqr_s'] / sp['median_s']:.1f}" if sp else "-"
    dense = size(st["dense_bytes"]) + ("" if st["dense_allocated"] else "*") if st else "-"
    return (f"{r['n']:>7} {r.get('nnz', '-'):>7} {r.get('colors', '-'):>3} | "
            f"{ms(r.get('prepare_s')):>6} {ms(r.get('first_eval_s')):>6} | "
            f"{ms(sp.get('median_s')):>7} {iqr:>5} "
            f"{ms(fwd.get('median_s')):>7} {ms(rev.get('median_s')):>7} "
            f"{ms(r.get('analytic', {}).get('median_s')):>6} "
            f"{(f'{su:.2f}x' if su else 'skip'):>7} | "
            f"{size(st['sparse_coo_bytes']) if st else '-':>6} {dense:>7} | "
            f"{'ok' if not r['problems'] else 'FAIL'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs1/results.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-run-time", type=float, default=0.2,
                    help="seconds per blocked_autorange measurement")
    ap.add_argument("--dense-limit-mib", type=float, default=64,
                    help="skip dense AD when its n*n*8 output exceeds this")
    args = ap.parse_args(argv)
    if min(args.sizes) < 3:
        ap.error("sizes must be at least 3")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    env = environment(args.seed)

    # Warm the process: imports, export machinery and the numba kernel compile here.
    prepare(broyden, -torch.ones(8, dtype=DTYPE)).jacobian(-torch.ones(8, dtype=DTYPE))
    reuse = reuse_check(args.seed)

    print(f"Broyden tridiagonal, {env['dtype']}, CPU, 1 thread | torch {env['torch']}, "
          f"numba {env['numba']} | {env['cpu']}")
    print("times in ms, storage in bytes; column definitions: --help\n")
    print(HEAD)
    print("-" * len(HEAD))

    results = []
    for n in args.sizes:
        r = run_size(n, args)
        results.append(r)
        print(row(r), flush=True)

    failed = [r for r in results if r["problems"]]
    dense_wins = [r["n"] for r in results if r.get("speedup_vs_best_dense", 2) < 1]
    skipped = [r["n"] for r in results if "dense_skipped" in r]
    print("-" * len(HEAD))
    print(f"dense AD faster at n = {', '.join(map(str, dense_wins)) or 'none'}"
          + (f"; not run at n = {', '.join(map(str, skipped))} "
             f"(output over {args.dense_limit_mib:g} MiB)" if skipped else ""))
    if not reuse["ok"]:
        print("FAIL reuse check: " + "; ".join(reuse["problems"]))
    for r in failed:
        print(f"FAIL n={r['n']}: " + "; ".join(r["problems"]))

    out = {"case_study": "cs1_broyden_tridiagonal", "environment": env,
           "settings": {"sizes": args.sizes, "min_run_time_s": args.min_run_time,
                        "dense_limit_mib": args.dense_limit_mib, "rtol": RTOL, "atol": ATOL,
                        "input": "x = -ones(n)", "prepare": "mode='auto', verify=True, "
                        "refine=False, chunk=None"},
           "validation": {"ok": reuse["ok"] and not failed, "reuse_check": reuse,
                          "failed_sizes": [r["n"] for r in failed]},
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"validation {'passed' if out['validation']['ok'] else 'FAILED'} "
          f"(reuse check + {len(results)} sizes); results in {args.out}")
    return 0 if out["validation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
