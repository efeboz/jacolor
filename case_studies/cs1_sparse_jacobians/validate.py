"""Correctness of jacolor's Broyden Jacobian against independent references.

    python -m case_studies.cs1_sparse_jacobians.validate --out /tmp/jacolor-cs1/validation.json
    python validate.py --out /tmp/jacolor-cs1/validation.json                 # this folder

For each size, one analysis is prepared at x = -1 and reused at every point:
x = -1, x = 1.5 (where the diagonal is exactly zero) and five random points.
Each Jacobian is compared with

  analytic    the hand-derived diagonals 3 - 2 x_i, -1, -2 (every size)
  sympy       a symbolic Jacobian of f, derived by SymPy (n <= 64)
  jacfwd/rev  dense autograd (n <= 2048)
  central FD  (f(x + h e_j) - f(x - h e_j)) / 2h for every column (n <= 2048);
              above that, J v against the same difference along random v

f is quadratic, so central differences have no truncation error and differ from
the true Jacobian only by rounding, about eps |f| / h. Two negative controls
check that a wrong pattern is caught rather than returned.
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import sympy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from case_studies.cs1_sparse_jacobians.run import DTYPE, analytic, broyden  # noqa: E402
from src import VerificationError, jacobian, pattern_from_pairs, prepare  # noqa: E402
from src.coloring import color_cols  # noqa: E402

DENSE_SIZES = [3, 8, 64, 256, 1024, 2048]
LARGE_SIZES = [10000, 100000]
SYMPY_MAX = 64
RANDOM_POINTS = 5
EXACT = dict(rtol=1e-10, atol=1e-12)  # references that are exact up to rounding
H = float(torch.finfo(DTYPE).eps) ** (1 / 3)  # central-difference step
FD_TOL = 1e-7  # bound on |J - J_fd| / max(1, |J|); rounding alone is ~1e-9 here


def points(n, seed):
    g = torch.Generator().manual_seed(seed + n)
    pts = [("x=-1", -torch.ones(n, dtype=DTYPE)), ("x=1.5", torch.full((n,), 1.5, dtype=DTYPE))]
    pts += [(f"rand{k}", 2 * torch.randn(n, generator=g, dtype=DTYPE)) for k in range(RANDOM_POINTS)]
    return pts


def sympy_jacobian(n):
    """Symbolic Jacobian of f, built from its definition, and its nonzero positions."""
    x = sympy.symbols(f"x0:{n}")
    f = [(3 - x[i]) * x[i] - (x[i - 1] if i > 0 else 0) - 2 * (x[i + 1] if i < n - 1 else 0) + 1
         for i in range(n)]
    J = sympy.Matrix(f).jacobian(x)
    pattern = {(i, j) for i in range(n) for j in range(n) if J[i, j] != 0}
    fn = sympy.lambdify([x], J, "numpy")
    return (lambda v: torch.tensor(np.asarray(fn(v.numpy()), dtype=np.float64))), pattern


def central_fd(x):
    """Full central-difference Jacobian, one pair of evaluations per column."""
    E = H * torch.eye(x.numel(), dtype=x.dtype)
    fb = torch.func.vmap(broyden)
    return ((fb(x + E) - fb(x - E)) / (2 * H)).T


def pattern_pairs(J):
    return set(map(tuple, J.indices().T.tolist()))


def check_size(n, seed, sym):
    out = {"n": n, "nnz_expected": 3 * n - 2, "problems": []}
    x0 = -torch.ones(n, dtype=DTYPE)
    p = prepare(broyden, x0, mode="auto", verify=True)
    out.update(colors=p.n_colors, mode=p.mode)

    # Pattern: detected against the analytic one, and against SymPy's nonzeros.
    got = pattern_pairs(p.jacobian(x0))
    want = pattern_pairs(analytic(x0))
    out["pattern"] = {"detected": len(got), "missing": len(want - got), "extra": len(got - want)}
    if want != got:
        out["problems"].append("detected pattern differs from analytic")
    if sym is not None:
        out["pattern"]["sympy_agrees"] = got == sym[1]
        if got != sym[1]:
            out["problems"].append("detected pattern differs from SymPy")

    err = {k: 0.0 for k in ("analytic", "sympy", "jacfwd", "jacrev", "fd")}
    dense = n <= DENSE_SIZES[-1]
    jf, jr = torch.func.jacfwd(broyden), torch.func.jacrev(broyden)
    for name, x in points(n, seed):
        J = p.jacobian(x)
        if p.status != "ok":
            out["problems"].append(f"{name}: verification {p.status}")
        R = analytic(x)
        if not torch.equal(J.indices(), R.indices()):
            out["problems"].append(f"{name}: pattern differs from analytic")
            continue
        err["analytic"] = max(err["analytic"], (J.values() - R.values()).abs().max().item())
        if not torch.allclose(J.values(), R.values(), **EXACT):
            out["problems"].append(f"{name}: values differ from analytic")
        if not dense:
            # J v against a central difference along v, never forming J densely.
            g = torch.Generator().manual_seed(seed + 7 * n)
            for _ in range(3):
                v = torch.randn(n, generator=g, dtype=DTYPE)
                fd = (broyden(x + H * v) - broyden(x - H * v)) / (2 * H)
                e = ((J @ v) - fd).abs().max().item() / max(1.0, fd.abs().max().item())
                err["fd"] = max(err["fd"], e)
            continue
        D = J.to_dense()
        refs = {"jacfwd": jf(x), "jacrev": jr(x)}
        if sym is not None:
            refs["sympy"] = sym[0](x)
        for k, ref in refs.items():
            err[k] = max(err[k], (D - ref).abs().max().item())
            if not torch.allclose(D, ref, **EXACT):
                out["problems"].append(f"{name}: differs from {k}")
        fd = central_fd(x)
        err["fd"] = max(err["fd"], ((D - fd).abs() / D.abs().clamp(min=1.0)).max().item())
    if err["fd"] > FD_TOL:
        out["problems"].append(f"central FD error {err['fd']:.2g} over {FD_TOL:g}")

    out["errors"] = {k: v for k, v in err.items()
                     if (k != "sympy" or sym is not None) and (dense or k in ("analytic", "fd"))}
    out["points"] = len(points(n, seed))
    out["fd_kind"] = "full" if dense else "directional"
    return out


def negative_controls():
    """A pattern missing the upper diagonal, and one holding two extra diagonals."""
    n = 64
    x = 2 * torch.randn(n, generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    i = np.arange(n)
    R = analytic(x).to_dense()
    res = {}

    lower = pattern_from_pairs(np.r_[i, i[1:]], np.r_[i, i[:-1]], (n, n))
    c = color_cols(lower)
    res["missing_upper_colors"] = c.n_colors
    try:
        jacobian(broyden, x, coloring=c, verify=True)
        res["missing_upper_verify"] = "not caught"
    except VerificationError as e:
        entry = re.search(r"entry \(\d+, \d+\)", str(e))
        res["missing_upper_verify"] = "VerificationError" + (f" at {entry.group()}" if entry else "")
    J = jacobian(broyden, x, coloring=c, verify=False).to_dense()
    res["missing_upper_silent_error"] = (J - R).abs().max().item()

    rows = np.r_[i, i[1:], i[:-1], i[2:], i[:-2]]
    cols = np.r_[i, i[:-1], i[1:], i[:-2], i[2:]]
    p = prepare(broyden, x, pattern=pattern_from_pairs(rows, cols, (n, n)))
    J = p.jacobian(x).to_dense()
    res["superset_colors"] = p.n_colors
    res["superset_status"] = p.status
    res["superset_error"] = (J - R).abs().max().item()
    return res


def fmt(e):
    return "0" if e == 0 else f"{e:.1e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs1/validation.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=DENSE_SIZES + LARGE_SIZES)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    print(f"Broyden tridiagonal, float64. One analysis per n, reused at {2 + RANDOM_POINTS} "
          f"points. Max |J - ref| (FD: relative), h = {H:.1e}\n")
    head = (f"{'n':>7} {'col':>3} | {'pattern':>16} | {'analytic':>8} {'sympy':>8} "
            f"{'jacfwd':>8} {'jacrev':>8} {'centralFD':>9} | check")
    print(head)
    print("-" * len(head))
    rows = []
    for n in args.sizes:
        sym = sympy_jacobian(n) if n <= SYMPY_MAX else None
        r = check_size(n, args.seed, sym)
        rows.append(r)
        e, pt = r["errors"], r["pattern"]
        pat = (f"{pt['detected']} = 3n-2" if not pt["missing"] and not pt["extra"]
               else f"-{pt['missing']} +{pt['extra']}")
        print(f"{n:>7} {r['colors']:>3} | {pat:>16} | {fmt(e['analytic']):>8} "
              f"{fmt(e['sympy']) if 'sympy' in e else '-':>8} "
              f"{fmt(e['jacfwd']) if 'jacfwd' in e else '-':>8} "
              f"{fmt(e['jacrev']) if 'jacrev' in e else '-':>8} "
              f"{fmt(e['fd']) + ('' if r['fd_kind'] == 'full' else '*'):>9} | "
              f"{'ok' if not r['problems'] else 'FAIL'}", flush=True)
    print("-" * len(head))
    print("pattern: detected entries, checked against the analytic pattern (and SymPy's "
          "nonzeros for n <= 64).\n*: directional central difference along 3 random v.")

    neg = negative_controls()
    print(f"\nwrong pattern, upper diagonal missing ({neg['missing_upper_colors']} colors): "
          f"{neg['missing_upper_verify']}; with verify=False it returns silently, max error "
          f"{neg['missing_upper_silent_error']:.2f}")
    print(f"superset pattern, two extra diagonals: {neg['superset_colors']} colors, "
          f"status {neg['superset_status']}, max error {fmt(neg['superset_error'])}")

    failed = [r["n"] for r in rows if r["problems"]]
    neg_ok = (neg["missing_upper_verify"].startswith("VerificationError")
              and neg["superset_error"] < 1e-12 and neg["superset_status"] == "ok")
    ok = not failed and neg_ok
    for r in rows:
        for msg in r["problems"]:
            print(f"FAIL n={r['n']}: {msg}")
    out = {"settings": {"seed": args.seed, "h": H, "fd_tol": FD_TOL, "exact": EXACT,
                        "random_points": RANDOM_POINTS, "sympy_max": SYMPY_MAX,
                        "torch": torch.__version__, "sympy": sympy.__version__},
           "sizes": rows, "negative_controls": neg, "ok": ok,
           "seconds": time.perf_counter() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nvalidation {'passed' if ok else 'FAILED'} in {out['seconds']:.0f} s; "
          f"results in {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
