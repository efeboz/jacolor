"""A short live demo of jacolor on the Broyden tridiagonal function.

    python demo.py                                            # this folder
    python -m case_studies.cs1_sparse_jacobians.demo          # repo root

Detects and colors the pattern, checks the values, compares speed with dense AD,
scales up, and shows the check catching a wrong pattern. Runs in a few seconds.
"""

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from case_studies.cs1_sparse_jacobians.run import DTYPE, analytic, broyden  # noqa: E402
from src import VerificationError, jacobian, pattern_from_pairs, prepare  # noqa: E402
from src.coloring import color_cols  # noqa: E402


def ms(stmt, **env):
    return 1e3 * Timer(stmt, globals=env, num_threads=1).blocked_autorange(min_run_time=0.1).median


def timed_prepare(x, **kw):
    t = time.perf_counter()
    p = prepare(broyden, x, **kw)
    return p, 1e3 * (time.perf_counter() - t)


def section(title):
    print(f"\n{title}")


def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    prepare(broyden, -torch.ones(8, dtype=DTYPE))  # warm up export and numba
    print("jacolor demo: Broyden tridiagonal f: R^n -> R^n, float64, CPU, 1 thread")

    section("1. detect and color  (n = 6)")
    x = -torch.ones(6, dtype=DTYPE)
    p, t = timed_prepare(x)
    s = p.summary()
    print(f"   pattern {s['nnz']} of {s['rows'] * s['cols']} entries, {s['colors']} colors "
          f"(lower bound {s['lower_bound']}), {s['mode']} mode, {t:.0f} ms")
    print(f"   column colors  {' '.join(map(str, p.coloring.colors))}")
    print("   J at x = -1:")
    for row in p.jacobian(x).to_dense().tolist():
        print("     " + " ".join(f"{v:3.0f}" if v else "  ." for v in row))

    section("2. correctness  (n = 2048, random x)")
    n = 2048
    x = 2 * torch.randn(n, generator=g, dtype=DTYPE)
    p, _ = timed_prepare(-torch.ones(n, dtype=DTYPE))  # prepared at one point, used at another
    J = p.jacobian(x)
    e_ref = (J.values() - analytic(x).values()).abs().max().item()
    e_rev = (J.to_dense() - torch.func.jacrev(broyden)(x)).abs().max().item()
    print(f"   max |J - analytic|  {e_ref:.1e}")
    print(f"   max |J - jacrev|    {e_rev:.1e}" if e_rev else "   max |J - jacrev|    0 (identical)")
    print(f"   verification        {p.status}")

    section(f"3. speed  (n = {n}, ms per Jacobian)")
    pr, _ = timed_prepare(x, mode="reverse")
    fwd, rev = torch.func.jacfwd(broyden), torch.func.jacrev(broyden)
    t_fwd, t_rev = ms("p.jacobian(x)", p=p, x=x), ms("p.jacobian(x)", p=pr, x=x)
    d_fwd, d_rev = ms("f(x)", f=fwd, x=x), ms("f(x)", f=rev, x=x)
    print(f"   jacolor  forward {t_fwd:.2f}   reverse {t_rev:.2f}")
    print(f"   dense    jacfwd  {d_fwd:.2f}   jacrev  {d_rev:.2f}")
    print(f"   best jacolor vs best dense: {min(d_fwd, d_rev) / min(t_fwd, t_rev):.1f}x faster")

    section("4. scale  (n = 100000)")
    n = 100_000
    x = 2 * torch.randn(n, generator=g, dtype=DTYPE)
    p, t = timed_prepare(x)
    J = p.jacobian(x)
    sparse = J.values().nbytes + J.indices().nbytes
    print(f"   prepare {t:.0f} ms once, then {ms('p.jacobian(x)', p=p, x=x):.1f} ms per Jacobian, "
          f"{p.n_colors} colors")
    print(f"   output {sparse / 2**20:.1f} MiB sparse, {n * n * 8 / 2**30:.1f} GiB if dense")

    section("5. safety  (pattern with the upper diagonal removed, n = 64)")
    n = 64
    x = 2 * torch.randn(n, generator=g, dtype=DTYPE)
    i = np.arange(n)
    wrong = color_cols(pattern_from_pairs(np.r_[i, i[1:]], np.r_[i, i[:-1]], (n, n)))
    try:
        jacobian(broyden, x, coloring=wrong)
        print("   not caught")
    except VerificationError as e:
        entry = re.search(r"entry \(\d+, \d+\)", str(e))
        print(f"   check on:  VerificationError at {entry.group() if entry else 'an entry'}")
    bad = jacobian(broyden, x, coloring=wrong, verify=False).to_dense()
    err = (bad - analytic(x).to_dense()).abs().max().item()
    print(f"   check off: returned silently, off by up to {err:.1f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
