"""Case study 2: trajectory reconstruction with a learned model, by derivative method.

    python -m case_studies.cs2_neural_smoothing.run --model /tmp/jacolor-cs2/model.pt --out /tmp/jacolor-cs2/results.json

For each sequence length and held-out trajectory, checks the residual and its
Jacobian, then solves the same least-squares problem with the same SciPy solver
and starting point, changing only where the Jacobian comes from:

  jacolor    prepare(residual, x0) once per reconstruction, to_scipy(p.jacobian(x))
  jacfwd     dense torch.func.jacfwd (T <= 256 and output <= 64 MiB)
  jacrev     dense torch.func.jacrev (same guard)
  sparse_fd  SciPy jac="2-point" with the expected pattern, which it is given
  local      2x2 blocks by vmap(jacrev(g)) assembled into the known layout: the
             structure-aware reference every method's solution is compared with

Prints one row per trajectory, times in ms. prep is jacolor's prepare() and 1st
its first Jacobian callback inside the solver, both medians over --repeats
reconstructions. jac and local are repeated Jacobian callbacks at x0 (jacolor
with to_scipy, and the local reference), medians over blocked_autorange. The
method columns are total reconstruction time: each method's own setup (prepare
for jacolor, building jac_sparsity for sparse_fd) plus the solve, median over
--repeats. nfev is jacolor's. q-RMSE is position error against the truth for
jacolor's reconstruction and for the interpolation it started from. Writes JSON
and an NPZ of trajectories beside it.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from torch.utils.benchmark import Timer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# CS1's environment record, and its warning filters: library warnings are errors.
from case_studies.cs1_sparse_jacobians.run import environment  # noqa: E402
from case_studies.cs2_neural_smoothing.problem import (  # noqa: E402
    DTYPE, GAP, KEEP, MU_PRIOR, SIGMA_OBS, SIGMA_PRIOR, expected_pattern, interpolate,
    load_model, local_reference, loop_residual, make_residual, shape, test_case)
from src import TraceUnchecked, VerificationInconclusive, prepare, to_scipy  # noqa: E402

LENGTHS = [64, 256, 1024, 4096]
SEEDS = [0, 1, 2]
REPEATS = 3
METHODS = ["jacolor", "jacfwd", "jacrev", "sparse_fd", "local"]
REFERENCE = "local"
SOLVER = dict(method="trf", tr_solver="lsmr", loss="linear", x_scale=1.0,
              ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=100)
RTOL, ATOL = 1e-10, 1e-12  # Jacobian values against AD references
DIRECTIONS = 3  # random directions for J v against central differences
FD_STEP = float(torch.finfo(DTYPE).eps) ** (1 / 3)  # times max(1, |x|_inf), for unit |v|_inf
FD_TOL = 1e-6  # |J v - fd|_inf / max(1, |J v|_inf); truncation and rounding are ~1e-9 here
OBJ_RTOL = 1e-4  # |cost - reference| / max(1, |reference|)
RMSE_ATOL = 1e-3  # |position RMSE - reference position RMSE|
DENSE_MAX_T = 256
MIB = 2 ** 20


# --- one reconstruction -------------------------------------------------------

def setup(method, residual, x0t, model, obs, sigma_dyn, T):
    """What a method builds before the solver starts, and the solver arguments it gives."""
    if method == "jacolor":
        p = prepare(residual, x0t, mode="auto", verify=True)
        return p, {"jac": lambda x: to_scipy(p.jacobian(torch.from_numpy(x)))}
    if method in ("jacfwd", "jacrev"):
        fn = getattr(torch.func, method)(residual)
        return fn, {"jac": lambda x: fn(torch.from_numpy(x)).numpy()}
    if method == "sparse_fd":
        return None, {"jac": "2-point", "jac_sparsity": expected_pattern(T, obs)}
    J = local_reference(model, obs, sigma_dyn)
    return J, {"jac": lambda x: J(torch.from_numpy(x))}


def solve(method, residual, x0, model, obs, sigma_dyn):
    """One complete reconstruction from a fresh solver: setup, then least_squares."""
    calls = {"fun": 0, "jac_s": []}

    def fun(x):
        calls["fun"] += 1
        return residual(torch.from_numpy(x)).numpy()

    T = len(x0) // 2
    t0 = time.perf_counter()
    obj, kw = setup(method, residual, torch.from_numpy(x0), model, obs, sigma_dyn, T)
    t1 = time.perf_counter()
    if callable(kw["jac"]):
        inner = kw["jac"]

        def jac(x):
            t = time.perf_counter()
            J = inner(x)
            calls["jac_s"].append(time.perf_counter() - t)
            return J

        kw["jac"] = jac
    r = least_squares(fun, x0, **kw, **SOLVER)
    t2 = time.perf_counter()
    rec = {"setup_s": t1 - t0, "solve_s": t2 - t1, "total_s": t2 - t0,
           "first_jac_s": calls["jac_s"][0] if calls["jac_s"] else None,
           "jac_total_s": sum(calls["jac_s"]) if calls["jac_s"] else None,
           "status": int(r.status), "message": r.message,
           # status 0 is the evaluation limit: a failed solve, not a timing sample
           "success": bool(r.success and r.status > 0),
           "cost": float(r.cost), "optimality": float(r.optimality),
           "nfev": int(r.nfev), "njev": None if r.njev is None else int(r.njev),
           "fun_calls": calls["fun"], "jac_calls": len(calls["jac_s"])}
    if method == "jacolor":
        rec["prepared"] = obj.summary()
    return r.x, rec


def stats(xs):
    xs = np.asarray(xs, dtype=float)
    return {"median": float(np.median(xs)), "q1": float(np.percentile(xs, 25)),
            "q3": float(np.percentile(xs, 75)), "iqr": float(np.subtract(*np.percentile(xs, [75, 25]))),
            "values": xs.tolist()}


def timed(stmt, min_run_time, **env):
    m = Timer(stmt=stmt, globals=env, num_threads=1).blocked_autorange(min_run_time=min_run_time)
    return {"median_s": m.median, "iqr_s": m.iqr,
            "q1_s": float(np.percentile(m.times, 25)), "q3_s": float(np.percentile(m.times, 75)),
            "samples": len(m.times), "per_sample": m.number_per_run}


def scores(Z, truth, obs, gap):
    missing = np.setdiff1d(np.arange(len(truth)), obs)
    e = Z - truth
    return {"rmse_q": float(np.sqrt((e[:, 0] ** 2).mean())),
            "rmse_q_missing": float(np.sqrt((e[missing, 0] ** 2).mean())),
            "rmse_q_gap": float(np.sqrt((e[gap[0]:gap[1], 0] ** 2).mean())),
            "rmse_v": float(np.sqrt((e[:, 1] ** 2).mean()))}


# --- correctness --------------------------------------------------------------

def compare(J, R):
    """Problems with scipy matrix J against reference R: coordinates, then values."""
    J, R = J.tocsr(), R.tocsr()
    J.sort_indices()
    R.sort_indices()
    if J.shape != R.shape or not (np.array_equal(J.indptr, R.indptr)
                                  and np.array_equal(J.indices, R.indices)):
        return f"coordinates differ: {J.nnz} entries against {R.nnz}"
    if not np.allclose(J.data, R.data, rtol=RTOL, atol=ATOL):
        return f"values differ by up to {np.abs(J.data - R.data).max():.3g}"
    return None


def validate(residual, p, x0t, case, model, sigma_dyn, dense_ok, rng):
    """Checks 1-3 and the first reuse point, outside every timer. Returns (record, problems)."""
    T, obs, y = len(case["truth"]), case["obs"], case["y"]
    out, problems = {}, []
    if T <= DENSE_MAX_T:
        d = (residual(x0t) - loop_residual(model, obs, y, sigma_dyn, x0t)).abs().max().item()
        out["loop_residual_max_abs"] = d
        if not torch.allclose(residual(x0t), loop_residual(model, obs, y, sigma_dyn, x0t),
                              rtol=RTOL, atol=ATOL):
            problems.append(f"residual differs from the loop by {d:.3g}")

    E = expected_pattern(T, obs)
    P = p.pattern
    out["pattern"] = {"detected": int(P.nnz), "expected": int(E.nnz),
                      "missing": int((E > P).nnz), "extra": int((P > E).nnz)}
    if out["pattern"]["missing"] or out["pattern"]["extra"]:
        problems.append(f"pattern: {out['pattern']}")

    J = to_scipy(p.jacobian(x0t))
    out["status_x0"] = p.status
    bad = compare(J, local_reference(model, obs, sigma_dyn)(x0t))
    out["local_at_x0"] = bad or "ok"
    if bad:
        problems.append(f"local reference at x0: {bad}")
    if dense_ok:
        Jd = J.toarray()
        for name in ("jacfwd", "jacrev"):
            D = getattr(torch.func, name)(residual)(x0t).numpy()
            err = float(np.abs(Jd - D).max())
            out[f"{name}_max_abs"] = err
            if not np.allclose(Jd, D, rtol=RTOL, atol=ATOL):
                problems.append(f"{name}: differs from jacolor by {err:.3g}")

    # J v against central differences of the complete residual.
    x = x0t.numpy()
    h = FD_STEP * max(1.0, np.abs(x).max())
    errs = []
    for _ in range(DIRECTIONS):
        v = rng.standard_normal(len(x))
        v /= np.abs(v).max()
        fd = (residual(torch.from_numpy(x + h * v)) - residual(torch.from_numpy(x - h * v))).numpy() / (2 * h)
        Jv = J @ v
        errs.append(float(np.abs(Jv - fd).max() / max(1.0, np.abs(Jv).max())))
    out["directional"] = {"step": h, "rel_err": errs, "tol": FD_TOL}
    if max(errs) > FD_TOL:
        problems.append(f"J v against central differences: {max(errs):.3g} > {FD_TOL:g}")
    return out, problems


def reuse(p, points, model, obs, sigma_dyn):
    """Check 4: one prepared analysis evaluated at several points, each against the reference."""
    ref = local_reference(model, obs, sigma_dyn)
    out, problems = {}, []
    for name, x in points:
        bad = compare(to_scipy(p.jacobian(x)), ref(x))
        out[name] = {"status": p.status, "local": bad or "ok"}
        if bad or p.status != "ok":
            problems.append(f"reuse at {name}: {bad or 'verification ' + str(p.status)}")
    return out, problems


# --- one trajectory -----------------------------------------------------------

def run_case(model, ckpt, T, seed, args):
    sigma_dyn = ckpt["sigma_dyn"]
    case = test_case(T, seed)
    obs, y, truth = case["obs"], case["y"], case["truth"]
    X0 = interpolate(T, obs, y)
    x0 = X0.ravel()
    x0t = torch.from_numpy(x0)
    residual = make_residual(model, obs, y, sigma_dyn)
    rows, cols, nnz = shape(T, len(obs))
    dense_bytes = rows * cols * 8
    r = {"T": T, "seed": seed, "stream": case["stream"], "K": len(obs), "gap": list(case["gap"]),
         "rows": rows, "cols": cols, "nnz_expected": nnz, "problems": [], "methods": {}}
    dense_ok = T <= args.dense_max_t and dense_bytes <= args.dense_limit_mib * MIB
    if not dense_ok:
        r["dense_skipped"] = (f"T = {T} > {args.dense_max_t}" if T > args.dense_max_t else
                              f"output {dense_bytes / MIB:.0f} MiB > {args.dense_limit_mib:g} MiB")
    rng = np.random.default_rng([T, seed, 7])

    try:
        p = prepare(residual, x0t, mode="auto", verify=True)  # the validation analysis
        r["validation"], problems = validate(residual, p, x0t, case, model, sigma_dyn, dense_ok, rng)
    except (TraceUnchecked, VerificationInconclusive) as e:
        r["problems"].append(f"jacolor: {type(e).__name__}: {e}")
        return r, {}
    r["problems"] += problems
    s = p.summary()
    r.update(mode=s["mode"], nnz=s["nnz"], colors=s["colors"], lower_bound=s["lower_bound"],
             reason=s["reason"])

    # Complete reconstructions, each repeat from a fresh solver and the same x0.
    sol = {}
    for m in METHODS:
        if m in ("jacfwd", "jacrev") and not dense_ok:
            continue
        reps = []
        for _ in range(args.repeats):
            try:
                x, rec = solve(m, residual, x0.copy(), model, obs, sigma_dyn)
            except (TraceUnchecked, VerificationInconclusive) as e:
                r["problems"].append(f"{m} solve: {type(e).__name__}: {e}")
                break
            reps.append(rec)
        if not reps:
            continue
        sol[m] = x.reshape(-1, 2)
        last = reps[-1]
        summary = {k: last[k] for k in ("status", "message", "cost", "optimality", "nfev",
                                        "njev", "fun_calls", "jac_calls")}
        summary["success"] = all(q["success"] for q in reps)
        summary["costs_identical"] = len({q["cost"] for q in reps}) == 1
        for k in ("setup_s", "solve_s", "total_s", "first_jac_s", "jac_total_s"):
            if last[k] is not None:
                summary[k] = stats([q[k] for q in reps])
        if m == "jacolor":
            summary["prepared"] = last["prepared"]
        summary["scores"] = scores(sol[m], truth, obs, case["gap"])
        r["methods"][m] = summary
    r["interpolation"] = scores(X0, truth, obs, case["gap"])

    # Check 4 at the recovered state; checks 5-6 against the reference solution.
    pts = [("x0", x0t), ("perturbed", x0t + 0.1 * torch.from_numpy(rng.standard_normal(2 * T)))]
    if "jacolor" in sol:
        pts.append(("recovered", torch.from_numpy(sol["jacolor"].ravel().copy())))
    try:
        r["reuse"], problems = reuse(p, pts, model, obs, sigma_dyn)
        r["problems"] += problems
    except (TraceUnchecked, VerificationInconclusive) as e:
        r["problems"].append(f"reuse: {type(e).__name__}: {e}")
    ref = r["methods"].get(REFERENCE)
    for m, s in r["methods"].items():
        if ref is None:
            s["accepted"] = False
            continue
        s["objective_rel_diff"] = abs(s["cost"] - ref["cost"]) / max(1.0, abs(ref["cost"]))
        s["rmse_q_diff"] = abs(s["scores"]["rmse_q"] - ref["scores"]["rmse_q"])
        s["accepted"] = bool(s["success"] and ref["success"] and s["objective_rel_diff"] <= OBJ_RTOL
                             and s["rmse_q_diff"] <= RMSE_ATOL)
        if not s["accepted"]:
            r["problems"].append(
                f"{m}: success {s['success']} ({s['message']}), objective diff "
                f"{s['objective_rel_diff']:.2e}, position RMSE diff {s['rmse_q_diff']:.2e}")
    if ref is None:
        r["problems"].append("the reference solve did not run")

    # Repeated Jacobian callbacks at the saved input x0, outside the solver.
    jacs = {"jacolor_torch": ("p.jacobian(x)", dict(p=p, x=x0t)),
            "jacolor": ("to_scipy(p.jacobian(x))", dict(p=p, x=x0t, to_scipy=to_scipy)),
            "local": ("J(x)", dict(J=local_reference(model, obs, sigma_dyn), x=x0t)),
            "residual": ("f(x).numpy()", dict(f=residual, x=x0t))}
    if dense_ok:
        for m in ("jacfwd", "jacrev"):
            jacs[m] = ("fn(x).numpy()", dict(fn=getattr(torch.func, m)(residual), x=x0t))
    r["repeated"] = {k: timed(stmt, args.min_run_time, **env) for k, (stmt, env) in jacs.items()}
    r["status"] = p.status
    if p.status != "ok":
        r["problems"].append(f"verification status {p.status!r} after timing")

    J = p.jacobian(x0t).coalesce()
    Js = to_scipy(J)
    r["storage"] = {"coo_bytes": J.values().nbytes + J.indices().nbytes,
                    "csr_bytes": Js.data.nbytes + Js.indices.nbytes + Js.indptr.nbytes,
                    "dense_bytes": dense_bytes, "dense_allocated": dense_ok}

    # Speedups only where both sides passed the quality checks.
    jc = r["methods"].get("jacolor", {})
    r["speedup"] = {}
    for m, s in r["methods"].items():
        if m != "jacolor" and jc.get("accepted") and s.get("accepted"):
            r["speedup"][m] = {k: s[k]["median"] / jc[k]["median"] for k in ("total_s", "solve_s")}

    arrays = {"truth": truth, "obs": obs, "y": y, "init": X0, "gap": np.array(case["gap"])}
    arrays.update({f"recon_{m}": z for m, z in sol.items()})
    return r, arrays


# --- driver -------------------------------------------------------------------

def warm(model, ckpt, args):
    """Imports, tracing, numba compilation and every solver path, on a disposable problem."""
    a = argparse.Namespace(**{**vars(args), "repeats": 1, "min_run_time": 0.01})
    run_case(model, ckpt, 16, 999, a)


def ms(s):
    if s is None:
        return "-"
    v = 1e3 * s
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


HEAD = (f"{'T':>5} {'s':>1} {'nnz':>6} {'col':>3} | {'prep':>5} {'1st':>5} {'jac':>5} {'local':>5} | "
        + " ".join(f"{m:>9}" for m in METHODS)
        + f" | {'nfev':>4} | {'q-RMSE':>6} {'interp':>6} | check")


def row(r):
    ms_ = r["methods"]
    jc = ms_.get("jacolor", {})
    rep = r.get("repeated", {})
    tot = " ".join(f"{ms(ms_[m]['total_s']['median']) if m in ms_ else '-':>9}" for m in METHODS)
    q = jc.get("scores", {}).get("rmse_q")
    return (f"{r['T']:>5} {r['seed']:>1} {r.get('nnz', '-'):>6} {r.get('colors', '-'):>3} | "
            f"{ms(jc.get('setup_s', {}).get('median')):>5} {ms(jc.get('first_jac_s', {}).get('median')):>5} "
            f"{ms(rep.get('jacolor', {}).get('median_s')):>5} {ms(rep.get('local', {}).get('median_s')):>5} | "
            f"{tot} | {jc.get('nfev', '-'):>4} | "
            f"{q if q is None else f'{q:.4f}':>6} {r.get('interpolation', {}).get('rmse_q', 0):.4f} | "
            f"{'ok' if not r['problems'] else 'FAIL'}")


def aggregate(results):
    """Per length: across trajectories, the median, min and max of per-trajectory medians."""
    out = {}
    for T in sorted({r["T"] for r in results}):
        rs = [r for r in results if r["T"] == T]
        a = {}
        for m in METHODS:
            for k in ("setup_s", "first_jac_s", "jac_total_s", "solve_s", "total_s"):
                v = [r["methods"][m][k]["median"] for r in rs if m in r["methods"] and k in r["methods"][m]]
                if v:
                    a.setdefault(m, {})[k] = {"median": float(np.median(v)), "min": min(v), "max": max(v)}
        for k in ("jacolor", "jacolor_torch", "local", "residual", "jacfwd", "jacrev"):
            v = [r["repeated"][k]["median_s"] for r in rs if k in r.get("repeated", {})]
            if v:
                a.setdefault("repeated", {})[k] = {"median": float(np.median(v)), "min": min(v), "max": max(v)}
        for k in ("total_s", "solve_s"):
            for m in METHODS:
                v = [r["speedup"][m][k] for r in rs if m in r.get("speedup", {})]
                if v:
                    a.setdefault(f"speedup_{k[:-2]}", {})[m] = {
                        "median": float(np.median(v)), "min": min(v), "max": max(v), "n": len(v)}
        out[str(T)] = a
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("/tmp/jacolor-cs2/model.pt"))
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs2/results.json"))
    ap.add_argument("--min-run-time", type=float, default=0.2,
                    help="seconds per blocked_autorange measurement")
    ap.add_argument("--dense-max-t", type=int, default=DENSE_MAX_T)
    ap.add_argument("--dense-limit-mib", type=float, default=64,
                    help="skip dense AD when its rows*cols*8 output exceeds this")
    args = ap.parse_args(argv)
    if min(args.lengths) < 3 * GAP:
        ap.error(f"lengths must be at least {3 * GAP}")

    torch.set_num_threads(1)
    model, ckpt = load_model(args.model)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    env = environment(None)
    root = Path(__file__).resolve().parents[2]
    env["git_dirty"] = bool(subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "case_studies/cs2_neural_smoothing"],
        capture_output=True, text=True, cwd=root).stdout.strip())
    env["thread_env"] = {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "VECLIB_MAXIMUM_THREADS")}
    env["model_sha256"] = digest

    warm(model, ckpt, args)

    print(f"learned-dynamics smoothing, {env['dtype']}, CPU, 1 thread | torch {env['torch']}, "
          f"scipy {env['scipy']}, numba {env['numba']} | {env['cpu']} | model {digest[:12]}")
    print("times in ms; total = setup + solve, median of repeats; column definitions: --help\n")
    print(HEAD)
    print("-" * len(HEAD))
    results, arrays = [], {}
    for T in args.lengths:
        for seed in args.seeds:
            r, a = run_case(model, ckpt, T, seed, args)
            results.append(r)
            arrays.update({f"T{T}_s{seed}_{k}": v for k, v in a.items()})
            print(row(r), flush=True)

    failed = [r for r in results if r["problems"]]
    print("-" * len(HEAD))
    for r in failed:
        print(f"FAIL T={r['T']} seed={r['seed']}: " + "; ".join(r["problems"]))
    agg = aggregate(results)
    for T, a in agg.items():
        su = a.get("speedup_total", {})
        print(f"T={T}: jacolor total-time speedup (median over trajectories) "
              + ", ".join(f"vs {m} {v['median']:.2f}x" for m, v in su.items())
              + "; solve only " + ", ".join(f"{m} {v['median']:.2f}x"
                                            for m, v in a.get("speedup_solve", {}).items()))

    out = {"case_study": "cs2_neural_smoothing", "environment": env,
           "settings": {"lengths": args.lengths, "seeds": args.seeds, "repeats": args.repeats,
                        "methods": METHODS, "reference": REFERENCE, "solver": SOLVER,
                        "min_run_time_s": args.min_run_time, "dense_max_t": args.dense_max_t,
                        "dense_limit_mib": args.dense_limit_mib, "rtol": RTOL, "atol": ATOL,
                        "directional": {"directions": DIRECTIONS, "step": "cbrt(eps) * max(1, |x|_inf)",
                                        "tol": FD_TOL},
                        "acceptance": {"objective_rtol": OBJ_RTOL, "rmse_q_atol": RMSE_ATOL},
                        "sigma_obs": SIGMA_OBS, "keep": KEEP, "gap": GAP, "mu_prior": MU_PRIOR,
                        "sigma_prior": SIGMA_PRIOR, "sigma_dyn": ckpt["sigma_dyn"],
                        "prepare": "mode='auto', verify=True, refine=False, chunk=None",
                        "init": "positions interpolated through observations, velocities np.gradient"},
           "model": {"path": str(args.model), "sha256": digest, "config": ckpt["config"],
                     "validation": ckpt["validation"], "training_s": ckpt["training_s"]},
           "validation": {"ok": not failed,
                          "failed_cases": [[r["T"], r["seed"]] for r in failed]},
           "aggregate": agg,
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(f"validation {'passed' if not failed else 'FAILED'} ({len(results)} trajectories); "
          f"results in {args.out} and {args.out.with_suffix('.npz')}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
