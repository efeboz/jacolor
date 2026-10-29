"""Case study 2: train the transition model once and save the checkpoint run.py uses.

    python -m case_studies.cs2_neural_smoothing.train --out /tmp/jacolor-cs2/model.pt  # repo root

One-step supervised training of g(z) = z + dt * net(z) on simulated oscillator
trajectories, with Adam on CPU in float64 and one thread. The checkpoint with the
lowest validation error is kept. Its validation next-state RMSE, per component
and floored, becomes sigma_dyn, the fixed weight of the dynamics residual.

Validation errors are reported against persistence (z_next = z) and a linear
transition fitted by least squares on the training pairs.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from case_studies.cs2_neural_smoothing.problem import (  # noqa: E402
    DT, SIGMA_DYN_FLOOR, SUBSTEPS, TRAIN_STREAM, VAL_STREAM, Dynamics, trajectories)

# max_updates was 2000 in the plan. The validation curve was still falling there and
# the model was no better than the linear baseline, so it was raised once, to
# 20000, after trying 10000 and 20000 on validation data. Nothing else was tuned.
CONFIG = {
    "n_train": 64, "n_val": 16, "length": 256, "hidden": 32, "dt": DT, "substeps": SUBSTEPS,
    "batch": 256, "lr": 1e-3, "max_updates": 20000, "eval_every": 25,
    "train_stream": TRAIN_STREAM, "val_stream": VAL_STREAM, "torch_seed": 0,
}


def pairs(Z):
    """(z_t, z_(t+1)) for every step of every trajectory, never across trajectories."""
    return Z[:, :-1].reshape(-1, 2), Z[:, 1:].reshape(-1, 2)


def rmse(pred, target):
    """Next-state RMSE per component."""
    return np.sqrt(((pred - target) ** 2).mean(axis=0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("/tmp/jacolor-cs2/model.pt"))
    ap.add_argument("--max-updates", type=int, default=CONFIG["max_updates"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"])
    args = ap.parse_args(argv)
    cfg = dict(CONFIG, max_updates=args.max_updates, lr=args.lr)

    torch.set_num_threads(1)
    torch.manual_seed(cfg["torch_seed"])
    X, Y = pairs(trajectories(cfg["train_stream"], cfg["n_train"], cfg["length"]))
    Xv, Yv = pairs(trajectories(cfg["val_stream"], cfg["n_val"], cfg["length"]))

    model = Dynamics(cfg["hidden"], cfg["dt"])
    D = (Y - X) / cfg["dt"]  # increment targets
    with torch.no_grad():  # training statistics only
        model.in_mean.copy_(torch.from_numpy(X.mean(0)))
        model.in_std.copy_(torch.from_numpy(X.std(0)))
        model.out_mean.copy_(torch.from_numpy(D.mean(0)))
        model.out_std.copy_(torch.from_numpy(D.std(0)))
    Xt, Dt = torch.from_numpy(X), torch.from_numpy(D)
    Xvt = torch.from_numpy(Xv)
    target = (Dt - model.out_mean) / model.out_std  # loss in standardized units

    def val_rmse():
        with torch.no_grad():
            return rmse(model(Xvt).numpy(), Yv)

    opt = torch.optim.Adam(model.net.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(cfg["torch_seed"])
    best = {"update": 0, "val_rmse": val_rmse(), "state": None}
    history = []
    t0 = time.perf_counter()
    perm, pos = torch.randperm(len(Xt), generator=g), 0
    for step in range(1, cfg["max_updates"] + 1):
        if pos + cfg["batch"] > len(perm):
            perm, pos = torch.randperm(len(Xt), generator=g), 0
        idx = perm[pos:pos + cfg["batch"]]
        pos += cfg["batch"]
        u = (Xt[idx] - model.in_mean) / model.in_std
        loss = ((model.net(u) - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["eval_every"] == 0 or step == cfg["max_updates"]:
            v = val_rmse()
            history.append({"update": step, "train_loss": loss.item(), "val_rmse": v.tolist()})
            if np.mean(v ** 2) < np.mean(best["val_rmse"] ** 2):
                best = {"update": step, "val_rmse": v,
                        "state": {k: t.clone() for k, t in model.state_dict().items()}}
    duration = time.perf_counter() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # Baselines on the same validation pairs.
    persistence = rmse(Xv, Yv)
    A, *_ = np.linalg.lstsq(np.hstack([X, np.ones((len(X), 1))]), Y, rcond=None)
    with np.errstate(all="ignore"):  # Accelerate's matmul raises spurious FP warnings
        linear = rmse(np.hstack([Xv, np.ones((len(Xv), 1))]) @ A, Yv)
    model_rmse = val_rmse()
    sigma_dyn = np.maximum(model_rmse, SIGMA_DYN_FLOOR)

    ckpt = {
        "config": cfg,
        "state_dict": model.state_dict(),
        "sigma_dyn": sigma_dyn.tolist(),
        "validation": {"model_rmse": model_rmse.tolist(), "persistence_rmse": persistence.tolist(),
                       "linear_rmse": linear.tolist(), "best_update": best["update"],
                       "pairs": len(Xv)},
        "train_pairs": len(X),
        "training_s": duration,
        "history": history,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    summary = {k: ckpt[k] for k in ("config", "sigma_dyn", "validation", "train_pairs", "training_s")}
    summary["sha256"] = digest
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))

    def fmt(v):
        return "[" + ", ".join(f"{x:.2e}" for x in v) + "]"

    print(f"{len(X)} training pairs, {len(Xv)} validation pairs, {cfg['max_updates']} updates "
          f"in {duration:.1f} s; best at update {best['update']}")
    print(f"validation next-state RMSE [q, v]: model {fmt(model_rmse)}, "
          f"linear {fmt(linear)}, persistence {fmt(persistence)}")
    print(f"sigma_dyn {fmt(sigma_dyn)}; checkpoint {args.out} (sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
