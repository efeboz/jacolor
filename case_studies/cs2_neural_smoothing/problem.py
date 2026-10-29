"""Case study 2: the oscillator, the learned transition model and the smoothing residual.

Everything train.py and run.py share lives here: the simulator that makes data
and scoring targets, the MLP that replaces it during reconstruction, the
observation pattern, the residual r(Z) whose sparse Jacobian jacolor detects,
the initial guess, and two references that know the residual's layout (the
expected pattern and a local-block Jacobian). See CASE_STUDY_2.md.

Residual rows, in order, for the flattened trajectory x = Z.reshape(-1):

    observation  (q_t - y_t) / sigma_obs                for t in O        K rows
    dynamics     (z_(t+1) - g(z_t)) / sigma_dyn         t = 0 .. T-2      2*(T-1) rows
    prior        (z_0 - mu_prior) / sigma_prior                           2 rows
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

DTYPE = torch.float64
DT = 0.05  # observation interval
SUBSTEPS = 5  # RK4 steps per interval
Q_RANGE, V_RANGE = 1.5, 1.0  # initial conditions uniform on [-1.5, 1.5] x [-1, 1]
SIGMA_OBS = 0.05
KEEP = 0.5  # probability that a measurement is retained
GAP = 8  # length of the removed interior interval
MU_PRIOR = (0.0, 0.0)
SIGMA_PRIOR = (2.0, 2.0)
SIGMA_DYN_FLOOR = 1e-3

# Streams for numpy.random.default_rng. Training, validation and testing never share one.
TRAIN_STREAM, VAL_STREAM, TEST_STREAM = 101, 202, 303


# --- the true system ---------------------------------------------------------

def oscillator(z):
    """dq/dt = v, dv/dt = -q - 0.2 q^3, for z[..., 0] = q and z[..., 1] = v."""
    q, v = z[..., 0], z[..., 1]
    return np.stack([v, -q - 0.2 * q ** 3], axis=-1)


def simulate(z0, T):
    """Trajectories of T states from initial states z0 of shape (B, 2), fixed-step RK4."""
    h = DT / SUBSTEPS
    Z = np.empty((len(z0), T, 2))
    z = np.asarray(z0, dtype=np.float64)
    Z[:, 0] = z
    for t in range(1, T):
        for _ in range(SUBSTEPS):
            k1 = oscillator(z)
            k2 = oscillator(z + h / 2 * k1)
            k3 = oscillator(z + h / 2 * k2)
            k4 = oscillator(z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Z[:, t] = z
    return Z


def initial_states(rng, n):
    return np.stack([rng.uniform(-Q_RANGE, Q_RANGE, n), rng.uniform(-V_RANGE, V_RANGE, n)], axis=1)


def trajectories(stream, n, T):
    """n clean trajectories of length T from their own random stream."""
    return simulate(initial_states(np.random.default_rng(stream), n), T)


# --- the learned transition --------------------------------------------------

class Dynamics(nn.Module):
    """g(z) = z + dt * net(z), with the standardization of inputs and targets built in.

    It consumes and returns physical states. The normalization is held in buffers,
    so it travels with the state dict.
    """

    def __init__(self, hidden=32, dt=DT):
        super().__init__()
        self.dt = dt
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        for name in ("in_mean", "in_std", "out_mean", "out_std"):
            self.register_buffer(name, torch.zeros(2, dtype=DTYPE) if "mean" in name
                                 else torch.ones(2, dtype=DTYPE))
        self.to(DTYPE)

    def increment(self, z):
        """net(z) in physical units: the predicted (z_next - z) / dt."""
        return self.net((z - self.in_mean) / self.in_std) * self.out_std + self.out_mean

    def forward(self, z):
        return z + self.dt * self.increment(z)


def load_model(path):
    """The frozen model and the checkpoint dict it came from."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Dynamics(ckpt["config"]["hidden"], ckpt["config"]["dt"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().requires_grad_(False)
    return model, ckpt


# --- the reconstruction problem ----------------------------------------------

def observe(Z, rng):
    """Observed indices and noisy positions for one clean trajectory Z of shape (T, 2).

    About half the measurements are kept, one interior interval of GAP samples is
    removed, and the first and last are always kept.
    """
    T = len(Z)
    keep = rng.random(T) < KEEP
    start = int(rng.integers(T // 4, 3 * T // 4 - GAP + 1))
    keep[start:start + GAP] = False
    keep[0] = keep[-1] = True
    obs = np.flatnonzero(keep)
    y = Z[obs, 0] + SIGMA_OBS * rng.standard_normal(len(obs))
    return obs, y, (start, start + GAP)


def test_case(T, seed):
    """A held-out trajectory with its observations. Its stream depends on (T, seed) only."""
    rng = np.random.default_rng([TEST_STREAM, T, seed])
    Z = simulate(initial_states(rng, 1), T)[0]
    obs, y, gap = observe(Z, rng)
    return {"truth": Z, "obs": obs, "y": y, "gap": gap,
            "stream": [TEST_STREAM, T, seed]}


def interpolate(T, obs, y):
    """The initial guess: positions linearly interpolated through the noisy observations,
    velocities by finite differences of that interpolation. Shape (T, 2)."""
    q = np.interp(np.arange(T), obs, y)
    return np.stack([q, np.gradient(q, DT)], axis=1)


def make_residual(model, obs, y, sigma_dyn):
    """r(x) for the flattened trajectory x of length 2*T. Everything but x is fixed."""
    obs = torch.as_tensor(obs, dtype=torch.long)
    y = torch.as_tensor(y, dtype=DTYPE)
    w_dyn = 1 / torch.as_tensor(sigma_dyn, dtype=DTYPE)
    mu = torch.tensor(MU_PRIOR, dtype=DTYPE)
    w_prior = 1 / torch.tensor(SIGMA_PRIOR, dtype=DTYPE)

    def residual(x):
        Z = x.reshape(-1, 2)
        r_obs = (Z[obs, 0] - y) / SIGMA_OBS
        r_dyn = (Z[1:] - model(Z[:-1])) * w_dyn
        r_prior = (Z[0] - mu) * w_prior
        return torch.cat([r_obs, r_dyn.reshape(-1), r_prior])

    return residual


def loop_residual(model, obs, y, sigma_dyn, x):
    """The same residual one term at a time, as a check on the vectorized one."""
    Z = x.reshape(-1, 2)
    out = [(Z[t, 0] - yt) / SIGMA_OBS for t, yt in zip(obs.tolist(), y.tolist())]
    for t in range(len(Z) - 1):
        nxt = model(Z[t:t + 1])[0]
        out += [(Z[t + 1, i] - nxt[i]) / sigma_dyn[i] for i in range(2)]
    out += [(Z[0, i] - MU_PRIOR[i]) / SIGMA_PRIOR[i] for i in range(2)]
    return torch.stack([torch.as_tensor(v, dtype=DTYPE) for v in out])


def shape(T, K):
    rows, cols = 2 * T + K, 2 * T
    return rows, cols, 6 * (T - 1) + K + 2


def _layout(T, obs):
    """Rows and columns of every structural entry, in the order the residual is stacked.

    Dynamics row (t, i) touches z_t (both components, through the network) and
    component i of z_(t+1).
    """
    K = len(obs)
    t = np.arange(T - 1)
    i = np.arange(2)
    row_dyn = K + 2 * t[:, None] + i[None, :]  # (T-1, 2)
    dyn_rows = np.concatenate([np.repeat(row_dyn[..., None], 2, axis=2).ravel(), row_dyn.ravel()])
    dyn_cols = np.concatenate([(2 * t[:, None, None] + i[None, None, :]).repeat(2, axis=1).ravel(),
                               (2 * (t[:, None] + 1) + i[None, :]).ravel()])
    rows = np.concatenate([np.arange(K), dyn_rows, K + 2 * (T - 1) + i])
    cols = np.concatenate([2 * np.asarray(obs), dyn_cols, i])
    return rows, cols


def expected_pattern(T, obs):
    """The structural pattern, from the residual's definition rather than from tracing."""
    rows, cols = _layout(T, obs)
    m, n, _ = shape(T, len(obs))
    return sp.csr_array((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(m, n))


def local_reference(model, obs, sigma_dyn):
    """A structure-aware Jacobian: each 2x2 transition Jacobian by vmap(jacrev(g)),
    assembled into the known residual blocks as scipy CSR.

    It is differentiated by the same autograd as everything else, so it checks
    jacolor's detection, coloring and assembly, not autograd itself.
    """
    obs = np.asarray(obs)
    w_dyn = 1 / np.asarray(sigma_dyn, dtype=np.float64)
    block = torch.func.vmap(torch.func.jacrev(model))
    K = len(obs)

    def jacobian(x):
        Z = x.reshape(-1, 2)
        T = len(Z)
        A = block(Z[:-1]).numpy()  # (T-1, 2, 2): d g(z_t)_i / d z_t,j
        rows, cols = _layout(T, obs)
        vals = np.concatenate([np.full(K, 1 / SIGMA_OBS),
                               (-A * w_dyn[None, :, None]).ravel(),
                               np.tile(w_dyn, T - 1),
                               1 / np.asarray(SIGMA_PRIOR)])
        m, n, _ = shape(T, K)
        return sp.csr_array((vals, (rows, cols)), shape=(m, n))

    return jacobian
