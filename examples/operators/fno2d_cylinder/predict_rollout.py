"""Autoregressive unsteady-flow rollout for the trained FNO2D cylinder model.

``fno2d_cylinder.py`` trains a *single-step* next-frame predictor (see its
module docstring). That alone only tests "does it predict the immediate next
frame well" -- it does not show whether the model has actually learned the
system's genuine *unsteady* dynamics (sustained periodic vortex shedding;
see the config.yaml comments and this session's verification work). This
script tests exactly that: seed the model with a real history window well
past the initial transient (in the verified periodic regime), then chain the
model's own predictions forward hundreds of steps with no ground truth in
the loop, and check whether the resulting trajectory keeps oscillating like
the true flow (and for how long, before any autoregressive drift sets in --
this model was NOT trained with pushforward/noise-injection stabilization,
so some drift over a long rollout is expected, not a bug; see
fno2d_cylinder.py's docstring for that documented limitation).

Run:
    python examples/operators/fno2d_cylinder/predict_rollout.py [config.yaml]
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flax import serialization

from underPINN.config.loader import load_config, cfg_get
from underPINN.nn.factory import build_model, network_config
from underPINN.utils.checkpoint import _msgpack_path

from datagen import solve_cylinder_flow


def rollout(predict_fn, params, seed_window: np.ndarray, re_val: float,
           fluid_mask: np.ndarray, n_steps: int) -> np.ndarray:
    """Chain ``predict_fn`` forward ``n_steps`` with no ground truth in the
    loop. ``seed_window``: (Nx, Ny, 2*prev_steps) u/v history, interleaved
    [u_1,v_1,...,u_k,v_k] (oldest first) -- matches
    ``fno2d_cylinder._build_pairs``'s layout. Returns (n_steps, Nx, Ny, 3)
    predicted (u, v, p)."""
    Nx, Ny, ch = seed_window.shape
    re_chan = np.full((Nx, Ny, 1), re_val, dtype=np.float32)
    mask_chan = fluid_mask.astype(np.float32)[..., None]
    window = seed_window.copy()
    preds = np.zeros((n_steps, Nx, Ny, 3), dtype=np.float32)
    for t in range(n_steps):
        x_in = jnp.array(np.concatenate([window, re_chan, mask_chan], axis=-1)[None])
        out = np.array(predict_fn(params, x_in))[0]     # (Nx, Ny, 3), already hard-masked
        preds[t] = out
        # shift window: drop oldest (u,v) pair, append the new prediction's (u,v)
        window = np.concatenate([window[:, :, 2:], out[:, :, 0:1], out[:, :, 1:2]], axis=-1)
    return preds


def main():
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else pathlib.Path(__file__).parent / "config.yaml")
    cfg = load_config(cfg_path)
    out_dir = cfg_get(cfg_get(cfg, "output", default=None), "dir",
                      default="outputs/fno2d_cylinder") or "outputs/fno2d_cylinder"
    roll_dir = os.path.join(out_dir, "rollout")
    os.makedirs(roll_dir, exist_ok=True)

    # --- load the trained model ---------------------------------------
    # ModelPredictor.from_checkpoint/from_meta both assume a point-network
    # 2-D (1, in_features) dummy input to rebuild the params template (see
    # ModelPredictor.from_meta's own docstring) -- wrong for a grid operator
    # like FNO2D, so build the template with a correctly-shaped 4-D dummy
    # input directly and load the raw checkpoint bytes into it.
    model = build_model(network_config(cfg))
    dummy = jnp.ones((1, cfg.data.Nx, cfg.data.Ny, 2 * cfg_get(cfg.data, "prev_steps", default=2) + 2))
    template = model.init(jax.random.PRNGKey(0), dummy)
    with open(_msgpack_path(out_dir), "rb") as f:
        params = serialization.from_bytes(template, f.read())
    print(f"Loaded checkpoint from {out_dir}/params.msgpack")

    prev_steps = cfg_get(cfg.data, "prev_steps", default=2)
    pred_steps = cfg_get(cfg.data, "pred_steps", default=1)
    dt = cfg.data.T / cfg.data.Nt

    # --- fresh validation trajectory: an unseen Re, safely inside the
    # verified-plateaued 148-170 band (train draws: 162.0/153.9/148.9/148.4/
    # 165.9/168.1; test: 159.3/168.9) --------------------------------------
    re_val = 158.0
    print(f"Generating validation trajectory at Re={re_val} (unseen during training)")
    U, V, P, mask = solve_cylinder_flow(
        re_val, cfg.data.T, cfg.geometry.Lx, cfg.geometry.Ly,
        cfg.data.Nx, cfg.data.Ny, cfg.data.Nt, cfg.geometry.cx, cfg.geometry.cy,
        cfg.physics.radius, cfg.physics.U_in, cfg_get(cfg.data, "seed", default=0),
        poisson_iters=cfg_get(cfg.data, "poisson_iters", default=80))
    fluid_mask = ~mask

    # --- seed the rollout well past the initial transient (t=40, verified
    # plateau region -- see config.yaml's comments) so this genuinely tests
    # periodic-regime prediction, not the (much easier, monotonic) startup
    # transient -----------------------------------------------------------
    t_start = 40.0
    s = int(t_start / dt)
    n_steps = 3000    # ~30 time units of rollout = ~5 shedding periods (period~5.9)

    seed_uv = np.stack([U[s:s + prev_steps], V[s:s + prev_steps]], axis=-1)
    seed_uv = np.moveaxis(seed_uv, 0, -2).reshape(cfg.data.Nx, cfg.data.Ny, -1)

    mask_b = jnp.array(fluid_mask.astype(np.float32))[None, :, :, None]

    def predict_fn(params, x_input):
        return model.apply(params, x_input) * mask_b

    print(f"Rolling out {n_steps} steps ({n_steps * dt * pred_steps:.1f} time units) "
         f"from t={t_start} with no ground truth in the loop...")
    preds = rollout(predict_fn, params, seed_uv, re_val, fluid_mask, n_steps)

    tgt_start = s + prev_steps + pred_steps - 1
    exact = np.stack([U[tgt_start:tgt_start + n_steps],
                      V[tgt_start:tgt_start + n_steps],
                      P[tgt_start:tgt_start + n_steps]], axis=-1)
    t_roll = t_start + dt * pred_steps * np.arange(n_steps)

    np.savez(os.path.join(roll_dir, "rollout.npz"),
            t=t_roll, u_pred=preds[..., 0], v_pred=preds[..., 1], p_pred=preds[..., 2],
            u_exact=exact[..., 0], v_exact=exact[..., 1], p_exact=exact[..., 2],
            re=re_val, fluid_mask=fluid_mask)
    print(f"Rollout saved -> {roll_dir}/rollout.npz")

    # --- per-step rel-L2 (error growth over the rollout) ------------------
    diff = preds - exact
    per_step_rel_l2 = np.linalg.norm(diff.reshape(n_steps, -1), axis=1) / (
        np.linalg.norm(exact.reshape(n_steps, -1), axis=1) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_roll, per_step_rel_l2)
    ax.set_xlabel("t")
    ax.set_ylabel("rel-L2 (u,v,p)")
    ax.set_title(f"Rollout error growth, Re={re_val}, seeded at t={t_start}")
    fig.tight_layout()
    fig.savefig(os.path.join(roll_dir, "rollout_error_growth.png"), dpi=110)
    plt.close(fig)

    # --- wake-probe signal: predicted vs exact, the direct "is it still
    # oscillating like the true flow" check used throughout this session's
    # verification work ------------------------------------------------
    dx, dy = cfg.geometry.Lx / cfg.data.Nx, cfg.geometry.Ly / cfg.data.Ny
    ix, iy = int(10.0 / dx), int(2.3 / dy)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_roll, exact[:, ix, iy, 1], label="exact", lw=1.5)
    ax.plot(t_roll, preds[:, ix, iy, 1], label="rollout (model, autoregressive)",
           lw=1.2, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel("v at probe (x=10, y=2.3)")
    ax.set_title(f"Unsteady rollout vs. exact, Re={re_val}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(roll_dir, "rollout_probe_signal.png"), dpi=110)
    plt.close(fig)

    # --- a few field snapshots across the rollout --------------------------
    xs = np.linspace(0.0, cfg.geometry.Lx, cfg.data.Nx, endpoint=False)
    ys = np.linspace(0.0, cfg.geometry.Ly, cfg.data.Ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    snap_idx = [0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]
    fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 8))
    for col, k in enumerate(snap_idx):
        axes[0, col].contourf(X, Y, preds[k, ..., 0], levels=40, cmap="turbo")
        axes[0, col].set_title(f"pred u, t={t_roll[k]:.1f}")
        axes[1, col].contourf(X, Y, exact[k, ..., 0], levels=40, cmap="turbo")
        axes[1, col].set_title(f"exact u, t={t_roll[k]:.1f}")
        axes[2, col].contourf(X, Y, np.abs(preds[k, ..., 0] - exact[k, ..., 0]),
                              levels=40, cmap="inferno")
        axes[2, col].set_title(f"|error|, t={t_roll[k]:.1f}")
        for ax in (axes[0, col], axes[1, col], axes[2, col]):
            ax.set_xlabel("x")
        axes[0, col].set_ylabel("y")
    fig.suptitle(f"Rollout snapshots, Re={re_val}, seeded t={t_start}")
    fig.tight_layout()
    fig.savefig(os.path.join(roll_dir, "rollout_snapshots.png"), dpi=110)
    plt.close(fig)

    print(f"rel-L2 at rollout start: {per_step_rel_l2[0]:.4f}")
    print(f"rel-L2 at rollout end:   {per_step_rel_l2[-1]:.4f}")
    print(f"rel-L2 max over rollout: {per_step_rel_l2.max():.4f}")
    print(f"Plots saved -> {roll_dir}/rollout_error_growth.png, "
         f"rollout_probe_signal.png, rollout_snapshots.png")


if __name__ == "__main__":
    main()
