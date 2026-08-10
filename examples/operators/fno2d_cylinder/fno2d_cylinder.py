"""Flow past a cylinder — Fourier Neural Operator (PINO) on incompressible
Navier-Stokes.

Reference trajectories are generated locally (see ``datagen.py`` in this
folder — a simple Chorin-projection solver on a channel with a circular
obstacle, masking velocity to zero inside the cylinder) across a range of
Reynolds numbers; the FNO2D operator (``out_channels=3`` for ``(u, v, p)``)
is trained to map a short history window (+ constant-Re channel) to the next
frame, combining a data-MSE loss with the incompressible-NS grid residual
(:class:`underPINN.pde.navier_stokes_2d_grid.CylinderNSGrid`).

Note: this trains a single-step next-frame predictor. Long autoregressive
rollouts of this kind of model often benefit from pushforward training +
noise injection (Brandstetter et al.) to stay stable over many steps — a
genuine refinement, but a materially different training loop, so it's left
out of this port to keep the shared
:class:`underPINN.solver.operator.OperatorSolver` generic across every FNO
example. The single-step model here is still a fully working, physically
correct PINO.

Run directly or via the CLI:

    python examples/operators/fno2d_cylinder/fno2d_cylinder.py
    python -m underPINN run examples/operators/fno2d_cylinder/config.yaml
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
import optax

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.navier_stokes_2d_grid import CylinderNSGrid
from underPINN.losses.operator_loss import OperatorLoss
from underPINN.solver.operator import OperatorSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.metrics import relative_l2_error, print_errors
from underPINN.postprocess.operators import plot_operator_loss, plot_prediction_2d

from datagen import solve_cylinder_flow  # local to this example


def _build_pairs(U, V, P, Re, prev_steps, pred_steps, pairs_per_traj, early_frac, seed):
    """Interleaved (u, v) history + constant-Re channel -> (u, v, p) target,
    matching the layout :class:`CylinderNSGrid` expects."""
    n_re, n_frames, Nx, Ny = U.shape
    last_start = max(int(early_frac * (n_frames - prev_steps - pred_steps)), 1)
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for i in range(n_re):
        starts = rng.integers(0, last_start, size=pairs_per_traj)
        for s in starts:
            u_hist = U[i, s:s + prev_steps]
            v_hist = V[i, s:s + prev_steps]
            tgt_idx = s + prev_steps + pred_steps - 1
            uv_hist = np.stack([u_hist, v_hist], axis=-1)              # (prev,Nx,Ny,2)
            uv_hist = np.moveaxis(uv_hist, 0, -2).reshape(Nx, Ny, -1)    # (Nx,Ny,2*prev)
            re_chan = np.full((Nx, Ny, 1), Re[i], dtype=np.float32)
            inp = np.concatenate([uv_hist, re_chan], axis=-1)
            tgt = np.stack([U[i, tgt_idx], V[i, tgt_idx], P[i, tgt_idx]], axis=-1)
            inputs.append(inp)
            targets.append(tgt)
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


def _make_data(data_cfg, physics_cfg, geom_cfg, seed: int):
    Nx = cfg_get(data_cfg, "Nx", default=64)
    Ny = cfg_get(data_cfg, "Ny", default=32)
    Nt = cfg_get(data_cfg, "Nt", default=400)
    T = cfg_get(data_cfg, "T", default=4.0)
    n_train_re = cfg_get(data_cfg, "n_train_re", default=6)
    n_test_re = cfg_get(data_cfg, "n_test_re", default=2)
    prev_steps = cfg_get(data_cfg, "prev_steps", default=2)
    pred_steps = cfg_get(data_cfg, "pred_steps", default=1)
    pairs_per_traj = cfg_get(data_cfg, "pairs_per_traj", default=8)
    early_frac = cfg_get(data_cfg, "early_frac", default=0.8)
    poisson_iters = cfg_get(data_cfg, "poisson_iters", default=80)

    Lx = cfg_get(geom_cfg, "Lx", default=8.0)
    Ly = cfg_get(geom_cfg, "Ly", default=4.0)
    cx = cfg_get(geom_cfg, "cx", default=2.0)
    cy = cfg_get(geom_cfg, "cy", default=2.0)
    U_in = cfg_get(physics_cfg, "U_in", default=1.0)
    r = cfg_get(physics_cfg, "radius", default=0.5)
    Re_lo = cfg_get(physics_cfg, "Re_lo", default=40.0)
    Re_hi = cfg_get(physics_cfg, "Re_hi", default=150.0)

    rng = np.random.default_rng(seed)
    Re_train = rng.uniform(Re_lo, Re_hi, n_train_re).astype(np.float32)
    Re_test = rng.uniform(Re_lo, Re_hi, n_test_re).astype(np.float32)

    def _solve_all(Re_vals):
        Us, Vs, Ps, mask = [], [], [], None
        for re in Re_vals:
            u, v, p, mask = solve_cylinder_flow(
                float(re), T, Lx, Ly, Nx, Ny, Nt, cx, cy, r, U_in, seed,
                poisson_iters=poisson_iters)
            Us.append(u)
            Vs.append(v)
            Ps.append(p)
        return np.stack(Us), np.stack(Vs), np.stack(Ps), mask

    U_train, V_train, P_train, mask = _solve_all(Re_train)
    U_test, V_test, P_test, _ = _solve_all(Re_test)

    x_train, y_train = _build_pairs(U_train, V_train, P_train, Re_train,
                                    prev_steps, pred_steps, pairs_per_traj,
                                    early_frac, seed)
    x_test, y_test = _build_pairs(U_test, V_test, P_test, Re_test,
                                  prev_steps, pred_steps, pairs_per_traj,
                                  early_frac, seed + 1)

    dt = T / Nt
    dx = Lx / Nx
    dy = Ly / Ny
    return (jnp.array(x_train), jnp.array(y_train),
            jnp.array(x_test), jnp.array(y_test), dt, dx, dy, jnp.array(mask))


def run_fno2d_cylinder(cfg) -> dict:
    """Train the FNO2D cylinder-flow PINO and save outputs."""
    tr = cfg.training
    seed = cfg_get(tr, "seed", default=0)
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/fno2d_cylinder") if out else "outputs/fno2d_cylinder"
    os.makedirs(out_dir, exist_ok=True)

    net_cfg = network_config(cfg)
    model = build_model(net_cfg)

    x_train, y_train, x_test, y_test, dt, dx, dy, mask = _make_data(
        cfg.data, cfg.physics, cfg.geometry, seed)
    pred_steps = cfg_get(cfg.data, "pred_steps", default=1)

    pde = CylinderNSGrid(model, dt=dt, dx=dx, dy=dy,
                         pred_steps=pred_steps, obstacle_mask=mask)
    loss = OperatorLoss(model.apply, pde,
                        rba=bool(cfg_get(cfg.loss, "rba", default=False)))

    epochs = tr.epochs
    lr = tr.lr
    lr_alpha = cfg_get(tr, "lr_alpha", default=0.01)
    log_every = cfg_get(tr, "log_every", default=50)
    batch_size = cfg_get(tr, "batch_size", default=16)

    solver = OperatorSolver(
        model, loss, lr=lr,
        pde_weight=cfg_get(cfg.loss, "pde_weight", default=0.05),
        pde_warmup_epochs=cfg_get(cfg.loss, "pde_warmup_epochs", default=0))
    solver.init(jax.random.PRNGKey(seed), jnp.ones((1,) + x_train.shape[1:]))

    callbacks = [ConsoleLogger(log_every=log_every)]
    tc = TrainingConfig(
        epochs=epochs, lr=lr,
        lr_schedule=optax.cosine_decay_schedule(lr, epochs, alpha=lr_alpha),
        batch_r=batch_size, log_every=log_every, callbacks=callbacks,
        n_scan_steps=cfg_get(tr, "n_scan_steps", default=1),
        out_dir=out_dir,
        save_restart_every=int(cfg_get(tr, "save_restart_every", default=0)),
    )
    solver.train(x_train, y_train, config=tc)

    u_pred_test = model.apply(solver.params, x_test)
    rel_l2 = relative_l2_error(u_pred_test, y_test)
    print_errors(u_pred_test, y_test, label="Test set (u,v,p)")

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    plot_operator_loss(
        {"loss": solver.loss_hist, "data": solver.data_hist, "pde": solver.pde_hist},
        os.path.join(out_dir, "loss.png"), title="FNO2D cylinder flow")

    Nx, Ny = x_test.shape[1], x_test.shape[2]
    xs = np.linspace(0.0, dx * Nx, Nx, endpoint=False)
    ys = np.linspace(0.0, dy * Ny, Ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    plot_prediction_2d(X, Y, np.array(u_pred_test[0, ..., 0]), np.array(y_test[0, ..., 0]),
                       os.path.join(out_dir, "prediction_u.png"),
                       title="FNO2D cylinder flow — u velocity, sample 0")

    save_predictions(out_dir, coords={"x": X, "y": Y},
                     outputs={"u_pred": np.array(u_pred_test[..., 0]),
                              "v_pred": np.array(u_pred_test[..., 1]),
                              "p_pred": np.array(u_pred_test[..., 2])},
                     exact={"u_exact": np.array(y_test[..., 0]),
                           "v_exact": np.array(y_test[..., 1]),
                           "p_exact": np.array(y_test[..., 2])})

    if (cfg_get(out, "save_params", default=True) if out else True):
        solver.save_checkpoint(out_dir, metadata={
            "problem": "fno2d_cylinder",
            "network": net_cfg,
            "physics": {"Re_lo": float(cfg.physics.Re_lo),
                       "Re_hi": float(cfg.physics.Re_hi),
                       "U_in": float(cfg.physics.U_in),
                       "radius": float(cfg.physics.radius)},
        })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "loss_hist": solver.loss_hist,
           "rel_l2": float(rel_l2)}


if __name__ == "__main__":
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else pathlib.Path(__file__).parent / "config.yaml")
    from underPINN.config.loader import load_config
    run_fno2d_cylinder(load_config(cfg_path))
