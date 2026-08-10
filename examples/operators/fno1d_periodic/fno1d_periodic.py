"""1-D periodic viscous Burgers — Fourier Neural Operator (PINO).

Trains an FNO1D to map a short history window (plus a constant-viscosity
channel) to the next frame, combining a data-MSE loss with a finite-difference
PDE-residual term (:class:`underPINN.pde.burgers_grid.BurgersGrid1D`) —
"physics-informed neural operator" style, so the model generalizes across a
*range* of viscosities from one training run instead of solving from scratch
for each one.

Run directly or via the CLI:

    python examples/operators/fno1d_periodic/fno1d_periodic.py
    python -m underPINN run examples/operators/fno1d_periodic/config.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp
import optax

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.burgers_grid import BurgersGrid1D
from underPINN.losses.operator_loss import OperatorLoss
from underPINN.solver.operator import OperatorSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.metrics import relative_l2_error, print_errors
from underPINN.utils.operator_datagen import solve_burgers1d, build_pairs_1d
from underPINN.postprocess.operators import plot_operator_loss, plot_prediction_1d


def _make_data(data_cfg, physics_cfg, seed: int):
    Nx = cfg_get(data_cfg, "Nx", default=64)
    Nt = cfg_get(data_cfg, "Nt", default=200)
    T = cfg_get(data_cfg, "T", default=1.0)
    Lx = cfg_get(data_cfg, "Lx", default=2.0 * np.pi)
    n_train = cfg_get(data_cfg, "n_train", default=400)
    n_test = cfg_get(data_cfg, "n_test", default=80)
    prev_steps = cfg_get(data_cfg, "prev_steps", default=1)
    pred_steps = cfg_get(data_cfg, "pred_steps", default=1)
    pairs_per_traj = cfg_get(data_cfg, "pairs_per_traj", default=5)
    early_frac = cfg_get(data_cfg, "early_frac", default=0.8)

    rng = np.random.default_rng(seed)
    nu_lo = cfg_get(physics_cfg, "nu_lo", default=0.005)
    nu_hi = cfg_get(physics_cfg, "nu_hi", default=0.05)
    nu_train = rng.uniform(nu_lo, nu_hi, n_train).astype(np.float32)
    nu_test = rng.uniform(nu_lo, nu_hi, n_test).astype(np.float32)

    snaps_train = solve_burgers1d(nu_train, T, Lx, Nx, Nt, n_train,
                                  periodic=True, seed=seed)
    snaps_test = solve_burgers1d(nu_test, T, Lx, Nx, Nt, n_test,
                                 periodic=True, seed=seed + 1)

    x_train, y_train = build_pairs_1d(snaps_train, nu_train, prev_steps, pred_steps,
                                      pairs_per_traj, early_frac, seed)
    x_test, y_test = build_pairs_1d(snaps_test, nu_test, prev_steps, pred_steps,
                                    pairs_per_traj, early_frac, seed + 1)
    dt = T / Nt
    dx = Lx / Nx
    return (jnp.array(x_train), jnp.array(y_train),
            jnp.array(x_test), jnp.array(y_test), dt, dx)


def run_fno1d_periodic(cfg) -> dict:
    """Train the FNO1D periodic-Burgers PINO and save outputs."""
    tr = cfg.training
    seed = cfg_get(tr, "seed", default=0)
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/fno1d_periodic") if out else "outputs/fno1d_periodic"
    os.makedirs(out_dir, exist_ok=True)

    net_cfg = network_config(cfg)
    model = build_model(net_cfg)

    x_train, y_train, x_test, y_test, dt, dx = _make_data(cfg.data, cfg.physics, seed)
    pred_steps = cfg_get(cfg.data, "pred_steps", default=1)

    pde = BurgersGrid1D(model, dt=dt, dx=dx, pred_steps=pred_steps, periodic=True)
    loss = OperatorLoss(model.apply, pde,
                        rba=bool(cfg_get(cfg.loss, "rba", default=False)))

    epochs = tr.epochs
    lr = tr.lr
    lr_alpha = cfg_get(tr, "lr_alpha", default=0.01)
    log_every = cfg_get(tr, "log_every", default=50)
    batch_size = cfg_get(tr, "batch_size", default=32)

    solver = OperatorSolver(
        model, loss, lr=lr,
        pde_weight=cfg_get(cfg.loss, "pde_weight", default=0.1),
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

    u_pred_test = model.apply(solver.params, x_test)[..., 0]
    rel_l2 = relative_l2_error(u_pred_test, y_test[..., 0])
    print_errors(u_pred_test, y_test[..., 0], label="Test set")

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    plot_operator_loss(
        {"loss": solver.loss_hist, "data": solver.data_hist, "pde": solver.pde_hist},
        os.path.join(out_dir, "loss.png"), title="FNO1D periodic Burgers")

    x_ax = np.linspace(0.0, dx * x_test.shape[1], x_test.shape[1], endpoint=False)
    plot_prediction_1d(x_ax, np.array(u_pred_test[0]), np.array(y_test[0, ..., 0]),
                       os.path.join(out_dir, "prediction.png"),
                       title="FNO1D periodic Burgers — sample 0")

    save_predictions(out_dir, coords={"x": x_ax},
                     outputs={"u_pred": np.array(u_pred_test)},
                     exact={"u_exact": np.array(y_test[..., 0])})

    if (cfg_get(out, "save_params", default=True) if out else True):
        solver.save_checkpoint(out_dir, metadata={
            "problem": "fno1d_periodic",
            "network": net_cfg,
            "physics": {"nu_lo": float(cfg.physics.nu_lo),
                       "nu_hi": float(cfg.physics.nu_hi)},
        })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "loss_hist": solver.loss_hist,
           "rel_l2": float(rel_l2)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_fno1d_periodic(load_config(cfg_path))
