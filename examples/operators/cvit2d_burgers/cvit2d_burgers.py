"""2-D periodic viscous Burgers — Continuous Vision Transformer (CViT PINO).

Wang et al., ICLR 2025. Unlike FNO (which is queried only on its native
grid), CViT decodes at arbitrary continuous coordinates via a cross-attention
decoder — but a finite-difference PDE residual needs values *on* a grid, so
:func:`underPINN.nn.operators.cvit_grid_predict` queries the model at every
grid point and reshapes back, letting the same
:class:`underPINN.pde.burgers_grid.BurgersGrid2D` residual used for FNO2D
serve CViT too.

Two things were empirically found critical for accuracy here (see the module
docstring of :func:`underPINN.nn.operators.cvit_grid_predict`):

* predicting the *increment* from the last history frame
  (``u_pred = u_prev + delta_scale * raw``) rather than the raw field, and
* matching the residual's advection stencil (``scheme: upwind``) to whatever
  stencil generated the training data.

Run directly or via the CLI:

    python examples/operators/cvit2d_burgers/cvit2d_burgers.py
    python -m underPINN run examples/operators/cvit2d_burgers/config.yaml
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
from underPINN.nn.operators import cvit_grid_predict
from underPINN.pde.burgers_grid import BurgersGrid2D
from underPINN.losses.operator_loss import OperatorLoss
from underPINN.solver.operator import OperatorSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.metrics import relative_l2_error, print_errors
from underPINN.utils.operator_datagen import solve_burgers2d, build_pairs_2d
from underPINN.postprocess.operators import plot_operator_loss, plot_prediction_2d


def _to_cvit_seq(inputs_fno: np.ndarray, prev_steps: int) -> np.ndarray:
    """``(M, Nx, Ny, prev_steps + 1)`` FNO-style window -> CViT's own
    ``(M, prev_steps, Nx, Ny, 2)`` window (channel 0 = u history, channel 1 =
    the constant-nu channel broadcast into every history frame)."""
    u_hist = inputs_fno[..., :-1]                              # (M, Nx, Ny, prev)
    nu_chan = inputs_fno[..., -1:]                              # (M, Nx, Ny, 1)
    u_hist_t = np.moveaxis(u_hist, -1, 1)[..., None]            # (M, prev, Nx, Ny, 1)
    nu_frame = np.repeat(nu_chan[:, None], prev_steps, axis=1)  # (M, prev, Nx, Ny, 1)
    return np.concatenate([u_hist_t, nu_frame], axis=-1)


def _make_data(data_cfg, physics_cfg, seed: int):
    Nx = cfg_get(data_cfg, "Nx", default=32)
    Nt = cfg_get(data_cfg, "Nt", default=300)
    T = cfg_get(data_cfg, "T", default=1.0)
    Lx = cfg_get(data_cfg, "Lx", default=2.0 * np.pi)
    Ly = cfg_get(data_cfg, "Ly", default=2.0 * np.pi)
    n_train = cfg_get(data_cfg, "n_train", default=200)
    n_test = cfg_get(data_cfg, "n_test", default=40)
    prev_steps = cfg_get(data_cfg, "prev_steps", default=4)
    pred_steps = cfg_get(data_cfg, "pred_steps", default=1)
    pairs_per_traj = cfg_get(data_cfg, "pairs_per_traj", default=4)
    early_frac = cfg_get(data_cfg, "early_frac", default=0.6)

    rng = np.random.default_rng(seed)
    nu_lo = cfg_get(physics_cfg, "nu_lo", default=0.01)
    nu_hi = cfg_get(physics_cfg, "nu_hi", default=0.05)
    nu_train = rng.uniform(nu_lo, nu_hi, n_train).astype(np.float32)
    nu_test = rng.uniform(nu_lo, nu_hi, n_test).astype(np.float32)

    snaps_train = solve_burgers2d(nu_train, T, Lx, Ly, Nx, Nt, n_train, seed=seed)
    snaps_test = solve_burgers2d(nu_test, T, Lx, Ly, Nx, Nt, n_test, seed=seed + 1)

    x_train, y_train = build_pairs_2d(snaps_train, nu_train, prev_steps, pred_steps,
                                      pairs_per_traj, early_frac, seed)
    x_test, y_test = build_pairs_2d(snaps_test, nu_test, prev_steps, pred_steps,
                                    pairs_per_traj, early_frac, seed + 1)
    dt = T / Nt
    dx = Lx / Nx
    dy = Ly / Nx
    return x_train, y_train, x_test, y_test, dt, dx, dy, prev_steps


def run_cvit2d_burgers(cfg) -> dict:
    """Train the CViT periodic-Burgers PINO and save outputs."""
    tr = cfg.training
    seed = cfg_get(tr, "seed", default=0)
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/cvit2d_burgers") if out else "outputs/cvit2d_burgers"
    os.makedirs(out_dir, exist_ok=True)

    net_cfg = network_config(cfg)
    model = build_model(net_cfg)

    (x_train_np, y_train_np, x_test_np, y_test_np,
     dt, dx, dy, prev_steps) = _make_data(cfg.data, cfg.physics, seed)

    u_prev_train = x_train_np[..., -2]           # (M, Nx, Ny) last history frame
    delta_scale = float(np.std(y_train_np[..., 0] - u_prev_train)) + 1e-8

    x_seq_train = _to_cvit_seq(x_train_np, prev_steps)
    x_seq_test = _to_cvit_seq(x_test_np, prev_steps)

    x_train = jnp.array(x_train_np)
    y_train = jnp.array(y_train_np)
    x_test = jnp.array(x_test_np)
    y_test = jnp.array(y_test_np)
    x_seq_train = jnp.array(x_seq_train)
    x_seq_test = jnp.array(x_seq_test)

    def predict_fn(params, x_seq):
        u_prev = x_seq[:, -1, :, :, 0]
        return cvit_grid_predict(model, params, x_seq, u_prev, delta_scale)

    pred_steps = cfg_get(cfg.data, "pred_steps", default=1)
    scheme = cfg_get(cfg.loss, "scheme", default="upwind")
    pde = BurgersGrid2D(model, dt=dt, dx=dx, dy=dy, pred_steps=pred_steps, scheme=scheme)
    loss = OperatorLoss(predict_fn, pde, rba=bool(cfg_get(cfg.loss, "rba", default=False)))

    epochs = tr.epochs
    lr = tr.lr
    lr_alpha = cfg_get(tr, "lr_alpha", default=0.01)
    log_every = cfg_get(tr, "log_every", default=50)
    batch_size = cfg_get(tr, "batch_size", default=16)

    solver = OperatorSolver(
        model, loss, lr=lr,
        pde_weight=cfg_get(cfg.loss, "pde_weight", default=0.1),
        pde_warmup_epochs=cfg_get(cfg.loss, "pde_warmup_epochs", default=0))
    coords_dummy = jnp.zeros((4, 2))
    solver.init(jax.random.PRNGKey(seed), jnp.ones((1,) + x_seq_train.shape[1:]), coords_dummy)

    callbacks = [ConsoleLogger(log_every=log_every)]
    tc = TrainingConfig(
        epochs=epochs, lr=lr,
        lr_schedule=optax.cosine_decay_schedule(lr, epochs, alpha=lr_alpha),
        batch_r=batch_size, log_every=log_every, callbacks=callbacks,
        n_scan_steps=cfg_get(tr, "n_scan_steps", default=1),
        out_dir=out_dir,
        save_restart_every=int(cfg_get(tr, "save_restart_every", default=0)),
    )
    solver.train(x_train, y_train, model_input=x_seq_train, config=tc)

    u_pred_test = predict_fn(solver.params, x_seq_test)[..., 0]
    rel_l2 = relative_l2_error(u_pred_test, y_test[..., 0])
    print_errors(u_pred_test, y_test[..., 0], label="Test set")

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    plot_operator_loss(
        {"loss": solver.loss_hist, "data": solver.data_hist, "pde": solver.pde_hist},
        os.path.join(out_dir, "loss.png"), title="CViT periodic Burgers")

    Nx = x_test.shape[1]
    xs = np.linspace(0.0, dx * Nx, Nx, endpoint=False)
    ys = np.linspace(0.0, dy * Nx, Nx, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    plot_prediction_2d(X, Y, np.array(u_pred_test[0]), np.array(y_test[0, ..., 0]),
                       os.path.join(out_dir, "prediction.png"),
                       title="CViT periodic Burgers — sample 0")

    save_predictions(out_dir, coords={"x": X, "y": Y},
                     outputs={"u_pred": np.array(u_pred_test)},
                     exact={"u_exact": np.array(y_test[..., 0])})

    if (cfg_get(out, "save_params", default=True) if out else True):
        solver.save_checkpoint(out_dir, metadata={
            "problem": "cvit2d_burgers",
            "network": net_cfg,
            "physics": {"nu_lo": float(cfg.physics.nu_lo),
                       "nu_hi": float(cfg.physics.nu_hi)},
            "delta_scale": delta_scale,
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
    run_cvit2d_burgers(load_config(cfg_path))
