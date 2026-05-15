"""1-D Burgers PINN.

Run directly or via the CLI:

    python examples/burgers/burgers.py                   # uses config.yaml
    python examples/burgers/burgers.py myconfig.yaml     # custom config
    python -m underPINN run examples/burgers/config.yaml

All parameters live in config.yaml — edit there, no code changes needed.
"""
from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.mlp import MLP, FourierMLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


def _build_model(net_cfg):
    layers = net_cfg.layers
    if cfg_get(net_cfg, "type", default="mlp") == "fourier_mlp":
        return FourierMLP(
            layers=layers,
            n_fourier=cfg_get(net_cfg, "n_fourier", default=16),
            sigma=cfg_get(net_cfg, "sigma", default=2.0),
        )
    return MLP(layers=layers)


def _make_data(data_cfg, seed: int):
    T    = data_cfg.T
    N_r  = cfg_get(data_cfg, "n_collocation", default=6000)
    N_ic = cfg_get(data_cfg, "n_ic", default=200)
    N_bc = cfg_get(data_cfg, "n_bc", default=300)

    rng = np.random.default_rng(seed)
    x_r  = rng.uniform(-1.0, 1.0, N_r).astype(np.float32)
    t_r  = rng.uniform(0.0,  T,   N_r).astype(np.float32)

    x_ic = np.linspace(-1.0, 1.0, N_ic, dtype=np.float32)
    u_ic = (-np.sin(np.pi * x_ic)).astype(np.float32)

    t_bc = rng.uniform(0.0, T, N_bc).astype(np.float32)
    x_bc = np.concatenate([np.full(N_bc, -1., np.float32),
                            np.full(N_bc,  1., np.float32)])
    t_bc = np.concatenate([t_bc, t_bc])
    u_bc = np.zeros_like(x_bc)

    return (jnp.array(x_r),  jnp.array(t_r),
            jnp.array(x_ic), jnp.array(u_ic),
            jnp.array(x_bc), jnp.array(t_bc), jnp.array(u_bc))


def run_burgers(cfg) -> dict:
    """Train a PINN on 1-D Burgers and save outputs."""
    tr   = cfg.training
    seed = cfg_get(tr, "seed",     default=0)
    out  = cfg_get(cfg, "output",  default=None)
    out_dir = cfg_get(out, "dir",  default="outputs/burgers") if out else "outputs/burgers"
    os.makedirs(out_dir, exist_ok=True)

    model  = _build_model(cfg.network)
    pde    = BurgersPDE(model, nu=cfg.physics.nu)
    loss   = PINNLoss(
        model, pde,
        ic_weight=cfg_get(cfg.loss, "ic_weight", default=100.0),
        bc_weight=cfg_get(cfg.loss, "bc_weight", default=10.0),
        rba=bool(cfg_get(cfg.loss, "rba", default=False)),
    )
    solver = FBPINNSolver(model, pde, loss=loss)
    solver.init(jax.random.PRNGKey(seed))

    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    patience  = cfg_get(tr, "early_stopping_patience", default=None)

    callbacks = [ConsoleLogger(log_every=log_every)]
    if patience:
        callbacks.append(EarlyStopping(patience=int(patience)))

    tc = TrainingConfig(
        epochs      = epochs,
        lr          = lr,
        lr_schedule = optax.cosine_decay_schedule(lr, epochs, alpha=lr_alpha),
        batch_r     = cfg_get(tr, "batch_r", default=2048),
        batch_i     = cfg_get(tr, "batch_i", default=256),
        batch_b     = cfg_get(tr, "batch_b", default=256),
        log_every   = log_every,
        callbacks   = callbacks,
        n_scan_steps        = cfg_get(tr, "n_scan_steps",        default=1),
        resample_period     = cfg_get(tr, "resample_period",     default=0),
        resample_candidates = cfg_get(tr, "resample_candidates", default=0),
        resample_k          = cfg_get(tr, "resample_k",          default=1.0),
    )

    data = _make_data(cfg.data, seed)
    solver.train(*data, config=tc)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    x_r, t_r = data[0], data[1]
    pts_r = jnp.stack([x_r, t_r], axis=1)
    u_pred_r = model.apply(solver.params, pts_r)[:, 0]
    save_predictions(
        out_dir,
        coords  = {"x": np.array(x_r), "t": np.array(t_r)},
        outputs = {"u_pred": np.array(u_pred_r)},
    )

    # ── Solution plot on a fine (x, t) grid ──────────────────────────────────
    T   = float(cfg.data.T)
    Nx, Nt = 200, 100
    x_plt = jnp.linspace(-1.0, 1.0, Nx)
    t_plt = jnp.linspace(0.0, T, Nt)
    XX, TT = jnp.meshgrid(x_plt, t_plt, indexing="ij")   # shape (Nx, Nt)
    pts_g  = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
    u_grid = np.array(model.apply(solver.params, pts_g)[:, 0].reshape(Nx, Nt))

    fig_s, ax_s = plt.subplots(figsize=(8, 4))
    cf = ax_s.contourf(np.array(x_plt), np.array(t_plt), u_grid.T,
                       50, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(cf, ax=ax_s, label="u")
    ax_s.set_xlabel("x"); ax_s.set_ylabel("t")
    ax_s.set_title(f"Burgers PINN  ν={cfg.physics.nu}")
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_s)

    # ── Loss history ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(solver.loss_hist, lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Burgers ν={cfg.physics.nu}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    cfg_out = cfg_get(cfg, "output", default=None)
    if cfg_get(cfg_out, "save_params", default=True) if cfg_out else True:
        net_cfg = cfg.network
        solver.save_checkpoint(out_dir, metadata={
            "problem": "burgers",
            "network": {"type": cfg_get(net_cfg, "type", default="mlp"),
                        "layers": list(net_cfg.layers)},
            "physics": {"nu": float(cfg.physics.nu)},
        })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "loss_hist": solver.loss_hist}


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_burgers(load_config(cfg_path))
