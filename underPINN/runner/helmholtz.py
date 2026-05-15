"""Runner for the 2-D Helmholtz equation.

Expected config sections
------------------------
problem  : helmholtz

network:
  type      : fourier_mlp     # or mlp
  layers    : [2, 128, 128, 128, 1]
  n_fourier : 32
  sigma     : 4.0             # should roughly match wavenumber k

physics:
  k : 4.0                     # wavenumber — Δu + k²u = f

data:
  n_collocation : 8000
  n_bc          : 600         # points per edge (4 edges → 4 × n_bc total)

training:
  epochs    : 10000
  lr        : 1.0e-3
  lr_alpha  : 0.01            # cosine-decay final LR = lr × lr_alpha
  batch_r   : 2048
  batch_b   : 256
  log_every : 1000
  seed      : 0

loss:
  bc_weight : 100.0
  rba       : true            # residual-based adaptive BC weighting

output:
  dir : outputs/helmholtz
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
from underPINN.nn.mlp import FourierMLP, MLP
from underPINN.pde.helmholtz import HelmholtzPDE
from underPINN.losses.steady_loss import SteadyLoss
from underPINN.solver.steady_solver import SteadySolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


def run_helmholtz(cfg) -> dict:
    """Train a PINN on the 2-D Helmholtz equation  Δu + k²u = f.

    Manufactured exact solution:  u(x,y) = sin(πx) sin(πy)
    Source term:  f = -(2π² − k²) sin(πx) sin(πy)
    BCs:  u = 0 on all four edges of [0,1]²
    """
    # ── Unpack ────────────────────────────────────────────────────────────────
    tr      = cfg.training
    seed    = cfg_get(tr, "seed",    default=0)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/helmholtz") if out else "outputs/helmholtz"
    os.makedirs(out_dir, exist_ok=True)

    k         = float(cfg.physics.k)
    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=1000)
    batch_r   = cfg_get(tr, "batch_r",   default=2048)
    batch_b   = cfg_get(tr, "batch_b",   default=256)

    N_r          = cfg_get(cfg.data, "n_collocation", default=8000)
    n_per_edge   = cfg_get(cfg.data, "n_bc",          default=600)
    bc_weight    = cfg_get(cfg.loss, "bc_weight",     default=100.0)
    rba          = cfg_get(cfg.loss, "rba",           default=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    net_cfg = cfg.network
    net_type = cfg_get(net_cfg, "type", default="fourier_mlp").lower()
    if net_type == "fourier_mlp":
        n_fourier = cfg_get(net_cfg, "n_fourier", default=32)
        sigma     = cfg_get(net_cfg, "sigma",     default=float(k))
        model = FourierMLP(layers=net_cfg.layers, n_fourier=n_fourier, sigma=sigma)
    else:
        model = MLP(layers=net_cfg.layers)

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    xy_r = jnp.array(rng.uniform(0.0, 1.0, (N_r, 2)).astype(np.float32))

    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float32)
    bottom = np.stack([t,              np.zeros_like(t)], axis=1)
    top    = np.stack([t,              np.ones_like(t)],  axis=1)
    left   = np.stack([np.zeros_like(t), t],              axis=1)
    right  = np.stack([np.ones_like(t),  t],              axis=1)
    xy_b = jnp.array(np.concatenate([bottom, top, left, right], axis=0))
    u_b  = jnp.zeros(xy_b.shape[0])

    # ── PDE + loss + solver ───────────────────────────────────────────────────
    pde    = HelmholtzPDE(model, k=k)
    loss   = SteadyLoss(model, pde, bc_weight=bc_weight, rba=rba)
    solver = SteadySolver(model, pde, loss)
    solver.init(jax.random.PRNGKey(seed))

    lr_sched = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        lr_schedule=lr_sched,
        batch_r=batch_r,
        batch_b=batch_b,
        seed=seed,
        log_every=log_every,
        callbacks=[
            ConsoleLogger(log_every=log_every),
            EarlyStopping(patience=int(cfg_get(tr, "early_stopping_patience",
                                               default=max(600, epochs // 15)))),
        ],
    )

    solver.train(xy_r, xy_b, u_b, config=config)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    N = 200
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

    u_pred  = model.apply(solver.params, grid)[:, 0].reshape(N, N)
    u_exact = pde.exact(grid).reshape(N, N)
    rel_l2  = float(jnp.linalg.norm(u_pred - u_exact) / (jnp.linalg.norm(u_exact) + 1e-10))
    max_ae  = float(jnp.max(jnp.abs(u_pred - u_exact)))
    print(f"\nRelative L2 error : {rel_l2:.4e}")
    print(f"Max absolute error: {max_ae:.4e}")

    # ── Save predictions at collocation points ─────────────────────────────────
    u_pred_r  = model.apply(solver.params, xy_r)[:, 0]
    u_exact_r = pde.exact(xy_r)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xy_r[:, 0]), "y": np.array(xy_r[:, 1])},
        outputs = {"u_pred": np.array(u_pred_r)},
        exact   = {"u_exact": np.array(u_exact_r)},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    x_np = np.array(x)
    y_np = np.array(y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes,
        [u_exact, u_pred, jnp.abs(u_pred - u_exact)],
        ["Exact  u(x,y)", "PINN  u(x,y)", f"|Error|  (Rel-L2={rel_l2:.2e})"],
    ):
        cf = ax.contourf(x_np, y_np, np.array(field), 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"Helmholtz  Δu + k²u = f   (k={k})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(solver.loss_hist, label="Total",  lw=1.2)
    ax2.semilogy(solver.pde_hist,  label="PDE",    alpha=0.7)
    ax2.semilogy(solver.bc_hist,   label="BC",     alpha=0.7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title(f"Helmholtz (k={k}) — training loss")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))

    # ── Model checkpoint ──────────────────────────────────────────────────────
    solver.save_checkpoint(out_dir, metadata={
        "problem": "helmholtz",
        "network": {"type": net_type, "layers": list(net_cfg.layers)},
        "physics": {"k": k},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "rel_l2": rel_l2, "max_ae": max_ae,
            "loss_hist": solver.loss_hist}
