"""Runner for the 2-D steady-state heat equation (Poisson problem).

Expected config sections
------------------------
problem  : heat_forward

network:
  type   : mlp            # or fourier_mlp
  layers : [2, 64, 64, 64, 1]

physics:
  alpha : 0.01            # thermal diffusivity (unused for the steady case
                          #  but kept for config compatibility)

data:
  n_collocation : 5000
  n_ic          : 200     # reused as n_per_edge for boundary sampling
  n_bc          : 300     # overrides n_ic if present

training:
  epochs    : 5000
  lr        : 1.0e-3
  lr_alpha  : 0.01
  batch_r   : 2048
  batch_b   : 256
  log_every : 500
  seed      : 0

loss:
  ic_weight : 100.0       # aliased as bc_weight for the steady problem
  bc_weight : 100.0
  rba       : true

output:
  dir : outputs/heat_forward
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
from underPINN.pde.heat import SteadyHeatPDE
from underPINN.losses.steady_loss import SteadyLoss
from underPINN.solver.steady_solver import SteadySolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


def _source(x, y):
    """f(x,y) = 2π² sin(πx) sin(πy) — RHS for the manufactured solution."""
    return 2.0 * jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def run_heat_forward(cfg) -> dict:
    """Train a PINN on the 2-D steady heat/Poisson problem  ∇²u = -f.

    Manufactured exact solution: u(x,y) = sin(πx) sin(πy)
    BCs: u = 0 on all four edges of [0,1]²
    """
    # ── Unpack ────────────────────────────────────────────────────────────────
    tr      = cfg.training
    seed    = cfg_get(tr, "seed",    default=0)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/heat_forward") if out else "outputs/heat_forward"
    os.makedirs(out_dir, exist_ok=True)

    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    batch_r   = cfg_get(tr, "batch_r",   default=2048)
    batch_b   = cfg_get(tr, "batch_b",   default=256)

    N_r        = cfg_get(cfg.data, "n_collocation", default=5000)
    # n_bc takes precedence; fall back to n_ic for configs that use the wave-style naming
    n_per_edge = cfg_get(cfg.data, "n_bc", default=None) or cfg_get(cfg.data, "n_ic", default=200)

    bc_weight  = cfg_get(cfg.loss, "bc_weight",
                         default=cfg_get(cfg.loss, "ic_weight", default=100.0))
    rba        = cfg_get(cfg.loss, "rba", default=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    net_cfg  = cfg.network
    net_type = cfg_get(net_cfg, "type", default="mlp").lower()
    if net_type == "fourier_mlp":
        n_fourier = cfg_get(net_cfg, "n_fourier", default=16)
        sigma     = cfg_get(net_cfg, "sigma",     default=2.0)
        model = FourierMLP(layers=net_cfg.layers, n_fourier=n_fourier, sigma=sigma)
    else:
        model = MLP(layers=net_cfg.layers)

    # ── Data ──────────────────────────────────────────────────────────────────
    rng  = np.random.default_rng(seed)
    xy_r = jnp.array(rng.uniform(0.0, 1.0, (N_r, 2)).astype(np.float32))

    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float32)
    bottom = np.stack([t,              np.zeros_like(t)], axis=1)
    top    = np.stack([t,              np.ones_like(t)],  axis=1)
    left   = np.stack([np.zeros_like(t), t],              axis=1)
    right  = np.stack([np.ones_like(t),  t],              axis=1)
    xy_b = jnp.array(np.concatenate([bottom, top, left, right], axis=0))
    u_b  = jnp.zeros(xy_b.shape[0])

    # ── PDE + loss + solver ───────────────────────────────────────────────────
    pde    = SteadyHeatPDE(model, source_fn=_source)
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
            EarlyStopping(patience=max(500, epochs // 10)),
        ],
    )

    solver.train(xy_r, xy_b, u_b, config=config)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    NX = NY = 100
    x   = np.linspace(0.0, 1.0, NX, dtype=np.float32)
    y   = np.linspace(0.0, 1.0, NY, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy_eval = jnp.array(np.stack([xx.ravel(), yy.ravel()], axis=1))

    u_pred  = pde.u(solver.params, xy_eval)
    u_exact = pde.exact(xy_eval)
    rel_l2  = float(jnp.linalg.norm(u_pred - u_exact) / (jnp.linalg.norm(u_exact) + 1e-10))
    max_ae  = float(jnp.max(jnp.abs(u_pred - u_exact)))
    print(f"\nRelative L2 error : {rel_l2:.4e}")
    print(f"Max absolute error: {max_ae:.4e}")

    # ── Save predictions ──────────────────────────────────────────────────────
    u_pred_r  = pde.u(solver.params, xy_r)
    u_exact_r = pde.exact(xy_r)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xy_r[:, 0]), "y": np.array(xy_r[:, 1])},
        outputs = {"u_pred": np.array(u_pred_r)},
        exact   = {"u_exact": np.array(u_exact_r)},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    u_pred_g  = np.array(u_pred.reshape(NX, NY))
    u_exact_g = np.array(u_exact.reshape(NX, NY))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes,
        [u_exact_g, u_pred_g, np.abs(u_pred_g - u_exact_g)],
        ["Exact  u(x,y)", "PINN  u(x,y)", f"|Error|  (Rel-L2={rel_l2:.2e})"],
    ):
        cf = ax.contourf(x, y, field, 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("2-D Steady Heat: ∇²u = -2π² sin(πx) sin(πy)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(solver.loss_hist, label="Total", lw=1.2)
    ax2.semilogy(solver.pde_hist,  label="PDE",   alpha=0.7)
    ax2.semilogy(solver.bc_hist,   label="BC",    alpha=0.7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("2-D Steady Heat — training loss")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "rel_l2": rel_l2, "max_ae": max_ae,
            "loss_hist": solver.loss_hist}
