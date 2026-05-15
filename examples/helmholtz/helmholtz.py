"""2-D Helmholtz Equation PINN.

Run directly or via the CLI:

    python examples/helmholtz/helmholtz.py                 # uses config.yaml
    python examples/helmholtz/helmholtz.py myconfig.yaml   # custom config
    python -m underPINN run examples/helmholtz/config.yaml

Manufactured exact solution: u(x,y) = sin(πx) sin(πy)
Source: f = -(2π² − k²) sin(πx) sin(πy),  BCs: u = 0 on all edges.
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
    """Train a PINN on the 2-D Helmholtz equation  Δu + k²u = f."""
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

    N_r        = cfg_get(cfg.data, "n_collocation", default=8000)
    n_per_edge = cfg_get(cfg.data, "n_bc",          default=600)
    bc_weight  = cfg_get(cfg.loss, "bc_weight",     default=100.0)
    rba        = cfg_get(cfg.loss, "rba",           default=True)

    net_cfg  = cfg.network
    net_type = cfg_get(net_cfg, "type", default="fourier_mlp").lower()
    if net_type == "fourier_mlp":
        model = FourierMLP(layers=net_cfg.layers,
                           n_fourier=cfg_get(net_cfg, "n_fourier", default=32),
                           sigma=cfg_get(net_cfg, "sigma", default=float(k)))
    else:
        model = MLP(layers=net_cfg.layers)

    rng  = np.random.default_rng(seed)
    xy_r = jnp.array(rng.uniform(0.0, 1.0, (N_r, 2)).astype(np.float32))

    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float32)
    xy_b = jnp.array(np.concatenate([
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([t, np.ones_like(t)],  axis=1),
        np.stack([np.zeros_like(t), t], axis=1),
        np.stack([np.ones_like(t),  t], axis=1),
    ]))
    u_b = jnp.zeros(xy_b.shape[0])

    pde    = HelmholtzPDE(model, k=k)
    loss   = SteadyLoss(model, pde, bc_weight=bc_weight, rba=rba)
    solver = SteadySolver(model, pde, loss)
    solver.init(jax.random.PRNGKey(seed))

    lr_sched = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    config = TrainingConfig(
        epochs=epochs, lr=lr, lr_schedule=lr_sched,
        batch_r=batch_r, batch_b=batch_b, seed=seed, log_every=log_every,
        callbacks=[
            ConsoleLogger(log_every=log_every),
            EarlyStopping(patience=int(cfg_get(tr, "early_stopping_patience",
                                               default=max(600, epochs // 15)))),
        ],
    )
    solver.train(xy_r, xy_b, u_b, config=config)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    N = 200
    x = jnp.linspace(0.0, 1.0, N); y = jnp.linspace(0.0, 1.0, N)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    grid   = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    u_pred  = model.apply(solver.params, grid)[:, 0].reshape(N, N)
    u_exact = pde.exact(grid).reshape(N, N)
    rel_l2  = float(jnp.linalg.norm(u_pred - u_exact) / (jnp.linalg.norm(u_exact) + 1e-10))
    max_ae  = float(jnp.max(jnp.abs(u_pred - u_exact)))
    print(f"\nRelative L2 error : {rel_l2:.4e}")
    print(f"Max absolute error: {max_ae:.4e}")

    u_pred_r  = model.apply(solver.params, xy_r)[:, 0]
    u_exact_r = pde.exact(xy_r)
    save_predictions(out_dir,
                     coords={"x": np.array(xy_r[:, 0]), "y": np.array(xy_r[:, 1])},
                     outputs={"u_pred": np.array(u_pred_r)},
                     exact={"u_exact": np.array(u_exact_r)})

    x_np, y_np = np.array(x), np.array(y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(axes,
                                 [u_exact, u_pred, jnp.abs(u_pred - u_exact)],
                                 ["Exact", "PINN", f"|Error| (Rel-L2={rel_l2:.2e})"]):
        cf = ax.contourf(x_np, y_np, np.array(field), 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"Helmholtz  Δu + k²u = f   (k={k})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(solver.loss_hist, label="Total", lw=1.2)
    ax2.semilogy(solver.pde_hist,  label="PDE",   alpha=0.7)
    ax2.semilogy(solver.bc_hist,   label="BC",    alpha=0.7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title(f"Helmholtz (k={k}) — training loss")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    solver.save_checkpoint(out_dir, metadata={
        "problem": "helmholtz",
        "network": {"type": net_type, "layers": list(net_cfg.layers)},
        "physics": {"k": k},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "rel_l2": rel_l2, "max_ae": max_ae,
            "loss_hist": solver.loss_hist}


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_helmholtz(load_config(cfg_path))
