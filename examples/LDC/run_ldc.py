"""2-D Lid-Driven Cavity (LDC) PINN.

Run directly or via the CLI:

    python examples/LDC/run_ldc.py              # uses config.yaml
    python examples/LDC/run_ldc.py myconfig.yaml
    python -m underPINN run examples/LDC/config.yaml

Geometry: unit square [0,1]²
  Lid: top wall (y=1), u=1, v=0
  No-slip: remaining three walls, u=v=0

Uses the FBPINN + SimpleGate architecture with LDCSolver.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.fbpinn import FBPINN
from underPINN.nn.attention import SimpleGate
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.solver.ldc_solver import LDCSolver, LDCInputWrapper
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


def _build_geometry(n_col: int, n_per_edge: int, seed: int):
    """Sample interior collocation and boundary points for the unit-square LDC."""
    try:
        from shapely.geometry import Polygon
        from underPINN.geometry.shapely_geom import ShapelyPolygon
        poly  = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        x_col = poly.sample(n_col, seed=seed)
    except Exception:
        # Fall back to uniform random if Shapely is unavailable
        rng   = np.random.default_rng(seed)
        x_col = rng.uniform(0.0, 1.0, (n_col, 2)).astype(np.float32)

    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float32)

    # Lid (y = 1): u = 1, v = 0
    x_lid    = np.stack([t,              np.ones_like(t)],  axis=1)
    # No-slip walls: left (x=0), right (x=1), bottom (y=0)
    w_left   = np.stack([np.zeros_like(t), t],              axis=1)
    w_right  = np.stack([np.ones_like(t),  t],              axis=1)
    w_bot    = np.stack([t,              np.zeros_like(t)], axis=1)
    x_noslip = np.concatenate([w_left, w_right, w_bot], axis=0)

    return x_col, x_lid, x_noslip


def run_ldc(cfg) -> dict:
    """Train a PINN on the 2-D Lid-Driven Cavity (LDC) problem."""
    # ── Unpack config ─────────────────────────────────────────────────────────
    net_cfg = cfg.network
    tr      = cfg.training
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/ldc") if out else "outputs/ldc"
    os.makedirs(out_dir, exist_ok=True)

    Re         = float(cfg.physics.Re)
    epochs     = int(tr.epochs)
    lr         = float(tr.lr)
    lr_alpha   = float(cfg_get(tr, "lr_alpha",   default=0.01))
    batch_r    = int(cfg_get(tr, "batch_r",      default=2048))
    log_every  = int(cfg_get(tr, "log_every",    default=500))
    patience   = int(cfg_get(tr, "early_stopping_patience", default=600))
    seed       = int(cfg_get(tr, "seed",         default=0))

    n_col      = int(cfg_get(cfg.data, "n_collocation", default=8000))
    n_per_edge = int(cfg_get(cfg.data, "n_bc",          default=800))

    print(f"LDC:  Re={Re},  epochs={epochs},  n_col={n_col}")

    # ── Model (single subdomain covering [0,1]²) ──────────────────────────────
    layers = list(net_cfg.layers)
    shifts = jnp.array([[0.5, 0.5]])
    xs_min = jnp.array([[0.0, 0.0]])
    xs_max = jnp.array([[1.0, 1.0]])
    smins  = jnp.array([[0.4, 0.4]])
    smaxs  = jnp.array([[0.4, 0.4]])

    model = FBPINN(
        layers=layers,
        shifts=shifts,
        xs_min=xs_min,
        xs_max=xs_max,
        smins=smins,
        smaxs=smaxs,
        attention_cls=SimpleGate,
    )
    pde = NavierStokesPDE(model, Re=Re)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )

    solver = LDCSolver(model, pde, optimizer=optimizer)
    solver.init(jax.random.PRNGKey(seed))

    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        lr_schedule=lr_sched,
        batch_r=batch_r,
        seed=seed,
        log_every=log_every,
        callbacks=[
            ConsoleLogger(log_every=log_every),
            # EarlyStopping(patience=patience),
        ],
    )

    # ── Geometry ──────────────────────────────────────────────────────────────
    print("Generating LDC geometry …")
    x_col, x_lid, x_noslip = _build_geometry(n_col, n_per_edge, seed=seed)

    inputs = LDCInputWrapper(
        col    = jnp.array(x_col,    dtype=jnp.float32),
        inlet  = jnp.array(x_lid,    dtype=jnp.float32),
        noslip = jnp.array(x_noslip, dtype=jnp.float32),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    solver.train(inputs, config=config)

    # ── Evaluate on grid ──────────────────────────────────────────────────────
    x = jnp.linspace(0.0, 1.0, 201)
    y = jnp.linspace(0.0, 1.0, 201)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    grid   = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    pred   = model.apply(solver.params, grid)
    u_g = np.array(pred[:, 0].reshape(201, 201))
    v_g = np.array(pred[:, 1].reshape(201, 201))
    p_g = np.array(pred[:, 2].reshape(201, 201))
    x_np = np.array(x);  y_np = np.array(y)

    # ── Save predictions at collocation points ────────────────────────────────
    pred_col = np.array(model.apply(solver.params, inputs.col))
    save_predictions(
        out_dir,
        coords  = {"x": np.array(inputs.col[:, 0]),
                   "y": np.array(inputs.col[:, 1])},
        outputs = {"u_pred": pred_col[:, 0],
                   "v_pred": pred_col[:, 1],
                   "p_pred": pred_col[:, 2]},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────


    nx, ny = 201, 201   # <-- keep these fixed
    df = pd.read_csv("re100.csv", skipinitialspace=True)
    df = df.sort_values(by=["y-coordinate", "x-coordinate"], ascending=[True, True]).reset_index(drop=True)

    x_c = df["x-coordinate"].values.reshape(ny, nx)
    y_c = df["y-coordinate"].values.reshape(ny, nx)
    p_c = df["pressure"].values.reshape(ny, nx)
    u_c = df["x-velocity"].values.reshape(ny, nx)
    v_c = df["y-velocity"].values.reshape(ny, nx)
    u_mag_c = np.sqrt(u_c**2 + v_c**2)

    plt.figure(figsize=(8,6))
    plt.plot(u_c[:, 100],  y_c[:,100],  'o', label='CFD Data', markerfacecolor='none', markeredgecolor='blue', markersize=6, markeredgewidth=1.5)
    plt.plot(u_g[100,:], YY[100,:], label='PINN Prediction', color='red', linewidth=2)
    plt.xlabel('U-velocity', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    # plt.title('Lid-Driven Cavity Flow at Re=1000', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Re100_comp.png"), bbox_inches='tight')
    plt.close()



    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, label in zip(axes, [u_g, v_g, p_g], ["u", "v", "p"]):
        cf = ax.contourf(x_np, y_np, field.T, levels=50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(f"PINN: {label}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"Lid-Driven Cavity  Re={Re}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(solver.loss_hist, label="Total", lw=1.2)
    if solver.phys_hist:
        ax2.semilogy(solver.phys_hist, label="PDE",  alpha=0.7)
    if solver.bc_hist:
        ax2.semilogy(solver.bc_hist,   label="BC",   alpha=0.7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title(f"LDC Re={Re} — training loss")
    ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    np.savez(os.path.join(out_dir, "solution.npz"),
             u=u_g, v=v_g, p=p_g, x=np.array(XX), y=np.array(YY))

    # ── Model checkpoint ──────────────────────────────────────────────────────
    solver.save_checkpoint(out_dir, metadata={
        "problem": "ldc",
        "network": {"type": "fbpinn", "layers": layers},
        "physics": {"Re": Re},
    })

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": solver.params, "loss_hist": solver.loss_hist}


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_ldc(load_config(cfg_path))
