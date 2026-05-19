"""2-D k-ε RANS — Backward-Facing Step (BFS) at Re = 10 000.

Run directly or via the CLI:

    python examples/K-Epsilon/turbulence.py              # uses config.yaml
    python examples/K-Epsilon/turbulence.py myconfig.yaml
    python -m underPINN run examples/K-Epsilon/config.yaml

Geometry (backward-facing step):
  Inlet : x = 0,    0.9423 ≤ y ≤ 1.9423   (u = 1, v = 0 + small k, ε)
  Outlet: x = 35,   0      ≤ y ≤ 1.9423   (zero pressure gradient)
  Walls : top (y=1.9423), bottom step, step-face, step-floor (no-slip u=v=0)

Network: FBPINN  (x, y) → (u, v, p, k, ε)
  k and ε are forced positive via softplus applied inside every subdomain net.

Data file: CSV with columns [x-coordinate, y-coordinate, x-velocity, y-velocity,
           pressure, turb-kinetic-energy, turb-diss-rate].  Specified in config as
           ``data_file``.  A sparse subset (default 500 points) is used for the
           supervised data-loss term.
"""
from __future__ import annotations

import os
import pathlib

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.fbpinn import FBPINN
from underPINN.pde.k_epsilon import KEpsilonPDE
from underPINN.solver.rans_solver import RANSSolver, RANSInputWrapper
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.io import save_predictions


# ---------------------------------------------------------------------------
# Output transform — enforces k > 0 and ε > 0 inside the subdomain networks
# ---------------------------------------------------------------------------

def k_eps_positivity(x):
    """Network output transform: (u, v, p, f_k, f_ε) → (u, v, p, k, ε).

    Applies softplus to the last two outputs so k > 0 and ε > 0 always,
    making the k-ε transport equations well-posed throughout training.
    Uses softplus rather than exp for numerical stability.
    """
    uvp = x[..., :3]
    k   = jax.nn.softplus(x[..., 3:4]) + 1e-8
    eps = jax.nn.softplus(x[..., 4:5]) + 1e-8
    return jnp.concatenate([uvp, k, eps], axis=-1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_bfs_geometry(n_col, n_inlet, n_outlet, n_noslip, seed):
    """Sample collocation and boundary points for the BFS domain."""
    try:
        from shapely.geometry import Polygon
        from underPINN.geometry.shapely_geom import ShapelyPolygon
        vertices = [(0, 1.9423), (35, 1.9423), (35, 0),
                    (5, 0),      (5, 0.9423),   (0, 0.9423)]
        poly  = ShapelyPolygon(vertices)
        x_col = poly.sample_near_boundary(n_col, decay=2.0, seed=seed)
    except Exception:
        # Fallback: uniform random rejection inside the step polygon
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < n_col:
            over = n_col * 4
            x = rng.uniform(0.0, 35.0, over).astype(np.float32)
            y = rng.uniform(0.0, 1.9423, over).astype(np.float32)
            # Keep points in the BFS domain (above step for x < 5)
            mask = (x >= 5.0) | (y >= 0.9423)
            valid = np.stack([x[mask], y[mask]], axis=1)
            pts.append(valid)
        x_col = np.concatenate(pts)[:n_col]

    # Inlet: x=0, 0.9423 ≤ y ≤ 1.9423
    y_in  = np.linspace(0.9423, 1.9423, n_inlet, dtype=np.float32)
    x_inlet = np.stack([np.zeros(n_inlet, np.float32), y_in], axis=1)

    # Outlet: x=35, 0 ≤ y ≤ 1.9423
    y_out   = np.linspace(0.0, 1.9423, n_outlet, dtype=np.float32)
    x_outlet = np.stack([np.full(n_outlet, 35.0, np.float32), y_out], axis=1)

    # No-slip walls (top, bottom after step, step-face, step-floor)
    n_each = max(n_noslip // 4, 50)
    w_top  = np.stack([np.linspace(0,  35, n_each, dtype=np.float32),
                       np.full(n_each, 1.9423, np.float32)], axis=1)
    w_bot  = np.stack([np.linspace(5,  35, n_each, dtype=np.float32),
                       np.zeros(n_each, np.float32)], axis=1)
    w_face = np.stack([np.full(n_each, 5.0, np.float32),
                       np.linspace(0, 0.9423, n_each, np.float32)], axis=1)
    w_floor = np.stack([np.linspace(0, 5, n_each, dtype=np.float32),
                        np.full(n_each, 0.9423, np.float32)], axis=1)
    x_noslip = np.concatenate([w_top, w_bot, w_face, w_floor], axis=0)

    return (x_col.astype(np.float32), x_inlet,
            x_outlet, x_noslip)


def _load_cfd_data(data_file: str, n_pts: int, seed: int):
    """Load sparse CFD reference data from CSV.

    Raises FileNotFoundError with a clear message if the file is missing.
    """
    import pandas as pd

    path = pathlib.Path(data_file)
    if not path.exists():
        raise FileNotFoundError(
            f"CFD reference data not found: {path}\n"
            f"Provide the CSV at that path or update 'data_file' in config.yaml.\n"
            f"Expected columns: x-coordinate, y-coordinate, x-velocity, "
            f"y-velocity, pressure, turb-kinetic-energy, turb-diss-rate"
        )

    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.strip()

    rng = np.random.default_rng(seed)
    n   = min(n_pts, len(df))
    idx = rng.choice(len(df), n, replace=False)
    sel = df.iloc[idx]

    x_data = sel[["x-coordinate", "y-coordinate"]].values.astype(np.float32)
    u_data = sel[["x-velocity", "y-velocity", "pressure",
                  "turb-kinetic-energy", "turb-diss-rate"]].values.astype(np.float32)
    return x_data, u_data


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_turbulence(cfg) -> dict:
    """Train a k-ε RANS PINN on the 2-D backward-facing step."""

    # ── Config ────────────────────────────────────────────────────────────────
    net_cfg  = cfg.network
    tr       = cfg.training
    out      = cfg_get(cfg, "output", default=None)
    out_dir  = cfg_get(out, "dir", default="outputs/k_epsilon") if out else "outputs/k_epsilon"
    os.makedirs(out_dir, exist_ok=True)

    Re         = float(cfg.physics.Re)
    epochs     = int(tr.epochs)
    lr         = float(tr.lr)
    batch_size = int(cfg_get(tr, "batch_size", default=2000))
    log_every  = int(cfg_get(tr, "log_every",  default=10))
    seed       = int(cfg_get(tr, "seed",        default=42))

    d = cfg.data
    n_col    = int(cfg_get(d, "n_collocation", default=80000))
    n_inlet  = int(cfg_get(d, "n_inlet",       default=200))
    n_outlet = int(cfg_get(d, "n_outlet",       default=200))
    n_noslip = int(cfg_get(d, "n_noslip",       default=900))
    n_data   = int(cfg_get(d, "n_data",         default=500))

    data_file = cfg_get(cfg, "data_file", default="Re10000")
    layers    = list(net_cfg.layers)

    print(f"k-ε RANS:  Re={Re},  epochs={epochs},  n_col={n_col}")

    # ── Geometry ──────────────────────────────────────────────────────────────
    print("Building BFS geometry …")
    x_col, x_inlet, x_outlet, x_noslip = _build_bfs_geometry(
        n_col, n_inlet, n_outlet, n_noslip, seed=seed)

    # ── CFD reference data ────────────────────────────────────────────────────
    print(f"Loading CFD reference data from '{data_file}' …")
    x_data, u_data = _load_cfd_data(data_file, n_data, seed=seed)
    print(f"  Loaded {len(x_data)} reference points.")

    inputs = RANSInputWrapper(
        col    = jnp.array(x_col),
        inlet  = jnp.array(x_inlet),
        noslip = jnp.array(x_noslip),
        outlet = jnp.array(x_outlet),
        data_x = jnp.array(x_data),
        data_u = jnp.array(u_data),
    )

    # ── Model — 3-subdomain FBPINN with k-ε positivity transform ─────────────
    # Three overlapping subdomains along x: [0,12], [12,24], [24,35]
    shifts = jnp.array([[6.0, 1.0], [18.0, 1.0], [30.0, 1.0]])
    xs_min = jnp.array([[0.0, 0.0],  [12.0, 0.0], [24.0, 0.0]])
    xs_max = jnp.array([[12.0, 2.0], [24.0, 2.0], [35.0, 2.0]])
    smins  = jnp.ones_like(xs_min)
    smaxs  = jnp.ones_like(xs_max)

    model = FBPINN(
        layers       = layers,
        shifts       = shifts,
        xs_min       = xs_min,
        xs_max       = xs_max,
        smins        = smins,
        smaxs        = smaxs,
        out_transform = k_eps_positivity,   # enforces k > 0, ε > 0
    )

    pde = KEpsilonPDE(model, Re=Re)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales=dict(cfg_get(
            tr, "lr_schedule",
            default={"boundaries_and_scales": {2000: 0.5, 4000: 0.5}}
        ).boundaries_and_scales if hasattr(
            cfg_get(tr, "lr_schedule", default=None), "boundaries_and_scales"
        ) else {int(k): float(v) for k, v in [("2000", 0.5), ("4000", 0.5)]})
    )
    optimizer = optax.adam(learning_rate=schedule)

    solver = RANSSolver(model, pde, optimizer)

    # Initialise parameters
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training …")
    final_params, loss_hist = solver.train(
        params, inputs,
        epochs=epochs, batch_size=batch_size, seed=seed)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # Predictions at interior collocation points
    pred_col = np.array(model.apply(final_params, inputs.col))
    save_predictions(
        out_dir,
        coords  = {"x": np.array(inputs.col[:, 0]),
                   "y": np.array(inputs.col[:, 1])},
        outputs = {"u_pred":   pred_col[:, 0],
                   "v_pred":   pred_col[:, 1],
                   "p_pred":   pred_col[:, 2],
                   "k_pred":   pred_col[:, 3],
                   "eps_pred": pred_col[:, 4]},
    )

    # ── Solution plots on a regular grid ─────────────────────────────────────
    Nx, Ny = 350, 50
    x_g = jnp.linspace(0.0, 35.0, Nx, dtype=jnp.float32)
    y_g = jnp.linspace(0.0, 1.9423, Ny, dtype=jnp.float32)
    XX, YY = jnp.meshgrid(x_g, y_g, indexing="ij")   # (Nx, Ny)
    grid   = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    pred_g = np.array(model.apply(final_params, grid))

    # Mask points inside the step (x < 5, y < 0.9423)
    mask = ~((np.array(XX.ravel()) < 5.0) & (np.array(YY.ravel()) < 0.9423))
    x_np = np.array(XX.ravel())[mask]
    y_np = np.array(YY.ravel())[mask]
    pred_m = pred_g[mask]

    def _plot_field(values, name, filename, cmap="jet"):
        fig, ax = plt.subplots(figsize=(15, 4))
        sc = ax.scatter(x_np, y_np, c=values, s=1, cmap=cmap)
        plt.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"PINN: {name}  (Re={Re})")
        # Draw step outline
        step_x = [0, 5, 5, 0]
        step_y = [0, 0, 0.9423, 0.9423]
        ax.fill(step_x, step_y, color="gray", alpha=0.8)
        fig.tight_layout()
        fig.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close(fig)

    _plot_field(pred_m[:, 0], "u",   os.path.join(out_dir, "solution_u.png"))
    _plot_field(pred_m[:, 1], "v",   os.path.join(out_dir, "solution_v.png"))
    _plot_field(pred_m[:, 2], "p",   os.path.join(out_dir, "solution_p.png"), cmap="plasma")
    _plot_field(pred_m[:, 3], "k",   os.path.join(out_dir, "solution_k.png"), cmap="hot")
    _plot_field(pred_m[:, 4], "eps", os.path.join(out_dir, "solution_eps.png"), cmap="hot")

    # ── Loss history ──────────────────────────────────────────────────────────
    if loss_hist:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.semilogy(loss_hist, lw=1.2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (avg over mini-batches)")
        ax.set_title(f"k-ε RANS BFS  Re={Re}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    save_checkpoint(final_params, out_dir, metadata={
        "problem": "k_epsilon",
        "network": {"type": "fbpinn", "layers": layers},
        "physics": {"Re": Re},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": final_params, "loss_hist": loss_hist}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    _HERE    = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_turbulence(load_config(cfg_path))
