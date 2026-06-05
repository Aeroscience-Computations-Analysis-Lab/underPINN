"""Pure PINN — 2-D Steady Flow over a Circular Cylinder.

Faithful port of the reference notebook *PINN_Re40_pure.ipynb*: steady,
incompressible flow over a cylinder.  No training data — only Navier–Stokes
PDE residuals + boundary conditions.

Run directly or via the CLI:

    python examples/cylinder/cylinder_flow.py              # uses config.yaml
    python examples/cylinder/cylinder_flow.py myconfig.yaml
    python -m underPINN run examples/cylinder/config.yaml

PDE  (steady incompressible Navier–Stokes, ν = 1/Re):

    ∇·u = 0
    (u·∇)u + ∇p − ν ∇²u = 0

Boundary conditions:

    • Inlet   (left edge,  x = xmin):  u = U∞,  v = 0
    • Outlet  (right edge, x = xmax):  p = 0
    • Cylinder (no-slip)            :  u = v = 0
    • Top / bottom walls (symmetry) :  v = 0      (steady cylinder wake is
                                                   symmetric about y = 0)

Collocation strategy (notebook recipe):

    A fixed uniform interior pool (exterior to the cylinder, with a small
    buffer) plus a denser wake pool downstream.  Both pools are re-sampled
    together every ``resample_period`` epochs.  Training is full-batch.

Network : plain tanh MLP, (x, y) → (u, v, p).
Schedule: Adam + StepLR (halve the LR every ``lr_step`` epochs).
Loss    : L = w_pde · L_pde + w_bc · L_bc      (L_bc = inlet+outlet+body+wall)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.mlp import MLP, GatedMLP
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.geometry.cylinder import Cylinder2D
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager


# ---------------------------------------------------------------------------
# Collocation sampling helpers  (notebook style)
# ---------------------------------------------------------------------------

def _sample_box_exterior(geom, n, xlo, xhi, ylo, yhi, buffer, seed) -> np.ndarray:
    """Uniform points in a box, rejecting the body interior + a buffer shell."""
    rng = np.random.default_rng(seed)
    pts: list = []
    while len(pts) < n:
        batch = rng.uniform([xlo, ylo], [xhi, yhi],
                            size=(max(n * 4, 20_000), 2)).astype(np.float32)
        batch = batch[~geom.is_inside(batch)]
        if buffer > 0.0 and len(batch):
            batch = batch[geom.sdf(batch) > buffer]
        pts.extend(batch.tolist())
    return np.array(pts[:n], dtype=np.float32)


def _build_collocation(geom, n_interior, n_wake, dom, wake_box, buffer, seed):
    """Interior background pool + denser downstream wake pool, concatenated."""
    xmin, xmax, ymin, ymax = dom
    wxmin, wxmax, wymin, wymax = wake_box
    interior = _sample_box_exterior(geom, n_interior, xmin, xmax, ymin, ymax,
                                    buffer, seed)
    if n_wake > 0:
        wake = _sample_box_exterior(geom, n_wake, wxmin, wxmax, wymin, wymax,
                                    buffer, seed + 1)
        return np.concatenate([interior, wake], axis=0)
    return interior


def _edge_vertical(xval, ymin, ymax, n):
    y = np.linspace(ymin, ymax, n, dtype=np.float32)
    return np.stack([np.full(n, xval, np.float32), y], axis=1)


def _edge_horizontal(yval, xmin, xmax, n):
    x = np.linspace(xmin, xmax, n, dtype=np.float32)
    return np.stack([x, np.full(n, yval, np.float32)], axis=1)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_cylinder(cfg) -> dict:
    """Train a pure PINN on steady incompressible NS over a circular cylinder."""
    ph      = cfg.physics
    tr      = cfg.training
    lw      = cfg.loss
    dom_c   = cfg_get(cfg, "domain", default=None)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/cylinder") if out else "outputs/cylinder"
    os.makedirs(out_dir, exist_ok=True)

    Re    = float(ph.Re)
    R     = float(cfg_get(ph, "R",     default=0.5))
    U_inf = float(cfg_get(ph, "U_inf", default=1.0))
    cx    = float(cfg_get(ph, "cx",    default=0.0))
    cy    = float(cfg_get(ph, "cy",    default=0.0))

    xmin = float(cfg_get(dom_c, "xmin", default=-5.0)) if dom_c else -5.0
    xmax = float(cfg_get(dom_c, "xmax", default=15.0)) if dom_c else 15.0
    ymin = float(cfg_get(dom_c, "ymin", default=-5.0)) if dom_c else -5.0
    ymax = float(cfg_get(dom_c, "ymax", default= 5.0)) if dom_c else  5.0

    # ── Collocation / BC point counts ─────────────────────────────────────────
    d = cfg.data
    n_interior = int(cfg_get(d, "n_interior", default=10000))
    n_wake     = int(cfg_get(d, "n_wake",     default=3000))
    n_inlet    = int(cfg_get(d, "n_inlet",    default=300))
    n_outlet   = int(cfg_get(d, "n_outlet",   default=300))
    n_body     = int(cfg_get(d, "n_body",     default=300))
    n_wall     = int(cfg_get(d, "n_wall",     default=300))
    buffer     = float(cfg_get(d, "buffer",   default=0.05))

    wake_xmin = float(cfg_get(d, "wake_xmin", default=0.5))
    wake_xmax = float(cfg_get(d, "wake_xmax", default=5.0))
    wake_ymin = float(cfg_get(d, "wake_ymin", default=-2.0))
    wake_ymax = float(cfg_get(d, "wake_ymax", default= 2.0))

    # ── Training hyper-parameters ─────────────────────────────────────────────
    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_step   = int(cfg_get(tr, "lr_step",  default=5000))
    lr_gamma  = float(cfg_get(tr, "lr_gamma", default=0.5))
    log_every = int(cfg_get(tr, "log_every", default=500))
    seed      = int(cfg_get(tr, "seed",      default=0))
    resample_period = int(cfg_get(tr, "resample_period", default=500))

    # ── Loss weights ──────────────────────────────────────────────────────────
    W_PDE = float(cfg_get(lw, "w_pde", default=10.0))
    W_BC  = float(cfg_get(lw, "w_bc",  default=1.0))

    # Top/bottom wall BC: symmetry (v=0) by default — steady cylinder flow is
    # symmetric about y=0.  Override with wall_bc: 'freestream' if desired.
    wall_bc = str(cfg_get(d, "wall_bc", default="symmetry")).lower()
    if wall_bc not in ("symmetry", "freestream"):
        raise ValueError(f"wall_bc must be 'symmetry' or 'freestream', got '{wall_bc}'.")
    wall_symmetry = (wall_bc == "symmetry")

    n_col = n_interior + n_wake
    print(f"Cylinder (pure PINN):  R={R} (D={2*R}),  Re={Re},  U∞={U_inf},  "
          f"epochs={epochs}")
    print(f"  Domain: x∈[{xmin}, {xmax}]  y∈[{ymin}, {ymax}]")
    print(f"  Collocation: {n_interior} interior + {n_wake} wake = {n_col} (full-batch)")
    print(f"  Wake box: x∈[{wake_xmin}, {wake_xmax}]  y∈[{wake_ymin}, {wake_ymax}]"
          f"  buffer={buffer}")
    _wall_desc = "v=0 symmetry" if wall_symmetry else "free-stream (U∞,0)"
    print(f"  BCs: {n_inlet} inlet (U∞,0) | {n_outlet} outlet (p=0) "
          f"| {n_body} no-slip | 2×{n_wall} top/bottom [{_wall_desc}]")
    if resample_period > 0:
        print(f"  Collocation re-sampled every {resample_period} epochs")

    # ── Geometry ──────────────────────────────────────────────────────────────
    cyl      = Cylinder2D(radius=R, center=(cx, cy))
    dom      = (xmin, xmax, ymin, ymax)
    wake_box = (wake_xmin, wake_xmax, wake_ymin, wake_ymax)

    print("  Sampling collocation points …")
    xy_col_j = jnp.array(_build_collocation(
        cyl, n_interior, n_wake, dom, wake_box, buffer, seed))

    xy_inlet_j  = jnp.array(_edge_vertical(xmin, ymin, ymax, n_inlet))
    xy_outlet_j = jnp.array(_edge_vertical(xmax, ymin, ymax, n_outlet))
    xy_top_j    = jnp.array(_edge_horizontal(ymax, xmin, xmax, n_wall))
    xy_bot_j    = jnp.array(_edge_horizontal(ymin, xmin, xmax, n_wall))
    xy_body_j   = jnp.array(cyl.surface_points(n=n_body), dtype=jnp.float32)

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_type = str(cfg_get(cfg.network, "type", default="mlp")).lower()
    _net_cls = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net_type)
    if _net_cls is None:
        raise ValueError(f"Unknown network type '{net_type}'. "
                         f"Choose 'mlp' or 'gated_mlp'.")
    model = _net_cls(layers=list(cfg.network.layers))
    print(f"  Network: {_net_cls.__name__}  layers={list(cfg.network.layers)}")
    pde   = NavierStokesPDE(model, Re=Re)

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {n_params:,}")

    # StepLR: lr · gamma^(epoch // lr_step)
    lr_sched  = optax.exponential_decay(
        init_value=lr, transition_steps=lr_step,
        decay_rate=lr_gamma, staircase=True)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    # ── Loss / step  (full-batch; BC sets captured in closure) ────────────────
    @jax.jit
    def step(params, state, col):
        def loss_fn(p):
            res   = pde.residual(p, col)                       # (N, 3)
            l_pde = jnp.sum(jnp.mean(res ** 2, axis=0))

            # Inlet: u = U∞, v = 0
            out_in = model.apply(p, xy_inlet_j)
            l_in   = (jnp.mean((out_in[:, 0] - U_inf) ** 2)
                      + jnp.mean(out_in[:, 1] ** 2))

            # Outlet: p = 0
            out_out = model.apply(p, xy_outlet_j)
            l_out   = jnp.mean(out_out[:, 2] ** 2)

            # Cylinder: no-slip u = v = 0
            out_b  = model.apply(p, xy_body_j)
            l_body = jnp.mean(out_b[:, 0] ** 2) + jnp.mean(out_b[:, 1] ** 2)

            # Top / bottom walls
            out_t  = model.apply(p, xy_top_j)
            out_bo = model.apply(p, xy_bot_j)
            if wall_symmetry:
                l_wall = jnp.mean(out_t[:, 1] ** 2) + jnp.mean(out_bo[:, 1] ** 2)
            else:
                l_wall = (jnp.mean((out_t[:, 0]  - U_inf) ** 2)
                          + jnp.mean(out_t[:, 1] ** 2)
                          + jnp.mean((out_bo[:, 0] - U_inf) ** 2)
                          + jnp.mean(out_bo[:, 1] ** 2))

            l_bc  = l_in + l_out + l_body + l_wall
            total = W_PDE * l_pde + W_BC * l_bc
            return total, (l_pde, l_bc)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, loss, aux

    # ── Restart ───────────────────────────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=1000))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])
    pde_hist  = hists.get("pde_hist",  [])
    bc_hist   = hists.get("bc_hist",   [])

    logger = ConsoleLogger(log_every=log_every)

    # ── Training loop ─────────────────────────────────────────────────────────
    try:
        for ep in range(start_ep, epochs):
            if resample_period > 0 and ep > 0 and ep % resample_period == 0:
                xy_col_j = jnp.array(_build_collocation(
                    cyl, n_interior, n_wake, dom, wake_box, buffer, seed + ep))

            params, opt_state, loss, (l_pde, l_bc) = step(
                params, opt_state, xy_col_j)
            loss_hist.append(float(loss))
            pde_hist.append(float(l_pde))
            bc_hist.append(float(l_bc))

            logs = {"loss": float(loss), "pde": float(l_pde), "bc": float(l_bc)}
            logger.on_epoch_end(ep, logs)
            restart.maybe_save(ep, params, opt_state,
                               {"loss_hist": loss_hist,
                                "pde_hist":  pde_hist,
                                "bc_hist":   bc_hist})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Visualisation ─────────────────────────────────────────────────────────
    print("\nEvaluating on prediction grid …")
    nx, ny = 400, 200
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xg, yg)
    grid_j = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)
    pred_g = np.array(model.apply(params, grid_j))
    U = pred_g[:, 0].reshape(ny, nx)
    V = pred_g[:, 1].reshape(ny, nx)
    P = pred_g[:, 2].reshape(ny, nx)

    inside = cyl.is_inside(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(ny, nx)
    for arr in (U, V, P):
        arr[inside] = np.nan

    # 3-panel stacked figure: u, v, p
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    for ax, field, cmap, title in zip(
        axes,
        (U, V, P),
        ("jet", "jet", "coolwarm"),
        ("Streamwise velocity  u", "Transverse velocity  v", "Pressure  p"),
    ):
        lim = np.nanmax(np.abs(field)) or 1.0
        cf  = ax.contourf(XX, YY, field, levels=60, cmap=cmap, vmin=-lim, vmax=lim)
        ax.fill(cyl.profile[:, 0], cyl.profile[:, 1], color="gray", zorder=5)
        ax.set_title(title, fontsize=13)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Cylinder | Re={Re} | D={2*R}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cylinder_fields.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Surface pressure on the cylinder ──────────────────────────────────────
    # Gauge set by the outlet (p=0), so  Cp = p / (½ U∞²)  with p∞ ≈ 0.
    theta_s = cyl.surface_angles(n=360)
    xy_surf = cyl.surface_points(n=360)
    p_surf  = np.array(model.apply(params, jnp.array(xy_surf, dtype=jnp.float32))[:, 2])
    q_inf   = 0.5 * U_inf ** 2
    Cp_surf = p_surf / (q_inf + 1e-14)
    theta_deg = np.degrees(theta_s)
    Cp_potential = 1.0 - 4.0 * np.sin(theta_s) ** 2   # inviscid reference

    fig_s, (axc, axp) = plt.subplots(1, 2, figsize=(14, 5))
    # Cp vs angle (θ=0 at the rear stagnation point on +x axis)
    order = np.argsort(theta_deg)
    axc.plot(theta_deg[order], Cp_surf[order], "b-", lw=1.5, label="PINN")
    axc.plot(theta_deg[order], Cp_potential[order], "k--", lw=1.0,
             label="Inviscid (1−4sin²θ)")
    axc.set_xlabel("θ (deg)  [0 = rear, 180 = front stagnation]")
    axc.set_ylabel("Cp")
    axc.set_title("Surface pressure coefficient  Cp(θ)")
    axc.legend()
    axc.grid(True, alpha=0.4)
    # Cp around the cylinder outline (polar-ish: plot on the circle)
    sc = axp.scatter(xy_surf[:, 0], xy_surf[:, 1], c=Cp_surf, cmap="coolwarm",
                     s=25, vmin=-np.max(np.abs(Cp_surf)), vmax=np.max(np.abs(Cp_surf)))
    axp.fill(cyl.profile[:, 0], cyl.profile[:, 1], color="lightgray", zorder=0)
    plt.colorbar(sc, ax=axp, label="Cp")
    axp.set_aspect("equal")
    axp.set_xlabel("x")
    axp.set_ylabel("y")
    axp.set_title("Cp on the cylinder surface")
    fig_s.suptitle(f"Cylinder surface pressure — Re={Re}", fontsize=13)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(out_dir, "cylinder_surface_pressure.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_s)

    # Loss convergence
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.semilogy(loss_hist, lw=0.8, label="Total")
    ax2.semilogy(pde_hist,  lw=0.8, alpha=0.7, label="PDE")
    ax2.semilogy(bc_hist,   lw=0.8, alpha=0.7, label="BC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title(f"PINN Training Loss Convergence — Re={Re}")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss_curve.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Velocity profile on a vertical line 2 diameters downstream of the centre
    x_probe = float(cx + 4.0 * R)
    y_line  = np.linspace(ymin, ymax, 500, dtype=np.float32)
    xy_line = jnp.array(np.stack([np.full(500, x_probe, np.float32), y_line], axis=1))
    u_line  = np.array(model.apply(params, xy_line)[:, 0])

    fig3, ax3 = plt.subplots(figsize=(6, 8))
    ax3.plot(u_line, y_line, "b-", label="PINN u-velocity")
    ax3.axvline(x=U_inf, color="r", ls="--", label=f"U∞={U_inf}")
    ax3.set_xlabel("u")
    ax3.set_ylabel("y")
    ax3.set_title(f"Velocity Profile at x={x_probe:.1f} (downstream)")
    ax3.legend()
    ax3.grid(True, alpha=0.4)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "velocity_profile.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── Save predictions + checkpoint ─────────────────────────────────────────
    pred_col = np.array(model.apply(params, xy_col_j))
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xy_col_j[:, 0]),
                   "y": np.array(xy_col_j[:, 1])},
        outputs = {"u_pred": pred_col[:, 0],
                   "v_pred": pred_col[:, 1],
                   "p_pred": pred_col[:, 2]},
    )
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))

    save_checkpoint(params, out_dir, metadata={
        "problem": "cylinder",
        "network": {"type": net_type, "layers": list(cfg.network.layers)},
        "physics": {"Re": Re, "R": R, "U_inf": U_inf, "cx": cx, "cy": cy},
        "results": {"final_loss": loss_hist[-1] if loss_hist else float("nan"),
                    "n_epochs": len(loss_hist)},
    })

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nFinal loss: {loss_hist[-1]:.4e}" if loss_hist else "\nNo epochs run.")
    print(f"Outputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist,
            "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_cylinder(load_config(cfg_path))
