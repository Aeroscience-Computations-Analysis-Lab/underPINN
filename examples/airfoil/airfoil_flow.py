"""Pure PINN — 2-D Steady Flow over a NACA Airfoil.

No training data is used — only Navier–Stokes PDE residuals + boundary
conditions.

Run directly or via the CLI:

    python examples/airfoil/airfoil_flow.py              # uses config.yaml
    python examples/airfoil/airfoil_flow.py myconfig.yaml
    python -m underPINN run examples/airfoil/config.yaml

PDE  (steady incompressible Navier–Stokes, ν = 1/Re):

    ∇·u = 0
    (u·∇)u + ∇p − ν ∇²u = 0

Angle of attack is imposed by **rotating the airfoil** (pitch by α about the
quarter-chord); the free-stream stays horizontal.  This avoids tilting the
inflow and keeps the inlet/outlet aligned with the domain edges.

Boundary conditions:

    • Inlet   (left edge,  x = xmin):  u = U∞,  v = 0   (horizontal free-stream)
    • Outlet  (right edge, x = xmax):  p = 0
    • Airfoil (no-slip)             :  u = v = 0
    • Top / bottom walls            :  v = 0 (symmetry, symmetric airfoil @ α=0)
                                       or (u, v) = (U∞, 0) (free-stream, else)

Collocation strategy (notebook recipe):

    A fixed uniform interior pool (exterior to the airfoil, with a small buffer)
    plus a denser wake pool downstream.  Both pools are re-sampled together
    every ``resample_period`` epochs so the network keeps seeing fresh points
    in the recirculation zone.  Training is full-batch (every epoch uses every
    collocation point), like the reference notebook.

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
from underPINN.geometry.airfoil import NACAAirfoil
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager


# ---------------------------------------------------------------------------
# Collocation sampling helpers  (notebook style)
# ---------------------------------------------------------------------------

def _sample_box_exterior(
    af: NACAAirfoil,
    n: int,
    xlo: float, xhi: float, ylo: float, yhi: float,
    buffer: float,
    seed: int,
) -> np.ndarray:
    """Uniform points in a box, rejecting the airfoil interior + a buffer shell.

    Mirrors the notebook's ``sqrt(x²+y²) > R + 0.05`` test (cylinder radius plus
    a small buffer), generalised to an arbitrary body via the signed-distance
    field: a point is kept iff it is outside the airfoil *and* at least
    ``buffer`` away from the surface.
    """
    rng = np.random.default_rng(seed)
    pts: list = []
    while len(pts) < n:
        batch = rng.uniform([xlo, ylo], [xhi, yhi],
                            size=(max(n * 4, 20_000), 2)).astype(np.float32)
        batch = batch[~af.is_inside(batch)]
        if buffer > 0.0 and len(batch):
            batch = batch[af.sdf(batch) > buffer]
        pts.extend(batch.tolist())
    return np.array(pts[:n], dtype=np.float32)


def _build_collocation(
    af, n_interior, n_wake, dom, wake_box, buffer, seed,
) -> np.ndarray:
    """Interior background pool + denser downstream wake pool, concatenated."""
    xmin, xmax, ymin, ymax = dom
    wxmin, wxmax, wymin, wymax = wake_box
    interior = _sample_box_exterior(af, n_interior, xmin, xmax, ymin, ymax,
                                    buffer, seed)
    if n_wake > 0:
        wake = _sample_box_exterior(af, n_wake, wxmin, wxmax, wymin, wymax,
                                    buffer, seed + 1)
        return np.concatenate([interior, wake], axis=0)
    return interior


def _edge_vertical(xval, ymin, ymax, n):
    """n points along a vertical edge at x = xval."""
    y = np.linspace(ymin, ymax, n, dtype=np.float32)
    return np.stack([np.full(n, xval, np.float32), y], axis=1)


def _edge_horizontal(yval, xmin, xmax, n):
    """n points along a horizontal edge at y = yval."""
    x = np.linspace(xmin, xmax, n, dtype=np.float32)
    return np.stack([x, np.full(n, yval, np.float32)], axis=1)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_airfoil(cfg) -> dict:
    """Train a pure PINN on steady incompressible NS around a NACA airfoil."""
    # ── Unpack config ─────────────────────────────────────────────────────────
    ph      = cfg.physics
    tr      = cfg.training
    lw      = cfg.loss
    dom_c   = cfg_get(cfg, "domain", default=None)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/airfoil") if out else "outputs/airfoil"
    os.makedirs(out_dir, exist_ok=True)

    Re     = float(ph.Re)
    aoa    = float(cfg_get(ph, "aoa",   default=0.0))
    naca   = str(cfg_get(ph,  "naca",   default="0012"))
    chord  = float(cfg_get(ph, "chord", default=1.0))
    U_inf  = float(cfg_get(ph, "U_inf", default=1.0))

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
    buffer     = float(cfg_get(d, "buffer",   default=0.02))

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

    # Angle of attack is realised by ROTATING THE AIRFOIL (see NACAAirfoil),
    # so the free-stream stays horizontal:  (u, v)_∞ = (U_inf, 0).
    u_in_val = U_inf
    v_in_val = 0.0

    # ── Geometry (airfoil pitched by the angle of attack) ─────────────────────
    af  = NACAAirfoil(naca=naca, chord=chord, aoa_deg=aoa)
    dom      = (xmin, xmax, ymin, ymax)
    wake_box = (wake_xmin, wake_xmax, wake_ymin, wake_ymax)

    # ── Top/bottom wall BC ────────────────────────────────────────────────────
    # A true symmetry plane (v=0) is only physical for a *symmetric* airfoil at
    # zero AoA.  For a cambered airfoil — or any non-zero AoA (rotated body) —
    # the flow carries lift and is not symmetric about y=0, so the far-field
    # top/bottom edges instead impose the free-stream velocity (u, v) = (U∞, 0).
    # ``wall_bc`` can be forced in config ('symmetry' | 'freestream').
    sym_ok      = af.is_symmetric and abs(aoa) < 1e-9
    wall_bc     = str(cfg_get(d, "wall_bc",
                              default="symmetry" if sym_ok else "freestream")).lower()
    if wall_bc not in ("symmetry", "freestream"):
        raise ValueError(f"wall_bc must be 'symmetry' or 'freestream', got '{wall_bc}'.")
    wall_symmetry = (wall_bc == "symmetry")

    n_col = n_interior + n_wake
    print(f"Airfoil (pure PINN): NACA {naca}"
          f"  ({'symmetric' if af.is_symmetric else 'cambered'}),"
          f"  Re={Re},  AoA={aoa}°,  epochs={epochs}")
    print(f"  Domain: x∈[{xmin}, {xmax}]  y∈[{ymin}, {ymax}]")
    print(f"  Collocation: {n_interior} interior + {n_wake} wake = {n_col} (full-batch)")
    print(f"  Wake box: x∈[{wake_xmin}, {wake_xmax}]  y∈[{wake_ymin}, {wake_ymax}]"
          f"  buffer={buffer}")
    _wall_desc = "v=0 symmetry" if wall_symmetry else "free-stream (u,v)"
    print(f"  BCs: {n_inlet} inlet (u,v) | {n_outlet} outlet (p=0) "
          f"| {n_body} no-slip | 2×{n_wall} top/bottom [{_wall_desc}]")
    if resample_period > 0:
        print(f"  Collocation re-sampled every {resample_period} epochs")

    print("  Sampling collocation points …")
    xy_col_j = jnp.array(_build_collocation(
        af, n_interior, n_wake, dom, wake_box, buffer, seed))

    # Boundary-condition point sets (fixed throughout training)
    xy_inlet_j  = jnp.array(_edge_vertical(xmin, ymin, ymax, n_inlet))
    xy_outlet_j = jnp.array(_edge_vertical(xmax, ymin, ymax, n_outlet))
    xy_top_j    = jnp.array(_edge_horizontal(ymax, xmin, xmax, n_wall))
    xy_bot_j    = jnp.array(_edge_horizontal(ymin, xmin, xmax, n_wall))
    xy_body_j   = jnp.array(af.surface_points(n=n_body), dtype=jnp.float32)

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
            # PDE residual: mean(cont²) + mean(momx²) + mean(momy²)
            res   = pde.residual(p, col)                       # (N, 3)
            l_pde = jnp.sum(jnp.mean(res ** 2, axis=0))

            # Inlet: u = U∞cosα, v = U∞sinα
            out_in = model.apply(p, xy_inlet_j)
            l_in   = (jnp.mean((out_in[:, 0] - u_in_val) ** 2)
                      + jnp.mean((out_in[:, 1] - v_in_val) ** 2))

            # Outlet: p = 0  (sets the pressure gauge)
            out_out = model.apply(p, xy_outlet_j)
            l_out   = jnp.mean(out_out[:, 2] ** 2)

            # Airfoil: no-slip u = v = 0
            out_b  = model.apply(p, xy_body_j)
            l_body = jnp.mean(out_b[:, 0] ** 2) + jnp.mean(out_b[:, 1] ** 2)

            # Top / bottom far-field walls
            out_t  = model.apply(p, xy_top_j)
            out_bo = model.apply(p, xy_bot_j)
            if wall_symmetry:
                # Symmetry plane: zero normal velocity v = 0
                l_wall = jnp.mean(out_t[:, 1] ** 2) + jnp.mean(out_bo[:, 1] ** 2)
            else:
                # Free-stream Dirichlet: (u, v) = U∞(cosα, sinα)
                l_wall = (jnp.mean((out_t[:, 0]  - u_in_val) ** 2)
                          + jnp.mean((out_t[:, 1]  - v_in_val) ** 2)
                          + jnp.mean((out_bo[:, 0] - u_in_val) ** 2)
                          + jnp.mean((out_bo[:, 1] - v_in_val) ** 2))

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
            # Re-sample collocation (interior + wake) periodically
            if resample_period > 0 and ep > 0 and ep % resample_period == 0:
                xy_col_j = jnp.array(_build_collocation(
                    af, n_interior, n_wake, dom, wake_box, buffer,
                    seed + ep))   # fresh seed → new points; shape unchanged

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

    # ── Visualisation (notebook style) ────────────────────────────────────────
    print("\nEvaluating on prediction grid …")
    nx, ny = 400, 200
    xg = np.linspace(xmin, xmax, nx)          # float64 → equally spaced for streamplot
    yg = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xg, yg)
    grid_j = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)
    pred_g = np.array(model.apply(params, grid_j))
    U = pred_g[:, 0].reshape(ny, nx)
    V = pred_g[:, 1].reshape(ny, nx)
    P = pred_g[:, 2].reshape(ny, nx)

    # Mask the airfoil interior
    inside = af.is_inside(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(ny, nx)
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
        ax.fill(af.profile[:, 0], af.profile[:, 1], color="gray", zorder=5)
        ax.set_title(title, fontsize=13)
        plt.colorbar(cf, ax=ax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")
    fig.suptitle(f"NACA {naca} | Re={Re} | AoA={aoa}°", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "airfoil_fields.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Surface pressure on the airfoil ───────────────────────────────────────
    # Pressure is sampled on the surface; the outlet BC (p=0) sets the gauge,
    # so the pressure coefficient is  Cp = p / (½ U∞²)  with p∞ ≈ 0.
    xy_surf, upper = af.surface_points(n=400, return_side=True)
    p_surf  = np.array(model.apply(params, jnp.array(xy_surf, dtype=jnp.float32))[:, 2])
    q_inf   = 0.5 * U_inf ** 2
    Cp_surf = p_surf / (q_inf + 1e-14)
    x_surf  = xy_surf[:, 0]    # physical x-coordinate of each surface point
    lower   = ~upper

    fig_s, (axp, axc) = plt.subplots(1, 2, figsize=(14, 5))
    # Raw surface pressure p(x)
    axp.plot(x_surf[upper], p_surf[upper], "b-o", ms=2.5, lw=1.2, label="Upper")
    axp.plot(x_surf[lower], p_surf[lower], "r-o", ms=2.5, lw=1.2, label="Lower")
    axp.axhline(0.0, color="k", lw=0.6, ls="--")
    axp.set_xlabel("x / c")
    axp.set_ylabel("p")
    axp.set_title("Surface pressure  p(x)")
    axp.legend()
    axp.grid(True, alpha=0.4)
    # Pressure coefficient Cp(x) — inverted y-axis (aerodynamics convention)
    axc.plot(x_surf[upper], Cp_surf[upper], "b-o", ms=2.5, lw=1.2, label="Upper")
    axc.plot(x_surf[lower], Cp_surf[lower], "r-o", ms=2.5, lw=1.2, label="Lower")
    axc.axhline(0.0, color="k", lw=0.6, ls="--")
    axc.invert_yaxis()
    axc.set_xlabel("x / c")
    axc.set_ylabel("Cp")
    axc.set_title("Pressure coefficient  Cp = p / (½ U∞²)")
    axc.legend()
    axc.grid(True, alpha=0.4)
    fig_s.suptitle(f"Airfoil surface pressure — NACA {naca} | Re={Re} | AoA={aoa}°",
                   fontsize=13)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(out_dir, "airfoil_surface_pressure.png"),
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

    # Velocity profile on a vertical line 2 chords downstream of the TE
    x_probe = float(chord + 2.0)
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
        "problem": "airfoil",
        "network": {"type": net_type, "layers": list(cfg.network.layers)},
        "physics": {"Re": Re, "aoa": aoa, "naca": naca,
                    "chord": chord, "U_inf": U_inf},
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
    run_airfoil(load_config(cfg_path))
