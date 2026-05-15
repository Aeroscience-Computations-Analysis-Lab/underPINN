"""Runner for steady flow over a NACA airfoil (incompressible NS).

Expected config sections
------------------------
problem  : airfoil   # run via:  python -m underPINN run examples/airfoil/config.yaml

network:
  type  : mlp
  layers: [2, 128, 128, 128, 128, 3]   # (x,y) → (u,v,p)

physics:
  Re    : 200.0
  aoa   : 5.0       # angle of attack in degrees
  naca  : "0012"    # 4-digit NACA profile
  chord : 1.0

domain:
  xmin: -5.0
  xmax: 15.0
  ymin: -8.0
  ymax:  8.0

data:
  n_exterior    : 40000   # uniform exterior collocation
  n_near_surface: 10000   # near-body refinement
  n_body_bc     : 2000    # no-slip wall points
  n_farfield_bc : 1600    # farfield boundary (400 per edge)

training:
  epochs                  : 10000
  lr                      : 1.0e-3
  lr_alpha                : 0.01
  batch_r                 : 2048
  batch_bc                : 512
  log_every               : 1000
  early_stopping_patience : 1000
  seed                    : 0

loss:
  w_pde  : 1.0
  w_body : 50.0    # no-slip BC on airfoil surface
  w_ff   : 10.0    # freestream farfield BC
  w_pref : 10.0    # pressure gauge (p=0 at upstream reference point)

output:
  dir: outputs/airfoil
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
from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.geometry.airfoil import NACAAirfoil
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# Aerodynamic post-processing helpers
# ---------------------------------------------------------------------------

def _compute_Cp(model, params, af, U_inf=1.0):
    """Surface pressure coefficient  Cp = (p − p∞) / (0.5 U∞²)."""
    xy_s  = jnp.array(af.surface_points(n=600))
    p_s   = model.apply(params, xy_s)[:, 2]          # PINN pressure
    p_ref = float(model.apply(params, jnp.array([[-4.9, 0.0]]))[0, 2])
    q_inf = 0.5 * U_inf ** 2
    Cp    = np.array((p_s - p_ref) / (q_inf + 1e-14))
    return np.array(xy_s), Cp


def _estimate_CL(xy_s, Cp):
    """Approximate lift coefficient via trapezoidal integration of Cp(x)."""
    x_s, y_s = xy_s[:, 0], xy_s[:, 1]
    top  = y_s >= 0
    bot  = y_s <  0
    CL_top = -np.trapz(Cp[top],  x_s[top])   if top.any()  else 0.0
    CL_bot =  np.trapz(Cp[bot],  x_s[bot])   if bot.any()  else 0.0
    return float(CL_top + CL_bot)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_airfoil(cfg) -> dict:
    """Train a PINN on steady incompressible NS around a NACA airfoil.

    ∇·u = 0,  (u·∇)u + ∇p − (1/Re)∇²u = 0

    BCs:
      • Farfield : u = U∞ cos α,  v = U∞ sin α
      • Airfoil  : u = v = 0  (no-slip)
      • Pressure : p = 0 at one upstream point (gauge condition)
    """
    # ── Unpack config ─────────────────────────────────────────────────────────
    ph      = cfg.physics
    tr      = cfg.training
    lw      = cfg.loss
    dom     = cfg_get(cfg, "domain", default=None)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/airfoil") if out else "outputs/airfoil"
    os.makedirs(out_dir, exist_ok=True)

    Re     = float(ph.Re)
    aoa    = float(cfg_get(ph, "aoa",   default=5.0))
    naca   = str(cfg_get(ph,  "naca",   default="0012"))
    chord  = float(cfg_get(ph, "chord", default=1.0))

    xmin = float(cfg_get(dom, "xmin", default=-5.0))  if dom else -5.0
    xmax = float(cfg_get(dom, "xmax", default=15.0))  if dom else 15.0
    ymin = float(cfg_get(dom, "ymin", default=-8.0))  if dom else -8.0
    ymax = float(cfg_get(dom, "ymax", default= 8.0))  if dom else  8.0

    n_ext  = int(cfg_get(cfg.data, "n_exterior",     default=40000))
    n_near = int(cfg_get(cfg.data, "n_near_surface",  default=10000))
    n_body = int(cfg_get(cfg.data, "n_body_bc",       default=2000))
    n_ff   = int(cfg_get(cfg.data, "n_farfield_bc",   default=1600))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",   default=0.01))
    batch_r   = int(cfg_get(tr, "batch_r",      default=2048))
    batch_bc  = int(cfg_get(tr, "batch_bc",     default=512))
    log_every = int(cfg_get(tr, "log_every",    default=1000))
    patience  = int(cfg_get(tr, "early_stopping_patience", default=1000))
    seed      = int(cfg_get(tr, "seed",         default=0))

    W_BODY = float(cfg_get(lw, "w_body",  default=50.0))
    W_FF   = float(cfg_get(lw, "w_ff",    default=10.0))
    W_PREF = float(cfg_get(lw, "w_pref",  default=10.0))

    U_INF     = 1.0
    alpha_rad = np.radians(aoa)
    u_ff_val  = U_INF * np.cos(alpha_rad)
    v_ff_val  = U_INF * np.sin(alpha_rad)

    print(f"Airfoil:  NACA {naca},  Re={Re},  AoA={aoa}°,  epochs={epochs}")

    # ── Geometry ──────────────────────────────────────────────────────────────
    af = NACAAirfoil(naca=naca, chord=chord)

    print("  Sampling exterior collocation points …")
    xy_far  = af.sample_exterior(n_ext,  xmin, xmax, ymin, ymax, seed=seed)
    xy_near = af.sample_near_surface(n_near, seed=seed + 1)
    xy_col  = np.concatenate([xy_far, xy_near], axis=0)

    print("  Sampling airfoil surface (no-slip) …")
    xy_body = af.surface_points(n=n_body)

    print("  Sampling farfield boundary …")
    n_per_edge = max(1, n_ff // 4)
    xy_ff      = af.farfield_boundary(n_per_edge=n_per_edge,
                                      xmin=xmin, xmax=xmax,
                                      ymin=ymin, ymax=ymax)
    u_ff = np.full(len(xy_ff), u_ff_val, dtype=np.float32)
    v_ff = np.full(len(xy_ff), v_ff_val, dtype=np.float32)

    # Pressure gauge: a single upstream reference point
    xy_pref = np.array([[-4.9, 0.0]], dtype=np.float32)

    # Convert to JAX arrays
    xy_col_j  = jnp.array(xy_col,   dtype=jnp.float32)
    xy_body_j = jnp.array(xy_body,  dtype=jnp.float32)
    xy_ff_j   = jnp.array(xy_ff,    dtype=jnp.float32)
    u_ff_j    = jnp.array(u_ff)
    v_ff_j    = jnp.array(v_ff)
    xy_pref_j = jnp.array(xy_pref)

    # ── Model + PDE ───────────────────────────────────────────────────────────
    model = MLP(layers=list(cfg.network.layers))
    pde   = NavierStokesPDE(model, Re=Re)

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    # ── JIT step ──────────────────────────────────────────────────────────────
    @jax.jit
    def step(params, state, col_b, ff_b, uff_b, vff_b):
        def loss_fn(p):
            res      = pde.residual(p, col_b)
            pde_loss = jnp.mean(res ** 2)

            out_body = model.apply(p, xy_body_j)
            l_body   = (jnp.mean(out_body[:, 0] ** 2)
                        + jnp.mean(out_body[:, 1] ** 2))

            out_ff = model.apply(p, ff_b)
            l_ff   = (jnp.mean((out_ff[:, 0] - uff_b) ** 2)
                      + jnp.mean((out_ff[:, 1] - vff_b) ** 2))

            l_pref = model.apply(p, xy_pref_j)[0, 2] ** 2

            total = pde_loss + W_BODY * l_body + W_FF * l_ff + W_PREF * l_pref
            return total, (pde_loss, l_body, l_ff)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, loss, aux

    # ── Training loop ─────────────────────────────────────────────────────────
    key       = jax.random.PRNGKey(seed + 99)
    logger    = ConsoleLogger(log_every=log_every)
    stopper   = EarlyStopping(patience=patience)
    n_col     = xy_col_j.shape[0]
    n_ff_pts  = xy_ff_j.shape[0]
    loss_hist, pde_hist, bc_hist = [], [], []

    try:
        for ep in range(epochs):
            key, k1, k2 = jax.random.split(key, 3)
            idx_col = safe_choice(k1, n_col,    batch_r)
            idx_ff  = safe_choice(k2, n_ff_pts, batch_bc)

            params, opt_state, loss, (pde_l, l_body, l_ff) = step(
                params, opt_state,
                xy_col_j[idx_col],
                xy_ff_j[idx_ff], u_ff_j[idx_ff], v_ff_j[idx_ff],
            )
            loss_hist.append(float(loss))
            pde_hist.append(float(pde_l))
            bc_hist.append(float(l_body + l_ff))

            logs = {"loss": float(loss), "pde": float(pde_l),
                    "bc": float(l_body + l_ff)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)
    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Post-processing ───────────────────────────────────────────────────────
    print("\nEvaluating on prediction grid …")
    Nx, Ny = 350, 180
    xg = np.linspace(xmin, xmax, Nx, dtype=np.float32)
    yg = np.linspace(ymin, ymax, Ny, dtype=np.float32)
    XX, YY = np.meshgrid(xg, yg)
    grid_j  = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel())], axis=1)
    pred_g  = np.array(model.apply(params, grid_j))
    u_grid  = pred_g[:, 0].reshape(Ny, Nx)
    v_grid  = pred_g[:, 1].reshape(Ny, Nx)
    p_grid  = pred_g[:, 2].reshape(Ny, Nx)

    inside  = af.is_inside(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(Ny, Nx)
    u_plot  = np.where(inside, np.nan, u_grid)
    v_plot  = np.where(inside, np.nan, v_grid)
    p_plot  = np.where(inside, np.nan, p_grid)

    # Fields plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, field, cmap, title in zip(
        axes,
        [u_plot, v_plot, p_plot],
        ["RdBu_r", "RdBu_r", "seismic"],
        ["Streamwise velocity  u", "Normal velocity  v", "Pressure  p"],
    ):
        lim = np.nanmax(np.abs(field)) or 1.0
        cf  = ax.contourf(xg, yg, field, 60, cmap=cmap, vmin=-lim, vmax=lim)
        plt.colorbar(cf, ax=ax, shrink=0.75)
        ax.fill(af.profile[:, 0], af.profile[:, 1], "k", zorder=5)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    fig.suptitle(f"NACA {naca} | Re={Re} | AoA={aoa}°", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "airfoil_fields.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Pressure coefficient + CL
    xy_s, Cp = _compute_Cp(model, params, af, U_INF)
    CL       = _estimate_CL(xy_s, Cp)
    print(f"\nEstimated CL ≈ {CL:.4f}  (pressure-only)")
    x_s, y_s = xy_s[:, 0], xy_s[:, 1]
    top_mask, bot_mask = y_s >= 0, y_s < 0
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(x_s[top_mask], Cp[top_mask], "b-o", ms=2.5, lw=1.2, label="Upper surface")
    ax3.plot(x_s[bot_mask], Cp[bot_mask], "r-o", ms=2.5, lw=1.2, label="Lower surface")
    ax3.axhline(0, color="k", lw=0.6, ls="--")
    ax3.invert_yaxis()
    ax3.set_xlabel("x / c"); ax3.set_ylabel("Cp")
    ax3.set_title(f"Pressure coefficient — NACA {naca} | Re={Re} | AoA={aoa}°"
                  f"\nEst. CL ≈ {CL:.3f}")
    ax3.legend(); fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "airfoil_Cp.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Loss history
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.semilogy(loss_hist, label="Total",  alpha=0.9)
    ax4.semilogy(pde_hist,  label="PDE",    alpha=0.75)
    ax4.semilogy(bc_hist,   label="BC",     alpha=0.75)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
    ax4.set_title(f"Airfoil PINN — Re={Re},  NACA {naca}")
    ax4.legend(); fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── Save predictions ──────────────────────────────────────────────────────
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

    # ── Model checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(params, out_dir, metadata={
        "problem": "airfoil",
        "network": {"type": "mlp", "layers": list(cfg.network.layers)},
        "physics": {"Re": Re, "aoa": aoa, "naca": naca, "chord": chord},
        "results": {"CL": CL, "n_epochs": len(loss_hist)},
    })

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist,
            "CL": CL, "n_epochs": len(loss_hist)}
