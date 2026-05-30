"""Steady flow over a NACA airfoil — incompressible NS PINN.

Run directly or via the CLI:

    python examples/airfoil/airfoil_flow.py              # uses config.yaml
    python examples/airfoil/airfoil_flow.py myconfig.yaml
    python -m underPINN run examples/airfoil/config.yaml

PDE: ∇·u = 0,  (u·∇)u + ∇p − (1/Re)∇²u = 0

BCs:
  • Farfield : u = U∞ cos α,  v = U∞ sin α
  • Airfoil  : u = v = 0  (no-slip)
  • Pressure : p = 0 at one upstream point (gauge)

Adaptive resampling (RAR-D)
---------------------------
Set ``training.resample_period > 0`` in config to periodically replace the
interior collocation pool with points sampled proportional to the local NS
residual magnitude.  Each resample evaluates the residual at a large
candidate pool (``resample_candidates × n_col`` points), then draws the
final set with weights  p ∝ |r|^resample_k.  This concentrates points in
high-error regions (shear layers, wake, boundary-layer separation) without
touching the fixed boundary condition points.

Outputs: field plots (u, v, p), Cp curve, estimated CL, loss history,
         params checkpoint.
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
from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.geometry.airfoil import NACAAirfoil
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice
from underPINN.utils.restart import RestartManager


# ---------------------------------------------------------------------------
# Aerodynamic post-processing helpers
# ---------------------------------------------------------------------------

def _compute_Cp(model, params, af, U_inf=1.0):
    """Surface pressure coefficient  Cp = (p − p∞) / (0.5 U∞²)."""
    xy_s  = jnp.array(af.surface_points(n=600))
    p_s   = model.apply(params, xy_s)[:, 2]
    p_ref = float(model.apply(params, jnp.array([[-4.9, 0.0]]))[0, 2])
    q_inf = 0.5 * U_inf ** 2
    Cp    = np.array((p_s - p_ref) / (q_inf + 1e-14))
    return np.array(xy_s), Cp


def _estimate_CL(xy_s, Cp):
    """Approximate lift coefficient via trapezoidal integration of Cp(x)."""
    x_s, y_s = xy_s[:, 0], xy_s[:, 1]
    top  = y_s >= 0
    bot  = y_s <  0
    CL_top = -np.trapz(Cp[top],  x_s[top])  if top.any()  else 0.0
    CL_bot =  np.trapz(Cp[bot],  x_s[bot])  if bot.any()  else 0.0
    return float(CL_top + CL_bot)


# ---------------------------------------------------------------------------
# RAR-D adaptive collocation resampling
# ---------------------------------------------------------------------------

def _rar_d_resample_col(
    pde,
    params,
    af: NACAAirfoil,
    n_col: int,
    near_frac: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    k: float,
    n_candidates: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Replace the interior collocation pool using RAR-D weighting.

    Generates ``n_candidates`` fresh domain points (preserving the original
    exterior / near-surface split ratio), evaluates the NS residual magnitude
    at each, then draws ``n_col`` replacement points proportional to
    ``|residual|^k``.  Returns a new ``(n_col, 2)`` JAX array.

    Parameters
    ----------
    pde :
        ``NavierStokesPDE`` instance — provides ``residual(params, xy)``.
    params :
        Current network parameters.
    af :
        ``NACAAirfoil`` geometry used to draw valid exterior candidates.
    n_col :
        Number of collocation points to return (same as current pool size).
    near_frac :
        Fraction of candidates to draw from the near-surface subdomain.
        Mirrors the original exterior / near-surface split.
    xmin, xmax, ymin, ymax :
        Far-field bounding box — passed to ``af.sample_exterior``.
    k :
        RAR-D exponent; ``p ∝ |r|^k``.  ``k=1`` is the standard choice.
        Higher values concentrate points more aggressively in high-error
        regions.
    n_candidates :
        Total candidate pool size (recommended: 5 × n_col).
    key :
        JAX PRNG key (consumed).

    Returns
    -------
    xy_new : (n_col, 2) jnp.ndarray
    """
    key, k1, k2 = jax.random.split(key, 3)
    # Convert JAX key to a NumPy integer seed for the geometry samplers
    np_seed = int(jax.random.randint(k1, (), 0, 2**31 - 1))

    # ── 1. Generate candidate pool ────────────────────────────────────────
    n_near_c = max(1, int(n_candidates * near_frac))
    n_ext_c  = n_candidates - n_near_c

    xy_ext_c  = af.sample_exterior(n_ext_c, xmin, xmax, ymin, ymax, seed=np_seed)
    xy_near_c = af.sample_near_surface(n_near_c, seed=np_seed + 1)
    xy_cand   = jnp.array(np.concatenate([xy_ext_c, xy_near_c], axis=0))

    # ── 2. NS residual magnitude at every candidate ───────────────────────
    # residual returns (n_cand, 3): [continuity, x-momentum, y-momentum]
    res     = pde.residual(params, xy_cand)
    res_mag = (jnp.sqrt(jnp.sum(res ** 2, axis=-1))
               if res.ndim > 1 else jnp.abs(res))   # (n_cand,)

    # ── 3. Build sampling weights  p ∝ |r|^k ─────────────────────────────
    w     = res_mag ** k
    total = w.sum()
    # Guard: if all residuals are zero (fully converged), use uniform weights
    w = jnp.where(total > 0.0, w / total,
                  jnp.ones_like(w) / n_candidates)

    # ── 4. Draw n_col replacement points ─────────────────────────────────
    idx = jax.random.choice(k2, n_candidates, shape=(n_col,),
                            replace=True, p=w)
    return xy_cand[idx]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_airfoil(cfg) -> dict:
    """Train a PINN on steady incompressible NS around a NACA airfoil."""
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

    # ── RAR-D adaptive resampling ─────────────────────────────────────────────
    resample_period     = int(cfg_get(tr, "resample_period",     default=0))
    resample_candidates = int(cfg_get(tr, "resample_candidates", default=5))
    resample_k          = float(cfg_get(tr, "resample_k",        default=1.0))
    _near_frac          = n_near / (n_ext + n_near)   # preserve original split

    W_BODY = float(cfg_get(lw, "w_body",  default=50.0))
    W_FF   = float(cfg_get(lw, "w_ff",    default=10.0))
    W_PREF = float(cfg_get(lw, "w_pref",  default=10.0))

    U_INF     = 1.0
    alpha_rad = np.radians(aoa)
    u_ff_val  = U_INF * np.cos(alpha_rad)
    v_ff_val  = U_INF * np.sin(alpha_rad)

    rar_info = (f"  RAR-D every {resample_period} ep, "
                f"pool={resample_candidates}×, k={resample_k}"
                if resample_period > 0 else "  RAR-D disabled (resample_period=0)")
    print(f"Airfoil:  NACA {naca},  Re={Re},  AoA={aoa}°,  epochs={epochs}")
    print(rar_info)

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
    # NOTE: xy_body_j and xy_pref_j are captured in the closure — they are fixed
    # boundary condition arrays that never change.  Only the interior collocation
    # mini-batch (col_b) is drawn fresh each step and resampled via RAR-D.
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
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])
    pde_hist  = hists.get("pde_hist",  [])
    bc_hist   = hists.get("bc_hist",   [])

    key       = jax.random.PRNGKey(seed + 99)
    logger    = ConsoleLogger(log_every=log_every)
    stopper   = EarlyStopping(patience=patience)
    n_col_pts = xy_col_j.shape[0]
    n_ff_pts  = xy_ff_j.shape[0]

    try:
        for ep in range(start_ep, epochs):
            # ── RAR-D adaptive resampling ─────────────────────────────────────
            # Skip epoch 0 — residuals are random before the first gradient step.
            # Resamples fire at ep == resample_period, 2×, 3×, …
            if resample_period > 0 and ep > 0 and ep % resample_period == 0:
                key, rkey = jax.random.split(key)
                n_cand    = resample_candidates * n_col_pts
                xy_col_j  = _rar_d_resample_col(
                    pde, params, af,
                    n_col=n_col_pts,
                    near_frac=_near_frac,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    k=resample_k,
                    n_candidates=n_cand,
                    key=rkey,
                )
                # n_col_pts stays the same — just the pool content changes
                print(f"  [RAR-D ep {ep:5d}] Resampled {n_col_pts} collocation points "
                      f"(pool={n_cand}, k={resample_k})")

            key, k1, k2 = jax.random.split(key, 3)
            idx_col = safe_choice(k1, n_col_pts, batch_r)
            idx_ff  = safe_choice(k2, n_ff_pts,  batch_bc)

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
            restart.maybe_save(ep, params, opt_state,
                               {"loss_hist": loss_hist,
                                "pde_hist":  pde_hist,
                                "bc_hist":   bc_hist})
    except StopIteration:
        pass

    restart.done()
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
    # Mark RAR-D resample epochs with vertical dashed lines
    if resample_period > 0:
        for rep_ep in range(resample_period, len(loss_hist), resample_period):
            ax4.axvline(rep_ep, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax4.axvline(resample_period, color="grey", lw=0.8, ls="--",
                    alpha=0.5, label="RAR-D resample")
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


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_airfoil(load_config(cfg_path))
