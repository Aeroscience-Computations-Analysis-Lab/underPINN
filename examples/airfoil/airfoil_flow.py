"""Steady flow over a NACA airfoil — incompressible NS PINN.

Run directly or via the CLI:

    python examples/airfoil/airfoil_flow.py              # uses config.yaml
    python examples/airfoil/airfoil_flow.py myconfig.yaml
    python -m underPINN run examples/airfoil/config.yaml

PDE: ∇·u = 0,  (u·∇)u + ∇p − (1/Re)∇²u = 0

BCs:
  • Farfield (left/right edges): u = U∞ cos α,  v = U∞ sin α
  • Symmetry (top/bottom edges): v = 0,  ∂u/∂y = 0
  • Airfoil  : u = v = 0  (no-slip)
  • Pressure : p = 0 at one upstream reference point (gauge)

At AoA=0 the flow is symmetric about y=0, so the top and bottom domain
boundaries act as symmetry planes: zero normal velocity (v=0, Dirichlet)
and zero wall-normal shear (∂u/∂y=0, Neumann via JAX autodiff).

Collocation strategy
---------------------
Points are split into three groups:

  1. **Fixed uniform** (n_fixed_uniform): drawn once from the exterior domain,
     uniform distribution — provide global background coverage throughout
     the far field.  Never resampled.

  2. **Fixed SDF** (n_fixed_sdf): drawn once with importance weights
     exp(−sdf / sdf_scale) so that point density is highest near the
     airfoil surface (boundary layer, leading/trailing edges, near wake).
     Never resampled.

  3. **Dynamic** (n_dynamic): drawn with a Gaussian spread about the wake
     centerline (a ray from the trailing edge in the free-stream direction).
     Points cluster densely along the wake axis and thin out laterally as
     N(0, σ²) in the perpendicular direction; the distribution is clipped
     to the wake bounding box.  Replaced every ``resample_period`` epochs
     using residual-based adaptive sampling (RAR-D, Lu et al. 2021).

Loss function
-------------
Simple MSE on each physical term::

    L = L_pde  +  w_body · L_body  +  w_ff · L_ff
      + w_pref · L_pref  +  w_sym · L_sym

Each term is a plain mean-squared residual.
Weights can be tuned in config.yaml.
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
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice
from underPINN.utils.restart import RestartManager


# ---------------------------------------------------------------------------
# Aerodynamic post-processing helpers
# ---------------------------------------------------------------------------

def _compute_Cp(model, params, af, U_inf=1.0, x_ref=-2.0):
    """Surface pressure coefficient  Cp = (p − p∞) / (0.5 U∞²).

    Parameters
    ----------
    x_ref : x-coordinate of the upstream reference point for p=0 gauge.
            Must lie inside the domain (default -2.0 for xmin=-2.5).
    """
    xy_s  = jnp.array(af.surface_points(n=600))
    p_s   = model.apply(params, xy_s)[:, 2]
    p_ref = float(model.apply(params, jnp.array([[x_ref, 0.0]]))[0, 2])
    q_inf = 0.5 * U_inf ** 2
    Cp    = np.array((p_s - p_ref) / (q_inf + 1e-14))
    return np.array(xy_s), Cp


def _estimate_CL(xy_s, Cp):
    """Approximate lift coefficient via trapezoidal integration of Cp(x)."""
    x_s, y_s = xy_s[:, 0], xy_s[:, 1]
    top  = y_s >= 0
    bot  = y_s <  0
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    CL_top = -_trapz(Cp[top], x_s[top]) if top.any() else 0.0
    CL_bot =  _trapz(Cp[bot], x_s[bot]) if bot.any() else 0.0
    return float(CL_top + CL_bot)


# ---------------------------------------------------------------------------
# Wake Gaussian sampling
# ---------------------------------------------------------------------------

def _sample_wake_gaussian(
    af: NACAAirfoil,
    n: int,
    wake_xmin: float,
    wake_xmax: float,
    wake_ymin: float,
    wake_ymax: float,
    sigma: float,
    aoa_deg: float = 0.0,
    chord: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Sample *n* wake points with 2-D isotropic Gaussian spread from the wake line.

    A seed position ``t`` is drawn uniformly along the wake centerline, and
    then independent Gaussian noise is added in **both** the along-wake and
    perpendicular directions::

        t  ~ Uniform(t_lo, t_hi)   — seed along the wake axis
        dt ~ Normal(0, sigma²)     — Gaussian spread along the wake axis
        s  ~ Normal(0, sigma²)     — Gaussian spread perpendicular to axis
        xy = TE  +  (t + dt)·d̂  +  s·n̂

    This produces a smooth 2-D Gaussian cloud around every point on the
    centerline: density peaks on the axis and decays isotropically in all
    directions with no hard cutoffs.  The ends of the wake region fade
    naturally instead of terminating abruptly.

    The bounding box ``[wake_xmin, wake_xmax] × [wake_ymin, wake_ymax]``
    still acts as a hard clip for outliers (set ymin/ymax ≈ ±4σ and
    xmax sufficiently past the last feature of interest).

    Parameters
    ----------
    sigma     : standard deviation in **both** along-axis and perpendicular
                directions (same units as chord).
    aoa_deg   : angle of attack in degrees (determines wake direction).
    chord     : chord length (trailing-edge x-coordinate for symmetric NACA).

    Returns
    -------
    xy : (n, 2) float32
    """
    rng   = np.random.default_rng(seed)
    alpha = np.radians(aoa_deg)

    # Unit vectors along and perpendicular to the wake centerline
    d    = np.array([ np.cos(alpha),  np.sin(alpha)], dtype=np.float64)
    perp = np.array([-np.sin(alpha),  np.cos(alpha)], dtype=np.float64)
    te   = np.array([chord, 0.0],                     dtype=np.float64)

    # t range: project wake box corners onto d̂ to span the full box
    corners = np.array([
        [wake_xmin, wake_ymin], [wake_xmin, wake_ymax],
        [wake_xmax, wake_ymin], [wake_xmax, wake_ymax],
    ], dtype=np.float64)
    t_proj = (corners - te) @ d
    t_lo, t_hi = float(t_proj.min()), float(t_proj.max())

    pts: list = []
    while len(pts) < n:
        batch = max(n * 6, 20_000)
        t  = rng.uniform(t_lo, t_hi, size=batch)   # seed along centerline
        dt = rng.normal(0.0, sigma,  size=batch)   # Gaussian spread along axis
        s  = rng.normal(0.0, sigma,  size=batch)   # Gaussian spread perpendicular
        xy = (te + (t + dt)[:, None] * d
                 + s[:, None]        * perp).astype(np.float32)

        # Hard clip to wake bounding box
        in_box = (
            (xy[:, 0] >= wake_xmin) & (xy[:, 0] <= wake_xmax) &
            (xy[:, 1] >= wake_ymin) & (xy[:, 1] <= wake_ymax)
        )
        xy = xy[in_box]
        if len(xy) == 0:
            continue

        # Discard points inside the airfoil body
        xy = xy[~af.is_inside(xy)]
        pts.extend(xy.tolist())

    return np.array(pts[:n], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dynamic (residual-adaptive) collocation resampling
# ---------------------------------------------------------------------------

def _resample_dynamic(
    pde,
    params,
    af: NACAAirfoil,
    n_dynamic: int,
    n_candidates: int,
    k: float,
    wake_xmin: float,
    wake_xmax: float,
    wake_ymin: float,
    wake_ymax: float,
    wake_sigma: float,
    aoa_deg: float,
    chord: float,
    key: jax.Array,
) -> tuple[jnp.ndarray, jax.Array]:
    """Replace the dynamic collocation pool using residual-based (RAR-D) sampling.

    Candidates are drawn from the wake subdomain using :func:`_sample_wake_gaussian`
    (Gaussian spread about the centerline), so the candidate density peaks along
    the wake axis where NS errors are typically largest.

    Then ``n_dynamic`` replacement points are chosen proportional to
    ``|NS residual|^k``.

    Parameters
    ----------
    pde                         : NavierStokesPDE — ``residual(params, xy) → (N, 3)``
    params                      : current network parameters
    af                          : NACAAirfoil geometry
    n_dynamic                   : size of the pool to return
    n_candidates                : total candidate pool (≥ 5 × n_dynamic)
    k                           : RAR-D weighting exponent (1.0 = standard)
    wake_xmin/xmax/ymin/ymax    : bounding box of the wake subdomain
    wake_sigma                  : Gaussian σ perpendicular to the wake centerline
    aoa_deg                     : angle of attack (degrees)
    chord                       : chord length
    key                         : JAX PRNG key (consumed); updated key also returned

    Returns
    -------
    xy_new : (n_dynamic, 2) jnp.ndarray
    key    : updated PRNG key
    """
    key, k1, k2 = jax.random.split(key, 3)
    np_seed = int(jax.random.randint(k1, (), 0, 2 ** 31 - 1))

    # Candidate pool: Gaussian-wake exterior points inside the subdomain
    xy_u    = _sample_wake_gaussian(
        af, n_candidates,
        wake_xmin, wake_xmax, wake_ymin, wake_ymax,
        sigma=wake_sigma, aoa_deg=aoa_deg, chord=chord,
        seed=np_seed,
    )
    xy_cand = jnp.array(xy_u)

    # NS residual magnitude at every candidate
    res     = pde.residual(params, xy_cand)          # (N_cand, 3)
    res_mag = jnp.sqrt(jnp.sum(res ** 2, axis=-1))  # (N_cand,)

    # Weights: p ∝ |r|^k  (uniform fallback when all residuals are zero)
    w     = res_mag ** k
    total = w.sum()
    w     = jnp.where(total > 0.0, w / total,
                      jnp.ones_like(w) / n_candidates)

    idx = jax.random.choice(k2, n_candidates, shape=(n_dynamic,),
                            replace=True, p=w)
    return xy_cand[idx], key


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

    Re    = float(ph.Re)
    aoa   = float(cfg_get(ph, "aoa",   default=5.0))
    naca  = str(cfg_get(ph,  "naca",   default="0012"))
    chord = float(cfg_get(ph, "chord", default=1.0))

    xmin = float(cfg_get(dom, "xmin", default=-5.0))  if dom else -5.0
    xmax = float(cfg_get(dom, "xmax", default=15.0))  if dom else 15.0
    ymin = float(cfg_get(dom, "ymin", default=-8.0))  if dom else -8.0
    ymax = float(cfg_get(dom, "ymax", default= 8.0))  if dom else  8.0

    # ── Collocation split ─────────────────────────────────────────────────────
    d = cfg.data
    n_fixed_unif = int(cfg_get(d, "n_fixed_uniform",  default=10000))
    n_fixed_sdf  = int(cfg_get(d, "n_fixed_sdf",      default=8000))
    n_dynamic    = int(cfg_get(d, "n_dynamic",         default=5000))
    sdf_scale    = float(cfg_get(d, "sdf_scale",       default=0.05))
    n_body       = int(cfg_get(d, "n_body_bc",         default=2000))
    n_ff         = int(cfg_get(d, "n_farfield_bc",     default=1600))
    n_sym        = int(cfg_get(d, "n_sym_bc",          default=400))

    # Wake subdomain for dynamic points
    wake_xmin  = float(cfg_get(d, "wake_xmin",  default=chord))
    wake_xmax  = float(cfg_get(d, "wake_xmax",  default=xmax))
    wake_ymin  = float(cfg_get(d, "wake_ymin",  default=-0.5))
    wake_ymax  = float(cfg_get(d, "wake_ymax",  default= 0.5))
    # Gaussian σ perpendicular to the wake centerline (in chord units)
    wake_sigma = float(cfg_get(d, "wake_sigma", default=0.15))

    # ── Training hyper-parameters ─────────────────────────────────────────────
    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",   default=0.01))
    batch_r   = int(cfg_get(tr, "batch_r",      default=2048))
    batch_bc  = int(cfg_get(tr, "batch_bc",     default=512))
    log_every = int(cfg_get(tr, "log_every",    default=1000))
    patience  = int(cfg_get(tr, "early_stopping_patience", default=1000))
    seed      = int(cfg_get(tr, "seed",         default=0))

    resample_period     = int(cfg_get(tr, "resample_period",     default=100))
    resample_candidates = int(cfg_get(tr, "resample_candidates", default=5))
    resample_k          = float(cfg_get(tr, "resample_k",        default=1.0))

    # ── Loss weights ──────────────────────────────────────────────────────────
    W_BODY = float(cfg_get(lw, "w_body",  default=1.0))
    W_FF   = float(cfg_get(lw, "w_ff",   default=1.0))
    W_PREF = float(cfg_get(lw, "w_pref", default=1.0))
    W_SYM  = float(cfg_get(lw, "w_sym",  default=5.0))

    U_INF     = 1.0
    alpha_rad = np.radians(aoa)
    u_ff_val  = U_INF * np.cos(alpha_rad)
    v_ff_val  = U_INF * np.sin(alpha_rad)

    n_col_total = n_fixed_unif + n_fixed_sdf + n_dynamic
    print(f"Airfoil: NACA {naca},  Re={Re},  AoA={aoa}°,  epochs={epochs}")
    print(f"  Collocation: {n_fixed_unif} uniform (fixed)"
          f"  +  {n_fixed_sdf} SDF-weighted (fixed)"
          f"  +  {n_dynamic} dynamic (wake)  =  {n_col_total} total")
    print(f"  Wake box: x=[{wake_xmin}, {wake_xmax}]  y=[{wake_ymin}, {wake_ymax}]"
          f"  σ={wake_sigma}")
    print(f"  BCs: {n_body} no-slip | {n_ff} farfield (left+right) "
          f"| {n_sym} symmetry (top+bottom)")
    if resample_period > 0:
        print(f"  Dynamic pool refreshed every {resample_period} epochs "
              f"(pool={resample_candidates}×n_dyn, k={resample_k})")
    else:
        print("  Dynamic resampling disabled (resample_period=0)")

    # ── Geometry ──────────────────────────────────────────────────────────────
    af = NACAAirfoil(naca=naca, chord=chord)

    # --- Fixed pool 1: uniform far-field background ---
    print("  Sampling fixed uniform collocation points …")
    xy_fixed_unif = jnp.array(
        af.sample_exterior(n_fixed_unif, xmin, xmax, ymin, ymax, seed=seed))

    # --- Fixed pool 2: SDF-weighted near-airfoil ---
    print(f"  Sampling fixed SDF-weighted points (sdf_scale={sdf_scale}) …")
    xy_fixed_sdf = jnp.array(
        af.sample_sdf_weighted(n_fixed_sdf, xmin, xmax, ymin, ymax,
                               sdf_scale=sdf_scale, seed=seed + 1))

    # --- Dynamic pool: uniform over the wake box initially ---
    # Uniform coverage ensures the network sees the full wake region from the
    # start.  After the first resample_period epochs the pool is replaced by
    # Gaussian-wake RAR-D points that concentrate on high-residual locations.
    print(f"  Sampling initial dynamic points (uniform in wake box) …")
    xy_dynamic = jnp.array(
        af.sample_exterior(n_dynamic,
                           wake_xmin, wake_xmax,
                           wake_ymin, wake_ymax,
                           seed=seed + 2)
    )

    # Full collocation pool = fixed_unif ‖ fixed_sdf ‖ dynamic
    # (Rebuilt after each resample — the fixed part never changes.)
    def _build_col():
        return jnp.concatenate([xy_fixed_unif, xy_fixed_sdf, xy_dynamic], axis=0)

    xy_col_j = _build_col()

    # --- Boundary condition points ---
    print("  Sampling airfoil surface (no-slip) …")
    xy_body_j = jnp.array(af.surface_points(n=n_body), dtype=jnp.float32)

    # ── Farfield BCs: left (inlet, x=xmin) and right (outlet, x=xmax) only ───
    # Top and bottom are treated as symmetry planes — see below.
    print("  Sampling farfield boundary (left + right edges only) …")
    n_per_ff = max(1, n_ff // 2)
    t_y_ff   = np.linspace(ymin, ymax, n_per_ff, dtype=np.float32)
    xy_left  = np.stack([np.full(n_per_ff, xmin, np.float32), t_y_ff], axis=1)
    xy_right = np.stack([np.full(n_per_ff, xmax, np.float32), t_y_ff], axis=1)
    xy_ff    = np.concatenate([xy_left, xy_right], axis=0)
    u_ff_arr = np.full(len(xy_ff), u_ff_val, dtype=np.float32)
    v_ff_arr = np.full(len(xy_ff), v_ff_val, dtype=np.float32)
    xy_ff_j  = jnp.array(xy_ff)
    u_ff_j   = jnp.array(u_ff_arr)
    v_ff_j   = jnp.array(v_ff_arr)

    # ── Symmetry BCs: top (y=ymax) and bottom (y=ymin) edges ─────────────────
    # At AoA=0 both horizontal boundaries are symmetry planes:
    #   v = 0           (zero normal velocity — Dirichlet)
    #   ∂u/∂y = 0       (zero wall-normal shear — Neumann via JAX grad)
    print("  Sampling symmetry boundary (top + bottom edges) …")
    n_per_sym = max(1, n_sym // 2)
    t_x_sym   = np.linspace(xmin, xmax, n_per_sym, dtype=np.float32)
    xy_top    = np.stack([t_x_sym, np.full(n_per_sym, ymax, np.float32)], axis=1)
    xy_bot    = np.stack([t_x_sym, np.full(n_per_sym, ymin, np.float32)], axis=1)
    xy_sym    = np.concatenate([xy_top, xy_bot], axis=0)
    xy_sym_j  = jnp.array(xy_sym)

    # Pressure gauge: a single upstream reference point (inside domain, x > xmin)
    xy_pref_j = jnp.array([[-2.0, 0.0]], dtype=jnp.float32)

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

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    # ── Loss function ─────────────────────────────────────────────────────────
    # Five MSE terms (weights from config.yaml).
    # xy_body_j, xy_pref_j are fixed BCs captured in closure.
    # Symmetry Neumann term uses JAX grad inside vmap — JIT-compiled cleanly.
    @jax.jit
    def step(params, state, col_b, ff_b, uff_b, vff_b, sym_b):
        def loss_fn(p):
            # 1. PDE residual — NS continuity + x/y momentum (N_b, 3)
            res      = pde.residual(p, col_b)
            pde_loss = jnp.mean(jnp.sum(res ** 2, axis=1))

            # 2. No-slip: u=0, v=0 on the airfoil body
            out_body  = model.apply(p, xy_body_j)
            body_loss = jnp.mean(out_body[:, 0] ** 2 + out_body[:, 1] ** 2)

            # 3. Farfield (left/right edges): match free-stream velocity
            out_ff  = model.apply(p, ff_b)
            ff_loss = jnp.mean((out_ff[:, 0] - uff_b) ** 2
                               + (out_ff[:, 1] - vff_b) ** 2)

            # 4. Pressure gauge: p=0 at one upstream reference point
            pref_loss = model.apply(p, xy_pref_j)[0, 2] ** 2

            # 5. Symmetry (top/bottom edges):
            #    (a) v = 0            — zero normal velocity (Dirichlet)
            #    (b) ∂u/∂y = 0        — zero wall-normal shear (Neumann)
            out_sym   = model.apply(p, sym_b)
            sym_v     = jnp.mean(out_sym[:, 1] ** 2)

            def _u_single(xy):
                """u at a single point xy (shape (2,))."""
                return model.apply(p, xy[None])[0, 0]

            dudy = jax.vmap(lambda xy: jax.grad(_u_single)(xy)[1])(sym_b)
            sym_dudy  = jnp.mean(dudy ** 2)
            sym_loss  = sym_v + sym_dudy

            total = (pde_loss
                     + W_BODY * body_loss
                     + W_FF   * ff_loss
                     + W_PREF * pref_loss
                     + W_SYM  * sym_loss)
            return total, (pde_loss, body_loss, ff_loss, sym_loss)

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

    key     = jax.random.PRNGKey(seed + 99)
    logger  = ConsoleLogger(log_every=log_every)
#    stopper = EarlyStopping(patience=patience)

    n_col_pts = xy_col_j.shape[0]
    n_ff_pts  = xy_ff_j.shape[0]
    n_sym_pts = xy_sym_j.shape[0]

    try:
        for ep in range(start_ep, epochs):

            # ── Refresh dynamic pool (skip ep=0 — residuals are random) ──────
            if resample_period > 0 and ep > 0 and ep % resample_period == 0:
                n_cand    = resample_candidates * n_dynamic
                xy_dynamic, key = _resample_dynamic(
                    pde, params, af,
                    n_dynamic=n_dynamic,
                    n_candidates=n_cand,
                    k=resample_k,
                    wake_xmin=wake_xmin, wake_xmax=wake_xmax,
                    wake_ymin=wake_ymin, wake_ymax=wake_ymax,
                    wake_sigma=wake_sigma, aoa_deg=aoa, chord=chord,
                    key=key,
                )
                # Rebuild full pool: fixed parts are unchanged
                xy_col_j  = _build_col()
                n_col_pts = xy_col_j.shape[0]
                print(f"  [ep {ep:5d}] Dynamic pool refreshed "
                      f"({n_dynamic} pts, pool={n_cand})")

            # ── Gradient step ─────────────────────────────────────────────────
            key, k1, k2, k3 = jax.random.split(key, 4)
            idx_col = safe_choice(k1, n_col_pts, batch_r)
            idx_ff  = safe_choice(k2, n_ff_pts,  batch_bc)
            idx_sym = safe_choice(k3, n_sym_pts, batch_bc)

            params, opt_state, loss, (pde_l, body_l, ff_l, sym_l) = step(
                params, opt_state,
                xy_col_j[idx_col],
                xy_ff_j[idx_ff], u_ff_j[idx_ff], v_ff_j[idx_ff],
                xy_sym_j[idx_sym],
            )
            loss_hist.append(float(loss))
            pde_hist.append(float(pde_l))
            bc_hist.append(float(body_l + ff_l + sym_l))

            logs = {"loss": float(loss), "pde": float(pde_l),
                    "bc": float(body_l + ff_l + sym_l)}
            logger.on_epoch_end(ep, logs)
 #           stopper.on_epoch_end(ep, logs)
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

    inside = af.is_inside(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(Ny, Nx)
    u_plot = np.where(inside, np.nan, u_grid)
    v_plot = np.where(inside, np.nan, v_grid)
    p_plot = np.where(inside, np.nan, p_grid)

    # Fields plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, field, cmap, title in zip(
        axes,
        [u_plot, v_plot, p_plot],
        ["jet", "jet", "seismic"],
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

    # Collocation point visualisation
    xy_col_np = np.array(xy_col_j)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    # Fixed uniform
    ax2.scatter(np.array(xy_fixed_unif)[:, 0],
                np.array(xy_fixed_unif)[:, 1],
                s=0.5, c="steelblue", alpha=0.3, label=f"Fixed uniform ({n_fixed_unif})")
    # Fixed SDF
    ax2.scatter(np.array(xy_fixed_sdf)[:, 0],
                np.array(xy_fixed_sdf)[:, 1],
                s=0.8, c="orange", alpha=0.5, label=f"Fixed SDF ({n_fixed_sdf})")
    # Dynamic (wake-confined)
    ax2.scatter(np.array(xy_dynamic)[:, 0],
                np.array(xy_dynamic)[:, 1],
                s=0.8, c="green", alpha=0.5,
                label=f"Dynamic / wake ({n_dynamic})")
    # Draw wake subdomain outline
    from matplotlib.patches import Rectangle
    wake_rect = Rectangle(
        (wake_xmin, wake_ymin),
        wake_xmax - wake_xmin, wake_ymax - wake_ymin,
        linewidth=1.2, edgecolor="green", facecolor="none",
        linestyle="--", zorder=6, label="Wake subdomain",
    )
    ax2.add_patch(wake_rect)
    ax2.fill(af.profile[:, 0], af.profile[:, 1], "k", zorder=5)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal")
    ax2.set_title(f"Collocation points — NACA {naca}")
    ax2.set_xlabel("x / c"); ax2.set_ylabel("y / c")
    ax2.legend(markerscale=8, fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "collocation_points.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Pressure coefficient + CL  (x_ref must lie inside the domain)
    xy_s, Cp = _compute_Cp(model, params, af, U_INF, x_ref=max(xmin + 0.5, -2.0))
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
    if resample_period > 0:
        for rep_ep in range(resample_period, len(loss_hist), resample_period):
            ax4.axvline(rep_ep, color="grey", lw=0.8, ls="--", alpha=0.4)
        ax4.axvline(resample_period, color="grey", lw=0.8, ls="--",
                    alpha=0.4, label="Dynamic resample")
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
        "network": {"type": net_type, "layers": list(cfg.network.layers)},
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
