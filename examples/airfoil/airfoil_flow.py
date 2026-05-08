"""
Steady Flow over NACA 0012 Airfoil — underPINN Example
=======================================================

Solves the steady incompressible Navier-Stokes equations around a NACA 0012
airfoil at a prescribed angle of attack (AoA) and Reynolds number:

    ∇·u = 0
    (u·∇)u + ∇p − (1/Re)∇²u = 0

Domain   : rectangular [-5, 15] × [-8, 8]  (chord c = 1, x ∈ [0, 1])
Model    : MLP([2, 128, 128, 128, 128, 3]) → (u, v, p)
BCs      :
    • Far-field  — u = U∞ cos α,  v = U∞ sin α   (freestream)
    • Airfoil    — u = v = 0                       (no-slip)
    • Pressure   — p = 0 at one upstream point     (gauge)

Outputs  :
    airfoil_fields.png      — u, v, p contour maps over the full domain
    airfoil_streamlines.png — speed + streamlines in the near-body region
    airfoil_Cp.png          — pressure coefficient Cp(x/c) upper & lower surface
    airfoil_loss.png        — training loss history

Key features demonstrated
--------------------------
• NACA 4-digit profile generation (NACAAirfoil geometry class)
• Near-surface collocation refinement for boundary-layer resolution
• Reuse of existing NavierStokesPDE — no new PDE class needed
• Pressure coefficient and lift coefficient estimation from PINN output
• ConsoleLogger + EarlyStopping callbacks
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.geometry.airfoil import NACAAirfoil
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping


# ---- Flow parameters ----
RE    = 200.0     # Reynolds number (laminar; NACA 0012 separates ~Re 500–1000)
AOA   = 5.0       # angle of attack [degrees]
U_INF = 1.0       # freestream speed magnitude

# ---- Rectangular computational domain ----
XMIN, XMAX = -5.0, 15.0   # ~5c upstream, 14c downstream
YMIN, YMAX = -8.0,  8.0   # ±8c lateral

# ---- Training hyper-parameters ----
EPOCHS    = 10000
BATCH_COL = 2048    # collocation mini-batch size
BATCH_FF  =  512    # far-field BC mini-batch
W_BODY    =  50.0   # no-slip BC weight (high: enforce strongly)
W_FF      =  10.0   # far-field BC weight
W_PREF    =  10.0   # pressure gauge weight


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_training_data():
    alpha_rad = np.radians(AOA)
    af = NACAAirfoil(naca="0012", chord=1.0)

    print("  Sampling exterior collocation points …")
    xy_far  = af.sample_exterior(40_000, XMIN, XMAX, YMIN, YMAX, seed=0)
    xy_near = af.sample_near_surface(10_000, seed=1)          # denser near body
    xy_col  = np.concatenate([xy_far, xy_near], axis=0)

    print("  Sampling airfoil surface (no-slip BC) …")
    xy_af = af.surface_points(n=1000)

    print("  Sampling far-field boundary …")
    xy_ff = af.farfield_boundary(n_per_edge=350,
                                 xmin=XMIN, xmax=XMAX,
                                 ymin=YMIN, ymax=YMAX)
    u_ff  = np.full(len(xy_ff), U_INF * np.cos(alpha_rad), dtype=np.float32)
    v_ff  = np.full(len(xy_ff), U_INF * np.sin(alpha_rad), dtype=np.float32)

    # Pressure reference: a single point far upstream on the centreline
    xy_pref = np.array([[-4.9, 0.0]], dtype=np.float32)

    return af, xy_col, xy_af, xy_ff, u_ff, v_ff, xy_pref


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def compute_Cp(model, params, af):
    """Pressure coefficient along the airfoil surface."""
    xy_s   = af.surface_points(n=600)
    pred_s = model.apply(params, jnp.array(xy_s))
    p_s    = np.array(pred_s[:, 2])
    # Cp = (p - p_inf) / (0.5 rho U_inf^2), with p_inf=0, rho=1, U_inf=1
    Cp     = 2.0 * p_s
    return xy_s, Cp


def estimate_CL(xy_s, Cp):
    """Rough lift coefficient from Cp integration (pressure-only, no viscous)."""
    x   = xy_s[:, 0]
    y   = xy_s[:, 1]
    top = y >= 0
    bot = y <  0

    idx_t = np.argsort(x[top]);  xu = x[top][idx_t];  Cu = Cp[top][idx_t]
    idx_b = np.argsort(x[bot]);  xl = x[bot][idx_b];  Cl = Cp[bot][idx_b]

    # CL ≈ (∫_lower Cp dx − ∫_upper Cp dx) / chord
    from numpy import trapz
    CL = (trapz(Cl, xl) - trapz(Cu, xu))
    return float(CL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())
    alpha_rad = np.radians(AOA)
    u_inf     = float(U_INF * np.cos(alpha_rad))
    v_inf     = float(U_INF * np.sin(alpha_rad))
    print(f"NACA 0012 | Re={RE} | AoA={AOA}° | u∞=({u_inf:.4f}, {v_inf:.4f})\n")

    # ---- Geometry + data ----
    print("Generating geometry …")
    af, xy_col, xy_af, xy_ff, u_ff, v_ff, xy_pref = make_training_data()
    print(f"  Collocation : {len(xy_col):,} pts")
    print(f"  Airfoil BC  : {len(xy_af):,} pts")
    print(f"  Far-field BC: {len(xy_ff):,} pts\n")

    # Convert to JAX arrays (kept in host memory; sliced each step)
    xy_col_j  = jnp.array(xy_col)
    xy_af_j   = jnp.array(xy_af)
    xy_ff_j   = jnp.array(xy_ff)
    u_ff_j    = jnp.array(u_ff)
    v_ff_j    = jnp.array(v_ff)
    xy_pref_j = jnp.array(xy_pref)

    # ---- Model ----
    model  = MLP(layers=[2, 128, 128, 128, 128, 3])
    pde    = NavierStokesPDE(model, Re=RE)
    params = model.init(jax.random.PRNGKey(42), jnp.ones((1, 2)))

    schedule  = optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    # ---- JIT-compiled step ----
    @jax.jit
    def step(params, opt_state, col_b, ff_b, uff_b, vff_b):
        def loss_fn(p):
            # NS residuals: continuity + x-momentum + y-momentum
            res      = pde.residual(p, col_b)        # (N, 3)
            pde_loss = jnp.mean(res ** 2)            # scalar mean over all 3

            # No-slip on airfoil (all surface points each step — small set)
            out_af = model.apply(p, xy_af_j)
            l_af   = jnp.mean(out_af[:, 0] ** 2) + jnp.mean(out_af[:, 1] ** 2)

            # Freestream at far-field (mini-batched)
            out_ff = model.apply(p, ff_b)
            l_ff   = (jnp.mean((out_ff[:, 0] - uff_b) ** 2)
                      + jnp.mean((out_ff[:, 1] - vff_b) ** 2))

            # Pressure gauge: p = 0 far upstream
            l_pref = model.apply(p, xy_pref_j)[0, 2] ** 2

            total = pde_loss + W_BODY * l_af + W_FF * l_ff + W_PREF * l_pref
            return total, (pde_loss, l_af, l_ff)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # ---- Training loop ----
    key      = jax.random.PRNGKey(0)
    logger   = ConsoleLogger(log_every=500)
    stopper  = EarlyStopping(patience=1000)
    loss_hist, pde_hist, bc_hist = [], [], []

    n_col = xy_col_j.shape[0]
    n_ff  = xy_ff_j.shape[0]

    try:
        for ep in range(EPOCHS):
            key, k1, k2 = jax.random.split(key, 3)

            idx_col = jax.random.choice(k1, n_col, (BATCH_COL,), replace=False)
            idx_ff  = jax.random.choice(k2, n_ff,  (BATCH_FF,),  replace=False)

            params, opt_state, loss, (pde_l, l_af, l_ff) = step(
                params, opt_state,
                xy_col_j[idx_col],
                xy_ff_j[idx_ff], u_ff_j[idx_ff], v_ff_j[idx_ff],
            )

            loss_val = float(loss)
            loss_hist.append(loss_val)
            pde_hist.append(float(pde_l))
            bc_hist.append(float(l_af + l_ff))

            logs = {"loss": loss_val, "pde": float(pde_l), "bc": float(l_af + l_ff)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)

    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1]})

    # ---- Post-processing ----
    print("\nEvaluating on prediction grid …")

    Nx, Ny  = 350, 180
    xg = np.linspace(XMIN, XMAX, Nx, dtype=np.float32)
    yg = np.linspace(YMIN, YMAX, Ny, dtype=np.float32)
    XX, YY  = np.meshgrid(xg, yg)
    grid_j  = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel())], axis=1)

    pred    = np.array(model.apply(params, grid_j))
    u_grid  = pred[:, 0].reshape(Ny, Nx)
    v_grid  = pred[:, 1].reshape(Ny, Nx)
    p_grid  = pred[:, 2].reshape(Ny, Nx)

    # Mask airfoil interior for cleaner plots
    inside  = af.is_inside(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(Ny, Nx)
    u_plot  = np.where(inside, np.nan, u_grid)
    v_plot  = np.where(inside, np.nan, v_grid)
    p_plot  = np.where(inside, np.nan, p_grid)

    # ---- Figure 1: u, v, p fields ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, field, cmap, title in zip(
        axes,
        [u_plot, v_plot, p_plot],
        ["RdBu_r", "RdBu_r", "seismic"],
        ["Streamwise velocity  u", "Normal velocity  v", "Pressure  p"],
    ):
        lim = np.nanmax(np.abs(field)) or 1.0
        cf  = ax.contourf(xg, yg, field, 60, cmap=cmap,
                          vmin=-lim, vmax=lim)
        plt.colorbar(cf, ax=ax, shrink=0.75)
        ax.fill(af.profile[:, 0], af.profile[:, 1], "k", zorder=5)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    fig.suptitle(f"NACA 0012 | Re={RE} | AoA={AOA}°", fontsize=13)
    fig.tight_layout()
    fig.savefig("airfoil_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: airfoil_fields.png")

    # ---- Figure 2: near-body streamlines ----
    # Restrict to near-body window to reduce NaN issues in streamplot
    xi = np.linspace(-1.5, 4.0, 250, dtype=np.float32)
    yi = np.linspace(-1.5, 1.5, 150, dtype=np.float32)
    XXi, YYi = np.meshgrid(xi, yi)
    grid_i  = jnp.stack([jnp.array(XXi.ravel()), jnp.array(YYi.ravel())], axis=1)
    pred_i  = np.array(model.apply(params, grid_i))
    ui = pred_i[:, 0].reshape(150, 250)
    vi = pred_i[:, 1].reshape(150, 250)
    inside_i = af.is_inside(
        np.stack([XXi.ravel(), YYi.ravel()], axis=1)
    ).reshape(150, 250)
    speed_i = np.where(inside_i, np.nan, np.sqrt(ui**2 + vi**2))
    ui      = np.where(inside_i, 0.0, ui)
    vi      = np.where(inside_i, 0.0, vi)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    cf2 = ax2.contourf(xi, yi, speed_i, 60, cmap="viridis")
    plt.colorbar(cf2, ax=ax2, label="|U| / U∞")
    ax2.streamplot(xi, yi, ui, vi, color="white", linewidth=0.6,
                   density=2.0, arrowsize=0.9)
    ax2.fill(af.profile[:, 0], af.profile[:, 1], "k", zorder=5)
    ax2.set_xlim(-1.5, 4.0); ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"Near-body flow: NACA 0012 | Re={RE} | AoA={AOA}°")
    ax2.set_xlabel("x / c"); ax2.set_ylabel("y / c")
    fig2.tight_layout()
    fig2.savefig("airfoil_streamlines.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: airfoil_streamlines.png")

    # ---- Figure 3: Pressure coefficient Cp ----
    xy_s, Cp = compute_Cp(model, params, af)
    x_s = xy_s[:, 0]
    y_s = xy_s[:, 1]
    top_mask = y_s >= 0
    bot_mask = y_s <  0

    CL = estimate_CL(xy_s, Cp)
    print(f"\nEstimated CL ≈ {CL:.4f}  (pressure-only; viscous contribution omitted)")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(x_s[top_mask], Cp[top_mask], "b-o", ms=2.5, lw=1.2, label="Upper surface")
    ax3.plot(x_s[bot_mask], Cp[bot_mask], "r-o", ms=2.5, lw=1.2, label="Lower surface")
    ax3.axhline(0, color="k", lw=0.6, ls="--")
    ax3.invert_yaxis()     # aerodynamics convention: suction (−Cp) plotted upward
    ax3.set_xlabel("x / c")
    ax3.set_ylabel("Cp")
    ax3.set_title(
        f"Pressure coefficient — NACA 0012 | Re={RE} | AoA={AOA}°"
        f"\nEstimated CL ≈ {CL:.3f}"
    )
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig("airfoil_Cp.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved: airfoil_Cp.png")

    # ---- Figure 4: Training loss ----
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.semilogy(loss_hist, label="Total",   alpha=0.9)
    ax4.semilogy(pde_hist,  label="PDE",     alpha=0.75)
    ax4.semilogy(bc_hist,   label="BC",      alpha=0.75)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
    ax4.set_title("Airfoil PINN — training history")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig("airfoil_loss.png", dpi=150)
    plt.close(fig4)
    print("Saved: airfoil_loss.png")


if __name__ == "__main__":
    main()
