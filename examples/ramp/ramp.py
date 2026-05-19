"""2-D Compressible Euler — Mach-3 flow over a 10° ramp.

Run directly or via the CLI:

    python examples/ramp/ramp.py                   # uses config.yaml
    python examples/ramp/ramp.py myconfig.yaml
    python -m underPINN run examples/ramp/config.yaml

Physics
-------
Steady, inviscid, compressible flow (Euler equations) in primitive variables
(ρ, u, v, p).  The exact solution is the classic oblique-shock wave:

  M_∞ = 3,  θ = 10°  →  shock angle β ≈ 27.4°,  post-shock M₂ ≈ 2.50

Non-dimensionalisation:  ρ_∞ = 1,  a_∞ = 1
  p_∞ = 1/γ ≈ 0.714,   u_∞ = M_∞ = 3,   v_∞ = 0

Network:  (x, y) → (f_ρ, f_u, f_v, f_p)
  Physical state recovered via softplus on ρ and p outputs (ensures > 0).

Boundary conditions:
  Inlet (x=0)         — all four primitive variables fixed to freestream
  Ramp wall (lower)   — slip: u·n = 0,  n = (−sin θ, cos θ)
  Upper farfield (y=H)— freestream values (undisturbed supersonic flow)
  Outlet (x=L)        — supersonic outflow; no BC imposed

Outputs written to outputs/ramp/:
  solution.png         — 2×3 contourf: ρ, u, v, p, Mach, entropy
  loss.png             — semilogy training history
  oblique_shock.png    — Mach number with analytical shock line overlay
  predictions.npz      — grid evaluation (x, y, rho, u, v, p, mach)
  loss_hist.npy
  params.msgpack + params_meta.json
"""
from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.config.loader import cfg_get, save_config
from underPINN.geometry.ramp import RampGeometry
from underPINN.nn.mlp import MLP
from underPINN.pde.compressible_euler import CompressibleEulerPDE
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ramp(cfg) -> dict:
    """Train a PINN on 2-D steady compressible Euler flow over a ramp."""

    # ── Config ────────────────────────────────────────────────────────────────
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/ramp") if out else "outputs/ramp"
    os.makedirs(out_dir, exist_ok=True)

    gamma     = float(ph.gamma)
    M_inf     = float(ph.M_inf)
    theta_deg = float(ph.theta_deg)

    L = float(cfg_get(cfg.geometry, "L", default=1.0))
    H = float(cfg_get(cfg.geometry, "H", default=0.8))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",   default=0.01))
    log_every = int(cfg_get(tr, "log_every",    default=500))
    patience  = int(cfg_get(tr, "early_stopping_patience", default=2000))
    seed      = int(cfg_get(tr, "seed",         default=0))
    batch_r   = int(cfg_get(tr, "batch_r",      default=2048))
    batch_bc  = int(cfg_get(tr, "batch_bc",     default=256))

    W_PDE   = float(cfg_get(lw, "w_pde",   default=1.0))
    W_INLET = float(cfg_get(lw, "w_inlet", default=200.0))
    W_WALL  = float(cfg_get(lw, "w_wall",  default=80.0))
    W_UPPER = float(cfg_get(lw, "w_upper", default=30.0))

    d = cfg.data
    n_int   = int(cfg_get(d, "n_interior", default=8000))
    n_in    = int(cfg_get(d, "n_inlet",    default=300))
    n_wall  = int(cfg_get(d, "n_wall",     default=400))
    n_upper = int(cfg_get(d, "n_upper",    default=200))

    print(f"Ramp PINN:  M={M_inf},  θ={theta_deg}°,  γ={gamma}")
    print(f"Domain:  x∈[0,{L}]  y∈[0,{H}]  (ramp wall at y=x·tan({theta_deg}°))")

    # ── Analytical oblique-shock solution ─────────────────────────────────────
    pde  = CompressibleEulerPDE(None, gamma=gamma)     # model set below
    shock = pde.oblique_shock(M_inf, theta_deg)
    print(f"\nAnalytical oblique shock:")
    print(f"  Shock angle β = {shock['beta_deg']:.2f}°")
    print(f"  Post-shock   M2={shock['M2']:.3f}  ρ2={shock['rho2']:.3f}"
          f"  u2={shock['u2']:.3f}  v2={shock['v2']:.3f}  p2={shock['p2']:.4f}\n")

    rho_inf, u_inf, v_inf, p_inf = pde.freestream(M_inf)

    # ── Geometry ──────────────────────────────────────────────────────────────
    geom = RampGeometry(theta_deg, L=L, H=H)

    xy_r    = jnp.array(geom.sample_interior(n_int,  seed=seed))
    xy_in   = jnp.array(geom.sample_inlet(n_in))
    xy_wall = jnp.array(geom.sample_ramp_wall(n_wall))
    xy_up   = jnp.array(geom.sample_upper(n_upper))

    nx, ny = geom.ramp_normal()    # scalar floats as jnp constants
    nx_j   = jnp.array(nx)
    ny_j   = jnp.array(ny)

    # ── Model ─────────────────────────────────────────────────────────────────
    layers = list(cfg.network.layers)
    model  = MLP(layers=layers)
    pde.model = model               # wire model into PDE

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    # ── JIT training step ─────────────────────────────────────────────────────
    @jax.jit
    def step(params, state, xy_r_, xy_in_, xy_wall_, xy_up_):
        def loss_fn(p):
            # PDE residuals
            cont, mom_x, mom_y, energy = pde.residual(p, xy_r_)
            pde_l = (jnp.mean(cont   ** 2) + jnp.mean(mom_x ** 2)
                     + jnp.mean(mom_y ** 2) + jnp.mean(energy ** 2))

            # Inlet BC — all four primitives fixed to freestream
            pv_in  = pde.apply(p, xy_in_)
            inlet_l = (jnp.mean((pv_in[:, 0] - rho_inf) ** 2)
                       + jnp.mean((pv_in[:, 1] - u_inf)  ** 2)
                       + jnp.mean((pv_in[:, 2] - v_inf)  ** 2)
                       + jnp.mean((pv_in[:, 3] - p_inf)  ** 2))

            # Ramp wall BC — slip: normal velocity = 0
            pv_w   = pde.apply(p, xy_wall_)
            u_w    = pv_w[:, 1];  v_w = pv_w[:, 2]
            wall_l = jnp.mean((u_w * nx_j + v_w * ny_j) ** 2)

            # Upper farfield BC — undisturbed freestream
            pv_up  = pde.apply(p, xy_up_)
            upper_l = (jnp.mean((pv_up[:, 0] - rho_inf) ** 2)
                       + jnp.mean((pv_up[:, 1] - u_inf)  ** 2)
                       + jnp.mean((pv_up[:, 2] - v_inf)  ** 2)
                       + jnp.mean((pv_up[:, 3] - p_inf)  ** 2))

            total = (W_PDE   * pde_l
                     + W_INLET * inlet_l
                     + W_WALL  * wall_l
                     + W_UPPER * upper_l)
            return total, (pde_l, inlet_l, wall_l, upper_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Training loop ─────────────────────────────────────────────────────────
    N_r    = xy_r.shape[0]
    N_in   = xy_in.shape[0]
    N_wall = xy_wall.shape[0]
    N_up   = xy_up.shape[0]

    logger   = ConsoleLogger(log_every=log_every)
    stopper  = EarlyStopping(patience=patience)
    loss_hist, pde_hist = [], []
    key = jax.random.PRNGKey(seed + 7)

    try:
        for ep in range(epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir  = safe_choice(k1, N_r,    batch_r)
            iin = safe_choice(k2, N_in,   min(batch_bc, N_in))
            iw  = safe_choice(k3, N_wall, min(batch_bc, N_wall))
            iu  = safe_choice(k4, N_up,   min(batch_bc, N_up))

            params, opt_state, total, (pl, il, wl, ul) = step(
                params, opt_state,
                xy_r[ir], xy_in[iin], xy_wall[iw], xy_up[iu])

            loss_hist.append(float(total))
            pde_hist.append(float(pl))

            logs = {"loss": float(total), "pde": float(pl),
                    "inlet": float(il), "wall": float(wl)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)
    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})
    print(f"  Stopped at epoch {len(loss_hist)}")

    # ── Grid evaluation ───────────────────────────────────────────────────────
    XX, YY, mask = geom.make_grid(Nx=200, Ny=160)
    pts = jnp.array(
        np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)
    pv_grid = np.array(pde.apply(params, pts))       # (Ny*Nx, 4)

    rho_g = pv_grid[:, 0].reshape(160, 200)
    u_g   = pv_grid[:, 1].reshape(160, 200)
    v_g   = pv_grid[:, 2].reshape(160, 200)
    p_g   = pv_grid[:, 3].reshape(160, 200)

    a_g    = np.sqrt(gamma * p_g / np.maximum(rho_g, 1e-9))
    mach_g = np.sqrt(u_g ** 2 + v_g ** 2) / np.maximum(a_g, 1e-9)

    # Isentropic entropy deviation: s/s_∞ - 1  (should be 0 everywhere for Euler)
    # s ∝ p / ρ^γ ;  s_∞ = p_inf / 1^γ = p_inf
    entr_g = (p_g / np.maximum(rho_g ** gamma, 1e-9)) / p_inf - 1.0

    # Mask sub-ramp points
    for arr in [rho_g, u_g, v_g, p_g, mach_g, entr_g]:
        arr[~mask] = np.nan

    # ── Solution contourf (2×3 grid) ─────────────────────────────────────────
    x_np = XX[0, :]          # (Nx,)  — x values along columns
    y_np = YY[:, 0]          # (Ny,)  — y values along rows

    fields = [
        (rho_g,  "ρ",    "plasma",  None, None),
        (u_g,    "u",    "RdBu_r",  None, None),
        (v_g,    "v",    "RdBu_r",  None, None),
        (p_g,    "p",    "viridis", None, None),
        (mach_g, "Mach", "jet",     0.0,  M_inf + 0.2),
        (entr_g, "Δs/s∞","seismic", -0.5, 0.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (field, label, cmap, vmin, vmax) in zip(axes.ravel(), fields):
        kw = dict(levels=50, cmap=cmap)
        if vmin is not None:
            kw.update(vmin=vmin, vmax=vmax)
        cf = ax.contourf(x_np, y_np, field, **kw)
        plt.colorbar(cf, ax=ax)
        # Draw ramp surface
        x_ramp = np.array([0.0, L])
        y_ramp = np.array([0.0, L * math.tan(math.radians(theta_deg))])
        ax.fill_between(x_ramp, y_ramp, -0.05, color="gray", alpha=0.4)
        ax.set_title(f"PINN: {label}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(0, L); ax.set_ylim(0, H)

    fig.suptitle(f"Compressible Euler — M={M_inf}, θ={theta_deg}°", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Mach number + analytical shock line ───────────────────────────────────
    beta_rad = math.radians(shock["beta_deg"])

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    cf2 = ax2.contourf(x_np, y_np, mach_g, levels=50, cmap="jet",
                       vmin=0.0, vmax=M_inf + 0.2)
    plt.colorbar(cf2, ax=ax2, label="Mach")

    # Analytical shock line from (0,0) at angle β
    x_shock = np.array([0.0, H / math.tan(beta_rad)])
    y_shock = x_shock * math.tan(beta_rad)
    y_shock = np.clip(y_shock, 0.0, H)
    ax2.plot(x_shock, y_shock, "w--", lw=2.0, label=f"Oblique shock  β={shock['beta_deg']:.1f}°")

    # Ramp surface
    x_ramp = np.array([0.0, L])
    y_ramp = np.array([0.0, L * math.tan(math.radians(theta_deg))])
    ax2.fill_between(x_ramp, y_ramp, -0.02, color="gray", alpha=0.5, label="Ramp")

    ax2.set_xlim(0, L); ax2.set_ylim(0, H)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title(f"Mach — M∞={M_inf}, θ={theta_deg}°  |  β_exact={shock['beta_deg']:.2f}°")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "oblique_shock.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Horizontal slice at y = H/2 (should show shock jump) ─────────────────
    # Pick the y-row closest to H/2
    j_mid  = int(np.argmin(np.abs(y_np - H / 2.0)))
    x_sl   = x_np
    mach_sl = mach_g[j_mid, :]
    rho_sl  = rho_g[j_mid, :]

    # Analytical shock x-position at y = y_np[j_mid]
    y_slice = float(y_np[j_mid])
    x_shock_sl = y_slice / math.tan(beta_rad) if math.tan(beta_rad) > 1e-9 else L + 1

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4))
    ax3a.plot(x_sl, mach_sl, "b-", lw=1.5, label="PINN Mach")
    ax3a.axvline(x_shock_sl, color="r", ls="--", lw=1.5, label=f"Shock (theory) x={x_shock_sl:.3f}")
    ax3a.axhline(shock["M2"], color="k", ls=":", lw=1.2, label=f"M2={shock['M2']:.3f}")
    ax3a.set_xlabel("x"); ax3a.set_ylabel("Mach")
    ax3a.set_title(f"Mach slice  y ≈ {y_slice:.3f}")
    ax3a.legend(fontsize=8)

    ax3b.plot(x_sl, rho_sl, "g-", lw=1.5, label="PINN ρ")
    ax3b.axvline(x_shock_sl, color="r", ls="--", lw=1.5, label="Shock (theory)")
    ax3b.axhline(shock["rho2"], color="k", ls=":", lw=1.2, label=f"ρ2={shock['rho2']:.3f}")
    ax3b.set_xlabel("x"); ax3b.set_ylabel("ρ")
    ax3b.set_title(f"Density slice  y ≈ {y_slice:.3f}")
    ax3b.legend(fontsize=8)

    fig3.suptitle(f"Horizontal slice at y ≈ {y_slice:.3f}  (M={M_inf}, θ={theta_deg}°)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "slice.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── Loss history ──────────────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(7, 3))
    ax4.semilogy(loss_hist, lw=1.2, label="Total")
    ax4.semilogy(pde_hist,  lw=1.0, alpha=0.7, label="PDE")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
    ax4.set_title(f"Ramp M={M_inf}, θ={theta_deg}°  — training loss")
    ax4.legend(); fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    np.savez(os.path.join(out_dir, "predictions.npz"),
             x=XX, y=YY, rho=rho_g, u=u_g, v=v_g, p=p_g, mach=mach_g)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # ── Checkpoint ────────────────────────────────────────────────────────────
    save_checkpoint(params, out_dir, metadata={
        "problem": "ramp",
        "network": {"type": cfg_get(cfg.network, "type", default="mlp"),
                    "layers": layers},
        "physics": {"gamma": gamma, "M_inf": M_inf, "theta_deg": theta_deg},
        "oblique_shock": shock,
    })

    print(f"\nOblique shock summary:")
    print(f"  β={shock['beta_deg']:.2f}°  M2={shock['M2']:.3f}  "
          f"ρ2/ρ∞={shock['rho2']:.3f}  p2/p∞={shock['p2']*gamma:.3f}")
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist, "oblique_shock": shock}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    _HERE    = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_ramp(load_config(cfg_path))
