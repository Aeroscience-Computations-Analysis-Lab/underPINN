"""Steady Rheological (Carreau) Pipe Flow PINN.

Run directly or via the CLI:

    python examples/pipe_flow_rheology/pipe_flow_rheology.py
    python examples/pipe_flow_rheology/pipe_flow_rheology.py myconfig.yaml
    python -m underPINN run examples/pipe_flow_rheology/config.yaml

3-D steady incompressible flow of a shear-thinning fluid (blood) through a
cylindrical pipe, using the **Carreau** rheological model of Nagargoje, Mishra
& Gupta, *Phys. Fluids* 33, 071904 (2021):

    μ(γ̇) = μ∞ + (μ0 − μ∞)[1 + (λ γ̇)²]^((n−1)/2)
    Table II (blood):  μ0 = 0.056,  μ∞ = 0.0035 Pa·s,  λ = 3.131 s,  n = 0.3568

Non-dimensional groups (length R, velocity U, viscosity μ∞):
    Re = ρ U R / μ∞,   β = μ0/μ∞,   Cu = λ U / R

The fully-developed Carreau profile (computed here by a 1-D ODE solve) is
imposed at the inlet, no-slip on the wall, and p = 0 at the outlet.  Because
the flow is fully developed, the exact solution is that profile everywhere —
used to validate the PINN and to contrast the shear-thinning (flatter) profile
with the Newtonian parabola.

Network: (x, y, z) → (u, v, w, p)
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
from underPINN.pde.carreau_ns_3d import CarreauNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# 1-D fully-developed Carreau pipe profile (non-dimensional, R*=1)
# ---------------------------------------------------------------------------

def _mu_star(a, beta, Cu, n):
    return 1.0 + (beta - 1.0) * (1.0 + (Cu * a) ** 2) ** ((n - 1.0) / 2.0)


def _shear_from_stress(tau_mag, beta, Cu, n):
    """Invert  μ*(a)·a = tau_mag  for the shear-rate magnitude a ≥ 0 (bisection).

    The shear stress μ*(a)·a is monotone increasing in a for a Carreau fluid,
    so the inverse is unique.
    """
    if tau_mag <= 0.0:
        return 0.0

    def f(a):
        return _mu_star(a, beta, Cu, n) * a - tau_mag

    lo, hi = 0.0, 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def carreau_developed_profile(beta, Cu, n, u_center=1.0, n_r=400):
    """Fully-developed Carreau velocity profile u*(r) on r ∈ [0, 1].

    Solves  μ*(|u*'|) u*' = ½ P r  with u*(1)=0, matching the centreline value
    ``u_center`` by adjusting the (non-dim) pressure-gradient constant P.

    Returns
    -------
    r      : (n_r,) radial grid in [0, 1]
    u      : (n_r,) velocity profile  (u[0] = u_center)
    P      : the matched pressure-gradient constant  (= Re · dp*/dx*)
    """
    r = np.linspace(0.0, 1.0, n_r)

    def profile_for_P(P):
        # shear magnitude a(r) from  μ*(a) a = ½ |P| r
        a = np.array([_shear_from_stress(0.5 * abs(P) * ri, beta, Cu, n) for ri in r])
        # u'(r) = -a ; integrate from wall inward: u(r) = ∫_r^1 a dr'
        # cumulative integral from 1 → r  (reverse cumulative trapezoid)
        du = a[:-1] + a[1:]
        dr = np.diff(r)
        seg = 0.5 * du * dr                          # trapezoid segments
        # u(r_i) = sum of segments from i..end
        u = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        return u

    # Bisection on |P| so that u(0) = u_center
    P_lo, P_hi = 1e-6, 1.0
    while profile_for_P(P_hi)[0] < u_center:
        P_hi *= 2.0
    for _ in range(80):
        P_mid = 0.5 * (P_lo + P_hi)
        if profile_for_P(P_mid)[0] < u_center:
            P_lo = P_mid
        else:
            P_hi = P_mid
    P = 0.5 * (P_lo + P_hi)
    u = profile_for_P(P)
    return r, u, -P            # P returned as Re·dp/dx (negative, favourable)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pipe_flow_rheology(cfg) -> dict:
    """Train a PINN on steady Carreau (shear-thinning) pipe flow."""
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/pipe_flow_rheology")
               if out else "outputs/pipe_flow_rheology")
    os.makedirs(out_dir, exist_ok=True)

    # ── Dimensional (paper Table II) → non-dimensional groups ─────────────────
    rho    = float(cfg_get(ph, "rho",    default=1060.0))
    mu0    = float(cfg_get(ph, "mu0",    default=0.056))
    mu_inf = float(cfg_get(ph, "mu_inf", default=0.0035))
    lam    = float(cfg_get(ph, "lam",    default=3.131))
    n      = float(cfg_get(ph, "n",      default=0.3568))
    R      = float(cfg_get(ph, "R",      default=0.004))     # pipe radius (m)
    L_phys = float(cfg_get(ph, "L",      default=0.04))      # pipe length (m)
    U      = float(cfg_get(ph, "U",      default=0.1))       # centreline velocity (m/s)
    x_lo   = float(cfg_get(ph, "x_lo",   default=0.0))

    Re   = rho * U * R / mu_inf
    beta = mu0 / mu_inf
    Cu   = lam * U / R

    # Work in non-dimensional units: lengths / R, velocities / U.
    R_nd = 1.0
    L_nd = L_phys / R
    x_lo_nd = x_lo / R
    x_hi_nd = x_lo_nd + L_nd
    x_mid_nd = 0.5 * (x_lo_nd + x_hi_nd)

    W_PDE    = float(cfg_get(lw, "w_pde",    default=1.0))
    W_WALL   = float(cfg_get(lw, "w_wall",   default=100.0))
    W_INLET  = float(cfg_get(lw, "w_inlet",  default=50.0))
    W_OUTLET = float(cfg_get(lw, "w_outlet", default=20.0))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    seed      = int(cfg_get(tr, "seed",        default=0))
    batch_r   = int(cfg_get(tr, "batch_r",     default=2048))
    batch_bc  = int(cfg_get(tr, "batch_bc",    default=512))

    d        = cfg.data
    n_int    = int(cfg_get(d, "n_interior", default=40000))
    n_wall   = int(cfg_get(d, "n_wall",     default=8000))
    n_inlet  = int(cfg_get(d, "n_inlet",    default=1500))
    n_outlet = int(cfg_get(d, "n_outlet",   default=1500))

    print("Carreau (shear-thinning) pipe flow")
    print(f"  Dimensional: ρ={rho}, μ0={mu0}, μ∞={mu_inf}, λ={lam}, n={n}, "
          f"R={R} m, U={U} m/s")
    print(f"  Non-dim:  Re={Re:.2f},  β=μ0/μ∞={beta:.2f},  Cu=λU/R={Cu:.2f},  n={n}")
    print(f"  Pipe (non-dim):  x ∈ [{x_lo_nd:.2f}, {x_hi_nd:.2f}],  R={R_nd}")

    # ── Fully-developed Carreau reference profile (non-dim, centreline = 1) ───
    r_ref, u_ref, P_ref = carreau_developed_profile(beta, Cu, n, u_center=1.0)
    dpdx_nd = P_ref / Re                       # non-dim dp*/dx*  (negative)

    def inlet_velocity_nd(yz_r):               # interpolate ref profile at radius
        return np.interp(yz_r, r_ref, u_ref).astype(np.float32)

    # ── Geometry + collocation (non-dim) ──────────────────────────────────────
    pipe    = Pipe(R=R_nd, L=L_nd, x_lo=x_lo_nd)
    xyz_r   = jnp.array(pipe.sample_interior(n_int,  seed=seed))
    xyz_w   = jnp.array(pipe.sample_wall(    n_wall, seed=seed + 1))
    xyz_in  = jnp.array(pipe.sample_inlet(   n_inlet, seed=seed + 2))
    xyz_out = jnp.array(pipe.sample_outlet(  n_outlet, seed=seed + 3))

    # inlet target velocity (Carreau developed profile)
    r_in   = np.sqrt(np.asarray(xyz_in[:, 1]) ** 2 + np.asarray(xyz_in[:, 2]) ** 2)
    u_in_t = jnp.array(inlet_velocity_nd(r_in))

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_type = str(cfg_get(cfg.network, "type", default="gated_mlp")).lower()
    _net_cls = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net_type, MLP)
    layers   = list(cfg.network.layers)
    model    = _net_cls(layers=layers)
    pde      = CarreauNS3DPDE(model, Re=Re, beta=beta, Cu=Cu, n=n)
    print(f"  Network: {_net_cls.__name__}  layers={layers}")

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 3)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, state, xyz_r, xyz_w, xyz_in, u_in, xyz_out):
        def loss_fn(p):
            res   = pde.residual(p, xyz_r)                 # (N, 4)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

            out_w  = model.apply(p, xyz_w)
            wall_l = jnp.mean(out_w[:, 0]**2 + out_w[:, 1]**2 + out_w[:, 2]**2)

            out_in = model.apply(p, xyz_in)
            in_l   = (jnp.mean((out_in[:, 0] - u_in) ** 2)
                      + jnp.mean(out_in[:, 1] ** 2)
                      + jnp.mean(out_in[:, 2] ** 2))

            out_out  = model.apply(p, xyz_out)
            outlet_l = jnp.mean(out_out[:, 3] ** 2)

            total = (W_PDE * pde_l + W_WALL * wall_l
                     + W_INLET * in_l + W_OUTLET * outlet_l)
            return total, (pde_l, wall_l, in_l, outlet_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Restart ───────────────────────────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    logger = ConsoleLogger(log_every=log_every)
    N_r, N_w = xyz_r.shape[0], xyz_w.shape[0]
    N_in, N_out = xyz_in.shape[0], xyz_out.shape[0]
    key = jax.random.PRNGKey(seed + 99)

    try:
        for ep in range(start_ep, epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir   = safe_choice(k1, N_r,   batch_r)
            iw   = safe_choice(k2, N_w,   batch_bc)
            iin  = safe_choice(k3, N_in,  min(batch_bc, N_in))
            iout = safe_choice(k4, N_out, min(batch_bc, N_out))

            params, opt_state, total, (pl, wl, il, ol) = step(
                params, opt_state,
                xyz_r[ir], xyz_w[iw], xyz_in[iin], u_in_t[iin], xyz_out[iout])
            loss_hist.append(float(total))
            logger.on_epoch_end(ep, {"loss": float(total), "pde": float(pl),
                                     "wall": float(wl), "inlet": float(il)})
            restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Validation: radial profile at mid-pipe vs Carreau reference ───────────
    Nr     = 120
    r_line = np.linspace(0.0, R_nd, Nr, dtype=np.float32)
    xyz_rp = jnp.array(np.stack([np.full(Nr, x_mid_nd, np.float32),
                                 r_line, np.zeros(Nr, np.float32)], axis=1))
    u_pinn  = np.array(model.apply(params, xyz_rp)[:, 0])
    u_carr  = np.interp(r_line, r_ref, u_ref)
    u_newt  = 1.0 - r_line ** 2                       # Newtonian parabola (centreline 1)
    rel_l2  = float(np.linalg.norm(u_pinn - u_carr) / (np.linalg.norm(u_carr) + 1e-12))
    print(f"\nRel-L² (PINN vs Carreau reference) for u(r): {rel_l2:.3e}")

    # apparent viscosity along the radius
    mu_pinn = np.array(pde.apparent_viscosity(params, xyz_rp))

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(r_line, u_newt, "k:",  lw=1.8, label="Newtonian parabola")
    ax1.plot(r_line, u_carr, "k-",  lw=2.0, label="Carreau reference")
    ax1.plot(r_line, u_pinn, "b--", lw=1.8, label="PINN")
    ax1.set_xlabel("r / R")
    ax1.set_ylabel("u / U")
    ax1.set_title(f"Axial velocity profile  (Re={Re:.0f}, Cu={Cu:.1f}, n={n})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(r_line, mu_pinn, "b-", lw=1.8, label="PINN  μ*(r)")
    ax2.axhline(beta, color="grey", ls=":", lw=1.0, label=f"β=μ0/μ∞={beta:.0f} (low shear)")
    ax2.axhline(1.0,  color="grey", ls="--", lw=1.0, label="1 (high shear)")
    ax2.set_xlabel("r / R")
    ax2.set_ylabel("μ* = μ / μ∞")
    ax2.set_title("Apparent viscosity (shear thinning)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("Carreau pipe flow — shear-thinning blood", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rheology_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cross-section contour of u at mid-pipe
    N_cs = 80
    yz   = np.linspace(-R_nd, R_nd, N_cs, dtype=np.float32)
    YY, ZZ = np.meshgrid(yz, yz)
    xyz_cs = jnp.array(np.stack([np.full(N_cs * N_cs, x_mid_nd, np.float32),
                                 YY.ravel(), ZZ.ravel()], axis=1))
    u_cs = np.array(model.apply(params, xyz_cs)[:, 0]).reshape(N_cs, N_cs)
    u_cs = np.where(YY ** 2 + ZZ ** 2 > R_nd ** 2, np.nan, u_cs)
    figc, axc = plt.subplots(figsize=(5, 4))
    cf = axc.contourf(yz, yz, u_cs, levels=50, cmap="jet")
    plt.colorbar(cf, ax=axc, label="u / U")
    axc.set_aspect("equal")
    axc.set_xlabel("y / R")
    axc.set_ylabel("z / R")
    axc.set_title(f"u cross-section at x={x_mid_nd:.1f}")
    figc.tight_layout()
    figc.savefig(os.path.join(out_dir, "rheology_crosssection.png"), dpi=150, bbox_inches="tight")
    plt.close(figc)

    # Loss
    figl, axl = plt.subplots(figsize=(7, 3))
    axl.semilogy(loss_hist, lw=1.0)
    axl.set_xlabel("Epoch")
    axl.set_ylabel("Loss")
    axl.set_title(f"Carreau pipe flow  Re={Re:.0f}")
    figl.tight_layout()
    figl.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(figl)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    uvwp = np.array(model.apply(params, xyz_r))
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xyz_r[:, 0]), "y": np.array(xyz_r[:, 1]),
                   "z": np.array(xyz_r[:, 2])},
        outputs = {"u_pred": uvwp[:, 0], "v_pred": uvwp[:, 1],
                   "w_pred": uvwp[:, 2], "p_pred": uvwp[:, 3]},
    )
    save_checkpoint(params, out_dir, metadata={
        "problem": "pipe_flow_rheology",
        "network": {"type": net_type, "layers": layers},
        "physics": {"rho": rho, "mu0": mu0, "mu_inf": mu_inf, "lam": lam, "n": n,
                    "R": R, "U": U, "Re": Re, "beta": beta, "Cu": Cu},
        "results": {"rel_l2_u": rel_l2, "dpdx_nd": dpdx_nd, "n_epochs": len(loss_hist)},
    })
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist, "rel_l2_u": rel_l2,
            "Re": Re, "beta": beta, "Cu": Cu, "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_pipe_flow_rheology(load_config(cfg_path))
