"""Steady Rheological (Carreau) Pipe Flow PINN.

Run directly or via the CLI:

    python examples/pipe_flow_rheology/pipe_flow_rheology.py
    python examples/pipe_flow_rheology/pipe_flow_rheology.py myconfig.yaml
    python -m underPINN run examples/pipe_flow_rheology/config.yaml

3-D steady incompressible flow of a shear-thinning fluid (blood) through a
cylindrical pipe, using the **Carreau** rheological model of Nagargoje, Mishra
& Gupta, *Phys. Fluids* 33, 071904 (2021).

The domain and Reynolds number are IDENTICAL to the Newtonian pipe case
(``examples/pipe_flow/pipe_flow.yaml``): x ∈ [-3.5, 3.5], R = 0.5 (D = 1),
Re = 40, inlet centreline velocity U_max = 2.0 — only the constitutive law
differs:

    μ*(γ̇) = 1 + (β − 1)[1 + (Cu γ̇)²]^((n−1)/2)
    blood (Table II):  β = μ0/μ∞ = 16,  n = 0.3568;  β = 1 ⇒ Newtonian.

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
from underPINN.pde.carreau_ns_3d import CarreauNS3DPDE, carreau_developed_profile
from underPINN.geometry.pipe import Pipe
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import safe_choice


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

    # ── Physics: SAME domain and Re as the Newtonian pipe case ────────────────
    # (examples/pipe_flow/pipe_flow.yaml) — only the constitutive law differs.
    Re    = float(cfg_get(ph, "Re",    default=40.0))
    R     = float(cfg_get(ph, "R",     default=0.5))
    L     = float(cfg_get(ph, "L",     default=7.0))
    x_lo  = float(cfg_get(ph, "x_lo",  default=-3.5))
    U_max = float(cfg_get(ph, "U_max", default=2.0))
    # Carreau rheology (blood: β=16, n=0.3568 from the paper Table II)
    beta  = float(cfg_get(ph, "beta",  default=16.0))
    Cu    = float(cfg_get(ph, "Cu",    default=10.0))
    n     = float(cfg_get(ph, "n",     default=0.3568))

    x_hi  = x_lo + L
    x_mid = 0.5 * (x_lo + x_hi)

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
    print(f"  Re={Re},  R={R} (D={2*R}),  x ∈ [{x_lo}, {x_hi}],  U_max={U_max}"
          f"   (same domain/Re as Newtonian pipe)")
    print(f"  Carreau:  β={beta},  Cu={Cu},  n={n}")

    # ── Fully-developed Carreau reference profile ──────────────────────────────
    # The 1-D solver works in unit coordinates (r*∈[0,1], u*∈[0,1]).  The PDE's
    # Carreau term uses the SIMULATION shear rate γ̇ = (U_max/R)·(du*/dr*), so
    # the unit-profile solver must use Cu_eff = Cu·U_max/R for consistency.
    Cu_eff = Cu * U_max / R
    r_ref, u_ref, P_ref = carreau_developed_profile(beta, Cu_eff, n, u_center=1.0)
    dpdx_nd = P_ref / Re                       # unit-system pressure gradient

    def inlet_velocity(yz_r):                  # physical radius → physical u
        return (U_max * np.interp(yz_r / R, r_ref, u_ref)).astype(np.float32)

    # ── Geometry + collocation (same units as the Newtonian case) ─────────────
    pipe    = Pipe(R=R, L=L, x_lo=x_lo)
    xyz_r   = jnp.array(pipe.sample_interior(n_int,  seed=seed))
    xyz_w   = jnp.array(pipe.sample_wall(    n_wall, seed=seed + 1))
    xyz_in  = jnp.array(pipe.sample_inlet(   n_inlet, seed=seed + 2))
    xyz_out = jnp.array(pipe.sample_outlet(  n_outlet, seed=seed + 3))

    # inlet target velocity (Carreau developed profile)
    r_in   = np.sqrt(np.asarray(xyz_in[:, 1]) ** 2 + np.asarray(xyz_in[:, 2]) ** 2)
    u_in_t = jnp.array(inlet_velocity(r_in))

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
    r_line = np.linspace(0.0, R, Nr, dtype=np.float32)
    xyz_rp = jnp.array(np.stack([np.full(Nr, x_mid, np.float32),
                                 r_line, np.zeros(Nr, np.float32)], axis=1))
    u_pinn  = np.array(model.apply(params, xyz_rp)[:, 0])
    u_carr  = U_max * np.interp(r_line / R, r_ref, u_ref)
    u_newt  = U_max * (1.0 - (r_line / R) ** 2)       # Newtonian parabola
    rel_l2  = float(np.linalg.norm(u_pinn - u_carr) / (np.linalg.norm(u_carr) + 1e-12))
    print(f"\nRel-L² (PINN vs Carreau reference) for u(r): {rel_l2:.3e}")

    # apparent viscosity along the radius
    mu_pinn = np.array(pde.apparent_viscosity(params, xyz_rp))

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(r_line, u_newt, "k:",  lw=1.8, label="Newtonian parabola")
    ax1.plot(r_line, u_carr, "k-",  lw=2.0, label="Carreau reference")
    ax1.plot(r_line, u_pinn, "b--", lw=1.8, label="PINN")
    ax1.set_xlabel("r")
    ax1.set_ylabel("u")
    ax1.set_title(f"Axial velocity profile  (Re={Re:.0f}, Cu={Cu:.1f}, n={n})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(r_line, mu_pinn, "b-", lw=1.8, label="PINN  μ*(r)")
    ax2.axhline(beta, color="grey", ls=":", lw=1.0, label=f"β=μ0/μ∞={beta:.0f} (low shear)")
    ax2.axhline(1.0,  color="grey", ls="--", lw=1.0, label="1 (high shear)")
    ax2.set_xlabel("r")
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
    yz   = np.linspace(-R, R, N_cs, dtype=np.float32)
    YY, ZZ = np.meshgrid(yz, yz)
    xyz_cs = jnp.array(np.stack([np.full(N_cs * N_cs, x_mid, np.float32),
                                 YY.ravel(), ZZ.ravel()], axis=1))
    u_cs = np.array(model.apply(params, xyz_cs)[:, 0]).reshape(N_cs, N_cs)
    u_cs = np.where(YY ** 2 + ZZ ** 2 > R ** 2, np.nan, u_cs)
    figc, axc = plt.subplots(figsize=(5, 4))
    cf = axc.contourf(yz, yz, u_cs, levels=50, cmap="jet")
    plt.colorbar(cf, ax=axc, label="u")
    axc.set_aspect("equal")
    axc.set_xlabel("y")
    axc.set_ylabel("z")
    axc.set_title(f"u cross-section at x={x_mid:.1f}")
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
        "physics": {"Re": Re, "R": R, "L": L, "x_lo": x_lo, "U_max": U_max,
                    "beta": beta, "Cu": Cu, "n": n},
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
