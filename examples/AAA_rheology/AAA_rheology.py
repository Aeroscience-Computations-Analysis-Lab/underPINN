"""Steady Rheological (Carreau) Flow through an Axisymmetric AAA.

Run directly or via the CLI:

    python examples/AAA_rheology/AAA_rheology.py
    python examples/AAA_rheology/AAA_rheology.py myconfig.yaml
    python -m underPINN run examples/AAA_rheology/config.yaml

Same shear-thinning (Carreau) blood rheology as the rheological pipe-flow case
(Nagargoje, Mishra & Gupta, *Phys. Fluids* 33, 071904 (2021)), but in the
axisymmetric **bulge / AAA** geometry:

    R(x) = R_vessel + (R_AAA − R_vessel) cos²(½π|x−x0|/(L_AAA/2))

    μ(γ̇) = μ∞ + (μ0 − μ∞)[1 + (λγ̇)²]^((n−1)/2)

The domain and Reynolds number are IDENTICAL to the Newtonian AAA case
(``examples/AAA/config.yaml``): x ∈ [-3.5, 3.5], R_vessel = 0.5, R_AAA = 1.0,
Re = 40, V_max = 2.0 — only the constitutive law differs:

    μ*(γ̇) = 1 + (β − 1)[1 + (Cu γ̇)²]^((n−1)/2)
    blood (Table II):  β = μ0/μ∞ = 16,  n = 0.3568;  β = 1 ⇒ Newtonian.

A fully-developed Carreau profile is imposed at the inlet (straight vessel),
no-slip on the curved wall, p = 0 at the outlet.  There is no analytic solution
in the bulge, so accuracy is assessed via the inlet/outlet flow-rate balance.

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
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.carreau_ns_3d import CarreauNS3DPDE, carreau_developed_profile
from underPINN.geometry.aaa import BulgeGeometry
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import safe_choice


def run_AAA_rheology(cfg) -> dict:
    """Train a Carreau (shear-thinning) PINN through an axisymmetric bulge."""
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/AAA_rheology")
               if out else "outputs/AAA_rheology")
    os.makedirs(out_dir, exist_ok=True)

    # ── Physics: SAME domain and Re as the Newtonian AAA case ─────────────────
    # (examples/AAA/config.yaml) — only the constitutive law differs.
    Re    = float(cfg_get(ph, "Re",       default=40.0))
    Rv    = float(cfg_get(ph, "R_vessel", default=0.5))
    Ra    = float(cfg_get(ph, "R_AAA",    default=1.0))
    L     = float(cfg_get(ph, "L",        default=7.0))
    x_lo  = float(cfg_get(ph, "x_lo",     default=-3.5))
    x0    = float(cfg_get(ph, "x0",       default=-2.0))
    La    = float(cfg_get(ph, "L_AAA",    default=1.5))
    V_max = float(cfg_get(ph, "V_max",    default=2.0))
    # Carreau rheology (blood: β=16, n=0.3568 from the paper Table II)
    beta  = float(cfg_get(ph, "beta",     default=16.0))
    Cu    = float(cfg_get(ph, "Cu",       default=10.0))
    n     = float(cfg_get(ph, "n",        default=0.3568))

    x_hi = x_lo + L

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
    n_int    = int(cfg_get(d, "n_interior", default=60000))
    n_wall   = int(cfg_get(d, "n_wall",     default=12000))
    n_inlet  = int(cfg_get(d, "n_inlet",    default=1500))
    n_outlet = int(cfg_get(d, "n_outlet",   default=1500))

    print("Carreau (shear-thinning) AAA flow")
    print(f"  Re={Re},  R_vessel={Rv},  R_AAA={Ra},  x ∈ [{x_lo}, {x_hi}], "
          f"x0={x0}, L_AAA={La}, V_max={V_max}   (same domain/Re as Newtonian AAA)")
    print(f"  Carreau:  β={beta},  Cu={Cu},  n={n}")

    # ── Carreau developed inlet profile ────────────────────────────────────────
    # 1-D solver works in unit coordinates (r*∈[0,1], u*∈[0,1]); the PDE's
    # Carreau term uses the simulation shear rate γ̇ = (V_max/Rv)·(du*/dr*),
    # so the unit-profile solver needs Cu_eff = Cu·V_max/Rv.
    Cu_eff = Cu * V_max / Rv
    r_ref, u_ref, _ = carreau_developed_profile(beta, Cu_eff, n, u_center=1.0)

    # ── Geometry + collocation (same units as the Newtonian case) ─────────────
    geom = BulgeGeometry(R_vessel=Rv, R_AAA=Ra, L=L,
                         x_lo=x_lo, x0=x0, L_AAA=La)
    xyz_r   = jnp.array(geom.sample_interior(n_int,  seed=seed))
    xyz_w   = jnp.array(geom.sample_wall(    n_wall, seed=seed + 1))
    xyz_in  = jnp.array(geom.sample_inlet(   n_inlet, seed=seed + 2))
    xyz_out = jnp.array(geom.sample_outlet(  n_outlet, seed=seed + 3))

    r_in   = np.sqrt(np.asarray(xyz_in[:, 1]) ** 2 + np.asarray(xyz_in[:, 2]) ** 2)
    u_in_t = jnp.array((V_max * np.interp(r_in / Rv, r_ref, u_ref)).astype(np.float32))

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_cfg  = network_config(cfg)
    net_type = net_cfg["type"]
    layers   = net_cfg["layers"]
    model    = build_model(net_cfg)
    pde      = CarreauNS3DPDE(model, Re=Re, beta=beta, Cu=Cu, n=n)
    print(f"  Network: {type(model).__name__} ({net_type})  layers={layers}")

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
            res   = pde.residual(p, xyz_r)
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

    # ── Flow-rate balance ─────────────────────────────────────────────────────
    xyz_in_v  = jnp.array(geom.sample_inlet(2000,  seed=77))
    xyz_out_v = jnp.array(geom.sample_outlet(2000, seed=78))
    A_vessel  = np.pi * Rv ** 2
    Q_in  = float(jnp.mean(model.apply(params, xyz_in_v)[:, 0]))  * A_vessel
    Q_out = float(jnp.mean(model.apply(params, xyz_out_v)[:, 0])) * A_vessel
    flow_balance = abs(Q_in - Q_out) / (abs(Q_in) + 1e-10)
    print(f"\nFlow-rate balance:  Q_in={Q_in:.4f}  Q_out={Q_out:.4f}"
          f"  rel_diff={flow_balance:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    # 1. Axial-velocity cross-section at the bulge centre
    N_cs = 80
    yz   = np.linspace(-Ra, Ra, N_cs, dtype=np.float32)
    YY, ZZ = np.meshgrid(yz, yz)
    xyz_cs = jnp.array(np.stack([np.full(N_cs * N_cs, float(x0), np.float32),
                                 YY.ravel(), ZZ.ravel()], axis=1))
    u_cs  = np.array(model.apply(params, xyz_cs)[:, 0]).reshape(N_cs, N_cs)
    R_x0  = float(geom.radius_at(np.array([x0]))[0])
    u_cs  = np.where(YY ** 2 + ZZ ** 2 > R_x0 ** 2, np.nan, u_cs)
    figc, axc = plt.subplots(figsize=(5, 4))
    cf = axc.contourf(yz, yz, u_cs, levels=50, cmap="jet")
    plt.colorbar(cf, ax=axc, label="u")
    axc.set_aspect("equal")
    axc.set_xlabel("y")
    axc.set_ylabel("z")
    axc.set_title(f"u at bulge centre x={x0:.1f}")
    figc.tight_layout()
    figc.savefig(os.path.join(out_dir, "AAA_crosssection.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(figc)

    # 2. Centreline velocity + wall profile
    N_ax = 240
    x_ax = np.linspace(x_lo, x_hi, N_ax, dtype=np.float32)
    xyz_ax = jnp.array(np.stack([x_ax, np.zeros(N_ax, np.float32),
                                 np.zeros(N_ax, np.float32)], axis=1))
    u_ctr = np.array(model.apply(params, xyz_ax)[:, 0])
    R_ax  = geom.radius_at(x_ax)
    figa, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(x_ax, u_ctr, "b-", lw=2.0, label="u centreline / U")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u (centreline)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(x_ax, R_ax, "r--", lw=1.5, label="R(x)")
    ax2.fill_between(x_ax,  R_ax,  Rv, alpha=0.12, color="r")
    ax2.fill_between(x_ax, -R_ax, -Rv, alpha=0.12, color="r")
    ax2.set_ylabel("R(x)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"Carreau AAA — centreline velocity  (Re={Re:.0f}, Cu={Cu:.1f})")
    ax1.grid(alpha=0.3)
    figa.tight_layout()
    figa.savefig(os.path.join(out_dir, "AAA_centreline.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(figa)

    # 3. Apparent-viscosity cross-section at the bulge centre
    mu_cs = np.array(pde.apparent_viscosity(params, xyz_cs)).reshape(N_cs, N_cs)
    mu_cs = np.where(YY ** 2 + ZZ ** 2 > R_x0 ** 2, np.nan, mu_cs)
    figm, axm = plt.subplots(figsize=(5, 4))
    cfm = axm.contourf(yz, yz, mu_cs, levels=50, cmap="viridis")
    plt.colorbar(cfm, ax=axm, label="μ* = μ/μ∞")
    axm.set_aspect("equal")
    axm.set_xlabel("y")
    axm.set_ylabel("z")
    axm.set_title(f"Apparent viscosity at x={x0:.1f}")
    figm.tight_layout()
    figm.savefig(os.path.join(out_dir, "AAA_viscosity.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(figm)

    # 4. Loss
    figl, axl = plt.subplots(figsize=(7, 3))
    axl.semilogy(loss_hist, lw=1.0)
    axl.set_xlabel("Epoch")
    axl.set_ylabel("Loss")
    axl.set_title(f"Carreau AAA flow  Re={Re:.0f}")
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
        "problem": "AAA_rheology",
        "network": net_cfg,
        "physics": {"Re": Re, "R_vessel": Rv, "R_AAA": Ra, "L": L,
                    "x_lo": x_lo, "x0": x0, "L_AAA": La, "V_max": V_max,
                    "beta": beta, "Cu": Cu, "n": n},
        "results": {"flow_balance": flow_balance, "n_epochs": len(loss_hist)},
    })
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist,
            "flow_balance": flow_balance, "Re": Re, "beta": beta, "Cu": Cu,
            "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_AAA_rheology(load_config(cfg_path))
