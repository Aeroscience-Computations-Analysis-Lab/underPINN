"""2-D Compressible Navier–Stokes — Mach-3 flow over a viscous compression ramp.

Shock / boundary-layer interaction (SBLI): supersonic M∞ = 3 flow over a flat
bottom wall followed by a compression ramp.  Unlike the inviscid Euler ramp,
the walls are **no-slip** and **isothermal** at the stagnation (total)
temperature, so a viscous boundary layer develops and interacts with the
ramp-induced oblique shock.

Run directly or via the CLI:

    python examples/ramp_ns/ramp_ns.py
    python examples/ramp_ns/ramp_ns.py myconfig.yaml
    python -m underPINN run examples/ramp_ns/config.yaml

PDE: 2-D steady compressible Navier–Stokes (``CompressibleNS2DPDE``), conservative
flux form with Newtonian viscous stresses + Fourier conduction (Re = 1e4, Pr = 0.72).

Network:  (x, y) → (f_ρ, f_u, f_v, f_T);  ρ, T via softplus positivity.

Boundary conditions
-------------------
* Inlet  (x = 0)         — freestream  ρ = 1, u = 1, v = 0, T = 1  (supersonic)
* Slip wall [0, slip_end]— free-slip / symmetry: no-penetration  v = 0 only
                           (a short leading run that removes the inlet/no-slip
                           corner conflict — at (0,0) the inlet's u=1 and the
                           no-slip u=0 disagree; slip only needs v=0, which agrees)
* No-slip wall (after)   — no-slip  u = v = 0,  isothermal  T = T₀ (total temp)
* Upper farfield (y = H) — freestream
* Outlet (x = L)         — supersonic outflow; no BC imposed
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from underPINN.config.loader import cfg_get, save_config
from underPINN.geometry.ramp import RampGeometry
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.compressible_ns_2d import CompressibleNS2DPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.io import save_predictions
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import rad_resample, safe_choice


def run_ramp_ns(cfg) -> dict:
    """Train a PINN on 2-D steady compressible NS over a viscous compression ramp."""
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/ramp_ns") if out else "outputs/ramp_ns"
    os.makedirs(out_dir, exist_ok=True)

    gamma     = float(cfg_get(ph, "gamma",     default=1.4))
    M_inf     = float(cfg_get(ph, "M_inf",     default=3.0))
    Re        = float(cfg_get(ph, "Re",        default=1.0e4))
    Pr        = float(cfg_get(ph, "Pr",        default=0.72))
    theta_deg = float(cfg_get(ph, "theta_deg", default=15.0))
    mu_law    = str(cfg_get(ph, "mu_law",      default="constant"))
    art_visc  = float(cfg_get(ph, "art_visc",  default=0.0))   # shock-capturing −ε∇²U
    av_sensor = str(cfg_get(ph, "av_sensor",   default="ducros"))  # 'ducros' | 'global'

    g  = cfg.geometry
    L          = float(cfg_get(g, "L",          default=2.0))
    H          = float(cfg_get(g, "H",          default=1.0))
    ramp_start = float(cfg_get(g, "ramp_start", default=0.8))
    slip_end   = float(cfg_get(g, "slip_end",   default=0.15))   # slip region [0, slip_end]

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    seed      = int(cfg_get(tr, "seed",        default=0))
    batch_r   = int(cfg_get(tr, "batch_r",     default=2048))
    batch_bc  = int(cfg_get(tr, "batch_bc",    default=256))

    rar_period = int(cfg_get(tr, "rar_period",     default=0))
    rar_cand   = int(cfg_get(tr, "rar_candidates", default=5))
    rar_k      = float(cfg_get(tr, "rar_k",        default=1.0))
    rar_c      = float(cfg_get(tr, "rar_c",        default=1.0))

    W_PDE   = float(cfg_get(lw, "w_pde",   default=1.0))
    W_INLET = float(cfg_get(lw, "w_inlet", default=100.0))
    W_WALL  = float(cfg_get(lw, "w_wall",  default=100.0))
    W_SLIP  = float(cfg_get(lw, "w_slip",  default=80.0))    # slip region (v=0)
    W_UPPER = float(cfg_get(lw, "w_upper", default=20.0))

    d = cfg.data
    n_int   = int(cfg_get(d, "n_interior", default=12000))
    # Hybrid interior collocation:  a FIXED uniform pool (global coverage) +
    # a RESIDUAL-ADAPTIVE pool that migrates to the boundary layer / shock.
    # Default split: half uniform, half adaptive (override n_uniform / n_adapt).
    n_uniform = int(cfg_get(d, "n_uniform", default=n_int // 2))
    n_adapt   = int(cfg_get(d, "n_adapt",   default=n_int - n_int // 2))
    n_bl      = int(cfg_get(d, "n_bl",      default=n_int // 2))   # wall-clustered (BL)
    bl_beta   = float(cfg_get(d, "bl_beta", default=4.0))         # BL clustering strength
    n_in    = int(cfg_get(d, "n_inlet",    default=400))
    n_wall  = int(cfg_get(d, "n_wall",     default=600))
    n_slip  = int(cfg_get(d, "n_slip",     default=200))
    n_upper = int(cfg_get(d, "n_upper",    default=300))
    # batch fractions:  adaptive (shock) + boundary-layer (wall);  rest = uniform
    adapt_frac = float(cfg_get(tr, "adapt_frac", default=0.4))
    bl_frac    = float(cfg_get(tr, "bl_frac",    default=0.3))

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_cfg  = network_config(cfg)
    net_type = net_cfg["type"]
    layers   = net_cfg["layers"]
    model    = build_model(net_cfg)
    pde      = CompressibleNS2DPDE(model, gamma=gamma, M_inf=M_inf, Re=Re, Pr=Pr,
                                   mu_law=mu_law, art_visc=art_visc,
                                   av_sensor=av_sensor)
    T0 = pde.total_temperature()
    rho_inf, u_inf, v_inf, T_inf = pde.freestream()

    print(f"Ramp NS (SBLI):  M={M_inf},  Re={Re:g},  Pr={Pr},  θ={theta_deg}°,  γ={gamma}")
    print(f"  Domain x∈[0,{L}] y∈[0,{H}];  flat wall on [0,{ramp_start}] then ramp")
    print(f"  Slip wall (v=0) on [0,{slip_end}], no-slip + isothermal T0={T0:.3f} after")
    print(f"  Viscosity law: {mu_law};  Network: {type(model).__name__} ({net_type})  layers={layers}")
    print(f"  Interior collocation: {n_uniform} uniform + {n_bl} wall-clustered(BL) "
          f"+ {n_adapt} adaptive (RAR every {rar_period} ep)")
    if art_visc > 0.0:
        print(f"  Artificial viscosity:  ε = {art_visc:g}  sensor={av_sensor} "
              f"({'shock-localised' if av_sensor == 'ducros' else 'global'})")

    # ── Geometry ──────────────────────────────────────────────────────────────
    geom = RampGeometry(theta_deg, L=L, H=H, ramp_start=ramp_start, slip_end=slip_end)
    # Three interior pools:
    #   • uniform        — fixed global coverage
    #   • wall-clustered — fixed near-wall prior (resolves the boundary layer)
    #   • adaptive       — RAR-resampled toward high residual (finds the shock)
    xy_uniform = jnp.array(geom.sample_interior(n_uniform, seed=seed))
    xy_bl      = jnp.array(geom.sample_boundary_layer(n_bl, beta=bl_beta, seed=seed + 7))
    xy_adapt   = jnp.array(geom.sample_interior(n_adapt,   seed=seed + 101))
    xy_in   = jnp.array(geom.sample_inlet(n_in))
    xy_w    = jnp.array(geom.sample_noslip_wall(n_wall))     # no-slip region
    xy_slip = jnp.array(geom.sample_slip_wall(n_slip))       # slip region (v=0)
    xy_up   = jnp.array(geom.sample_upper(n_upper))

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # ── JIT step ──────────────────────────────────────────────────────────────
    @jax.jit
    def step(params, state, ru_b, rbl_b, ra_b, in_b, w_b, slip_b, up_b):
        def loss_fn(p):
            # PDE residual on the UNION of the uniform + BL + adaptive batches
            r_b   = jnp.concatenate([ru_b, rbl_b, ra_b], axis=0)
            res   = pde.residual(p, r_b)                       # (N, 4)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

            pv_in = pde.apply(p, in_b)                         # freestream inlet
            in_l  = (jnp.mean((pv_in[:, 0] - rho_inf) ** 2)
                     + jnp.mean((pv_in[:, 1] - u_inf) ** 2)
                     + jnp.mean((pv_in[:, 2] - v_inf) ** 2)
                     + jnp.mean((pv_in[:, 3] - T_inf) ** 2))

            pv_w  = pde.apply(p, w_b)                          # no-slip + isothermal
            wall_l = (jnp.mean(pv_w[:, 1] ** 2)                # u = 0
                      + jnp.mean(pv_w[:, 2] ** 2)              # v = 0
                      + jnp.mean((pv_w[:, 3] - T0) ** 2))      # T = T0

            pv_s  = pde.apply(p, slip_b)                       # slip wall: v = 0 only
            slip_l = jnp.mean(pv_s[:, 2] ** 2)

            pv_up = pde.apply(p, up_b)                         # freestream farfield
            up_l  = (jnp.mean((pv_up[:, 0] - rho_inf) ** 2)
                     + jnp.mean((pv_up[:, 1] - u_inf) ** 2)
                     + jnp.mean((pv_up[:, 2] - v_inf) ** 2)
                     + jnp.mean((pv_up[:, 3] - T_inf) ** 2))

            total = (W_PDE * pde_l + W_INLET * in_l + W_WALL * wall_l
                     + W_SLIP * slip_l + W_UPPER * up_l)
            return total, (pde_l, in_l, wall_l, slip_l, up_l)

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
    N_uni, N_blp, N_adp = xy_uniform.shape[0], xy_bl.shape[0], xy_adapt.shape[0]
    N_in, N_w, N_up = xy_in.shape[0], xy_w.shape[0], xy_up.shape[0]
    N_slip = xy_slip.shape[0]
    # split the PDE batch:  adaptive (shock) + BL (wall) + uniform (rest)
    batch_adapt = max(1, min(int(round(adapt_frac * batch_r)), N_adp))
    batch_bl    = max(1, min(int(round(bl_frac    * batch_r)), N_blp))
    batch_uni   = max(1, min(batch_r - batch_adapt - batch_bl, N_uni))
    key = jax.random.PRNGKey(seed + 7)

    # Restore RNG key + (RAR-resampled) ADAPTIVE pool for a clean resume.
    # The uniform and wall-clustered pools are deterministic ⇒ re-derived.
    saved_state = restart.restore_arrays()
    if "key" in saved_state:
        key = jnp.asarray(saved_state["key"], dtype=jnp.uint32)
    if "xy_adapt" in saved_state:
        xy_adapt = jnp.asarray(saved_state["xy_adapt"])
        N_adp = xy_adapt.shape[0]

    try:
        for ep in range(start_ep, epochs):
            # RAR: migrate ONLY the adaptive pool toward high-residual regions
            # (mainly the ramp shock / SBLI); uniform + BL pools stay fixed.
            if rar_period > 0 and ep > 0 and ep % rar_period == 0:
                xy_adapt = jnp.array(rad_resample(
                    pde, params, geom.sample_interior,
                    n_keep=n_adapt, n_candidates=rar_cand * n_adapt,
                    k=rar_k, c=rar_c, seed=seed + ep))
                N_adp = xy_adapt.shape[0]
                print(f"  [ep {ep:5d}] RAR resampled ADAPTIVE pool ({n_adapt} pts) "
                      f"→ shock / SBLI")

            key, k1u, k1b, k1a, k2, k3, k4, k5 = jax.random.split(key, 8)
            iru = safe_choice(k1u, N_uni,  batch_uni)
            irb = safe_choice(k1b, N_blp,  batch_bl)
            ira = safe_choice(k1a, N_adp,  batch_adapt)
            iin = safe_choice(k2,  N_in,   min(batch_bc, N_in))
            iw  = safe_choice(k3,  N_w,    min(batch_bc, N_w))
            isl = safe_choice(k4,  N_slip, min(batch_bc, N_slip))
            iu  = safe_choice(k5,  N_up,   min(batch_bc, N_up))

            params, opt_state, total, (pl, il, wl, sl, ul) = step(
                params, opt_state, xy_uniform[iru], xy_bl[irb], xy_adapt[ira],
                xy_in[iin], xy_w[iw], xy_slip[isl], xy_up[iu])

            loss_hist.append(float(total))
            logger.on_epoch_end(ep, {"loss": float(total), "pde": float(pl),
                                     "inlet": float(il), "wall": float(wl),
                                     "slip": float(sl), "upper": float(ul)})
            restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist},
                               arrays={"xy_adapt": np.asarray(xy_adapt),
                                       "key": np.asarray(key)})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Evaluate on a grid + plots ────────────────────────────────────────────
    XX, YY, mask = geom.make_grid(Nx=220, Ny=160)
    pts  = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32))
    prim = np.array(pde.apply(params, pts))
    pres = np.array(pde.pressure(params, pts))
    mach = np.array(pde.mach(params, pts))
    Ny, Nx = XX.shape

    def _grid(a):
        out = a.reshape(Ny, Nx).copy()
        out[~mask] = np.nan
        return out

    rho_g, u_g, v_g, T_g = (_grid(prim[:, i]) for i in range(4))
    p_g, mach_g = _grid(pres), _grid(mach)

    fields = [(rho_g, "ρ"), (u_g, "u"), (v_g, "v"),
              (T_g, "T"), (p_g, "p"), (mach_g, "Mach")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for ax, (fld, name) in zip(axes.ravel(), fields):
        cf = ax.contourf(XX, YY, fld, levels=80, cmap="turbo")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        ax.plot([0, ramp_start, L],
                [0, 0, (L - ramp_start) * np.tan(np.radians(theta_deg))],
                "k-", lw=1.5)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle(f"Compression-ramp NS — M={M_inf}, Re={Re:g}, θ={theta_deg}°, "
                 f"T_wall=T0={T0:.2f}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Collocation map: uniform base + wall-clustered (BL) + adaptive (shock) ─
    xu, xb, xa = np.asarray(xy_uniform), np.asarray(xy_bl), np.asarray(xy_adapt)
    figc, axc = plt.subplots(figsize=(11, 4.2))
    axc.scatter(xu[:, 0], xu[:, 1], s=2, c="#b8c4d0", alpha=0.5,
                label=f"uniform ({len(xu)})", rasterized=True)
    axc.scatter(xb[:, 0], xb[:, 1], s=3, c="#1d9e75", alpha=0.6,
                label=f"wall-clustered / BL ({len(xb)})", rasterized=True)
    axc.scatter(xa[:, 0], xa[:, 1], s=4, c="#d1495b", alpha=0.7,
                label=f"adaptive / shock ({len(xa)})", rasterized=True)
    axc.plot([0, ramp_start, L],
             [0, 0, (L - ramp_start) * np.tan(np.radians(theta_deg))],
             "k-", lw=1.6)
    axc.set_xlim(0, L)
    axc.set_ylim(0, H)
    axc.set_aspect("equal")
    axc.set_xlabel("x")
    axc.set_ylabel("y")
    axc.legend(loc="upper left", framealpha=0.9, markerscale=2)
    axc.set_title("Interior collocation — uniform + wall-clustered (BL) + "
                  "residual-adaptive (shock)", fontweight="bold")
    figc.tight_layout()
    figc.savefig(os.path.join(out_dir, "collocation.png"), dpi=200,
                 bbox_inches="tight")
    plt.close(figc)

    fig2, ax2 = plt.subplots(figsize=(8, 3.4))
    ax2.semilogy(loss_hist, lw=1.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total loss")
    ax2.set_title("Training loss")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "loss.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    save_predictions(
        out_dir,
        coords  = {"x": XX.ravel(), "y": YY.ravel()},
        outputs = {"rho": prim[:, 0], "u": prim[:, 1], "v": prim[:, 2],
                   "T": prim[:, 3], "p": pres, "mach": mach},
        filename="predictions.npz",
    )
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    save_checkpoint(params, out_dir, metadata={
        "problem": "ramp_ns",
        "network": net_cfg,
        "physics": {"gamma": gamma, "M_inf": M_inf, "Re": Re, "Pr": Pr,
                    "theta_deg": theta_deg, "T0": T0, "mu_law": mu_law,
                    "art_visc": art_visc, "L": L, "H": H,
                    "ramp_start": ramp_start, "slip_end": slip_end},
        "results": {"n_epochs": len(loss_hist)},
    })
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist, "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_ramp_ns(load_config(cfg_path))
