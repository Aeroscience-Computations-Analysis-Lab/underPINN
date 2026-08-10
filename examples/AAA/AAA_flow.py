"""3-D Steady Axisymmetric Bulge (AAA-like) PINN.

Run directly or via the CLI:

    python examples/AAA/AAA_flow.py           # uses config.yaml
    python examples/AAA/AAA_flow.py myconfig.yaml
    python -m underPINN run examples/AAA/config.yaml

Solves steady 3-D incompressible NS inside an axisymmetric vessel with a
local outward bulge.  The vessel cross-section varies along x as

    R(x) = R_vessel + (R_AAA − R_vessel) · cos(π/2 · |x−x0|/half_bulge)²

A parabolic (Poiseuille-like) inflow is imposed on the inlet disk; no-slip on
the curved wall; zero-pressure on the outlet disk.  There is no known exact
analytical solution, so accuracy is assessed by a flow-rate balance.

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
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.aaa import BulgeGeometry
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import safe_choice


def run_AAA_flow(cfg) -> dict:
    """Train a PINN on 3-D steady flow through an axisymmetric bulge."""
    # ── Unpack ─────────────────────────────────────────────────────────────────
    ph   = cfg.physics
    tr   = cfg.training
    lw   = cfg.loss
    seed = cfg_get(tr,  "seed",   default=0)
    out  = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/AAA_flow") if out else "outputs/AAA_flow"
    os.makedirs(out_dir, exist_ok=True)

    Re         = float(ph.Re)
    R_vessel   = float(ph.R_vessel)
    R_AAA = float(ph.R_AAA)
    L          = float(ph.L)
    x_lo       = float(cfg_get(ph, "x_lo",       default=-3.5))
    x0         = float(cfg_get(ph, "x0",          default=1.5))
    L_AAA = float(cfg_get(ph, "L_AAA",  default=1.5))
    V_max      = float(ph.V_max)
    x_hi       = x_lo + L

    W_PDE    = cfg_get(lw, "w_pde",    default=1.0)
    W_WALL   = cfg_get(lw, "w_wall",   default=100.0)
    W_INLET  = cfg_get(lw, "w_inlet",  default=50.0)
    W_OUTLET = cfg_get(lw, "w_outlet", default=20.0)

    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    batch_r   = cfg_get(tr, "batch_r",   default=256)
    batch_bc  = cfg_get(tr, "batch_bc",  default=128)

    print(f"AAA flow:  Re={Re},  R_vessel={R_vessel},  R_AAA={R_AAA}")
    print(f"  x ∈ [{x_lo}, {x_hi}],  bulge centre x0={x0},  L_AAA={L_AAA}")
    print(f"  V_max={V_max}")

    # ── Geometry + collocation data ────────────────────────────────────────────
    geom = BulgeGeometry(
        R_vessel=R_vessel, R_AAA=R_AAA,
        L=L, x_lo=x_lo, x0=x0, L_AAA=L_AAA,
    )
    d = cfg.data
    xyz_r   = jnp.array(geom.sample_interior(cfg_get(d, "n_interior", default=5000), seed=seed))
    xyz_w   = jnp.array(geom.sample_wall(    cfg_get(d, "n_wall",     default=1500), seed=seed+1))
    xyz_in  = jnp.array(geom.sample_inlet(   cfg_get(d, "n_inlet",    default=400),  seed=seed+2))
    xyz_out = jnp.array(geom.sample_outlet(  cfg_get(d, "n_outlet",   default=400),  seed=seed+3))

    def inlet_velocity(xyz):
        r2 = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
        return V_max * (1.0 - r2 / R_vessel ** 2)

    # ── Model + PDE ────────────────────────────────────────────────────────────
    net_cfg  = network_config(cfg)
    net_type = net_cfg["type"]
    model    = build_model(net_cfg)
    print(f"  Network: {type(model).__name__} ({net_type})  layers={list(cfg.network.layers)}")

    pde    = SteadyNS3DPDE(model, Re=Re)
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 3)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # ── JIT step ───────────────────────────────────────────────────────────────
    @jax.jit
    def step(params, state, xyz_r, xyz_w, xyz_in, xyz_out):
        def loss_fn(p):
            _res   = pde.residual(p, xyz_r)           # (N, 4)
            pde_l  = jnp.mean(jnp.sum(_res ** 2, axis=-1))

            out_w  = model.apply(p, xyz_w)
            wall_l = jnp.mean(out_w[:, 0]**2 + out_w[:, 1]**2 + out_w[:, 2]**2)

            out_in     = model.apply(p, xyz_in)
            u_in_exact = inlet_velocity(xyz_in)
            in_l       = (jnp.mean((out_in[:, 0] - u_in_exact)**2)
                          + jnp.mean(out_in[:, 1]**2)
                          + jnp.mean(out_in[:, 2]**2))

            out_out  = model.apply(p, xyz_out)
            outlet_l = jnp.mean(out_out[:, 3]**2)

            total = (W_PDE * pde_l + W_WALL * wall_l
                     + W_INLET * in_l + W_OUTLET * outlet_l)
            return total, (pde_l, wall_l, in_l, outlet_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Restart ────────────────────────────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart  = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    # ── Training loop ──────────────────────────────────────────────────────────
    N_r  = xyz_r.shape[0]
    N_w  = xyz_w.shape[0]
    N_in  = xyz_in.shape[0]
    N_out = xyz_out.shape[0]

    logger = ConsoleLogger(log_every=log_every)
    key    = jax.random.PRNGKey(seed + 99)

    try:
        for ep in range(start_ep, epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir   = safe_choice(k1, N_r,   batch_r)
            iw   = safe_choice(k2, N_w,   batch_bc)
            iin  = safe_choice(k3, N_in,  min(batch_bc, N_in))
            iout = safe_choice(k4, N_out, min(batch_bc, N_out))

            params, opt_state, total, (pl, wl, il, ol) = step(
                params, opt_state,
                xyz_r[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
            loss_hist.append(float(total))

            logs = {"loss": float(total), "pde": float(pl),
                    "wall": float(wl), "inlet": float(il)}
            logger.on_epoch_end(ep, logs)
            restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Save ───────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # Save predictions at interior collocation points
    uvwp_pred = model.apply(params, xyz_r)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xyz_r[:, 0]),
                   "y": np.array(xyz_r[:, 1]),
                   "z": np.array(xyz_r[:, 2])},
        outputs = {"u_pred": np.array(uvwp_pred[:, 0]),
                   "v_pred": np.array(uvwp_pred[:, 1]),
                   "w_pred": np.array(uvwp_pred[:, 2]),
                   "p_pred": np.array(uvwp_pred[:, 3])},
    )

    # ── Flow-rate balance (continuity sanity check) ────────────────────────────
    # Approximate volumetric flow rate at inlet and outlet via MC integration
    xyz_in_val  = jnp.array(geom.sample_inlet( 2000, seed=77))
    xyz_out_val = jnp.array(geom.sample_outlet(2000, seed=78))
    u_in_val    = model.apply(params, xyz_in_val)[:, 0]
    u_out_val   = model.apply(params, xyz_out_val)[:, 0]
    A_vessel    = np.pi * R_vessel ** 2
    Q_in  = float(jnp.mean(u_in_val))  * A_vessel
    Q_out = float(jnp.mean(u_out_val)) * A_vessel
    flow_balance = abs(Q_in - Q_out) / (abs(Q_in) + 1e-10)
    print(f"\nFlow-rate balance:  Q_in={Q_in:.4f}  Q_out={Q_out:.4f}"
          f"  rel_diff={flow_balance:.4f}")

    # ── Solution visualisation ─────────────────────────────────────────────────
    # 1. Cross-section contourf of u(y,z) at x = x0  (bulge centre)
    N_cs  = 80
    R_cs  = R_AAA        # plot covers the maximum cross-section
    y_cs  = np.linspace(-R_cs, R_cs, N_cs, dtype=np.float32)
    z_cs  = np.linspace(-R_cs, R_cs, N_cs, dtype=np.float32)
    YY_cs, ZZ_cs = np.meshgrid(y_cs, z_cs)
    x_cs   = np.full(N_cs * N_cs, float(x0), dtype=np.float32)
    xyz_cs = jnp.array(np.stack([x_cs, YY_cs.ravel(), ZZ_cs.ravel()], axis=1))

    pred_cs = np.array(model.apply(params, xyz_cs))
    u_cs    = pred_cs[:, 0].reshape(N_cs, N_cs)

    # Mask outside the local cross-section radius at x0
    R_x0   = float(geom.radius_at(np.array([x0]))[0])
    outside = YY_cs ** 2 + ZZ_cs ** 2 > R_x0 ** 2
    u_cs    = np.where(outside, np.nan, u_cs)

    fig_s, ax_s = plt.subplots(figsize=(5, 4))
    cf = ax_s.contourf(y_cs, z_cs, u_cs, levels=50, cmap="jet")
    plt.colorbar(cf, ax=ax_s, label="u")
    ax_s.set_title(f"Axial velocity u at x={x0:.2f}  (bulge centre, Re={Re})")
    ax_s.set_xlabel("y")
    ax_s.set_ylabel("z")
    ax_s.set_aspect("equal")
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(out_dir, "solution_crosssection.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_s)

    # 2. Centreline axial velocity u(x, 0, 0) — highlights bulge effect
    N_ax  = 200
    x_ax  = np.linspace(x_lo, x_hi, N_ax, dtype=np.float32)
    xyz_ax = jnp.array(np.stack(
        [x_ax,
         np.zeros(N_ax, dtype=np.float32),
         np.zeros(N_ax, dtype=np.float32)], axis=1))
    u_centre = np.array(model.apply(params, xyz_ax)[:, 0])
    R_ax     = geom.radius_at(x_ax)

    fig_a, ax1 = plt.subplots(figsize=(8, 4))
    col_u = "#0072B2"
    col_r = "#D55E00"
    ax1.plot(x_ax, u_centre, color=col_u, lw=2.0, label="u (centreline)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u (centreline)", color=col_u)
    ax1.tick_params(axis="y", labelcolor=col_u)
    ax2 = ax1.twinx()
    ax2.plot(x_ax, R_ax, color=col_r, lw=1.5, ls="--", label="R(x)")
    ax2.fill_between(x_ax,  R_ax, R_vessel, alpha=0.12, color=col_r)
    ax2.fill_between(x_ax, -R_ax, -R_vessel, alpha=0.12, color=col_r)
    ax2.set_ylabel("R(x)", color=col_r)
    ax2.tick_params(axis="y", labelcolor=col_r)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")
    ax1.set_title(f"Centreline velocity vs bulge profile  Re={Re}")
    ax1.grid(True, alpha=0.3)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(out_dir, "solution_centreline.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_a)

    # 3. Loss plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(loss_hist, lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"AAA Flow  Re={Re}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    save_checkpoint(params, out_dir, metadata={
        "problem": "AAA_flow",
        "network": net_cfg,
        "physics": {"Re": Re, "R_vessel": R_vessel, "R_AAA": R_AAA,
                    "L": L, "x_lo": x_lo, "x0": x0, "L_AAA": L_AAA,
                    "V_max": V_max},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": params, "loss_hist": loss_hist,
            "flow_balance": flow_balance}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml"
    )
    from underPINN.config.loader import load_config
    run_AAA_flow(load_config(cfg_path))
