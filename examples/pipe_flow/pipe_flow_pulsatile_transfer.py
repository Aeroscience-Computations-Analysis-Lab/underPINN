"""3-D Unsteady Pulsatile Pipe Flow via Time-Marching Transfer Learning.

Run directly or via the CLI:

    python examples/pipe_flow/pipe_flow_pulsatile_transfer.py
    python examples/pipe_flow/pipe_flow_pulsatile_transfer.py myconfig.yaml
    python -m underPINN run examples/pipe_flow/pipe_flow_pulsatile_transfer.yaml

The geometry and inflow profile are identical to the steady pipe-flow case
(``examples/pipe_flow/pipe_flow.py``): a cylinder of radius R over
x ∈ [x_lo, x_lo + L] with a parabolic Poiseuille inlet of peak velocity V_max.
The flow is made **unsteady** by modulating the inlet peak sinusoidally:

    u_inlet(r, t) = (V_max + V_amp · sin(2π t / T_period)) · (1 − r²/R²)
    v = w = 0   at the inlet

The full 3-D unsteady incompressible Navier–Stokes system is solved:

    ∇·u = 0
    u_t + (u·∇)u = −∇p + ν Δu          (ν = 1/Re)

Boundary / initial conditions
-----------------------------
    • Inlet  (x = x_lo): pulsatile parabolic velocity (above)
    • Wall   (r = R)   : no-slip  u = v = w = 0
    • Outlet (x = x_hi): p = 0
    • t = 0            : steady Poiseuille profile  u = V_max(1−r²/R²)

Long-time integration — Time-Marching Transfer Learning
-------------------------------------------------------
Representing a long horizon with a single network is hard (spectral bias,
causality violation).  Instead the horizon [0, T_total] is split into K
windows of length dT.  Each window is trained on its own local time τ ∈ [0, dT]
(absolute time t = k·dT + τ) and:

  • **warm-starts** its network from the previous window's trained weights
    (transfer learning → each window converges in far fewer epochs), and
  • uses the previous window's **end-state** (at τ = dT) as its initial
    condition, so information propagates forward in time.

An optional no-transfer baseline (each window cold-started from scratch) is
run for comparison to quantify the transfer-learning speed-up.

Network: (x, y, z, t) → (u, v, w, p)
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
from underPINN.pde.navier_stokes_3d import UnsteadyNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# Per-window collocation data
# ---------------------------------------------------------------------------

def _window_data(pipe: Pipe, dT: float, d, seed: int):
    """Sample one window's collocation points with local time τ ∈ [0, dT].

    Returns a dict of jnp arrays:
        xyzt_r   (N_r, 4)   interior  PDE points
        xyzt_w   (N_w, 4)   wall      no-slip points
        xyzt_in  (N_in, 4)  inlet     points (+ time)
        xyzt_out (N_out, 4) outlet    points
        xyz_ic   (N_ic, 3)  interior  IC points (time appended at use)
    """
    rng = np.random.default_rng(seed)
    n_r   = int(cfg_get(d, "n_interior", default=5000))
    n_w   = int(cfg_get(d, "n_wall",     default=1500))
    n_in  = int(cfg_get(d, "n_inlet",    default=400))
    n_out = int(cfg_get(d, "n_outlet",   default=400))
    n_ic  = int(cfg_get(d, "n_ic",       default=1000))

    def _attach_t(xyz, n):
        t = rng.uniform(0.0, dT, size=(n, 1)).astype(np.float32)
        return np.concatenate([xyz, t], axis=1)

    xyz_r   = pipe.sample_interior(n_r,  seed=seed)
    xyz_w   = pipe.sample_wall(    n_w,  seed=seed + 1)
    xyz_in  = pipe.sample_inlet(   n_in, seed=seed + 2)
    xyz_out = pipe.sample_outlet(  n_out, seed=seed + 3)
    xyz_ic  = pipe.sample_interior(n_ic, seed=seed + 4)

    return {
        "xyzt_r":   jnp.array(_attach_t(xyz_r,   len(xyz_r))),
        "xyzt_w":   jnp.array(_attach_t(xyz_w,   len(xyz_w))),
        "xyzt_in":  jnp.array(_attach_t(xyz_in,  len(xyz_in))),
        "xyzt_out": jnp.array(_attach_t(xyz_out, len(xyz_out))),
        "xyz_ic":   jnp.array(xyz_ic),
    }


# ---------------------------------------------------------------------------
# Single-window training
# ---------------------------------------------------------------------------

def _train_window(
    pde, model, data, *, R, V_max, V_amp, T_period, t0, dT,
    ic_uvw, epochs, lr, lr_alpha, weights, batch_r, batch_bc,
    init_params, seed, label,
):
    """Train one time window τ ∈ [0, dT] (absolute t = t0 + τ).

    ``ic_uvw`` : (N_ic, 3) target (u, v, w) at τ = 0 (the window's initial
    condition — either the steady profile or the previous window's end-state).
    Warm-starts from *init_params* when provided.
    """
    w_pde, w_wall, w_inlet, w_outlet, w_ic = weights
    omega = 2.0 * np.pi / T_period

    xyzt_r   = data["xyzt_r"]
    xyzt_w   = data["xyzt_w"]
    xyzt_in  = data["xyzt_in"]
    xyzt_out = data["xyzt_out"]
    xyz_ic   = data["xyz_ic"]
    # IC points at τ = 0
    xyzt_ic0 = jnp.concatenate([xyz_ic, jnp.zeros((xyz_ic.shape[0], 1))], axis=1)
    ic_u, ic_v, ic_w = ic_uvw[:, 0], ic_uvw[:, 1], ic_uvw[:, 2]

    key    = jax.random.PRNGKey(seed)
    params = (model.init(key, jnp.ones((1, 4)))
              if init_params is None else init_params)

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=max(epochs, 1),
                                            alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, state, r_b, w_b, in_b, out_b):
        def loss_fn(p):
            # PDE residual
            res   = pde.residual(p, r_b)                       # (N, 4)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

            # Wall no-slip
            out_w  = model.apply(p, w_b)
            wall_l = jnp.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)

            # Inlet: pulsatile parabolic profile (same shape as steady case)
            out_in = model.apply(p, in_b)
            r2     = in_b[:, 1] ** 2 + in_b[:, 2] ** 2
            shape  = 1.0 - r2 / R ** 2
            t_abs  = t0 + in_b[:, 3]
            u_tgt  = (V_max + V_amp * jnp.sin(omega * t_abs)) * shape
            in_l   = (jnp.mean((out_in[:, 0] - u_tgt) ** 2)
                      + jnp.mean(out_in[:, 1] ** 2)
                      + jnp.mean(out_in[:, 2] ** 2))

            # Outlet: p = 0
            out_out  = model.apply(p, out_b)
            outlet_l = jnp.mean(out_out[:, 3] ** 2)

            # Initial condition at τ = 0  (window-to-window continuity)
            out_ic = model.apply(p, xyzt_ic0)
            ic_l   = jnp.mean((out_ic[:, 0] - ic_u) ** 2
                              + (out_ic[:, 1] - ic_v) ** 2
                              + (out_ic[:, 2] - ic_w) ** 2)

            total = (w_pde * pde_l + w_wall * wall_l + w_inlet * in_l
                     + w_outlet * outlet_l + w_ic * ic_l)
            return total, (pde_l, wall_l, in_l, outlet_l, ic_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    N_r, N_w   = xyzt_r.shape[0], xyzt_w.shape[0]
    N_in, N_out = xyzt_in.shape[0], xyzt_out.shape[0]
    key = jax.random.PRNGKey(seed + 99)
    loss_hist = []

    for ep in range(epochs):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir  = safe_choice(k1, N_r,   batch_r)
        iw  = safe_choice(k2, N_w,   batch_bc)
        iin = safe_choice(k3, N_in,  min(batch_bc, N_in))
        iout = safe_choice(k4, N_out, min(batch_bc, N_out))

        params, opt_state, total, (pl, wl, il, ol, icl) = step(
            params, opt_state,
            xyzt_r[ir], xyzt_w[iw], xyzt_in[iin], xyzt_out[iout])
        loss_hist.append(float(total))

        if ep % max(epochs // 5, 1) == 0 or ep == epochs - 1:
            print(f"    [{label}] ep {ep:5d} | total {total:.3e} "
                  f"| pde {pl:.2e} | wall {wl:.2e} | in {il:.2e} "
                  f"| out {ol:.2e} | ic {icl:.2e}")

    return params, loss_hist


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_uvw(model, params, xyz, tau):
    """(u, v, w) at spatial points xyz (N,3) and scalar local time tau."""
    xyzt = jnp.concatenate([xyz, jnp.full((xyz.shape[0], 1), tau)], axis=1)
    return np.array(model.apply(params, xyzt)[:, :3])


def _centreline_u(model, params, x_val, tau):
    """Axial velocity u at the pipe centreline point (x_val, 0, 0), time tau."""
    xyzt = jnp.array([[x_val, 0.0, 0.0, tau]], dtype=jnp.float32)
    return float(model.apply(params, xyzt)[0, 0])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pipe_flow_pulsatile_transfer(cfg) -> dict:
    """Time-marching transfer-learning PINN for 3-D pulsatile pipe flow."""
    ph  = cfg.physics
    d   = cfg.data
    lw  = cfg.loss
    tm  = cfg_get(cfg, "time_marching", default=None)
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/pipe_flow_pulsatile_transfer")
               if out else "outputs/pipe_flow_pulsatile_transfer")
    os.makedirs(out_dir, exist_ok=True)

    # ── Physics (same dimensions / profile as the steady pipe case) ───────────
    Re       = float(cfg_get(ph, "Re",       default=40.0))
    R        = float(cfg_get(ph, "R",        default=0.5))
    L        = float(cfg_get(ph, "L",        default=7.0))
    x_lo     = float(cfg_get(ph, "x_lo",     default=-3.5))
    V_max    = float(cfg_get(ph, "V_max",    default=2.0))
    V_amp    = float(cfg_get(ph, "V_amp",    default=1.0))
    T_period = float(cfg_get(ph, "T_period", default=1.0))
    x_hi     = x_lo + L
    x_mid    = 0.5 * (x_lo + x_hi)
    omega    = 2.0 * np.pi / T_period

    # ── Time-marching parameters ──────────────────────────────────────────────
    T_total      = float(cfg_get(tm, "T_total",        default=4.0))
    dT           = float(cfg_get(tm, "dT",             default=0.5))
    n_first      = int(cfg_get(tm,   "n_first_epochs", default=8000))
    n_warm       = int(cfg_get(tm,   "n_warm_epochs",  default=3000))
    n_cold       = int(cfg_get(tm,   "n_cold_epochs",  default=8000))
    cmp_cold     = bool(cfg_get(tm,  "compare_no_transfer", default=True))
    first_lr     = float(cfg_get(tm, "first_lr",       default=1e-3))
    warm_lr      = float(cfg_get(tm, "warm_lr",        default=5e-4))
    cold_lr      = float(cfg_get(tm, "cold_lr",        default=1e-3))
    lr_alpha     = float(cfg_get(tm, "lr_alpha",       default=0.01))
    n_windows    = max(1, int(round(T_total / dT)))

    batch_r  = int(cfg_get(d, "batch_r",  default=2048))
    batch_bc = int(cfg_get(d, "batch_bc", default=512))
    weights  = (float(cfg_get(lw, "w_pde",    default=1.0)),
                float(cfg_get(lw, "w_wall",   default=100.0)),
                float(cfg_get(lw, "w_inlet",  default=50.0)),
                float(cfg_get(lw, "w_outlet", default=20.0)),
                float(cfg_get(lw, "w_ic",     default=100.0)))

    seed = int(cfg_get(tm, "seed", default=0))

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_type = str(cfg_get(cfg.network, "type", default="gated_mlp")).lower()
    _net_cls = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net_type, MLP)
    layers   = list(cfg.network.layers)
    model    = _net_cls(layers=layers)
    pde      = UnsteadyNS3DPDE(model, Re=Re)
    pipe     = Pipe(R=R, L=L, x_lo=x_lo)

    print(f"Pulsatile pipe flow (3-D unsteady):  Re={Re},  R={R} (D={2*R}),  "
          f"x ∈ [{x_lo}, {x_hi}]")
    print(f"  Inlet peak(t) = {V_max} + {V_amp}·sin(2π t/{T_period})  "
          f"(parabolic profile)")
    print(f"  Network: {_net_cls.__name__}  layers={layers}")
    print(f"  Time-marching: T_total={T_total}, dT={dT} → {n_windows} windows")
    print(f"  Epochs: first={n_first}, warm={n_warm}"
          + (f", cold(baseline)={n_cold}" if cmp_cold else ""))

    # ── Steady Poiseuille initial condition (t = 0) ───────────────────────────
    def _steady_uvw(xyz):
        r2 = np.asarray(xyz[:, 1]) ** 2 + np.asarray(xyz[:, 2]) ** 2
        u  = V_max * (1.0 - r2 / R ** 2)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)], axis=1).astype(np.float32)

    # =====================================================================
    #  Transfer chain  (warm-started, time-marching)
    # =====================================================================
    print("\n" + "=" * 60)
    print("Transfer chain — warm-started time marching")
    print("=" * 60)

    tf_params_list = []
    tf_loss_all    = []
    prev_params    = None
    for k in range(n_windows):
        t0   = k * dT
        data = _window_data(pipe, dT, d, seed=1000 + k)
        # Initial condition for this window
        if k == 0:
            ic_uvw = jnp.array(_steady_uvw(np.array(data["xyz_ic"])))
        else:
            ic_uvw = jnp.array(_eval_uvw(model, prev_params,
                                         data["xyz_ic"], tau=dT))
        epochs = n_first if k == 0 else n_warm
        lr     = first_lr if k == 0 else warm_lr
        print(f"\n  Window {k+1}/{n_windows}  t ∈ [{t0:.2f}, {t0+dT:.2f}]"
              f"  ({'cold' if k == 0 else 'warm'}, {epochs} ep)")
        params, lh = _train_window(
            pde, model, data,
            R=R, V_max=V_max, V_amp=V_amp, T_period=T_period, t0=t0, dT=dT,
            ic_uvw=ic_uvw, epochs=epochs, lr=lr, lr_alpha=lr_alpha,
            weights=weights, batch_r=batch_r, batch_bc=batch_bc,
            init_params=prev_params, seed=seed + k, label=f"tf-w{k+1}")
        tf_params_list.append(params)
        tf_loss_all.append(lh)
        prev_params = params

    # =====================================================================
    #  No-transfer baseline  (each window cold-started)
    # =====================================================================
    cold_loss_all = []
    if cmp_cold:
        print("\n" + "=" * 60)
        print("Baseline chain — no transfer (cold start each window)")
        print("=" * 60)
        prev_cold = None
        for k in range(n_windows):
            t0   = k * dT
            data = _window_data(pipe, dT, d, seed=2000 + k)
            if k == 0:
                ic_uvw = jnp.array(_steady_uvw(np.array(data["xyz_ic"])))
            else:
                ic_uvw = jnp.array(_eval_uvw(model, prev_cold,
                                             data["xyz_ic"], tau=dT))
            print(f"\n  Window {k+1}/{n_windows}  (cold, {n_cold} ep)")
            params_c, lh_c = _train_window(
                pde, model, data,
                R=R, V_max=V_max, V_amp=V_amp, T_period=T_period, t0=t0, dT=dT,
                ic_uvw=ic_uvw, epochs=n_cold, lr=cold_lr, lr_alpha=lr_alpha,
                weights=weights, batch_r=batch_r, batch_bc=batch_bc,
                init_params=None, seed=seed + 500 + k, label=f"cold-w{k+1}")
            cold_loss_all.append(lh_c)
            prev_cold = params_c

    # =====================================================================
    #  Full-horizon prediction stitched from the transfer windows
    # =====================================================================
    def predict_uvwp(t_abs, xyz):
        """(u,v,w,p) at absolute time t_abs using the owning window's params."""
        k   = min(int(t_abs // dT), n_windows - 1)
        tau = t_abs - k * dT
        xyzt = jnp.concatenate(
            [xyz, jnp.full((xyz.shape[0], 1), float(tau))], axis=1)
        return np.array(model.apply(tf_params_list[k], xyzt))

    # ── Centreline velocity over the whole horizon ────────────────────────────
    t_line = np.linspace(0.0, T_total, 400, dtype=np.float32)
    u_ctr  = np.array([
        _centreline_u(model, tf_params_list[min(int(t // dT), n_windows - 1)],
                      x_mid, float(t - min(int(t // dT), n_windows - 1) * dT))
        for t in t_line])
    inlet_peak = V_max + V_amp * np.sin(omega * t_line)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_line, inlet_peak, "k--", lw=1.5, label="Inlet peak forcing")
    ax.plot(t_line, u_ctr, "b-", lw=1.8, label=f"PINN centreline u @ x={x_mid:.1f}")
    for k in range(1, n_windows):
        ax.axvline(k * dT, color="grey", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("u (centreline)")
    ax.set_title(f"Pulsatile pipe — centreline velocity over {n_windows} windows "
                 f"(Re={Re})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "centreline_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Stitched loss history (transfer chain) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    offset = 0
    for k, lh in enumerate(tf_loss_all):
        xs = np.arange(offset, offset + len(lh))
        ax.semilogy(xs, lh, lw=1.0, label=f"window {k+1}" if k < 8 else None)
        offset += len(lh)
        ax.axvline(offset, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Cumulative epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Transfer chain — per-window loss (warm starts converge fast)")
    ax.grid(alpha=0.3, which="both")
    if n_windows <= 8:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transfer_loss_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Transfer vs no-transfer: per-window cost & final loss ─────────────────
    if cmp_cold and cold_loss_all:
        wins      = np.arange(1, n_windows + 1)
        tf_final  = [lh[-1] for lh in tf_loss_all]
        cold_final = [lh[-1] for lh in cold_loss_all]
        tf_epochs  = [len(lh) for lh in tf_loss_all]
        cold_epochs = [len(lh) for lh in cold_loss_all]

        fig, (axl, axe) = plt.subplots(1, 2, figsize=(13, 4))
        width = 0.38
        axl.bar(wins - width / 2, tf_final,   width, label="Transfer (warm)",
                color="#0072B2")
        axl.bar(wins + width / 2, cold_final, width, label="No transfer (cold)",
                color="#D55E00")
        axl.set_yscale("log")
        axl.set_xlabel("Window")
        axl.set_ylabel("Final loss")
        axl.set_title("Final loss per window")
        axl.set_xticks(wins)
        axl.legend(fontsize=9)
        axl.grid(axis="y", alpha=0.3, which="both")

        axe.bar(wins - width / 2, tf_epochs,   width, label="Transfer (warm)",
                color="#0072B2")
        axe.bar(wins + width / 2, cold_epochs, width, label="No transfer (cold)",
                color="#D55E00")
        axe.set_xlabel("Window")
        axe.set_ylabel("Training epochs")
        axe.set_title(f"Epochs per window  "
                      f"(transfer total {sum(tf_epochs):,} vs "
                      f"cold {sum(cold_epochs):,})")
        axe.set_xticks(wins)
        axe.legend(fontsize=9)
        axe.grid(axis="y", alpha=0.3)
        fig.suptitle("Time-marching: transfer learning vs cold start")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "transfer_vs_cold.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Radial profiles at phase snapshots across the horizon ─────────────────
    Nr    = 80
    r_arr = np.linspace(0.0, R, Nr, dtype=np.float32)
    xyz_r = np.stack([np.full(Nr, x_mid, np.float32), r_arr,
                      np.zeros(Nr, np.float32)], axis=1)
    snaps = np.linspace(0.0, T_total, 4, endpoint=False) + 0.25 * dT
    fig, axes = plt.subplots(1, len(snaps), figsize=(15, 4))
    for ax, ts in zip(np.atleast_1d(axes), snaps):
        uvwp = predict_uvwp(float(ts), jnp.array(xyz_r))
        peak = V_max + V_amp * np.sin(omega * ts)
        u_qs = peak * (1.0 - r_arr ** 2 / R ** 2)   # quasi-steady reference
        ax.plot(r_arr, u_qs, "k--", lw=1.3, label="Quasi-steady")
        ax.plot(r_arr, uvwp[:, 0], "b-", lw=1.8, label="PINN")
        ax.set_title(f"t = {ts:.2f}")
        ax.set_xlabel("r")
        ax.set_ylabel("u")
        ax.grid(alpha=0.3)
        if ts == snaps[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"Radial velocity profiles at x={x_mid:.1f}  (Re={Re})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radial_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save final-window checkpoint + config ─────────────────────────────────
    save_checkpoint(tf_params_list[-1], out_dir, stem="params_final_window",
                    metadata={
                        "problem": "pipe_flow_pulsatile_transfer",
                        "method":  "time_marching_transfer",
                        "network": {"type": net_type, "layers": layers},
                        "physics": {"Re": Re, "R": R, "L": L, "x_lo": x_lo,
                                    "V_max": V_max, "V_amp": V_amp,
                                    "T_period": T_period},
                        "time_marching": {"T_total": T_total, "dT": dT,
                                          "n_windows": n_windows},
                    })
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    flat_loss = [v for lh in tf_loss_all for v in lh]
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(flat_loss))
    print(f"\nAll outputs saved to: {out_dir}/")

    result = {
        "params": tf_params_list[-1],
        "params_per_window": tf_params_list,
        "loss_hist": flat_loss,
        "n_windows": n_windows,
        "transfer_window_final": [lh[-1] for lh in tf_loss_all],
    }
    if cmp_cold and cold_loss_all:
        result["cold_window_final"] = [lh[-1] for lh in cold_loss_all]
    return result


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else _HERE / "pipe_flow_pulsatile_transfer.yaml"
    )
    from underPINN.config.loader import load_config
    run_pipe_flow_pulsatile_transfer(load_config(cfg_path))
