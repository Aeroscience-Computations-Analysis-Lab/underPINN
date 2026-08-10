"""Shared time-marching transfer-learning loop for pulsatile 3-D flow cases.

Used by:

* ``examples/pipe_flow/pipe_flow_pulsatile_transfer.py``     — Newtonian pipe
* ``examples/AAA/AAA_pulsatile_transfer.py``                 — Newtonian AAA
* ``examples/pipe_flow_rheology/pipe_flow_rheology_pulsatile.py`` — Carreau pipe
* ``examples/AAA_rheology/AAA_rheology_pulsatile.py``        — Carreau AAA

Each case provides a ``problem_spec`` dict with:

* ``problem``                — name (string, matches the dispatch key)
* ``geom``                   — geometry object with ``sample_{interior,wall,inlet,outlet}``
* ``pde``                    — PDE object whose ``residual(params, xyzt)`` returns (N, 4)
* ``inlet_target_fn(xyz, t_abs)`` — jnp callable → axial u target at inlet pts
* ``steady_uvw_fn(xyz)``     — jnp callable → (u, v, w) at t=0 (the IC)
* ``physics_dict``           — physics record to embed in saved metadata
* ``label``                  — human-readable title for prints/plots

The helper handles: window data sampling (with overlapping stride), the train
loop, window-level restart, per-window checkpoint saving, the time→window
index, plots (centreline + radial profiles + transfer/cold comparison) and
prediction stitching.  The example scripts stay short and case-specific.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.factory import build_model, network_config
from underPINN.utils.checkpoint import load_checkpoint, read_metadata, save_checkpoint
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# Per-window collocation data
# ---------------------------------------------------------------------------

def _window_data(geom, dT: float, d, seed: int):
    rng = np.random.default_rng(seed)
    n_r   = int(cfg_get(d, "n_interior", default=5000))
    n_w   = int(cfg_get(d, "n_wall",     default=1500))
    n_in  = int(cfg_get(d, "n_inlet",    default=400))
    n_out = int(cfg_get(d, "n_outlet",   default=400))
    n_ic  = int(cfg_get(d, "n_ic",       default=1000))

    def _attach_t(xyz, n):
        t = rng.uniform(0.0, dT, size=(n, 1)).astype(np.float32)
        return np.concatenate([xyz, t], axis=1)

    xyz_r   = geom.sample_interior(n_r,  seed=seed)
    xyz_w   = geom.sample_wall(    n_w,  seed=seed + 1)
    xyz_in  = geom.sample_inlet(   n_in, seed=seed + 2)
    xyz_out = geom.sample_outlet(  n_out, seed=seed + 3)
    xyz_ic  = geom.sample_interior(n_ic, seed=seed + 4)

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
    pde, model, data, *, inlet_target_fn, t0, dT,
    ic_uvw, epochs, lr, lr_alpha, weights, batch_r, batch_bc,
    init_params, seed, label,
):
    w_pde, w_wall, w_inlet, w_outlet, w_ic = weights

    xyzt_r   = data["xyzt_r"]
    xyzt_w   = data["xyzt_w"]
    xyzt_in  = data["xyzt_in"]
    xyzt_out = data["xyzt_out"]
    xyz_ic   = data["xyz_ic"]
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
            res   = pde.residual(p, r_b)
            pde_l = jnp.mean(jnp.sum(res ** 2, axis=-1))

            out_w  = model.apply(p, w_b)
            wall_l = jnp.mean(out_w[:, 0] ** 2 + out_w[:, 1] ** 2 + out_w[:, 2] ** 2)

            out_in = model.apply(p, in_b)
            u_tgt  = inlet_target_fn(in_b[:, :3], t0 + in_b[:, 3])
            in_l   = (jnp.mean((out_in[:, 0] - u_tgt) ** 2)
                      + jnp.mean(out_in[:, 1] ** 2)
                      + jnp.mean(out_in[:, 2] ** 2))

            out_out  = model.apply(p, out_b)
            outlet_l = jnp.mean(out_out[:, 3] ** 2)

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


def _eval_uvw(model, params, xyz, tau):
    """Evaluate (u, v, w) at local time tau on xyz (N, 3)."""
    xyzt = jnp.concatenate([xyz, jnp.full((xyz.shape[0], 1), tau)], axis=1)
    return np.array(model.apply(params, xyzt)[:, :3])


def _centreline_u(model, params, x_val, tau):
    xyzt = jnp.array([[x_val, 0.0, 0.0, tau]], dtype=jnp.float32)
    return float(model.apply(params, xyzt)[0, 0])


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_pulsatile_time_march(
    cfg,
    *,
    problem_spec: dict,
    out_dir_default: str,
) -> dict:
    """Time-marching transfer-learning PINN for a pulsatile 3-D case.

    Parameters
    ----------
    cfg            : SimpleNamespace (the loaded YAML config).
    problem_spec   : per-case wiring (see module docstring).
    out_dir_default: fallback output directory when ``cfg.output.dir`` is absent.
    """
    d   = cfg.data
    lw  = cfg.loss
    tm  = cfg_get(cfg, "time_marching", default=None)
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default=out_dir_default)
               if out else out_dir_default)
    os.makedirs(out_dir, exist_ok=True)

    label          = problem_spec["label"]
    geom           = problem_spec["geom"]
    pde            = problem_spec["pde"]
    inlet_target   = problem_spec["inlet_target_fn"]
    steady_uvw_fn  = problem_spec["steady_uvw_fn"]
    physics_record = problem_spec["physics_dict"]
    problem_name   = problem_spec["problem"]
    plot_extent    = problem_spec.get("plot_extent")    # (x_lo, x_hi, R_for_centreline)

    # ── Time-marching parameters ─────────────────────────────────────────────
    T_total      = float(cfg_get(tm, "T_total",        default=4.0))
    dT           = float(cfg_get(tm, "dT",             default=0.5))
    stride       = float(cfg_get(tm, "stride",         default=dT))
    if stride <= 0.0 or stride > dT:
        raise ValueError(f"time_marching.stride must be in (0, dT]; got "
                         f"stride={stride}, dT={dT}.")
    n_first      = int(cfg_get(tm,   "n_first_epochs", default=8000))
    n_warm       = int(cfg_get(tm,   "n_warm_epochs",  default=3000))
    n_cold       = int(cfg_get(tm,   "n_cold_epochs",  default=8000))
    cmp_cold     = bool(cfg_get(tm,  "compare_no_transfer", default=True))
    first_lr     = float(cfg_get(tm, "first_lr",       default=1e-3))
    warm_lr      = float(cfg_get(tm, "warm_lr",        default=5e-4))
    cold_lr      = float(cfg_get(tm, "cold_lr",        default=1e-3))
    lr_alpha     = float(cfg_get(tm, "lr_alpha",       default=0.01))
    n_windows    = 1 if T_total <= dT else int(np.ceil((T_total - dT) / stride)) + 1

    batch_r  = int(cfg_get(d, "batch_r",  default=2048))
    batch_bc = int(cfg_get(d, "batch_bc", default=512))
    weights  = (float(cfg_get(lw, "w_pde",    default=1.0)),
                float(cfg_get(lw, "w_wall",   default=100.0)),
                float(cfg_get(lw, "w_inlet",  default=50.0)),
                float(cfg_get(lw, "w_outlet", default=20.0)),
                float(cfg_get(lw, "w_ic",     default=100.0)))
    seed     = int(cfg_get(tm, "seed", default=0))

    # ── Model (already built and wired into pde by the caller) ────────────────
    model    = pde.model
    net_cfg  = network_config(cfg)              # round-trips to prediction
    layers   = net_cfg["layers"]

    overlap_pct = 100.0 * max(0.0, (dT - stride)) / dT if dT > 0 else 0.0
    print(f"{label} (3-D unsteady)")
    print(f"  Network: {type(model).__name__}  layers={layers}")
    print(f"  Time-marching: T_total={T_total}, dT={dT}, stride={stride}"
          f"  →  {n_windows} windows ({overlap_pct:.0f}% overlap)")
    print(f"  Epochs: first={n_first}, warm={n_warm}"
          + (f", cold(baseline)={n_cold}" if cmp_cold else ""))

    # =====================================================================
    #  Transfer chain  (warm-started, time-marching, overlapping or not)
    # =====================================================================
    print("\n" + "=" * 60)
    print("Transfer chain — warm-started time marching")
    print("=" * 60)

    ckpt_dir = os.path.join(out_dir, "windows")
    os.makedirs(ckpt_dir, exist_ok=True)
    base_meta = {
        "problem": problem_name,
        "method":  "time_marching_transfer",
        "network": net_cfg,
        "physics": physics_record,
        "time_marching": {"T_total": T_total, "dT": dT, "stride": stride,
                          "n_windows": n_windows},
    }

    # Up-front time→window index — depends only on config, so prediction works
    # even if the run is interrupted before the very end.
    with open(os.path.join(out_dir, "windows_index.json"), "w") as f:
        json.dump({
            "n_windows": n_windows, "dT": dT, "stride": stride,
            "T_total": T_total,
            "network": net_cfg,
            "physics": physics_record,
            "windows": [
                {"index": k, "t0": k * stride, "t1": k * stride + dT,
                 "checkpoint": f"windows/params_window_{k:03d}.msgpack",
                 "note": "evaluate at local time tau = t_abs - t0, input (x,y,z,tau)"}
                for k in range(n_windows)
            ],
        }, f, indent=2)

    tf_params_list = []
    tf_loss_all    = []
    prev_params    = None

    # Window-level restart — reload the contiguous run of completed windows.
    restart_enabled = bool(cfg_get(tm, "restart", default=True))
    resume_from = 0
    if restart_enabled:
        completed = []
        for k in range(n_windows):
            mp = os.path.join(ckpt_dir, f"params_window_{k:03d}.msgpack")
            if os.path.exists(mp):
                completed.append(k)
            else:
                break
        ok = True
        idx_path = os.path.join(out_dir, "windows_index.json")
        if completed and os.path.exists(idx_path):
            try:
                with open(idx_path) as f:
                    prev_idx = json.load(f)
                if (abs(float(prev_idx.get("dT", dT)) - dT) > 1e-12
                        or abs(float(prev_idx.get("stride", dT)) - stride) > 1e-12
                        or int(prev_idx.get("n_windows", n_windows)) != n_windows):
                    ok = False
            except Exception:
                ok = False
        if completed and ok:
            for k in completed:
                mp = os.path.join(ckpt_dir, f"params_window_{k:03d}.msgpack")
                tf_params_list.append(load_checkpoint(model, mp))
                meta = read_metadata(mp) or {}
                fl   = meta.get("window", {}).get("final_loss")
                tf_loss_all.append([float(fl)] if fl is not None else [float("nan")])
            prev_params = tf_params_list[-1]
            resume_from = len(completed)
            print(f"\n  [Restart] Found {resume_from}/{n_windows} completed "
                  f"window(s) → resuming at window {resume_from + 1}.")
            if resume_from > 0:
                cmp_cold = False
        elif completed and not ok:
            print("\n  [Restart] Saved windows don't match current config "
                  "(dT / stride / n_windows changed) — retraining from window 1.")

    for k in range(resume_from, n_windows):
        t0   = k * stride
        data = _window_data(geom, dT, d, seed=1000 + k)
        if k == 0:
            ic_uvw = jnp.array(steady_uvw_fn(np.array(data["xyz_ic"])))
        else:
            # Previous window's clock at the same absolute moment = τ_prev = stride
            ic_uvw = jnp.array(_eval_uvw(model, prev_params,
                                         data["xyz_ic"], tau=stride))
        epochs = n_first if k == 0 else n_warm
        lr     = first_lr if k == 0 else warm_lr
        print(f"\n  Window {k+1}/{n_windows}  t ∈ [{t0:.3f}, {t0+dT:.3f}]"
              f"  ({'cold' if k == 0 else 'warm'}, {epochs} ep)")
        params, lh = _train_window(
            pde, model, data,
            inlet_target_fn=inlet_target, t0=t0, dT=dT,
            ic_uvw=ic_uvw, epochs=epochs, lr=lr, lr_alpha=lr_alpha,
            weights=weights, batch_r=batch_r, batch_bc=batch_bc,
            init_params=prev_params, seed=seed + k, label=f"tf-w{k+1}")
        tf_params_list.append(params)
        tf_loss_all.append(lh)
        prev_params = params

        save_checkpoint(
            params, ckpt_dir, stem=f"params_window_{k:03d}",
            metadata={**base_meta,
                      "window": {"index": k, "t0": t0, "t1": t0 + dT,
                                 "local_time_range": [0.0, dT],
                                 "final_loss": lh[-1] if lh else None}},
        )

    # ──────────── No-transfer baseline (each window cold-started) ─────────
    cold_loss_all: list = []
    if cmp_cold:
        print("\n" + "=" * 60)
        print("Baseline chain — no transfer (cold start each window)")
        print("=" * 60)
        prev_cold = None
        for k in range(n_windows):
            t0   = k * stride
            data = _window_data(geom, dT, d, seed=2000 + k)
            if k == 0:
                ic_uvw = jnp.array(steady_uvw_fn(np.array(data["xyz_ic"])))
            else:
                ic_uvw = jnp.array(_eval_uvw(model, prev_cold,
                                             data["xyz_ic"], tau=stride))
            print(f"\n  Window {k+1}/{n_windows}  (cold, {n_cold} ep)")
            params_c, lh_c = _train_window(
                pde, model, data,
                inlet_target_fn=inlet_target, t0=t0, dT=dT,
                ic_uvw=ic_uvw, epochs=n_cold, lr=cold_lr, lr_alpha=lr_alpha,
                weights=weights, batch_r=batch_r, batch_bc=batch_bc,
                init_params=None, seed=seed + 500 + k, label=f"cold-w{k+1}")
            cold_loss_all.append(lh_c)
            prev_cold = params_c

    # =====================================================================
    #  Plots
    # =====================================================================
    def _window_for(t_abs):
        return min(int(t_abs // stride), n_windows - 1)

    # ── Centreline velocity over the whole horizon ──────────────────────────
    x_mid = plot_extent[2] if plot_extent is not None else 0.0

    t_line = np.linspace(0.0, T_total, 400, dtype=np.float32)
    u_ctr  = np.array([
        _centreline_u(model, tf_params_list[_window_for(t)],
                      x_mid, float(t - _window_for(t) * stride))
        for t in t_line])

    # Inlet peak forcing (drawn from physics record)
    V_max    = float(physics_record.get("V_max",    1.0))
    V_amp    = float(physics_record.get("V_amp",    0.5))
    T_period = float(physics_record.get("T_period", 1.0))
    omega    = 2.0 * np.pi / T_period
    inlet_peak = V_max + V_amp * np.sin(omega * t_line)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_line, inlet_peak, "k--", lw=1.5, label="Inlet peak forcing")
    ax.plot(t_line, u_ctr, "b-", lw=1.8, label=f"PINN centreline u @ x={x_mid:.2f}")
    for k in range(1, n_windows):
        ax.axvline(k * stride, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("u (centreline)")
    ax.set_title(f"{label} — centreline velocity over {n_windows} windows")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "centreline_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Stitched loss history ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    offset = 0
    for k, lh in enumerate(tf_loss_all):
        xs = np.arange(offset, offset + len(lh))
        ax.semilogy(xs, lh, lw=1.0, label=f"window {k+1}" if k < 8 else None)
        offset += len(lh)
        ax.axvline(offset, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Cumulative epoch")
    ax.set_ylabel("Total loss")
    ax.set_title(f"{label} — per-window loss")
    ax.grid(alpha=0.3, which="both")
    if n_windows <= 8:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transfer_loss_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Transfer vs cold comparison ─────────────────────────────────────────
    if cmp_cold and cold_loss_all:
        wins      = np.arange(1, n_windows + 1)
        tf_final  = [lh[-1] for lh in tf_loss_all]
        cold_final = [lh[-1] for lh in cold_loss_all]
        tf_ep     = [len(lh) for lh in tf_loss_all]
        cold_ep   = [len(lh) for lh in cold_loss_all]
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
        axe.bar(wins - width / 2, tf_ep,   width, label="Transfer (warm)",
                color="#0072B2")
        axe.bar(wins + width / 2, cold_ep, width, label="No transfer (cold)",
                color="#D55E00")
        axe.set_xlabel("Window")
        axe.set_ylabel("Training epochs")
        axe.set_title(f"Epochs per window "
                      f"(transfer total {sum(tf_ep):,} vs cold {sum(cold_ep):,})")
        axe.set_xticks(wins)
        axe.legend(fontsize=9)
        axe.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{label} — transfer learning vs cold start")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "transfer_vs_cold.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Final-window checkpoint alias + config save ─────────────────────────
    save_checkpoint(tf_params_list[-1], out_dir, stem="params_final_window",
                    metadata={**base_meta,
                              "window": {"index": n_windows - 1,
                                         "t0": (n_windows - 1) * stride,
                                         "t1": (n_windows - 1) * stride + dT}})
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


def make_model_from_cfg(cfg) -> tuple:
    """Build the network referred to by the config; returns (model, type_str).

    Supports ``mlp``, ``gated_mlp``, ``fourier_mlp`` and ``temporal_fourier``
    (the temporal-Fourier network derives ω = 2π/T_period from ``physics``).
    """
    net_cfg = network_config(cfg)
    return build_model(net_cfg), net_cfg["type"]


def cosine_squared_inlet_factory(
    R: float, V_max: float, V_amp: float, T_period: float
) -> Callable:
    """Pulsatile parabolic inlet target u(r, t) = (V_max + V_amp·sin ωt)·(1 − r²/R²)."""
    omega = 2.0 * np.pi / T_period

    def fn(xyz, t_abs):
        r2    = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
        shape = 1.0 - r2 / R ** 2
        return (V_max + V_amp * jnp.sin(omega * t_abs)) * shape
    return fn


def carreau_inlet_factory(
    R: float, V_max: float, V_amp: float, T_period: float,
    beta: float, Cu: float, n: float,
) -> Callable:
    """Pulsatile Carreau inlet — developed Carreau profile modulated in time.

    Uses the unit-coordinate developed profile and rescales: u(r,t) = (V_max +
    V_amp sin ωt) · u_carr(r/R) / u_carr(0).  The 1-D solver is invoked with
    Cu_eff = Cu·V_max/R so the imposed inlet is consistent with the PDE's
    constitutive law at the simulation scale.
    """
    omega = 2.0 * np.pi / T_period
    # Pre-tabulate the developed Carreau profile (unit coords, centreline = 1).
    from underPINN.pde.carreau_ns_3d import carreau_developed_profile
    Cu_eff = Cu * V_max / R
    r_ref, u_ref, _ = carreau_developed_profile(beta, Cu_eff, n, u_center=1.0)
    r_ref_j = jnp.asarray(r_ref.astype(np.float32))
    u_ref_j = jnp.asarray(u_ref.astype(np.float32))

    def fn(xyz, t_abs):
        r = jnp.sqrt(xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
        shape = jnp.interp(r / R, r_ref_j, u_ref_j)        # u_carr(r/R), max ≈ 1
        return (V_max + V_amp * jnp.sin(omega * t_abs)) * shape
    return fn


def parabolic_steady_uvw_factory(R: float, V_max: float) -> Callable:
    """Steady Poiseuille IC at t=0: u = V_max(1 − r²/R²), v = w = 0."""
    def fn(xyz):
        r2 = np.asarray(xyz[:, 1]) ** 2 + np.asarray(xyz[:, 2]) ** 2
        u  = V_max * (1.0 - r2 / R ** 2)
        return np.stack([u, np.zeros_like(u), np.zeros_like(u)],
                        axis=1).astype(np.float32)
    return fn


def carreau_steady_uvw_factory(
    R: float, V_max: float, beta: float, Cu: float, n: float,
) -> Callable:
    """Steady Carreau-developed IC at t=0 (host-side; NumPy/SciPy)."""
    from underPINN.pde.carreau_ns_3d import carreau_developed_profile
    Cu_eff = Cu * V_max / R
    r_ref, u_ref, _ = carreau_developed_profile(beta, Cu_eff, n, u_center=1.0)

    def fn(xyz):
        r = np.sqrt(np.asarray(xyz[:, 1]) ** 2 + np.asarray(xyz[:, 2]) ** 2)
        u = V_max * np.interp(r / R, r_ref, u_ref)
        z = np.zeros_like(u)
        return np.stack([u, z, z], axis=1).astype(np.float32)
    return fn
