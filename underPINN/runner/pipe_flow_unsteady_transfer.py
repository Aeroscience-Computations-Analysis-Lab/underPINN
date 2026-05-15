"""Runner for unsteady 3-D pipe flow — transfer learning.

Two strategies driven by YAML config:

  Strategy 1 — Re Transfer
    Source: Re = 10, t ∈ [0, T] → n_source_epochs
    Target: Re = 50, t ∈ [0, T]
      • Transfer: warm start → n_transfer_epochs
      • Scratch : random init → n_scratch_epochs

  Strategy 2 — Temporal Transfer
    Phase 1: Re fixed, t ∈ [0, T1] → n_phase1_epochs
    Phase 2: Re fixed, t ∈ [0, T2] (extended)
      • Transfer: warm start → n_transfer_epochs
      • Scratch : random init → n_scratch_epochs

Expected config sections
------------------------
problem  : pipe_flow_unsteady_transfer

network:
  layers: [3, 64, 64, 64, 64, 1]   # (y,z,t) → u

physics:
  R    : 0.5
  U_max: 1.0

re_transfer:
  source_Re: 10.0
  target_Re: 50.0
  T        : 3.0
  n_source_epochs  : 2500
  n_transfer_epochs: 2000
  n_scratch_epochs : 4500
  source_lr  : 1.0e-3
  transfer_lr: 3.0e-4
  scratch_lr : 1.0e-3

temporal_transfer:
  Re  : 10.0
  T1  : 1.0
  T2  : 3.0
  n_phase1_epochs  : 2500
  n_transfer_epochs: 2000
  n_scratch_epochs : 4500
  phase1_lr  : 1.0e-3
  transfer_lr: 3.0e-4
  scratch_lr : 1.0e-3

data:
  n_collocation: 6000
  n_ic         : 600
  n_bc         : 800
  batch_r      : 512
  batch_ic     : 200
  batch_bc     : 200

loss:
  w_pde: 1.0
  w_ic : 100.0
  w_bc : 50.0

output:
  dir: outputs/pipe_flow_unsteady_transfer
"""

from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.mlp import MLP
from underPINN.pde.pipe_flow_unsteady import UnsteadyPipeFlowPDE
from underPINN.utils.io import save_predictions
from underPINN.utils.sampling import safe_choice


# ---------------------------------------------------------------------------
# Shared sampling helpers
# ---------------------------------------------------------------------------

def _disk_yz(n: int, R: float, seed: int) -> np.ndarray:
    """Uniform (y, z) inside disk of radius R (rejection sampling)."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(a) for a in out) < n:
        y = rng.uniform(-R, R, 4 * n).astype(np.float32)
        z = rng.uniform(-R, R, 4 * n).astype(np.float32)
        k = y ** 2 + z ** 2 <= R ** 2
        out.append(np.column_stack([y[k], z[k]]))
    return np.concatenate(out)[:n]


def _make_data(T: float, N_r: int, N_ic: int, N_bc: int, R: float, seed: int = 0):
    rng    = np.random.default_rng(seed)
    yz_r   = _disk_yz(N_r,  R, seed)
    t_r    = rng.uniform(0.0, T, N_r).astype(np.float32)
    yz_ic  = _disk_yz(N_ic, R, seed + 1)
    theta  = rng.uniform(0.0, 2 * np.pi, N_bc).astype(np.float32)
    yz_bc  = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    t_bc   = rng.uniform(0.0, T, N_bc).astype(np.float32)
    return (jnp.array(yz_r),  jnp.array(t_r),
            jnp.array(yz_ic),
            jnp.array(yz_bc), jnp.array(t_bc))


def _run_training(pde, data, epochs: int, lr: float, R: float,
                  batch_r: int, batch_ic: int, batch_bc: int,
                  w_pde: float, w_ic: float, w_bc: float,
                  init_params=None, seed: int = 0, label: str = "",
                  log_every: int = 500):
    """Train the unsteady pipe PDE; optionally warm-start from *init_params*."""
    yz_r, t_r, yz_ic, yz_bc, t_bc = data

    key    = jax.random.PRNGKey(seed)
    params = (pde.model.init(key, jnp.ones((1, 3)))
              if init_params is None else init_params)

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, state, yz_r_b, t_r_b, yz_ic_b, yz_bc_b, t_bc_b):
        def loss_fn(p):
            pde_l = jnp.mean(pde.residual(p, yz_r_b, t_r_b) ** 2)
            t_z   = jnp.zeros(yz_ic_b.shape[0])
            ic_l  = jnp.mean(pde.u(p, yz_ic_b, t_z) ** 2)
            bc_l  = jnp.mean(pde.u(p, yz_bc_b, t_bc_b) ** 2)
            total = w_pde * pde_l + w_ic * ic_l + w_bc * bc_l
            return total, (pde_l, ic_l, bc_l)
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state      = optimizer.update(grads, state)
        params              = optax.apply_updates(params, updates)
        return params, state, total, aux

    N_r, N_ic, N_bc = yz_r.shape[0], yz_ic.shape[0], yz_bc.shape[0]
    key       = jax.random.PRNGKey(seed + 99)
    loss_hist = []

    for ep in range(epochs):
        key, k1, k2, k3 = jax.random.split(key, 4)
        ir = safe_choice(k1, N_r,  batch_r)
        ii = safe_choice(k2, N_ic, batch_ic)
        ib = safe_choice(k3, N_bc, batch_bc)

        params, opt_state, total, (pl, il, bl) = step(
            params, opt_state,
            yz_r[ir], t_r[ir], yz_ic[ii], yz_bc[ib], t_bc[ib],
        )
        loss_hist.append(float(total))

        if ep % log_every == 0 or ep == epochs - 1:
            tag = f"[{label}] " if label else ""
            print(f"{tag}Epoch {ep:5d} | total {total:.3e} "
                  f"| pde {pl:.3e} | ic {il:.3e} | bc {bl:.3e}")

    return params, loss_hist


def _rel_l2(model, params, pde, yz_val, t_val: float) -> float:
    u_ex  = pde.exact(np.array(yz_val), t_val)
    yzt   = jnp.concatenate([yz_val, jnp.full((yz_val.shape[0], 1), t_val)], axis=1)
    u_pr  = np.array(model.apply(params, yzt)[:, 0])
    return float(np.linalg.norm(u_pr - u_ex) / (np.linalg.norm(u_ex) + 1e-10))


def _eval_radial(model, params, r_arr, t_val: float) -> np.ndarray:
    yz  = jnp.column_stack([jnp.array(r_arr), jnp.zeros(len(r_arr))])
    yzt = jnp.concatenate([yz, jnp.full((len(r_arr), 1), t_val)], axis=1)
    return np.array(model.apply(params, yzt)[:, 0])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pipe_flow_unsteady_transfer(cfg) -> dict:
    """Run Re Transfer and/or Temporal Transfer on unsteady 3-D pipe flow."""
    # ── Unpack shared params ──────────────────────────────────────────────────
    ph  = cfg.physics
    d   = cfg.data
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/pipe_flow_unsteady_transfer")
               if out else "outputs/pipe_flow_unsteady_transfer")
    os.makedirs(out_dir, exist_ok=True)

    R      = float(cfg_get(ph, "R",     default=0.5))
    U_max  = float(cfg_get(ph, "U_max", default=1.0))
    layers = list(cfg.network.layers)

    N_r    = int(cfg_get(d,  "n_collocation", default=6000))
    N_ic   = int(cfg_get(d,  "n_ic",          default=600))
    N_bc   = int(cfg_get(d,  "n_bc",          default=800))
    B_r    = int(cfg_get(d,  "batch_r",       default=512))
    B_ic   = int(cfg_get(d,  "batch_ic",      default=200))
    B_bc   = int(cfg_get(d,  "batch_bc",      default=200))

    w_pde  = float(cfg_get(lw, "w_pde", default=1.0))
    w_ic   = float(cfg_get(lw, "w_ic",  default=100.0))
    w_bc   = float(cfg_get(lw, "w_bc",  default=50.0))

    results = {}

    # ── Strategy 1 — Re Transfer ──────────────────────────────────────────────
    rt = cfg_get(cfg, "re_transfer", default=None)
    if rt is not None:
        print("\n" + "=" * 60)
        print("Strategy 1: Re Transfer")
        print("=" * 60)

        src_Re  = float(cfg_get(rt, "source_Re",        default=10.0))
        tgt_Re  = float(cfg_get(rt, "target_Re",        default=50.0))
        T       = float(cfg_get(rt, "T",                default=3.0))
        N_src   = int(cfg_get(rt,   "n_source_epochs",   default=2500))
        N_tf    = int(cfg_get(rt,   "n_transfer_epochs", default=2000))
        N_sc    = int(cfg_get(rt,   "n_scratch_epochs",  default=4500))
        src_lr  = float(cfg_get(rt, "source_lr",         default=1e-3))
        tf_lr   = float(cfg_get(rt, "transfer_lr",       default=3e-4))
        sc_lr   = float(cfg_get(rt, "scratch_lr",        default=1e-3))

        data_src = _make_data(T, N_r, N_ic, N_bc, R, seed=0)
        data_tgt = _make_data(T, N_r, N_ic, N_bc, R, seed=1)

        print(f"\n[Source]   Re={src_Re}, {N_src} epochs")
        model_src = MLP(layers=layers)
        pde_src   = UnsteadyPipeFlowPDE(model_src, Re=src_Re, R=R, U_max=U_max)
        src_params, _ = _run_training(
            pde_src, data_src, N_src, src_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, seed=0, label="Source")

        print(f"\n[Transfer] Re={tgt_Re}, {N_tf} fine-tune epochs")
        model_tf = MLP(layers=layers)
        pde_tf   = UnsteadyPipeFlowPDE(model_tf, Re=tgt_Re, R=R, U_max=U_max)
        tf_params, hist_tf = _run_training(
            pde_tf, data_tgt, N_tf, tf_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, init_params=src_params, seed=1, label="Transfer")

        print(f"\n[Scratch]  Re={tgt_Re}, {N_sc} epochs from random init")
        model_sc = MLP(layers=layers)
        pde_sc   = UnsteadyPipeFlowPDE(model_sc, Re=tgt_Re, R=R, U_max=U_max)
        sc_params, hist_sc = _run_training(
            pde_sc, data_tgt, N_sc, sc_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, seed=2, label="Scratch")

        # Validation
        yz_val = jnp.array(_disk_yz(2000, R, seed=99))
        for t_check in [1.0, T]:
            e_tf = _rel_l2(model_tf, tf_params, pde_tf, yz_val, t_check)
            e_sc = _rel_l2(model_sc, sc_params, pde_sc, yz_val, t_check)
            print(f"Rel-L² at t={t_check:.1f}: Transfer {e_tf:.3e} | Scratch {e_sc:.3e}")

        # Loss convergence plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax = axes[0]
        ax.semilogy(hist_tf, "b-",  lw=1.5, label=f"Transfer (warm, {N_tf} ep)")
        ax.semilogy(hist_sc[N_src:], "r--", lw=1.5,
                    label=f"Scratch Re={tgt_Re} (same range)")
        ax.axhline(hist_sc[N_src - 1], color="gray", ls=":", lw=1)
        ax.set_xlabel("Fine-tune epoch"); ax.set_ylabel("Total loss")
        ax.set_title(f"Re Transfer  {src_Re} → {tgt_Re}"); ax.legend(fontsize=8)
        ax = axes[1]
        ax.semilogy(hist_sc[:N_src], "r-",  lw=1.2, label=f"Scratch (first {N_src} ep)")
        ax.axhline(hist_tf[0], color="b", ls="--", lw=1.2,
                   label=f"Transfer start = {hist_tf[0]:.2e}")
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
        fig.suptitle("Unsteady Pipe Flow — Re Transfer")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "re_transfer_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Radial profiles
        Nr    = 80
        r_arr = np.linspace(0.0, R, Nr, dtype=np.float32)
        t_snaps = [0.5 * T, T]
        fig, axes = plt.subplots(1, len(t_snaps), figsize=(12, 4))
        for ax, ts in zip(np.atleast_1d(axes), t_snaps):
            u_ex = np.array(pde_tf.exact(
                np.column_stack([r_arr, np.zeros(Nr)]), ts))
            u_tf_ = _eval_radial(model_tf, tf_params, r_arr, ts)
            u_sc_ = _eval_radial(model_sc, sc_params, r_arr, ts)
            ax.plot(r_arr, u_ex,  "k-",  lw=2.0, label="Exact")
            ax.plot(r_arr, u_tf_, "b--", lw=1.8, label="Transfer")
            ax.plot(r_arr, u_sc_, "r:",  lw=1.8, label="Scratch")
            ax.set_title(f"t = {ts:.1f}"); ax.set_xlabel("r"); ax.set_ylabel("u")
            ax.grid(ls="--", alpha=0.4)
            if ts == t_snaps[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"Unsteady Pipe Flow Re={tgt_Re}: Re Transfer profiles")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "re_transfer_profiles.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save predictions
        yz_r, t_r, _, _, _ = data_tgt
        for lbl, mdl, prm in [("transfer", model_tf, tf_params),
                               ("scratch",  model_sc, sc_params)]:
            yzt    = jnp.concatenate([yz_r, t_r[:, None]], axis=1)
            u_pred = np.array(mdl.apply(prm, yzt)[:, 0])
            save_predictions(
                out_dir,
                coords  = {"y": np.array(yz_r[:, 0]),
                           "z": np.array(yz_r[:, 1]),
                           "t": np.array(t_r)},
                outputs = {"u_pred": u_pred},
                filename=f"predictions_re_transfer_{lbl}.npz",
            )

        results["re_transfer"] = {
            "transfer_final": hist_tf[-1],
            "scratch_final":  hist_sc[-1],
        }

    # ── Strategy 2 — Temporal Transfer ────────────────────────────────────────
    tt = cfg_get(cfg, "temporal_transfer", default=None)
    if tt is not None:
        print("\n" + "=" * 60)
        print("Strategy 2: Temporal Transfer")
        print("=" * 60)

        Re    = float(cfg_get(tt, "Re",               default=10.0))
        T1    = float(cfg_get(tt, "T1",               default=1.0))
        T2    = float(cfg_get(tt, "T2",               default=3.0))
        N_p1  = int(cfg_get(tt,   "n_phase1_epochs",   default=2500))
        N_tf  = int(cfg_get(tt,   "n_transfer_epochs", default=2000))
        N_sc  = int(cfg_get(tt,   "n_scratch_epochs",  default=4500))
        p1_lr = float(cfg_get(tt, "phase1_lr",         default=1e-3))
        tf_lr = float(cfg_get(tt, "transfer_lr",       default=3e-4))
        sc_lr = float(cfg_get(tt, "scratch_lr",        default=1e-3))

        data_p1 = _make_data(T1, N_r, N_ic, N_bc, R, seed=0)
        data_p2 = _make_data(T2, N_r, N_ic, N_bc, R, seed=0)

        print(f"\n[Phase 1]  Re={Re}, t∈[0,{T1}], {N_p1} epochs")
        model_p1 = MLP(layers=layers)
        pde_p1   = UnsteadyPipeFlowPDE(model_p1, Re=Re, R=R, U_max=U_max)
        p1_params, _ = _run_training(
            pde_p1, data_p1, N_p1, p1_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, seed=0, label="Phase1")

        print(f"\n[Transfer] Re={Re}, t∈[0,{T2}], {N_tf} fine-tune epochs")
        model_p2 = MLP(layers=layers)
        pde_p2   = UnsteadyPipeFlowPDE(model_p2, Re=Re, R=R, U_max=U_max)
        p2_params, hist_p2 = _run_training(
            pde_p2, data_p2, N_tf, tf_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, init_params=p1_params, seed=1, label="Transfer")

        print(f"\n[Scratch]  Re={Re}, t∈[0,{T2}], {N_sc} epochs from random init")
        model_sc2 = MLP(layers=layers)
        pde_sc2   = UnsteadyPipeFlowPDE(model_sc2, Re=Re, R=R, U_max=U_max)
        sc2_params, hist_sc2 = _run_training(
            pde_sc2, data_p2, N_sc, sc_lr, R, B_r, B_ic, B_bc,
            w_pde, w_ic, w_bc, seed=2, label="Scratch")

        # Loss plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(hist_p2, "b-",  lw=1.5,
                    label=f"Temporal transfer (warm, {N_tf} ep on [0,{T2}])")
        ax.semilogy(hist_sc2[N_p1:], "r--", lw=1.5,
                    label=f"Scratch on [0,{T2}] (same epoch range)")
        ax.set_xlabel("Fine-tune epoch"); ax.set_ylabel("Total loss")
        ax.set_title(f"Temporal Transfer — Unsteady Pipe Re={Re}\n"
                     f"Phase-1 [0,{T1}] → Phase-2 [0,{T2}]")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "temporal_transfer_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Radial profiles at multiple times
        Nr    = 80
        r_arr = np.linspace(0.0, R, Nr, dtype=np.float32)
        t_snaps = [t for t in [0.5, 1.0, 1.5, T2] if t <= T2]
        fig, axes = plt.subplots(1, len(t_snaps), figsize=(14, 4))
        for ax, ts in zip(np.atleast_1d(axes), t_snaps):
            u_ex  = np.array(pde_p2.exact(
                np.column_stack([r_arr, np.zeros(Nr)]), ts))
            u_p2_ = _eval_radial(model_p2,  p2_params,  r_arr, ts)
            u_sc_ = _eval_radial(model_sc2, sc2_params, r_arr, ts)
            ax.plot(r_arr, u_ex,  "k-",  lw=2.0, label="Exact")
            ax.plot(r_arr, u_p2_, "b--", lw=1.8, label="Transfer")
            ax.plot(r_arr, u_sc_, "r:",  lw=1.8, label="Scratch")
            ax.set_title(f"t = {ts:.1f}"); ax.set_xlabel("r"); ax.set_ylabel("u")
            ax.grid(ls="--", alpha=0.4)
            if ts == t_snaps[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"Temporal Transfer — Unsteady Pipe Re={Re}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "temporal_transfer_profiles.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save predictions
        yz_r, t_r, _, _, _ = data_p2
        for lbl, mdl, prm in [("transfer", model_p2,  p2_params),
                               ("scratch",  model_sc2, sc2_params)]:
            yzt    = jnp.concatenate([yz_r, t_r[:, None]], axis=1)
            u_pred = np.array(mdl.apply(prm, yzt)[:, 0])
            save_predictions(
                out_dir,
                coords  = {"y": np.array(yz_r[:, 0]),
                           "z": np.array(yz_r[:, 1]),
                           "t": np.array(t_r)},
                outputs = {"u_pred": u_pred},
                filename=f"predictions_temporal_{lbl}.npz",
            )

        results["temporal_transfer"] = {
            "transfer_final": hist_p2[-1],
            "scratch_final":  hist_sc2[-1],
        }

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nAll outputs saved to: {out_dir}/")
    return results
