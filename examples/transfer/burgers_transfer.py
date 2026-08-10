"""Burgers Transfer Learning PINN.

Run directly or via the CLI:

    python examples/transfer/burgers_transfer.py              # uses burgers_transfer.yaml
    python examples/transfer/burgers_transfer.py myconfig.yaml
    python -m underPINN run examples/transfer/burgers_transfer.yaml

Two transfer learning strategies on 1-D Burgers; each benchmarked against scratch:

  Strategy 1 — Parameter Transfer
    Source: ν_src (easy)  →  n_source_epochs
    Target: ν_tgt (hard)
      • Transfer: warm start → n_transfer_epochs
      • Scratch : random init → n_scratch_epochs  (= n_source + n_transfer)

  Strategy 2 — Temporal Transfer
    Phase 1: ν fixed, t ∈ [0, T1]  →  n_phase1_epochs
    Phase 2: ν fixed, t ∈ [0, T2]  (extended)
      • Transfer: warm start → n_transfer_epochs
      • Scratch : random init → n_scratch_epochs
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
from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_data(T: float, N_r: int, N_ic: int, N_bc: int, seed: int = 0):
    rng  = np.random.default_rng(seed)
    x_r  = rng.uniform(-1.0, 1.0, N_r).astype(np.float32)
    t_r  = rng.uniform(0.0,  T,   N_r).astype(np.float32)
    x_ic = np.linspace(-1.0, 1.0, N_ic, dtype=np.float32)
    u_ic = (-np.sin(np.pi * x_ic)).astype(np.float32)
    t_bc = rng.uniform(0.0, T, N_bc).astype(np.float32)
    x_bc = np.concatenate([np.full(N_bc, -1., np.float32),
                            np.full(N_bc,  1., np.float32)])
    t_bc = np.concatenate([t_bc, t_bc])
    u_bc = np.zeros_like(x_bc)
    return (
        jnp.array(x_r),  jnp.array(t_r),
        jnp.array(x_ic), jnp.array(u_ic),
        jnp.array(x_bc), jnp.array(t_bc), jnp.array(u_bc),
    )


def _build_solver(layers, nu, epochs, lr, ic_w, bc_w, rba, log_every, seed):
    model  = MLP(layers=layers)
    pde    = BurgersPDE(model, nu=nu)
    loss   = PINNLoss(model, pde, ic_weight=ic_w, bc_weight=bc_w,
                      loss_type="l2", rba=rba)
    solver = FBPINNSolver(model, pde, loss=loss)
    solver.init(jax.random.PRNGKey(seed))
    sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2)
    config = TrainingConfig(
        epochs=epochs, lr=lr, lr_schedule=sched,
        batch_r=2048, batch_i=256, batch_b=256,
        log_every=log_every,
        callbacks=[ConsoleLogger(log_every=log_every)],
    )
    return model, solver, config


def _eval_grid(model, params, T: float, Nx: int = 200, Nt: int = 100):
    x = jnp.linspace(-1.0, 1.0, Nx)
    t = jnp.linspace(0.0,   T,  Nt)
    XX, TT = jnp.meshgrid(x, t, indexing="ij")
    pts    = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
    u      = model.apply(params, pts)[:, 0].reshape(Nx, Nt)
    return np.array(x), np.array(t), np.array(u)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_burgers_transfer(cfg) -> dict:
    """Run both Burgers transfer learning strategies driven by YAML config."""
    # ── Unpack shared config ──────────────────────────────────────────────────
    d   = cfg.data
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/burgers_transfer") if out else "outputs/burgers_transfer"
    os.makedirs(out_dir, exist_ok=True)

    layers    = list(cfg.network.layers)
    T_default = float(cfg_get(cfg.physics, "T", default=2.0))
    N_r  = int(cfg_get(d, "n_collocation", default=6000))
    N_ic = int(cfg_get(d, "n_ic",          default=200))
    N_bc = int(cfg_get(d, "n_bc",          default=300))
    ic_w = float(cfg_get(lw, "ic_weight",  default=100.0))
    bc_w = float(cfg_get(lw, "bc_weight",  default=10.0))
    rba  = bool(cfg_get(lw,  "rba",        default=True))

    log_every = 500
    results   = {}

    # ── Strategy 1 — Parameter Transfer ──────────────────────────────────────
    pt = cfg_get(cfg, "parameter_transfer", default=None)
    if pt is not None:
        print("\n" + "=" * 60)
        print("Strategy 1: Parameter Transfer")
        print("=" * 60)

        src_nu = float(cfg_get(pt, "source_nu",        default=0.05))
        tgt_nu = float(cfg_get(pt, "target_nu",        default=0.01))
        N_src  = int(cfg_get(pt,   "n_source_epochs",   default=2000))
        N_tf   = int(cfg_get(pt,   "n_transfer_epochs", default=2000))
        N_sc   = int(cfg_get(pt,   "n_scratch_epochs",  default=4000))
        src_lr = float(cfg_get(pt, "source_lr",         default=1e-3))
        tf_lr  = float(cfg_get(pt, "transfer_lr",       default=3e-4))
        sc_lr  = float(cfg_get(pt, "scratch_lr",        default=1e-3))

        data_src = _make_data(T_default, N_r, N_ic, N_bc, seed=0)
        data_tgt = _make_data(T_default, N_r, N_ic, N_bc, seed=1)

        print(f"\n[Source]   ν={src_nu}, {N_src} epochs")
        _, solver_src, cfg_src = _build_solver(
            layers, src_nu, N_src, src_lr, ic_w, bc_w, rba, log_every, 0)
        solver_src.train(*data_src, config=cfg_src)

        print(f"\n[Transfer] ν={tgt_nu}, {N_tf} fine-tune epochs")
        model_tf, solver_tf, cfg_tf = _build_solver(
            layers, tgt_nu, N_tf, tf_lr, ic_w, bc_w, rba, log_every, 1)
        solver_tf.load_params(solver_src.params)
        solver_tf.train(*data_tgt, config=cfg_tf)

        print(f"\n[Scratch]  ν={tgt_nu}, {N_sc} epochs from random init")
        model_sc, solver_sc, cfg_sc = _build_solver(
            layers, tgt_nu, N_sc, sc_lr, ic_w, bc_w, rba, log_every, 2)
        solver_sc.train(*data_tgt, config=cfg_sc)

        # Convergence plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax = axes[0]
        ax.semilogy(solver_tf.loss_hist, "b-",  lw=1.5,
                    label=f"Transfer (warm, {N_tf} ep)")
        ax.semilogy(solver_sc.loss_hist[N_src:], "r--", lw=1.5,
                    label="Scratch (same epoch range)")
        ax.axhline(solver_sc.loss_hist[N_src - 1], color="gray", ls=":", lw=1)
        ax.set_xlabel("Fine-tune epoch"); ax.set_ylabel("Total loss")
        ax.set_title(f"Parameter Transfer\nν = {src_nu} → {tgt_nu}")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.semilogy(solver_src.loss_hist, "g-",  lw=1.2, label=f"Source ν={src_nu}")
        ax.semilogy(solver_sc.loss_hist[:N_src], "r-",   lw=1.2,
                    label=f"Scratch ν={tgt_nu} (first {N_src} ep)")
        ax.axhline(solver_tf.loss_hist[0], color="b", ls="--", lw=1.2,
                   label=f"Transfer start = {solver_tf.loss_hist[0]:.2e}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
        ax.set_title("Source vs Scratch (first phase)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "param_transfer_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Solution plots
        x_tf, t_tf, u_tf = _eval_grid(model_tf, solver_tf.params, T_default)
        x_sc, t_sc, u_sc = _eval_grid(model_sc, solver_sc.params, T_default)
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, u, title in zip(axes2, [u_tf, u_sc],
                                 [f"Transfer  ν={tgt_nu}", f"Scratch  ν={tgt_nu}"]):
            cf = ax.contourf(x_tf, t_tf, u.T, 50, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            plt.colorbar(cf, ax=ax)
            ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("t")
        fig2.suptitle("Burgers: Parameter Transfer solution comparison")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "param_transfer_soln.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # Save predictions
        x_r, t_r = data_tgt[0], data_tgt[1]
        pts_r = jnp.stack([x_r, t_r], axis=1)
        for lbl, mdl, slv in [("transfer", model_tf, solver_tf),
                               ("scratch",  model_sc, solver_sc)]:
            u_pred_r = mdl.apply(slv.params, pts_r)[:, 0]
            save_predictions(
                out_dir,
                coords  = {"x": np.array(x_r), "t": np.array(t_r)},
                outputs = {"u_pred": np.array(u_pred_r)},
                filename=f"predictions_param_{lbl}.npz",
            )

        # Save checkpoints
        _net_meta = {"type": "mlp", "layers": layers}
        solver_tf.save_checkpoint(out_dir, stem="params_param_transfer", metadata={
            "problem": "burgers_transfer", "strategy": "parameter_transfer",
            "network": _net_meta,
            "physics": {"source_nu": src_nu, "target_nu": tgt_nu},
            "final_loss": solver_tf.loss_hist[-1],
        })
        solver_sc.save_checkpoint(out_dir, stem="params_param_scratch", metadata={
            "problem": "burgers_transfer", "strategy": "parameter_scratch",
            "network": _net_meta, "physics": {"nu": tgt_nu},
            "final_loss": solver_sc.loss_hist[-1],
        })

        results["param_transfer"] = {
            "transfer_final": solver_tf.loss_hist[-1],
            "scratch_final":  solver_sc.loss_hist[-1],
        }
        print(f"\nParam transfer final: {solver_tf.loss_hist[-1]:.3e}  "
              f"| Scratch final: {solver_sc.loss_hist[-1]:.3e}")

    # ── Strategy 2 — Temporal Transfer ────────────────────────────────────────
    tt = cfg_get(cfg, "temporal_transfer", default=None)
    if tt is not None:
        print("\n" + "=" * 60)
        print("Strategy 2: Temporal Transfer")
        print("=" * 60)

        nu   = float(cfg_get(tt, "nu",                default=0.01))
        T1   = float(cfg_get(tt, "T1",               default=1.0))
        T2   = float(cfg_get(tt, "T2",               default=2.0))
        N_p1 = int(cfg_get(tt,  "n_phase1_epochs",   default=2000))
        N_tf  = int(cfg_get(tt,  "n_transfer_epochs", default=2000))
        N_sc  = int(cfg_get(tt,  "n_scratch_epochs",  default=4000))
        p1_lr = float(cfg_get(tt, "phase1_lr",        default=1e-3))
        tf_lr = float(cfg_get(tt, "transfer_lr",      default=3e-4))
        sc_lr = float(cfg_get(tt, "scratch_lr",       default=1e-3))

        data_p1 = _make_data(T1, N_r, N_ic, N_bc, seed=0)
        data_p2 = _make_data(T2, N_r, N_ic, N_bc, seed=0)

        print(f"\n[Phase 1]  ν={nu}, t∈[0,{T1}], {N_p1} epochs")
        _, solver_p1, cfg_p1 = _build_solver(
            layers, nu, N_p1, p1_lr, ic_w, bc_w, rba, log_every, 0)
        solver_p1.train(*data_p1, config=cfg_p1)

        print(f"\n[Transfer] ν={nu}, t∈[0,{T2}], {N_tf} fine-tune epochs")
        model_p2, solver_p2, cfg_p2 = _build_solver(
            layers, nu, N_tf, tf_lr, ic_w, bc_w, rba, log_every, 1)
        solver_p2.load_params(solver_p1.params)
        solver_p2.train(*data_p2, config=cfg_p2)

        print(f"\n[Scratch]  ν={nu}, t∈[0,{T2}], {N_sc} epochs from random init")
        model_sc, solver_sc, cfg_sc = _build_solver(
            layers, nu, N_sc, sc_lr, ic_w, bc_w, rba, log_every, 2)
        solver_sc.train(*data_p2, config=cfg_sc)

        # Convergence plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(solver_p2.loss_hist, "b-",  lw=1.5,
                    label=f"Temporal transfer (warm, {N_tf} ep on [0,{T2}])")
        ax.semilogy(solver_sc.loss_hist[N_p1:], "r--", lw=1.5,
                    label=f"Scratch on [0,{T2}] (same epoch range)")
        ax.set_xlabel("Fine-tune epoch on extended domain")
        ax.set_ylabel("Total loss")
        ax.set_title(f"Temporal Transfer — Burgers  ν={nu}\n"
                     f"Phase-1 [0,{T1}] → Phase-2 [0,{T2}]")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "temporal_transfer_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Solution snapshots
        Nx = 200
        x_plot   = jnp.linspace(-1.0, 1.0, Nx)
        snapshots = [s for s in [0.5, 1.0, 1.5, T2] if s <= T2]
        fig2, axes2 = plt.subplots(1, len(snapshots), figsize=(14, 3.5))
        for ax, t_snap in zip(np.atleast_1d(axes2), snapshots):
            t_arr = jnp.full(Nx, t_snap)
            pts   = jnp.stack([x_plot, t_arr], axis=1)
            u_tf  = np.array(model_p2.apply(solver_p2.params, pts)[:, 0])
            u_sc  = np.array(model_sc.apply(solver_sc.params, pts)[:, 0])
            ax.plot(x_plot, u_tf, "b-",  lw=1.5, label="Transfer")
            ax.plot(x_plot, u_sc, "r--", lw=1.5, label="Scratch")
            ax.axhline(0, color="k", lw=0.4, ls="--")
            ax.set_title(f"t = {t_snap}"); ax.set_xlabel("x"); ax.set_ylabel("u")
            if t_snap == snapshots[0]:
                ax.legend(fontsize=7)
        fig2.suptitle(f"Temporal Transfer: Burgers ν={nu}, t∈[0,{T2}]")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "temporal_transfer_soln.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # Save predictions
        x_r, t_r = data_p2[0], data_p2[1]
        pts_r = jnp.stack([x_r, t_r], axis=1)
        for lbl, mdl, slv in [("transfer", model_p2, solver_p2),
                               ("scratch",  model_sc, solver_sc)]:
            u_pred_r = mdl.apply(slv.params, pts_r)[:, 0]
            save_predictions(
                out_dir,
                coords  = {"x": np.array(x_r), "t": np.array(t_r)},
                outputs = {"u_pred": np.array(u_pred_r)},
                filename=f"predictions_temporal_{lbl}.npz",
            )

        # Save checkpoints
        _net_meta = {"type": "mlp", "layers": layers}
        solver_p2.save_checkpoint(out_dir, stem="params_temporal_transfer", metadata={
            "problem": "burgers_transfer", "strategy": "temporal_transfer",
            "network": _net_meta,
            "physics": {"nu": nu, "T1": T1, "T2": T2},
            "final_loss": solver_p2.loss_hist[-1],
        })
        solver_sc.save_checkpoint(out_dir, stem="params_temporal_scratch", metadata={
            "problem": "burgers_transfer", "strategy": "temporal_scratch",
            "network": _net_meta, "physics": {"nu": nu, "T": T2},
            "final_loss": solver_sc.loss_hist[-1],
        })

        results["temporal_transfer"] = {
            "transfer_final": solver_p2.loss_hist[-1],
            "scratch_final":  solver_sc.loss_hist[-1],
        }
        print(f"\nTemporal transfer final: {solver_p2.loss_hist[-1]:.3e}  "
              f"| Scratch final: {solver_sc.loss_hist[-1]:.3e}")

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nAll outputs saved to: {out_dir}/")
    return results


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
        else _HERE / "burgers_transfer.yaml"
    )
    from underPINN.config.loader import load_config
    run_burgers_transfer(load_config(cfg_path))
