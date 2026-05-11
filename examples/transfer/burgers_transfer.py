"""
Transfer Learning — 1-D Burgers Equation
==========================================

Demonstrates two independent transfer learning strategies on Burgers:

    u_t + u u_x = ν u_xx,   x ∈ [-1, 1],  t ∈ [0, T]
    IC : u(x, 0) = −sin(πx)
    BC : u(±1, t) = 0

═══════════════════════════════════════════════════════════
Strategy 1 — Parameter Transfer  (different viscosity ν)
═══════════════════════════════════════════════════════════
Source  : ν = 0.05  (diffusion-dominated, easy)  →  2 000 epochs
Target  : ν = 0.01  (convection-dominated, hard)
  • Transfer : initialise from source weights → 2 000 fine-tune epochs
  • Scratch  : random init → 4 000 epochs  (same total compute)

═══════════════════════════════════════════════════════════
Strategy 2 — Temporal Transfer  (extend time horizon)
═══════════════════════════════════════════════════════════
Phase 1 : ν = 0.01, t ∈ [0, 1]              → 2 000 epochs
Phase 2 : ν = 0.01, t ∈ [0, 2] (extended)
  • Transfer : initialise from phase-1 weights → 2 000 fine-tune epochs
  • Scratch  : random init on [0, 2]           → 4 000 epochs

Both strategies plot loss curves showing that transfer converges
faster and/or reaches lower loss than training from scratch with
the same total epoch budget.

Outputs
-------
transfer_param_loss.png   — strategy 1 convergence comparison
transfer_param_soln.png   — strategy 1 solution at final epoch
transfer_temporal_loss.png — strategy 2 convergence comparison
transfer_temporal_soln.png — strategy 2 solution snapshots
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


# ── Architecture (shared across all runs) ──────────────────────────────────
LAYERS = [2, 64, 64, 64, 64, 1]

# ── Loss weights ────────────────────────────────────────────────────────────
IC_W = 100.0
BC_W =  10.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_data(T: float, N_r=6000, N_ic=200, N_bc=300, seed=0):
    """Random collocation + linspace IC/BC for 1-D Burgers."""
    rng = np.random.default_rng(seed)

    x_r = rng.uniform(-1.0, 1.0, N_r).astype(np.float32)
    t_r = rng.uniform(0.0,  T,   N_r).astype(np.float32)

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


def build_solver(nu: float, epochs: int, lr: float = 1e-3, seed: int = 0):
    """Create a fresh (model, solver, config) triple."""
    model  = MLP(layers=LAYERS)
    pde    = BurgersPDE(model, nu=nu)
    loss   = PINNLoss(model, pde, ic_weight=IC_W, bc_weight=BC_W,
                      loss_type="l2", rba=True)
    solver = FBPINNSolver(model, pde, loss=loss)
    solver.init(jax.random.PRNGKey(seed))

    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        lr_schedule=optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2),
        batch_r=2048, batch_i=256, batch_b=256,
        log_every=500,
        callbacks=[ConsoleLogger(log_every=500)],
    )
    return model, solver, config


def eval_solution(model, params, T: float, nu: float, Nx=200, Nt=100):
    x  = jnp.linspace(-1.0, 1.0, Nx)
    t  = jnp.linspace(0.0,  T,   Nt)
    XX, TT = jnp.meshgrid(x, t, indexing="ij")
    pts    = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
    u      = model.apply(params, pts)[:, 0].reshape(Nx, Nt)
    return np.array(x), np.array(t), np.array(u)


# ---------------------------------------------------------------------------
# Strategy 1 — Parameter Transfer
# ---------------------------------------------------------------------------

def run_parameter_transfer():
    print("\n" + "=" * 60)
    print("Strategy 1: Parameter Transfer  ν=0.05 → ν=0.01")
    print("=" * 60)

    T = 2.0
    data_src = make_data(T, seed=0)   # source and target share the same domain
    data_tgt = make_data(T, seed=1)   # slight variation for target

    N_SRC = 2000
    N_TGT = 2000   # fine-tune budget
    N_SC  = N_SRC + N_TGT   # scratch gets same total

    # ── Source training (ν = 0.05) ──────────────────────────────────────────
    print(f"\n[Source]  ν=0.05, {N_SRC} epochs")
    _, solver_src, cfg_src = build_solver(nu=0.05, epochs=N_SRC, lr=1e-3)
    solver_src.train(*data_src, config=cfg_src)
    source_params = solver_src.params

    # ── Target: transfer (ν = 0.01) ─────────────────────────────────────────
    print(f"\n[Transfer]  ν=0.01, {N_TGT} fine-tune epochs")
    model_tf, solver_tf, cfg_tf = build_solver(nu=0.01, epochs=N_TGT, lr=3e-4)
    solver_tf.load_params(source_params)          # ← warm start
    solver_tf.train(*data_tgt, config=cfg_tf)

    # ── Target: scratch (ν = 0.01) ──────────────────────────────────────────
    print(f"\n[Scratch]  ν=0.01, {N_SC} epochs from random init")
    model_sc, solver_sc, cfg_sc = build_solver(nu=0.01, epochs=N_SC, lr=1e-3)
    solver_sc.train(*data_tgt, config=cfg_sc)

    # ── Convergence plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(solver_tf.loss_hist, "b-",  lw=1.5, label=f"Transfer (warm start, {N_TGT} ep)")
    ax.semilogy(solver_sc.loss_hist[N_SRC:], "r--", lw=1.5,
                label=f"Scratch (same epoch range)")
    ax.axhline(solver_sc.loss_hist[N_SRC - 1], color="gray", ls=":", lw=1,
               label=f"Scratch loss after {N_SRC} ep")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Parameter Transfer — Burgers\nν = 0.05 → 0.01")
    ax.legend(fontsize=8)

    # Source training loss (reference)
    ax2 = axes[1]
    ax2.semilogy(solver_src.loss_hist,  "g-",  lw=1.2, label=f"Source ν=0.05 ({N_SRC} ep)")
    ax2.semilogy(solver_sc.loss_hist[:N_SRC], "r-",  lw=1.2, label=f"Scratch ν=0.01 (first {N_SRC} ep)")
    # Show transfer starting point
    ax2.axhline(solver_tf.loss_hist[0], color="b", ls="--", lw=1.2,
                label=f"Transfer start loss = {solver_tf.loss_hist[0]:.2e}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total loss")
    ax2.set_title("Source vs Scratch (first phase)\n(transfer starts where source ends)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("transfer_param_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: transfer_param_loss.png")

    # ── Solution comparison ────────────────────────────────────────────────
    x_tf, t_tf, u_tf = eval_solution(model_tf, solver_tf.params, T, 0.01)
    x_sc, t_sc, u_sc = eval_solution(model_sc, solver_sc.params, T, 0.01)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, u, title in zip(axes2,
                             [u_tf, u_sc],
                             ["Transfer  ν=0.01", "Scratch  ν=0.01"]):
        cf = ax.contourf(x_tf, t_tf, u, 50, cmap="RdBu_r",
                         vmin=-1.0, vmax=1.0)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("t")
    fig2.suptitle("Burgers: Parameter Transfer solution comparison")
    fig2.tight_layout()
    fig2.savefig("transfer_param_soln.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: transfer_param_soln.png")

    # ── Save predictions at collocation points ───────────────────────────────
    x_r, t_r = data_tgt[0], data_tgt[1]
    pts_r = jnp.stack([x_r, t_r], axis=1)
    for label, model_, solver_ in [("transfer", model_tf, solver_tf),
                                   ("scratch",  model_sc, solver_sc)]:
        u_pred_r = model_.apply(solver_.params, pts_r)[:, 0]
        save_predictions(
            ".",
            coords  = {"x": x_r, "t": t_r},
            outputs = {"u_pred": u_pred_r},
            filename=f"predictions_burgers_param_{label}.npz",
        )

    tf_final = solver_tf.loss_hist[-1]
    sc_final = solver_sc.loss_hist[-1]
    print(f"\nFinal loss — Transfer: {tf_final:.3e}  |  Scratch: {sc_final:.3e}")
    print(f"Transfer loss at start of fine-tuning: {solver_tf.loss_hist[0]:.3e}")


# ---------------------------------------------------------------------------
# Strategy 2 — Temporal Transfer
# ---------------------------------------------------------------------------

def run_temporal_transfer():
    print("\n" + "=" * 60)
    print("Strategy 2: Temporal Transfer  t∈[0,1] → t∈[0,2]")
    print("=" * 60)

    NU   = 0.01
    T1, T2 = 1.0, 2.0
    N_P1 = 2000
    N_P2 = 2000
    N_SC = N_P1 + N_P2

    data_p1 = make_data(T1, seed=0)
    data_p2 = make_data(T2, seed=0)

    # ── Phase 1: short horizon ───────────────────────────────────────────────
    print(f"\n[Phase 1]  ν={NU}, t∈[0,{T1}], {N_P1} epochs")
    _, solver_p1, cfg_p1 = build_solver(nu=NU, epochs=N_P1, lr=1e-3)
    solver_p1.train(*data_p1, config=cfg_p1)
    phase1_params = solver_p1.params

    # ── Phase 2: extended horizon via transfer ───────────────────────────────
    print(f"\n[Transfer] ν={NU}, t∈[0,{T2}], {N_P2} fine-tune epochs")
    model_p2, solver_p2, cfg_p2 = build_solver(nu=NU, epochs=N_P2, lr=3e-4)
    solver_p2.load_params(phase1_params)          # ← temporal warm start
    solver_p2.train(*data_p2, config=cfg_p2)

    # ── Scratch on full horizon ──────────────────────────────────────────────
    print(f"\n[Scratch]  ν={NU}, t∈[0,{T2}], {N_SC} epochs from random init")
    model_sc, solver_sc, cfg_sc = build_solver(nu=NU, epochs=N_SC, lr=1e-3)
    solver_sc.train(*data_p2, config=cfg_sc)

    # ── Convergence plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(solver_p2.loss_hist, "b-",  lw=1.5,
                label=f"Temporal transfer (warm start, {N_P2} ep on [0,{T2}])")
    ax.semilogy(solver_sc.loss_hist[N_P1:], "r--", lw=1.5,
                label=f"Scratch on [0,{T2}] (same epoch range)")
    ax.set_xlabel("Fine-tuning epoch on extended domain")
    ax.set_ylabel("Total loss")
    ax.set_title(f"Temporal Transfer — Burgers  ν={NU}\n"
                 f"Phase-1 train on [0,{T1}], phase-2 extend to [0,{T2}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig("transfer_temporal_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: transfer_temporal_loss.png")

    # ── Solution snapshots ───────────────────────────────────────────────────
    Nx = 200
    x_plot = jnp.linspace(-1.0, 1.0, Nx)
    snapshots = [0.5, 1.0, 1.5, 2.0]

    fig2, axes2 = plt.subplots(1, len(snapshots), figsize=(14, 3.5))
    for ax, t_snap in zip(axes2, snapshots):
        t_arr = jnp.full(Nx, t_snap)
        pts   = jnp.stack([x_plot, t_arr], axis=1)
        u_tf  = np.array(model_p2.apply(solver_p2.params, pts)[:, 0])
        u_sc  = np.array(model_sc.apply(solver_sc.params, pts)[:, 0])
        u_p1  = np.array(solver_p1.model.apply(phase1_params, pts)[:, 0])

        ax.plot(x_plot, u_tf, "b-",  lw=1.5, label="Transfer")
        ax.plot(x_plot, u_sc, "r--", lw=1.5, label="Scratch")
        if t_snap <= T1:
            ax.plot(x_plot, u_p1, "g:", lw=1.2, label="Phase-1 (src)")
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_title(f"t = {t_snap}")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        if t_snap == snapshots[0]:
            ax.legend(fontsize=7)

    fig2.suptitle(f"Temporal Transfer: Burgers ν={NU}, t∈[0,{T2}]")
    fig2.tight_layout()
    fig2.savefig("transfer_temporal_soln.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: transfer_temporal_soln.png")

    # ── Save predictions at collocation points ───────────────────────────────
    x_r, t_r = data_p2[0], data_p2[1]
    pts_r = jnp.stack([x_r, t_r], axis=1)
    for label, model_, solver_ in [("transfer", model_p2, solver_p2),
                                   ("scratch",  model_sc, solver_sc)]:
        u_pred_r = model_.apply(solver_.params, pts_r)[:, 0]
        save_predictions(
            ".",
            coords  = {"x": x_r, "t": t_r},
            outputs = {"u_pred": u_pred_r},
            filename=f"predictions_burgers_temporal_{label}.npz",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())
    run_parameter_transfer()
    run_temporal_transfer()
    print("\nAll transfer learning experiments complete.")


if __name__ == "__main__":
    main()
