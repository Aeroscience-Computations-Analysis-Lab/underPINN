"""
Transfer Learning — 2-D Unsteady Heat Equation
================================================

Demonstrates two transfer learning strategies on the 2-D heat equation:

    u_t = α (u_xx + u_yy),   (x,y) ∈ [0,1]²,   t ∈ [0, T]
    IC : u(x, y, 0) = sin(πx) sin(πy)
    BC : u = 0 on all four edges
    Exact : u = sin(πx) sin(πy) exp(−2α π² t)

Network input dimension is 3: (x, y, t) → u.

═══════════════════════════════════════════════════════════
Strategy 1 — Parameter Transfer  (different diffusivity α)
═══════════════════════════════════════════════════════════
Source  : α = 0.10  (fast diffusion, easy)  →  2 500 epochs
Target  : α = 0.01  (slow diffusion, hard)
  • Transfer : warm start → 2 000 fine-tune epochs
  • Scratch  : random init → 4 500 epochs  (same total compute)

Quantitative metric: relative L2 error vs epoch on a fixed validation set.

═══════════════════════════════════════════════════════════
Strategy 2 — Temporal Transfer  (extend time horizon)
═══════════════════════════════════════════════════════════
Phase 1 : α = 0.01, t ∈ [0, 0.3]            → 2 500 epochs
Phase 2 : α = 0.01, t ∈ [0, 0.8] (extended)
  • Transfer : warm start → 2 000 fine-tune epochs
  • Scratch  : random init → 4 500 epochs

Outputs
-------
heat2d_param_transfer.png   — strategy 1: loss + error curves
heat2d_param_solution.png   — strategy 1: solution maps at t=0.4
heat2d_temporal_transfer.png — strategy 2: loss + error curves
heat2d_temporal_solution.png — strategy 2: solution maps at t=0.6
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


# ── Architecture ─────────────────────────────────────────────────────────────
LAYERS = [3, 64, 64, 64, 64, 1]   # input dim = 3: (x, y, t)

# ── Loss weights ─────────────────────────────────────────────────────────────
IC_W = 100.0
BC_W =  20.0

# ── Validation grid (shared across all runs) ─────────────────────────────────
_NV  = 30
_xv  = np.linspace(0.0, 1.0, _NV, dtype=np.float32)
_yv  = np.linspace(0.0, 1.0, _NV, dtype=np.float32)
_XV, _YV = np.meshgrid(_xv, _yv)
_XY_VAL  = np.stack([_XV.ravel(), _YV.ravel()], axis=1)    # (NV², 2)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_data(T: float, N_r=6000, N_ic=400, N_per_edge=100, seed=0):
    rng = np.random.default_rng(seed)

    # Collocation: (x,y,t) in [0,1]² × [0,T]
    x_r = rng.uniform(0.0, 1.0, N_r).astype(np.float32)
    y_r = rng.uniform(0.0, 1.0, N_r).astype(np.float32)
    t_r = rng.uniform(0.0, T,   N_r).astype(np.float32)
    xy_r = np.stack([x_r, y_r], axis=1)

    # IC: (x,y) at t=0
    xi  = rng.uniform(0.0, 1.0, N_ic).astype(np.float32)
    yi  = rng.uniform(0.0, 1.0, N_ic).astype(np.float32)
    xy_i = np.stack([xi, yi], axis=1)
    u_i  = (np.sin(np.pi * xi) * np.sin(np.pi * yi)).astype(np.float32)

    # BC: four edges at random times
    t_edge = rng.uniform(0.0, T, N_per_edge).astype(np.float32)
    s      = np.linspace(0.0, 1.0, N_per_edge, dtype=np.float32)
    bottom = np.stack([s,               np.zeros_like(s)], axis=1)
    top    = np.stack([s,               np.ones_like(s)],  axis=1)
    left   = np.stack([np.zeros_like(s), s],               axis=1)
    right  = np.stack([np.ones_like(s),  s],               axis=1)
    xy_b   = np.tile(np.concatenate([bottom, top, left, right], axis=0), (1, 1))
    t_b    = np.tile(t_edge, 4).astype(np.float32)
    u_b    = np.zeros(len(t_b), dtype=np.float32)

    return (
        jnp.array(xy_r), jnp.array(t_r),
        jnp.array(xy_i), jnp.array(u_i),
        jnp.array(xy_b), jnp.array(t_b), jnp.array(u_b),
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def make_step_fn(model, pde, optimizer, alpha,
                 xy_i, u_i, xy_b, t_b, u_b, batch_r=1024):
    """Return a JIT-compiled step that mini-batches collocation only."""

    @jax.jit
    def step(params, opt_state, xy_r_b, t_r_b):
        def loss_fn(p):
            res   = pde.residual(p, xy_r_b, t_r_b, alpha=alpha)
            pde_l = jnp.mean(res ** 2)

            u_ic  = pde.u(p, xy_i, jnp.zeros(len(xy_i)))
            ic_l  = jnp.mean((u_ic - u_i) ** 2)

            u_bc  = pde.u(p, xy_b, t_b)
            bc_l  = jnp.mean((u_bc - u_b) ** 2)

            total = pde_l + IC_W * ic_l + BC_W * bc_l
            return total, (pde_l, ic_l, bc_l)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    return step


def rel_l2(u_pred, u_exact):
    return float(
        jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
        / (jnp.sqrt(jnp.mean(u_exact ** 2)) + 1e-10)
    )


def run_training(model, pde, data, epochs, lr, alpha,
                 init_params=None, seed=0, label="",
                 val_every=100, val_t=None):
    """
    Full training loop for UnsteadyHeat2DPDE.

    Returns (final_params, loss_hist, err_hist).
    err_hist[k] is the relative L2 error evaluated every val_every epochs.
    """
    xy_r, t_r, xy_i, u_i, xy_b, t_b, u_b = data
    schedule  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    if init_params is None:
        params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 3)))
    else:
        params = init_params

    opt_state = optimizer.init(params)
    step_fn   = make_step_fn(model, pde, optimizer, alpha, xy_i, u_i, xy_b, t_b, u_b)

    logger  = ConsoleLogger(log_every=500)
    stopper = EarlyStopping(patience=600)
    key     = jax.random.PRNGKey(seed + 99)

    n_r      = xy_r.shape[0]
    BATCH_R  = 1024
    loss_hist = []
    err_hist  = []

    try:
        for ep in range(epochs):
            key, k = jax.random.split(key)
            idx     = jax.random.choice(k, n_r, (BATCH_R,), replace=False)
            params, opt_state, loss, _ = step_fn(
                params, opt_state, xy_r[idx], t_r[idx]
            )
            loss_hist.append(float(loss))

            # Validation error at a fixed time slice
            if val_t is not None and ep % val_every == 0:
                tv  = jnp.full(_XY_VAL.shape[0], val_t, dtype=jnp.float32)
                u_p = pde.u(params, jnp.array(_XY_VAL), tv)
                u_e = pde.exact(jnp.array(_XY_VAL), tv, alpha=alpha)
                err_hist.append((ep, rel_l2(u_p, u_e)))

            logs = {"loss": float(loss)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)

    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1]})
    return params, loss_hist, err_hist


# ---------------------------------------------------------------------------
# Solution field helper
# ---------------------------------------------------------------------------

def solution_map(model, params, pde, alpha, t_slice, N=80):
    x  = np.linspace(0.0, 1.0, N, dtype=np.float32)
    y  = np.linspace(0.0, 1.0, N, dtype=np.float32)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    xy = jnp.stack([jnp.array(XX.ravel()), jnp.array(YY.ravel())], axis=1)
    tv = jnp.full(N * N, t_slice, dtype=jnp.float32)
    u_pred  = np.array(pde.u(params, xy, tv)).reshape(N, N)
    u_exact = np.array(pde.exact(xy, tv, alpha=alpha)).reshape(N, N)
    return x, y, u_pred, u_exact


def plot_solution_row(axes_row, model, params, pde, alpha, t_slice, title_prefix):
    x, y, u_p, u_e = solution_map(model, params, pde, alpha, t_slice)
    err = np.abs(u_p - u_e)
    for ax, field, label in zip(axes_row,
                                [u_e, u_p, err],
                                ["Exact", "PINN", "|Error|"]):
        cf = ax.contourf(x, y, field, 40, cmap="jet")
        plt.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_title(f"{title_prefix} — {label}  (t={t_slice})")
        ax.set_xlabel("x"); ax.set_ylabel("y")


# ---------------------------------------------------------------------------
# Strategy 1 — Parameter Transfer
# ---------------------------------------------------------------------------

def run_parameter_transfer():
    print("\n" + "=" * 60)
    print("Strategy 1: Parameter Transfer  α=0.10 → α=0.01")
    print("=" * 60)

    T      = 0.5
    N_SRC  = 2500
    N_TGT  = 2000
    N_SC   = N_SRC + N_TGT
    VAL_T  = 0.4   # time slice for error evaluation

    data = make_data(T, seed=0)

    # ── Source (α=0.10) ──────────────────────────────────────────────────────
    print(f"\n[Source]  α=0.10, {N_SRC} epochs, T={T}")
    model_src = MLP(layers=LAYERS)
    pde_src   = UnsteadyHeat2DPDE(model_src, alpha=0.10)
    src_params, src_loss, _ = run_training(
        model_src, pde_src, data, N_SRC, lr=1e-3, alpha=0.10, label="Source"
    )

    # ── Transfer (α=0.01, warm start) ────────────────────────────────────────
    print(f"\n[Transfer]  α=0.01, {N_TGT} epochs from source weights")
    model_tf = MLP(layers=LAYERS)
    pde_tf   = UnsteadyHeat2DPDE(model_tf, alpha=0.01)
    tf_params, tf_loss, tf_err = run_training(
        model_tf, pde_tf, data, N_TGT, lr=3e-4, alpha=0.01,
        init_params=src_params,
        label="Transfer", val_t=VAL_T,
    )

    # ── Scratch (α=0.01, random init) ────────────────────────────────────────
    print(f"\n[Scratch]  α=0.01, {N_SC} epochs from random init")
    model_sc = MLP(layers=LAYERS)
    pde_sc   = UnsteadyHeat2DPDE(model_sc, alpha=0.01)
    sc_params, sc_loss, sc_err = run_training(
        model_sc, pde_sc, data, N_SC, lr=1e-3, alpha=0.01,
        label="Scratch", val_t=VAL_T,
    )

    # ── Convergence + error plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(tf_loss,              "b-",  lw=1.5, label=f"Transfer ({N_TGT} ep)")
    ax.semilogy(sc_loss[N_SRC:],      "r--", lw=1.5, label=f"Scratch (last {N_TGT} ep)")
    ax.set_xlabel("Fine-tuning epoch"); ax.set_ylabel("Loss")
    ax.set_title("Parameter Transfer — 2-D Heat\nα = 0.10 → 0.01"); ax.legend()

    ax2 = axes[1]
    if tf_err and sc_err:
        ep_tf, err_tf = zip(*tf_err)
        ep_sc, err_sc = zip(*sc_err)
        ax2.semilogy(ep_tf, err_tf, "b-",  lw=1.5, label=f"Transfer (t={VAL_T})")
        ax2.semilogy([e + N_SRC for e in ep_sc], err_sc, "r--", lw=1.5,
                     label=f"Scratch (t={VAL_T})")
        ax2.axvline(N_SRC, color="gray", ls=":", lw=1, label="Scratch epoch 0")
        ax2.set_xlabel("Total epoch"); ax2.set_ylabel(f"Rel-L2 error at t={VAL_T}")
        ax2.set_title("Accuracy vs total compute")
        ax2.legend()

    fig.tight_layout()
    fig.savefig("heat2d_param_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: heat2d_param_transfer.png")

    # ── Solution maps ─────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    plot_solution_row(axes2[0], model_tf, tf_params, pde_tf, 0.01, VAL_T, "Transfer α=0.01")
    plot_solution_row(axes2[1], model_sc, sc_params, pde_sc, 0.01, VAL_T, "Scratch  α=0.01")
    fig2.suptitle(f"2-D Heat: Parameter Transfer — t={VAL_T}")
    fig2.tight_layout()
    fig2.savefig("heat2d_param_solution.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: heat2d_param_solution.png")

    print(f"\nFinal loss — Transfer: {tf_loss[-1]:.3e}  |  Scratch: {sc_loss[-1]:.3e}")
    if tf_err and sc_err:
        print(f"Final Rel-L2 (t={VAL_T}) — Transfer: {tf_err[-1][1]:.3e}  "
              f"|  Scratch: {sc_err[-1][1]:.3e}")

    # ── Save predictions at collocation points ────────────────────────────────
    xy_r, t_r = data[0], data[1]
    for label, model_, params_, alpha in [("transfer", model_tf, tf_params, 0.01),
                                          ("scratch",  model_sc, sc_params, 0.01)]:
        xyt = jnp.concatenate([xy_r, t_r[:, None]], axis=1)
        u_pred_r  = pde_tf.u(params_, xy_r, t_r)
        u_exact_r = pde_tf.exact(xy_r, t_r, alpha=alpha)
        save_predictions(
            ".",
            coords  = {"x": np.array(xy_r[:, 0]),
                       "y": np.array(xy_r[:, 1]),
                       "t": np.array(t_r)},
            outputs = {"u_pred": u_pred_r},
            exact   = {"u_exact": u_exact_r},
            filename=f"predictions_heat2d_param_{label}.npz",
        )


# ---------------------------------------------------------------------------
# Strategy 2 — Temporal Transfer
# ---------------------------------------------------------------------------

def run_temporal_transfer():
    print("\n" + "=" * 60)
    print("Strategy 2: Temporal Transfer  t∈[0,0.3] → t∈[0,0.8]")
    print("=" * 60)

    ALPHA  = 0.01
    T1, T2 = 0.3, 0.8
    N_P1   = 2500
    N_P2   = 2000
    N_SC   = N_P1 + N_P2
    VAL_T  = 0.6   # in extended domain

    data_p1 = make_data(T1, seed=0)
    data_p2 = make_data(T2, seed=0)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\n[Phase 1]  α={ALPHA}, t∈[0,{T1}], {N_P1} epochs")
    model_p1 = MLP(layers=LAYERS)
    pde_p1   = UnsteadyHeat2DPDE(model_p1, alpha=ALPHA)
    p1_params, p1_loss, _ = run_training(
        model_p1, pde_p1, data_p1, N_P1, lr=1e-3, alpha=ALPHA, label="Phase-1"
    )

    # ── Temporal Transfer ─────────────────────────────────────────────────────
    print(f"\n[Transfer]  α={ALPHA}, t∈[0,{T2}], {N_P2} epochs from phase-1")
    model_tf = MLP(layers=LAYERS)
    pde_tf   = UnsteadyHeat2DPDE(model_tf, alpha=ALPHA)
    tf_params, tf_loss, tf_err = run_training(
        model_tf, pde_tf, data_p2, N_P2, lr=3e-4, alpha=ALPHA,
        init_params=p1_params,
        label="Temporal-TF", val_t=VAL_T,
    )

    # ── Scratch ───────────────────────────────────────────────────────────────
    print(f"\n[Scratch]  α={ALPHA}, t∈[0,{T2}], {N_SC} epochs from random init")
    model_sc = MLP(layers=LAYERS)
    pde_sc   = UnsteadyHeat2DPDE(model_sc, alpha=ALPHA)
    sc_params, sc_loss, sc_err = run_training(
        model_sc, pde_sc, data_p2, N_SC, lr=1e-3, alpha=ALPHA,
        label="Scratch", val_t=VAL_T,
    )

    # ── Convergence + error plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(tf_loss,         "b-",  lw=1.5, label=f"Transfer ({N_P2} ep on [0,{T2}])")
    ax.semilogy(sc_loss[N_P1:],  "r--", lw=1.5, label=f"Scratch (last {N_P2} ep)")
    ax.set_xlabel("Fine-tuning epoch on extended domain")
    ax.set_ylabel("Loss")
    ax.set_title(f"Temporal Transfer — 2-D Heat  α={ALPHA}\n"
                 f"Phase-1: [0,{T1}]  →  Phase-2: [0,{T2}]")
    ax.legend()

    ax2 = axes[1]
    if tf_err and sc_err:
        ep_tf, err_tf = zip(*tf_err)
        ep_sc, err_sc = zip(*sc_err)
        ax2.semilogy(ep_tf, err_tf, "b-",  lw=1.5, label=f"Transfer (t={VAL_T})")
        ax2.semilogy([e + N_P1 for e in ep_sc], err_sc, "r--", lw=1.5,
                     label=f"Scratch (t={VAL_T})")
        ax2.axvline(N_P1, color="gray", ls=":", lw=1, label="Scratch epoch 0")
        ax2.set_xlabel("Total epoch"); ax2.set_ylabel(f"Rel-L2 at t={VAL_T}")
        ax2.set_title("Accuracy vs total compute"); ax2.legend()

    fig.tight_layout()
    fig.savefig("heat2d_temporal_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: heat2d_temporal_transfer.png")

    # ── Solution maps at t = VAL_T (in extended domain) ──────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    plot_solution_row(axes2[0], model_tf, tf_params, pde_tf, ALPHA, VAL_T,
                      f"Transfer (t∈[0,{T2}])")
    plot_solution_row(axes2[1], model_sc, sc_params, pde_sc, ALPHA, VAL_T,
                      f"Scratch  (t∈[0,{T2}])")
    fig2.suptitle(f"2-D Heat: Temporal Transfer — α={ALPHA}, t={VAL_T}")
    fig2.tight_layout()
    fig2.savefig("heat2d_temporal_solution.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: heat2d_temporal_solution.png")

    print(f"\nFinal loss — Transfer: {tf_loss[-1]:.3e}  |  Scratch: {sc_loss[-1]:.3e}")
    if tf_err and sc_err:
        print(f"Final Rel-L2 (t={VAL_T}) — Transfer: {tf_err[-1][1]:.3e}  "
              f"|  Scratch: {sc_err[-1][1]:.3e}")

    # ── Save predictions at collocation points ────────────────────────────────
    xy_r, t_r = data_p2[0], data_p2[1]
    for label, model_, params_ in [("transfer", model_tf, tf_params),
                                   ("scratch",  model_sc, sc_params)]:
        u_pred_r  = pde_tf.u(params_, xy_r, t_r)
        u_exact_r = pde_tf.exact(xy_r, t_r, alpha=ALPHA)
        save_predictions(
            ".",
            coords  = {"x": np.array(xy_r[:, 0]),
                       "y": np.array(xy_r[:, 1]),
                       "t": np.array(t_r)},
            outputs = {"u_pred": u_pred_r},
            exact   = {"u_exact": u_exact_r},
            filename=f"predictions_heat2d_temporal_{label}.npz",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())
    run_parameter_transfer()
    run_temporal_transfer()
    print("\nAll 2-D heat transfer experiments complete.")


if __name__ == "__main__":
    main()
