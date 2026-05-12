"""
Unsteady 3-D Pipe Flow — Transfer Learning
===========================================

Cross-section formulation: (y, z, t) → u

PDE:   u_t = G + ν(u_yy + u_zz),   r < R,  t ∈ [0, T]
       G   = 4 ν U_max / R²          (pressure-gradient forcing)
IC :   u(y, z, 0) = 0               (fluid at rest)
BC :   u = 0  at  r = R             (no-slip)

Exact: Bessel series (Stokes starting flow)
       u(r,t) = U_max(1−r²/R²) − 8U_max Σ J₀(αₙr/R)/(αₙ³J₁(αₙ)) exp(−αₙ²νt/R²)

════════════════════════════════════════════════════════
Strategy 1 — Re Transfer  (ν = 1/Re)
════════════════════════════════════════════════════════
Source : Re = 10, t ∈ [0, 3]  →  2 500 epochs  (fast transient, τ = R²Re = 2.5)
Target : Re = 50, t ∈ [0, 3]                    (slow transient, τ = 12.5)
  Transfer : warm start         →  2 000 fine-tune epochs (lr = 3e-4)
  Scratch  : random init        →  4 500 epochs            (lr = 1e-3)

════════════════════════════════════════════════════════
Strategy 2 — Temporal Transfer  (extend time horizon)
════════════════════════════════════════════════════════
Re = 10 throughout.
Phase 1 : t ∈ [0, T1=1.0]    →  2 500 epochs
Phase 2 : t ∈ [0, T2=3.0]
  Transfer : warm start         →  2 000 fine-tune epochs (lr = 3e-4)
  Scratch  : random init        →  4 500 epochs            (lr = 1e-3)

Outputs
-------
pipe_unsteady_re_loss.png         — strategy 1 convergence
pipe_unsteady_re_profiles.png     — strategy 1 radial profiles vs exact
pipe_unsteady_re_spacetime.png    — strategy 1 space-time comparison
pipe_unsteady_time_loss.png       — strategy 2 convergence
pipe_unsteady_time_profiles.png   — strategy 2 snapshot profiles vs exact
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.pipe_flow_unsteady import UnsteadyPipeFlowPDE
from underPINN.utils.io import save_predictions
from underPINN.utils.sampling import safe_choice


# ── Problem parameters ────────────────────────────────────────────────────────
R     = 0.5
U_MAX = 1.0

# ── Loss weights ──────────────────────────────────────────────────────────────
W_PDE = 1.0
W_IC  = 100.0
W_BC  = 50.0

# ── Training batch sizes ──────────────────────────────────────────────────────
BATCH_R  = 512
BATCH_IC = 200
BATCH_BC = 200

LOG_EVERY = 500

LAYERS = [3, 64, 64, 64, 64, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _disk_yz(n: int, seed: int) -> np.ndarray:
    """Uniform (y, z) inside disk of radius R (rejection from bounding square)."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(a) for a in out) < n:
        y = rng.uniform(-R, R, 4 * n).astype(np.float32)
        z = rng.uniform(-R, R, 4 * n).astype(np.float32)
        k = y ** 2 + z ** 2 <= R ** 2
        out.append(np.column_stack([y[k], z[k]]))
    return np.concatenate(out)[:n]


def make_data(T: float, N_r: int = 6000, N_ic: int = 600,
              N_bc: int = 800, seed: int = 0):
    """Generate all training arrays for time horizon [0, T]."""
    rng = np.random.default_rng(seed)

    yz_r = _disk_yz(N_r, seed)
    t_r  = rng.uniform(0, T, N_r).astype(np.float32)

    yz_ic = _disk_yz(N_ic, seed + 1)

    theta  = rng.uniform(0, 2 * np.pi, N_bc).astype(np.float32)
    yz_bc  = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    t_bc   = rng.uniform(0, T, N_bc).astype(np.float32)

    return (jnp.array(yz_r),  jnp.array(t_r),
            jnp.array(yz_ic),
            jnp.array(yz_bc), jnp.array(t_bc))


# ─────────────────────────────────────────────────────────────────────────────
# Training infrastructure
# ─────────────────────────────────────────────────────────────────────────────

def _make_step(pde, optimizer):
    """Build a JIT-compiled gradient step (captures pde and optimizer)."""

    @jax.jit
    def step(params, opt_state, yz_r, t_r, yz_ic, yz_bc, t_bc):
        def loss_fn(p):
            pde_l = jnp.mean(pde.residual(p, yz_r, t_r) ** 2)

            t_zero = jnp.zeros(yz_ic.shape[0])
            ic_l   = jnp.mean(pde.u(p, yz_ic, t_zero) ** 2)

            bc_l   = jnp.mean(pde.u(p, yz_bc, t_bc) ** 2)

            total  = W_PDE * pde_l + W_IC * ic_l + W_BC * bc_l
            return total, (pde_l, ic_l, bc_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state  = optimizer.update(grads, opt_state)
        params              = optax.apply_updates(params, updates)
        return params, opt_state, total, aux

    return step


def run_training(pde, data, epochs: int, lr: float,
                 init_params=None, seed: int = 0, label: str = ""):
    """Train the model, optionally warm-starting from *init_params*.

    Parameters
    ----------
    pde         : UnsteadyPipeFlowPDE instance (owns model + physics)
    data        : tuple from make_data()
    epochs      : number of gradient steps
    lr          : peak learning rate
    init_params : if given, skip random init (transfer learning warm start)
    label       : short string for console progress lines
    """
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
    step_fn   = _make_step(pde, optimizer)

    N_r, N_ic, N_bc = yz_r.shape[0], yz_ic.shape[0], yz_bc.shape[0]
    key = jax.random.PRNGKey(seed + 99)

    loss_hist = []
    for ep in range(epochs):
        key, k1, k2, k3 = jax.random.split(key, 4)
        ir = safe_choice(k1, N_r,  BATCH_R)
        ii = safe_choice(k2, N_ic, BATCH_IC)
        ib = safe_choice(k3, N_bc, BATCH_BC)

        params, opt_state, total, (pl, il, bl) = step_fn(
            params, opt_state,
            yz_r[ir], t_r[ir],
            yz_ic[ii],
            yz_bc[ib], t_bc[ib],
        )
        loss_hist.append(float(total))

        if ep % LOG_EVERY == 0 or ep == epochs - 1:
            tag = f"[{label}] " if label else ""
            print(f"{tag}Epoch {ep:5d} | total {total:.3e} | "
                  f"pde {pl:.3e} | ic {il:.3e} | bc {bl:.3e}")

    return params, loss_hist


def rel_l2(model, params, pde, yz_val, t_val: float) -> float:
    """Relative L² error vs exact Bessel solution at time t_val."""
    u_ex   = pde.exact(yz_val, t_val)
    yzt    = jnp.concatenate(
        [yz_val, jnp.full((yz_val.shape[0], 1), t_val)], axis=1)
    u_pred = np.array(model.apply(params, yzt)[:, 0])
    return float(np.linalg.norm(u_pred - u_ex)
                 / (np.linalg.norm(u_ex) + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def _eval_radial(model, params, r_arr, t_val: float):
    """Evaluate u along the y-axis (z=0) at a fixed time."""
    Nr  = len(r_arr)
    yz  = jnp.column_stack([jnp.array(r_arr), jnp.zeros(Nr)])
    yzt = jnp.concatenate([yz, jnp.full((Nr, 1), t_val)], axis=1)
    return np.array(model.apply(params, yzt)[:, 0])


def _exact_radial(pde, r_arr, t_val: float):
    """Exact u along the y-axis (z=0) at a fixed time."""
    yz = np.column_stack([np.array(r_arr), np.zeros(len(r_arr))])
    return pde.exact(yz, t_val)


def _exact_spacetime(pde, r_arr, t_arr):
    """Exact u(r, t) on a 2D grid for space-time plots."""
    yz = np.column_stack([np.array(r_arr), np.zeros(len(r_arr))])
    return np.array([pde.exact(yz, float(t)) for t in t_arr])   # (Nt, Nr)


def _pred_spacetime(model, params, r_arr, t_arr):
    """PINN u(r, t) on a 2D grid."""
    Nr, Nt = len(r_arr), len(t_arr)
    RR, TT = np.meshgrid(r_arr, t_arr, indexing="ij")          # (Nr, Nt)
    yz  = np.column_stack([RR.ravel(), np.zeros(Nr * Nt, dtype=np.float32)])
    yzt = jnp.concatenate([jnp.array(yz),
                           jnp.array(TT.ravel()[:, None])], axis=1)
    return np.array(model.apply(params, yzt)[:, 0]).reshape(Nr, Nt)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Re Transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_re_transfer():
    print("\n" + "=" * 60)
    print("Strategy 1: Re Transfer  Re=10 → Re=50")
    print("=" * 60)

    T     = 3.0
    N_SRC = 2500
    N_TF  = 2000
    N_SC  = N_SRC + N_TF

    data_src = make_data(T, seed=0)
    data_tgt = make_data(T, seed=1)

    # ── Source: Re = 10 ───────────────────────────────────────────────────────
    print(f"\n[Source]   Re=10,  {N_SRC} epochs  (τ = {R**2*10:.1f})")
    model_src = MLP(layers=LAYERS)
    pde_src   = UnsteadyPipeFlowPDE(model_src, Re=10.0, R=R, U_max=U_MAX)
    src_params, _ = run_training(pde_src, data_src, N_SRC, lr=1e-3,
                                 seed=0, label="Source")

    # ── Transfer: Re = 50, warm start ─────────────────────────────────────────
    print(f"\n[Transfer] Re=50, {N_TF} fine-tune epochs  (τ = {R**2*50:.1f})")
    model_tf = MLP(layers=LAYERS)
    pde_tf   = UnsteadyPipeFlowPDE(model_tf, Re=50.0, R=R, U_max=U_MAX)
    tf_params, hist_tf = run_training(
        pde_tf, data_tgt, N_TF, lr=3e-4,
        init_params=src_params, seed=1, label="Transfer")

    # ── Scratch: Re = 50 ──────────────────────────────────────────────────────
    print(f"\n[Scratch]  Re=50, {N_SC} epochs from random init")
    model_sc = MLP(layers=LAYERS)
    pde_sc   = UnsteadyPipeFlowPDE(model_sc, Re=50.0, R=R, U_max=U_MAX)
    sc_params, hist_sc = run_training(
        pde_sc, data_tgt, N_SC, lr=1e-3, seed=2, label="Scratch")

    # ── Validation ────────────────────────────────────────────────────────────
    yz_val = jnp.array(_disk_yz(2000, seed=99))
    for t_check in [1.0, T]:
        err_tf = rel_l2(model_tf, tf_params, pde_tf, yz_val, t_check)
        err_sc = rel_l2(model_sc, sc_params, pde_sc, yz_val, t_check)
        print(f"Rel-L² at t={t_check:.1f}: Transfer {err_tf:.3e}  |  "
              f"Scratch {err_sc:.3e}")

    # ── Save predictions at collocation points ────────────────────────────────
    yz_r, t_r, _, _, _ = data_tgt
    for label, model_, params_ in [("transfer", model_tf, tf_params),
                                   ("scratch",  model_sc, sc_params)]:
        yzt = jnp.concatenate([yz_r, t_r[:, None]], axis=1)
        u_pred = np.array(model_.apply(params_, yzt)[:, 0])
        u_ex   = pde_tf.exact(np.array(yz_r), float(t_r.mean()))  # approx exact at mean t
        save_predictions(
            ".",
            coords  = {"y": np.array(yz_r[:, 0]),
                       "z": np.array(yz_r[:, 1]),
                       "t": np.array(t_r)},
            outputs = {"u_pred": u_pred},
            filename=f"predictions_re_transfer_{label}.npz",
        )

    # ── Loss convergence plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(hist_tf, "b-",  lw=1.5, label=f"Transfer (warm, {N_TF} ep)")
    ax.semilogy(hist_sc[N_SRC:], "r--", lw=1.5,
                label=f"Scratch Re=50 (same ep range)")
    ax.axhline(hist_sc[N_SRC - 1], color="gray", ls=":", lw=1,
               label=f"Scratch after {N_SRC} ep")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Re Transfer  Re=10 → 50\nFine-tuning phase")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(hist_sc[:N_SRC], "r-",  lw=1.2, label=f"Scratch (first {N_SRC} ep)")
    ax.axhline(hist_tf[0], color="b", ls="--", lw=1.2,
               label=f"Transfer start loss = {hist_tf[0]:.2e}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("Scratch vs Transfer starting loss")
    ax.legend(fontsize=8)

    fig.suptitle("Unsteady Pipe Flow — Re Transfer")
    fig.tight_layout()
    fig.savefig("pipe_unsteady_re_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_unsteady_re_loss.png")

    # ── Radial profiles at multiple times ─────────────────────────────────────
    Nr    = 80
    r_arr = np.linspace(0, R, Nr, dtype=np.float32)
    t_snaps = [0.5, 1.5, T]

    fig, axes = plt.subplots(1, len(t_snaps), figsize=(14, 4))
    for ax, ts in zip(axes, t_snaps):
        u_ex = _exact_radial(pde_tf, r_arr, ts)
        u_tf = _eval_radial(model_tf, tf_params, r_arr, ts)
        u_sc = _eval_radial(model_sc, sc_params, r_arr, ts)

        ax.plot(r_arr, u_ex, "k-",  lw=2.0, label="Exact")
        ax.plot(r_arr, u_tf, "b--", lw=1.8, label="Transfer")
        ax.plot(r_arr, u_sc, "r:",  lw=1.8, label="Scratch")
        ax.set_title(f"t = {ts:.1f}")
        ax.set_xlabel("r"); ax.set_ylabel("u")
        ax.grid(ls="--", alpha=0.4)
        if ts == t_snaps[0]:
            ax.legend(fontsize=8)

    fig.suptitle(f"Unsteady Pipe Flow Re=50: Radial profiles (Re Transfer)")
    fig.tight_layout()
    fig.savefig("pipe_unsteady_re_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_unsteady_re_profiles.png")

    # ── Space-time contours ───────────────────────────────────────────────────
    Nr, Nt = 60, 80
    r_grid = np.linspace(0, R, Nr, dtype=np.float32)
    t_grid = np.linspace(0, T, Nt, dtype=np.float32)

    u_ex   = _exact_spacetime(pde_tf, r_grid, t_grid).T     # (Nr, Nt)
    u_tf_g = _pred_spacetime(model_tf, tf_params, r_grid, t_grid)
    u_sc_g = _pred_spacetime(model_sc, sc_params, r_grid, t_grid)

    vmin, vmax = 0.0, U_MAX
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Exact", f"Transfer Re=50", f"Scratch Re=50"]
    for ax, u, title in zip(axes, [u_ex, u_tf_g, u_sc_g], titles):
        cf = ax.contourf(t_grid, r_grid, u, 40, cmap="viridis",
                         vmin=vmin, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="u")
        ax.set_xlabel("t"); ax.set_ylabel("r")
        ax.set_title(title)

    fig.suptitle("Unsteady Pipe Flow: space-time  u(r, t)  — Re Transfer")
    fig.tight_layout()
    fig.savefig("pipe_unsteady_re_spacetime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_unsteady_re_spacetime.png")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Temporal Transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_temporal_transfer():
    print("\n" + "=" * 60)
    print("Strategy 2: Temporal Transfer  t∈[0, 1] → t∈[0, 3]")
    print("=" * 60)

    RE    = 10.0
    T1, T2 = 1.0, 3.0
    N_P1  = 2500
    N_P2  = 2000
    N_SC  = N_P1 + N_P2

    data_p1 = make_data(T1, seed=0)
    data_p2 = make_data(T2, seed=0)

    # ── Phase 1: short horizon ────────────────────────────────────────────────
    print(f"\n[Phase 1]  Re={RE}, t∈[0,{T1}], {N_P1} epochs")
    model_p1 = MLP(layers=LAYERS)
    pde_p1   = UnsteadyPipeFlowPDE(model_p1, Re=RE, R=R, U_max=U_MAX)
    p1_params, _ = run_training(pde_p1, data_p1, N_P1, lr=1e-3,
                                 seed=0, label="Phase1")

    # ── Transfer: extend to [0, T2] ───────────────────────────────────────────
    print(f"\n[Transfer] Re={RE}, t∈[0,{T2}], {N_P2} fine-tune epochs (warm start)")
    model_p2 = MLP(layers=LAYERS)
    pde_p2   = UnsteadyPipeFlowPDE(model_p2, Re=RE, R=R, U_max=U_MAX)
    p2_params, hist_p2 = run_training(
        pde_p2, data_p2, N_P2, lr=3e-4,
        init_params=p1_params, seed=1, label="Transfer")

    # ── Scratch: full horizon ─────────────────────────────────────────────────
    print(f"\n[Scratch]  Re={RE}, t∈[0,{T2}], {N_SC} epochs from random init")
    model_sc = MLP(layers=LAYERS)
    pde_sc   = UnsteadyPipeFlowPDE(model_sc, Re=RE, R=R, U_max=U_MAX)
    sc_params, hist_sc = run_training(
        pde_sc, data_p2, N_SC, lr=1e-3, seed=2, label="Scratch")

    # ── Validation at extended time (not in phase-1 training) ─────────────────
    yz_val = jnp.array(_disk_yz(2000, seed=99))
    t_ext  = T1 + 0.5 * (T2 - T1)          # midpoint of extended region
    err_tf = rel_l2(model_p2, p2_params, pde_p2, yz_val, t_ext)
    err_sc = rel_l2(model_sc, sc_params, pde_sc, yz_val, t_ext)
    print(f"\nRel-L² at t={t_ext:.2f} (extended): "
          f"Transfer {err_tf:.3e}  |  Scratch {err_sc:.3e}")

    # ── Save predictions at collocation points ────────────────────────────────
    yz_r, t_r, _, _, _ = data_p2
    for label, model_, params_ in [("transfer", model_p2, p2_params),
                                   ("scratch",  model_sc, sc_params)]:
        yzt = jnp.concatenate([yz_r, t_r[:, None]], axis=1)
        u_pred = np.array(model_.apply(params_, yzt)[:, 0])
        save_predictions(
            ".",
            coords  = {"y": np.array(yz_r[:, 0]),
                       "z": np.array(yz_r[:, 1]),
                       "t": np.array(t_r)},
            outputs = {"u_pred": u_pred},
            filename=f"predictions_time_transfer_{label}.npz",
        )

    # ── Loss convergence plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(hist_p2, "b-",  lw=1.5,
                label=f"Temporal transfer (warm, {N_P2} ep on [0,{T2}])")
    ax.semilogy(hist_sc[N_P1:], "r--", lw=1.5,
                label=f"Scratch [0,{T2}] (same ep range)")
    ax.set_xlabel("Fine-tuning epoch on extended domain")
    ax.set_ylabel("Total loss")
    ax.set_title(f"Temporal Transfer — Unsteady Pipe Flow  Re={RE}\n"
                 f"Phase-1 on [0,{T1}], Phase-2 extended to [0,{T2}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("pipe_unsteady_time_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_unsteady_time_loss.png")

    # ── Snapshot profiles ─────────────────────────────────────────────────────
    Nr    = 80
    r_arr = np.linspace(0, R, Nr, dtype=np.float32)
    t_snaps = [0.5, T1, 2.0, T2]

    fig, axes = plt.subplots(1, len(t_snaps), figsize=(14, 4))
    for ax, ts in zip(axes, t_snaps):
        u_ex = _exact_radial(pde_p2, r_arr, ts)
        u_tf = _eval_radial(model_p2, p2_params, r_arr, ts)
        u_sc = _eval_radial(model_sc, sc_params, r_arr, ts)
        u_p1 = _eval_radial(model_p1, p1_params, r_arr, ts)

        ax.plot(r_arr, u_ex, "k-",  lw=2.0, label="Exact")
        ax.plot(r_arr, u_tf, "b--", lw=1.8, label="Transfer")
        ax.plot(r_arr, u_sc, "r:",  lw=1.8, label="Scratch")
        if ts <= T1:
            ax.plot(r_arr, u_p1, "g-.", lw=1.2, label="Phase-1 (src)")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"t = {ts:.1f}" + ("  [ext]" if ts > T1 else ""))
        ax.set_xlabel("r"); ax.set_ylabel("u")
        ax.grid(ls="--", alpha=0.4)
        if ts == t_snaps[0]:
            ax.legend(fontsize=7)

    fig.suptitle(f"Unsteady Pipe Flow Re={RE}: Temporal Transfer snapshot profiles")
    fig.tight_layout()
    fig.savefig("pipe_unsteady_time_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pipe_unsteady_time_profiles.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("JAX devices:", jax.devices())
    run_re_transfer()
    run_temporal_transfer()
    print("\nAll unsteady pipe flow transfer learning experiments complete.")


if __name__ == "__main__":
    main()
