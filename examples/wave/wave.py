"""
Wave Equation — underPINN Example
==================================

Solves the 1-D wave equation using a Physics-Informed Neural Network:

    u_tt - c² u_xx = 0,    x ∈ [0, 1],  t ∈ [0, 1]

    IC  : u(x, 0)  = sin(πx)          (displacement)
          u_t(x, 0) = 0                (velocity)
    BC  : u(0, t)  = u(1, t) = 0      (fixed ends)

    Exact solution: u(x, t) = sin(πx) cos(c π t)

Key features demonstrated
--------------------------
- Hyperbolic (wave) PDE — two ICs (displacement AND velocity)
- FourierMLP for oscillatory space-time solutions
- Custom training loop with separate IC / IC-derivative / BC terms
- ConsoleLogger + EarlyStopping callbacks
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from underPINN.nn.mlp import FourierMLP
from underPINN.pde.wave import WavePDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping


# ---- Hyper-parameters ----
C      = 1.0    # wave speed
T_MAX  = 1.0    # time horizon
EPOCHS = 6000

IC_W     = 100.0   # weight for u(x,0) = sin(πx)
IC_DOT_W = 100.0   # weight for u_t(x,0) = 0
BC_W     =  10.0   # weight for wall BCs


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_data(N_r=8000, N_ic=300, N_bc=300, seed=42):
    rng = np.random.default_rng(seed)

    # Interior collocation
    x_r = rng.uniform(0.0, 1.0, N_r).astype(np.float32)
    t_r = rng.uniform(0.0, T_MAX, N_r).astype(np.float32)

    # Initial condition slice (t = 0)
    x_ic = np.linspace(0.0, 1.0, N_ic, dtype=np.float32)

    # Boundary: left wall (x=0) and right wall (x=1) at random times
    t_bc = rng.uniform(0.0, T_MAX, N_bc).astype(np.float32)
    x_bc = np.concatenate([np.zeros(N_bc, np.float32),
                            np.ones(N_bc,  np.float32)])
    t_bc = np.concatenate([t_bc, t_bc])

    return (
        jnp.array(x_r),  jnp.array(t_r),
        jnp.array(x_ic),
        jnp.array(x_bc), jnp.array(t_bc),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())
    print(f"Wave equation: c={C}, T={T_MAX}, epochs={EPOCHS}")

    # FourierMLP — encodes inputs as [sin(Bx), cos(Bx)] before MLP
    # sigma=2 gives frequencies tuned to the sin(πx)cos(πt) solution
    model = FourierMLP(layers=[2, 64, 64, 64, 1], n_fourier=16, sigma=2.0)
    pde   = WavePDE(model, c=C)

    key    = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 2)))

    schedule  = optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)

    x_r, t_r, x_ic, x_bc, t_bc = make_data()

    u_ic     = jnp.sin(jnp.pi * x_ic)    # u(x, 0) = sin(πx)
    u_ic_dot = jnp.zeros_like(x_ic)       # u_t(x, 0) = 0
    u_bc     = jnp.zeros_like(x_bc)       # u = 0 at walls

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            # PDE residual: u_tt - c²u_xx
            res     = pde.residual(p, x_r, t_r)
            pde_l   = jnp.mean(res ** 2)

            # IC displacement: u(x,0) = sin(πx)
            u_pred  = pde.u(p, x_ic, jnp.zeros_like(x_ic))
            ic_l    = jnp.mean((u_pred - u_ic) ** 2)

            # IC velocity: u_t(x,0) = 0  (requires a Jacobian)
            ut_pred  = pde.u_t(p, x_ic, jnp.zeros_like(x_ic))
            ic_dot_l = jnp.mean((ut_pred - u_ic_dot) ** 2)

            # BC: fixed ends
            u_bc_pred = pde.u(p, x_bc, t_bc)
            bc_l      = jnp.mean((u_bc_pred - u_bc) ** 2)

            total = pde_l + IC_W * ic_l + IC_DOT_W * ic_dot_l + BC_W * bc_l
            return total, (pde_l, ic_l, ic_dot_l, bc_l)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    logger  = ConsoleLogger(log_every=500)
    stopper = EarlyStopping(patience=500)
    loss_hist = []

    try:
        for ep in range(EPOCHS):
            params, opt_state, loss, (pde_l, ic_l, ic_dot_l, bc_l) = step(params, opt_state)
            loss_hist.append(float(loss))
            logs = {
                "loss": float(loss),
                "pde":  float(pde_l),
                "ic":   float(ic_l),
                "bc":   float(bc_l),
            }
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)
    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1]})

    # ---- Evaluate on grid ----
    Nx, Nt = 200, 200
    x_test = jnp.linspace(0.0, 1.0, Nx)
    t_test = jnp.linspace(0.0, T_MAX, Nt)
    XX, TT = jnp.meshgrid(x_test, t_test, indexing="ij")
    pts    = jnp.stack([XX.ravel(), TT.ravel()], axis=1)

    u_pred  = model.apply(params, pts)[:, 0].reshape(Nx, Nt)
    u_exact = pde.exact(XX.ravel(), TT.ravel()).reshape(Nx, Nt)

    rel_l2 = float(
        jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
        / jnp.sqrt(jnp.mean(u_exact ** 2))
    )
    max_err = float(jnp.max(jnp.abs(u_pred - u_exact)))
    print(f"\nRelative L2 error : {rel_l2:.4e}")
    print(f"Max absolute error: {max_err:.4e}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, field, title in zip(
        axes,
        [u_exact, u_pred, jnp.abs(u_pred - u_exact)],
        ["Exact  u(x,t)", "PINN  u(x,t)", f"|Error|  (Rel-L2={rel_l2:.2e})"],
    ):
        cf = ax.contourf(np.array(x_test), np.array(t_test),
                         np.array(field).T, 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    fig.suptitle(f"Wave equation  u_tt = c² u_xx  (c={C})")
    fig.tight_layout()
    fig.savefig("wave_solution.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(loss_hist)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Wave equation — training loss")
    fig2.tight_layout()
    fig2.savefig("wave_loss.png", dpi=150)
    plt.close(fig2)

    print("Plots saved: wave_solution.png, wave_loss.png")


if __name__ == "__main__":
    main()
