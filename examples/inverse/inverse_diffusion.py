"""
Inverse Problem: Diffusivity Identification — underPINN Example
================================================================

Identifies an unknown thermal diffusivity α from sparse, noisy measurements
of the diffusion (heat) equation solution:

    u_t = α u_xx,    x ∈ [0, 1],  t ∈ [0, 1]

    IC  : u(x, 0) = sin(πx)
    BC  : u(0, t) = u(1, t) = 0
    Exact: u(x, t) = sin(πx) exp(-α π² t)

Setup
------
- True α = 0.01
- Initial guess α₀ = 0.10  (10× wrong)
- 200 noisy observations from exact solution (1% Gaussian noise)

The NN weights and log(α) are optimised jointly with a single Adam step.
log-parameterisation ensures α stays strictly positive throughout training.

Key features demonstrated
--------------------------
- Inverse PINN: simultaneous NN + physics-parameter identification
- Joint parameter pytree {"nn": ..., "log_alpha": ...}
- Data loss term alongside PDE / IC / BC residuals
- α convergence plot alongside training loss
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.diffusion import DiffusionPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


# ---- Problem parameters ----
ALPHA_TRUE = 0.01   # true diffusivity
ALPHA_INIT = 0.10   # initial guess (10× too large)
T_MAX      = 1.0
EPOCHS     = 8000

# ---- Loss weights ----
IC_W   = 100.0
BC_W   =  10.0
DATA_W = 100.0   # sparse measurements carry a heavy weight


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_observations(N_obs=200, noise_frac=0.01, seed=7):
    """Sparse, noisy point measurements from the exact solution."""
    rng = np.random.default_rng(seed)
    x   = rng.uniform(0.05, 0.95, N_obs).astype(np.float32)
    t   = rng.uniform(0.10, T_MAX, N_obs).astype(np.float32)   # avoid t=0 (covered by IC)
    u   = np.sin(np.pi * x) * np.exp(-ALPHA_TRUE * np.pi ** 2 * t)
    u  += (noise_frac * rng.standard_normal(N_obs)).astype(np.float32)
    return jnp.array(x), jnp.array(t), jnp.array(u)


def make_collocation(N_r=6000, N_ic=300, N_bc=300, seed=42):
    rng = np.random.default_rng(seed)

    x_r = rng.uniform(0.0, 1.0, N_r).astype(np.float32)
    t_r = rng.uniform(0.0, T_MAX, N_r).astype(np.float32)

    x_ic = np.linspace(0.0, 1.0, N_ic, dtype=np.float32)

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
    print(f"Inverse diffusion: α_true={ALPHA_TRUE}, α_init={ALPHA_INIT}, epochs={EPOCHS}")

    model = MLP(layers=[2, 64, 64, 64, 1])
    pde   = DiffusionPDE(model, alpha=ALPHA_TRUE)   # self.alpha unused — we pass alpha explicitly

    # Joint parameter pytree: NN weights + log(α)
    nn_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    all_params = {
        "nn":        nn_params,
        "log_alpha": jnp.array(np.log(ALPHA_INIT), dtype=jnp.float32),
    }

    schedule  = optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(all_params)

    x_r, t_r, x_ic, x_bc, t_bc = make_collocation()
    x_obs, t_obs, u_obs         = make_observations()

    u_ic = jnp.sin(jnp.pi * x_ic)   # u(x, 0) = sin(πx)
    u_bc = jnp.zeros_like(x_bc)      # u = 0 at walls

    @jax.jit
    def step(all_params, opt_state):
        def loss_fn(p):
            nn_p  = p["nn"]
            alpha = jnp.exp(p["log_alpha"])   # positivity guaranteed

            # PDE: u_t = α u_xx
            res   = pde.residual(nn_p, x_r, t_r, alpha=alpha)
            pde_l = jnp.mean(res ** 2)

            # IC
            ic_l = jnp.mean((pde.u(nn_p, x_ic, jnp.zeros_like(x_ic)) - u_ic) ** 2)

            # BC
            bc_l = jnp.mean((pde.u(nn_p, x_bc, t_bc) - u_bc) ** 2)

            # Sparse observations
            data_l = jnp.mean((pde.u(nn_p, x_obs, t_obs) - u_obs) ** 2)

            total = pde_l + IC_W * ic_l + BC_W * bc_l + DATA_W * data_l
            return total, (pde_l, ic_l, bc_l, data_l)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        all_params = optax.apply_updates(all_params, updates)
        return all_params, opt_state, loss, aux

    logger  = ConsoleLogger(log_every=500)
    stopper = EarlyStopping(patience=600)

    loss_hist  = []
    alpha_hist = []

    try:
        for ep in range(EPOCHS):
            all_params, opt_state, loss, (pde_l, ic_l, bc_l, data_l) = step(
                all_params, opt_state
            )
            alpha_now = float(jnp.exp(all_params["log_alpha"]))
            loss_hist.append(float(loss))
            alpha_hist.append(alpha_now)

            logs = {"loss": float(loss), "pde": float(pde_l),
                    "ic": float(ic_l), "bc": float(bc_l)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)

            if ep % 500 == 0:
                print(f"  → α = {alpha_now:.6f}  (true = {ALPHA_TRUE:.6f})")

    except StopIteration:
        pass

    alpha_final = float(jnp.exp(all_params["log_alpha"]))
    rel_err_alpha = abs(alpha_final - ALPHA_TRUE) / ALPHA_TRUE
    print(f"\n{'='*45}")
    print(f"Identified α = {alpha_final:.6f}")
    print(f"True      α = {ALPHA_TRUE:.6f}")
    print(f"Relative error: {rel_err_alpha*100:.2f}%")

    # ---- Solution accuracy ----
    x_test = jnp.linspace(0.0, 1.0, 200)
    t_test = jnp.linspace(0.0, T_MAX, 200)
    XX, TT = jnp.meshgrid(x_test, t_test, indexing="ij")
    pts    = jnp.stack([XX.ravel(), TT.ravel()], axis=1)

    u_pred  = model.apply(all_params["nn"], pts)[:, 0].reshape(200, 200)
    u_exact = pde.exact(XX.ravel(), TT.ravel(), alpha=ALPHA_TRUE).reshape(200, 200)

    rel_l2 = float(
        jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
        / jnp.sqrt(jnp.mean(u_exact ** 2))
    )
    print(f"Solution Rel-L2 error: {rel_l2:.4e}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(loss_hist)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("Training loss")

    axes[1].axhline(ALPHA_TRUE, color="r", linestyle="--",
                    linewidth=2, label=f"True α = {ALPHA_TRUE}")
    axes[1].axhline(ALPHA_INIT, color="gray", linestyle=":",
                    linewidth=1.5, label=f"Init  α = {ALPHA_INIT}")
    axes[1].plot(alpha_hist, "b-", linewidth=1.5, label="Identified α")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("α")
    axes[1].set_title(f"Diffusivity convergence  (final error {rel_err_alpha*100:.2f}%)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("inverse_diffusion_training.png", dpi=150)
    plt.close(fig)

    # Solution comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    x_np = np.array(x_test)
    t_np = np.array(t_test)
    for ax, field, title in zip(
        axes2,
        [u_exact, u_pred, jnp.abs(u_pred - u_exact)],
        ["Exact  u(x,t)", "PINN  u(x,t)", f"|Error|  (Rel-L2={rel_l2:.2e})"],
    ):
        cf = ax.contourf(x_np, t_np, np.array(field).T, 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        # Mark observation locations on all panels
        ax.scatter(np.array(x_obs), np.array(t_obs),
                   c="white", s=4, alpha=0.5, label="obs")

    fig2.suptitle(f"Inverse diffusion  u_t = α u_xx  "
                  f"(α_true={ALPHA_TRUE}, α_ident={alpha_final:.4f})")
    fig2.tight_layout()
    fig2.savefig("inverse_diffusion_solution.png", dpi=150)
    plt.close(fig2)

    print("Plots saved: inverse_diffusion_training.png, inverse_diffusion_solution.png")

    # ---- Save predictions at collocation points (also saves identified α) ----
    nn_p     = all_params["nn"]
    u_pred_r = model.apply(nn_p, jnp.stack([x_r, t_r], axis=1))[:, 0]
    u_ex_r   = pde.exact(x_r, t_r, alpha=ALPHA_TRUE)
    save_predictions(
        ".",
        coords  = {"x": x_r, "t": t_r},
        outputs = {"u_pred": u_pred_r,
                   "alpha_identified": np.array([alpha_final], dtype=np.float32)},
        exact   = {"u_exact": u_ex_r},
    )


if __name__ == "__main__":
    main()
