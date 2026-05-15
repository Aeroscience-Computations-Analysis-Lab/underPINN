"""
1-D Heat Equation — Inverse Problem
=====================================

Identifies an unknown thermal diffusivity α from sparse, noisy measurements:

    u_t = α u_xx,    x ∈ [0, 1],  t ∈ [0, 1]
    IC  : u(x, 0) = sin(πx)
    BC  : u(0, t) = u(1, t) = 0
    Exact: u(x, t) = sin(πx) exp(−α π² t)

Setup
------
- True α = 0.01
- Initial guess α₀ = 0.10  (10× wrong)
- Configurable noisy observations from exact solution

The NN weights and log(α) are optimised jointly.
log-parameterisation keeps α strictly positive throughout training.

Run via
-------
    python -m underPINN run examples/heat/heat_inverse.yaml
    python examples/heat/inverse.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP
from underPINN.pde.diffusion import DiffusionPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.seed import set_seed


# ── Default hyper-parameters (overridden by runner when called via CLI) ────
ALPHA_TRUE  = 0.01
ALPHA_INIT  = 0.10
T_MAX       = 1.0
EPOCHS      = 8000
N_OBS       = 200
NOISE       = 0.01
IC_W        = 100.0
BC_W        = 10.0
DATA_W      = 100.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_observations(N_obs=N_OBS, noise_frac=NOISE, seed=7):
    rng = np.random.default_rng(seed)
    x   = rng.uniform(0.05, 0.95, N_obs).astype(np.float32)
    t   = rng.uniform(0.10, T_MAX, N_obs).astype(np.float32)
    u   = np.sin(np.pi * x) * np.exp(-ALPHA_TRUE * np.pi ** 2 * t)
    u  += (noise_frac * rng.standard_normal(N_obs)).astype(np.float32)
    return jnp.array(x), jnp.array(t), jnp.array(u)


def make_collocation(N_r=6000, N_ic=300, N_bc=300, seed=42):
    rng  = np.random.default_rng(seed)
    x_r  = rng.uniform(0.0, 1.0, N_r).astype(np.float32)
    t_r  = rng.uniform(0.0, T_MAX, N_r).astype(np.float32)
    x_ic = np.linspace(0.0, 1.0, N_ic, dtype=np.float32)
    t_bc = rng.uniform(0.0, T_MAX, N_bc).astype(np.float32)
    x_bc = np.concatenate([np.zeros(N_bc, np.float32), np.ones(N_bc, np.float32)])
    t_bc = np.concatenate([t_bc, t_bc])
    return (jnp.array(x_r), jnp.array(t_r),
            jnp.array(x_ic),
            jnp.array(x_bc), jnp.array(t_bc))


# ---------------------------------------------------------------------------
# Main training function (also called by the CLI runner)
# ---------------------------------------------------------------------------

def run_inverse(
    alpha_true=ALPHA_TRUE,
    alpha_init=ALPHA_INIT,
    epochs=EPOCHS,
    n_obs=N_OBS,
    noise=NOISE,
    n_r=6000, n_ic=300, n_bc=300,
    ic_w=IC_W, bc_w=BC_W, data_w=DATA_W,
    lr=1e-3, lr_alpha=0.01,
    log_every=500,
    patience=600,
    seed=0,
    out_dir="outputs/heat_inverse",
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    key = set_seed(seed)
    print(f"Inverse heat:  α_true={alpha_true},  α_init={alpha_init},  epochs={epochs}")

    model = MLP(layers=[2, 64, 64, 64, 1])
    pde   = DiffusionPDE(model, alpha=alpha_true)

    nn_params  = model.init(key, jnp.ones((1, 2)))
    all_params = {
        "nn":        nn_params,
        "log_alpha": jnp.array(np.log(alpha_init), dtype=jnp.float32),
    }

    schedule  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(all_params)

    x_r, t_r, x_ic, x_bc, t_bc = make_collocation(n_r, n_ic, n_bc, seed=seed + 42)
    x_obs, t_obs, u_obs         = make_observations(n_obs, noise, seed=seed + 7)
    u_ic = jnp.sin(jnp.pi * x_ic)
    u_bc = jnp.zeros_like(x_bc)

    @jax.jit
    def step(params, state):
        def loss_fn(p):
            alpha = jnp.exp(p["log_alpha"])
            res   = pde.residual(p["nn"], x_r, t_r, alpha=alpha)
            pde_l = jnp.mean(res ** 2)
            ic_l  = jnp.mean((pde.u(p["nn"], x_ic, jnp.zeros_like(x_ic)) - u_ic) ** 2)
            bc_l  = jnp.mean((pde.u(p["nn"], x_bc, t_bc) - u_bc) ** 2)
            data_l = jnp.mean((pde.u(p["nn"], x_obs, t_obs) - u_obs) ** 2)
            total  = pde_l + ic_w * ic_l + bc_w * bc_l + data_w * data_l
            return total, (pde_l, ic_l, bc_l, data_l)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state     = optimizer.update(grads, state)
        params             = optax.apply_updates(params, updates)
        return params, state, loss, aux

    logger  = ConsoleLogger(log_every=log_every)
    stopper = EarlyStopping(patience=patience)
    loss_hist, alpha_hist = [], []

    try:
        for ep in range(epochs):
            all_params, opt_state, loss, (pde_l, ic_l, bc_l, data_l) = step(
                all_params, opt_state)
            alpha_now = float(jnp.exp(all_params["log_alpha"]))
            loss_hist.append(float(loss))
            alpha_hist.append(alpha_now)

            logs = {"loss": float(loss), "pde": float(pde_l),
                    "ic": float(ic_l), "bc": float(bc_l)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)

            if ep % log_every == 0:
                print(f"  α = {alpha_now:.6f}  (true = {alpha_true:.6f})")
    except StopIteration:
        pass

    alpha_final   = float(jnp.exp(all_params["log_alpha"]))
    rel_err_alpha = abs(alpha_final - alpha_true) / alpha_true
    print(f"\nIdentified α = {alpha_final:.6f}  |  true α = {alpha_true:.6f}  "
          f"|  error = {rel_err_alpha*100:.2f}%")

    # ── Save predictions ──────────────────────────────────────────────────
    nn_p     = all_params["nn"]
    u_pred_r = model.apply(nn_p, jnp.stack([x_r, t_r], axis=1))[:, 0]
    u_ex_r   = pde.exact(x_r, t_r, alpha=alpha_true)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(x_r), "t": np.array(t_r)},
        outputs = {"u_pred": np.array(u_pred_r),
                   "alpha_identified": np.array([alpha_final], dtype=np.float32)},
        exact   = {"u_exact": np.array(u_ex_r)},
    )

    np.save(f"{out_dir}/loss_hist.npy", np.array(loss_hist))
    np.save(f"{out_dir}/alpha_hist.npy", np.array(alpha_hist))

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(loss_hist)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total loss")
    axes[0].set_title("Training loss")

    axes[1].axhline(alpha_true, color="r", ls="--", lw=2,
                    label=f"True α = {alpha_true}")
    axes[1].axhline(alpha_init, color="gray", ls=":", lw=1.5,
                    label=f"Init  α = {alpha_init}")
    axes[1].plot(alpha_hist, "b-", lw=1.5, label="Identified α")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("α")
    axes[1].set_title(f"Diffusivity convergence  (error {rel_err_alpha*100:.2f}%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOutputs saved to: {out_dir}/")

    return {"alpha_identified": alpha_final, "rel_err_alpha": rel_err_alpha,
            "loss_hist": loss_hist}


if __name__ == "__main__":
    run_inverse()
