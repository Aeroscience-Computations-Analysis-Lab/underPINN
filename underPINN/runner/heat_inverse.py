"""Runner for the 1-D heat equation — inverse problem (identify α).

Expected config sections
------------------------
problem  : heat_inverse   # or inverse_diffusion

physics:
  alpha_true  : 0.01    # true (hidden) diffusivity
  alpha_init  : 0.10    # initial guess  (log-parameterised; must be > 0)
  noise_level : 0.01    # Gaussian noise fraction on synthetic observations

data:
  T              : 1.0
  n_collocation  : 6000
  n_ic           : 300
  n_bc           : 300
  n_observations : 200   # sparse noisy measurement points

training:
  epochs                  : 8000
  lr                      : 1.0e-3
  lr_alpha                : 0.01   # cosine-decay final lr factor
  log_every               : 500
  early_stopping_patience : 600
  seed                    : 0

loss:
  ic_weight  : 100.0
  bc_weight  : 10.0
  data_weight: 100.0   # weight on sparse observations

output:
  dir: outputs/heat_inverse
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
from underPINN.pde.diffusion import DiffusionPDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.seed import set_seed


def run_heat_inverse(cfg) -> dict:
    """Identify thermal diffusivity α from sparse, noisy observations.

    The network weights and ``log(α)`` are optimised jointly.
    The log-parameterisation keeps α strictly positive throughout training.

    PDE: u_t = α u_xx,  x ∈ [0,1],  t ∈ [0,T]
    IC : u(x, 0) = sin(πx)
    BC : u(0, t) = u(1, t) = 0
    Exact: u(x, t) = sin(πx) exp(−α π² t)
    """
    # ── Unpack config ─────────────────────────────────────────────────────────
    ph  = cfg.physics
    tr  = cfg.training
    d   = cfg.data
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/heat_inverse") if out else "outputs/heat_inverse"
    os.makedirs(out_dir, exist_ok=True)

    alpha_true = float(cfg_get(ph, "alpha_true",  default=0.01))
    alpha_init = float(cfg_get(ph, "alpha_init",  default=0.10))
    noise      = float(cfg_get(ph, "noise_level", default=0.01))
    T          = float(cfg_get(d,  "T",           default=1.0))

    N_r   = int(cfg_get(d, "n_collocation",  default=6000))
    N_ic  = int(cfg_get(d, "n_ic",          default=300))
    N_bc  = int(cfg_get(d, "n_bc",          default=300))
    N_obs = int(cfg_get(d, "n_observations", default=200))

    IC_W   = float(cfg_get(lw, "ic_weight",   default=100.0))
    BC_W   = float(cfg_get(lw, "bc_weight",   default=10.0))
    DATA_W = float(cfg_get(lw, "data_weight", default=100.0))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    patience  = int(cfg_get(tr, "early_stopping_patience", default=600))
    seed      = int(cfg_get(tr, "seed",        default=0))

    print(f"Heat inverse:  α_true={alpha_true},  α_init={alpha_init},  "
          f"epochs={epochs},  N_obs={N_obs}")

    # ── Seeding ───────────────────────────────────────────────────────────────
    key = set_seed(seed)

    # ── Model + PDE ───────────────────────────────────────────────────────────
    model = MLP(layers=cfg_get(cfg.network, "layers", default=[2, 64, 64, 64, 1]))
    pde   = DiffusionPDE(model, alpha=alpha_true)

    nn_params  = model.init(key, jnp.ones((1, 2)))
    all_params = {
        "nn":        nn_params,
        "log_alpha": jnp.array(np.log(alpha_init), dtype=jnp.float32),
    }

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_sched),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(all_params)

    # ── Data ──────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed + 42)
    x_r  = jnp.array(rng.uniform(0.0, 1.0, N_r).astype(np.float32))
    t_r  = jnp.array(rng.uniform(0.0, T,   N_r).astype(np.float32))
    x_ic = jnp.array(np.linspace(0.0, 1.0, N_ic, dtype=np.float32))
    u_ic = jnp.sin(jnp.pi * x_ic)

    t_bc_half = rng.uniform(0.0, T, N_bc).astype(np.float32)
    x_bc = jnp.array(np.concatenate([np.zeros(N_bc, np.float32),
                                      np.ones(N_bc,  np.float32)]))
    t_bc = jnp.array(np.concatenate([t_bc_half, t_bc_half]))
    u_bc = jnp.zeros_like(x_bc)

    # Synthetic observations (noisy measurements of the exact solution)
    rng2 = np.random.default_rng(seed + 7)
    x_obs = rng2.uniform(0.05, 0.95, N_obs).astype(np.float32)
    t_obs = rng2.uniform(0.10, T,    N_obs).astype(np.float32)
    u_obs = (np.sin(np.pi * x_obs) * np.exp(-alpha_true * np.pi ** 2 * t_obs)
             + (noise * rng2.standard_normal(N_obs)).astype(np.float32))
    x_obs = jnp.array(x_obs);  t_obs = jnp.array(t_obs);  u_obs = jnp.array(u_obs)

    # ── JIT step ──────────────────────────────────────────────────────────────
    @jax.jit
    def step(params, state):
        def loss_fn(p):
            alpha  = jnp.exp(p["log_alpha"])
            res    = pde.residual(p["nn"], x_r, t_r, alpha=alpha)
            pde_l  = jnp.mean(res ** 2)
            ic_l   = jnp.mean((pde.u(p["nn"], x_ic, jnp.zeros_like(x_ic)) - u_ic) ** 2)
            bc_l   = jnp.mean((pde.u(p["nn"], x_bc, t_bc) - u_bc) ** 2)
            data_l = jnp.mean((pde.u(p["nn"], x_obs, t_obs) - u_obs) ** 2)
            total  = pde_l + IC_W * ic_l + BC_W * bc_l + DATA_W * data_l
            return total, (pde_l, ic_l, bc_l, data_l)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state     = optimizer.update(grads, state)
        params             = optax.apply_updates(params, updates)
        return params, state, loss, aux

    # ── Training loop ─────────────────────────────────────────────────────────
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

    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Results ───────────────────────────────────────────────────────────────
    alpha_final   = float(jnp.exp(all_params["log_alpha"]))
    rel_err_alpha = abs(alpha_final - alpha_true) / (alpha_true + 1e-12)
    print(f"\nIdentified α = {alpha_final:.6f}  |  true α = {alpha_true:.6f}  "
          f"|  error = {rel_err_alpha * 100:.2f}%")

    # ── Save predictions ──────────────────────────────────────────────────────
    nn_p     = all_params["nn"]
    u_pred_r = model.apply(nn_p, jnp.stack([x_r, t_r], axis=1))[:, 0]
    u_ex_r   = pde.exact(x_r, t_r, alpha=alpha_true)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(x_r), "t": np.array(t_r)},
        outputs = {"u_pred":           np.array(u_pred_r),
                   "alpha_identified": np.array([alpha_final], dtype=np.float32)},
        exact   = {"u_exact":          np.array(u_ex_r)},
    )

    np.save(os.path.join(out_dir, "loss_hist.npy"),  np.array(loss_hist))
    np.save(os.path.join(out_dir, "alpha_hist.npy"), np.array(alpha_hist))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(loss_hist)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total loss")
    axes[0].set_title("Training loss")

    axes[1].axhline(alpha_true, color="r",    ls="--", lw=2,
                    label=f"True  α = {alpha_true}")
    axes[1].axhline(alpha_init, color="gray", ls=":",  lw=1.5,
                    label=f"Init  α = {alpha_init}")
    axes[1].plot(alpha_hist, "b-", lw=1.5, label="Identified α")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("α")
    axes[1].set_title(f"Diffusivity convergence  (error {rel_err_alpha * 100:.2f}%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    print(f"\nOutputs saved to: {out_dir}/")

    return {"alpha_identified": alpha_final,
            "rel_err_alpha":    rel_err_alpha,
            "loss_hist":        loss_hist}


# Allow `problem: inverse_diffusion` to map to the same runner.
run_inverse_diffusion = run_heat_inverse
