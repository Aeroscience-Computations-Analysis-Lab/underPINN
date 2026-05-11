"""Runner for 3-D steady Hagen-Poiseuille pipe flow.

Expected config sections
------------------------
problem  : pipe_flow

network:
  layers : [3, 64, 64, 64, 64, 4]   # (x,y,z) → (u,v,w,p)

physics:
  Re    : 10.0
  R     : 0.5
  L     : 2.0
  U_max : 1.0

data:
  n_interior : 5000
  n_wall     : 1500
  n_inlet    : 400
  n_outlet   : 400

training:
  epochs    : 5000
  lr        : 1.0e-3
  lr_alpha  : 0.01
  batch_r   : 256
  batch_bc  : 128
  log_every : 500
  seed      : 0

loss:
  w_pde    : 1.0
  w_wall   : 100.0
  w_inlet  : 50.0
  w_outlet : 20.0

output:
  dir : outputs/pipe_flow
"""

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
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.utils.io import save_predictions


def run_pipe_flow(cfg) -> dict:
    """Train a PINN on 3-D steady Hagen-Poiseuille pipe flow."""
    # ── Unpack ────────────────────────────────────────────────────────────────
    ph   = cfg.physics
    tr   = cfg.training
    lw   = cfg.loss
    seed = cfg_get(tr,  "seed",    default=0)
    out  = cfg_get(cfg, "output",  default=None)
    out_dir = cfg_get(out, "dir",  default="outputs/pipe_flow") if out else "outputs/pipe_flow"
    os.makedirs(out_dir, exist_ok=True)

    Re, R, L, U_max = ph.Re, ph.R, ph.L, ph.U_max
    W_PDE    = cfg_get(lw, "w_pde",    default=1.0)
    W_WALL   = cfg_get(lw, "w_wall",   default=100.0)
    W_INLET  = cfg_get(lw, "w_inlet",  default=50.0)
    W_OUTLET = cfg_get(lw, "w_outlet", default=20.0)

    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    batch_r   = cfg_get(tr, "batch_r",   default=256)
    batch_bc  = cfg_get(tr, "batch_bc",  default=128)

    # ── Geometry + data ───────────────────────────────────────────────────────
    pipe  = Pipe(R=R, L=L)
    d     = cfg.data
    xyz_r   = jnp.array(pipe.sample_interior(cfg_get(d, "n_interior", default=5000), seed=seed))
    xyz_w   = jnp.array(pipe.sample_wall(    cfg_get(d, "n_wall",     default=1500), seed=seed+1))
    xyz_in  = jnp.array(pipe.sample_inlet(   cfg_get(d, "n_inlet",    default=400),  seed=seed+2))
    xyz_out = jnp.array(pipe.sample_outlet(  cfg_get(d, "n_outlet",   default=400),  seed=seed+3))

    def inlet_velocity(xyz):
        r2 = xyz[:, 1] ** 2 + xyz[:, 2] ** 2
        return U_max * (1.0 - r2 / R ** 2)

    # ── Model + PDE ───────────────────────────────────────────────────────────
    model = MLP(layers=cfg.network.layers)
    pde   = SteadyNS3DPDE(model, Re=Re)

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 3)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # ── JIT step ──────────────────────────────────────────────────────────────
    @jax.jit
    def step(params, state, xyz_r, xyz_w, xyz_in, xyz_out):
        def loss_fn(p):
            cont, mx, my, mz = pde.residual(p, xyz_r)
            pde_l  = (jnp.mean(cont**2) + jnp.mean(mx**2)
                      + jnp.mean(my**2) + jnp.mean(mz**2))

            out_w  = model.apply(p, xyz_w)
            wall_l = jnp.mean(out_w[:, 0]**2 + out_w[:, 1]**2 + out_w[:, 2]**2)

            out_in     = model.apply(p, xyz_in)
            u_in_exact = inlet_velocity(xyz_in)
            in_l       = (jnp.mean((out_in[:, 0] - u_in_exact)**2)
                          + jnp.mean(out_in[:, 1]**2)
                          + jnp.mean(out_in[:, 2]**2))

            out_out  = model.apply(p, xyz_out)
            outlet_l = jnp.mean(out_out[:, 3]**2)

            total = (W_PDE * pde_l + W_WALL * wall_l
                     + W_INLET * in_l + W_OUTLET * outlet_l)
            return total, (pde_l, wall_l, in_l, outlet_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Training loop ──────────────────────────────────────────────────────────
    N_r = xyz_r.shape[0]; N_w = xyz_w.shape[0]
    N_in = xyz_in.shape[0]; N_out = xyz_out.shape[0]
    loss_hist = []
    key = jax.random.PRNGKey(seed + 99)

    for ep in range(epochs):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        ir   = jax.random.choice(k1, N_r,   (batch_r,),              replace=False)
        iw   = jax.random.choice(k2, N_w,   (batch_bc,),             replace=False)
        iin  = jax.random.choice(k3, N_in,  (min(batch_bc, N_in),),  replace=False)
        iout = jax.random.choice(k4, N_out, (min(batch_bc, N_out),), replace=False)

        params, opt_state, total, (pl, wl, il, ol) = step(
            params, opt_state,
            xyz_r[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
        loss_hist.append(float(total))

        if ep % log_every == 0 or ep == epochs - 1:
            print(f"Epoch {ep:5d} | total {total:.3e} | pde {pl:.3e} "
                  f"| wall {wl:.3e} | inlet {il:.3e} | outlet {ol:.3e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # Predictions at interior collocation points + Hagen-Poiseuille exact
    uvwp_pred = model.apply(params, xyz_r)   # (N_r, 4)
    u_ex, v_ex, w_ex, p_ex = pde.exact_poiseuille(xyz_r, R=R, U_max=U_max, L=L)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(xyz_r[:, 0]),
                   "y": np.array(xyz_r[:, 1]),
                   "z": np.array(xyz_r[:, 2])},
        outputs = {"u_pred": np.array(uvwp_pred[:, 0]),
                   "v_pred": np.array(uvwp_pred[:, 1]),
                   "w_pred": np.array(uvwp_pred[:, 2]),
                   "p_pred": np.array(uvwp_pred[:, 3])},
        exact   = {"u_exact": np.array(u_ex),
                   "v_exact": np.array(v_ex),
                   "w_exact": np.array(w_ex),
                   "p_exact": np.array(p_ex)},
    )

    # Relative L² vs Poiseuille exact
    xyz_val = jnp.array(pipe.sample_interior(3000, seed=99))
    u_p, v_p, w_p, p_p = pde.exact_poiseuille(xyz_val, R=R, U_max=U_max, L=L)
    out_val = model.apply(params, xyz_val)
    def rel_l2(pred, exact):
        return float(jnp.linalg.norm(pred - exact) / (jnp.linalg.norm(exact) + 1e-10))
    errs = {k: rel_l2(out_val[:, i], v)
            for i, (k, v) in enumerate(zip("uvwp", [u_p, v_p, w_p, p_p]))}
    print("\nRel-L² vs Hagen-Poiseuille exact:", errs)

    # Loss plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(loss_hist, lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"3-D Pipe Flow  Re={Re}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": params, "loss_hist": loss_hist, "rel_l2": errs}
