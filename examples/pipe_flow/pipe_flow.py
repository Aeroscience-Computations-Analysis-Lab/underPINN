"""3-D Steady Hagen-Poiseuille Pipe Flow PINN.

Run directly or via the CLI:

    python examples/pipe_flow/pipe_flow.py              # uses pipe_flow.yaml
    python examples/pipe_flow/pipe_flow.py myconfig.yaml
    python -m underPINN run examples/pipe_flow/pipe_flow.yaml

Solves steady 3-D incompressible NS inside a cylinder; recovers the exact
parabolic Hagen-Poiseuille profile.

Network: (x, y, z) → (u, v, w, p)
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
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.pipe import Pipe
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice


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
    patience  = int(cfg_get(tr, "early_stopping_patience", default=600))
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

    # ── Training loop ─────────────────────────────────────────────────────────
    N_r = xyz_r.shape[0]; N_w = xyz_w.shape[0]
    N_in = xyz_in.shape[0]; N_out = xyz_out.shape[0]

    logger  = ConsoleLogger(log_every=log_every)
    stopper = EarlyStopping(patience=patience)
    loss_hist = []
    key = jax.random.PRNGKey(seed + 99)

    try:
        for ep in range(epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir   = safe_choice(k1, N_r,   batch_r)
            iw   = safe_choice(k2, N_w,   batch_bc)
            iin  = safe_choice(k3, N_in,  min(batch_bc, N_in))
            iout = safe_choice(k4, N_out, min(batch_bc, N_out))

            params, opt_state, total, (pl, wl, il, ol) = step(
                params, opt_state,
                xyz_r[ir], xyz_w[iw], xyz_in[iin], xyz_out[iout])
            loss_hist.append(float(total))

            logs = {"loss": float(total), "pde": float(pl),
                    "wall": float(wl), "inlet": float(il)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)
    except StopIteration:
        pass

    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    # Predictions at interior collocation points + Hagen-Poiseuille exact
    uvwp_pred = model.apply(params, xyz_r)
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

    # ── Model checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(params, out_dir, metadata={
        "problem": "pipe_flow",
        "network": {"type": "mlp", "layers": list(cfg.network.layers)},
        "physics": {"Re": float(Re), "R": float(R), "L": float(L)},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": params, "loss_hist": loss_hist, "rel_l2": errs}


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "pipe_flow.yaml")
    from underPINN.config.loader import load_config
    run_pipe_flow(load_config(cfg_path))
