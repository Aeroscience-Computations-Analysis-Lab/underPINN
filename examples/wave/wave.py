"""1-D Wave Equation PINN.

Run directly or via the CLI:

    python examples/wave/wave.py                   # uses config.yaml
    python examples/wave/wave.py myconfig.yaml     # custom config
    python -m underPINN run examples/wave/config.yaml

IC: u(x,0) = sin(πx),  u_t(x,0) = 0    BC: u(±1,t) = 0
Exact: sin(πx) cos(cπt)
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
from underPINN.nn.mlp import FourierMLP, MLP
from underPINN.pde.wave import WavePDE
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.sampling import safe_choice
from underPINN.utils.restart import RestartManager


def run_wave(cfg) -> dict:
    """Train a PINN on the 1-D wave equation  u_tt = c² u_xx."""
    tr      = cfg.training
    seed    = cfg_get(tr, "seed",    default=0)
    out     = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir",   default="outputs/wave") if out else "outputs/wave"
    os.makedirs(out_dir, exist_ok=True)

    c         = cfg.physics.c
    T         = cfg.data.T
    epochs    = tr.epochs
    lr        = tr.lr
    lr_alpha  = cfg_get(tr, "lr_alpha",  default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    patience  = int(cfg_get(tr, "early_stopping_patience", default=600))

    N_r  = cfg_get(cfg.data, "n_collocation", default=6000)
    N_ic = cfg_get(cfg.data, "n_ic",          default=300)
    N_bc = cfg_get(cfg.data, "n_bc",          default=300)

    IC_W     = cfg_get(cfg.loss, "ic_weight",     default=100.0)
    IC_DOT_W = cfg_get(cfg.loss, "ic_dot_weight", default=100.0)
    BC_W     = cfg_get(cfg.loss, "bc_weight",     default=10.0)

    net_cfg   = cfg.network
    n_fourier = cfg_get(net_cfg, "n_fourier", default=16)
    sigma     = cfg_get(net_cfg, "sigma",     default=max(2.0, float(c) * np.pi))
    model = FourierMLP(layers=net_cfg.layers, n_fourier=n_fourier, sigma=sigma)
    pde   = WavePDE(model, c=c)

    rng = np.random.default_rng(seed)
    x_r = jnp.array(rng.uniform(-1, 1, N_r).astype(np.float32))
    t_r = jnp.array(rng.uniform(0, T, N_r).astype(np.float32))

    x_ic = jnp.array(np.linspace(-1, 1, N_ic, dtype=np.float32))
    u_ic = jnp.array(np.sin(np.pi * np.linspace(-1, 1, N_ic)).astype(np.float32))

    t_bc = rng.uniform(0, T, N_bc).astype(np.float32)
    x_bc = jnp.array(np.concatenate([np.full(N_bc, -1., np.float32),
                                      np.full(N_bc,  1., np.float32)]))
    t_bc = jnp.array(np.concatenate([t_bc, t_bc]))

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 2)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, state, x_r, t_r, x_ic, u_ic, x_bc, t_bc):
        def loss_fn(p):
            res   = pde.residual(p, x_r, t_r)
            pde_l = jnp.mean(res ** 2)
            ic_l  = jnp.mean((pde.u(p, x_ic, jnp.zeros_like(x_ic)) - u_ic) ** 2)
            ut    = pde.u_t(p, x_ic, jnp.zeros_like(x_ic))
            dot_l = jnp.mean(ut ** 2)
            bc_l  = jnp.mean(pde.u(p, x_bc, t_bc) ** 2)
            total = pde_l + IC_W * ic_l + IC_DOT_W * dot_l + BC_W * bc_l
            return total, (pde_l, ic_l, dot_l, bc_l)
        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    N_R, N_IC, N_BC = x_r.shape[0], x_ic.shape[0], x_bc.shape[0]
    bR = cfg_get(tr, "batch_r", default=2048)
    bI = cfg_get(tr, "batch_i", default=256)
    bB = cfg_get(tr, "batch_b", default=256)

    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    logger  = ConsoleLogger(log_every=log_every)
    stopper = EarlyStopping(patience=patience)
    key = jax.random.PRNGKey(seed + 99)

    try:
        for ep in range(start_ep, epochs):
            key, k1, k2, k3 = jax.random.split(key, 4)
            ir = safe_choice(k1, N_R,  bR)
            ii = safe_choice(k2, N_IC, bI)
            ib = safe_choice(k3, N_BC, bB)
            params, opt_state, total, (pl, il, dl, bl) = step(
                params, opt_state,
                x_r[ir], t_r[ir], x_ic[ii], u_ic[ii], x_bc[ib], t_bc[ib])
            loss_hist.append(float(total))
            logs = {"loss": float(total), "pde": float(pl),
                    "ic": float(il), "bc": float(bl)}
            logger.on_epoch_end(ep, logs)
            stopper.on_epoch_end(ep, logs)
            restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    pts_r    = jnp.stack([x_r, t_r], axis=1)
    u_pred_r = model.apply(params, pts_r)[:, 0]
    u_exact_r = jnp.sin(jnp.pi * x_r) * jnp.cos(c * jnp.pi * t_r)
    save_predictions(
        out_dir,
        coords  = {"x": np.array(x_r), "t": np.array(t_r)},
        outputs = {"u_pred": np.array(u_pred_r)},
        exact   = {"u_exact": np.array(u_exact_r)},
    )

    Nx, Nt = 200, 100
    x_plt = jnp.linspace(-1, 1, Nx)
    t_plt = jnp.linspace(0, T, Nt)
    XX, TT = jnp.meshgrid(x_plt, t_plt, indexing="ij")
    pts = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
    u_pred  = model.apply(params, pts)[:, 0].reshape(Nx, Nt)
    u_exact = jnp.sin(jnp.pi * XX) * jnp.cos(c * jnp.pi * TT)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, u, title in zip(axes, [np.array(u_exact), np.array(u_pred), np.array(u_exact - u_pred)],
                             ["Exact", "PINN", "Error"]):
        cf = ax.contourf(np.array(x_plt), np.array(t_plt), u.T, 40,
                         cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("t")
    fig.suptitle(f"Wave equation  c = {c}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_checkpoint(params, out_dir, metadata={
        "problem": "wave",
        "network": {"type": "fourier_mlp", "layers": list(net_cfg.layers),
                    "n_fourier": cfg_get(net_cfg, "n_fourier", default=16),
                    "sigma":     cfg_get(net_cfg, "sigma",     default=2.0)},
        "physics": {"c": float(c)},
    })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": params, "loss_hist": loss_hist}


if __name__ == "__main__":
    import sys, pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_wave(load_config(cfg_path))
