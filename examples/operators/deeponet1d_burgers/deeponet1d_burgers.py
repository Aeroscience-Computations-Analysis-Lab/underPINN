"""1-D Dirichlet viscous Burgers — DeepONet (physics-informed, self-supervised).

Unlike the FNO/PINO examples, this DeepONet never sees a full-field ground
truth: the branch net's ``m`` sensors already sample the initial condition
exactly, so that supervision is free, and the interior is fit purely by
minimizing the autodiff PDE residual
(:class:`underPINN.pde.burgers_deeponet.DeepONetBurgersPDE`) at randomly
sampled collocation points — a different, older training philosophy from the
grid-residual PINO examples, included here for contrast.

Run directly or via the CLI:

    python examples/operators/deeponet1d_burgers/deeponet1d_burgers.py
    python -m underPINN run examples/operators/deeponet1d_burgers/config.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp
import optax

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.factory import build_model, network_config
from underPINN.pde.burgers_deeponet import DeepONetBurgersPDE
from underPINN.losses.operator_loss import DeepONetLoss
from underPINN.solver.operator import DeepONetSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.io import save_predictions
from underPINN.utils.metrics import relative_l2_error, print_errors
from underPINN.utils.operator_datagen import random_ic_1d, burgers1d_exact
from underPINN.postprocess.operators import plot_operator_loss, plot_prediction_1d


def _make_ic_pool(data_cfg, seed: int):
    m = cfg_get(data_cfg, "m", default=101)
    x_lo = cfg_get(data_cfg, "x_lo", default=-1.0)
    x_hi = cfg_get(data_cfg, "x_hi", default=1.0)
    n_pool = cfg_get(data_cfg, "n_ic_pool", default=2000)
    Lx = x_hi - x_lo

    rng = np.random.default_rng(seed)
    x_sensors = np.linspace(x_lo, x_hi, m).astype(np.float32)
    u_pool = np.stack([random_ic_1d(rng, m, Lx, periodic=False)
                       for _ in range(n_pool)])
    return jnp.array(u_pool), jnp.array(x_sensors)


def run_deeponet1d_burgers(cfg) -> dict:
    """Train the physics-informed DeepONet on 1-D Dirichlet Burgers."""
    tr = cfg.training
    seed = cfg_get(tr, "seed", default=0)
    out = cfg_get(cfg, "output", default=None)
    out_dir = cfg_get(out, "dir", default="outputs/deeponet1d_burgers") if out else "outputs/deeponet1d_burgers"
    os.makedirs(out_dir, exist_ok=True)

    net_cfg = network_config(cfg)
    model = build_model(net_cfg)

    u_pool, x_sensors = _make_ic_pool(cfg.data, seed)
    m = x_sensors.shape[0]

    nu = float(cfg.physics.nu)
    pde = DeepONetBurgersPDE(model, nu=nu)
    loss = DeepONetLoss(
        model, pde,
        ics_weight=cfg_get(cfg.loss, "ics_weight", default=20.0),
        bcs_weight=cfg_get(cfg.loss, "bcs_weight", default=1.0),
        periodic=bool(cfg_get(cfg.loss, "periodic", default=False)),
        rba=bool(cfg_get(cfg.loss, "rba", default=False)),
    )

    epochs = tr.epochs
    lr = tr.lr
    lr_alpha = cfg_get(tr, "lr_alpha", default=0.01)
    log_every = cfg_get(tr, "log_every", default=500)
    batch_size = cfg_get(tr, "batch_size", default=64)

    solver = DeepONetSolver(
        model, loss, lr=lr,
        pde_weight=cfg_get(cfg.loss, "pde_weight", default=1.0),
        pde_warmup_epochs=cfg_get(cfg.loss, "pde_warmup_epochs", default=0))
    solver.init(jax.random.PRNGKey(seed), jnp.ones((m,)), jnp.ones((2,)))

    n_bc = cfg_get(cfg.loss, "n_bc", default=100)
    t_hi = cfg_get(cfg.loss, "t_hi", default=1.0)
    x_lo = cfg_get(cfg.data, "x_lo", default=-1.0)
    x_hi = cfg_get(cfg.data, "x_hi", default=1.0)
    t_bc = jnp.linspace(0.0, t_hi, n_bc)
    xt_bc_lo = jnp.stack([jnp.full((n_bc,), x_lo), t_bc], axis=1)
    xt_bc_hi = jnp.stack([jnp.full((n_bc,), x_hi), t_bc], axis=1)

    callbacks = [ConsoleLogger(log_every=log_every)]
    tc = TrainingConfig(
        epochs=epochs, lr=lr,
        lr_schedule=optax.cosine_decay_schedule(lr, epochs, alpha=lr_alpha),
        batch_i=batch_size, log_every=log_every, callbacks=callbacks,
    )
    solver.train(u_pool, x_sensors, xt_bc_lo, xt_bc_hi,
                n_r=cfg_get(cfg.loss, "n_r", default=2000),
                x_range=(x_lo, x_hi), t_range=(0.0, t_hi), config=tc)

    # ── Evaluate against the Cole-Hopf exact solution for the classic
    #    single-mode IC u0(x) = -sin(pi x) (independent of the training pool).
    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(x_lo, x_hi, Nx_eval)
    t_eval = np.linspace(0.0, t_hi, Nt_eval)
    u_exact = burgers1d_exact(x_eval, t_eval, nu=nu, u0_mode=1)   # (Nx, Nt)

    u_ic_eval = jnp.array((-np.sin(np.pi * x_eval)).astype(np.float32))
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    xt_query = jnp.array(np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32))
    u_pred_flat = jax.vmap(
        lambda xt: model.apply(solver.params, u_ic_eval, xt)
    )(xt_query)
    u_pred_eval = np.array(u_pred_flat).reshape(Nx_eval, Nt_eval)

    rel_l2 = relative_l2_error(u_pred_eval, u_exact)
    print_errors(u_pred_eval, u_exact, label="Cole-Hopf exact (k=1 IC)")

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(solver.loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    plot_operator_loss(
        {"loss": solver.loss_hist, "ics": solver.ics_hist,
         "bcs": solver.bcs_hist, "res": solver.res_hist},
        os.path.join(out_dir, "loss.png"), title="DeepONet1D Burgers")

    mid_t = Nt_eval // 2
    plot_prediction_1d(x_eval, u_pred_eval[:, mid_t], u_exact[:, mid_t],
                       os.path.join(out_dir, "prediction.png"),
                       title=f"DeepONet1D Burgers — t={t_eval[mid_t]:.2f}")

    save_predictions(out_dir, coords={"x": x_eval, "t": t_eval},
                     outputs={"u_pred": u_pred_eval}, exact={"u_exact": u_exact})

    if (cfg_get(out, "save_params", default=True) if out else True):
        solver.save_checkpoint(out_dir, metadata={
            "problem": "deeponet1d_burgers",
            "network": net_cfg,
            "physics": {"nu": nu},
        })

    print(f"\nOutputs saved to: {out_dir}/")
    return {"params": solver.params, "loss_hist": solver.loss_hist,
           "rel_l2": float(rel_l2)}


if __name__ == "__main__":
    import sys
    import pathlib
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_deeponet1d_burgers(load_config(cfg_path))
