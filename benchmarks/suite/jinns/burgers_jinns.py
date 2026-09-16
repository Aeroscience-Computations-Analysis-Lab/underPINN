"""1-D viscous Burgers, solved with jinns (https://gitlab.com/mia_jinns/jinns) --
a genuine third-party, JAX-native PINN framework comparison, matching
``benchmarks/suite/baselines/burgers_baselines.py`` /
``benchmarks/suite/physicsnemo/compare_underpinn.py`` as closely as
jinns' loss/data-generator API allows:

  * same physics   : u_t + u*u_x = nu*u_xx, nu=0.01, domain x in [-1,1],
                     t in [0, 1.5], IC u(x,0)=-sin(pi x), BC u(+-1,t)=0
  * same network   : 5 hidden layers x 64 units, tanh
  * same batch     : full-batch every step (jinns' CubicMeshPDENonStatio
                     uses the entire n/ni/nb pool every iteration unless a
                     `*_batch_size` override is given -- none is given here,
                     matching burgers_baselines.py's un-minibatched
                     convention)
  * same optimizer : plain Adam with a cosine decay schedule (jinns' own
                     example notebook for this problem uses its
                     natural-gradient optimizer `vanilla_ngd`; we use Adam
                     instead specifically to avoid confounding the
                     framework comparison with an optimizer-choice
                     difference, matching underPINN's own choice)
  * same reference : scored against the identical Cole-Hopf exact solution
                     underPINN itself uses (underPINN.utils.operator_datagen
                     .burgers1d_exact) -- not a separately-computed number.

One real API detail worth flagging because it is easy to get wrong (we
checked jinns' own ``BurgersEquation.equation`` source rather than assume):
jinns normalizes the PINN's time input to [0, 1] and expects ``Tmax`` as a
separate scalar multiplying the dynamic-loss residual, i.e. the network is
queried at ``t_tilde = t / Tmax``, not physical time directly. We therefore
build the domain generator over ``t_tilde in [0, 1]`` and pass
``Tmax=T_MAX`` to ``BurgersEquation`` and to every physical-time query
below, rather than the more natural-looking (but wrong) ``tmax=T_MAX``.

jinns' ``Dirichlet`` boundary condition is a soft penalty (squared network
output at the boundary, added to the loss) exactly like underPINN's own
boundary loss term -- not a hard architectural constraint -- confirmed by
reading ``jinns.loss.Dirichlet.equation_u``'s source, so this axis is
matched too.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python burgers_jinns.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "baselines"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import equinox as eqx                                          # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jinns                                                    # noqa: E402
import numpy as np                                              # noqa: E402
import optax                                                    # noqa: E402
from jax import random                                          # noqa: E402

from common import base_parser, jax_device_info, save_result, warn_if_cpu  # noqa: E402

from underPINN.utils.operator_datagen import burgers1d_exact    # noqa: E402
from underPINN.utils.metrics import relative_l2_error           # noqa: E402

NU = 0.01
T_MAX = 1.5
N_R, N_IC, N_BC = 20000, 200, 300   # N_BC is per side in our convention -> nb=2*N_BC
W_PDE, W_IC, W_BC = 1.0, 100.0, 10.0
LAYERS_HIDDEN, LAYER_SIZE = 5, 64
LR = 1e-3


def u0(x):
    return -jnp.sin(jnp.pi * x)


def run_jinns(epochs: int, seed: int) -> dict:
    key = random.PRNGKey(seed)

    eqx_list = ((eqx.nn.Linear, 2, LAYER_SIZE), (jax.nn.tanh,))
    for _ in range(LAYERS_HIDDEN - 1):
        eqx_list += ((eqx.nn.Linear, LAYER_SIZE, LAYER_SIZE), (jax.nn.tanh,))
    eqx_list += ((eqx.nn.Linear, LAYER_SIZE, 1),)

    key, subkey = random.split(key)
    u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio")

    key, subkey = random.split(key)
    # t_tilde in [0, 1] -- see module docstring; T_MAX enters via BurgersEquation.
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey, n=N_R, nb=2 * N_BC, ni=N_IC, dim=1,
        min_pts=(-1.0,), max_pts=(1.0,), tmin=0.0, tmax=1.0, method="uniform")

    nu = jnp.array(NU)
    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={"nu": nu})

    be_loss = jinns.loss.BurgersEquation(Tmax=T_MAX)
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=W_PDE, initial_condition=W_IC, boundary_loss=W_BC)
    loss = jinns.loss.LossPDENonStatio(
        u=u_pinn, loss_weights=loss_weights, dynamic_loss=be_loss,
        boundary_condition=jinns.loss.Dirichlet(),
        initial_condition_fun=u0, params=init_params)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    tx = optax.adam(sched)

    # Untimed warmup: a tiny n_iter call to trigger jinns' internal JIT
    # compilation, from the same freshly-initialized params/data used for
    # the real run below (so the timed call starts from an untrained state,
    # not from wherever the warmup call's few steps left it).
    jinns.solve(init_params=init_params, data=train_data, optimizer=tx,
               loss=loss, n_iter=2, verbose=False)

    t0 = time.perf_counter()
    (final_params, loss_values, loss_by_term, *_rest) = jinns.solve(
        init_params=init_params, data=train_data, optimizer=tx, loss=loss,
        n_iter=epochs, verbose=False)
    jax.block_until_ready(final_params.nn_params)
    wall = time.perf_counter() - t0
    final_loss = float(loss_values[-1])

    # ── score against the identical Cole-Hopf reference underPINN/PhysicsNeMo use ──
    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    u_exact = burgers1d_exact(x_eval, t_eval, nu=NU, u0_mode=1)

    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    t_tilde = (TT / T_MAX).ravel().astype(np.float32)
    x_flat = XX.ravel().astype(np.float32)
    tx_query = jnp.stack([jnp.array(t_tilde), jnp.array(x_flat)], axis=1)

    def u_single(tx_i):
        return u_pinn(tx_i, final_params)[0]

    u_pred = np.array(jax.vmap(u_single)(tx_query)).reshape(Nx_eval, Nt_eval)
    rel_l2 = float(relative_l2_error(u_pred, u_exact))

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_cole_hopf": rel_l2,
            "n_r": N_R, "n_ic": N_IC, "n_bc": 2 * N_BC, "layers_hidden": LAYERS_HIDDEN,
            "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX Burgers, scored on the same Cole-Hopf "
                     "reference as the underPINN/PhysicsNeMo comparison")
    ap.set_defaults(epochs=5000)
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    r = run_jinns(args.epochs, args.seed)
    print(f"jinns: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(Cole-Hopf)={r['rel_l2_vs_cole_hopf']:.4e}")

    save_result("jinns_compare_burgers", {
        "problem": "burgers_1d", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
