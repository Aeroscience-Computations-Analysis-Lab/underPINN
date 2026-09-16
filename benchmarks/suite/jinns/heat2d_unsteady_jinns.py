"""2-D unsteady diffusion / heat equation, solved with jinns
(https://gitlab.com/mia_jinns/jinns) -- a fourth jinns comparison,
matching
``benchmarks/suite/physicsnemo/compare_underpinn_multi.py::run_heat2d_unsteady``
as closely as jinns' loss/data-generator API allows:

  * same physics   : u_t = alpha*(u_xx+u_yy), alpha=0.01, domain
                     (x,y) in [0,1]^2, t in [0,1], IC u(x,y,0)=sin(pi x)
                     sin(pi y), BC u=0 on all four edges -- matches
                     underPINN.pde.heat2d_unsteady.UnsteadyHeat2DPDE's own
                     docstring's canonical test case
  * same network   : plain MLP, 4 hidden layers x 64 units, tanh (matches
                     underPINN's own choice -- layers=[3,64,64,64,64,1],
                     from examples/transfer/heat2d_transfer.py's LAYERS)
  * same batching  : minibatched every step (batch_r=2048/8000,
                     batch_i=256/400, batch_b=256/1200), matching
                     underPINN's own config. See heat2d_jinns.py's module
                     docstring for the (already-documented, not repeated
                     here) real minibatching-semantics difference between
                     jinns' sequential-chunk-with-reshuffle convention and
                     underPINN's fresh-random-draw-every-step convention.
  * same optimizer : plain Adam with a cosine decay schedule
  * same reference : scored against the identical exact solution
                     underPINN itself uses (sin(pi x)sin(pi y)exp(-2 alpha
                     pi^2 t))

Unlike diffusion_jinns.py (which writes an explicit DiffusionEquation for
readability), this script reuses jinns' own **built-in** ``FisherKPP``
dynamic loss directly, with its reaction coefficients zeroed
(``r=0, g=0``). Confirmed from ``FisherKPP.equation``'s source, not
assumed from the class name: with r=g=0 the reaction term
``-u*(r - g*u)`` becomes identically zero, leaving exactly
``u_t = D*laplacian(u)`` -- the diffusion equation, with D=alpha.
``FisherKPP`` already supports an arbitrary spatial dimension via its
``dim_x`` field, so this needs no new DynamicLoss subclass at all -- the
most idiomatic way to express this particular equation in jinns as
shipped.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python heat2d_unsteady_jinns.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
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

ALPHA = 0.01
T_MAX = 1.0
N_R, N_IC, N_BC = 8000, 400, 300   # N_BC is per edge -> nb = 4*N_BC
BATCH_R, BATCH_I, BATCH_B = 2048, 256, 256
IC_W, BC_W = 100.0, 100.0
LAYERS_HIDDEN, LAYER_SIZE = 4, 64
LR = 1e-3


def u0(xy):
    return jnp.sin(jnp.pi * xy[..., 0]) * jnp.sin(jnp.pi * xy[..., 1])


def run_jinns(epochs: int, seed: int) -> dict:
    key = random.PRNGKey(seed)

    eqx_list = ((eqx.nn.Linear, 3, LAYER_SIZE), (jax.nn.tanh,))
    for _ in range(LAYERS_HIDDEN - 1):
        eqx_list += ((eqx.nn.Linear, LAYER_SIZE, LAYER_SIZE), (jax.nn.tanh,))
    eqx_list += ((eqx.nn.Linear, LAYER_SIZE, 1),)

    key, subkey = random.split(key)
    u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio")

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey, n=N_R, nb=4 * N_BC, ni=N_IC, dim=2,
        min_pts=(0.0, 0.0), max_pts=(1.0, 1.0), tmin=0.0, tmax=1.0,
        method="uniform",
        domain_batch_size=BATCH_R, initial_batch_size=BATCH_I,
        border_batch_size=BATCH_B)

    eq_params = {"D": jnp.array(ALPHA), "r": jnp.array(0.0), "g": jnp.array(0.0)}
    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params=eq_params)

    fisher_kpp_as_diffusion = jinns.loss.FisherKPP(Tmax=T_MAX, dim_x=2)
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0, initial_condition=IC_W, boundary_loss=BC_W)
    loss = jinns.loss.LossPDENonStatio(
        u=u_pinn, loss_weights=loss_weights, dynamic_loss=fisher_kpp_as_diffusion,
        boundary_condition=jinns.loss.Dirichlet(),
        initial_condition_fun=u0, params=init_params)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    tx = optax.adam(sched)

    jinns.solve(init_params=init_params, data=train_data, optimizer=tx,
               loss=loss, n_iter=2, verbose=False)

    t0 = time.perf_counter()
    (final_params, loss_values, loss_by_term, *_rest) = jinns.solve(
        init_params=init_params, data=train_data, optimizer=tx, loss=loss,
        n_iter=epochs, verbose=False)
    jax.block_until_ready(final_params.nn_params)
    wall = time.perf_counter() - t0
    final_loss = float(loss_values[-1])

    N_eval, Nt_eval = 41, 21
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, YY, TT = np.meshgrid(x_eval, y_eval, t_eval, indexing="ij")
    u_exact = (np.sin(np.pi * XX) * np.sin(np.pi * YY)
              * np.exp(-2.0 * ALPHA * np.pi ** 2 * TT))

    t_tilde = (TT / T_MAX).ravel().astype(np.float32)
    txy_query = jnp.stack([jnp.array(t_tilde),
                           jnp.array(XX.ravel(), "f4"),
                           jnp.array(YY.ravel(), "f4")], axis=1)

    def u_single(txy_i):
        return u_pinn(txy_i, final_params)[0]

    u_pred = np.array(jax.vmap(u_single)(txy_query)).reshape(N_eval, N_eval, Nt_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_exact": rel_l2,
            "n_r": N_R, "n_ic": N_IC, "n_bc": 4 * N_BC,
            "batch_r": BATCH_R, "batch_i": BATCH_I, "batch_b": BATCH_B,
            "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX Heat2D-unsteady, scored on the same exact "
                     "reference as the underPINN/PhysicsNeMo comparison")
    ap.set_defaults(epochs=5000)
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    r = run_jinns(args.epochs, args.seed)
    print(f"jinns: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(exact)={r['rel_l2_vs_exact']:.4e}")

    save_result("jinns_compare_heat2d_unsteady", {
        "problem": "heat2d_unsteady", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
