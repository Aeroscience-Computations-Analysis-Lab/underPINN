"""1-D diffusion (heat) equation, solved with jinns
(https://gitlab.com/mia_jinns/jinns) -- a third jinns comparison, alongside
``burgers_jinns.py`` and ``heat2d_jinns.py``, matching
``benchmarks/suite/physicsnemo/compare_underpinn_multi.py::run_diffusion``
as closely as jinns' loss/data-generator API allows:

  * same physics   : u_t = alpha*u_xx, alpha=0.01, domain x in [0,1],
                     t in [0,1], IC u(x,0)=sin(pi x), BC u(0,t)=u(1,t)=0 --
                     matches underPINN.pde.diffusion.DiffusionPDE's own
                     docstring's canonical test case (no pre-existing
                     example script to match against, unlike Burgers/Heat,
                     so this config is built directly from that docstring)
  * same network   : plain MLP, 3 hidden layers x 64 units, tanh (matches
                     underPINN's own choice for this problem -- no
                     Fourier-embedding architecture gap here)
  * same batching  : minibatched every step (batch_r=2048 of a 5000-point
                     interior pool, batch_i=256/batch_b=256 for IC/BC),
                     using jinns' domain_batch_size/initial_batch_size/
                     border_batch_size -- NOT full-batch like the Burgers
                     comparison. See heat2d_jinns.py's module docstring for
                     the (already-documented, not repeated here) real
                     minibatching-semantics difference between jinns'
                     sequential-chunk-with-reshuffle convention and
                     underPINN's fresh-random-draw-every-step convention.
  * same optimizer : plain Adam with a cosine decay schedule
  * same reference : scored against the identical exact solution
                     underPINN itself uses (sin(pi x) exp(-alpha pi^2 t))

jinns has no built-in pure-diffusion dynamic loss, but its built-in
``FisherKPP`` (D*laplacian + reaction term) reduces exactly to diffusion
when its reaction coefficients are zeroed (r=0, g=0) -- confirmed from
``FisherKPP.equation``'s source (the reaction term becomes
``-u*(0 - 0*u) = 0``) rather than assumed from the class name. This script
still writes an explicit ``DiffusionEquation(PDENonStatio)`` subclass
instead of reusing ``FisherKPP`` with zeroed reaction coefficients, purely
for readability (no ``eq_params.r``/``.g`` noise in the config); see
``heat2d_unsteady_jinns.py`` for the reuse-``FisherKPP`` version of this
same trick applied to the 2-D case.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python diffusion_jinns.py
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
from jinns.loss._operators import laplacian_rev                 # noqa: E402

from common import base_parser, jax_device_info, save_result, warn_if_cpu  # noqa: E402

ALPHA = 0.01
T_MAX = 1.0
N_R, N_IC, N_BC = 5000, 300, 300   # N_BC is per side -> nb = 2*N_BC
BATCH_R, BATCH_I, BATCH_B = 2048, 256, 256
IC_W, BC_W = 100.0, 100.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 64
LR = 1e-3


def u0(x):
    return jnp.sin(jnp.pi * x)


class DiffusionEquation(jinns.loss.PDENonStatio):
    """u_t - alpha*u_xx = 0, i.e. du_dt + Tmax*(-alpha*laplacian) -- same
    Tmax-scaling convention as jinns' own BurgersEquation/FisherKPP
    (confirmed from their source, see burgers_jinns.py's module docstring
    for the full derivation)."""

    alpha: float = eqx.field(default=ALPHA, static=True)

    def equation(self, t_x, u, params):
        u_ = lambda t_x: u(t_x, params)[0]
        du_dt = jax.grad(u_)(t_x)[0]
        lap = laplacian_rev(t_x, u, params)[..., None]
        return du_dt + self.Tmax * (-self.alpha * lap)


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
    # t_tilde in [0, 1] -- Tmax enters via DiffusionEquation, see above.
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey, n=N_R, nb=2 * N_BC, ni=N_IC, dim=1,
        min_pts=(0.0,), max_pts=(1.0,), tmin=0.0, tmax=1.0, method="uniform",
        domain_batch_size=BATCH_R, initial_batch_size=BATCH_I,
        border_batch_size=BATCH_B)

    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})

    diff_loss = DiffusionEquation(Tmax=T_MAX)
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0, initial_condition=IC_W, boundary_loss=BC_W)
    loss = jinns.loss.LossPDENonStatio(
        u=u_pinn, loss_weights=loss_weights, dynamic_loss=diff_loss,
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

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(0.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.exp(-ALPHA * np.pi ** 2 * TT)

    t_tilde = (TT / T_MAX).ravel().astype(np.float32)
    x_flat = XX.ravel().astype(np.float32)
    tx_query = jnp.stack([jnp.array(t_tilde), jnp.array(x_flat)], axis=1)

    def u_single(tx_i):
        return u_pinn(tx_i, final_params)[0]

    u_pred = np.array(jax.vmap(u_single)(tx_query)).reshape(Nx_eval, Nt_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_exact": rel_l2,
            "n_r": N_R, "n_ic": N_IC, "n_bc": 2 * N_BC,
            "batch_r": BATCH_R, "batch_i": BATCH_I, "batch_b": BATCH_B,
            "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX Diffusion1D, scored on the same exact "
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

    save_result("jinns_compare_diffusion1d", {
        "problem": "diffusion_1d", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
