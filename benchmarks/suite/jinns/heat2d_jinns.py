"""2-D steady heat / Poisson equation, solved with jinns
(https://gitlab.com/mia_jinns/jinns) -- a second, independent jinns
comparison alongside ``burgers_jinns.py``, matching
``benchmarks/suite/physicsnemo/compare_underpinn_multi.py::run_heat``
as closely as jinns' loss/data-generator API allows:

  * same physics   : nabla^2 u + f = 0, f(x,y) = 2*pi^2*sin(pi x)*sin(pi y),
                     domain (x,y) in [0,1]^2, u=0 on all four edges,
                     exact u(x,y) = sin(pi x) sin(pi y)
  * same network   : plain MLP, 3 hidden layers x 64 units, tanh -- chosen
                     deliberately over underPINN's Fourier-embedding
                     problems (Burgers/Wave/Helmholtz all use FourierMLP)
                     specifically to avoid the "no jinns equivalent
                     architecture" gap noted for the PhysicsNeMo
                     comparison's FourierMLP/GatedMLP problems; Heat 2D is
                     the one underPINN benchmark-suite problem in the
                     PhysicsNeMo/jinns comparison set that already uses a
                     plain MLP on the underPINN side too
  * same batching  : minibatched every step (batch_r=2048 of a 5000-point
                     interior pool, batch_b=256 of a 1200-point boundary
                     pool) -- NOT full-batch like the Burgers comparison,
                     matching underPINN's own real Heat 2D config. See the
                     module-level note below on a real, unavoidable
                     semantic difference in *how* jinns minibatches versus
                     underPINN's own convention.
  * same optimizer : plain Adam with a cosine decay schedule
  * same reference : scored against the identical exact solution
                     underPINN itself uses (sin(pi x) sin(pi y))

A real minibatching-semantics difference, checked rather than assumed
(read ``jinns.data._CubicMeshPDEStatio.inside_batch``'s source directly):
jinns' ``omega_batch_size`` walks the shuffled n-point pool in sequential,
non-overlapping chunks, reshuffling once the pool is exhausted -- standard
epoch-based SGD. underPINN's ``safe_choice`` instead draws a fresh
independent random index set from the full pool on every single step (see
``underPINN/utils/sampling.py``). Both are "minibatch SGD with the same
per-step batch size drawn from the same candidate pool," but they are not
the same resampling scheme bit-for-bit -- flagged explicitly rather than
described as identical.

jinns does not ship a built-in Poisson/Helmholtz dynamic loss, so this uses
its documented extension point instead: a ``jinns.loss.PDEStatio``
subclass implementing ``equation(x, u, params)``, using jinns' own
``laplacian_rev`` operator (the same operator jinns' own built-in
``FisherKPP`` dynamic loss uses internally) -- this is the idiomatic way a
jinns user would write a custom stationary PDE, not a workaround.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python heat2d_jinns.py
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

N_R, N_BC = 5000, 300         # N_BC is per edge -> nb = 4*N_BC total
BATCH_R, BATCH_B = 2048, 256
BC_W = 100.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 64
LR = 1e-3

# A real, source-confirmed aggregation confound (not a naming mismatch):
# jinns.loss.LossPDEStatio's boundary_loss reduction is "mean within each
# of the 4 facets, then SUM across facets" (jinns/loss/_LossPDE.py's
# _reduction_functions -- jax.tree.reduce(jnp.add, ...) over
# jax.tree.map(mean_sum_reduction, ...)), not a single pooled mean over a
# mixed-facet minibatch the way underPINN's bc_l = jnp.mean(...) is. At
# matched BC_W, jinns' aggregate boundary term is therefore ~4x larger for
# the same per-point residual scale. BC_W_JINNS below lets --bc-w-jinns
# test a compensating 1/4 weight directly rather than only noting the
# difference in prose.


class PoissonEquation(jinns.loss.PDEStatio):
    """nabla^2 u + f = 0, f = 2*pi^2*sin(pi x)*sin(pi y). Mirrors
    underPINN.pde.heat.SteadyHeatPDE's sign convention exactly
    (nabla^2 u + f = 0, verified against that file's own docstring, not
    guessed from the PDE name)."""

    def equation(self, x, u, params):
        lap = laplacian_rev(x, u, params)[..., None]
        f = 2.0 * jnp.pi ** 2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
        return lap + f[..., None]


def run_jinns(epochs: int, seed: int, bc_w: float = BC_W) -> dict:
    key = random.PRNGKey(seed)

    eqx_list = ((eqx.nn.Linear, 2, LAYER_SIZE), (jax.nn.tanh,))
    for _ in range(LAYERS_HIDDEN - 1):
        eqx_list += ((eqx.nn.Linear, LAYER_SIZE, LAYER_SIZE), (jax.nn.tanh,))
    eqx_list += ((eqx.nn.Linear, LAYER_SIZE, 1),)

    key, subkey = random.split(key)
    u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDEStatio")

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey, n=N_R, nb=4 * N_BC,
        omega_batch_size=BATCH_R, omega_border_batch_size=BATCH_B,
        dim=2, min_pts=(0.0, 0.0), max_pts=(1.0, 1.0), method="uniform")

    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})

    poisson_loss = PoissonEquation()
    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=1.0, boundary_loss=bc_w)
    loss = jinns.loss.LossPDEStatio(
        u=u_pinn, loss_weights=loss_weights, dynamic_loss=poisson_loss,
        boundary_condition=jinns.loss.Dirichlet(), params=init_params)

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    tx = optax.adam(sched)

    # Untimed warmup, matching burgers_jinns.py's convention.
    jinns.solve(init_params=init_params, data=train_data, optimizer=tx,
               loss=loss, n_iter=2, verbose=False)

    t0 = time.perf_counter()
    (final_params, loss_values, loss_by_term, *_rest) = jinns.solve(
        init_params=init_params, data=train_data, optimizer=tx, loss=loss,
        n_iter=epochs, verbose=False)
    jax.block_until_ready(final_params.nn_params)
    wall = time.perf_counter() - t0
    final_loss = float(loss_values[-1])

    # ── score against the identical exact solution underPINN uses ──
    N_eval = 101
    x_eval = np.linspace(0.0, 1.0, N_eval)
    y_eval = np.linspace(0.0, 1.0, N_eval)
    XX, YY = np.meshgrid(x_eval, y_eval, indexing="ij")
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    xy_query = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)

    def u_single(xy_i):
        return u_pinn(xy_i, final_params)[0]

    u_pred = np.array(jax.vmap(u_single)(xy_query)).reshape(N_eval, N_eval)
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_exact": rel_l2,
            "n_r": N_R, "n_bc": 4 * N_BC, "batch_r": BATCH_R, "batch_b": BATCH_B,
            "bc_w": bc_w,
            "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX Heat2D, scored on the same exact reference "
                     "as the underPINN/PhysicsNeMo comparison")
    ap.set_defaults(epochs=5000)
    ap.add_argument("--bc-w-jinns", type=float, default=BC_W,
                    help="boundary loss weight passed to jinns (see the "
                         "sum-over-facets vs pooled-mean note above); "
                         "default matches underPINN's nominal BC_W=100")
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}   bc_w_jinns: {args.bc_w_jinns}\n")

    r = run_jinns(args.epochs, args.seed, bc_w=args.bc_w_jinns)
    print(f"jinns: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(exact)={r['rel_l2_vs_exact']:.4e}")

    save_result("jinns_compare_heat2d", {
        "problem": "heat_2d", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
