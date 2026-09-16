"""ODE harmonic oscillator, solved with jinns
(https://gitlab.com/mia_jinns/jinns) -- a fifth jinns comparison, matching
``benchmarks/suite/physicsnemo/compare_underpinn_multi.py::run_ode_harmonic``
as closely as jinns' loss/data-generator API allows:

  * same physics   : u'' + omega^2 u = 0, omega=2, t in [0, T_MAX=5], IC
                     u(0)=1, u'(0)=0, exact u(t) = cos(omega t)
  * same network   : plain MLP, 3 hidden layers x 64 units, tanh (matches
                     underPINN's own choice)
  * same batching  : minibatched every step (batch_r=4096 of a 10,000-point
                     pool), using jinns' DataGeneratorODE's
                     temporal_batch_size
  * same optimizer : plain Adam with a cosine decay schedule
  * same reference : scored against the identical exact solution
                     underPINN itself uses (cos(omega t))

This is the one problem in the PhysicsNeMo comparison where underPINN
*lost* to PhysicsNeMo on accuracy (rel L2 0.83 vs. PhysicsNeMo's 0.043, a
19x gap) -- making a genuine jinns data point here more informative than
on a problem where underPINN's own accuracy is already strong.

**A real API limitation, found rather than assumed, that changes this
script's structure from the other four in this directory:**
``jinns.loss.LossODE``'s ``initial_condition`` argument only supports a
*value* constraint, ``u(t0) = u0`` (confirmed from ``_LossODE.py``'s
docstring and its ``initial_condition_check`` usage) -- there is no
built-in mechanism for a *derivative* initial condition like this
problem's ``u'(0) = 0``, unlike ``jinns.loss.BurgersEquation`` /
``FisherKPP``'s dynamic-loss terms, which can freely differentiate the
network internally. Rather than skip this problem or silently drop the
``u'(0)=0`` constraint (which would not be a matched comparison), this
script bypasses ``LossODE``/``jinns.solve`` and instead writes a small
composite loss directly: it calls a ``jinns.loss.ODE`` subclass
(``HarmonicOscillatorEquation``, defined below -- jinns itself has no
built-in harmonic-oscillator equation, so this is user code built on
jinns' real ``ODE`` extension point, the same one ``GeneralizedLotkaVolterra``
uses in jinns' own source) for the interior residual, and adds the
derivative-IC term with a plain ``jax.grad`` call, exactly mirroring
``underPINN.pde.ode.HarmonicOscillatorODE.ut``'s own formula. Everything else -- the PINN class (``jinns.nn.PINN_MLP``), the
collocation sampler (``jinns.data.DataGeneratorODE``), and the equation
residual itself -- is still genuine jinns machinery; only the top-level
training loop (``jax.value_and_grad`` + ``optax``, not
``jinns.solve``) is hand-rolled, and disclosed as such rather than
presented as an unmodified ``jinns.solve`` run like the other four scripts
here.

Run (from this directory, using the jinns venv):
    ./.venv/bin/python ode_harmonic_jinns.py
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

OMEGA, T_MAX, U0, V0 = 2.0, 5.0, 1.0, 0.0
N_R = 10000
BATCH_R = 4096
IC_W, IC_DOT_W = 100.0, 100.0
LAYERS_HIDDEN, LAYER_SIZE = 3, 64
LR = 1e-3


class HarmonicOscillatorEquation(jinns.loss.ODE):
    """u'' + omega^2 u = 0. Tmax-scaled the same way jinns' own
    BurgersEquation/FisherKPP scale their spatial/reaction terms (see
    burgers_jinns.py's docstring): with t_tilde = t/Tmax,
    d^2/dt_physical^2 = (1/Tmax^2) d^2/dt_tilde^2, so multiplying the whole
    physical equation by Tmax^2 gives d^2u/dt_tilde^2 + Tmax^2*omega^2*u=0."""

    omega: float = eqx.field(default=2.0, static=True)

    def equation(self, t, u, params):
        def u_(t):
            return u(t, params)[0]

        def ut_(t):
            return jax.grad(u_)(t)[0]

        utt = jax.grad(ut_)(t)[0]
        return jnp.array([utt + self.Tmax ** 2 * self.omega ** 2 * u_(t)])


def run_jinns(epochs: int, seed: int) -> dict:
    key = random.PRNGKey(seed)

    eqx_list = ((eqx.nn.Linear, 1, LAYER_SIZE), (jax.nn.tanh,))
    for _ in range(LAYERS_HIDDEN - 1):
        eqx_list += ((eqx.nn.Linear, LAYER_SIZE, LAYER_SIZE), (jax.nn.tanh,))
    eqx_list += ((eqx.nn.Linear, LAYER_SIZE, 1),)

    key, subkey = random.split(key)
    u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE")

    key, subkey = random.split(key)
    # t_tilde in [0, 1] -- Tmax enters via HarmonicOscillatorEquation, same
    # convention as the other four scripts in this directory.
    train_data = jinns.data.DataGeneratorODE(
        key=subkey, nt=N_R, tmin=0.0, tmax=1.0,
        temporal_batch_size=BATCH_R, method="uniform")

    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})

    ho_loss = HarmonicOscillatorEquation(Tmax=T_MAX, omega=OMEGA)
    t0_tilde = jnp.zeros((1,))

    def loss_fn(params, t_batch):
        # interior residual, vmapped exactly the way jinns' own LossODE
        # vmaps a dynamic loss internally (jinns.loss._loss_utils
        # .vmap_loss_fun_classical: jax.vmap(fun, in_axes=(0, None))).
        res = jax.vmap(lambda t_i: ho_loss.equation(t_i, u_pinn, params),
                       in_axes=(0,))(t_batch)
        pde_l = jnp.mean(res ** 2)

        # value IC: u(0) = U0 (physical t=0 <-> t_tilde=0)
        u0_pred = u_pinn(t0_tilde, params)[0]
        ic_l = (u0_pred - U0) ** 2

        # derivative IC: u'(0) = V0, physical derivative = (1/Tmax) *
        # d/dt_tilde, matching underPINN.pde.ode.HarmonicOscillatorODE.ut's
        # own formula exactly (a single jax.grad of the scalar network
        # output, not a jinns built-in -- this is the constraint
        # LossODE's initial_condition mechanism cannot express, see the
        # module docstring).
        def u_(t):
            return u_pinn(t, params)[0]
        ut0_pred = jax.grad(u_)(t0_tilde)[0] / T_MAX
        ic_dot_l = (ut0_pred - V0) ** 2

        return pde_l + IC_W * ic_l + IC_DOT_W * ic_dot_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    tx = optax.adam(sched)
    opt_state = tx.init(init_params)

    @jax.jit
    def step(params, opt_state, data):
        data, batch = data.get_batch()
        loss, grads = jax.value_and_grad(loss_fn)(params, batch.temporal_batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, data, loss

    # untimed warmup, matching the other scripts' convention
    p, s, d = init_params, opt_state, train_data
    for _ in range(2):
        p, s, d, warm_loss = step(p, s, d)
    jax.block_until_ready(warm_loss)

    params, opt_state, data = init_params, opt_state, train_data
    t0 = time.perf_counter()
    loss = None
    for _ in range(epochs):
        params, opt_state, data, loss = step(params, opt_state, data)
    jax.block_until_ready(loss)
    wall = time.perf_counter() - t0
    final_loss = float(loss)

    t_eval = np.linspace(0.0, T_MAX, 2000).astype(np.float32)
    u_exact = U0 * np.cos(OMEGA * t_eval)
    t_tilde_eval = jnp.array((t_eval / T_MAX)[:, None])

    def u_single(t_i):
        return u_pinn(t_i, params)[0]

    u_pred = np.array(jax.vmap(u_single)(t_tilde_eval))
    rel_l2 = float(np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-10))

    return {"framework": "jinns", "jinns_version": "1.10.0", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_exact": rel_l2,
            "n_r": N_R, "batch_r": BATCH_R,
            "layers_hidden": LAYERS_HIDDEN, "layer_size": LAYER_SIZE}


def main() -> int:
    ap = base_parser("jinns JAX ODE Harmonic, scored on the same exact "
                     "reference as the underPINN/PhysicsNeMo comparison")
    ap.set_defaults(epochs=3000)
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    r = run_jinns(args.epochs, args.seed)
    print(f"jinns: {r['ms_per_epoch']:.3f} ms/ep  final_loss={r['final_loss']:.4e}  "
         f"rel_L2(exact)={r['rel_l2_vs_exact']:.4e}")

    save_result("jinns_compare_ode_harmonic", {
        "problem": "ode_harmonic", "device": info, **r,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
