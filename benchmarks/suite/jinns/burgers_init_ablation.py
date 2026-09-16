"""Isolates ONE variable from the underPINN-vs-jinns Burgers comparison
(../jinns/README.md): weight/bias initialization scheme.

We confirmed by reading source directly (not assuming from argument names)
that ``eqx.nn.Linear`` (jinns' network layer) defaults to
``Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`` for *both* weight and bias,
whereas Flax's ``nn.Dense`` (underPINN's layer) defaults to
``lecun_normal()`` (variance-scaling truncated-normal, fan_in-based) for the
weight and *zero* for the bias -- a real difference, not the same scheme
under a different name. This script reruns underPINN's own Burgers case
with the network's kernel/bias initializers swapped to reproduce Equinox's
exact scheme layer-by-layer (correct per-layer fan_in for the bias bound
too, not just the weight), everything else identical to
``../physicsnemo/compare_underpinn.py``, to measure how much of the
underPINN-vs-jinns accuracy gap this one variable explains.

Run:
    python benchmarks/suite/jinns/burgers_init_ablation.py --epochs 5000
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "baselines"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import numpy as np                                             # noqa: E402
import optax                                                   # noqa: E402
from flax import linen as fnn                                  # noqa: E402

from common import base_parser, jax_device_info, save_result, timed, warn_if_cpu  # noqa: E402
from burgers_baselines import (LAYERS, LR, N_BC, N_IC, N_R,     # noqa: E402
                               NU, T_MAX, W_BC, W_IC, make_data)

from underPINN.utils.operator_datagen import burgers1d_exact    # noqa: E402
from underPINN.utils.metrics import relative_l2_error           # noqa: E402


def equinox_style_init(fan_in: int):
    """Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) -- eqx.nn.Linear's default,
    confirmed from its source (equinox/nn/_linear.py::default_init /
    Linear.__init__), reproduced here for both kernel *and* bias of one
    Flax Dense layer with this layer's actual fan_in (Flax's own
    ``bias_init`` signature has no way to see the paired kernel's fan_in,
    so it must be supplied explicitly per layer, not inferred from shape)."""
    lim = 1.0 / jnp.sqrt(fan_in)
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-lim, maxval=lim)
    return init


class MLPNetEquinoxInit(fnn.Module):
    @fnn.compact
    def __call__(self, xt):
        h = xt
        fan_in = xt.shape[-1]
        for w in LAYERS[1:-1]:
            init = equinox_style_init(fan_in)
            h = jnp.tanh(fnn.Dense(w, kernel_init=init, bias_init=init)(h))
            fan_in = w
        init = equinox_style_init(fan_in)
        return fnn.Dense(LAYERS[-1], kernel_init=init, bias_init=init)(h)


def run(epochs: int, seed: int) -> dict:
    data = make_data(seed)
    x_r, t_r, x_ic, u_ic, x_bc, t_bc = data

    model = MLPNetEquinoxInit()
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 2)))

    def u_single(p, xt):
        return model.apply(p, xt[None, :])[0, 0]

    jac = jax.vmap(jax.jacfwd(u_single, argnums=1), in_axes=(None, 0))
    hess = jax.vmap(jax.hessian(u_single, argnums=1), in_axes=(None, 0))

    XR = jnp.stack([jnp.array(x_r), jnp.array(t_r)], axis=1)
    XI = jnp.stack([jnp.array(x_ic), jnp.zeros_like(jnp.array(x_ic))], axis=1)
    UI = jnp.array(u_ic)
    XB = jnp.stack([jnp.array(x_bc), jnp.array(t_bc)], axis=1)

    def loss_fn(p):
        J = jac(p, XR)
        H = hess(p, XR)
        u = model.apply(p, XR)[:, 0]
        res = J[:, 1] + u * J[:, 0] - NU * H[:, 0, 0]
        pde_l = jnp.mean(res ** 2)
        ic_l = jnp.mean((model.apply(p, XI)[:, 0] - UI) ** 2)
        bc_l = jnp.mean(model.apply(p, XB)[:, 0] ** 2)
        return pde_l + W_IC * ic_l + W_BC * bc_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    state0 = opt.init(params0)
    _p, _s, warm = step(params0, state0)
    warm.block_until_ready()

    def train():
        p, s, loss = params0, state0, None
        for _ in range(epochs):
            p, s, loss = step(p, s)
        loss.block_until_ready()
        return p, float(loss)

    (final_params, final_loss), wall = timed(train)

    Nx_eval, Nt_eval = 101, 41
    x_eval = np.linspace(-1.0, 1.0, Nx_eval)
    t_eval = np.linspace(0.0, T_MAX, Nt_eval)
    u_exact = burgers1d_exact(x_eval, t_eval, nu=NU, u0_mode=1)
    XX, TT = np.meshgrid(x_eval, t_eval, indexing="ij")
    xt_query = jnp.array(np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32))
    u_pred = np.array(model.apply(final_params, xt_query)[:, 0]).reshape(Nx_eval, Nt_eval)
    rel_l2 = float(relative_l2_error(u_pred, u_exact))

    return {"framework": "underpinn-jax-jit-equinox-init", "epochs": epochs,
            "wall_s": wall, "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": final_loss, "rel_l2_vs_cole_hopf": rel_l2}


def main() -> int:
    ap = base_parser("underPINN Burgers with Equinox-style init, isolating "
                     "the init-scheme variable from the jinns comparison")
    ap.set_defaults(epochs=5000)
    args = ap.parse_args()
    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    r = run(args.epochs, args.seed)
    print(f"underPINN (Equinox-style init): {r['ms_per_epoch']:.3f} ms/ep  "
         f"rel_L2(Cole-Hopf)={r['rel_l2_vs_cole_hopf']:.4e}  "
         f"(default-init baseline: 0.2605)")
    save_result("jinns_underpinn_equinox_init_ablation", {"device": info, **r})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
