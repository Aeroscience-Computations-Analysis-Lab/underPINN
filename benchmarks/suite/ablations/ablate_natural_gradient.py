"""Ablation: Gauss-Newton / natural-gradient training vs. Adam, on the ODE
harmonic oscillator.

A reviewer asked for discussion of second-order / natural-gradient training
(citing recent "D-NGD"-style work) alongside underPINN's Adam-based solvers.
``underPINN/training/natural_gradient.py::train_gauss_newton`` adds a
Levenberg-Marquardt-damped Gauss-Newton trainer (see its docstring for what
it is and, importantly, is *not* a reproduction of) as a genuine alternative
to Adam. This script measures whether it actually helps, on a problem small
enough for exact Gauss-Newton to be tractable at all: the harmonic
oscillator ODE (``u'' + omega^2 u = 0``,
:class:`underPINN.pde.ode.HarmonicOscillatorODE`), matching
``examples/ode/ode_test.py``'s physics and IC weighting exactly, but with a
much smaller network (order-100 parameters, not the example's default) so
that forming an explicit Gauss-Newton matrix every step is cheap.

Two arms, same network initialisation, same collocation points, same epoch
budget:

  adam            underPINN's standard Adam + cosine-decay training
  gauss_newton    Levenberg-Marquardt-damped Gauss-Newton (this module)

Scored on relative L^2 against the exact solution
``u(t) = cos(omega t)``. Wall-clock is reported alongside epoch count because
a Gauss-Newton epoch is far more expensive than an Adam epoch (it forms and
solves an explicit parameter-count-squared linear system) -- "did fewer,
costlier steps still win on wall-clock" is the fair question, not raw epoch
count.

Run:
    python benchmarks/suite/ablations/ablate_natural_gradient.py --epochs 300
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import optax                                                       # noqa: E402

from common import base_parser, jax_device_info, save_result, timed, warn_if_cpu  # noqa: E402

from underPINN.nn.mlp import MLP                                    # noqa: E402
from underPINN.pde.ode import HarmonicOscillatorODE                 # noqa: E402
from underPINN.training.natural_gradient import train_gauss_newton  # noqa: E402
from underPINN.utils.metrics import relative_l2_error               # noqa: E402

OMEGA = 2.0
T_MAX = 10.0
U0, V0 = 1.0, 0.0
N_COL = 200                # small on purpose -- keeps Gauss-Newton tractable
LAYERS = [1, 20, 20, 1]    # ~500 params
LR = 1e-3
W_IC = 100.0
W_IC_DOT = 100.0


def make_problem():
    t_r = jnp.linspace(0.0, T_MAX, N_COL)
    t_ic = jnp.array([0.0])
    u_ic = jnp.array([U0])
    u_ic_dot = jnp.array([V0])
    return t_r, t_ic, u_ic, u_ic_dot


def _rel_l2(pde, params) -> float:
    t_test = jnp.linspace(0.0, T_MAX, 2000)
    u_pred = pde.u(params, t_test)
    u_exact = pde.exact(t_test)
    return float(relative_l2_error(u_pred, u_exact))


def run_adam(epochs: int, seed: int, problem) -> dict:
    t_r, t_ic, u_ic, u_ic_dot = problem
    model = MLP(layers=LAYERS)
    pde = HarmonicOscillatorODE(model, omega=OMEGA)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 1)))

    def loss_fn(p):
        r_pde = pde.residual(p, t_r)
        pde_l = jnp.mean(r_pde ** 2)
        ic_l = jnp.mean((pde.u(p, t_ic) - u_ic) ** 2)
        ic_dot_l = jnp.mean((pde.ut(p, t_ic) - u_ic_dot) ** 2)
        return pde_l + W_IC * ic_l + W_IC_DOT * ic_dot_l

    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-2)
    opt = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(sched),
                      optax.scale(-1.0))

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    state = opt.init(params0)
    _p, _s, warm = step(params0, state)
    warm.block_until_ready()               # compile once, untimed

    def train():
        p, s, loss = params0, state, None
        for _ in range(epochs):
            p, s, loss = step(p, s)
        loss.block_until_ready()
        return p, float(loss)

    (final_params, final_loss), wall = timed(train)
    return {"method": "adam", "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs, "final_loss": final_loss,
            "rel_l2": _rel_l2(pde, final_params)}


def run_gauss_newton(epochs: int, seed: int, problem) -> dict:
    t_r, t_ic, u_ic, u_ic_dot = problem
    model = MLP(layers=LAYERS)
    pde = HarmonicOscillatorODE(model, omega=OMEGA)
    params0 = model.init(jax.random.PRNGKey(seed), jnp.ones((1, 1)))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params0))

    def residual_fn(p):
        r_pde = pde.residual(p, t_r)                                   # (N_COL,)
        r_ic = jnp.sqrt(W_IC) * (pde.u(p, t_ic) - u_ic)                # (1,)
        r_ic_dot = jnp.sqrt(W_IC_DOT) * (pde.ut(p, t_ic) - u_ic_dot)   # (1,)
        return jnp.concatenate([r_pde, r_ic, r_ic_dot])

    def train():
        final_params, loss_hist, damping_hist = train_gauss_newton(
            residual_fn, params0, epochs=epochs, damping0=1e-2)
        return final_params, loss_hist, damping_hist

    (final_params, loss_hist, damping_hist), wall = timed(train)
    return {"method": "gauss_newton", "epochs": epochs, "wall_s": wall,
            "ms_per_epoch": 1e3 * wall / epochs,
            "final_loss": loss_hist[-1], "final_damping": damping_hist[-1],
            "n_params": n_params,
            "rel_l2": _rel_l2(pde, final_params)}


def main() -> int:
    ap = base_parser("Ablate Gauss-Newton vs Adam on the ODE harmonic oscillator")
    ap.set_defaults(epochs=2000)
    ap.add_argument("--methods", nargs="*", default=["adam", "gauss_newton"],
                    choices=["adam", "gauss_newton"])
    args = ap.parse_args()

    info = jax_device_info(require_gpu=not args.allow_cpu)
    warn_if_cpu(info)
    print(f"JAX backend: {info['platform']} ({info['device_name']})")
    print(f"Epochs: {args.epochs}   seed: {args.seed}\n")

    problem = make_problem()
    rows = {}
    runners = {"adam": run_adam, "gauss_newton": run_gauss_newton}
    for name in args.methods:
        print(f"--- {name}")
        try:
            r = runners[name](args.epochs, args.seed, problem)
            rows[name] = r
            print(f"    {r['wall_s']:8.3f}s total  {r['ms_per_epoch']:8.3f} ms/ep "
                  f" loss={r['final_loss']:.4e}  rel_L2={r['rel_l2']:.4e}")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            rows[name] = {"error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in rows.items() if "error" not in v}
    if "adam" in ok and "gauss_newton" in ok:
        print("\n" + "=" * 84)
        print(f"{'method':13s} {'wall_s':>9s} {'ms/ep':>9s} {'rel L2':>11s} "
              f"{'vs adam (L2)':>13s}")
        print("-" * 84)
        base = ok["adam"]["rel_l2"]
        for name in ("adam", "gauss_newton"):
            r = ok[name]
            rel = f"{base / r['rel_l2']:.2f}x" if name != "adam" else "-"
            print(f"{name:13s} {r['wall_s']:9.3f} {r['ms_per_epoch']:9.3f} "
                  f"{r['rel_l2']:11.4e} {rel:>13s}")
        print("=" * 84)
        wall_ratio = ok["adam"]["wall_s"] / ok["gauss_newton"]["wall_s"]
        acc_ratio = base / ok["gauss_newton"]["rel_l2"]
        print(f"\nAt {args.epochs} epochs each: Gauss-Newton is "
              f"{'faster' if wall_ratio > 1 else 'slower'} on wall-clock "
              f"({1/wall_ratio if wall_ratio < 1 else wall_ratio:.2f}x) and "
              f"{'more' if acc_ratio > 1 else 'less'} accurate "
              f"({acc_ratio if acc_ratio > 1 else 1/acc_ratio:.2f}x rel-L2 "
              f"{'improvement' if acc_ratio > 1 else 'degradation'}).")

    save_result("ablation_natural_gradient_ode_harmonic", {
        "problem": "ode_harmonic_oscillator", "epochs": args.epochs,
        "seed": args.seed, "device": info,
        "metric": "relative L2 vs exact cos(omega t) solution",
        "network_layers": LAYERS, "methods": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
