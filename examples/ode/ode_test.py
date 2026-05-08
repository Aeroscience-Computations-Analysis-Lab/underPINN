"""
ODE Test Cases for underPINN
============================

Test 1 — Exponential Decay:
    du/dt + 2u = 0,  u(0) = 1
    Exact: u(t) = exp(-2t),  t in [0, 2]

Test 2 — Harmonic Oscillator:
    d²u/dt² + 4u = 0,  u(0) = 1, u'(0) = 0
    Exact: u(t) = cos(2t),  t in [0, 2pi]

Demonstrates the production-style API:
  - TrainingConfig  (unified hyperparameters)
  - ConsoleLogger   (replaces hardcoded prints)
  - EarlyStopping   (stops when loss plateaus)
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from underPINN.nn.mlp import MLP
from underPINN.pde.ode import ExponentialDecayODE, HarmonicOscillatorODE
from underPINN.losses.ode_loss import ODELoss
from underPINN.solver.ode_solver import ODESolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.metrics import print_errors
from underPINN.utils.plotting import plot_ode_result


# ---------------------------------------------------------------------------
# Test 1: Exponential Decay  du/dt + 2u = 0
# ---------------------------------------------------------------------------

def test_exponential_decay():
    print("\n" + "=" * 60)
    print("Test 1: Exponential Decay  du/dt + 2u = 0")
    print("=" * 60)

    LAM = 2.0
    T_MAX = 2.0
    EPOCHS = 3000

    t_r  = jnp.linspace(0.0, T_MAX, 500)
    t_ic = jnp.array([0.0])
    u_ic = jnp.array([1.0])

    config = TrainingConfig(
        epochs=EPOCHS,
        lr=1e-3,
        lr_schedule=optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2),
        log_every=500,
        callbacks=[
            ConsoleLogger(log_every=500),
            EarlyStopping(patience=300),
        ],
    )

    model  = MLP(layers=[1, 64, 64, 64, 1])
    pde    = ExponentialDecayODE(model, lam=LAM)
    loss   = ODELoss(model, pde, ic_weight=100.0)
    solver = ODESolver(model, pde, loss)

    solver.init(jax.random.PRNGKey(0))
    solver.train(t_r, t_ic, u_ic, config=config)

    t_test  = jnp.linspace(0.0, T_MAX, 1000)
    u_pred  = pde.u(solver.params, t_test)
    u_exact = pde.exact(t_test)

    print_errors(u_pred, u_exact, label="Exp Decay")

    plot_ode_result(
        t_test, u_pred, u_exact,
        solver.loss_hist, solver.pde_hist, solver.ic_hist,
        title=f"Exponential Decay: du/dt + {LAM}u = 0",
        filename="ode_exponential_decay.png",
    )

    return float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)) / jnp.sqrt(jnp.mean(u_exact ** 2)))


# ---------------------------------------------------------------------------
# Test 2: Harmonic Oscillator  d²u/dt² + 4u = 0
# ---------------------------------------------------------------------------

def test_harmonic_oscillator():
    print("\n" + "=" * 60)
    print("Test 2: Harmonic Oscillator  d²u/dt² + 4u = 0")
    print("=" * 60)

    OMEGA = 2.0
    T_MAX = 2 * np.pi
    EPOCHS = 5000

    t_r      = jnp.linspace(0.0, T_MAX, 800)
    t_ic     = jnp.array([0.0])
    u_ic     = jnp.array([1.0])
    u_ic_dot = jnp.array([0.0])

    config = TrainingConfig(
        epochs=EPOCHS,
        lr=1e-3,
        lr_schedule=optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2),
        log_every=500,
        callbacks=[
            ConsoleLogger(log_every=500),
            EarlyStopping(patience=400),
        ],
    )

    model  = MLP(layers=[1, 64, 64, 64, 1])
    pde    = HarmonicOscillatorODE(model, omega=OMEGA)
    loss   = ODELoss(model, pde, ic_weight=100.0, ic_derivative_weight=100.0)
    solver = ODESolver(model, pde, loss)

    solver.init(jax.random.PRNGKey(42))
    solver.train(t_r, t_ic, u_ic, u_ic_dot=u_ic_dot, config=config)

    t_test  = jnp.linspace(0.0, T_MAX, 1000)
    u_pred  = pde.u(solver.params, t_test)
    u_exact = pde.exact(t_test)

    print_errors(u_pred, u_exact, label="Harmonic")

    plot_ode_result(
        t_test, u_pred, u_exact,
        solver.loss_hist, solver.pde_hist, solver.ic_hist,
        title=f"Harmonic Oscillator: d²u/dt² + {OMEGA**2:.0f}u = 0",
        filename="ode_harmonic_oscillator.png",
    )

    return float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)) / jnp.sqrt(jnp.mean(u_exact ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())

    rel_err_decay    = test_exponential_decay()
    rel_err_harmonic = test_harmonic_oscillator()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exponential Decay   — Rel-L2: {rel_err_decay:.4e}")
    print(f"  Harmonic Oscillator — Rel-L2: {rel_err_harmonic:.4e}")

    assert rel_err_decay    < 1e-2, f"Exp decay error too large: {rel_err_decay:.4e}"
    assert rel_err_harmonic < 5e-2, f"Harmonic error too large:  {rel_err_harmonic:.4e}"

    print("\nAll ODE tests PASSED.")


if __name__ == "__main__":
    main()
