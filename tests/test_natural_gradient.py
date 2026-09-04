"""Tests for underPINN/training/natural_gradient.py (Gauss-Newton / LM
natural-gradient training).

Covers:
* Exact linear least-squares converges in a single accepted step (Gauss-Newton
  is exact -- not an approximation -- for a linear residual).
* Loss is monotonically non-increasing by construction (accept/reject logic).
* Damping grows on a rejected step and shrinks on an accepted one.
* Converges a small nonlinear (MLP) PINN-style residual to a low loss.
* ``final_params`` round-trips through the same pytree structure as ``params0``.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as fnn

from underPINN.training.natural_gradient import (
    gauss_newton_step,
    train_gauss_newton,
)


# ---------------------------------------------------------------------------
# Exact linear least squares: Gauss-Newton should solve it in one step.
# ---------------------------------------------------------------------------

class TestLinearLeastSquares:
    def _problem(self, seed=0):
        rng = np.random.default_rng(seed)
        A = jnp.array(rng.normal(size=(20, 5)).astype(np.float32))
        theta_true = jnp.array(rng.normal(size=(5,)).astype(np.float32))
        b = A @ theta_true

        def residual_fn(theta):
            return A @ theta - b
        return residual_fn, theta_true

    def test_single_step_reaches_near_zero_loss(self):
        residual_fn, theta_true = self._problem()
        theta0 = jnp.zeros_like(theta_true)
        final, loss_hist, _ = train_gauss_newton(
            residual_fn, theta0, epochs=1, damping0=0.0)
        assert loss_hist[-1] < 1e-8
        np.testing.assert_allclose(np.array(final), np.array(theta_true),
                                   atol=1e-3)

    def test_zero_damping_matches_normal_equations(self):
        residual_fn, theta_true = self._problem(seed=1)
        theta0 = jnp.zeros_like(theta_true)
        trial = gauss_newton_step(residual_fn, theta0, damping=0.0)
        np.testing.assert_allclose(np.array(trial), np.array(theta_true),
                                   atol=1e-3)


# ---------------------------------------------------------------------------
# Loss monotonicity + damping adaptation invariants
# ---------------------------------------------------------------------------

class TestAcceptRejectInvariants:
    def _rosenbrock_like_residual(self, theta):
        # A mildly nonlinear 2-D residual (banana-shaped valley) -- a classic
        # stress test for Gauss-Newton/LM damping control.
        x, y = theta[0], theta[1]
        return jnp.array([10.0 * (y - x ** 2), 1.0 - x])

    def test_loss_is_monotonically_non_increasing(self):
        theta0 = jnp.array([-1.5, 2.0])
        _, loss_hist, _ = train_gauss_newton(
            self._rosenbrock_like_residual, theta0, epochs=30, damping0=1.0,
            log_every=0)
        diffs = np.diff(loss_hist)
        assert np.all(diffs <= 1e-9), "loss increased on some accepted step"

    def test_damping_grows_after_a_forced_rejection(self):
        # Start damping absurdly small on a nonlinear residual so the first
        # (near-undamped Gauss-Newton) step overshoots and is rejected.
        theta0 = jnp.array([-1.5, 2.0])
        _, _, damping_hist = train_gauss_newton(
            self._rosenbrock_like_residual, theta0, epochs=1, damping0=1e-12)
        assert damping_hist[-1] > damping_hist[0]

    def test_damping_shrinks_after_a_good_step(self):
        # Large initial damping ~ gradient descent with a tiny step -> the
        # very first step should be accepted (loss can only improve at worst
        # marginally, never overshoot), shrinking damping.
        theta0 = jnp.array([-1.5, 2.0])
        _, loss_hist, damping_hist = train_gauss_newton(
            self._rosenbrock_like_residual, theta0, epochs=1, damping0=1e6)
        assert loss_hist[-1] <= loss_hist[0]
        assert damping_hist[-1] < damping_hist[0]

    def test_final_loss_lower_than_initial(self):
        theta0 = jnp.array([-1.5, 2.0])
        _, loss_hist, _ = train_gauss_newton(
            self._rosenbrock_like_residual, theta0, epochs=50, damping0=1.0)
        assert loss_hist[-1] < loss_hist[0]


# ---------------------------------------------------------------------------
# Small nonlinear PINN-style problem (MLP fitting a smooth target function)
# ---------------------------------------------------------------------------

class TestSmallMLPProblem:
    def test_converges_and_preserves_pytree_structure(self):
        class TinyMLP(fnn.Module):
            @fnn.compact
            def __call__(self, x):
                h = jnp.tanh(fnn.Dense(8)(x))
                return fnn.Dense(1)(h)

        model = TinyMLP()
        params0 = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))

        x = jnp.linspace(-1.0, 1.0, 30)[:, None]
        y_target = jnp.sin(2.0 * jnp.pi * x[:, 0])

        def residual_fn(params):
            pred = model.apply(params, x)[:, 0]
            return pred - y_target

        final_params, loss_hist, _ = train_gauss_newton(
            residual_fn, params0, epochs=40, damping0=1e-2)

        # Same pytree structure (Flax FrozenDict / dict of arrays) preserved.
        assert jax.tree_util.tree_structure(final_params) == \
            jax.tree_util.tree_structure(params0)
        # Meaningfully fit the target -- well below the untrained loss.
        assert loss_hist[-1] < 0.05 * loss_hist[0]
