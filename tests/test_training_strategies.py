"""Tests for underPINN.training.strategies and nn.embeddings.PeriodicInput."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from underPINN.nn.embeddings import PeriodicInput
from underPINN.nn.mlp import MLP
from underPINN.training.strategies import causal_pde_loss, ema_update, grad_norm_weights, lbfgs_refine


def test_causal_loss_matches_hand_computation():
    res = jnp.array([1.0, 1.0, 2.0, 2.0])
    t = jnp.array([0.1, 0.2, 0.6, 0.7])
    loss, w, L = causal_pde_loss(res, t, (0.0, 1.0), n_bins=2, eps=1.0)
    np.testing.assert_allclose(np.asarray(L), [1.0, 4.0])
    np.testing.assert_allclose(np.asarray(w), [1.0, np.exp(-1.0)], rtol=1e-6)
    np.testing.assert_allclose(float(loss), (1.0 + np.exp(-1.0) * 4.0) / 2, rtol=1e-6)


def test_causal_weights_carry_no_gradient():
    t = jnp.linspace(0, 1, 8)
    g = jax.grad(lambda a: causal_pde_loss(a * jnp.ones(8), t, (0.0, 1.0), 4, 1.0)[0])(1.0)
    # with w frozen, d/da mean(w * a^2) = 2a * mean(w)
    _, w, _ = causal_pde_loss(jnp.ones(8), t, (0.0, 1.0), 4, 1.0)
    np.testing.assert_allclose(float(g), 2.0 * float(jnp.mean(w)), rtol=1e-6)


def test_causal_reduces_to_mean_for_eps_zero():
    r = jax.random.normal(jax.random.PRNGKey(0), (64,))
    t = jax.random.uniform(jax.random.PRNGKey(1), (64,))
    loss, w, L = causal_pde_loss(r, t, (0.0, 1.0), n_bins=8, eps=0.0)
    np.testing.assert_allclose(np.asarray(w), 1.0)
    np.testing.assert_allclose(float(loss), float(jnp.mean(L)), rtol=1e-6)


def test_grad_norm_weights_equalise_gradients():
    f = lambda q: {"a": jnp.sum(q["w"] ** 2), "b": 9 * jnp.sum(q["w"] ** 2)}          # noqa: E731
    lam = grad_norm_weights(f, {"w": jnp.ones(3)}, ["a", "b"])
    np.testing.assert_allclose(float(lam["a"]), 10.0, rtol=1e-6)
    np.testing.assert_allclose(float(lam["b"]), 10.0 / 9.0, rtol=1e-6)
    # weighted gradients now have equal norms
    ga = float(lam["a"]) * 2 * np.sqrt(3)
    gb = float(lam["b"]) * 18 * np.sqrt(3)
    assert ga == pytest.approx(gb)
    np.testing.assert_allclose(float(ema_update({"a": 1.0}, {"a": 11.0}, 0.9)["a"]), 2.0)


def test_lbfgs_reaches_analytic_minimum():
    # per coordinate: 2(w-3) + 0.4 w^3 = 0
    f = lambda q: jnp.sum((q["w"] - 3.0) ** 2) + 0.1 * jnp.sum(q["w"] ** 4)            # noqa: E731
    q, h = lbfgs_refine(f, {"w": jnp.zeros(4)}, max_iters=200)
    w = float(q["w"][0])
    assert abs(2 * (w - 3) + 0.4 * w ** 3) < 1e-3
    assert h["best"] <= h["loss"][0]


def test_periodic_input_is_exactly_periodic_with_derivatives():
    m = PeriodicInput(MLP(layers=[3, 16, 1]), lo=(0.0, 0.0), hi=(2 * np.pi, 1.0), periods={0: 2 * np.pi})
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    f = lambda z: m.apply(p, z[None])[0, 0]                                            # noqa: E731
    z = jnp.array([0.3, 0.4])
    zp = z.at[0].add(2 * np.pi)
    assert abs(float(f(z) - f(zp))) < 1e-5
    np.testing.assert_allclose(np.asarray(jax.hessian(f)(z)), np.asarray(jax.hessian(f)(zp)), atol=1e-4)


def test_lbfgs_handles_parameters_named_value():
    """optax's value_and_grad_from_state looks up 'value' by name in the whole optimiser
    state; attention layers have a 'value' parameter, which used to crash lbfgs_refine."""
    params = {"value": {"w": jnp.zeros(3)}, "key": {"w": jnp.ones(2)}}
    f = lambda q: jnp.sum((q["value"]["w"] - 1.0) ** 2) + jnp.sum(q["key"]["w"] ** 2)  # noqa: E731
    q, h = lbfgs_refine(f, params, max_iters=50)
    np.testing.assert_allclose(np.asarray(q["value"]["w"]), 1.0, atol=1e-4)
    assert h["best"] < 1e-6


def test_periodic_input_accepts_pairs_and_negative_index():
    """Int-keyed dict attributes break flax 0.8.x init; periods are stored as pairs, and a
    negative index (time as the last column) must not be duplicated as a plain input."""
    kw = dict(lo=(0.0, 0.0), hi=(1.0, 2 * np.pi))
    m_dict = PeriodicInput(MLP(layers=[3, 8, 1]), periods={-1: 2 * np.pi}, **kw)
    m_pair = PeriodicInput(MLP(layers=[3, 8, 1]), periods=((1, 2 * np.pi),), **kw)
    p = m_dict.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))      # 3 = cos, sin + 1 plain input
    z = jnp.array([[0.3, 0.4], [0.3, 0.4 + 2 * np.pi]])
    np.testing.assert_allclose(np.asarray(m_dict.apply(p, z)), np.asarray(m_pair.apply(p, z)))
    assert abs(float(m_dict.apply(p, z)[0, 0] - m_dict.apply(p, z)[1, 0])) < 1e-5


def test_lbfgs_history_and_best_are_consistent():
    """hist = [loss(params0), ..., loss(final)] with no duplicate, and 'best' is the loss of
    the returned parameters."""
    f = lambda q: jnp.sum((q["w"] - 3.0) ** 2) + 0.1 * jnp.sum(q["w"] ** 4)            # noqa: E731
    p0 = {"w": jnp.zeros(4)}
    for n in (1, 3):
        q, h = lbfgs_refine(f, p0, max_iters=n, patience=10 ** 9)
        assert len(h["loss"]) == n + 1 and h["iters"] == n
        assert h["loss"][0] == pytest.approx(float(f(p0)))
        assert h["loss"][1] < h["loss"][0]                       # no repeated initial value
        assert h["best"] == pytest.approx(float(f(q))) == pytest.approx(min(h["loss"]))
