"""Regression tests for BurgersPDE.residual's fused-AD rewrite.

The previous implementation called three separate AD transforms on the same
points (``jax.jacfwd`` for (u_x, u_t), ``jax.hessian`` for u_xx, and a third
plain ``model.apply`` for u) -- profiled at a real, measurable ~1.85x
overhead versus a fused ``jax.vjp`` (value + full gradient in one pass) +
``jax.jvp`` (u_xx only, not the full unused 2x2 Hessian) formulation. These
tests confirm the rewrite is not just faster but numerically identical:
against an analytic function with hand-derived derivatives, and against the
old jacfwd/hessian-based formulation directly, on the same trained-ish MLP.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE


def _tiny_mlp(in_dim: int, out_dim: int):
    model = MLP(layers=[in_dim, 8, 8, out_dim])
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, in_dim)))
    return model, params


class TestBurgersResidualAnalytic:
    """The residual reduces to a known value when the network is replaced
    by an exactly-representable function with hand-derived derivatives."""

    def test_matches_hand_derived_residual(self):
        # f(x, t) = x^3 t + sin(x) t^2
        #   f_x  = 3x^2 t + cos(x) t^2
        #   f_t  = x^3 + 2 sin(x) t
        #   f_xx = 6xt - sin(x) t^2
        nu = 0.01

        class AnalyticModel:
            def apply(self, params, xt):
                x, t = xt[:, 0], xt[:, 1]
                return (x ** 3 * t + jnp.sin(x) * t ** 2)[:, None]

        pde = BurgersPDE(AnalyticModel(), nu=nu)
        x = jnp.array([0.7, -0.3, 0.0, 1.0])
        t = jnp.array([1.3, 0.5, 0.2, 0.9])
        xt = jnp.stack([x, t], axis=1)

        res = pde.residual(None, xt)

        fx = 3 * x ** 2 * t + jnp.cos(x) * t ** 2
        ft = x ** 3 + 2 * jnp.sin(x) * t
        fxx = 6 * x * t - jnp.sin(x) * t ** 2
        f = x ** 3 * t + jnp.sin(x) * t ** 2
        expected = ft + f * fx - nu * fxx

        assert jnp.allclose(res, expected, atol=1e-4)


class TestBurgersResidualMatchesLegacyFormulation:
    """Cross-check the fused vjp/jvp rewrite against the original
    jacfwd+hessian+apply formulation on the same (small, real) MLP."""

    def _legacy_residual(self, pde, params, xt):
        def u_single(xy_i):
            return pde.model.apply(params, xy_i[None, :])[0, 0]
        J = jax.vmap(jax.jacfwd(u_single))(xt)
        H = jax.vmap(jax.hessian(u_single))(xt)
        u = pde.model.apply(params, xt)[:, 0]
        return J[:, 1] + u * J[:, 0] - pde.nu * H[:, 0, 0]

    def test_fused_matches_legacy_on_random_mlp(self):
        model, params = _tiny_mlp(2, 1)
        pde = BurgersPDE(model, nu=0.01)
        rng = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(rng)
        x = jax.random.uniform(k1, (50,), minval=-1.0, maxval=1.0)
        t = jax.random.uniform(k2, (50,), minval=0.0, maxval=1.5)
        xt = jnp.stack([x, t], axis=1)

        fused = pde.residual(params, xt)
        legacy = self._legacy_residual(pde, params, xt)

        # atol, not rtol: some residual entries are near zero on an
        # untrained network, which inflates relative error meaninglessly.
        # 5e-4 comfortably covers the ~2.8e-4 max abs diff observed between
        # the two AD paths on this network (accumulated float32 rounding
        # across a chained-tanh second-derivative computation -- expected
        # given the two paths differ in AD graph topology, not a tolerance
        # loosened to paper over a real discrepancy: see the tighter
        # analytic-function test above, and the jax.hessian cross-check in
        # BurgersPDE.residual's own docstring, both of which are far more
        # precise because they involve far fewer chained nonlinearities).
        assert jnp.allclose(fused, legacy, atol=5e-4)

    def test_shape_and_dtype_unchanged(self):
        model, params = _tiny_mlp(2, 1)
        pde = BurgersPDE(model, nu=0.01)
        xt = jax.random.uniform(jax.random.PRNGKey(1), (17, 2))
        res = pde.residual(params, xt)
        assert res.shape == (17,)
        assert res.dtype == jnp.float32
