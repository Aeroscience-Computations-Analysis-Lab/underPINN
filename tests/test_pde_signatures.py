"""Tests for PDE residual signature unification.

Verifies that every updated PDE accepts a single packed coordinate array
``xy`` of shape ``(N, D)`` and returns a ``jnp.ndarray`` (never a tuple).

Numeric regression: each test compares the residual evaluated with the new
packed API against the value computed by the legacy split-arg code to
confirm no logic was changed — only the interface.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

from underPINN.nn.mlp import MLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_mlp(in_dim: int, out_dim: int):
    """Deterministically-initialised tiny MLP for tests."""
    model  = MLP(layers=[in_dim, 8, 8, out_dim])
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, in_dim)))
    return model, params


# ---------------------------------------------------------------------------
# #3 — BurgersPDE
# ---------------------------------------------------------------------------

class TestBurgersPDESignature:
    def setup_method(self):
        from underPINN.pde.burgers import BurgersPDE
        self.model, self.params = _tiny_mlp(2, 1)
        self.pde = BurgersPDE(self.model, nu=0.01)
        rng = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(rng)
        self.x  = jax.random.uniform(k1, (10,), minval=-1.0, maxval=1.0)
        self.t  = jax.random.uniform(k2, (10,), minval=0.0,  maxval=1.0)

    def test_packed_array_accepted(self):
        xt = jnp.stack([self.x, self.t], axis=1)
        res = self.pde.residual(self.params, xt)
        assert res.shape == (10,)

    def test_returns_ndarray_not_tuple(self):
        xt = jnp.stack([self.x, self.t], axis=1)
        res = self.pde.residual(self.params, xt)
        assert isinstance(res, jnp.ndarray)

    def test_numeric_regression(self):
        """Packed-API result must be identical to old split-arg path."""
        xt = jnp.stack([self.x, self.t], axis=1)
        res_new = self.pde.residual(self.params, xt)

        # Replicate legacy logic inline so no old code dependency
        def _legacy(params, x, t):
            xy = jnp.stack([x, t], axis=1)
            def u_s(xy_i):
                return self.model.apply(params, xy_i[None, :])[0, 0]
            J   = jax.vmap(jax.jacfwd(u_s))(xy)
            H   = jax.vmap(jax.hessian(u_s))(xy)
            ux  = J[:, 0]; ut = J[:, 1]; uxx = H[:, 0, 0]
            u   = self.model.apply(params, xy)[:, 0]
            return ut + u * ux - 0.01 * uxx

        res_old = _legacy(self.params, self.x, self.t)
        assert jnp.allclose(res_new, res_old, atol=1e-5)


# ---------------------------------------------------------------------------
# #3 — WavePDE
# ---------------------------------------------------------------------------

class TestWavePDESignature:
    def setup_method(self):
        from underPINN.pde.wave import WavePDE
        self.model, self.params = _tiny_mlp(2, 1)
        self.pde = WavePDE(self.model, c=1.0)
        rng = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(rng)
        self.x = jax.random.uniform(k1, (8,))
        self.t = jax.random.uniform(k2, (8,))

    def test_packed_array_accepted(self):
        res = self.pde.residual(self.params, jnp.stack([self.x, self.t], axis=1))
        assert res.shape == (8,)

    def test_returns_ndarray_not_tuple(self):
        res = self.pde.residual(self.params, jnp.stack([self.x, self.t], axis=1))
        assert isinstance(res, jnp.ndarray)

    def test_numeric_regression(self):
        xt = jnp.stack([self.x, self.t], axis=1)
        res_new = self.pde.residual(self.params, xt)

        def _legacy(params, x, t):
            xy = jnp.stack([x, t], axis=1)
            def u_s(xy_i):
                return self.model.apply(params, xy_i[None, :])[0, 0]
            H = jax.vmap(jax.hessian(u_s))(xy)
            return H[:, 1, 1] - 1.0 ** 2 * H[:, 0, 0]

        assert jnp.allclose(res_new, _legacy(self.params, self.x, self.t), atol=1e-5)


# ---------------------------------------------------------------------------
# #3 — DiffusionPDE
# ---------------------------------------------------------------------------

class TestDiffusionPDESignature:
    def setup_method(self):
        from underPINN.pde.diffusion import DiffusionPDE
        self.model, self.params = _tiny_mlp(2, 1)
        self.pde = DiffusionPDE(self.model, alpha=0.01)
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        self.x = jax.random.uniform(k1, (6,))
        self.t = jax.random.uniform(k2, (6,))

    def test_packed_array_accepted(self):
        xt  = jnp.stack([self.x, self.t], axis=1)
        res = self.pde.residual(self.params, xt)
        assert res.shape == (6,)

    def test_alpha_override(self):
        xt   = jnp.stack([self.x, self.t], axis=1)
        res1 = self.pde.residual(self.params, xt, alpha=0.01)
        res2 = self.pde.residual(self.params, xt, alpha=1.0)
        assert not jnp.allclose(res1, res2)

    def test_returns_ndarray_not_tuple(self):
        res = self.pde.residual(self.params, jnp.stack([self.x, self.t], axis=1))
        assert isinstance(res, jnp.ndarray)


# ---------------------------------------------------------------------------
# #3 — UnsteadyHeat2DPDE
# ---------------------------------------------------------------------------

class TestHeat2DPDESignature:
    def setup_method(self):
        from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE
        self.model, self.params = _tiny_mlp(3, 1)
        self.pde = UnsteadyHeat2DPDE(self.model, alpha=0.01)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
        self.x  = jax.random.uniform(k1, (7,))
        self.y  = jax.random.uniform(k2, (7,))
        self.t  = jax.random.uniform(k3, (7,))

    def test_packed_array_accepted(self):
        xyt = jnp.stack([self.x, self.y, self.t], axis=1)
        res = self.pde.residual(self.params, xyt)
        assert res.shape == (7,)

    def test_returns_ndarray_not_tuple(self):
        xyt = jnp.stack([self.x, self.y, self.t], axis=1)
        res = self.pde.residual(self.params, xyt)
        assert isinstance(res, jnp.ndarray)

    def test_numeric_regression(self):
        xyt     = jnp.stack([self.x, self.y, self.t], axis=1)
        res_new = self.pde.residual(self.params, xyt)

        # Replicate legacy code inline
        def _legacy(params, xy, t, alpha=0.01):
            xyt_ = jnp.concatenate([xy, t[:, None]], axis=1)
            def u_s(xyt_i):
                return self.model.apply(params, xyt_i[None, :])[0, 0]
            J = jax.vmap(jax.jacfwd(u_s))(xyt_)
            H = jax.vmap(jax.hessian(u_s))(xyt_)
            return J[:, 2] - alpha * (H[:, 0, 0] + H[:, 1, 1])

        xy_old = jnp.stack([self.x, self.y], axis=1)
        assert jnp.allclose(res_new, _legacy(self.params, xy_old, self.t), atol=1e-5)


# ---------------------------------------------------------------------------
# #3 — UnsteadyPipeFlowPDE
# ---------------------------------------------------------------------------

class TestPipeFlowUnsteadyPDESignature:
    def setup_method(self):
        from underPINN.pde.pipe_flow_unsteady import UnsteadyPipeFlowPDE
        self.model, self.params = _tiny_mlp(3, 1)
        self.pde = UnsteadyPipeFlowPDE(self.model, Re=10.0, R=0.5, U_max=1.0)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(9), 3)
        self.y = jax.random.uniform(k1, (5,), minval=-0.4, maxval=0.4)
        self.z = jax.random.uniform(k2, (5,), minval=-0.4, maxval=0.4)
        self.t = jax.random.uniform(k3, (5,), minval=0.0,  maxval=1.0)

    def test_packed_array_accepted(self):
        yzt = jnp.stack([self.y, self.z, self.t], axis=1)
        res = self.pde.residual(self.params, yzt)
        assert res.shape == (5,)

    def test_returns_ndarray_not_tuple(self):
        yzt = jnp.stack([self.y, self.z, self.t], axis=1)
        res = self.pde.residual(self.params, yzt)
        assert isinstance(res, jnp.ndarray)


# ---------------------------------------------------------------------------
# #3 — CompressibleEulerPDE: return type fix (tuple → stacked)
# ---------------------------------------------------------------------------

class TestCompressibleEulerReturnType:
    def setup_method(self):
        from underPINN.pde.compressible_euler import CompressibleEulerPDE
        self.model, self.params = _tiny_mlp(2, 4)
        self.pde = CompressibleEulerPDE(self.model, gamma=1.4)
        xy = jax.random.uniform(jax.random.PRNGKey(11), (6, 2))
        self.xy = xy

    def test_returns_ndarray_not_tuple(self):
        res = self.pde.residual(self.params, self.xy)
        assert isinstance(res, jnp.ndarray), "residual must return ndarray, not tuple"

    def test_shape_is_n_by_4(self):
        res = self.pde.residual(self.params, self.xy)
        assert res.shape == (6, 4), f"expected (6,4), got {res.shape}"

    def test_columns_match_legacy_tuple(self):
        """Each column of the stacked result equals the corresponding legacy scalar."""
        res = self.pde.residual(self.params, self.xy)

        # Replicate the legacy per-component computation to check columns
        import math
        from underPINN.pde.compressible_euler import CompressibleEulerPDE as _PDE

        # Re-run with a fresh PDE that exposes the old tuple logic
        gamma = 1.4
        eps   = 1e-6

        def _phys(xy_i):
            raw = self.model.apply(self.params, xy_i[None, :])[0]
            return jnp.stack([
                jax.nn.softplus(raw[0]) + eps,
                raw[1], raw[2],
                jax.nn.softplus(raw[3]) + eps,
            ])

        J   = jax.vmap(jax.jacfwd(_phys))(self.xy)
        pv  = self.pde.apply(self.params, self.xy)
        rho = pv[:, 0]; u = pv[:, 1]; v = pv[:, 2]; p = pv[:, 3]
        cont   = J[:,0,0]*u + rho*J[:,1,0] + J[:,0,1]*v + rho*J[:,2,1]
        mom_x  = rho*(u*J[:,1,0] + v*J[:,1,1]) + J[:,3,0]
        mom_y  = rho*(u*J[:,2,0] + v*J[:,2,1]) + J[:,3,1]
        energy = u*J[:,3,0] + v*J[:,3,1] + gamma*p*(J[:,1,0]+J[:,2,1])

        assert jnp.allclose(res[:, 0], cont,   atol=1e-5)
        assert jnp.allclose(res[:, 1], mom_x,  atol=1e-5)
        assert jnp.allclose(res[:, 2], mom_y,  atol=1e-5)
        assert jnp.allclose(res[:, 3], energy, atol=1e-5)


# ---------------------------------------------------------------------------
# #3 — SteadyNS3DPDE: return type fix (tuple → stacked)
# ---------------------------------------------------------------------------

class TestSteadyNS3DReturnType:
    def setup_method(self):
        from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
        self.model, self.params = _tiny_mlp(3, 4)
        self.pde  = SteadyNS3DPDE(self.model, Re=100.0)
        self.xyz  = jax.random.uniform(jax.random.PRNGKey(13), (5, 3))

    def test_returns_ndarray_not_tuple(self):
        res = self.pde.residual(self.params, self.xyz)
        assert isinstance(res, jnp.ndarray)

    def test_shape_is_n_by_4(self):
        res = self.pde.residual(self.params, self.xyz)
        assert res.shape == (5, 4)
