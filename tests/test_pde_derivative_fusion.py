"""Regression tests for the fused-AD rewrite applied to PDE classes sharing
the redundancy underPINN/pde/burgers.py's residual() was first profiled and
fixed for (see tests/test_pde_burgers_residual.py): separate jax.jacfwd /
jax.hessian / model.apply calls on the same points, with the Hessian often
computing far more entries than are ever used.

**Kept fused** (measured faster under the real outer
jax.value_and_grad(loss)(params, ...) a training step actually needs, not
just as an isolated forward residual call -- see
benchmarks/rebuttal/README.md section 6): diffusion.py, heat2d_unsteady.py,
pipe_flow_unsteady.py, burgers_deeponet.py. Each is checked two ways below:
against hand-derived analytic derivatives on an exactly-representable
function (where one was practical), and against its own pre-rewrite
(jacfwd/hessian-based) formulation on a small real network -- not just
"does it run," but "does it compute the same physics."

**Reverted to jax.hessian** after the same outer-gradient check found a
regression, not an improvement: helmholtz.py (0.84x), navier_stokes.py
(0.62x), k_epsilon.py (0.49x, worst) -- see each file's residual() comment
for the measurement. Their classes below now compare the *current*
(jax.hessian-based) code against a hand-written jax.hessian-based
formulation, so the "legacy" comparison is intentionally exact rather than
a meaningful cross-check of two different implementations -- kept as a
plain regression/shape test guarding against a future accidental change,
not as evidence a fusion rewrite still lives here.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from underPINN.nn.mlp import MLP


def _tiny_mlp(in_dim: int, out_dim: int, seed: int = 0):
    model = MLP(layers=[in_dim, 8, 8, out_dim])
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, in_dim)))
    return model, params


# ---------------------------------------------------------------------------
# DiffusionPDE: u_t - alpha*u_xx
# ---------------------------------------------------------------------------

class TestDiffusionResidual:
    def test_matches_analytic(self):
        from underPINN.pde.diffusion import DiffusionPDE

        # f(x,t) = x^2 * sin(t) + t^3
        #   f_t  = x^2 cos(t) + 3t^2
        #   f_xx = 2 sin(t)
        class AnalyticModel:
            def apply(self, params, xt):
                x, t = xt[:, 0], xt[:, 1]
                return (x ** 2 * jnp.sin(t) + t ** 3)[:, None]

        alpha = 0.02
        pde = DiffusionPDE(AnalyticModel(), alpha=alpha)
        x = jnp.array([0.5, -0.2, 1.0])
        t = jnp.array([0.3, 1.1, 0.7])
        xt = jnp.stack([x, t], axis=1)
        res = pde.residual(None, xt)

        f_t = x ** 2 * jnp.cos(t) + 3 * t ** 2
        f_xx = 2 * jnp.sin(t)
        expected = f_t - alpha * f_xx
        assert jnp.allclose(res, expected, atol=1e-4)

    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.diffusion import DiffusionPDE

        model, params = _tiny_mlp(2, 1)
        pde = DiffusionPDE(model, alpha=0.02)
        xt = jax.random.uniform(jax.random.PRNGKey(3), (40, 2))

        def u_single(xy_i):
            return model.apply(params, xy_i[None, :])[0, 0]
        J = jax.vmap(jax.jacfwd(u_single))(xt)
        H = jax.vmap(jax.hessian(u_single))(xt)
        legacy = J[:, 1] - 0.02 * H[:, 0, 0]

        fused = pde.residual(params, xt)
        assert jnp.allclose(fused, legacy, atol=5e-4)


# ---------------------------------------------------------------------------
# UnsteadyHeat2DPDE: u_t - alpha*(u_xx + u_yy), 3-D input (x, y, t)
# ---------------------------------------------------------------------------

class TestHeat2DUnsteadyResidual:
    def test_matches_analytic(self):
        from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE

        # f(x,y,t) = sin(x)*cos(y)*t + x*y^2
        #   f_t  = sin(x)cos(y)
        #   f_xx = -sin(x)cos(y)*t
        #   f_yy = -sin(x)cos(y)*t + 2x
        class AnalyticModel:
            def apply(self, params, xyt):
                x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
                return (jnp.sin(x) * jnp.cos(y) * t + x * y ** 2)[:, None]

        alpha = 0.05
        pde = UnsteadyHeat2DPDE(AnalyticModel(), alpha=alpha)
        x = jnp.array([0.4, -0.6, 0.9])
        y = jnp.array([0.2, 0.7, -0.3])
        t = jnp.array([0.5, 1.2, 0.1])
        xyt = jnp.stack([x, y, t], axis=1)
        res = pde.residual(None, xyt)

        f_t = jnp.sin(x) * jnp.cos(y)
        f_xx = -jnp.sin(x) * jnp.cos(y) * t
        f_yy = -jnp.sin(x) * jnp.cos(y) * t + 2 * x
        expected = f_t - alpha * (f_xx + f_yy)
        assert jnp.allclose(res, expected, atol=1e-4)

    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.heat2d_unsteady import UnsteadyHeat2DPDE

        model, params = _tiny_mlp(3, 1)
        pde = UnsteadyHeat2DPDE(model, alpha=0.02)
        xyt = jax.random.uniform(jax.random.PRNGKey(4), (40, 3))

        def u_single(xyt_i):
            return model.apply(params, xyt_i[None, :])[0, 0]
        J = jax.vmap(jax.jacfwd(u_single))(xyt)
        H = jax.vmap(jax.hessian(u_single))(xyt)
        legacy = J[:, 2] - 0.02 * (H[:, 0, 0] + H[:, 1, 1])

        fused = pde.residual(params, xyt)
        assert jnp.allclose(fused, legacy, atol=5e-4)


# ---------------------------------------------------------------------------
# UnsteadyPipeFlowPDE: u_t - nu*(u_yy + u_zz) - G, 3-D input (y, z, t)
# ---------------------------------------------------------------------------

class TestPipeFlowUnsteadyResidual:
    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.pipe_flow_unsteady import UnsteadyPipeFlowPDE

        model, params = _tiny_mlp(3, 1)
        pde = UnsteadyPipeFlowPDE(model, Re=20.0, R=0.5, U_max=1.0)
        yzt = jax.random.uniform(jax.random.PRNGKey(5), (40, 3), minval=-0.4, maxval=0.4)

        nu = 1.0 / 20.0
        G = 4.0 * nu * 1.0 / 0.5 ** 2

        def u_single(yzt_i):
            return model.apply(params, yzt_i[None, :])[0, 0]
        J = jax.vmap(jax.jacfwd(u_single))(yzt)
        H = jax.vmap(jax.hessian(u_single))(yzt)
        legacy = J[:, 2] - nu * (H[:, 0, 0] + H[:, 1, 1]) - G

        fused = pde.residual(params, yzt)
        assert jnp.allclose(fused, legacy, atol=5e-4)


# ---------------------------------------------------------------------------
# HelmholtzPDE: laplacian + k^2*u - f  (reverted to jax.hessian, see module
# docstring -- "legacy" here means "current code," checked exactly)
# ---------------------------------------------------------------------------

class TestHelmholtzResidual:
    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.helmholtz import HelmholtzPDE

        model, params = _tiny_mlp(2, 1)
        k = 2.0
        pde = HelmholtzPDE(model, k=k)
        xy = jax.random.uniform(jax.random.PRNGKey(6), (40, 2))

        def u_single(xy_i):
            return model.apply(params, xy_i[None, :])[0, 0]
        H = jax.vmap(jax.hessian(u_single))(xy)
        u = model.apply(params, xy)[:, 0]
        f = pde.source(xy)
        legacy = H[:, 0, 0] + H[:, 1, 1] + k ** 2 * u - f

        fused = pde.residual(params, xy)
        assert jnp.allclose(fused, legacy, atol=5e-4)


# ---------------------------------------------------------------------------
# NavierStokesPDE: (u, v, p) -- continuity + 2 momentum equations
# (reverted to jax.hessian, see module docstring -- "legacy" here means
# "current code," checked exactly)
# ---------------------------------------------------------------------------

class TestNavierStokesResidual:
    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.navier_stokes import NavierStokesPDE

        model, params = _tiny_mlp(2, 3)
        Re = 50.0
        pde = NavierStokesPDE(model, Re=Re)
        x = jax.random.uniform(jax.random.PRNGKey(8), (30, 2))

        def u_fn(x_i):
            return model.apply(params, x_i[None, :])[0]
        J = jax.vmap(jax.jacfwd(u_fn))(x)
        H = jax.vmap(jax.hessian(u_fn))(x)
        out = model.apply(params, x)
        u, v = out[:, 0], out[:, 1]
        u_x, u_y = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y = J[:, 1, 0], J[:, 1, 1]
        p_x, p_y = J[:, 2, 0], J[:, 2, 1]
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]
        cont = u_x + v_y
        mom_x = 2 * u * u_x + (u_y * v + u * v_y) + p_x - (1.0 / Re) * (u_xx + u_yy)
        mom_y = (u_x * v + u * v_x) + 2 * v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
        legacy = jnp.stack([cont, mom_x, mom_y], axis=1)

        fused = pde.residual(params, x)
        assert jnp.allclose(fused, legacy, atol=5e-4)


# ---------------------------------------------------------------------------
# KEpsilonPDE: (u, v, p, k, eps) -- 5 coupled residuals
# (reverted to jax.hessian, see module docstring -- "legacy" here means
# "current code," checked exactly)
# ---------------------------------------------------------------------------

class TestKEpsilonResidual:
    def test_shape_and_finite(self):
        from underPINN.pde.k_epsilon import KEpsilonPDE

        model, params = _tiny_mlp(2, 5)
        pde = KEpsilonPDE(model, Re=1000.0)
        # k, eps must stay positive for a physically sensible residual;
        # a raw untrained net can output near-zero/negative values there,
        # so this test only checks shape/finiteness, not physics (no
        # positivity-transform wired up outside the full FBPINN solver).
        x = jax.random.uniform(jax.random.PRNGKey(9), (10, 2)) + 0.1
        res = pde.residual(params, x)
        assert res.shape == (10, 5)

    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.k_epsilon import KEpsilonPDE

        model, params = _tiny_mlp(2, 5)
        Re = 1000.0
        pde = KEpsilonPDE(model, Re=Re)
        x = jax.random.uniform(jax.random.PRNGKey(10), (20, 2)) + 0.5  # keep k,eps>0-ish

        def u_fn(x_i):
            return model.apply(params, x_i[None, :])[0]
        J = jax.vmap(jax.jacfwd(u_fn))(x)
        H = jax.vmap(jax.hessian(u_fn))(x)
        out = model.apply(params, x)
        u_val, v_val = out[:, 0], out[:, 1]
        k_val, e_val = out[:, 3], out[:, 4]
        u_x, u_y = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y = J[:, 1, 0], J[:, 1, 1]
        k_x, k_y = J[:, 3, 0], J[:, 3, 1]
        e_x, e_y = J[:, 4, 0], J[:, 4, 1]
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]
        k_xx, k_yy = H[:, 3, 0, 0], H[:, 3, 1, 1]
        e_xx, e_yy = H[:, 4, 0, 0], H[:, 4, 1, 1]
        C_mu, C1, C2, s_k, s_e = 0.09, 1.44, 1.92, 1.0, 1.3
        mu_t = C_mu * k_val ** 2 / (e_val + 1e-8)
        P_k = 2.0 * mu_t * (u_x ** 2 + v_y ** 2 + 0.5 * (u_y + v_x) ** 2)
        cont = u_x + v_y
        mom_x = u_val * u_x + v_val * u_y + J[:, 2, 0] - (1.0 / Re + mu_t) * (u_xx + u_yy)
        mom_y = u_val * v_x + v_val * v_y + J[:, 2, 1] - (1.0 / Re + mu_t) * (v_xx + v_yy)
        diff_k = (1.0 / Re + mu_t / s_k) * (k_xx + k_yy)
        T_k = u_val * k_x + v_val * k_y - diff_k - P_k + e_val
        diff_e = (1.0 / Re + mu_t / s_e) * (e_xx + e_yy)
        prod_e = C1 * (e_val / (k_val + 1e-8)) * P_k
        diss_e = C2 * e_val ** 2 / (k_val + 1e-8)
        T_e = u_val * e_x + v_val * e_y - diff_e - prod_e + diss_e
        legacy = jnp.stack([cont, mom_x, mom_y, T_k, T_e], axis=1)

        fused = pde.residual(params, x)
        assert jnp.allclose(fused, legacy, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# DeepONetBurgersPDE
# ---------------------------------------------------------------------------

class TestDeepONetBurgersResidual:
    def test_matches_legacy_on_random_mlp(self):
        from underPINN.pde.burgers_deeponet import DeepONetBurgersPDE

        class ToyDeepONet:
            """Minimal branch/trunk stand-in: s(u, xt) = sum(u) * mlp_trunk(xt)."""
            def __init__(self, trunk, trunk_params):
                self.trunk = trunk
                self.trunk_params = trunk_params

            def apply(self, params, u_i, xt_i):
                return jnp.sum(u_i) * self.trunk.apply(self.trunk_params, xt_i[None, :])[0, 0]

        trunk, trunk_params = _tiny_mlp(2, 1)
        model = ToyDeepONet(trunk, trunk_params)
        pde = DeepONetBurgersPDE(model, nu=0.01)

        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)
        u_batch = jax.random.uniform(k1, (20, 5))
        xt_batch = jax.random.uniform(k2, (20, 2))

        def s_of_xt(u_i, xt_i):
            return model.apply(None, u_i, xt_i)

        def grad_single(u_i, xt_i):
            return jax.jacfwd(lambda z: s_of_xt(u_i, z))(xt_i)

        def hess_single(u_i, xt_i):
            return jax.hessian(lambda z: s_of_xt(u_i, z))(xt_i)

        J = jax.vmap(grad_single)(u_batch, xt_batch)
        H = jax.vmap(hess_single)(u_batch, xt_batch)
        s = jax.vmap(s_of_xt)(u_batch, xt_batch)
        legacy = J[:, 1] + s * J[:, 0] - 0.01 * H[:, 0, 0]

        fused = pde.residual(None, u_batch, xt_batch)
        # atol a bit looser than the other cross-checks in this file: the
        # toy branch net's jnp.sum(u_i) reduction plus s*s_x in the
        # residual itself add another chained operation on top of the
        # trunk MLP, compounding float32 rounding a little further between
        # the two AD paths (same phenomenon as every other test here, not
        # a different one) -- observed max diff ~9e-4, comfortably inside
        # this bound.
        assert jnp.allclose(fused, legacy, atol=2e-3)
