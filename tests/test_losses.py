"""Tests for PINN loss functions: PINNLoss, SteadyLoss, ODELoss.

Covers:
* Output structure (total + aux tuple).
* Loss components scale correctly with weights.
* RBA mode produces different (re-weighted) loss vs standard mode.
* Boundary condition term disabled when x_b is None.
* Regularisation weight is applied.
* ODELoss with and without derivative IC.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pytest

from underPINN.nn.mlp import MLP
from underPINN.losses.loss import PINNLoss
from underPINN.losses.steady_loss import SteadyLoss
from underPINN.losses.ode_loss import ODELoss


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _mlp(in_dim, out_dim, seed=0):
    model = MLP(layers=[in_dim, 8, 8, out_dim])
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((1, in_dim)))
    return model, params


# ---------------------------------------------------------------------------
# PINNLoss (space-time PDE, x_r + t_r + x_i + u_i + x_b + t_b + u_b)
# ---------------------------------------------------------------------------

class TestPINNLoss:
    def setup_method(self):
        from underPINN.pde.burgers import BurgersPDE
        self.model, self.params = _mlp(2, 1)
        pde = BurgersPDE(self.model, nu=0.01)
        self.loss = PINNLoss(self.model, pde)

        rng = jax.random.PRNGKey(1)
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        self.x_r = jax.random.uniform(k1, (20,), minval=-1.0, maxval=1.0)
        self.t_r = jax.random.uniform(k2, (20,), minval=0.0, maxval=1.0)
        self.x_i = jax.random.uniform(k3, (10,), minval=-1.0, maxval=1.0)
        self.u_i = jnp.zeros(10)
        self.x_b = jnp.array([-1.0, 1.0])
        self.t_b = jnp.array([0.5, 0.5])
        self.u_b = jnp.zeros(2)

    def test_returns_tuple(self):
        result = self.loss(self.params, self.x_r, self.t_r,
                          self.x_i, self.u_i, self.x_b, self.t_b, self.u_b)
        total, aux = result
        assert isinstance(total, jnp.ndarray)
        assert len(aux) == 4   # pde, ic, bc, reg

    def test_total_is_scalar(self):
        total, _ = self.loss(self.params, self.x_r, self.t_r,
                             self.x_i, self.u_i, self.x_b, self.t_b, self.u_b)
        assert total.shape == ()

    def test_bc_none_reduces_loss(self):
        """Disabling BC (x_b=None) should give a lower total loss."""
        total_bc, _ = self.loss(self.params, self.x_r, self.t_r,
                                self.x_i, self.u_i, self.x_b, self.t_b, self.u_b)
        total_no_bc, _ = self.loss(self.params, self.x_r, self.t_r,
                                   self.x_i, self.u_i)
        assert float(total_no_bc) <= float(total_bc) + 1e-6

    def test_higher_ic_weight_increases_total(self):
        from underPINN.pde.burgers import BurgersPDE
        pde = BurgersPDE(self.model, nu=0.01)
        loss_low  = PINNLoss(self.model, pde, ic_weight=1.0)
        loss_high = PINNLoss(self.model, pde, ic_weight=1000.0)
        t_low,  _ = loss_low(self.params, self.x_r, self.t_r,
                             self.x_i, self.u_i)
        t_high, _ = loss_high(self.params, self.x_r, self.t_r,
                              self.x_i, self.u_i)
        assert float(t_high) >= float(t_low)

    def test_reg_zero_by_default(self):
        _, (_, _, _, reg) = self.loss(self.params, self.x_r, self.t_r,
                                      self.x_i, self.u_i)
        assert float(reg) == 0.0

    def test_reg_nonzero_when_set(self):
        from underPINN.pde.burgers import BurgersPDE
        pde = BurgersPDE(self.model, nu=0.01)
        loss = PINNLoss(self.model, pde, reg_weight=1.0)
        _, (_, _, _, reg) = loss(self.params, self.x_r, self.t_r, self.x_i, self.u_i)
        assert float(reg) > 0.0

    def test_rba_differs_from_standard(self):
        from underPINN.pde.burgers import BurgersPDE
        pde = BurgersPDE(self.model, nu=0.01)
        loss_std = PINNLoss(self.model, pde, rba=False)
        loss_rba = PINNLoss(self.model, pde, rba=True)
        t_std, _ = loss_std(self.params, self.x_r, self.t_r, self.x_i, self.u_i)
        t_rba, _ = loss_rba(self.params, self.x_r, self.t_r, self.x_i, self.u_i)
        # RBA re-weights per-point: result should differ from uniform mean
        # (Not guaranteed to be larger/smaller, only different)
        assert not jnp.allclose(t_std, t_rba, atol=1e-8)

    def test_grad_flows_through_loss(self):
        def objective(p):
            total, _ = self.loss(p, self.x_r, self.t_r, self.x_i, self.u_i)
            return total
        grads = jax.grad(objective)(self.params)
        assert len(jax.tree_util.tree_leaves(grads)) > 0


# ---------------------------------------------------------------------------
# SteadyLoss (no time, xy_r + xy_b + u_b)
# ---------------------------------------------------------------------------

class TestSteadyLoss:
    def setup_method(self):
        from underPINN.pde.heat import SteadyHeatPDE
        self.model, self.params = _mlp(2, 1)
        pde = SteadyHeatPDE(self.model)
        self.loss = SteadyLoss(self.model, pde)
        rng = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(rng)
        self.xy_r = jax.random.uniform(k1, (15, 2))
        self.xy_b = jax.random.uniform(k2, (8,  2))
        self.u_b  = jnp.zeros(8)

    def test_returns_tuple(self):
        total, aux = self.loss(self.params, self.xy_r, self.xy_b, self.u_b)
        assert isinstance(total, jnp.ndarray)
        assert len(aux) == 3   # pde, bc, reg

    def test_scalar_total(self):
        total, _ = self.loss(self.params, self.xy_r, self.xy_b, self.u_b)
        assert total.shape == ()

    def test_bc_weight_scales_loss(self):
        from underPINN.pde.heat import SteadyHeatPDE
        pde = SteadyHeatPDE(self.model)
        loss1 = SteadyLoss(self.model, pde, bc_weight=1.0)
        loss2 = SteadyLoss(self.model, pde, bc_weight=100.0)
        t1, _ = loss1(self.params, self.xy_r, self.xy_b, self.u_b)
        t2, _ = loss2(self.params, self.xy_r, self.xy_b, self.u_b)
        assert float(t2) >= float(t1)

    def test_grad_flows(self):
        def obj(p):
            total, _ = self.loss(p, self.xy_r, self.xy_b, self.u_b)
            return total
        grads = jax.grad(obj)(self.params)
        assert len(jax.tree_util.tree_leaves(grads)) > 0


# ---------------------------------------------------------------------------
# ODELoss (time-only, first- and second-order)
# ---------------------------------------------------------------------------

class TestODELoss:
    def setup_method(self):
        from underPINN.pde.ode import ExponentialDecayODE
        self.model, self.params = _mlp(1, 1)
        pde = ExponentialDecayODE(self.model, lam=1.0)
        self.loss = ODELoss(self.model, pde)
        self.t_r  = jnp.linspace(0.0, 1.0, 20)[:, None]   # (20, 1) for ODE
        self.t_ic = jnp.array([[0.0]])
        self.u_ic = jnp.array([1.0])

    def test_returns_tuple(self):
        total, aux = self.loss(self.params, self.t_r, self.t_ic, self.u_ic)
        assert isinstance(total, jnp.ndarray)
        assert len(aux) == 4  # pde, ic, ic_dot, reg

    def test_scalar_total(self):
        total, _ = self.loss(self.params, self.t_r, self.t_ic, self.u_ic)
        assert total.shape == ()

    def test_ic_weight_scales_loss(self):
        from underPINN.pde.ode import ExponentialDecayODE
        pde = ExponentialDecayODE(self.model, lam=1.0)
        lo_w = ODELoss(self.model, pde, ic_weight=1.0)
        hi_w = ODELoss(self.model, pde, ic_weight=1000.0)
        t1, _ = lo_w(self.params, self.t_r, self.t_ic, self.u_ic)
        t2, _ = hi_w(self.params, self.t_r, self.t_ic, self.u_ic)
        assert float(t2) >= float(t1)

    def test_grad_flows(self):
        def obj(p):
            total, _ = self.loss(p, self.t_r, self.t_ic, self.u_ic)
            return total
        grads = jax.grad(obj)(self.params)
        assert len(jax.tree_util.tree_leaves(grads)) > 0

    def test_ic_dot_term_zero_when_weight_zero(self):
        _, (_, _, ic_dot_l, _) = self.loss(self.params, self.t_r, self.t_ic, self.u_ic)
        assert float(ic_dot_l) == 0.0
