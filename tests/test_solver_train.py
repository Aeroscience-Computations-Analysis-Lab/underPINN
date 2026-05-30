"""Short end-to-end training tests for FBPINNSolver, ODESolver, SteadySolver.

Each test runs only a handful of epochs (5–10) to keep CI fast.
We verify:
* Loss history grows by exactly the number of epochs run.
* Parameters change (network actually updates).
* Config path and legacy kwargs both work.
* EarlyStopping fires and terminates training cleanly.
* RAR-D resampling runs without error (SteadySolver).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from underPINN.nn.mlp import MLP
from underPINN.core.config import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_EPOCHS = 5   # keep very short for CI


def _allclose_tree(t1, t2):
    leaves1 = jax.tree_util.tree_leaves(t1)
    leaves2 = jax.tree_util.tree_leaves(t2)
    return all(jnp.allclose(a, b) for a, b in zip(leaves1, leaves2))


# ---------------------------------------------------------------------------
# ODESolver — ExponentialDecayODE
# ---------------------------------------------------------------------------

class TestODESolverTrain:
    def setup_method(self):
        from underPINN.pde.ode import ExponentialDecayODE
        from underPINN.losses.ode_loss import ODELoss
        from underPINN.solver.ode_solver import ODESolver

        self.model = MLP(layers=[1, 16, 16, 1])
        pde   = ExponentialDecayODE(self.model, lam=1.0)
        loss  = ODELoss(self.model, pde)
        self.solver = ODESolver(self.model, pde, loss)
        self.solver.init(jax.random.PRNGKey(0))

        self.t_r  = jnp.linspace(0.0, 2.0, 50)
        self.t_ic = jnp.array([0.0])
        self.u_ic = jnp.array([1.0])

    def test_loss_hist_grows(self):
        self.solver.train(self.t_r, self.t_ic, self.u_ic, epochs=N_EPOCHS)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_params_change(self):
        p0 = jax.tree_util.tree_leaves(self.solver.params)[0].copy()
        self.solver.train(self.t_r, self.t_ic, self.u_ic, epochs=N_EPOCHS)
        p1 = jax.tree_util.tree_leaves(self.solver.params)[0]
        assert not jnp.allclose(p0, p1), "params did not change after training"

    def test_config_path(self):
        cfg = TrainingConfig(epochs=N_EPOCHS, lr=1e-3)
        self.solver.train(self.t_r, self.t_ic, self.u_ic, config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_early_stopping_terminates(self):
        from underPINN.callbacks.early_stopping import EarlyStopping
        # patience=1 → should fire after 2 epochs with no improvement
        cfg = TrainingConfig(
            epochs=1000,
            lr=1e-3,
            callbacks=[EarlyStopping(patience=1, min_delta=1e10)],
        )
        self.solver.train(self.t_r, self.t_ic, self.u_ic, config=cfg)
        # Should stop well before 1000
        assert len(self.solver.loss_hist) < 100

    def test_scan_mode(self):
        cfg = TrainingConfig(epochs=N_EPOCHS * 4, lr=1e-3, n_scan_steps=N_EPOCHS)
        self.solver.train(self.t_r, self.t_ic, self.u_ic, config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS * 4


# ---------------------------------------------------------------------------
# FBPINNSolver — BurgersPDE
# ---------------------------------------------------------------------------

class TestFBPINNSolverTrain:
    def setup_method(self):
        from underPINN.pde.burgers import BurgersPDE
        from underPINN.losses.loss import PINNLoss
        from underPINN.solver.fbpinn import FBPINNSolver

        self.model = MLP(layers=[2, 16, 16, 1])
        pde   = BurgersPDE(self.model, nu=0.01)
        loss  = PINNLoss(self.model, pde)
        self.solver = FBPINNSolver(self.model, pde, loss)
        self.solver.init(jax.random.PRNGKey(1))

        rng = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(rng, 3)
        self.x_r = jax.random.uniform(k1, (50,), minval=-1.0, maxval=1.0)
        self.t_r = jax.random.uniform(k2, (50,), minval=0.0,  maxval=1.0)
        self.x_i = jax.random.uniform(k3, (10,), minval=-1.0, maxval=1.0)
        self.u_i = jnp.zeros(10)
        self.x_b = jnp.array([-1.0, 1.0])
        self.t_b = jnp.array([0.5, 0.5])
        self.u_b = jnp.zeros(2)

    def _train(self, **kw):
        self.solver.train(
            self.x_r, self.t_r, self.x_i, self.u_i,
            self.x_b, self.t_b, self.u_b, **kw)

    def test_loss_hist_grows(self):
        self._train(epochs=N_EPOCHS)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_params_change(self):
        p0 = jax.tree_util.tree_leaves(self.solver.params)[0].copy()
        self._train(epochs=N_EPOCHS)
        p1 = jax.tree_util.tree_leaves(self.solver.params)[0]
        assert not jnp.allclose(p0, p1)

    def test_config_path(self):
        cfg = TrainingConfig(epochs=N_EPOCHS, lr=1e-3, batch_r=30, batch_i=10, batch_b=2)
        self._train(config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_scan_mode(self):
        cfg = TrainingConfig(
            epochs=N_EPOCHS * 4, lr=1e-3,
            batch_r=30, batch_i=10, batch_b=2,
            n_scan_steps=N_EPOCHS,
        )
        self._train(config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS * 4

    def test_rar_d_runs_without_error(self):
        """resample_period=1 triggers RAR-D on every epoch — just check no error."""
        cfg = TrainingConfig(
            epochs=N_EPOCHS, lr=1e-3,
            batch_r=30, batch_i=10, batch_b=2,
            resample_period=2,
        )
        self._train(config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_early_stopping(self):
        from underPINN.callbacks.early_stopping import EarlyStopping
        cfg = TrainingConfig(
            epochs=1000, lr=1e-3,
            batch_r=30, batch_i=10, batch_b=2,
            callbacks=[EarlyStopping(patience=1, min_delta=1e10)],
        )
        self._train(config=cfg)
        assert len(self.solver.loss_hist) < 100


# ---------------------------------------------------------------------------
# SteadySolver — SteadyHeatPDE
# ---------------------------------------------------------------------------

class TestSteadySolverTrain:
    def setup_method(self):
        from underPINN.pde.heat import SteadyHeatPDE
        from underPINN.losses.steady_loss import SteadyLoss
        from underPINN.solver.steady_solver import SteadySolver

        self.model = MLP(layers=[2, 16, 16, 1])
        pde   = SteadyHeatPDE(self.model)
        loss  = SteadyLoss(self.model, pde)
        self.solver = SteadySolver(self.model, pde, loss)
        self.solver.init(jax.random.PRNGKey(3))

        rng = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(rng)
        self.xy_r = jax.random.uniform(k1, (40, 2))
        self.xy_b = jax.random.uniform(k2, (20, 2))
        self.u_b  = jnp.zeros(20)

    def test_loss_hist_grows(self):
        self.solver.train(self.xy_r, self.xy_b, self.u_b, epochs=N_EPOCHS)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_params_change(self):
        p0 = jax.tree_util.tree_leaves(self.solver.params)[0].copy()
        self.solver.train(self.xy_r, self.xy_b, self.u_b, epochs=N_EPOCHS)
        p1 = jax.tree_util.tree_leaves(self.solver.params)[0]
        assert not jnp.allclose(p0, p1)

    def test_config_path(self):
        cfg = TrainingConfig(epochs=N_EPOCHS, lr=1e-3, batch_r=30, batch_b=15)
        self.solver.train(self.xy_r, self.xy_b, self.u_b, config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_rar_d_runs_without_error(self):
        """RAR-D on SteadySolver with packed xy_r — must not raise."""
        cfg = TrainingConfig(
            epochs=N_EPOCHS, lr=1e-3, batch_r=30, batch_b=15,
            resample_period=2,
        )
        self.solver.train(self.xy_r, self.xy_b, self.u_b, config=cfg)
        assert len(self.solver.loss_hist) == N_EPOCHS

    def test_rar_d_custom_sampler(self):
        """Custom candidate_sampler (packed) is called during RAR-D."""
        called = [0]
        def sampler(n, key):
            called[0] += 1
            return jax.random.uniform(key, (n, 2))

        cfg = TrainingConfig(
            epochs=N_EPOCHS, lr=1e-3, batch_r=30, batch_b=15,
            resample_period=1,
            candidate_sampler=sampler,
        )
        self.solver.train(self.xy_r, self.xy_b, self.u_b, config=cfg)
        assert called[0] == N_EPOCHS   # fires every epoch

    def test_early_stopping(self):
        from underPINN.callbacks.early_stopping import EarlyStopping
        cfg = TrainingConfig(
            epochs=1000, lr=1e-3, batch_r=30, batch_b=15,
            callbacks=[EarlyStopping(patience=1, min_delta=1e10)],
        )
        self.solver.train(self.xy_r, self.xy_b, self.u_b, config=cfg)
        assert len(self.solver.loss_hist) < 100
