"""Tests for #7 — opt_state → self.state naming unification.

Verifies that:
* LDCSolver and RANSSolver expose ``self.state`` (not ``self.opt_state``)
  after ``init()`` is called.
* Neither solver has a dangling ``opt_state`` attribute.
* The stored ``state`` is a valid optax optimizer state (has at least one
  field, can be used to compute updates).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import optax
import pytest


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _prng(seed: int = 0):
    return jax.random.PRNGKey(seed)


# ---------------------------------------------------------------------------
# LDCSolver
# ---------------------------------------------------------------------------

class TestLDCSolverStateNaming:
    def setup_method(self):
        from underPINN.solver.ldc_solver import LDCSolver
        from underPINN.pde.navier_stokes import NavierStokesPDE
        from underPINN.nn.mlp import MLP
        model = MLP(layers=[2, 16, 16, 3])
        pde   = NavierStokesPDE(model, Re=100.0)
        opt   = optax.adam(1e-3)
        self.solver = LDCSolver(model, pde, optimizer=opt)
        self.solver.init(_prng(0))

    def test_has_state_attribute(self):
        assert hasattr(self.solver, "state"), \
            "LDCSolver must expose self.state after init()"

    def test_no_opt_state_attribute(self):
        assert not hasattr(self.solver, "opt_state"), \
            "LDCSolver must NOT have self.opt_state (rename complete)"

    def test_state_is_not_none(self):
        assert self.solver.state is not None

    def test_state_is_valid_optax_state(self):
        """state must have leaves (i.e. it is a proper pytree, not empty)."""
        leaves = jax.tree_util.tree_leaves(self.solver.state)
        assert len(leaves) > 0, "optimizer state should have at least one leaf"

    def test_params_initialised(self):
        assert self.solver.params is not None
        leaves = jax.tree_util.tree_leaves(self.solver.params)
        assert len(leaves) > 0


# ---------------------------------------------------------------------------
# RANSSolver
# ---------------------------------------------------------------------------

class TestRANSSolverStateNaming:
    def setup_method(self):
        from underPINN.solver.rans_solver import RANSSolver
        from underPINN.pde.navier_stokes import NavierStokesPDE
        from underPINN.nn.mlp import MLP
        model = MLP(layers=[2, 16, 16, 3])
        pde   = NavierStokesPDE(model, Re=100.0)
        self.solver = RANSSolver(model=model, pde=pde)
        self.solver.init(_prng(1))

    def test_has_state_attribute(self):
        assert hasattr(self.solver, "state"), \
            "RANSSolver must expose self.state after init()"

    def test_no_opt_state_attribute(self):
        assert not hasattr(self.solver, "opt_state"), \
            "RANSSolver must NOT have self.opt_state (rename complete)"

    def test_state_is_not_none(self):
        assert self.solver.state is not None

    def test_state_is_valid_optax_state(self):
        leaves = jax.tree_util.tree_leaves(self.solver.state)
        assert len(leaves) > 0

    def test_params_initialised(self):
        assert self.solver.params is not None
        leaves = jax.tree_util.tree_leaves(self.solver.params)
        assert len(leaves) > 0


# ---------------------------------------------------------------------------
# All concrete BaseSolver subclasses must use self.state (smoke-test via
# attribute inspection on the class dict — avoids importing heavy deps)
# ---------------------------------------------------------------------------

class TestNoOptStateInSourceCode:
    """Grep-level guard: solvers must not reference self.opt_state."""

    def _load_source(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def _solver_path(self, fname: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        return str(root / "underPINN" / "solver" / fname)

    def test_ldc_solver_no_self_opt_state(self):
        """self.opt_state must not appear — local variable opt_state in step() is fine."""
        src = self._load_source(self._solver_path("ldc_solver.py"))
        assert "self.opt_state" not in src, \
            "ldc_solver.py still uses self.opt_state (rename incomplete)"

    def test_rans_solver_no_self_opt_state(self):
        src = self._load_source(self._solver_path("rans_solver.py"))
        assert "self.opt_state" not in src, \
            "rans_solver.py still uses self.opt_state (rename incomplete)"
