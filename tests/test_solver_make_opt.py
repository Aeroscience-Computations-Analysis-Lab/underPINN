"""Tests for #10 — _make_opt deduplication onto BaseSolver.

Verifies that:
* ``_make_opt`` is accessible on every concrete solver (inherited, not
  duplicated locally).
* No concrete solver module defines its own ``_make_opt`` (source-level
  guard).
* The method returns the correct optax transform depending on whether a
  schedule is supplied.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import optax
import pytest

from underPINN.core.base import BaseSolver


# ---------------------------------------------------------------------------
# Behavioural tests (via BaseSolver directly — no need to instantiate)
# ---------------------------------------------------------------------------

class TestMakeOptBehaviour:
    """_make_opt is a @staticmethod so we can call it without an instance."""

    def test_no_schedule_returns_adam(self):
        opt = BaseSolver._make_opt(lr=1e-3, lr_schedule=None)
        # Adam produces a named tuple / GradientTransformation with init/update
        assert hasattr(opt, "init") and hasattr(opt, "update")

    def test_no_schedule_produces_usable_state(self):
        opt    = BaseSolver._make_opt(lr=1e-3, lr_schedule=None)
        params = {"w": jnp.zeros((4,))}
        state  = opt.init(params)
        grads  = {"w": jnp.ones((4,))}
        updates, _ = opt.update(grads, state)
        # Adam with lr=1e-3 should produce small updates
        assert jnp.all(jnp.abs(updates["w"]) < 0.01)

    def test_with_schedule_returns_chain(self):
        sched  = optax.cosine_decay_schedule(1e-3, decay_steps=1000)
        opt    = BaseSolver._make_opt(lr=1e-3, lr_schedule=sched)
        # chain transforms have 'inner_state' in their update closure
        assert hasattr(opt, "init") and hasattr(opt, "update")

    def test_with_schedule_produces_usable_state(self):
        sched  = optax.cosine_decay_schedule(1e-3, decay_steps=1000)
        opt    = BaseSolver._make_opt(lr=1e-3, lr_schedule=sched)
        params = {"w": jnp.zeros((4,))}
        state  = opt.init(params)
        grads  = {"w": jnp.ones((4,))}
        updates, new_state = opt.update(grads, state)
        assert updates["w"].shape == (4,)

    def test_schedule_and_no_schedule_give_different_updates(self):
        """Chain-with-schedule should differ from plain Adam after 1 step."""
        sched   = optax.constant_schedule(2.0)   # scale by 2 to make it visibly different
        opt_sc  = BaseSolver._make_opt(lr=1e-3, lr_schedule=sched)
        opt_no  = BaseSolver._make_opt(lr=1e-3, lr_schedule=None)
        params  = {"w": jnp.zeros((4,))}
        grads   = {"w": jnp.ones((4,))}
        upd_sc, _  = opt_sc.update(grads, opt_sc.init(params))
        upd_no, _  = opt_no.update(grads, opt_no.init(params))
        assert not jnp.allclose(upd_sc["w"], upd_no["w"]), \
            "Scheduled and non-scheduled opts should differ"


# ---------------------------------------------------------------------------
# Inheritance: _make_opt defined ONLY on BaseSolver, not on subclasses
# ---------------------------------------------------------------------------

class TestMakeOptNotDuplicated:
    """Verify no subclass re-defines _make_opt (would shadow the base)."""

    SOLVER_MODULES = [
        ("underPINN.solver.fbpinn",       "FBPINNSolver"),
        ("underPINN.solver.ode_solver",   "ODESolver"),
        ("underPINN.solver.steady_solver","SteadySolver"),
        ("underPINN.solver.ldc_solver",   "LDCSolver"),
        ("underPINN.solver.rans_solver",  "RANSSolver"),
    ]

    def test_make_opt_not_in_subclass_dict(self):
        import importlib
        for mod_path, cls_name in self.SOLVER_MODULES:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            assert "_make_opt" not in cls.__dict__, (
                f"{cls_name} re-defines _make_opt in its own __dict__; "
                "it should inherit from BaseSolver instead"
            )

    def test_all_subclasses_inherit_make_opt(self):
        import importlib
        for mod_path, cls_name in self.SOLVER_MODULES:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            assert hasattr(cls, "_make_opt"), \
                f"{cls_name} does not have _make_opt (not inheriting BaseSolver?)"
            # Confirm the method resolves to BaseSolver's version
            assert cls._make_opt is BaseSolver._make_opt, (
                f"{cls_name}._make_opt is not the same object as BaseSolver._make_opt"
            )


# ---------------------------------------------------------------------------
# Source-level guard: no 'def _make_opt' in any solver file except base.py
# ---------------------------------------------------------------------------

class TestMakeOptSourceGuard:
    SOLVER_FILES = [
        "fbpinn.py",
        "ode_solver.py",
        "steady_solver.py",
        "ldc_solver.py",
        "rans_solver.py",
    ]

    def _solver_src(self, fname: str) -> str:
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        p = root / "underPINN" / "solver" / fname
        return p.read_text()

    @pytest.mark.parametrize("fname", SOLVER_FILES)
    def test_no_def_make_opt_in_solver(self, fname):
        src = self._solver_src(fname)
        assert "def _make_opt" not in src, \
            f"{fname} still defines its own _make_opt — should be removed"
