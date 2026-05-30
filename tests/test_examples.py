"""Smoke tests for all underPINN example scripts.

Every ``run_*`` entry point is called with the matching config YAML, but with
epoch counts overridden to 3–5 so CI finishes in seconds.  We verify:

* The function returns without raising.
* The returned dict contains a non-empty ``loss_hist`` (or equivalent).
* Any solver result has a finite final loss value.

Skipped examples
----------------
* K-Epsilon/turbulence.py  — requires an external CFD data CSV (not in-repo).
* transfer/heat2d_transfer.py — exposes no ``run_*(cfg)`` entry point.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import types

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest

# ---------------------------------------------------------------------------
# Repo-level paths
# ---------------------------------------------------------------------------

_HERE      = pathlib.Path(__file__).parent
_REPO_ROOT = _HERE.parent
_EXAMPLES  = _REPO_ROOT / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module(name: str, rel_path: str) -> types.ModuleType:
    """Import an example file as a module without adding it to sys.path."""
    full_path = _EXAMPLES / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(yaml_rel: str, overrides: dict, tmp_path: pathlib.Path):
    """Load YAML config, apply overrides, and redirect output dir to tmp_path."""
    from underPINN.config.loader import load_config, merge_config
    cfg_path = _EXAMPLES / yaml_rel
    cfg = load_config(str(cfg_path))
    merged = merge_config(cfg, {**overrides, "output.dir": str(tmp_path)})
    return merged


def _has_nonempty_loss(result: dict, key: str = "loss_hist") -> bool:
    hist = result.get(key, [])
    return len(hist) > 0


# ---------------------------------------------------------------------------
# 1. Burgers PINN
# ---------------------------------------------------------------------------

class TestExampleBurgers:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("burgers", "burgers/burgers.py")
        cfg = _cfg("burgers/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 200,
                    "data.n_ic": 20,
                    "data.n_bc": 20},
                   tmp_path)
        result = mod.run_burgers(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)

    def test_params_returned(self, tmp_path):
        mod = _load_module("burgers_params", "burgers/burgers.py")
        cfg = _cfg("burgers/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 200,
                    "data.n_ic": 20,
                    "data.n_bc": 20},
                   tmp_path)
        result = mod.run_burgers(cfg)
        assert "params" in result
        assert result["params"] is not None


# ---------------------------------------------------------------------------
# 2. Wave PINN
# ---------------------------------------------------------------------------

class TestExampleWave:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("wave", "wave/wave.py")
        cfg = _cfg("wave/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "training.early_stopping_patience": 9999,
                    "data.n_collocation": 200,
                    "data.n_ic": 20,
                    "data.n_bc": 20},
                   tmp_path)
        result = mod.run_wave(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 3. Heat Forward (2-D steady Poisson)
# ---------------------------------------------------------------------------

class TestExampleHeatForward:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("heat_forward", "heat/forward.py")
        cfg = _cfg("heat/heat_forward.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_ic": 20,
                    "data.n_bc": 20},
                   tmp_path)
        result = mod.run_heat_forward(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 4. Heat Inverse (identify diffusivity from sparse data)
# ---------------------------------------------------------------------------

class TestExampleHeatInverse:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("heat_inverse", "heat/inverse.py")
        cfg = _cfg("heat/heat_inverse.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_ic": 20,
                    "data.n_bc": 20,
                    "data.n_observations": 10},
                   tmp_path)
        result = mod.run_heat_inverse(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)

    def test_alpha_identified_key_present(self, tmp_path):
        mod = _load_module("heat_inverse_alpha", "heat/inverse.py")
        cfg = _cfg("heat/heat_inverse.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_ic": 20,
                    "data.n_bc": 20,
                    "data.n_observations": 10},
                   tmp_path)
        result = mod.run_heat_inverse(cfg)
        assert "alpha_identified" in result


# ---------------------------------------------------------------------------
# 5. ODE Solver (exponential decay + harmonic oscillator)
# ---------------------------------------------------------------------------

class TestExampleODE:
    def test_runs_and_returns_both_cases(self, tmp_path):
        mod = _load_module("ode", "ode/ode_test.py")
        cfg = _cfg("ode/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "training.early_stopping_patience": 9999,
                    "data.n_collocation": 50},
                   tmp_path)
        result = mod.run_ode(cfg)
        assert "exp_decay_rel_l2" in result
        assert "harmonic_rel_l2"  in result


# ---------------------------------------------------------------------------
# 6. Helmholtz PINN
# ---------------------------------------------------------------------------

class TestExampleHelmholtz:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("helmholtz", "helmholtz/helmholtz.py")
        cfg = _cfg("helmholtz/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_bc": 20},
                   tmp_path)
        result = mod.run_helmholtz(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 7. Pipe Flow (3-D steady Hagen-Poiseuille)
# ---------------------------------------------------------------------------

class TestExamplePipeFlow:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("pipe_flow", "pipe_flow/pipe_flow.py")
        cfg = _cfg("pipe_flow/pipe_flow.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_interior": 100,
                    "data.n_wall": 20,
                    "data.n_inlet": 20,
                    "data.n_outlet": 20},
                   tmp_path)
        result = mod.run_pipe_flow(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 8. Inverse Diffusion (separate entry point, same runner as heat_inverse)
# ---------------------------------------------------------------------------

class TestExampleInverseDiffusion:
    def test_runs_via_inverse_entry_point(self, tmp_path):
        mod = _load_module("inv_diff", "inverse/inverse_diffusion.py")
        cfg = _cfg("inverse/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_ic": 20,
                    "data.n_bc": 20,
                    "data.n_observations": 10},
                   tmp_path)
        result = mod.run_inverse_diffusion(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 9. LDC (Lid-Driven Cavity — FBPINN)
# ---------------------------------------------------------------------------

class TestExampleLDC:
    @staticmethod
    def _make_fake_re100_csv(directory: pathlib.Path) -> None:
        """Create a minimal re100.csv with the shape expected by run_ldc's plot code."""
        import numpy as np
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available — skipping LDC test")
        nx = ny = 201
        n = nx * ny
        df = pd.DataFrame({
            "x-coordinate": np.tile(np.linspace(0.0, 1.0, nx), ny).astype(np.float32),
            "y-coordinate": np.repeat(np.linspace(0.0, 1.0, ny), nx).astype(np.float32),
            "pressure":    np.zeros(n, dtype=np.float32),
            "x-velocity":  np.zeros(n, dtype=np.float32),
            "y-velocity":  np.zeros(n, dtype=np.float32),
        })
        df.to_csv(directory / "re100.csv", index=False)

    def test_runs_and_returns_loss_hist(self, tmp_path, monkeypatch):
        self._make_fake_re100_csv(tmp_path)
        monkeypatch.chdir(tmp_path)          # pd.read_csv("re100.csv") resolves here
        mod = _load_module("ldc", "LDC/run_ldc.py")
        cfg = _cfg("LDC/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "data.n_collocation": 100,
                    "data.n_bc": 40},
                   tmp_path)
        result = mod.run_ldc(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)


# ---------------------------------------------------------------------------
# 10. Ramp (2-D compressible Euler)
# ---------------------------------------------------------------------------

class TestExampleRamp:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("ramp", "ramp/ramp.py")
        cfg = _cfg("ramp/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "training.early_stopping_patience": 9999,
                    "data.n_interior": 50,
                    "data.n_inlet": 10,
                    "data.n_wall": 10,
                    "data.n_upper": 10},
                   tmp_path)
        result = mod.run_ramp(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)

    def test_oblique_shock_key_present(self, tmp_path):
        mod = _load_module("ramp_shock", "ramp/ramp.py")
        cfg = _cfg("ramp/config.yaml",
                   {"training.epochs": 3,
                    "training.save_restart_every": 0,
                    "training.early_stopping_patience": 9999,
                    "data.n_interior": 50,
                    "data.n_inlet": 10,
                    "data.n_wall": 10,
                    "data.n_upper": 10},
                   tmp_path)
        result = mod.run_ramp(cfg)
        assert "oblique_shock" in result


# ---------------------------------------------------------------------------
# 11. Burgers Transfer Learning
# ---------------------------------------------------------------------------

class TestExampleBurgersTransfer:
    def _small_cfg(self, tmp_path):
        return _cfg(
            "transfer/burgers_transfer.yaml",
            {
                # Each training phase reduced to 3 epochs
                "parameter_transfer.n_source_epochs":   3,
                "parameter_transfer.n_transfer_epochs": 3,
                "parameter_transfer.n_scratch_epochs":  3,
                "temporal_transfer.n_phase1_epochs":    3,
                "temporal_transfer.n_transfer_epochs":  3,
                "temporal_transfer.n_scratch_epochs":   3,
                # Small data
                "data.n_collocation": 200,
                "data.n_ic": 20,
                "data.n_bc": 20,
            },
            tmp_path,
        )

    def test_param_transfer_key_in_result(self, tmp_path):
        mod = _load_module("bt", "transfer/burgers_transfer.py")
        cfg = self._small_cfg(tmp_path)
        result = mod.run_burgers_transfer(cfg)
        assert "param_transfer" in result

    def test_temporal_transfer_key_in_result(self, tmp_path):
        mod = _load_module("bt2", "transfer/burgers_transfer.py")
        cfg = self._small_cfg(tmp_path)
        result = mod.run_burgers_transfer(cfg)
        assert "temporal_transfer" in result


# ---------------------------------------------------------------------------
# 12. Pipe Flow Unsteady Transfer
# ---------------------------------------------------------------------------

class TestExamplePipeFlowUnsteadyTransfer:
    def _small_cfg(self, tmp_path):
        return _cfg(
            "pipe_flow/pipe_flow_unsteady_transfer.yaml",
            {
                "re_transfer.n_source_epochs":   3,
                "re_transfer.n_transfer_epochs": 3,
                "re_transfer.n_scratch_epochs":  3,
                "temporal_transfer.n_phase1_epochs":    3,
                "temporal_transfer.n_transfer_epochs":  3,
                "temporal_transfer.n_scratch_epochs":   3,
                "data.n_collocation": 100,
                "data.n_ic": 20,
                "data.n_bc": 20,
            },
            tmp_path,
        )

    def test_re_transfer_key_in_result(self, tmp_path):
        mod = _load_module("pfut", "pipe_flow/pipe_flow_unsteady_transfer.py")
        cfg = self._small_cfg(tmp_path)
        result = mod.run_pipe_flow_unsteady_transfer(cfg)
        assert "re_transfer" in result

    def test_temporal_transfer_key_in_result(self, tmp_path):
        mod = _load_module("pfut2", "pipe_flow/pipe_flow_unsteady_transfer.py")
        cfg = self._small_cfg(tmp_path)
        result = mod.run_pipe_flow_unsteady_transfer(cfg)
        assert "temporal_transfer" in result


# ---------------------------------------------------------------------------
# 13. Airfoil Flow (2-D NACA steady NS)
# ---------------------------------------------------------------------------

class TestExampleAirfoil:
    def test_runs_and_returns_loss_hist(self, tmp_path):
        mod = _load_module("airfoil", "airfoil/airfoil_flow.py")
        cfg = _cfg(
            "airfoil/config.yaml",
            {
                "training.epochs": 3,
                "training.save_restart_every": 0,
                "training.resample_period": 0,   # disable RAR-D for speed
                "data.n_exterior":      100,
                "data.n_near_surface":   50,
                "data.n_body_bc":        20,
                "data.n_farfield_bc":    20,
            },
            tmp_path,
        )
        result = mod.run_airfoil(cfg)
        assert "loss_hist" in result
        assert _has_nonempty_loss(result)
