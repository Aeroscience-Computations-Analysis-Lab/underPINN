"""Central runner registry for underPINN CLI.

Runner logic lives in the example scripts themselves — each example folder is
self-contained (script + YAML).  This file maps problem names to those scripts
via dynamic import so no runner code is duplicated inside underPINN.

To add a new problem:
  1. Create  examples/<name>/  with your script and a YAML config.
  2. Add one line to ``_REGISTRY`` below — that's it.
"""

from __future__ import annotations

import importlib.util
import pathlib

# ── Path registry ─────────────────────────────────────────────────────────────
# Maps  problem_name  →  (relative_path_to_script, function_name)
# Paths are relative to the repository root (parent of underPINN/).

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent   # → repo root

_REGISTRY: dict[str, tuple[str, str]] = {
    # Core physics benchmarks
    "burgers":      ("examples/burgers/burgers.py",         "run_burgers"),
    "wave":         ("examples/wave/wave.py",                "run_wave"),
    "helmholtz":    ("examples/helmholtz/helmholtz.py",      "run_helmholtz"),
    "heat_forward": ("examples/heat/forward.py",             "run_heat_forward"),
    "ode":          ("examples/ode/ode_test.py",             "run_ode"),

    # Fluid dynamics
    "ldc":          ("examples/LDC/run_ldc.py",             "run_ldc"),
    "airfoil":      ("examples/airfoil/airfoil_flow.py",    "run_airfoil"),
    "pipe_flow":    ("examples/pipe_flow/pipe_flow.py",     "run_pipe_flow"),

    # Inverse problems
    "heat_inverse":      ("examples/heat/inverse.py",        "run_heat_inverse"),
    "inverse_diffusion": ("examples/heat/inverse.py",        "run_heat_inverse"),

    # Transfer learning
    "burgers_transfer": (
        "examples/transfer/burgers_transfer.py",
        "run_burgers_transfer",
    ),
    "pipe_flow_unsteady_transfer": (
        "examples/pipe_flow/pipe_flow_unsteady_transfer.py",
        "run_pipe_flow_unsteady_transfer",
    ),
}


# ── Dynamic loader ────────────────────────────────────────────────────────────

def _load_runner(rel_path: str, fn_name: str):
    """Dynamically import *fn_name* from the script at *rel_path*."""
    abs_path = _REPO_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Runner script not found: {abs_path}\n"
            f"Expected at '{rel_path}' relative to the repo root."
        )
    spec = importlib.util.spec_from_file_location("_runner_mod", abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def get_runner(problem: str):
    """Return the runner callable for *problem*, raising a helpful error if unknown."""
    if problem not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown problem '{problem}'.\n"
            f"Available runners: {known}\n"
            f"To add a new runner, add one entry to underPINN/runner/dispatch.py."
        )
    rel_path, fn_name = _REGISTRY[problem]
    return _load_runner(rel_path, fn_name)


def list_problems() -> list[str]:
    """Return sorted list of registered problem names."""
    return sorted(_REGISTRY)
