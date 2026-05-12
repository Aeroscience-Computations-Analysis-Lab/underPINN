"""Central runner registry for underPINN CLI.

To add a new problem runner:
  1. Write  underPINN/runner/<name>.py  with a ``run_<name>(cfg)`` function.
  2. Import it below and add it to ``_REGISTRY``.
"""

from underPINN.runner.burgers      import run_burgers
from underPINN.runner.wave         import run_wave
from underPINN.runner.pipe_flow    import run_pipe_flow
from underPINN.runner.helmholtz    import run_helmholtz
from underPINN.runner.heat_forward import run_heat_forward
from underPINN.runner.ode          import run_ode

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps the string value of ``problem:`` in a config YAML to a runner callable.

_REGISTRY: dict = {
    "burgers":      run_burgers,
    "wave":         run_wave,
    "pipe_flow":    run_pipe_flow,
    "helmholtz":    run_helmholtz,
    "heat_forward": run_heat_forward,
    "ode":          run_ode,
}


def get_runner(problem: str):
    """Return the runner function for *problem*, raising a helpful error if unknown."""
    if problem not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown problem '{problem}'. "
            f"Available runners: {known}\n"
            f"To add a new runner, see underPINN/runner/dispatch.py."
        )
    return _REGISTRY[problem]


def list_problems() -> list:
    """Return sorted list of registered problem names."""
    return sorted(_REGISTRY)
