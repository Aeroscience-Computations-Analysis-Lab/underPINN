"""2-D Inverse Diffusion PINN — entry point.

Identifies thermal diffusivity α from sparse, noisy observations.
Runner logic lives in examples/PINNs/heat/inverse.py (shared with heat_inverse).

Run directly or via the CLI:

    python examples/PINNs/inverse/inverse_diffusion.py              # uses config.yaml
    python examples/PINNs/inverse/inverse_diffusion.py myconfig.yaml
    python -m underPINN run examples/PINNs/inverse/config.yaml
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Dynamically load run_heat_inverse from examples/PINNs/heat/inverse.py.
# Resolved as a sibling of this example's own directory rather than by walking
# up to the repo root, so it survives the whole examples/PINNs/ tree moving.
_PINNS_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HEAT_INV   = _PINNS_ROOT / "heat" / "inverse.py"

_spec = importlib.util.spec_from_file_location("_heat_inverse", _HEAT_INV)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_inverse_diffusion = _mod.run_heat_inverse


if __name__ == "__main__":
    import sys
    _HERE    = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_inverse_diffusion(load_config(cfg_path))
