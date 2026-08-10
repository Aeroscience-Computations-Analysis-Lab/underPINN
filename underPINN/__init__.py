# underPINN/__init__.py

# ── TPU: force full-precision matmuls ─────────────────────────────────────────
# JAX on TPU runs float32 matmuls at reduced (bfloat16-input) precision on the
# MXU by default.  PINNs differentiate the network twice (PDE residuals via
# Hessians), so bf16 matmul noise corrupts the residual and stalls training.
# "highest" restores true float32 matmuls (slower, but required for accuracy).
# No effect on CPU/GPU.  Override by setting JAX_DEFAULT_MATMUL_PRECISION.
import os as _os

from ._version import __version__, version_tag  # calver: "2605", "v2605"

try:
    import jax as _jax
    if (_jax.default_backend() == "tpu"
            and "JAX_DEFAULT_MATMUL_PRECISION" not in _os.environ):
        _jax.config.update("jax_default_matmul_precision", "highest")
        print("[underPINN] TPU detected — matmul precision set to 'highest' "
              "(full float32; needed for accurate PDE residuals).")
except Exception:  # noqa: BLE001, S110 -- pragma: no cover (best-effort; never block import)
    pass

from .benchmark_utils import *
from .geometry import *
from .nn import *
from .pde import *
from .solver import *

__all__ = [
    "geometry",
    "nn",
    "pde",
    "solver",
    "benchmark_utils",
    "__version__",
    "version_tag",
]   