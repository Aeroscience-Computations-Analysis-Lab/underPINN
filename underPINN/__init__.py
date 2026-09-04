# underPINN/__init__.py

# ── TPU: force full-precision matmuls ─────────────────────────────────────────
# JAX on TPU runs float32 matmuls at reduced (bfloat16-input) precision on the
# MXU by default.  PINNs differentiate the network twice (PDE residuals via
# Hessians), so bf16 matmul noise corrupts the residual and stalls training.
# "float32" restores true float32 matmuls (slower, but required for accuracy).
# No effect on CPU/GPU.  Override by setting JAX_DEFAULT_MATMUL_PRECISION.
#
# Use the canonical name "float32", not the "highest" alias: jax.config's
# validator accepts only ('bfloat16', 'tensorfloat32', 'float32') on the JAX
# versions this project supports (>=0.4.26) and raises ValueError on the
# aliases. Because the block below swallows exceptions, passing "highest" here
# fails *silently* -- the precision is never raised and PDE residuals quietly
# lose accuracy on TPU, which is the exact failure this code exists to prevent.
import os as _os

from ._version import __version__, version_tag  # calver: "2605", "v2605"

try:
    import jax as _jax
    if (_jax.default_backend() == "tpu"
            and "JAX_DEFAULT_MATMUL_PRECISION" not in _os.environ):
        _jax.config.update("jax_default_matmul_precision", "float32")
        print("[underPINN] TPU detected — matmul precision set to float32 "
              "(full precision; needed for accurate PDE residuals).")
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