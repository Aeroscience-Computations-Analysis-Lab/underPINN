# underPINN/__init__.py

from ._version import __version__, version_tag   # calver: "2605", "v2605"

from .geometry import *
from .nn import *
from .pde import *
from .solver import *
from .benchmark_utils import *

__all__ = [
    "geometry",
    "nn",
    "pde",
    "solver",
    "benchmark_utils",
    "__version__",
    "version_tag",
]   