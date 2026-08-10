"""Calendar versioning for underPINN — YYMM format.

The version is derived from the **current date at import time**, so the
installed distribution is automatically tagged with the month it was built.

  May  2026  →  __version__ = "2605"   tag = "v2605"
  Jan  2027  →  __version__ = "2701"   tag = "v2701"
  Dec  2027  →  __version__ = "2712"   tag = "v2712"

The distribution wheel / dist-info folder will therefore be named::

  underPINN-2605-py3-none-any.whl
  underPINN-2605.dist-info/

Usage
-----
From Python::

    import underPINN
    print(underPINN.__version__)       # "2605"
    print(underPINN.version_tag)       # "v2605"

From the CLI::

    python -c "import underPINN; print(underPINN.version_tag)"
    python -m underPINN --version

pip::

    pip show underPINN                  # Version: 2605
"""
from __future__ import annotations

from datetime import datetime as _dt


def _calver() -> str:
    """Return YYMM string for the current date (e.g. '2605' for May 2026)."""
    now = _dt.now()
    return f"{now.year % 100:02d}{now.month:02d}"


#: Numeric version string used by pip / setuptools: ``"2605"``
__version__: str = _calver()

#: Human-readable label with ``v`` prefix: ``"v2605"``
version_tag: str = f"v{__version__}"
