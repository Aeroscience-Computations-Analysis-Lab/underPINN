"""I/O helpers — save model predictions to disk.

Typical use inside a runner::

    from underPINN.utils.io import save_predictions

    save_predictions(
        out_dir,
        coords   = {"x": x_r, "t": t_r},
        outputs  = {"u_pred": u_pred},
        exact    = {"u_exact": u_exact},   # optional
    )
    # → outputs/predictions.npz
"""

from __future__ import annotations

import os
import numpy as np


def save_predictions(
    out_dir: str,
    coords: dict,
    outputs: dict,
    exact: dict | None = None,
    filename: str = "predictions.npz",
) -> str:
    """Save collocation-point coordinates, PINN predictions, and (optionally)
    exact solution values to a compressed NumPy archive.

    Parameters
    ----------
    out_dir  : Directory where the file is written (created if absent).
    coords   : Coordinate arrays, e.g. ``{"x": x_r, "t": t_r}``.
               Keys are used verbatim as array names in the archive.
    outputs  : PINN output arrays, e.g. ``{"u_pred": u_pred}``.
    exact    : Optional ground-truth arrays, e.g. ``{"u_exact": u_exact}``.
               Pass ``None`` or ``{}`` to omit.
    filename : Archive filename (default ``predictions.npz``).

    Returns
    -------
    str
        Absolute path of the saved file.

    Notes
    -----
    All arrays are converted to ``float32`` NumPy arrays before saving to
    keep file sizes small.  JAX / PyTorch tensors are handled transparently.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _to_np(v):
        if hasattr(v, "numpy"):          # jax.Array / torch.Tensor
            v = v.__array__() if hasattr(v, "__array__") else np.array(v)
        return np.asarray(v, dtype=np.float32)

    arrays: dict[str, np.ndarray] = {}
    for k, v in coords.items():
        arrays[k] = _to_np(v)
    for k, v in outputs.items():
        arrays[k] = _to_np(v)
    for k, v in (exact or {}).items():
        arrays[k] = _to_np(v)

    path = os.path.join(out_dir, filename)
    np.savez_compressed(path, **arrays)
    print(f"Predictions saved → {path}")
    return path
