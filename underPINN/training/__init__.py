"""underPINN.training — collocation-point management and training algorithms.

Two distinct concerns live here:

* **Adaptive collocation** (``resample.py``) — RAR-D resampling that moves
  collocation points toward high-residual regions during training.
* **Second-order training** (``natural_gradient.py``) — Levenberg-Marquardt-damped
  Gauss-Newton / natural-gradient optimization, an alternative to the Adam-based
  solvers for small, ill-conditioned problems.
"""

from underPINN.training.natural_gradient import (
    gauss_newton_step,
    train_gauss_newton,
)
from underPINN.training.resample import rar_d_resample, rar_d_resample_split

__all__ = [
    "gauss_newton_step",
    "rar_d_resample",
    "rar_d_resample_split",
    "train_gauss_newton",
]
