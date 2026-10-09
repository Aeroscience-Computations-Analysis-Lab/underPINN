"""underPINN.training — collocation-point management and training algorithms.

Three distinct concerns live here:

* **Adaptive collocation** (``resample.py``) — RAR-D resampling that moves
  collocation points toward high-residual regions during training.
* **Second-order training** (``natural_gradient.py``) — Levenberg-Marquardt-damped
  Gauss-Newton / natural-gradient optimization, an alternative to the Adam-based
  solvers for small, ill-conditioned problems.
* **Data parallelism** (``parallel.py``) — splitting each mini-batch across
  devices with replicated parameters; enabled per run via
  ``TrainingConfig(data_parallel=True)``.
"""

from underPINN.training.natural_gradient import (
    gauss_newton_step,
    train_gauss_newton,
)
from underPINN.training.parallel import (
    DataParallel,
    data_parallel_from_cfg,
    make_data_parallel,
    pad_to_multiple,
)
from underPINN.training.resample import rar_d_resample, rar_d_resample_split

__all__ = [
    "DataParallel",
    "data_parallel_from_cfg",
    "gauss_newton_step",
    "make_data_parallel",
    "pad_to_multiple",
    "rar_d_resample",
    "rar_d_resample_split",
    "train_gauss_newton",
]
