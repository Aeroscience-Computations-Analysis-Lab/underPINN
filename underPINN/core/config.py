from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TrainingConfig:
    """Unified configuration object passed to any BaseSolver.

    Replaces scattered keyword arguments in solver.train() calls.

    Performance knobs
    -----------------
    n_scan_steps : int
        Number of gradient steps to fuse into a single ``jax.lax.scan`` XLA
        kernel.  The default (``1``) keeps the original Python-loop behaviour.
        Larger values (e.g. 100–500) dramatically reduce Python dispatch
        overhead on GPU at the cost of coarser callback granularity — callbacks
        fire every *n_scan_steps* epochs rather than every epoch.
    resample_period : int
        Interval (in *outer* steps, i.e. every ``n_scan_steps`` epochs) at
        which RAR-D adaptive collocation resampling is applied.
        ``0`` disables resampling entirely (default).
    resample_candidates : int
        Candidate pool size used by RAR-D.  ``0`` → ``5 × batch_r``.
    resample_k : float
        Exponent in the RAR-D weight distribution  p ∝ |residual|^k.
        ``1.0`` is the standard choice; larger values focus more aggressively
        on high-residual regions.
    candidate_sampler : callable, optional
        Callable that draws fresh candidate points from the problem domain.
        The expected signature depends on which solver uses this config:

        * Solvers that use **packed** collocation (e.g. :class:`SteadySolver`):
          ``fn(n: int, key) → xy``  where ``xy`` is an ``(n, D)`` array.
        * Solvers that use **split** collocation (e.g. :class:`FBPINNSolver`):
          ``fn(n: int, key) → (x, t)``  returning a pair of arrays.

        When *None* (default) bootstrap resampling from the current
        collocation set is used — adequate for most problems but inferior to
        a proper domain sampler.
    data_parallel : bool
        Split every mini-batch across all visible devices along its point
        axis, replicating parameters and optimizer state. The gradient is the
        same as single-device full-batch training (it is data parallelism,
        not a different algorithm). Any batch size works on any device
        count: a batch that does not divide is padded by repeating at most
        D-1 of its own points (reported at start-up); divisible batches are
        not padded and stay exactly equivalent. A no-op on a single device.
        See :mod:`underPINN.training.parallel`.
    data_parallel_devices : int, optional
        How many devices ``data_parallel`` uses: the first *N* visible ones.
        ``None`` (default) uses all of them. Any count works; batches that
        do not divide by it are padded (see ``data_parallel``). Asking for more
        devices than are visible raises; choose *which* GPUs with
        ``CUDA_VISIBLE_DEVICES``. Ignored, with a warning, when
        ``data_parallel`` is false.
    model_parallel : bool
        Split an FBPINN's subdomain networks across devices (model
        parallelism); see :mod:`underPINN.training.model_parallel`. The
        model must be built with ``mp.wrap(FBPINN(...))``. Same loss,
        gradients and initialisation as single-device training. Cannot
        yet be combined with ``data_parallel``.
    model_parallel_devices : int, optional
        How many devices ``model_parallel`` uses (first *N* visible;
        ``None`` = all). Any count works: subdomains that do not divide
        are padded with exact-zero dummies.

    Examples
    --------
    >>> import optax
    >>> from underPINN.callbacks.logging import ConsoleLogger
    >>> from underPINN.callbacks.early_stopping import EarlyStopping
    >>> config = TrainingConfig(
    ...     epochs=5000,
    ...     lr=1e-3,
    ...     lr_schedule=optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    ...     log_every=500,
    ...     n_scan_steps=100,        # fuse 100 steps into one XLA kernel
    ...     resample_period=5,       # RAR-D every 5 outer steps (= 500 epochs)
    ...     callbacks=[ConsoleLogger(), EarlyStopping(patience=300)],
    ... )
    """

    # ------------------------------------------------------------------ #
    # Core hyper-parameters                                                #
    # ------------------------------------------------------------------ #
    epochs: int = 1000
    lr: float = 1e-3
    lr_schedule: Any = None        # any optax schedule; overrides lr when set
    batch_r: int = 4096            # collocation batch
    batch_i: int = 512             # initial-condition batch
    batch_b: int = 512             # boundary-condition batch
    log_every: int = 100           # print every N epochs (used by ConsoleLogger)
    seed: int = 0
    callbacks: List[Any] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Performance knobs                                                    #
    # ------------------------------------------------------------------ #
    n_scan_steps: int = 1          # lax.scan chunk size; 1 = Python loop
    resample_period: int = 0       # outer steps between RAR-D resamplings; 0 = off
    resample_candidates: int = 0   # candidate pool size; 0 → 5 × batch_r
    resample_k: float = 1.0        # |residual|^k weighting exponent
    candidate_sampler: Optional[Any] = None  # fn(n, key) → (x, t) or None
    data_parallel: bool = False    # split batches across all visible devices
                                   # (see underPINN.training.parallel); a batch
                                   # not divisible by the count is padded
    data_parallel_devices: Optional[int] = None  # use only the first N
                                   # visible devices; None = all of them
    model_parallel: bool = False   # split FBPINN subdomain networks across
                                   # devices (underPINN.training.model_parallel)
    model_parallel_devices: Optional[int] = None  # first N visible; None = all

    # ------------------------------------------------------------------ #
    # Restart / resume                                                     #
    # ------------------------------------------------------------------ #
    out_dir: str = ""              # output dir; when non-empty, enables auto-restart
    save_restart_every: int = 500  # save restart snapshot every N epochs; 0 = off

    # ------------------------------------------------------------------ #
    # Validation                                                            #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Validate all fields eagerly so errors surface before JAX traces.

        Raises
        ------
        ValueError
            Descriptive message that names the offending field and its value.
        """
        _pos_int = {
            "epochs":      self.epochs,
            "batch_r":     self.batch_r,
            "batch_i":     self.batch_i,
            "batch_b":     self.batch_b,
            "log_every":   self.log_every,
            "n_scan_steps": self.n_scan_steps,
        }
        for name, val in _pos_int.items():
            if not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"TrainingConfig.{name} must be a positive integer, got {val!r}"
                )

        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise ValueError(
                f"TrainingConfig.lr must be a positive float, got {self.lr!r}"
            )

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(
                f"TrainingConfig.seed must be a non-negative integer, got {self.seed!r}"
            )

        if not isinstance(self.resample_k, (int, float)) or self.resample_k <= 0:
            raise ValueError(
                f"TrainingConfig.resample_k must be a positive float, got {self.resample_k!r}"
            )

        if self.resample_period < 0:
            raise ValueError(
                f"TrainingConfig.resample_period must be >= 0, got {self.resample_period!r}"
            )

        if self.resample_candidates < 0:
            raise ValueError(
                f"TrainingConfig.resample_candidates must be >= 0, got {self.resample_candidates!r}"
            )

        _dpd = self.data_parallel_devices
        # bool is an int in Python; refuse it so `true` is not read as 1 device.
        if _dpd is not None and (isinstance(_dpd, bool)
                                 or not isinstance(_dpd, int) or _dpd < 1):
            raise ValueError(
                f"TrainingConfig.data_parallel_devices must be a positive int "
                f"or None (all visible devices), got {_dpd!r}"
            )

        _mpd = self.model_parallel_devices
        if _mpd is not None and (isinstance(_mpd, bool)
                                 or not isinstance(_mpd, int) or _mpd < 1):
            raise ValueError(
                f"TrainingConfig.model_parallel_devices must be a positive int "
                f"or None (all visible devices), got {_mpd!r}"
            )

        if self.data_parallel and self.model_parallel:
            raise ValueError(
                "TrainingConfig: data_parallel and model_parallel are both on; "
                "combining them in one run is not supported yet -- enable one."
            )

        if not isinstance(self.save_restart_every, int) or self.save_restart_every < 0:
            raise ValueError(
                f"TrainingConfig.save_restart_every must be a non-negative int, "
                f"got {self.save_restart_every!r}"
            )
