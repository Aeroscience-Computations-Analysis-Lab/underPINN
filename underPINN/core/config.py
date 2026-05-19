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
        ``fn(n: int, key) → (x, t)`` that draws fresh candidate points from
        the problem domain.  When *None* (default) bootstrap resampling from
        the current collocation set is used — adequate for most problems but
        inferior to a proper domain sampler.

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

        if not isinstance(self.save_restart_every, int) or self.save_restart_every < 0:
            raise ValueError(
                f"TrainingConfig.save_restart_every must be a non-negative int, "
                f"got {self.save_restart_every!r}"
            )
