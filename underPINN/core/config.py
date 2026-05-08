from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class TrainingConfig:
    """Unified configuration object passed to any BaseSolver.

    Replaces scattered keyword arguments in solver.train() calls.

    Examples
    --------
    >>> import optax
    >>> from underPINN.callbacks.logging import ConsoleLogger
    >>> from underPINN.callbacks.early_stopping import EarlyStopping
    >>> config = TrainingConfig(
    ...     epochs=3000,
    ...     lr=1e-3,
    ...     lr_schedule=optax.cosine_decay_schedule(1e-3, 3000, alpha=1e-2),
    ...     log_every=500,
    ...     callbacks=[ConsoleLogger(), EarlyStopping(patience=300)],
    ... )
    """

    epochs: int = 1000
    lr: float = 1e-3
    lr_schedule: Any = None        # any optax schedule; overrides lr when set
    batch_r: int = 4096            # collocation batch
    batch_i: int = 512             # initial-condition batch
    batch_b: int = 512             # boundary-condition batch
    log_every: int = 100           # print every N epochs (used by ConsoleLogger)
    seed: int = 0
    callbacks: List[Any] = field(default_factory=list)
