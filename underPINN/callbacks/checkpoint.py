"""ModelCheckpoint callback — save model weights whenever a monitored metric improves.

Usage in a runner::

    from underPINN.callbacks.checkpoint import ModelCheckpoint

    config = TrainingConfig(
        epochs=5000,
        callbacks=[
            ConsoleLogger(log_every=500),
            EarlyStopping(patience=400),
            ModelCheckpoint(out_dir="outputs/burgers/",
                            monitor="loss", save_best_only=True),
        ],
    )
    solver.train(*data, config=config)

The callback writes ``params.msgpack`` (and optionally ``params_meta.json``)
to *out_dir* every time the monitored loss improves.  Set
``save_best_only=False`` to checkpoint every ``period`` epochs instead.

Parameters (all keyword)
------------------------
out_dir        : Directory for checkpoint files.
solver_ref     : The solver object to read ``.params`` from.  **Must be set**
                 before the callback fires.  Solvers that understand
                 :class:`TrainingConfig` set this automatically via
                 :meth:`~underPINN.core.base.BaseSolver._attach_checkpoint_cb`.
monitor        : Log key to watch (default ``"loss"``).
mode           : ``"min"`` (default) or ``"max"``.
save_best_only : Save only when *monitor* improves (default ``True``).
                 When ``False``, saves every *period* epochs.
period         : Epoch interval when ``save_best_only=False`` (default 100).
stem           : Filename stem (default ``"params"``).
metadata       : Optional dict written to the JSON sidecar.
verbose        : Print a message when a checkpoint is written (default ``True``).
"""

from __future__ import annotations

import math
from typing import Optional

from underPINN.callbacks.base import Callback
from underPINN.utils.checkpoint import save_checkpoint


class ModelCheckpoint(Callback):
    """Save model parameters when a monitored metric improves (or periodically)."""

    def __init__(
        self,
        out_dir: str,
        *,
        solver_ref=None,
        monitor: str = "loss",
        mode: str = "min",
        save_best_only: bool = True,
        period: int = 100,
        stem: str = "params",
        metadata: Optional[dict] = None,
        verbose: bool = True,
    ):
        self.out_dir       = out_dir
        self.solver_ref    = solver_ref   # set by solver or user
        self.monitor       = monitor
        self.save_best_only = save_best_only
        self.period        = period
        self.stem          = stem
        self.metadata      = metadata
        self.verbose       = verbose

        if mode == "min":
            self._is_better = lambda new, best: new < best
            self._best      = math.inf
        elif mode == "max":
            self._is_better = lambda new, best: new > best
            self._best      = -math.inf
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    # ------------------------------------------------------------------
    # Callback interface
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if self.solver_ref is None:
            return   # no params source registered yet

        if self.save_best_only:
            value = logs.get(self.monitor)
            if value is None:
                return
            if self._is_better(float(value), self._best):
                self._best = float(value)
                self._write(epoch, float(value))
        else:
            if epoch % self.period == 0:
                self._write(epoch)

    def on_train_end(self, logs: dict) -> None:
        """Always write a final checkpoint when training ends."""
        if self.solver_ref is not None:
            self._write(epoch=None, value=None, suffix="_final")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, epoch, value=None, suffix: str = "") -> None:
        params = self.solver_ref.params
        meta   = dict(self.metadata) if self.metadata else {}
        if epoch is not None:
            meta["saved_at_epoch"] = epoch
        if value is not None:
            meta[f"best_{self.monitor}"] = value

        save_checkpoint(
            params,
            self.out_dir,
            stem=self.stem + suffix,
            metadata=meta if meta else None,
        )
        if self.verbose and epoch is not None:
            tag = f"{self.monitor}={value:.4e}" if value is not None else ""
            print(f"  [ModelCheckpoint] epoch {epoch:5d}  {tag}  → {self.out_dir}/{self.stem}{suffix}.msgpack")
