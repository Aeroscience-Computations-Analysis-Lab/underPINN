"""ConsoleLogger — logs training progress via the standard ``logging`` module.

Using ``logging`` instead of bare ``print`` lets callers silence all underPINN
output with a single line::

    import logging
    logging.getLogger("underPINN").setLevel(logging.WARNING)

or configure it to write to a file, add timestamps, etc.
"""

from __future__ import annotations

import logging
import time

from .base import Callback

_log = logging.getLogger("underPINN.training")

# Attach a default StreamHandler only when the root logger has no handlers
# (i.e. the user has not configured logging themselves).  This avoids double-
# printing in applications that call ``logging.basicConfig()``.
if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


class ConsoleLogger(Callback):
    """Logs a formatted loss table every *log_every* epochs.

    Output goes through ``logging.getLogger("underPINN.training")`` at the
    ``INFO`` level, so it can be silenced or redirected without touching this
    class::

        import logging
        logging.getLogger("underPINN").setLevel(logging.WARNING)  # silence
        logging.getLogger("underPINN").setLevel(logging.DEBUG)    # verbose

    Parameters
    ----------
    log_every:
        Emit a log line every this many epochs (default: 100).
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self._start: float | None = None

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if self._start is None:
            self._start = time.time()

        if epoch % self.log_every == 0:
            elapsed = time.time() - self._start
            parts = [f"Epoch {epoch:5d}"]
            for key in ("loss", "pde", "ic", "bc", "ic_dot"):
                if key in logs:
                    parts.append(f"{key.upper()} {logs[key]:.3e}")
            parts.append(f"Time {elapsed:.2f}s")
            _log.info(" | ".join(parts))

    def on_train_end(self, logs: dict) -> None:
        _log.info(
            "Training complete — final loss %.3e",
            logs.get("loss", float("nan")),
        )
