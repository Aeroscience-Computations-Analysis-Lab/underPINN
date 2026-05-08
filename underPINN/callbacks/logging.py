import time
from .base import Callback


class ConsoleLogger(Callback):
    """Prints a formatted loss table to stdout every `log_every` epochs.

    Replaces the hardcoded print statements inside each solver.
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self._start = None

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
            print(" | ".join(parts))

    def on_train_end(self, logs: dict) -> None:
        print(f"Training complete — final loss {logs.get('loss', float('nan')):.3e}")
