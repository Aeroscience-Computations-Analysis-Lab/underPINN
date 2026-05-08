from .base import Callback


class EarlyStopping(Callback):
    """Stops training when total loss stops improving.

    Raises `StopIteration` inside `on_epoch_end`; solvers catch this to
    exit their training loop cleanly.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change in loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 200, min_delta: float = 1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._wait = 0

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        loss = logs.get("loss", float("inf"))
        if loss < self._best - self.min_delta:
            self._best = loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(
                    f"EarlyStopping: no improvement for {self.patience} epochs "
                    f"(best={self._best:.3e}). Stopping at epoch {epoch}."
                )
                raise StopIteration
