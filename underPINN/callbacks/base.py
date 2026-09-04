from abc import ABC


class Callback(ABC):
    """Base class for training callbacks.

    Override whichever hooks you need; default implementations are no-ops.
    """

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Called after every training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        logs : dict
            Keys: 'loss', 'pde', 'ic', and any extra loss components.
        """

    def on_train_end(self, logs: dict) -> None:
        """Called once when training finishes.

        Parameters
        ----------
        logs : dict
            Final epoch logs (same structure as on_epoch_end).
        """
