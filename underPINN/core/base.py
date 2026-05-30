from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional


class BasePDE(ABC):
    """Contract for every physics operator (ODE or PDE).

    Subclasses must implement `residual`. Optionally override `u` and `exact`.
    """

    @abstractmethod
    def residual(self, params, *args) -> Any:
        """Compute the PDE/ODE residual at collocation points."""
        ...


class BaseLoss(ABC):
    """Contract for loss functions used in PINN training."""

    @abstractmethod
    def __call__(self, params, *args, **kwargs) -> Tuple[float, tuple]:
        """Return (total_loss, tuple_of_components)."""
        ...


class BaseSolver(ABC):
    """Contract for training-loop orchestrators.

    In addition to the abstract interface, every concrete solver inherits
    two ready-to-use checkpoint helpers:

    * :meth:`save_checkpoint` — serialise ``self.params`` to disk.
    * :meth:`restore_checkpoint` — load params from disk and reset the
      optimiser state (calls ``load_params`` if available, otherwise sets
      ``self.params`` directly).

    These are implemented here in terms of
    :mod:`underPINN.utils.checkpoint` so subclasses get them for free.
    """

    @abstractmethod
    def init(self, key) -> None:
        """Initialise network parameters and optimizer state."""
        ...

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Run the training loop."""
        ...

    def evaluate(self, *args, **kwargs) -> Any:
        """Optional post-training evaluation hook."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Checkpoint helpers (concrete — available on every solver)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        out_dir: str,
        stem: str = "params",
        metadata: Optional[dict] = None,
    ) -> tuple:
        """Save ``self.params`` to *out_dir* as a Flax msgpack checkpoint.

        Parameters
        ----------
        out_dir  : Output directory (created if absent).
        stem     : Filename stem (default ``"params"``).
        metadata : Optional JSON-serialisable dict written alongside the
                   checkpoint.  Include at minimum
                   ``{"problem": ..., "network": {"type": ..., "layers": ...}}``
                   so :class:`~underPINN.utils.checkpoint.ModelPredictor`
                   can auto-rebuild the model later.

        Returns
        -------
        (params_path, meta_path)  — absolute paths of the files written.
        """
        from underPINN.utils.checkpoint import save_checkpoint
        return save_checkpoint(self.params, out_dir, stem=stem, metadata=metadata)

    def restore_checkpoint(self, path: str) -> None:
        """Load params from *path* and reset the optimiser.

        *path* may be a directory (looks for ``params.msgpack`` inside) or
        the direct path to a ``.msgpack`` file.

        The solver must have been initialised (``init`` called) at least once
        before calling this so that the parameter template is available.
        """
        from underPINN.utils.checkpoint import load_checkpoint
        new_params = load_checkpoint(self.params, path)   # self.params as template
        # Delegate to load_params if available (resets opt state + histories)
        if hasattr(self, "load_params"):
            self.load_params(new_params)
        else:
            self.params = new_params
            if hasattr(self, "opt"):
                # FBPINNSolver / ODESolver use self.state; LDCSolver uses self.opt_state
                if hasattr(self, "state"):
                    self.state = self.opt.init(new_params)
                elif hasattr(self, "opt_state"):
                    self.opt_state = self.opt.init(new_params)

    # ------------------------------------------------------------------
    # Internal: wire ModelCheckpoint callbacks to self
    # ------------------------------------------------------------------

    def _attach_checkpoint_callbacks(self, callbacks: list) -> None:
        """Give every ModelCheckpoint callback in *callbacks* a reference to self."""
        try:
            from underPINN.callbacks.checkpoint import ModelCheckpoint
        except ImportError:
            return
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.solver_ref is None:
                cb.solver_ref = self
