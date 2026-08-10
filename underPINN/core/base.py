from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional


class BasePDE(ABC):
    """Contract for every physics operator (ODE or PDE).

    Subclasses must implement `residual`. Optionally override `u` and `exact`.

    Signature convention
    --------------------
    All PDEs use a **single packed coordinate array**::

        residual(self, params, xy: jnp.ndarray) -> jnp.ndarray

    ``xy`` has shape ``(N, D)`` where D is the total number of input
    dimensions (space + time concatenated).  Examples:

    * 1-D + time  (Burgers, Wave, Diffusion)   → D = 2,  xy[:, 0] = x, xy[:, 1] = t
    * 2-D + time  (Heat2D, UnsteadyPipe)       → D = 3,  xy[:, 0:2] = (x,y), xy[:, 2] = t
    * 2-D steady  (NavierStokes, Helmholtz)    → D = 2
    * 3-D steady  (SteadyNS3D)                → D = 3
    * ODE         (ExpDecay, HarmonicOsc)      → D = 1  (time only)

    Return type
    -----------
    Always a ``jnp.ndarray``.  Scalar PDEs return shape ``(N,)``; multi-equation
    PDEs return shape ``(N, K)`` where K is the number of equations.
    Never a tuple.
    """

    @abstractmethod
    def residual(self, params, xy) -> Any:
        """Compute the PDE/ODE residual at collocation points.

        Parameters
        ----------
        params : Flax/pytree parameter tree.
        xy     : ``(N, D)`` packed coordinate array (see class docstring).

        Returns
        -------
        ``jnp.ndarray`` of shape ``(N,)`` or ``(N, K)``.
        """
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
            if hasattr(self, "opt") and hasattr(self, "state"):
                self.state = self.opt.init(new_params)

    # ------------------------------------------------------------------
    # Shared optimizer factory (identical across all solvers)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_opt(lr, lr_schedule):
        """Build an optax optimizer from a learning rate and optional schedule.

        Parameters
        ----------
        lr           : Base learning rate (float).
        lr_schedule  : An optax schedule object, or ``None``.  When provided,
                       Adam scaling is combined with the schedule via
                       ``optax.chain``; otherwise plain ``optax.adam(lr)``
                       is returned.

        Returns
        -------
        An optax ``GradientTransformation``.
        """
        import optax
        if lr_schedule is not None:
            return optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1.0),
            )
        return optax.adam(lr)

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
