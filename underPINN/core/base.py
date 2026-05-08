from abc import ABC, abstractmethod
from typing import Tuple, Any


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
    """Contract for training-loop orchestrators."""

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
