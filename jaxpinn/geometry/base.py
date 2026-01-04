from abc import ABC, abstractmethod
import numpy as np


class Geometry(ABC):
    """
    Abstract geometry class.
    """

    @abstractmethod
    def contains(self, x: np.ndarray) -> np.ndarray:
        """
        Check if points are inside geometry.
        x : (N, D)
        returns : (N,) boolean mask
        """
        pass

    @abstractmethod
    def sample(self, n: int, seed=None) -> np.ndarray:
        """
        Sample n points inside geometry.
        returns : (n, D)
        """
        pass

    @abstractmethod
    def bounding_box(self):
        """
        Return (min, max) bounds.
        """
        pass
