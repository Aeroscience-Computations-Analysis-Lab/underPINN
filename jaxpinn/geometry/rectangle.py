import numpy as np
from .base import Geometry


class Rectangle(Geometry):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def contains(self, x):
        return (
            (x[:, 0] >= self.xmin)
            & (x[:, 0] <= self.xmax)
            & (x[:, 1] >= self.ymin)
            & (x[:, 1] <= self.ymax)
        )

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xmin, self.xmax, size=n)
        y = rng.uniform(self.ymin, self.ymax, size=n)
        return np.stack([x, y], axis=1)

    def bounding_box(self):
        return (
            np.array([self.xmin, self.ymin]),
            np.array([self.xmax, self.ymax]),
        )
