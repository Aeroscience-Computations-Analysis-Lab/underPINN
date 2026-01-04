import numpy as np
from .base import Geometry


class Interval(Geometry):
    def __init__(self, xmin, xmax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)

    def contains(self, x):
        x = np.asarray(x)
        return (x[:, 0] >= self.xmin) & (x[:, 0] <= self.xmax)

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xmin, self.xmax, size=(n, 1))
        return x

    def bounding_box(self):
        return np.array([self.xmin]), np.array([self.xmax])
