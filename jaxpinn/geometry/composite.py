import numpy as np
from .base import Geometry


class CompositeGeometry(Geometry):
    def __init__(self, geom_a, geom_b, op="union"):
        self.A = geom_a
        self.B = geom_b
        self.op = op

    def contains(self, x):
        a = self.A.contains(x)
        b = self.B.contains(x)

        if self.op == "union":
            return a | b
        elif self.op == "intersection":
            return a & b
        elif self.op == "difference":
            return a & (~b)
        else:
            raise ValueError("Invalid operation")

    def sample(self, n, seed=None):
        # rejection sampling from bounding box
        rng = np.random.default_rng(seed)
        lo, hi = self.bounding_box()

        samples = []
        while len(samples) < n:
            x = rng.uniform(lo, hi, size=(n, lo.shape[0]))
            mask = self.contains(x)
            samples.append(x[mask])

        return np.concatenate(samples, axis=0)[:n]

    def bounding_box(self):
        lo1, hi1 = self.A.bounding_box()
        lo2, hi2 = self.B.bounding_box()
        return np.minimum(lo1, lo2), np.maximum(hi1, hi2)
