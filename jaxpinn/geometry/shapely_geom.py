import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep
from .base import Geometry


class ShapelyGeometry(Geometry):
    def __init__(self, shapely_object):
        self.geom = prep(shapely_object)
        self.bounds = shapely_object.bounds

    def contains(self, x):
        return np.array([
            self.geom.contains(Point(p))
            for p in x
        ])

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        xmin, ymin, xmax, ymax = self.bounds

        pts = []
        while len(pts) < n:
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if self.geom.contains(Point(x, y)):
                pts.append([x, y])
        return np.array(pts)

    def bounding_box(self):
        xmin, ymin, xmax, ymax = self.bounds
        return np.array([xmin, ymin]), np.array([xmax, ymax])
