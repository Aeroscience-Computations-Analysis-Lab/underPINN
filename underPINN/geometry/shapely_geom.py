'''
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
'''

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from scipy.spatial import KDTree
from .base import Geometry


class ShapelyPolygon(Geometry):
    def __init__(self, vertices):
        self.poly = Polygon(vertices)
        self.prep = prep(self.poly)
        self.bounds = self.poly.bounds

    def contains(self, x):
        return np.array([self.prep.contains(Point(p)) for p in x])

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        xmin, ymin, xmax, ymax = self.bounds
        pts = []

        while len(pts) < n:
            cand = rng.uniform([xmin, ymin], [xmax, ymax], size=(n, 2))
            mask = self.contains(cand)
            pts.extend(cand[mask])

        return np.array(pts[:n])

    def sample_near_boundary(self, n, decay=2.0, seed=None):
        rng = np.random.default_rng(seed)
        boundary = np.array(self.poly.exterior.coords)
        tree = KDTree(boundary)

        pts = []
        xmin, ymin, xmax, ymax = self.bounds

        while len(pts) < n:
            cand = rng.uniform([xmin, ymin], [xmax, ymax], size=(n, 2))
            cand = cand[self.contains(cand)]

            d, _ = tree.query(cand)
            w = np.exp(-decay * d / (d.max() + 1e-8))
            w /= w.sum()

            sel = rng.choice(len(cand), size=min(n - len(pts), len(cand)), p=w)
            pts.extend(cand[sel])

        return np.array(pts)
    
    def bounding_box(self):
        xmin, ymin, xmax, ymax = self.bounds
        return np.array([xmin, ymin]), np.array([xmax, ymax])
