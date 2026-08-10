import numpy as np


def sample_interior(geom, n, seed=None):
    return geom.sample(n, seed)


def sample_where(geom, predicate, n, seed=None):
    """
    Sample points satisfying a custom condition.
    predicate: function(x) -> bool mask
    """
    pts = []
    rng = np.random.default_rng(seed)

    while len(pts) < n:
        x = geom.sample(n, seed=rng.integers(1e9))
        mask = predicate(x)
        pts.append(x[mask])

    return np.concatenate(pts, axis=0)[:n]
