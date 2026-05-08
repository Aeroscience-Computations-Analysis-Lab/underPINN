import numpy as np


class Pipe:
    """Cylindrical pipe aligned with the x-axis.

    Domain: x ∈ [0, L],  r = √(y²+z²) ≤ R.

    Parameters
    ----------
    R : pipe radius
    L : pipe length
    """

    def __init__(self, R: float = 0.5, L: float = 2.0):
        self.R = R
        self.L = L

    def sample_interior(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample inside the cylinder (rejection from box)."""
        rng = np.random.default_rng(seed)
        collected = []
        while sum(len(c) for c in collected) < n:
            xs = rng.uniform(0.0,     self.L, 4 * n).astype(np.float32)
            ys = rng.uniform(-self.R, self.R, 4 * n).astype(np.float32)
            zs = rng.uniform(-self.R, self.R, 4 * n).astype(np.float32)
            keep = ys ** 2 + zs ** 2 <= self.R ** 2
            collected.append(np.column_stack([xs[keep], ys[keep], zs[keep]]))
        return np.concatenate(collected)[:n]

    def sample_wall(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the cylindrical wall r = R."""
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2 * np.pi, n).astype(np.float32)
        x = rng.uniform(0.0, self.L, n).astype(np.float32)
        y = (self.R * np.cos(theta)).astype(np.float32)
        z = (self.R * np.sin(theta)).astype(np.float32)
        return np.column_stack([x, y, z])

    def sample_inlet(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the inlet disk x = 0."""
        rng = np.random.default_rng(seed)
        collected = []
        while sum(len(c) for c in collected) < n:
            ys = rng.uniform(-self.R, self.R, 4 * n).astype(np.float32)
            zs = rng.uniform(-self.R, self.R, 4 * n).astype(np.float32)
            keep = ys ** 2 + zs ** 2 <= self.R ** 2
            xs = np.zeros(keep.sum(), dtype=np.float32)
            collected.append(np.column_stack([xs, ys[keep], zs[keep]]))
        return np.concatenate(collected)[:n]

    def sample_outlet(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the outlet disk x = L."""
        pts = self.sample_inlet(n, seed).copy()
        pts[:, 0] = self.L
        return pts
