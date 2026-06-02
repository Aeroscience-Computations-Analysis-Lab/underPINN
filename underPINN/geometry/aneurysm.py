"""Axisymmetric bulge (aneurysm-like) geometry aligned with the x-axis."""
import numpy as np


class BulgeGeometry:
    """Axisymmetric bulge geometry — straight vessel with a local outward bulge.

    The cross-section is always circular; its radius varies along the axial
    direction x according to a cosine-squared profile:

        R(x) = R_vessel + (R_aneurysm - R_vessel)
                        * cos(π/2 · |x - x0| / half_bulge)²
               for |x - x0| ≤ half_bulge, else  R_vessel

    This gives a smooth (C¹-continuous) transition between the straight-vessel
    radius R_vessel and the peak bulge radius R_aneurysm at x = x0.

    Parameters
    ----------
    R_vessel   : radius of the straight vessel sections
    R_aneurysm : maximum radius at the centre of the bulge
    L          : total domain length;  x ∈ [x_lo, x_lo + L]
    x_lo       : x-coordinate of the inlet face  (default −L/2)
    x0         : x-coordinate of the bulge centre
    L_aneurysm : full width of the bulge region along x
    """

    def __init__(
        self,
        R_vessel:   float = 0.5,
        R_aneurysm: float = 1.0,
        L:          float = 7.0,
        x_lo:       float = -3.5,
        x0:         float = 1.5,
        L_aneurysm: float = 1.5,
    ):
        self.R_vessel   = float(R_vessel)
        self.R_aneurysm = float(R_aneurysm)
        self.L          = float(L)
        self.x_lo       = float(x_lo)
        self.x_hi       = float(x_lo) + float(L)
        self.x0         = float(x0)
        self.half_bulge = float(L_aneurysm) / 2.0

    # ------------------------------------------------------------------
    # Radius profile
    # ------------------------------------------------------------------

    def radius_at(self, x: np.ndarray) -> np.ndarray:
        """Local cross-section radius R(x) — vectorised over x."""
        x   = np.asarray(x, dtype=np.float64)
        dx  = np.abs(x - self.x0)
        cos_sq = np.cos(0.5 * np.pi * np.clip(dx / self.half_bulge, 0.0, 1.0)) ** 2
        inside = dx <= self.half_bulge
        r = np.where(
            inside,
            self.R_vessel + (self.R_aneurysm - self.R_vessel) * cos_sq,
            self.R_vessel,
        )
        return r.astype(np.float32)

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample inside the bulge domain (rejection from box)."""
        rng   = np.random.default_rng(seed)
        R_max = self.R_aneurysm
        collected = []
        while sum(len(c) for c in collected) < n:
            batch = max(4 * n, 2048)
            xs = rng.uniform(self.x_lo, self.x_hi, batch).astype(np.float32)
            ys = rng.uniform(-R_max, R_max, batch).astype(np.float32)
            zs = rng.uniform(-R_max, R_max, batch).astype(np.float32)
            r_local = self.radius_at(xs)
            keep    = ys ** 2 + zs ** 2 <= r_local ** 2
            collected.append(np.column_stack([xs[keep], ys[keep], zs[keep]]))
        return np.concatenate(collected)[:n]

    def sample_wall(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the variable-radius no-slip wall."""
        rng   = np.random.default_rng(seed)
        x     = rng.uniform(self.x_lo, self.x_hi, n).astype(np.float32)
        theta = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        r_loc = self.radius_at(x)
        y = (r_loc * np.cos(theta)).astype(np.float32)
        z = (r_loc * np.sin(theta)).astype(np.float32)
        return np.column_stack([x, y, z])

    def sample_inlet(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the inlet disk x = x_lo  (radius R_vessel)."""
        rng = np.random.default_rng(seed)
        collected = []
        while sum(len(c) for c in collected) < n:
            batch = max(4 * n, 1024)
            ys = rng.uniform(-self.R_vessel, self.R_vessel, batch).astype(np.float32)
            zs = rng.uniform(-self.R_vessel, self.R_vessel, batch).astype(np.float32)
            keep = ys ** 2 + zs ** 2 <= self.R_vessel ** 2
            xs = np.full(keep.sum(), self.x_lo, dtype=np.float32)
            collected.append(np.column_stack([xs, ys[keep], zs[keep]]))
        return np.concatenate(collected)[:n]

    def sample_outlet(self, n: int, seed: int = 0) -> np.ndarray:
        """Uniform random sample on the outlet disk x = x_hi  (radius R_vessel)."""
        pts = self.sample_inlet(n, seed).copy()
        pts[:, 0] = self.x_hi
        return pts
