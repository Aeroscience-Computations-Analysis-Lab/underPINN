"""2-D circular cylinder geometry for cross-flow PINN problems."""
import numpy as np


class Cylinder2D:
    """A 2-D circular cylinder for external cross-flow problems.

    Mirrors the public interface of :class:`NACAAirfoil` so the same
    collocation-sampling helpers can be reused:

        * ``profile``          — surface polygon (for plotting)
        * ``is_inside(xy)``    — boolean interior mask
        * ``sdf(xy)``          — unsigned distance to the surface
        * ``surface_points``   — boundary points (+ optional upper/lower mask)
        * ``sample_exterior``  — rejection-sampled exterior collocation points

    Parameters
    ----------
    radius : cylinder radius R  (diameter D = 2R)
    center : (cx, cy) centre of the cylinder
    """

    def __init__(self, radius: float = 0.5, center=(0.0, 0.0),
                 n_profile: int = 200):
        self.R  = float(radius)
        self.cx = float(center[0])
        self.cy = float(center[1])
        theta = np.linspace(0.0, 2.0 * np.pi, n_profile, endpoint=False)
        self._coords = np.stack([self.cx + self.R * np.cos(theta),
                                 self.cy + self.R * np.sin(theta)], axis=1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def profile(self) -> np.ndarray:
        """(N, 2) closed surface polygon (for ``ax.fill``)."""
        return self._coords.copy()

    def is_inside(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask: True if a point lies strictly inside the cylinder."""
        xy = np.asarray(xy)
        return (xy[:, 0] - self.cx) ** 2 + (xy[:, 1] - self.cy) ** 2 < self.R ** 2

    def sdf(self, xy: np.ndarray) -> np.ndarray:
        """Unsigned distance from each point to the cylinder surface."""
        xy = np.asarray(xy, dtype=np.float64)
        r  = np.sqrt((xy[:, 0] - self.cx) ** 2 + (xy[:, 1] - self.cy) ** 2)
        return np.abs(r - self.R).astype(np.float32)

    def surface_angles(self, n: int = 400) -> np.ndarray:
        """Polar angles θ ∈ [0, 2π) of evenly spaced surface points."""
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def surface_points(self, n: int = 400, return_side: bool = False):
        """Evenly spaced surface points → (n, 2) float32.

        When ``return_side=True`` also returns a boolean ``upper`` mask
        (True for the upper half, y ≥ cy).
        """
        theta = self.surface_angles(n)
        pts = np.stack([self.cx + self.R * np.cos(theta),
                        self.cy + self.R * np.sin(theta)], axis=1).astype(np.float32)
        if not return_side:
            return pts
        upper = np.sin(theta) >= 0.0
        return pts, upper

    def sample_exterior(
        self,
        n: int,
        xmin: float = -5.0,
        xmax: float = 15.0,
        ymin: float = -5.0,
        ymax: float =  5.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Rejection-sample n collocation points exterior to the cylinder."""
        rng = np.random.default_rng(seed)
        pts: list = []
        while len(pts) < n:
            batch = rng.uniform([xmin, ymin], [xmax, ymax],
                                size=(max(n * 4, 20_000), 2)).astype(np.float32)
            pts.extend(batch[~self.is_inside(batch)].tolist())
        return np.array(pts[:n], dtype=np.float32)
