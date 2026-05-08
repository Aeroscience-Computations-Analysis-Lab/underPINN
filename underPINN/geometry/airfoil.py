import numpy as np


class NACAAirfoil:
    """NACA 4-digit series airfoil geometry for PINN applications.

    Generates the surface profile (upper + lower), samples boundary
    points, and samples collocation points exterior to the airfoil
    using rejection sampling.

    Parameters
    ----------
    naca  : 4-character NACA designation, e.g. "0012", "2412"
    chord : chord length (default 1.0, airfoil spans x ∈ [0, chord])

    Examples
    --------
    >>> af = NACAAirfoil("0012")
    >>> xy_surface = af.surface_points(500)      # (500, 2) BC points
    >>> xy_col     = af.sample_exterior(40000)   # (40000, 2) interior pts
    >>> xy_ff      = af.farfield_boundary(300)   # (1200, 2) far-field pts
    """

    def __init__(self, naca: str = "0012", chord: float = 1.0):
        if len(naca) != 4 or not naca.isdigit():
            raise ValueError("naca must be a 4-digit string, e.g. '0012'")
        self.naca  = naca
        self.chord = chord
        self.m = int(naca[0]) / 100   # max camber fraction
        self.p = int(naca[1]) / 10    # chordwise location of max camber
        self.t = int(naca[2:]) / 100  # max thickness fraction
        self._coords = self._generate(200)   # closed surface polygon

    # ------------------------------------------------------------------
    # Profile geometry (NACA 4-digit equations)
    # ------------------------------------------------------------------

    def _thickness(self, xc):
        t = self.t
        return (t / 0.2) * (
            0.2969 * np.sqrt(np.clip(xc, 0.0, None))
            - 0.1260 * xc
            - 0.3516 * xc ** 2
            + 0.2843 * xc ** 3
            - 0.1015 * xc ** 4
        )

    def _camber_slope(self, xc):
        m, p = self.m, self.p
        if m == 0:
            return np.zeros_like(xc), np.zeros_like(xc)
        yc  = np.where(xc < p,
                       (m / p**2) * (2*p*xc - xc**2),
                       (m / (1-p)**2) * (1 - 2*p + 2*p*xc - xc**2))
        dyc = np.where(xc < p,
                       (2*m / p**2) * (p - xc),
                       (2*m / (1-p)**2) * (p - xc))
        return yc, dyc

    def _generate(self, n: int) -> np.ndarray:
        c    = self.chord
        beta = np.linspace(0.0, np.pi, n)
        xc   = 0.5 * (1.0 - np.cos(beta))          # cosine spacing ∈ [0, 1]

        yt       = self._thickness(xc) * c
        yc, dyc  = self._camber_slope(xc)
        yc      *= c
        theta    = np.arctan(dyc)

        xu = xc * c - yt * np.sin(theta)
        yu = yc     + yt * np.cos(theta)
        xl = xc * c + yt * np.sin(theta)
        yl = yc     - yt * np.cos(theta)

        upper = np.stack([xu, yu], axis=1)         # LE → TE
        lower = np.stack([xl, yl], axis=1)[::-1]   # TE → LE
        return np.concatenate([upper, lower[1:-1]], axis=0)  # no duplicate LE/TE

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def profile(self) -> np.ndarray:
        """(N_profile, 2) closed surface polygon (counterclockwise from LE)."""
        return self._coords.copy()

    def surface_points(self, n: int = 500) -> np.ndarray:
        """Arc-length-uniformly resampled surface points → (n, 2) float32."""
        p  = self._coords
        ds = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
        s  = np.concatenate([[0.0], np.cumsum(ds)])
        sq = np.linspace(0.0, s[-1], n)
        x  = np.interp(sq, s, p[:, 0])
        y  = np.interp(sq, s, p[:, 1])
        return np.stack([x, y], axis=1).astype(np.float32)

    def is_inside(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask: True if a point lies strictly inside the airfoil.

        Uses ``matplotlib.path.Path.contains_points`` (ray casting,
        fully vectorised — no Python loop over points).
        """
        from matplotlib.path import Path
        closed = np.vstack([self._coords, self._coords[:1]])
        return Path(closed).contains_points(xy)

    def sample_exterior(
        self,
        n: int,
        xmin: float = -5.0,
        xmax: float = 15.0,
        ymin: float = -8.0,
        ymax: float =  8.0,
        seed: int = 0,
    ) -> np.ndarray:
        """Rejection-sample n collocation points exterior to the airfoil.

        Points are uniform in the rectangular domain minus the airfoil.
        """
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < n:
            batch = rng.uniform([xmin, ymin], [xmax, ymax],
                                size=(max(n * 4, 20_000), 2)).astype(np.float32)
            pts.extend(batch[~self.is_inside(batch)].tolist())
        return np.array(pts[:n], dtype=np.float32)

    def sample_near_surface(
        self,
        n: int,
        x_lo: float = -0.2,
        x_hi: float =  1.2,
        y_lo: float = -0.5,
        y_hi: float =  0.5,
        seed: int = 1,
    ) -> np.ndarray:
        """Dense exterior sample in the near-surface bounding box.

        Adds resolution in the boundary layer and near wake without
        increasing the far-field collocation density.
        """
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < n:
            batch = rng.uniform([x_lo, y_lo], [x_hi, y_hi],
                                size=(max(n * 6, 20_000), 2)).astype(np.float32)
            pts.extend(batch[~self.is_inside(batch)].tolist())
        return np.array(pts[:n], dtype=np.float32)

    def farfield_boundary(
        self,
        n_per_edge: int = 300,
        xmin: float = -5.0,
        xmax: float = 15.0,
        ymin: float = -8.0,
        ymax: float =  8.0,
    ) -> np.ndarray:
        """Sample points on the four edges of a rectangular far-field boundary."""
        t_x = np.linspace(xmin, xmax, n_per_edge, dtype=np.float32)
        t_y = np.linspace(ymin, ymax, n_per_edge, dtype=np.float32)
        bottom = np.stack([t_x, np.full(n_per_edge, ymin, np.float32)], axis=1)
        top    = np.stack([t_x, np.full(n_per_edge, ymax, np.float32)], axis=1)
        left   = np.stack([np.full(n_per_edge, xmin, np.float32), t_y], axis=1)
        right  = np.stack([np.full(n_per_edge, xmax, np.float32), t_y], axis=1)
        return np.concatenate([bottom, top, left, right], axis=0)
