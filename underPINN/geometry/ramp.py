"""Ramp geometry for 2-D supersonic / compressible flow problems.

Domain (trapezoidal):
  x ∈ [0, L]
  y ∈ [y_wall(x), H]   where  y_wall(x) = x · tan(θ)

Boundaries:
  Inlet  : x = 0,  y ∈ [0, H]                        (supersonic inflow — all vars specified)
  Wall   : y = x · tan(θ)                             (slip for Euler: u·n = 0)
  Upper  : y = H                                       (freestream farfield)
  Outlet : x = L,  y ∈ [L·tan(θ), H]                 (supersonic outflow — no BC needed)

The outward normal to the ramp wall (pointing into the fluid) is:
  n = (−sin θ,  cos θ)
"""
from __future__ import annotations

import math
import numpy as np


class RampGeometry:
    """2-D wedge/ramp domain above a flat inclined lower wall.

    Parameters
    ----------
    theta_deg : float
        Ramp angle in degrees (measured from horizontal).
    L : float
        Domain length in x (default 1.0).
    H : float
        Domain height (y extent at x = 0, default 0.8).
    """

    def __init__(self, theta_deg: float, L: float = 1.0, H: float = 0.8,
                 ramp_start: float = 0.0, slip_end: float = 0.0):
        self.theta     = math.radians(theta_deg)
        self.tan_theta = math.tan(self.theta)
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        self.L = float(L)
        self.H = float(H)
        # x at which the ramp begins.  0 ⇒ a pure inclined ramp from the inlet
        # (Euler case).  > 0 ⇒ a flat bottom wall on [0, ramp_start] followed by
        # a compression ramp — the SBLI / compression-corner geometry.
        self.ramp_start = float(ramp_start)
        # x up to which the flat bottom is a *slip* wall (v = 0 only); the
        # no-slip wall then runs from slip_end to L.  0 ⇒ no slip region.
        self.slip_end = float(slip_end)

    # ------------------------------------------------------------------
    # Lower wall  y_wall(x)
    # ------------------------------------------------------------------

    def y_wall(self, x):
        """Lower-wall height: flat (=0) before ``ramp_start``, then inclined."""
        return np.maximum(0.0, (np.asarray(x) - self.ramp_start) * self.tan_theta)

    # ------------------------------------------------------------------
    # Interior
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, seed: int = 0) -> np.ndarray:
        """Sample *n* points uniformly inside the trapezoidal domain.

        Uses rejection sampling: draw from [0,L]×[0,H], keep those
        strictly above the lower wall.
        """
        rng  = np.random.default_rng(seed)
        pts  = []
        need = n
        while need > 0:
            over = max(need * 4, 512)
            x = rng.uniform(0.0, self.L, over).astype(np.float32)
            y = rng.uniform(0.0, self.H, over).astype(np.float32)
            mask  = y > self.y_wall(x) + 1e-4        # strictly inside
            batch = np.stack([x[mask], y[mask]], axis=1)
            pts.append(batch[:need])
            need -= len(batch[:need])
        return np.concatenate(pts, axis=0)[:n]

    # ------------------------------------------------------------------
    # Boundaries
    # ------------------------------------------------------------------

    def sample_boundary_layer(self, n: int, beta: float = 4.0,
                              seed: int = 0) -> np.ndarray:
        """Interior points clustered toward the lower wall (BL resolution).

        For each x the wall-normal coordinate is geometrically **stretched** so
        the point density is highest at the wall and decays into the freestream
        — exactly the bias a boundary-layer CFD mesh uses.  ``beta`` controls the
        clustering (higher ⇒ tighter to the wall).  Guarantees the thin viscous
        layer is resolved from the first epoch, independent of the residual.
        """
        rng = np.random.default_rng(seed)
        x   = rng.uniform(0.0, self.L, n).astype(np.float32)
        yw  = self.y_wall(x)
        xi  = rng.uniform(0.0, 1.0, n)
        frac = np.expm1(beta * xi) / np.expm1(beta)      # ∈ [0,1], dense near 0
        frac = np.clip(frac, 0.0, 0.999)                 # keep strictly interior
        y = (yw + frac * (self.H - yw)).astype(np.float32)
        return np.stack([x, y], axis=1)

    def sample_inlet(self, n: int) -> np.ndarray:
        """Inlet: x = 0, y ∈ [0, H]  (uniform spacing)."""
        y = np.linspace(0.0, self.H, n, dtype=np.float32)
        x = np.zeros(n, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def sample_ramp_wall(self, n: int) -> np.ndarray:
        """Ramp wall: y = x · tan(θ),  x ∈ [0, L]  (uniform spacing)."""
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = (x * self.tan_theta).astype(np.float32)
        return np.stack([x, y], axis=1)

    def sample_lower_wall(self, n: int) -> np.ndarray:
        """Full lower wall y = y_wall(x), x ∈ [0, L] (flat bottom + ramp).

        For ``ramp_start = 0`` this is identical to :meth:`sample_ramp_wall`.
        Used for the no-slip viscous case (no normal needed for no-slip).
        """
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = self.y_wall(x).astype(np.float32)
        return np.stack([x, y], axis=1)

    def sample_slip_wall(self, n: int) -> np.ndarray:
        """Slip region on the flat bottom: x ∈ [0, slip_end], y = 0.

        Free-slip / symmetry wall — only no-penetration (v = 0) is imposed.
        """
        x = np.linspace(0.0, self.slip_end, n, dtype=np.float32)
        y = self.y_wall(x).astype(np.float32)        # 0 on the flat part
        return np.stack([x, y], axis=1)

    def sample_noslip_wall(self, n: int) -> np.ndarray:
        """No-slip region: x ∈ [slip_end, L], y = y_wall(x) (flat tail + ramp)."""
        x = np.linspace(self.slip_end, self.L, n, dtype=np.float32)
        y = self.y_wall(x).astype(np.float32)
        return np.stack([x, y], axis=1)

    def sample_upper(self, n: int) -> np.ndarray:
        """Upper farfield: y = H,  x ∈ [0, L]  (uniform spacing)."""
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = np.full(n, self.H, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def sample_outlet(self, n: int) -> np.ndarray:
        """Outlet: x = L,  y ∈ [y_wall(L), H]  (supersonic — reference only)."""
        y_lo = float(self.y_wall(self.L))
        y    = np.linspace(y_lo, self.H, n, dtype=np.float32)
        x    = np.full(n, self.L, dtype=np.float32)
        return np.stack([x, y], axis=1)

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def ramp_normal(self) -> np.ndarray:
        """Outward unit normal to the ramp wall (pointing into the fluid).

        The ramp tangent is (cos θ, sin θ); rotating 90° CCW gives the
        normal pointing away from the solid:  n = (−sin θ, cos θ).
        """
        return np.array([-self.sin_theta, self.cos_theta], dtype=np.float32)

    # ------------------------------------------------------------------
    # Grid (for evaluation / visualisation)
    # ------------------------------------------------------------------

    def make_grid(self, Nx: int = 200, Ny: int = 160):
        """Create a regular (x, y) grid with points below the ramp masked.

        Returns
        -------
        XX, YY : (Ny, Nx) numpy arrays (meshgrid, indexing='xy')
        mask   : boolean (Ny, Nx), True for interior points
        """
        x_arr = np.linspace(0.0, self.L, Nx, dtype=np.float32)
        y_arr = np.linspace(0.0, self.H, Ny, dtype=np.float32)
        XX, YY = np.meshgrid(x_arr, y_arr)              # (Ny, Nx)
        mask   = YY > self.y_wall(XX) + 1e-4
        return XX, YY, mask
