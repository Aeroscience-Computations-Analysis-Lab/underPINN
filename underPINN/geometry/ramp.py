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

    def __init__(self, theta_deg: float, L: float = 1.0, H: float = 0.8):
        self.theta     = math.radians(theta_deg)
        self.tan_theta = math.tan(self.theta)
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        self.L = float(L)
        self.H = float(H)

    # ------------------------------------------------------------------
    # Interior
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, seed: int = 0) -> np.ndarray:
        """Sample *n* points uniformly inside the trapezoidal domain.

        Uses rejection sampling: draw from [0,L]×[0,H], keep those
        strictly above the ramp surface.
        """
        rng  = np.random.default_rng(seed)
        pts  = []
        need = n
        while need > 0:
            over = max(need * 4, 512)
            x = rng.uniform(0.0, self.L, over).astype(np.float32)
            y = rng.uniform(0.0, self.H, over).astype(np.float32)
            mask  = y > x * self.tan_theta + 1e-4   # strictly inside
            batch = np.stack([x[mask], y[mask]], axis=1)
            pts.append(batch[:need])
            need -= len(batch[:need])
        return np.concatenate(pts, axis=0)[:n]

    # ------------------------------------------------------------------
    # Boundaries
    # ------------------------------------------------------------------

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

    def sample_upper(self, n: int) -> np.ndarray:
        """Upper farfield: y = H,  x ∈ [0, L]  (uniform spacing)."""
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = np.full(n, self.H, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def sample_outlet(self, n: int) -> np.ndarray:
        """Outlet: x = L,  y ∈ [L·tan(θ), H]  (supersonic — reference only)."""
        y_lo = self.L * self.tan_theta
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
        mask   = YY > XX * self.tan_theta + 1e-4
        return XX, YY, mask
