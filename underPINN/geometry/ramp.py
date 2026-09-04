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

    def sample_interior(self, n: int, seed: int = 0, x_min: float = 0.0) -> np.ndarray:
        """Sample *n* points uniformly inside the trapezoidal domain.

        Uses rejection sampling: draw from [x_min,L]×[0,H], keep those
        strictly above the lower wall.

        ``x_min`` restricts the sampled region to ``x >= x_min`` — useful for
        residual-adaptive resampling, where the inlet/wall corner at x=0 (and
        any other BC-transition point, e.g. the slip/no-slip switch) is a
        geometric singularity with a much larger PDE residual than any real
        downstream feature (a shock), so an unrestricted adaptive pool gets
        dominated by that corner instead of the feature it's meant to find.
        """
        rng  = np.random.default_rng(seed)
        pts  = []
        need = n
        while need > 0:
            over = max(need * 4, 512)
            x = rng.uniform(x_min, self.L, over).astype(np.float32)
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


class CompressionExpansionRampGeometry:
    """2-D wedge/ramp domain with **two** wall corners: a concave
    (compression, oblique-shock) corner at ``x1`` followed by a convex
    (expansion, Prandtl-Meyer fan) corner at ``x2`` -- a compression-
    expansion ramp, as opposed to :class:`RampGeometry`'s single
    compression-only corner.

    Domain (piecewise-linear lower wall, three segments):
      x in [0, x1)   : flat,    y_wall = 0
      x in [x1, x2)  : incline, y_wall = (x - x1) tan(theta1)      -- rises
      x in [x2, L]   : flat,    y_wall = (x2 - x1) tan(theta1)     -- back
                                          to horizontal (theta2 = 0 case);
                                          more generally continues at
                                          tan(theta2) < tan(theta1)

    Boundaries:
      Inlet  : x = 0, y in [0, H]                    (supersonic inflow)
      Wall   : the three-segment y_wall(x) above       (slip: u.n = 0,
                                                          segment-dependent n)
      Upper  : y = H                                    (freestream farfield)
      Outlet : x = L, y in [y_wall(L), H]              (supersonic outflow)
    """

    def __init__(self, theta1_deg: float, theta2_deg: float = 0.0,
                 x1: float = 0.4, x2: float = 1.0,
                 L: float = 1.6, H: float = 0.8):
        if not (0.0 < x1 < x2 < L):
            raise ValueError(f"require 0 < x1 < x2 < L, got x1={x1}, x2={x2}, L={L}")
        self.theta1_deg, self.theta2_deg = float(theta1_deg), float(theta2_deg)
        self.theta1 = math.radians(theta1_deg)
        self.theta2 = math.radians(theta2_deg)
        self.x1, self.x2 = float(x1), float(x2)
        self.L, self.H = float(L), float(H)
        self.y_corner1 = 0.0
        self.y_corner2 = (self.x2 - self.x1) * math.tan(self.theta1)

    # ------------------------------------------------------------------
    # Lower wall  y_wall(x)  (vectorised, piecewise-linear, continuous)
    # ------------------------------------------------------------------

    def y_wall(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x)
        seg2 = (x >= self.x1) & (x < self.x2)
        seg3 = x >= self.x2
        y = np.where(seg2, (x - self.x1) * math.tan(self.theta1), y)
        y = np.where(seg3, self.y_corner2 + (x - self.x2) * math.tan(self.theta2), y)
        return y.astype(np.float32)

    def wall_segment(self, x):
        """Which wall segment each x falls in: 0 (flat), 1 (compression
        incline), or 2 (post-expansion segment) -- used to pick the correct
        per-point wall normal for the slip boundary condition, since a
        single fixed normal (as in :class:`RampGeometry`) only applies to
        one constant-angle segment here."""
        x = np.asarray(x, dtype=np.float64)
        seg = np.zeros_like(x, dtype=np.int32)
        seg = np.where(x >= self.x1, 1, seg)
        seg = np.where(x >= self.x2, 2, seg)
        return seg

    def wall_normals(self, x):
        """Outward unit normal (pointing into the fluid) at each x,
        selected by wall segment. Segment 0/2 (flat, angle theta1/theta2
        respectively -- theta2=0 for segment 0): n=(-sin(theta), cos(theta));
        segment 1 (compression incline, angle theta1): same formula with
        theta1."""
        seg = self.wall_segment(x)
        theta = np.where(seg == 1, self.theta1,
                        np.where(seg == 2, self.theta2, 0.0))
        nx = -np.sin(theta)
        ny = np.cos(theta)
        return nx.astype(np.float32), ny.astype(np.float32)

    # ------------------------------------------------------------------
    # Interior
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, seed: int = 0, x_min: float = 0.0) -> np.ndarray:
        """Rejection-sample *n* points uniformly inside the domain, above
        the (now piecewise) lower wall -- same method as
        :meth:`RampGeometry.sample_interior`."""
        rng = np.random.default_rng(seed)
        pts = []
        need = n
        while need > 0:
            over = max(need * 4, 512)
            x = rng.uniform(x_min, self.L, over).astype(np.float32)
            y = rng.uniform(0.0, self.H, over).astype(np.float32)
            mask = y > self.y_wall(x) + 1e-4
            batch = np.stack([x[mask], y[mask]], axis=1)
            pts.append(batch[:need])
            need -= len(batch[:need])
        return np.concatenate(pts, axis=0)[:n]

    # ------------------------------------------------------------------
    # Boundaries
    # ------------------------------------------------------------------

    def sample_inlet(self, n: int) -> np.ndarray:
        y = np.linspace(0.0, self.H, n, dtype=np.float32)
        x = np.zeros(n, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def sample_wall(self, n: int) -> np.ndarray:
        """Full lower wall (all three segments), uniformly spaced in x."""
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = self.y_wall(x)
        return np.stack([x, y], axis=1)

    def sample_upper(self, n: int) -> np.ndarray:
        x = np.linspace(0.0, self.L, n, dtype=np.float32)
        y = np.full(n, self.H, dtype=np.float32)
        return np.stack([x, y], axis=1)

    def sample_outlet(self, n: int) -> np.ndarray:
        y_lo = float(self.y_wall(np.array([self.L]))[0])
        y = np.linspace(y_lo, self.H, n, dtype=np.float32)
        x = np.full(n, self.L, dtype=np.float32)
        return np.stack([x, y], axis=1)

    # ------------------------------------------------------------------
    # Grid (for evaluation / visualisation)
    # ------------------------------------------------------------------

    def make_grid(self, Nx: int = 240, Ny: int = 160):
        x_arr = np.linspace(0.0, self.L, Nx, dtype=np.float32)
        y_arr = np.linspace(0.0, self.H, Ny, dtype=np.float32)
        XX, YY = np.meshgrid(x_arr, y_arr)
        mask = YY > self.y_wall(XX) + 1e-4
        return XX, YY, mask
