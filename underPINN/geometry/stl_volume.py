"""Sample collocation points from a watertight 3-D STL geometry.

Builds the interior / wall / inlet / outlet point clouds directly from an STL
surface, so a new patient-specific case is a drop-in (just supply the STL
files).  Backed by :mod:`trimesh` — install it with ``pip install trimesh``.

Typical layout
--------------
This mirrors the surface decomposition used by the NVIDIA PhysicsNeMo
aneurysm tutorial.  ``closed_stl`` is the watertight outer surface (the
interior is rejection-sampled inside it).  The remaining surfaces tag the
boundary conditions:

* ``inlet_stl``    — inlet cap (parabolic velocity prescribed here)
* ``outlet_stl``   — outlet cap (zero-pressure / outflow)
* ``noslip_stl``   — the vessel wall (no-slip).  When supplied, wall points
  are sampled *directly* from this surface, which already excludes the caps
  (``closed = noslip + inlet + outlet``), so no cap-carving heuristic runs.
* ``integral_stl`` — a mid-stream planar disk on which mass-flow continuity
  (∫ ρ u·n dA = Q) is enforced.  When supplied, the plane points / normal /
  area come straight from this mesh instead of cutting a slice through the
  closed surface.

Use
---
>>> geom = STLVolumeGeometry(
...     closed_stl  ="aneurysm_closed.stl",
...     inlet_stl   ="aneurysm_inlet.stl",
...     outlet_stl  ="aneurysm_outlet.stl",
...     noslip_stl  ="aneurysm_noslip.stl",
...     integral_stl="aneurysm_integral.stl",
...     center=[-18.4, -50.3, 12.8],  scale=0.4,
... )
>>> xyz_int = geom.sample_interior(50_000, seed=0)
>>> xyz_in  = geom.sample_inlet  (2_000,  seed=1)
>>> xyz_out = geom.sample_outlet (2_000,  seed=2)
>>> xyz_w   = geom.sample_wall   (10_000, seed=3)
>>> xyz_q, n_q, A_q = geom.sample_integral(2_000, seed=4)

All four samplers honor the affine map ``x_sim = (x_raw − center) · scale``
applied at load time, so the returned coordinates match the simulation frame.

Generated point clouds are cached to ``<cache_dir>/<key>.npz`` (where
``<key>`` hashes the STL paths + transform + the requested ``n``) so re-runs
return instantly.  Pass ``cache_dir=None`` to disable caching.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Optional, Sequence

import numpy as np


def _need_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:                                # pragma: no cover
        raise ImportError(
            "STLVolumeGeometry requires the 'trimesh' package.\n"
            "Install it with:  pip install trimesh"
        ) from e


def _affine(pts: np.ndarray, center, scale: float) -> np.ndarray:
    return (np.asarray(pts) - np.asarray(center)) * scale


# ---------------------------------------------------------------------------
# Point-in-mesh: ray casting (+x direction) with 2-D spatial-hash pre-filter
# ---------------------------------------------------------------------------
# A naive vectorized ray-cast against a 37k-triangle mesh is O(N·T) — fine
# for a few hundred queries, hopeless for ~10⁵.  rtree (libspatialindex) is
# the standard BVH backend used by trimesh, but installing it isn't always
# convenient.  Instead, build a tiny pure-NumPy spatial hash binned in (y, z)
# — the dimensions perpendicular to the +x ray.  Each cell records the
# triangles whose (y, z) bbox overlaps it; querying a point retrieves only
# those triangles, typically 1–2 % of the total.  Möller–Trumbore then runs
# on the filtered set.  No external dependency, ~50–200× faster than naive.

class _RayCaster:
    """Encapsulates the precomputed triangle data + (y, z) spatial hash."""

    def __init__(self, triangles: np.ndarray, grid: int = 64):
        tri = np.asarray(triangles, dtype=np.float64)
        self.tri = tri
        self.v0  = tri[:, 0]
        self.e1  = tri[:, 1] - self.v0
        self.e2  = tri[:, 2] - self.v0

        # +x ray direction.  pvec = d×e2 simplifies to (0, e2_z, -e2_y)
        d = np.array([1.0, 0.0, 0.0])
        pvec = np.cross(d, self.e2)                       # (T, 3)
        det  = np.einsum("ij,ij->i", self.e1, pvec)       # (T,)
        self.pvec  = pvec
        self.inv_d = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)
        self.valid_tri = np.abs(det) > 1e-12

        # Per-triangle bounding boxes
        ty_min = tri[:, :, 1].min(axis=1)
        ty_max = tri[:, :, 1].max(axis=1)
        tz_min = tri[:, :, 2].min(axis=1)
        tz_max = tri[:, :, 2].max(axis=1)
        tx_max = tri[:, :, 0].max(axis=1)
        self.ty_min, self.ty_max = ty_min, ty_max
        self.tz_min, self.tz_max = tz_min, tz_max
        self.tx_max = tx_max

        # 2-D spatial hash on (y, z).  Each cell stores triangle indices that
        # overlap it.  Stored as a flat (cell_idx → tri_idx) list-of-arrays.
        y_lo, y_hi = ty_min.min(), ty_max.max()
        z_lo, z_hi = tz_min.min(), tz_max.max()
        # tiny pad to avoid edge cases
        pad_y = max(1e-9, (y_hi - y_lo) * 1e-9)
        pad_z = max(1e-9, (z_hi - z_lo) * 1e-9)
        self.y_lo, self.y_hi = y_lo - pad_y, y_hi + pad_y
        self.z_lo, self.z_hi = z_lo - pad_z, z_hi + pad_z
        self.G = int(grid)
        self.dy = (self.y_hi - self.y_lo) / self.G
        self.dz = (self.z_hi - self.z_lo) / self.G

        # For each triangle, list of cells (iy, iz) it overlaps
        iy0 = np.clip(((ty_min - self.y_lo) / self.dy).astype(int),
                      0, self.G - 1)
        iy1 = np.clip(((ty_max - self.y_lo) / self.dy).astype(int),
                      0, self.G - 1)
        iz0 = np.clip(((tz_min - self.z_lo) / self.dz).astype(int),
                      0, self.G - 1)
        iz1 = np.clip(((tz_max - self.z_lo) / self.dz).astype(int),
                      0, self.G - 1)

        cells: list[list[int]] = [[] for _ in range(self.G * self.G)]
        for t in range(len(tri)):
            for iy in range(iy0[t], iy1[t] + 1):
                base = iy * self.G
                for iz in range(iz0[t], iz1[t] + 1):
                    cells[base + iz].append(t)
        self.cells = [np.asarray(c, dtype=np.int64) for c in cells]

    def _cell(self, y: float, z: float) -> int:
        iy = int(np.clip(int((y - self.y_lo) / self.dy), 0, self.G - 1))
        iz = int(np.clip(int((z - self.z_lo) / self.dz), 0, self.G - 1))
        return iy * self.G + iz

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: True if a point is inside the closed (watertight) mesh."""
        P = np.asarray(points, dtype=np.float64)
        out = np.zeros(len(P), dtype=bool)
        for i in range(len(P)):
            px, py, pz = P[i]
            cand = self.cells[self._cell(py, pz)]
            if len(cand) == 0:
                continue
            # AABB filter on the cell's triangle list
            m = (self.ty_min[cand] <= py) & (py <= self.ty_max[cand]) \
                & (self.tz_min[cand] <= pz) & (pz <= self.tz_max[cand]) \
                & (self.tx_max[cand] >= px) & self.valid_tri[cand]
            idx = cand[m]
            if len(idx) == 0:
                continue
            # Möller-Trumbore for ray +x on the filtered triangles
            tvec = P[i] - self.v0[idx]                       # (k, 3)
            u    = (tvec * self.pvec[idx]).sum(axis=1) * self.inv_d[idx]
            qvec = np.cross(tvec, self.e1[idx])
            v    = qvec[:, 0] * self.inv_d[idx]             # qvec · (1,0,0)
            t    = (qvec * self.e2[idx]).sum(axis=1) * self.inv_d[idx]
            hit  = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
            out[i] = (hit.sum() % 2 == 1)
        return out


def _polygon_area(poly2d: np.ndarray) -> float:
    """Shoelace area of a 2-D polygon (vertices in order, open form)."""
    x = poly2d[:, 0]
    y = poly2d[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_in_polygon_2d(poly: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting: True if a point is inside the 2-D polygon."""
    x1, y1 = poly[:, 0], poly[:, 1]                          # (V,)
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)                # (V,)
    px = points[:, 0:1]
    py = points[:, 1:2]                                      # (N, 1)
    straddles = (y1[None, :] <= py) != (y2[None, :] <= py)   # (N, V)
    denom = (y2 - y1)[None, :]
    safe  = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    x_int = x1[None, :] + (py - y1[None, :]) * (x2 - x1)[None, :] / safe
    crossings = np.sum(straddles & (x_int > px), axis=1)
    return (crossings % 2) == 1


def _sample_polygon_uniform(poly: np.ndarray, n: int, rng) -> np.ndarray:
    """Uniform area-weighted sample of *n* points inside a 2-D polygon
    (rejection in the bounding box)."""
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    out: list = []
    while sum(len(a) for a in out) < n:
        batch = rng.uniform(lo, hi, size=(max(4 * n, 1024), 2))
        out.append(batch[_point_in_polygon_2d(poly, batch)])
    return np.concatenate(out, axis=0)[:n]


class STLVolumeGeometry:
    """Sample collocation points from an STL surface mesh.

    Parameters
    ----------
    closed_stl    : path to the watertight outer surface (required)
    inlet_stl     : optional STL of the inlet cap (a small disk-like surface)
    outlet_stl    : optional STL of the outlet cap
    noslip_stl    : optional STL of the vessel wall (no-slip surface).  When
                    given, wall points are sampled directly from it and the
                    cap-carving heuristic is skipped.
    integral_stl  : optional STL of the mid-stream mass-flow plane.  When
                    given, ``sample_integral`` reads its points / normal /
                    area straight off this mesh.
    center, scale : affine map  x_sim = (x_raw − center) · scale  (default identity)
    cache_dir     : where to cache generated point clouds (None disables)
    cap_pad       : extra radius used when identifying closed-surface points
                    that fall on the inlet/outlet caps, so they are excluded
                    from no-slip wall samples (in simulation units, default 0.05).
                    Only used as a fallback when ``noslip_stl`` is not supplied.
    """

    def __init__(
        self,
        closed_stl,
        inlet_stl: Optional[str] = None,
        outlet_stl: Optional[str] = None,
        noslip_stl: Optional[str] = None,
        integral_stl: Optional[str] = None,
        center: Optional[Sequence[float]] = None,
        scale: float = 1.0,
        cache_dir: Optional[str] = "outputs/_stl_cache",
        cap_pad: float = 0.05,
    ):
        trimesh = _need_trimesh()

        self.center  = np.asarray(center if center is not None else [0.0, 0.0, 0.0],
                                  dtype=np.float64)
        self.scale   = float(scale)
        self.cap_pad = float(cap_pad)

        # ── Load + transform the closed (watertight) surface ─────────────────
        self._closed_path = str(closed_stl)
        m_closed = trimesh.load_mesh(self._closed_path, force="mesh")
        m_closed.apply_translation(-self.center)
        m_closed.apply_scale(self.scale)
        self.mesh = m_closed
        self.bounds = m_closed.bounds.astype(np.float64)  # (2, 3) min/max

        # ── Load + transform the inlet/outlet cap surfaces (optional) ────────
        self._inlet_path  = str(inlet_stl)  if inlet_stl  else None
        self._outlet_path = str(outlet_stl) if outlet_stl else None
        self.inlet_mesh   = self._load_cap(inlet_stl)
        self.outlet_mesh  = self._load_cap(outlet_stl)

        # ── Load + transform the dedicated wall / integral surfaces ──────────
        self._noslip_path   = str(noslip_stl)   if noslip_stl   else None
        self._integral_path = str(integral_stl) if integral_stl else None
        self.noslip_mesh    = self._load_cap(noslip_stl)
        self.integral_mesh  = self._load_cap(integral_stl)

        # Cap descriptors (centroid + bounding sphere radius) — used to
        # carve the matching region out of the closed surface when picking
        # wall (no-slip) points.
        self.inlet_center  = (self.inlet_mesh.centroid
                              if self.inlet_mesh is not None else None)
        self.outlet_center = (self.outlet_mesh.centroid
                              if self.outlet_mesh is not None else None)
        self.inlet_radius  = self._cap_radius(self.inlet_mesh)
        self.outlet_radius = self._cap_radius(self.outlet_mesh)

        # Caching
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy ray-caster (built on first interior-sample call)
        self._caster: Optional[_RayCaster] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cap(self, path):
        if path is None:
            return None
        trimesh = _need_trimesh()
        m = trimesh.load_mesh(str(path), force="mesh")
        m.apply_translation(-self.center)
        m.apply_scale(self.scale)
        return m

    @staticmethod
    def _cap_radius(cap_mesh):
        if cap_mesh is None:
            return 0.0
        d = cap_mesh.vertices - cap_mesh.centroid
        return float(np.linalg.norm(d, axis=1).max())

    def _cache_key(self, kind: str, n: int, seed: int) -> str:
        h = hashlib.md5(
            f"{self._closed_path}|{self._inlet_path}|{self._outlet_path}|"
            f"{self._noslip_path}|{self._integral_path}|"
            f"{self.center.tolist()}|{self.scale}|{self.cap_pad}|"
            f"{kind}|{n}|{seed}".encode()
        ).hexdigest()[:16]
        return h

    def _cache_load(self, kind, n, seed):
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{kind}_{self._cache_key(kind, n, seed)}.npz"
        if p.exists():
            return np.load(p)["xyz"].astype(np.float32)
        return None

    def _cache_save(self, kind, n, seed, xyz):
        if self.cache_dir is None:
            return
        p = self.cache_dir / f"{kind}_{self._cache_key(kind, n, seed)}.npz"
        np.savez(p, xyz=xyz.astype(np.float32))

    def _is_on_cap(self, pts: np.ndarray) -> np.ndarray:
        """Boolean: True if a point is closer to an inlet/outlet centre than
        that cap's radius + cap_pad — i.e. it belongs to a cap face."""
        mask = np.zeros(len(pts), dtype=bool)
        if self.inlet_center is not None:
            d = np.linalg.norm(pts - self.inlet_center, axis=1)
            mask |= d <= self.inlet_radius + self.cap_pad
        if self.outlet_center is not None:
            d = np.linalg.norm(pts - self.outlet_center, axis=1)
            mask |= d <= self.outlet_radius + self.cap_pad
        return mask

    # ------------------------------------------------------------------
    # Public samplers
    # ------------------------------------------------------------------

    def sample_interior(self, n: int, seed: int = 0) -> np.ndarray:
        """*n* points uniformly distributed inside the closed mesh."""
        cached = self._cache_load("interior", n, seed)
        if cached is not None:
            return cached

        rng = np.random.default_rng(seed)
        lo, hi = self.bounds[0], self.bounds[1]

        # Build the spatial-hash ray-caster once; reuse across batches.
        if self._caster is None:
            self._caster = _RayCaster(self.mesh.triangles)

        pts: list = []
        batch = max(4 * n, 5000)
        while sum(len(a) for a in pts) < n:
            cand   = rng.uniform(lo, hi, size=(batch, 3))
            inside = self._caster.contains(cand)
            pts.append(cand[inside])
            if len(pts[-1]) < batch * 0.02:
                batch *= 2
        xyz = np.concatenate(pts, axis=0)[:n].astype(np.float32)
        self._cache_save("interior", n, seed, xyz)
        return xyz

    def sample_inlet(self, n: int, seed: int = 0) -> np.ndarray:
        if self.inlet_mesh is None:
            raise ValueError("No inlet STL provided.")
        return self._sample_cap(self.inlet_mesh, n, seed, "inlet")

    def sample_outlet(self, n: int, seed: int = 0) -> np.ndarray:
        if self.outlet_mesh is None:
            raise ValueError("No outlet STL provided.")
        return self._sample_cap(self.outlet_mesh, n, seed, "outlet")

    def _sample_cap(self, mesh, n, seed, kind):
        cached = self._cache_load(kind, n, seed)
        if cached is not None:
            return cached
        trimesh = _need_trimesh()
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        xyz = pts.astype(np.float32)
        self._cache_save(kind, n, seed, xyz)
        return xyz

    def sample_plane(
        self,
        origin: Sequence[float],
        normal: Sequence[float],
        n: int,
        seed: int = 0,
    ) -> tuple:
        """Sample *n* points on the planar cross-section of the closed mesh.

        The plane (``origin``, ``normal``) is sliced through the watertight
        surface; the resulting closed polygons are sampled uniformly by area.
        Returns ``(xyz, normals, area)`` where every point's normal is the
        plane normal — exactly what the integral-continuity loss needs.

        Useful for defining a virtual flow-rate plane when no STL of the
        cross-section was supplied (here, the source ``aneurysm_integral.stl``
        is 0 bytes — so we cut a slice ourselves).
        """
        cached_xyz = self._cache_load("plane", n, seed)
        if cached_xyz is not None:
            # Cached: recompute normals (cheap; constants per call) + area
            normal_n = np.asarray(normal, np.float64)
            normal_n = normal_n / np.linalg.norm(normal_n)
            normals = np.broadcast_to(normal_n.astype(np.float32),
                                      cached_xyz.shape)
            # Area is recomputed below from the actual section
            polys2d, _, _, _ = self._slice_section(origin, normal)
            area = float(sum(_polygon_area(p) for p in polys2d))
            return cached_xyz, normals.copy(), area

        polys2d, e1, e2, plane_origin = self._slice_section(origin, normal)
        # Sample each polygon proportional to its area
        areas = np.array([_polygon_area(p) for p in polys2d])
        total = float(areas.sum())
        if total <= 0.0:
            raise ValueError("Plane does not cut the closed mesh.")
        counts = np.maximum(1, np.round(n * areas / total).astype(int))
        counts[-1] = n - counts[:-1].sum()                # exact total = n
        rng = np.random.default_rng(seed)
        pts2d: list = []
        for poly, k in zip(polys2d, counts):
            if k <= 0:
                continue
            pts2d.append(_sample_polygon_uniform(poly, k, rng))
        pts2d = np.concatenate(pts2d, axis=0)
        # Lift back to 3-D
        xyz = (plane_origin
               + np.outer(pts2d[:, 0], e1)
               + np.outer(pts2d[:, 1], e2)).astype(np.float32)
        normal_n = np.asarray(normal, np.float64)
        normal_n = normal_n / np.linalg.norm(normal_n)
        normals = np.broadcast_to(normal_n.astype(np.float32),
                                  (n, 3)).copy()
        self._cache_save("plane", n, seed, xyz)
        return xyz, normals, total

    def sample_integral(self, n: int, seed: int = 0,
                        orient: Optional[Sequence[float]] = None) -> tuple:
        """Sample *n* points on the dedicated integral-plane STL.

        This is the NVIDIA-tutorial path: the mid-stream mass-flow surface is
        supplied as its own (planar) STL, so the points, the unit normal and
        the area all come straight off that mesh — no slicing of the closed
        surface needed.  Returns ``(xyz, normals, area)`` with every row of
        ``normals`` equal to the plane's unit normal, exactly what the
        integral-continuity loss consumes.

        ``orient`` (optional) is a reference direction; the returned normal is
        flipped to point along it (use the inlet→outlet flow direction so the
        flux sign is positive downstream).
        """
        if self.integral_mesh is None:
            raise ValueError("No integral STL provided.")
        trimesh = _need_trimesh()
        cached = self._cache_load("integral", n, seed)
        normal_n = self.integral_normal
        if orient is not None:
            ref = np.asarray(orient, np.float64)
            if float(np.dot(normal_n, ref)) < 0.0:
                normal_n = -normal_n
        area = self.integral_area
        if cached is not None:
            normals = np.broadcast_to(normal_n.astype(np.float32),
                                      cached.shape).copy()
            return cached, normals, area
        pts, _ = trimesh.sample.sample_surface(self.integral_mesh, n, seed=seed)
        xyz = pts.astype(np.float32)
        normals = np.broadcast_to(normal_n.astype(np.float32), xyz.shape).copy()
        self._cache_save("integral", n, seed, xyz)
        return xyz, normals, area

    def _slice_section(self, origin, normal):
        """Cut the closed mesh with the plane and return projected 2-D polygons."""
        plane_origin = np.asarray(origin, dtype=np.float64)
        plane_normal = np.asarray(normal, dtype=np.float64)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        sec = self.mesh.section(plane_origin=plane_origin,
                                plane_normal=plane_normal)
        if sec is None:
            raise ValueError("Plane does not intersect the closed mesh.")
        # Build a local 2-D basis on the plane
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(plane_normal, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(plane_normal, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(plane_normal, e1)
        # Extract each closed polygon from the section entities
        verts = sec.vertices
        polys2d = []
        for ent in sec.entities:
            idx = np.asarray(ent.points)
            poly3 = verts[idx]
            poly2 = np.stack([(poly3 - plane_origin) @ e1,
                              (poly3 - plane_origin) @ e2], axis=1)
            # close the loop if needed
            if not np.allclose(poly2[0], poly2[-1]):
                poly2 = np.vstack([poly2, poly2[0:1]])
            polys2d.append(poly2[:-1])                     # store open-form
        return polys2d, e1, e2, plane_origin

    # ------------------------------------------------------------------
    # Cap diagnostics (centroid + area + radius from the supplied caps)
    # ------------------------------------------------------------------

    @property
    def inlet_area(self) -> float:
        return float(self.inlet_mesh.area) if self.inlet_mesh is not None else 0.0

    @property
    def outlet_area(self) -> float:
        return float(self.outlet_mesh.area) if self.outlet_mesh is not None else 0.0

    @property
    def inlet_normal(self) -> Optional[np.ndarray]:
        if self.inlet_mesh is None:
            return None
        n = self.inlet_mesh.face_normals.mean(axis=0)
        return n / (np.linalg.norm(n) + 1e-12)

    @property
    def outlet_normal(self) -> Optional[np.ndarray]:
        if self.outlet_mesh is None:
            return None
        n = self.outlet_mesh.face_normals.mean(axis=0)
        return n / (np.linalg.norm(n) + 1e-12)

    @property
    def integral_area(self) -> float:
        return (float(self.integral_mesh.area)
                if self.integral_mesh is not None else 0.0)

    @property
    def integral_center(self) -> Optional[np.ndarray]:
        return (np.asarray(self.integral_mesh.centroid, np.float64)
                if self.integral_mesh is not None else None)

    @property
    def integral_normal(self) -> Optional[np.ndarray]:
        if self.integral_mesh is None:
            return None
        n = self.integral_mesh.face_normals.mean(axis=0)
        return n / (np.linalg.norm(n) + 1e-12)

    def sample_wall(self, n: int, seed: int = 0) -> np.ndarray:
        """*n* no-slip wall points.

        When a dedicated ``noslip_stl`` was supplied (the NVIDIA-tutorial
        layout, ``closed = noslip + inlet + outlet``), points are sampled
        directly from that wall surface — it already excludes the caps, so the
        sample is exact.  Otherwise we fall back to sampling the closed surface
        and carving out the inlet/outlet caps with the ``cap_pad`` heuristic.
        """
        cached = self._cache_load("wall", n, seed)
        if cached is not None:
            return cached

        trimesh = _need_trimesh()

        if self.noslip_mesh is not None:
            pts, _ = trimesh.sample.sample_surface(self.noslip_mesh, n, seed=seed)
            xyz = pts.astype(np.float32)
            self._cache_save("wall", n, seed, xyz)
            return xyz

        out: list = []
        oversample = 2 if (self.inlet_mesh is not None
                           or self.outlet_mesh is not None) else 1
        while sum(len(a) for a in out) < n:
            pts, _ = trimesh.sample.sample_surface(
                self.mesh, oversample * n, seed=seed)
            keep = ~self._is_on_cap(np.asarray(pts))
            out.append(pts[keep])
            seed += 1
            oversample *= 2
        xyz = np.concatenate(out, axis=0)[:n].astype(np.float32)
        self._cache_save("wall", n, seed, xyz)
        return xyz

    def sample_wall_with_normals(self, n: int, seed: int = 0) -> tuple:
        """*n* wall points together with their outward unit normals.

        Returns ``(xyz, normals)`` — both ``(n, 3)`` float32.  Normals come from
        the face the sample landed on, so they are exact for the triangulated
        wall.  Uses the dedicated ``noslip_stl`` surface when available, else
        the closed surface with the inlet/outlet caps carved out.  Needed for
        wall-shear-stress (traction · n) post-processing.
        """
        trimesh = _need_trimesh()
        mesh = self.noslip_mesh if self.noslip_mesh is not None else self.mesh
        carve = self.noslip_mesh is None and (self.inlet_mesh is not None
                                              or self.outlet_mesh is not None)
        pts_out: list = []
        nrm_out: list = []
        oversample = 2 if carve else 1
        s = int(seed)
        while sum(len(a) for a in pts_out) < n:
            pts, fid = trimesh.sample.sample_surface(mesh, oversample * n, seed=s)
            nrm = mesh.face_normals[fid]
            if carve:
                keep = ~self._is_on_cap(np.asarray(pts))
                pts, nrm = pts[keep], nrm[keep]
            pts_out.append(np.asarray(pts))
            nrm_out.append(np.asarray(nrm))
            s += 1
            oversample *= 2
        xyz = np.concatenate(pts_out, axis=0)[:n].astype(np.float32)
        normals = np.concatenate(nrm_out, axis=0)[:n].astype(np.float32)
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        return xyz, normals

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def info(self) -> dict:
        return {
            "n_vertices": int(len(self.mesh.vertices)),
            "n_faces":    int(len(self.mesh.faces)),
            "is_watertight": bool(self.mesh.is_watertight),
            "bounds":     self.bounds.tolist(),
            "volume":     float(self.mesh.volume),
            "inlet_center": (None if self.inlet_center is None
                             else self.inlet_center.tolist()),
            "inlet_radius": self.inlet_radius,
            "outlet_center": (None if self.outlet_center is None
                              else self.outlet_center.tolist()),
            "outlet_radius": self.outlet_radius,
            "noslip_faces": (None if self.noslip_mesh is None
                             else int(len(self.noslip_mesh.faces))),
            "integral_area": (None if self.integral_mesh is None
                              else self.integral_area),
            "integral_center": (None if self.integral_center is None
                                else self.integral_center.tolist()),
        }
