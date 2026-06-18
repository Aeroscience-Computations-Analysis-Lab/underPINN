"""Write ParaView-readable ``.vtu`` files (VTK XML UnstructuredGrid).

Pure-NumPy, no external dependency — the XML is emitted directly in ASCII so
the output opens straight in ParaView / VTK / PyVista.  Two writers cover the
common PINN post-processing needs:

* :func:`save_vtu_points`  — a point cloud (one ``VTK_VERTEX`` per point), e.g.
  the interior collocation points carrying the predicted flow field.
* :func:`save_vtu_surface` — a triangulated surface (``VTK_TRIANGLE`` cells),
  e.g. the vessel wall carrying pressure / wall-shear-stress.

Point data may be scalar ``(N,)`` or vector ``(N, 3)`` (written with three
components, the natural choice for velocity).  In ParaView a point cloud can be
rendered directly (Point Gaussian), turned into glyphs, meshed with *Delaunay
3D*, or used as a seed/volume for *Stream Tracer*.
"""
from __future__ import annotations

import io
import pathlib
from typing import Mapping

import numpy as np

# VTK cell type ids
_VTK_VERTEX = 1
_VTK_TRIANGLE = 5


def _ascii(arr, fmt: str) -> str:
    """Flatten *arr* to a single whitespace-separated ASCII line."""
    a = np.asarray(arr).ravel()
    buf = io.StringIO()
    np.savetxt(buf, a[None, :], fmt=fmt)
    return buf.getvalue().strip()


def _point_data_xml(point_data: Mapping[str, np.ndarray], n_pts: int) -> str:
    if not point_data:
        return "      <PointData></PointData>\n"
    lines = ["      <PointData>\n"]
    for name, arr in point_data.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            ncomp = 1
        elif a.ndim == 2 and a.shape[1] in (1, 3):
            ncomp = a.shape[1]
        else:
            raise ValueError(
                f"point_data['{name}'] must be (N,), (N,1) or (N,3); got {a.shape}")
        if a.shape[0] != n_pts:
            raise ValueError(
                f"point_data['{name}'] has {a.shape[0]} rows, expected {n_pts}")
        lines.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="ascii">\n')
        lines.append("          " + _ascii(a, "%.7g") + "\n")
        lines.append("        </DataArray>\n")
    lines.append("      </PointData>\n")
    return "".join(lines)


def _write_vtu(path, points, connectivity, offsets, types,
               point_data) -> str:
    points = np.asarray(points, dtype=np.float64)
    n_pts = points.shape[0]
    n_cells = len(offsets)
    xml = []
    xml.append('<?xml version="1.0"?>\n')
    xml.append('<VTKFile type="UnstructuredGrid" version="0.1" '
               'byte_order="LittleEndian">\n')
    xml.append("  <UnstructuredGrid>\n")
    xml.append(f'    <Piece NumberOfPoints="{n_pts}" '
               f'NumberOfCells="{n_cells}">\n')
    # Points
    xml.append("      <Points>\n")
    xml.append('        <DataArray type="Float32" NumberOfComponents="3" '
               'format="ascii">\n')
    xml.append("          " + _ascii(points, "%.7g") + "\n")
    xml.append("        </DataArray>\n")
    xml.append("      </Points>\n")
    # Cells
    xml.append("      <Cells>\n")
    xml.append('        <DataArray type="Int64" Name="connectivity" '
               'format="ascii">\n')
    xml.append("          " + _ascii(connectivity, "%d") + "\n")
    xml.append("        </DataArray>\n")
    xml.append('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
    xml.append("          " + _ascii(offsets, "%d") + "\n")
    xml.append("        </DataArray>\n")
    xml.append('        <DataArray type="UInt8" Name="types" format="ascii">\n')
    xml.append("          " + _ascii(types, "%d") + "\n")
    xml.append("        </DataArray>\n")
    xml.append("      </Cells>\n")
    # Point data
    xml.append(_point_data_xml(point_data, n_pts))
    xml.append("    </Piece>\n")
    xml.append("  </UnstructuredGrid>\n")
    xml.append("</VTKFile>\n")

    path = str(path)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(xml))
    return path


def save_vtu_points(path, points, point_data: Mapping[str, np.ndarray] | None = None
                    ) -> str:
    """Write *points* ``(N, 3)`` as a ``.vtu`` point cloud (one vertex per point).

    ``point_data`` maps a field name to a ``(N,)`` scalar or ``(N, 3)`` vector
    array (e.g. ``{"velocity": uvw, "pressure": p}``).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    connectivity = np.arange(n, dtype=np.int64)
    offsets = np.arange(1, n + 1, dtype=np.int64)
    types = np.full(n, _VTK_VERTEX, dtype=np.uint8)
    return _write_vtu(path, points, connectivity, offsets, types,
                      point_data or {})


def save_vtu_surface(path, points, triangles,
                     point_data: Mapping[str, np.ndarray] | None = None) -> str:
    """Write a triangulated surface ``(points, triangles)`` as a ``.vtu``.

    ``points`` is ``(N, 3)``; ``triangles`` is ``(M, 3)`` integer vertex
    indices.  ``point_data`` is per-vertex (same convention as
    :func:`save_vtu_points`).
    """
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"triangles must be (M, 3); got {tris.shape}")
    m = tris.shape[0]
    connectivity = tris.ravel()
    offsets = np.arange(3, 3 * m + 1, 3, dtype=np.int64)
    types = np.full(m, _VTK_TRIANGLE, dtype=np.uint8)
    return _write_vtu(path, points, connectivity, offsets, types,
                      point_data or {})
