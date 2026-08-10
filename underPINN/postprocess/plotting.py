"""Publication-quality matplotlib defaults + shared field-plot helpers.

Importing this module applies a consistent figure style (perceptually-ordered
``turbo`` colormap, 220-dpi output, clean spines/typography) used across the
post-processing utilities and example visualisations.  Public helpers:

* :data:`CMAP`        — the default colormap name.
* :func:`field`       — high-quality filled contour for a smooth scalar field.
* :func:`cbar`        — a tidy, compact colorbar.
* :func:`save_fig`    — save + close at the standard DPI.

The leading-underscore aliases ``_CMAP / _field / _cbar / _save`` are kept for
backward compatibility with existing call sites.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi":        120,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "font.size":         11,
    "axes.titlesize":    12.5,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#444444",
    "axes.grid":         False,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "lines.linewidth":   1.9,
    "lines.antialiased": True,
    "legend.frameon":    False,
    "legend.fontsize":   9.5,
    "image.cmap":        "turbo",
})

CMAP = "turbo"        # perceptually-ordered replacement for the old "jet"
SAVE_DPI = 220


def field(ax, X, Y, C, levels=120, cmap=CMAP, **kw):
    """High-quality filled contour for a smooth scalar field."""
    return ax.contourf(X, Y, C, levels=levels, cmap=cmap, antialiased=True, **kw)


def cbar(fig, mappable, ax, label, **kw):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, **kw)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.6)
    return cb


def save_fig(fig, path):
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# Backward-compatible aliases (previously private helpers in the examples).
_CMAP = CMAP
_field = field
_cbar = cbar
_save = save_fig
