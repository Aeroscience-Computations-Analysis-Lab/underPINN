"""underPINN.utils.pgf_export

Export the *same arrays* your matplotlib plot() methods already compute
(u_pred, u_exact, loss_hist, ...) to plain-text data files that PGFPlots
reads directly. This gives you paper-quality, vector, LaTeX-native figures
whose fonts/sizes match the rest of the document, with no re-implementation
of any solver/PDE logic -- only the rendering backend changes.

Two data shapes are covered, matching the two panel families used across
underPINN's evaluators:

  1. Continuous 2-D fields / heatmaps (Burgers, Wave space-time; Helmholtz,
     SteadyHeat (x,y); Ramp/RampNS Mach fields; PipeFlow cross-section)
     -> save_heatmap_png()  RECOMMENDED. Rasterizes only the color data
        (no text) at high DPI; the .tex template supplies real vector
        axes/ticks/colorbar on top. This is the approach that actually
        compiles at full simulation resolution (200x100 etc.) -- see the
        note in save_heatmap_png()'s docstring.
     -> save_surf_dat()     Fully-vector alternative via \\addplot3[surf].
        Only practical for small grids (roughly <=2000-3000 points total
        across ALL panels in one document) before pdfTeX's default memory
        limit is exceeded; kept here for reference / small benchmark runs.

  2. Line data  (loss curves, ODE solution vs. exact, radial profiles,
     Toro3 rho/u/p profiles) -- these have only hundreds to a few thousand
     rows and are comfortably vector.
     -> save_lines_dat()  for \\addplot table[...] line panels

Usage pattern inside an evaluator (see the patch snippets in
burgers_pgf_patch.py / ode_harmonic_pgf_patch.py for concrete examples):

    from underPINN.utils.pgf_export import save_surf_dat, save_lines_dat

    def plot_pgf(self, out_dir, suffix=""):
        ... compute u_pred, u_exact, x_plt, t_plt, exactly as in plot() ...
        rows, cols = save_surf_dat(f"{out_dir}/burgers_pinn.dat",  x_plt, t_plt, u_pred)
        save_surf_dat(f"{out_dir}/burgers_exact.dat", x_plt, t_plt, u_exact)
        save_surf_dat(f"{out_dir}/burgers_err.dat",   x_plt, t_plt, np.abs(u_pred - u_exact))
        save_lines_dat(f"{out_dir}/burgers_loss.dat",
                       epoch=np.arange(1, len(self._loss_hist) + 1),
                       total=self._loss_hist, pde=self._pde_hist)
        return rows, cols   # feed into \\pgfplotsset{... mesh/rows=<rows> ...}
"""

from __future__ import annotations

import numpy as np


'''
def save_heatmap_png(path: str, x_vals, y_vals, Z, cmap, vmin, vmax, dpi: int = 400):
    """
    Saves a complete 2-D heatmap figure including axes, ticks, and a colorbar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    Z = np.asarray(Z, dtype=float)
    assert Z.shape == (len(x_vals), len(y_vals)), (
        f"Z.shape={Z.shape} must equal (len(x_vals), len(y_vals))"
    )
    
    # 1. Create a standard figure and axis
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    
    # 2. Plot the data
    im = ax.pcolormesh(x_vals, y_vals, Z.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto")
    
    # 3. Add axes labels and a colorbar
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    
    # 4. Save the figure with a tight layout so nothing is cut off
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    
    return float(x_vals.min()), float(x_vals.max()), float(y_vals.min()), float(y_vals.max())
'''

def save_heatmap_png(path: str, x_vals, y_vals, Z, cmap, vmin, vmax, dpi: int = 400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    Z = np.asarray(Z, dtype=float)
    
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire canvas, no margins
    ax.set_axis_off()                # Remove axes/text
    ax.pcolormesh(x_vals, y_vals, Z.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)
    return float(x_vals.min()), float(x_vals.max()), float(y_vals.min()), float(y_vals.max())

def save_bar_dat(path: str, categories, **series) -> list[str]:
    """Write a categorical table for bar / grouped-bar PGFPlots charts
    (e.g. underPINN's ms_per_epoch.png, accuracy_summary_bar.png).

    Parameters
    ----------
    path : output file path
    categories : list of N short, PGFPlots-safe string keys -- one per bar
        group (use something like `r.problem`, e.g. "burgers", NOT the
        fancy display label "1-D Burgers (nu=0.01)"; symbolic x coords in
        PGFPlots are comma-parsed, so parentheses are fine but you don't
        want spaces/commas fighting the parser). Pair with
        `format_xticklabels()` below to show the fancy labels anyway.
    **series : named numeric columns, one per bar series, e.g.
        `epoch_1000=[...], epoch_5000=[...]` for a grouped bar chart, or a
        single `value=[...]` for a plain bar chart.

    Returns the list of series names actually written.
    """
    names = list(series.keys())
    arrs = [np.asarray(series[n], dtype=float) for n in names]
    n = len(categories)
    assert all(len(a) == n for a in arrs), "all series must match len(categories)"
    with open(path, "w") as f:
        f.write("category " + " ".join(names) + "\n")
        for i, cat in enumerate(categories):
            vals = " ".join(
                "nan" if not np.isfinite(a[i]) else f"{a[i]:.8g}" for a in arrs
            )
            f.write(f"{cat} {vals}\n")
    return names


def format_xticklabels(display_labels) -> str:
    """Return a ready-to-paste PGFPlots `xticklabels={...}` value.

    Each label is wrapped in its own `{...}` group, so labels containing
    commas, parentheses, or Greek letters (e.g. "1-D Burgers (nu=0.01)")
    are safe -- braced groups protect their contents from PGFPlots' own
    comma-separated-list parsing, which plain symbolic x coordinates do not
    survive if you try to put the fancy label directly there.

    Example:
        format_xticklabels(["1-D Burgers (nu=0.01)", "2-D Helmholtz (k=1)"])
        -> "{{1-D Burgers (nu=0.01)},{2-D Helmholtz (k=1)}}"
    Paste directly after `xticklabels=` in the axis options.
    """
    return "{" + ",".join("{" + str(lbl) + "}" for lbl in display_labels) + "}"


def save_surf_dat(path: str, x_vals, y_vals, Z, nan_value: str = "nan") -> tuple[int, int]:
    """Write a 2-D field to gnuplot/PGFPlots "non-uniform matrix" format.

    Parameters
    ----------
    path : output file path (e.g. "outputs/burgers_pinn.dat")
    x_vals : 1-D array, length Nx  (fast-varying axis, e.g. space)
    y_vals : 1-D array, length Ny  (slow-varying axis, e.g. time)
    Z : 2-D array, shape (Nx, Ny)  -- Z[i, j] corresponds to (x_vals[i], y_vals[j])
        This is exactly the shape you get from
        `np.meshgrid(x, y, indexing="ij")` + a model evaluation, which is
        what every evaluator in benchmarks.py already produces.
    nan_value : string written for masked/invalid points (e.g. outside a
        wedge or pipe cross-section). Requires pgfplots >= 1.16 to render
        as a gap; older versions may show artifacts -- see note at bottom
        of this file if you hit that.

    Returns
    -------
    (mesh_rows, mesh_cols) = (len(y_vals), len(x_vals))
        Pass these into the .tex file as
        `mesh/rows=<mesh_rows>, mesh/cols=<mesh_cols>`.

    Format written (blank line = new "row", i.e. new y_vals[j]):

        x0 y0 z00
        x1 y0 z10
        ...
        xN y0 zN0
        <blank line>
        x0 y1 z01
        ...
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    Z = np.asarray(Z, dtype=float)
    assert Z.shape == (len(x_vals), len(y_vals)), (
        f"Z.shape={Z.shape} must equal (len(x_vals), len(y_vals))="
        f"({len(x_vals)}, {len(y_vals)}) -- did you mean Z.T?"
    )
    with open(path, "w") as f:
        f.write("x y z\n")
        for j, yv in enumerate(y_vals):
            for i, xv in enumerate(x_vals):
                z = Z[i, j]
                zz = nan_value if not np.isfinite(z) else f"{z:.6g}"
                f.write(f"{xv:.6g} {yv:.6g} {zz}\n")
            f.write("\n")
    return len(y_vals), len(x_vals)


def save_lines_dat(path: str, **columns) -> list[str]:
    """Write any number of equal-length 1-D arrays as a tab-separated table
    with a header row, for `\\addplot table[x=..., y=...] {file.dat};`.

    Example:
        save_lines_dat("loss.dat", epoch=np.arange(1, N+1),
                        total=loss_hist, pde=pde_hist)
    writes columns "epoch total pde", letting the .tex pick any x/y pair
    (e.g. semilogy on `total`, dashed `pde` on the same axes).

    NaN entries (e.g. an evaluator that has no separate pde_hist) are
    written as "nan" and silently skipped by PGFPlots when that column is
    plotted with `unbounded coords=jump`.

    Returns the list of column names actually written, for convenience.
    """
    names = list(columns.keys())
    arrs = [np.asarray(columns[n], dtype=float) for n in names]
    n = len(arrs[0])
    assert all(len(a) == n for a in arrs), "all columns must have equal length"
    with open(path, "w") as f:
        f.write(" ".join(names) + "\n")
        for row in range(n):
            vals = []
            for a in arrs:
                v = a[row]
                vals.append("nan" if not np.isfinite(v) else f"{v:.8g}")
            f.write(" ".join(vals) + "\n")
    return names


# -----------------------------------------------------------------------
# Note on masked domains (Ramp/RampNS wedge, PipeFlow circular section):
#
# These evaluators build XX/YY over a bounding rectangle and mask out
# points outside the physical domain (mask=False -> NaN). save_surf_dat()
# still writes a full rectangular grid with "nan" at masked points, which
# recent pgfplots (>= 1.16, i.e. any TeX Live 2021+) renders as a hole in
# the surface/heatmap. If your TeX Live is older and you see stray
# triangles at the domain boundary, the robust fallback is to crop the
# *rectangle* itself to the largest inscribed axis-aligned box that is
# fully inside the physical domain, and add the true domain boundary
# (wedge wall, pipe circle) as a separate `\addplot[thick]` line drawn from
# the analytic geometry you already have (e.g. `geom.ramp_normal()`,
# the shock beta angle, or the circle x^2+y^2=R^2) rather than trying to
# infer it from the masked matrix.
# -----------------------------------------------------------------------