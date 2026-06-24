"""Comparative study: pulsatile AAA flow — Newtonian vs Carreau (non-Newtonian).

Both ``AAA_pulsatile_transfer`` (Newtonian) and ``AAA_rheology_pulsatile``
(shear-thinning Carreau) save the same windowed time-marching format, so
:class:`PulsatilePredictor` loads either.  This module overlays / side-by-sides
the two solutions through the axisymmetric AAA bulge (the comparison is
bulge-aware — the local radius R(x) is reconstructed from the saved physics):

* centreline velocity vs time at the bulge      (both + inlet forcing)
* radial velocity profiles at the bulge centre  (both overlaid, per phase)
* wall shear rate at the bulge vs time          (key hemodynamic difference)
* axial-plane velocity field  Newtonian | Carreau | difference
* centreline space-time map                     (both + difference)
* **video** — side-by-side animated axial-plane velocity over the cardiac cycle
* (optional) 3-D bulge-conforming VTU time-series per case → ParaView video

Usage (CLI)
-----------
    python examples/AAA/compare_aaa_pulsatile.py \
        outputs/AAA_pulsatile_transfer \
        outputs/AAA_rheology_pulsatile \
        --plot --fields --video --t 2.7
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Reuse the predictor + publication-quality plot helpers from the pipe module.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pipe_flow"))
from predict_pulsatile import (  # noqa: E402
    PulsatilePredictor, _field, _cbar, _save, _CMAP,
)

_COL_A = "#1f6fd6"      # Newtonian (blue)
_COL_B = "#d1495b"      # Carreau / non-Newtonian (warm red)
_DIFF_CMAP = "RdBu_r"


class AAAComparison:
    """Overlay two pulsatile AAA solutions through the bulge (Newtonian vs Carreau)."""

    def __init__(self, dir_a: str, dir_b: str,
                 label_a: str = "Newtonian", label_b: str = "Carreau"):
        self.A = PulsatilePredictor(dir_a)
        self.B = PulsatilePredictor(dir_b)
        self.la, self.lb = label_a, label_b
        self.out_dir = dir_a
        p = self.A.physics
        self.R_vessel = float(p.get("R_vessel", 0.5))
        self.R_AAA    = float(p.get("R_AAA", 1.0))
        self.x0       = float(p.get("x0", -2.0))
        self.L_AAA    = float(p.get("L_AAA", 1.5))
        self.x_lo     = float(p.get("x_lo", -3.5))
        self.L        = float(p.get("L", 7.0))
        self.x_hi     = self.x_lo + self.L
        self.Re       = float(p.get("Re", 40.0))
        self.V_max    = float(p.get("V_max", 2.0))
        self.V_amp    = float(p.get("V_amp", 1.0))
        self.Tper     = float(p.get("T_period", 1.0))
        self.omega    = 2.0 * np.pi / self.Tper
        self.Rmax     = max(self.R_vessel, self.R_AAA)
        self.t_max    = min(self.A.t_max, self.B.t_max)

    # ------------------------------------------------------------------
    def R(self, x):
        """Local AAA cross-section radius R(x) (cos² bulge profile)."""
        half = self.L_AAA / 2.0
        dx   = np.abs(np.asarray(x, dtype=float) - self.x0)
        cos2 = np.cos(0.5 * np.pi * np.clip(dx / half, 0.0, 1.0)) ** 2
        return np.where(dx <= half,
                        self.R_vessel + (self.R_AAA - self.R_vessel) * cos2,
                        self.R_vessel)

    def _u_centre(self, pred, ts, x):
        return np.array([pred.predict(float(t), np.array([[x, 0, 0]]))[0, 0]
                         for t in ts])

    def _u_radial(self, pred, t, x, r_arr):
        xyz = np.stack([np.full_like(r_arr, x), r_arr, np.zeros_like(r_arr)],
                       axis=1).astype(np.float32)
        return pred.predict(float(t), xyz)[:, 0]

    # ------------------------------------------------------------------
    # 1) centreline velocity vs time at the bulge centre
    # ------------------------------------------------------------------
    def centreline_timeseries(self) -> str:
        ts = np.linspace(0.0, self.t_max, 400, endpoint=False)
        ua = self._u_centre(self.A, ts, self.x0)
        ub = self._u_centre(self.B, ts, self.x0)
        peak = self.V_max + self.V_amp * np.sin(self.omega * ts)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, peak, "--", color="#888888", lw=1.5, label="Inlet peak forcing")
        ax.plot(ts, ua, "-", color=_COL_A, lw=2.2, label=self.la)
        ax.plot(ts, ub, "-", color=_COL_B, lw=2.2, label=self.lb)
        ax.set_xlabel("t")
        ax.set_ylabel("u  (bulge centreline)")
        ax.set_title(f"AAA bulge centreline velocity — {self.la} vs {self.lb}   "
                     f"(Re = {self.Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_aaa_centreline.png"))

    # ------------------------------------------------------------------
    # 2) radial profiles at the bulge centre
    # ------------------------------------------------------------------
    def radial_profiles(self, n_snaps: int = 4) -> str:
        r_arr = np.linspace(0.0, self.R_AAA, 120, dtype=np.float32)
        snaps = (np.linspace(0.0, self.t_max, n_snaps, endpoint=False)
                 + 0.25 * self.A.dT)
        fig, axes = plt.subplots(1, n_snaps, figsize=(3.8 * n_snaps, 4),
                                 sharey=True)
        for ax, tt in zip(np.atleast_1d(axes), snaps):
            ua = self._u_radial(self.A, tt, self.x0, r_arr)
            ub = self._u_radial(self.B, tt, self.x0, r_arr)
            ax.plot(r_arr, ua, "-", color=_COL_A, lw=2.2, label=self.la)
            ax.plot(r_arr, ub, "-", color=_COL_B, lw=2.2, label=self.lb)
            ax.axvline(self.R_vessel, color="0.7", ls=":", lw=1.0)
            ax.set_title(f"t = {tt:.2f}")
            ax.set_xlabel("r")
            ax.grid(True, alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if tt == snaps[0]:
                ax.set_ylabel("u  (axial)")
                ax.legend(loc="lower left")
        fig.suptitle(f"Radial profiles at bulge centre x = {self.x0:g}   "
                     f"({self.la} vs {self.lb},  Re = {self.Re:g})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_aaa_radial_profiles.png"))

    # ------------------------------------------------------------------
    # 3) wall shear rate at the bulge vs time
    # ------------------------------------------------------------------
    def wall_shear_rate(self) -> str:
        ts    = np.linspace(0.0, self.t_max, 240, endpoint=False)
        r_arr = np.linspace(0.0, self.R_AAA, 120, dtype=np.float32)

        def gw(pred):
            out = np.empty(len(ts))
            for i, t in enumerate(ts):
                u = self._u_radial(pred, t, self.x0, r_arr)
                out[i] = abs(np.polyfit(r_arr[-6:], u[-6:], 1)[0])
            return out

        wa, wb = gw(self.A), gw(self.B)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, wa, "-", color=_COL_A, lw=2.2, label=self.la)
        ax.plot(ts, wb, "-", color=_COL_B, lw=2.2, label=self.lb)
        ax.set_xlabel("t")
        ax.set_ylabel(r"wall shear rate  $|\partial u/\partial r|$ at bulge")
        ax.set_title(f"AAA wall shear rate — {self.la} vs {self.lb}   "
                     f"(Re = {self.Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_aaa_wallshear.png"))

    # ------------------------------------------------------------------
    # 4) axial-plane velocity field:  A | B | (B − A)
    # ------------------------------------------------------------------
    def _axial_grid(self, Nx=300, Ny=120):
        xg = np.linspace(self.x_lo, self.x_hi, Nx)
        yg = np.linspace(-self.Rmax, self.Rmax, Ny)
        XX, YY = np.meshgrid(xg, yg)
        base = np.stack([XX.ravel(), YY.ravel(),
                         np.zeros(XX.size, np.float32)], axis=1).astype(np.float32)
        mask = np.abs(YY) <= self.R(XX)
        return xg, yg, XX, YY, base, mask

    def _walls(self, ax, xg):
        ax.plot(xg, self.R(xg),  "k-", lw=1.5)
        ax.plot(xg, -self.R(xg), "k-", lw=1.5)

    def axial_plane(self, t_field: float) -> str:
        xg, yg, XX, YY, base, mask = self._axial_grid()
        ua = np.where(mask, self.A.predict(float(t_field), base)[:, 0].reshape(YY.shape), np.nan)
        ub = np.where(mask, self.B.predict(float(t_field), base)[:, 0].reshape(YY.shape), np.nan)
        diff = ub - ua
        vmin = float(np.nanmin([np.nanmin(ua), np.nanmin(ub)]))
        vmax = float(np.nanmax([np.nanmax(ua), np.nanmax(ub)]))
        levs = np.linspace(vmin, vmax, 120)
        dmax = float(np.nanmax(np.abs(diff))) or 1e-9
        dlev = np.linspace(-dmax, dmax, 120)

        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        for ax, fld, ttl, lv, cm, lab in (
                (axes[0], ua,   self.la,                 levs, _CMAP, "u"),
                (axes[1], ub,   self.lb,                 levs, _CMAP, "u"),
                (axes[2], diff, f"{self.lb} − {self.la}", dlev, _DIFF_CMAP, "Δu")):
            cf = _field(ax, xg, yg, fld, levels=lv, cmap=cm)
            _cbar(fig, cf, ax, lab)
            self._walls(ax, xg)
            ax.set_aspect("equal")
            ax.set_ylim(-self.Rmax, self.Rmax)
            ax.set_ylabel("y")
            ax.set_title(ttl)
        axes[-1].set_xlabel("x  (streamwise)")
        fig.suptitle(f"AAA axial-plane velocity (z = 0) at t = {t_field:.2f}   "
                     f"(Re = {self.Re:g})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.out_dir, f"compare_aaa_axial_t{t_field:.2f}.png"))

    # ------------------------------------------------------------------
    # 5) centreline space-time:  A | B | (B − A)
    # ------------------------------------------------------------------
    def spacetime(self, n_x: int = 240, n_t: int = 240) -> str:
        xs = np.linspace(self.x_lo, self.x_hi, n_x)
        ts = np.linspace(0.0, self.t_max, n_t, endpoint=False)
        base = np.stack([xs, np.zeros(n_x), np.zeros(n_x)], axis=1).astype(np.float32)
        Ua = np.empty((n_t, n_x))
        Ub = np.empty((n_t, n_x))
        for j, t in enumerate(ts):
            Ua[j] = self.A.predict(float(t), base)[:, 0]
            Ub[j] = self.B.predict(float(t), base)[:, 0]
        diff = Ub - Ua
        vmin = float(min(Ua.min(), Ub.min()))
        vmax = float(max(Ua.max(), Ub.max()))
        levs = np.linspace(vmin, vmax, 120)
        dmax = float(np.max(np.abs(diff))) or 1e-9
        dlev = np.linspace(-dmax, dmax, 120)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for ax, fld, ttl, lv, cm, lab in (
                (axes[0], Ua.T,   self.la,                 levs, _CMAP, "u"),
                (axes[1], Ub.T,   self.lb,                 levs, _CMAP, "u"),
                (axes[2], diff.T, f"{self.lb} − {self.la}", dlev, _DIFF_CMAP, "Δu")):
            cf = _field(ax, ts, xs, fld, levels=lv, cmap=cm)
            _cbar(fig, cf, ax, lab)
            ax.axhline(self.x0 - self.L_AAA / 2, color="0.5", ls=":", lw=0.8)
            ax.axhline(self.x0 + self.L_AAA / 2, color="0.5", ls=":", lw=0.8)
            ax.set_xlabel("t")
            ax.set_ylabel("x  (streamwise)")
            ax.set_title(ttl)
        fig.suptitle(f"AAA centreline axial velocity — space–time   "
                     f"(Re = {self.Re:g})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, "compare_aaa_spacetime.png"))

    # ------------------------------------------------------------------
    # 6) VIDEO — side-by-side animated axial-plane velocity (Newtonian / Carreau)
    # ------------------------------------------------------------------
    def save_video(self, n_frames: int = 60, fps: int = 12,
                   filename: str = "compare_aaa_velocity.gif") -> str:
        xg, yg, XX, YY, base, mask = self._axial_grid(Nx=280, Ny=110)
        times = np.linspace(0.0, self.t_max, n_frames, endpoint=False)
        fa, fb = [], []
        for t in times:
            fa.append(np.where(mask, self.A.predict(float(t), base)[:, 0].reshape(YY.shape), np.nan))
            fb.append(np.where(mask, self.B.predict(float(t), base)[:, 0].reshape(YY.shape), np.nan))
        vmin = float(np.nanmin([np.nanmin(f) for f in fa + fb]))
        vmax = float(np.nanmax([np.nanmax(f) for f in fa + fb]))
        levels = np.linspace(vmin, vmax, 100)

        fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=_CMAP)
        fig.colorbar(sm, ax=[axA, axB], fraction=0.025, pad=0.02,
                     label="u  (axial velocity)")

        def _draw(i):
            for ax, frames, lab in ((axA, fa, self.la), (axB, fb, self.lb)):
                ax.clear()
                _field(ax, xg, yg, frames[i], levels=levels)
                self._walls(ax, xg)
                ax.set_aspect("equal")
                ax.set_ylim(-self.Rmax, self.Rmax)
                ax.set_ylabel("y")
                ax.set_title(f"{lab}     t = {times[i]:.3f}")
            axB.set_xlabel("x  (streamwise)")

        anim = FuncAnimation(fig, _draw, frames=len(times), interval=1000.0 / fps)
        out = os.path.join(self.out_dir, filename)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  saved comparison video ({len(times)} frames) → {out}")
        return out

    # ------------------------------------------------------------------
    # 7) 3-D bulge-conforming VTU time-series per case (→ ParaView video)
    # ------------------------------------------------------------------
    def save_vtu_timeseries(self, pred: PulsatilePredictor, name: str,
                            n_frames: int = 60, n_axial: int = 120,
                            n_radial: int = 10, n_theta: int = 22) -> str:
        from underPINN.utils.vtk_io import save_vtu_points, save_pvd

        xg = np.linspace(self.x_lo, self.x_hi, n_axial)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        pts = []
        for x in xg:
            Rl = float(self.R(x))
            pts.append((x, 0.0, 0.0))                       # axis
            for r in np.linspace(0.0, Rl, n_radial)[1:]:
                for t in th:
                    pts.append((x, r * np.cos(t), r * np.sin(t)))
        xyz = np.asarray(pts, dtype=np.float32)             # bulge-conforming cloud

        series_dir = os.path.join(self.out_dir, f"vtu_{name}")
        os.makedirs(series_dir, exist_ok=True)
        times = np.linspace(0.0, self.t_max, n_frames, endpoint=False)
        entries = []
        for k, t in enumerate(times):
            uvwp = pred.predict(float(t), xyz)
            vel  = uvwp[:, :3]
            vtu = save_vtu_points(
                os.path.join(series_dir, f"{name}_{k:04d}.vtu"),
                xyz.astype(np.float64),
                {"velocity": vel, "pressure": uvwp[:, 3],
                 "speed": np.linalg.norm(vel, axis=1)})
            entries.append((float(t), vtu))
        pvd = save_pvd(os.path.join(self.out_dir, f"{name}.pvd"), entries)
        print(f"  saved 3-D VTU series ({name}): {len(times)} frames "
              f"× {xyz.shape[0]} pts → {pvd}")
        return pvd

    # ------------------------------------------------------------------
    def save_all(self, t_field: float | None = None, video: bool = True,
                 vtu: bool = False) -> list[str]:
        out = [self.centreline_timeseries(), self.radial_profiles(),
               self.wall_shear_rate(), self.spacetime()]
        if t_field is not None:
            out.append(self.axial_plane(t_field))
        if video:
            out.append(self.save_video())
        if vtu:
            out.append(self.save_vtu_timeseries(self.A, "newtonian"))
            out.append(self.save_vtu_timeseries(self.B, "carreau"))
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare pulsatile AAA flow — Newtonian vs Carreau.")
    ap.add_argument("dir_a", help="Newtonian AAA output dir "
                                  "(e.g. outputs/AAA_pulsatile_transfer)")
    ap.add_argument("dir_b", help="Carreau AAA output dir "
                                  "(e.g. outputs/AAA_rheology_pulsatile)")
    ap.add_argument("--label-a", default="Newtonian")
    ap.add_argument("--label-b", default="Carreau")
    ap.add_argument("--t", type=float, default=None,
                    help="time for the axial-plane velocity-field comparison")
    ap.add_argument("--plot", action="store_true",
                    help="centreline, radial, wall-shear, space-time figures")
    ap.add_argument("--fields", action="store_true",
                    help="axial-plane velocity field A|B|diff (needs --t)")
    ap.add_argument("--video", action="store_true",
                    help="side-by-side animated velocity video (GIF)")
    ap.add_argument("--vtu", action="store_true",
                    help="3-D bulge VTU time-series per case (ParaView video)")
    ap.add_argument("--all", action="store_true", help="every output")
    args = ap.parse_args()

    cmp = AAAComparison(args.dir_a, args.dir_b,
                        label_a=args.label_a, label_b=args.label_b)
    saved: list[str] = []
    if args.all:
        saved += cmp.save_all(t_field=args.t, video=True, vtu=args.vtu)
    else:
        if args.plot:
            saved += [cmp.centreline_timeseries(), cmp.radial_profiles(),
                      cmp.wall_shear_rate(), cmp.spacetime()]
        if args.fields:
            if args.t is None:
                ap.error("--fields needs --t")
            saved.append(cmp.axial_plane(args.t))
        if args.video:
            saved.append(cmp.save_video())
        if args.vtu:
            saved.append(cmp.save_vtu_timeseries(cmp.A, "newtonian"))
            saved.append(cmp.save_vtu_timeseries(cmp.B, "carreau"))
    if not saved:
        ap.error("nothing to do — pass --plot, --fields (+--t), --video, --vtu, or --all")
    for p in saved:
        print("Saved:", p)
