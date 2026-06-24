"""Post-process a single pulsatile AAA solution — Carreau (non-Newtonian) only.

Loads one windowed time-marching AAA run (e.g. ``AAA_rheology_pulsatile``) and
produces bulge-aware figures, a velocity-field animation and a 3-D VTU series.
The local aneurysm radius R(x) is reconstructed from the saved physics, so all
masks / walls / cross-sections follow the bulge.  Works for any AAA pulsatile
output dir; here it targets the shear-thinning Carreau case.

Produces (into ``out_dir``):
  predict_aaa_centreline.png      bulge-centreline velocity vs time
  predict_aaa_radial_profiles.png radial profiles at the bulge centre (per phase)
  predict_aaa_wallshear.png       wall shear rate at the bulge vs time
  predict_aaa_axial_t{t}.png      axial-plane velocity + streamlines at time t
  predict_aaa_spacetime.png       centreline u(x, t) space-time map
  predict_aaa_velocity.gif        animated axial-plane velocity over the cycle
  aaa_pulsatile.pvd               3-D bulge VTU time-series (ParaView → video)

Usage (CLI)
-----------
    python examples/AAA_rheology/predict_aaa_pulsatile.py \
        outputs/AAA_rheology_pulsatile --plot --field --t 2.7 --video --vtu
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

_COL = "#1f6fd6"


class AAABulgePredictor:
    """Single-case post-processor for a pulsatile AAA (bulge) solution."""

    def __init__(self, out_dir: str, label: str = "Carreau"):
        self.pred  = PulsatilePredictor(out_dir)
        self.out_dir = out_dir
        self.label = label
        p = self.pred.physics
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
        self.t_max    = self.pred.t_max

    # ------------------------------------------------------------------
    def R(self, x):
        """Local AAA cross-section radius R(x) (cos² bulge profile)."""
        half = self.L_AAA / 2.0
        dx   = np.abs(np.asarray(x, dtype=float) - self.x0)
        cos2 = np.cos(0.5 * np.pi * np.clip(dx / half, 0.0, 1.0)) ** 2
        return np.where(dx <= half,
                        self.R_vessel + (self.R_AAA - self.R_vessel) * cos2,
                        self.R_vessel)

    def _walls(self, ax, xg):
        ax.plot(xg, self.R(xg),  "k-", lw=1.5)
        ax.plot(xg, -self.R(xg), "k-", lw=1.5)

    def _u_radial(self, t, x, r_arr):
        xyz = np.stack([np.full_like(r_arr, x), r_arr, np.zeros_like(r_arr)],
                       axis=1).astype(np.float32)
        return self.pred.predict(float(t), xyz)[:, 0]

    # ------------------------------------------------------------------
    # 1) centreline velocity vs time at the bulge centre
    # ------------------------------------------------------------------
    def centreline_timeseries(self) -> str:
        ts = np.linspace(0.0, self.t_max, 400, endpoint=False)
        uc = np.array([self.pred.predict(float(t), np.array([[self.x0, 0, 0]]))[0, 0]
                       for t in ts])
        peak = self.V_max + self.V_amp * np.sin(self.omega * ts)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, peak, "--", color="#888888", lw=1.5, label="Inlet peak forcing")
        ax.plot(ts, uc, "-", color=_COL, lw=2.2, label=f"{self.label} bulge u")
        ax.set_xlabel("t")
        ax.set_ylabel("u  (bulge centreline)")
        ax.set_title(f"AAA bulge centreline velocity — {self.label}   "
                     f"(Re = {self.Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, "predict_aaa_centreline.png"))

    # ------------------------------------------------------------------
    # 2) radial profiles at the bulge centre
    # ------------------------------------------------------------------
    def radial_profiles(self, n_snaps: int = 4) -> str:
        r_arr = np.linspace(0.0, self.R_AAA, 120, dtype=np.float32)
        snaps = (np.linspace(0.0, self.t_max, n_snaps, endpoint=False)
                 + 0.25 * self.pred.dT)
        fig, axes = plt.subplots(1, n_snaps, figsize=(3.8 * n_snaps, 4),
                                 sharey=True)
        for ax, tt in zip(np.atleast_1d(axes), snaps):
            u = self._u_radial(tt, self.x0, r_arr)
            ax.fill_between(r_arr, u, color=_COL, alpha=0.10)
            ax.plot(r_arr, u, "-", color=_COL, lw=2.2)
            ax.axvline(self.R_vessel, color="0.7", ls=":", lw=1.0)
            ax.set_title(f"t = {tt:.2f}")
            ax.set_xlabel("r")
            ax.grid(True, alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if tt == snaps[0]:
                ax.set_ylabel("u  (axial)")
        fig.suptitle(f"Radial velocity profiles at bulge centre x = {self.x0:g}   "
                     f"({self.label},  Re = {self.Re:g})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "predict_aaa_radial_profiles.png"))

    # ------------------------------------------------------------------
    # 3) wall shear rate at the bulge vs time
    # ------------------------------------------------------------------
    def wall_shear_rate(self) -> str:
        ts    = np.linspace(0.0, self.t_max, 240, endpoint=False)
        r_arr = np.linspace(0.0, self.R_AAA, 120, dtype=np.float32)
        w = np.empty(len(ts))
        for i, t in enumerate(ts):
            u = self._u_radial(t, self.x0, r_arr)
            w[i] = abs(np.polyfit(r_arr[-6:], u[-6:], 1)[0])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, w, "-", color=_COL, lw=2.2)
        ax.set_xlabel("t")
        ax.set_ylabel(r"wall shear rate  $|\partial u/\partial r|$ at bulge")
        ax.set_title(f"AAA wall shear rate — {self.label}   (Re = {self.Re:g})")
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, "predict_aaa_wallshear.png"))

    # ------------------------------------------------------------------
    # 4) axial-plane velocity + streamlines at a time
    # ------------------------------------------------------------------
    def _axial_grid(self, Nx=300, Ny=120):
        xg = np.linspace(self.x_lo, self.x_hi, Nx)
        yg = np.linspace(-self.Rmax, self.Rmax, Ny)
        XX, YY = np.meshgrid(xg, yg)
        base = np.stack([XX.ravel(), YY.ravel(),
                         np.zeros(XX.size, np.float32)], axis=1).astype(np.float32)
        mask = np.abs(YY) <= self.R(XX)
        return xg, yg, YY, base, mask

    def axial_plane(self, t_field: float) -> str:
        xg, yg, YY, base, mask = self._axial_grid()
        pred = self.pred.predict(float(t_field), base)
        U = np.where(mask, pred[:, 0].reshape(YY.shape), np.nan)
        V = np.where(mask, pred[:, 1].reshape(YY.shape), np.nan)
        fig, ax = plt.subplots(figsize=(13, 4.2))
        cf = _field(ax, xg, yg, U, levels=120)
        _cbar(fig, cf, ax, "u  (axial velocity)")
        ax.streamplot(xg, yg, np.nan_to_num(U), np.nan_to_num(V),
                      density=(2.4, 0.7), color=(0, 0, 0, 0.5),
                      linewidth=0.6, arrowsize=0.8)
        self._walls(ax, xg)
        ax.set_aspect("equal")
        ax.set_ylim(-self.Rmax, self.Rmax)
        ax.set_xlabel("x  (streamwise)")
        ax.set_ylabel("y")
        ax.set_title(f"AAA axial-plane velocity & streamlines — {self.label}   "
                     f"(t = {t_field:.2f},  Re = {self.Re:g})",
                     fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.out_dir, f"predict_aaa_axial_t{t_field:.2f}.png"))

    # ------------------------------------------------------------------
    # 5) centreline space-time
    # ------------------------------------------------------------------
    def spacetime(self, n_x: int = 240, n_t: int = 240) -> str:
        xs = np.linspace(self.x_lo, self.x_hi, n_x)
        ts = np.linspace(0.0, self.t_max, n_t, endpoint=False)
        base = np.stack([xs, np.zeros(n_x), np.zeros(n_x)], axis=1).astype(np.float32)
        U = np.empty((n_t, n_x))
        for j, t in enumerate(ts):
            U[j] = self.pred.predict(float(t), base)[:, 0]
        fig, ax = plt.subplots(figsize=(11, 4))
        cf = _field(ax, ts, xs, U.T, levels=120)
        _cbar(fig, cf, ax, "u  (centreline)")
        ax.axhline(self.x0 - self.L_AAA / 2, color="0.6", ls=":", lw=0.9)
        ax.axhline(self.x0 + self.L_AAA / 2, color="0.6", ls=":", lw=0.9)
        ax.set_xlabel("t")
        ax.set_ylabel("x  (streamwise)")
        ax.set_title(f"AAA centreline axial velocity — space–time ({self.label}, "
                     f"Re = {self.Re:g})")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, "predict_aaa_spacetime.png"))

    # ------------------------------------------------------------------
    # 6) VIDEO — animated axial-plane velocity over the cycle
    # ------------------------------------------------------------------
    def save_video(self, n_frames: int = 60, fps: int = 12,
                   filename: str = "predict_aaa_velocity.gif") -> str:
        xg, yg, YY, base, mask = self._axial_grid(Nx=280, Ny=110)
        times = np.linspace(0.0, self.t_max, n_frames, endpoint=False)
        frames = [np.where(mask, self.pred.predict(float(t), base)[:, 0].reshape(YY.shape), np.nan)
                  for t in times]
        vmin = float(np.nanmin([np.nanmin(f) for f in frames]))
        vmax = float(np.nanmax([np.nanmax(f) for f in frames]))
        levels = np.linspace(vmin, vmax, 100)

        fig, ax = plt.subplots(figsize=(13, 3.8))
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=_CMAP)
        fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="u  (axial velocity)")

        def _draw(i):
            ax.clear()
            _field(ax, xg, yg, frames[i], levels=levels)
            self._walls(ax, xg)
            ax.set_aspect("equal")
            ax.set_ylim(-self.Rmax, self.Rmax)
            ax.set_xlabel("x  (streamwise)")
            ax.set_ylabel("y")
            ax.set_title(f"AAA {self.label} — u(x, y)  z = 0     t = {times[i]:.3f}     "
                         f"(Re = {self.Re:g})")

        anim = FuncAnimation(fig, _draw, frames=len(times), interval=1000.0 / fps)
        out = os.path.join(self.out_dir, filename)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  saved video ({len(times)} frames) → {out}")
        return out

    # ------------------------------------------------------------------
    # 7) 3-D bulge-conforming VTU time-series (→ ParaView video)
    # ------------------------------------------------------------------
    def save_vtu_timeseries(self, n_frames: int = 60, n_axial: int = 120,
                            n_radial: int = 10, n_theta: int = 22,
                            name: str = "aaa_pulsatile") -> str:
        from underPINN.utils.vtk_io import save_vtu_points, save_pvd

        xg = np.linspace(self.x_lo, self.x_hi, n_axial)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        pts = []
        for x in xg:
            Rl = float(self.R(x))
            pts.append((x, 0.0, 0.0))
            for r in np.linspace(0.0, Rl, n_radial)[1:]:
                for t in th:
                    pts.append((x, r * np.cos(t), r * np.sin(t)))
        xyz = np.asarray(pts, dtype=np.float32)

        series_dir = os.path.join(self.out_dir, f"vtu_{name}")
        os.makedirs(series_dir, exist_ok=True)
        times = np.linspace(0.0, self.t_max, n_frames, endpoint=False)
        entries = []
        for k, t in enumerate(times):
            uvwp = self.pred.predict(float(t), xyz)
            vel  = uvwp[:, :3]
            vtu = save_vtu_points(
                os.path.join(series_dir, f"{name}_{k:04d}.vtu"),
                xyz.astype(np.float64),
                {"velocity": vel, "pressure": uvwp[:, 3],
                 "speed": np.linalg.norm(vel, axis=1)})
            entries.append((float(t), vtu))
        pvd = save_pvd(os.path.join(self.out_dir, f"{name}.pvd"), entries)
        print(f"  saved 3-D VTU series: {len(times)} frames × {xyz.shape[0]} pts "
              f"→ {pvd}")
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
            out.append(self.save_vtu_timeseries())
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Post-process a single pulsatile AAA (Carreau) solution.")
    ap.add_argument("out_dir", help="AAA pulsatile output dir "
                                    "(e.g. outputs/AAA_rheology_pulsatile)")
    ap.add_argument("--label", default="Carreau")
    ap.add_argument("--t", type=float, default=None,
                    help="time for the axial-plane velocity field")
    ap.add_argument("--plot", action="store_true",
                    help="centreline, radial, wall-shear, space-time figures")
    ap.add_argument("--field", action="store_true",
                    help="axial-plane velocity + streamlines (needs --t)")
    ap.add_argument("--video", action="store_true",
                    help="animated axial-plane velocity (GIF)")
    ap.add_argument("--vtu", action="store_true",
                    help="3-D bulge VTU time-series (ParaView video)")
    ap.add_argument("--all", action="store_true", help="every output")
    args = ap.parse_args()

    aaa = AAABulgePredictor(args.out_dir, label=args.label)
    saved: list[str] = []
    if args.all:
        saved += aaa.save_all(t_field=args.t, video=True, vtu=args.vtu)
    else:
        if args.plot:
            saved += [aaa.centreline_timeseries(), aaa.radial_profiles(),
                      aaa.wall_shear_rate(), aaa.spacetime()]
        if args.field:
            if args.t is None:
                ap.error("--field needs --t")
            saved.append(aaa.axial_plane(args.t))
        if args.video:
            saved.append(aaa.save_video())
        if args.vtu:
            saved.append(aaa.save_vtu_timeseries())
    if not saved:
        ap.error("nothing to do — pass --plot, --field (+--t), --video, --vtu, or --all")
    for p in saved:
        print("Saved:", p)
