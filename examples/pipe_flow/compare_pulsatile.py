"""Compare two trained pulsatile pipe-flow solutions — Newtonian vs Carreau.

Both the Newtonian (``pipe_flow_pulsatile_transfer``) and the non-Newtonian
(``pipe_flow_rheology_pulsatile``) cases save the same windowed time-marching
format, so :class:`PulsatilePredictor` loads either one.  This module overlays /
side-by-sides the two solutions to make the shear-thinning effect visible:

* centreline velocity vs time            (both curves + inlet forcing)
* radial velocity profiles at 4 phases   (both overlaid — parabola vs blunt)
* wall shear rate vs time                (Carreau ⇒ steeper near-wall gradient)
* cross-section at a chosen time         (Newtonian | Carreau | difference)
* streamwise plane at a chosen time      (Newtonian over Carreau)
* centreline space-time map              (Newtonian | Carreau | difference)

Usage (CLI)
-----------
    python examples/pipe_flow/compare_pulsatile.py \
        outputs/pipe_flow_pulsatile_transfer \
        outputs/pipe_flow_rheology_pulsatile \
        --plot --fields --spacetime --t 2.7

Usage (library)
---------------
    from examples.pipe_flow.compare_pulsatile import PulsatileComparison
    cmp = PulsatileComparison("outputs/pipe_flow_pulsatile_transfer",
                              "outputs/pipe_flow_rheology_pulsatile")
    cmp.save_all(t_field=2.7)
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.postprocess import PulsatilePredictor, _field, _cbar, _save, _CMAP

_COL_A = "#1f6fd6"      # Newtonian  (blue)
_COL_B = "#d1495b"      # Carreau / non-Newtonian (warm red)
_DIFF_CMAP = "RdBu_r"   # divergent map for (B − A)


class PulsatileComparison:
    """Overlay two pulsatile pipe solutions for a Newtonian-vs-Carreau study."""

    def __init__(self, dir_a: str, dir_b: str,
                 label_a: str = "Newtonian", label_b: str = "Carreau"):
        self.A = PulsatilePredictor(dir_a)
        self.B = PulsatilePredictor(dir_b)
        self.la, self.lb = label_a, label_b
        self.out_dir = dir_a            # comparison PNGs land next to case A
        # Shared domain from case A (the two cases are set up on the same domain).
        p = self.A.physics
        self.R    = float(p.get("R", 0.5))
        self.Vmax = float(p.get("V_max", 2.0))
        self.Vamp = float(p.get("V_amp", 1.0))
        self.Tper = float(p.get("T_period", 1.0))
        self.x_lo = float(p.get("x_lo", -3.5))
        self.L    = float(p.get("L", 7.0))
        self.Re   = float(p.get("Re", 40.0))
        self.x_mid = self.x_lo + 0.5 * self.L
        self.omega = 2.0 * np.pi / self.Tper
        self.t_max = min(self.A.t_max, self.B.t_max)   # common horizon
        rb = float(self.B.physics.get("R", self.R))
        if abs(rb - self.R) > 1e-6:
            print(f"  [compare] WARNING: radii differ (A={self.R}, B={rb}); "
                  f"using A's domain for the shared axes.")

    # ------------------------------------------------------------------
    def _u_centreline(self, pred, ts):
        return np.array([pred.predict(float(t), np.array([[self.x_mid, 0, 0]]))[0, 0]
                         for t in ts])

    def _u_radial(self, pred, t, r_arr):
        xyz = np.stack([np.full_like(r_arr, self.x_mid), r_arr,
                        np.zeros_like(r_arr)], axis=1).astype(np.float32)
        return pred.predict(float(t), xyz)[:, 0]

    def _wall_shear_rate(self, pred, ts, r_arr):
        """|du/dr| at r = R from a robust last-6-point linear fit."""
        out = np.empty(len(ts))
        for i, t in enumerate(ts):
            u = self._u_radial(pred, t, r_arr)
            out[i] = abs(np.polyfit(r_arr[-6:], u[-6:], 1)[0])
        return out

    # ------------------------------------------------------------------
    # 1) centreline velocity vs time
    # ------------------------------------------------------------------
    def centreline_timeseries(self) -> str:
        ts   = np.linspace(0.0, self.t_max, 400, endpoint=False)
        uc_a = self._u_centreline(self.A, ts)
        uc_b = self._u_centreline(self.B, ts)
        peak = self.Vmax + self.Vamp * np.sin(self.omega * ts)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, peak, "--", color="#888888", lw=1.5, label="Inlet peak forcing")
        ax.plot(ts, uc_a, "-", color=_COL_A, lw=2.2, label=self.la)
        ax.plot(ts, uc_b, "-", color=_COL_B, lw=2.2, label=self.lb)
        ax.set_xlabel("t")
        ax.set_ylabel("u  (centreline)")
        ax.set_title(f"Centreline velocity — {self.la} vs {self.lb}   "
                     f"(Re = {self.Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_centreline_timeseries.png"))

    # ------------------------------------------------------------------
    # 2) radial profiles at 4 phase snapshots
    # ------------------------------------------------------------------
    def radial_profiles(self, n_snaps: int = 4) -> str:
        r_arr = np.linspace(0.0, self.R, 120, dtype=np.float32)
        snaps = (np.linspace(0.0, self.t_max, n_snaps, endpoint=False)
                 + 0.25 * self.A.dT)
        fig, axes = plt.subplots(1, n_snaps, figsize=(3.8 * n_snaps, 4),
                                 sharey=True)
        for ax, tt in zip(np.atleast_1d(axes), snaps):
            ua = self._u_radial(self.A, tt, r_arr)
            ub = self._u_radial(self.B, tt, r_arr)
            ax.plot(r_arr, ua, "-", color=_COL_A, lw=2.2, label=self.la)
            ax.plot(r_arr, ub, "-", color=_COL_B, lw=2.2, label=self.lb)
            ax.set_title(f"t = {tt:.2f}")
            ax.set_xlabel("r")
            ax.grid(True, alpha=0.25)
            ax.margins(x=0.02)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if tt == snaps[0]:
                ax.set_ylabel("u  (axial)")
                ax.legend(loc="lower left")
        fig.suptitle(f"Radial velocity profiles at x = {self.x_mid:.1f}   "
                     f"({self.la} vs {self.lb},  Re = {self.Re:g})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_radial_profiles.png"))

    # ------------------------------------------------------------------
    # 3) wall shear rate vs time
    # ------------------------------------------------------------------
    def wall_shear_rate(self) -> str:
        ts    = np.linspace(0.0, self.t_max, 240, endpoint=False)
        r_arr = np.linspace(0.0, self.R, 120, dtype=np.float32)
        wa = self._wall_shear_rate(self.A, ts, r_arr)
        wb = self._wall_shear_rate(self.B, ts, r_arr)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts, wa, "-", color=_COL_A, lw=2.2, label=self.la)
        ax.plot(ts, wb, "-", color=_COL_B, lw=2.2, label=self.lb)
        ax.set_xlabel("t")
        ax.set_ylabel(r"wall shear rate  $|\partial u/\partial r|_{r=R}$")
        ax.set_title(f"Wall shear rate — {self.la} vs {self.lb}   "
                     f"(Re = {self.Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir,
                                       "compare_wallshear_timeseries.png"))

    # ------------------------------------------------------------------
    # 4) cross-section at a time:  A | B | (B − A)
    # ------------------------------------------------------------------
    def crosssection(self, t_field: float) -> str:
        N  = 160
        yy = np.linspace(-self.R, self.R, N, dtype=np.float32)
        YY, ZZ = np.meshgrid(yy, yy)
        xyz = np.stack([np.full(N * N, self.x_mid, np.float32),
                        YY.ravel(), ZZ.ravel()], axis=1)
        mask = YY ** 2 + ZZ ** 2 > self.R ** 2
        ua = np.where(mask, np.nan,
                      self.A.predict(float(t_field), xyz)[:, 0].reshape(N, N))
        ub = np.where(mask, np.nan,
                      self.B.predict(float(t_field), xyz)[:, 0].reshape(N, N))
        diff = ub - ua

        vmin = float(np.nanmin([np.nanmin(ua), np.nanmin(ub)]))
        vmax = float(np.nanmax([np.nanmax(ua), np.nanmax(ub)]))
        levs = np.linspace(vmin, vmax, 120)
        dmax = float(np.nanmax(np.abs(diff))) or 1e-9
        dlev = np.linspace(-dmax, dmax, 120)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, fld, ttl, lv, cm in (
                (axes[0], ua, self.la, levs, _CMAP),
                (axes[1], ub, self.lb, levs, _CMAP),
                (axes[2], diff, f"{self.lb} − {self.la}", dlev, _DIFF_CMAP)):
            wall = plt.Circle((0, 0), self.R, fill=False, lw=1.5,
                              edgecolor="#222222")
            ax.add_patch(wall)
            cf = _field(ax, yy, yy, fld, levels=lv, cmap=cm)
            try:
                cf.set_clip_path(wall)
            except Exception:
                pass
            _cbar(fig, cf, ax, "u" if cm == _CMAP else "Δu")
            ax.set_aspect("equal")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_title(ttl)
        fig.suptitle(f"Axial velocity cross-section at x = {self.x_mid:.1f}, "
                     f"t = {t_field:.2f}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.out_dir, f"compare_crosssection_t{t_field:.2f}.png"))

    # ------------------------------------------------------------------
    # 5) streamwise plane at a time: Newtonian over Carreau
    # ------------------------------------------------------------------
    def axial_plane(self, t_field: float) -> str:
        x_hi = self.x_lo + self.L
        Nx, Ny = 300, 70
        xg = np.linspace(self.x_lo, x_hi, Nx)
        yg = np.linspace(-self.R, self.R, Ny)
        XX, YY = np.meshgrid(xg, yg)
        base = np.stack([XX.ravel(), YY.ravel(),
                         np.zeros(XX.size, np.float32)], axis=1).astype(np.float32)
        Ua = self.A.predict(float(t_field), base)[:, 0].reshape(Ny, Nx)
        Ub = self.B.predict(float(t_field), base)[:, 0].reshape(Ny, Nx)
        vmin = float(min(Ua.min(), Ub.min()))
        vmax = float(max(Ua.max(), Ub.max()))
        levs = np.linspace(vmin, vmax, 120)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
        for ax, U, ttl in ((axes[0], Ua, self.la), (axes[1], Ub, self.lb)):
            cf = _field(ax, xg, yg, U, levels=levs)
            _cbar(fig, cf, ax, "u  (streamwise)")
            ax.axhline(self.R,  color="#222222", lw=1.5)
            ax.axhline(-self.R, color="#222222", lw=1.5)
            ax.set_ylim(-self.R, self.R)
            ax.set_aspect("equal")
            ax.set_ylabel("y")
            ax.set_title(ttl)
        axes[-1].set_xlabel("x  (streamwise)")
        fig.suptitle(f"Streamwise plane z = 0 at t = {t_field:.2f}   "
                     f"(Re = {self.Re:g})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.out_dir, f"compare_axial_plane_t{t_field:.2f}.png"))

    # ------------------------------------------------------------------
    # 6) centreline space-time:  A | B | (B − A)
    # ------------------------------------------------------------------
    def spacetime(self, n_x: int = 240, n_t: int = 240) -> str:
        x_hi = self.x_lo + self.L
        xs = np.linspace(self.x_lo, x_hi, n_x)
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
                (axes[0], Ua.T,   self.la,                levs, _CMAP, "u"),
                (axes[1], Ub.T,   self.lb,                levs, _CMAP, "u"),
                (axes[2], diff.T, f"{self.lb} − {self.la}", dlev, _DIFF_CMAP, "Δu")):
            cf = _field(ax, ts, xs, fld, levels=lv, cmap=cm)
            _cbar(fig, cf, ax, lab)
            ax.set_xlabel("t")
            ax.set_ylabel("x  (streamwise)")
            ax.set_title(ttl)
        fig.suptitle(f"Centreline axial velocity — space–time   "
                     f"(Re = {self.Re:g})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, "compare_spacetime.png"))

    # ------------------------------------------------------------------
    def save_all(self, t_field: float | None = None) -> list[str]:
        out = [self.centreline_timeseries(),
               self.radial_profiles(),
               self.wall_shear_rate(),
               self.spacetime()]
        if t_field is not None:
            out.append(self.crosssection(t_field))
            out.append(self.axial_plane(t_field))
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare Newtonian vs non-Newtonian pulsatile pipe flow.")
    ap.add_argument("dir_a", help="Newtonian output dir "
                                  "(e.g. outputs/pipe_flow_pulsatile_transfer)")
    ap.add_argument("dir_b", help="non-Newtonian output dir "
                                  "(e.g. outputs/pipe_flow_rheology_pulsatile)")
    ap.add_argument("--label-a", default="Newtonian")
    ap.add_argument("--label-b", default="Carreau")
    ap.add_argument("--t", type=float, default=None,
                    help="time for the cross-section / streamwise-plane fields")
    ap.add_argument("--plot", action="store_true",
                    help="centreline, radial profiles, wall shear, space-time")
    ap.add_argument("--fields", action="store_true",
                    help="cross-section + streamwise plane (needs --t)")
    ap.add_argument("--spacetime", action="store_true",
                    help="centreline space-time comparison only")
    ap.add_argument("--all", action="store_true",
                    help="every comparison figure (uses --t for the fields)")
    args = ap.parse_args()

    cmp = PulsatileComparison(args.dir_a, args.dir_b,
                              label_a=args.label_a, label_b=args.label_b)
    saved: list[str] = []

    if args.all:
        saved += cmp.save_all(t_field=args.t)
    else:
        if args.plot:
            saved += [cmp.centreline_timeseries(), cmp.radial_profiles(),
                      cmp.wall_shear_rate()]
        if args.spacetime:
            saved.append(cmp.spacetime())
        if args.fields:
            if args.t is None:
                ap.error("--fields needs --t (the snapshot time)")
            saved += [cmp.crosssection(args.t), cmp.axial_plane(args.t)]

    if not saved:
        ap.error("nothing to do — pass --plot, --spacetime, --fields (+--t), or --all")
    for p in saved:
        print("Saved:", p)
