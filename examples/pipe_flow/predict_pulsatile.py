"""Pulsatile-pipe post-processing (figures, animation, 3-D VTU) + CLI.

The geometry-agnostic core (window loading + :meth:`predict`) and the shared
plot helpers now live in the library — ``underPINN.postprocess`` — so they can
be reused without ``sys.path`` hacks.  This module adds the **pipe-specific**
views (cylinder cross-sections, radial profiles along y, the z=0 streamwise
plane, a cylindrical VTU cloud) as a thin subclass, plus the CLI.

    from underPINN.postprocess import PulsatilePredictor    # generic core
    from examples.pipe_flow.predict_pulsatile import PipePulsatilePredictor

Usage (CLI)
-----------
    python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer \
        --plot --t 2.7 --spacetime --animate --vtu
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.postprocess import PulsatilePredictor, _field, _cbar, _save, _CMAP  # noqa: F401


class PipePulsatilePredictor(PulsatilePredictor):
    """Pulsatile-pipe figures / animation / VTU on top of the generic core."""

    def save_plots(self, t_field: float | None = None) -> list[str]:
        """Save figures (PNG) reconstructed from the window checkpoints.

        Produces:
          * centreline velocity vs time over the covered horizon
          * radial profiles at four phase snapshots
          * (optional) an axial-velocity cross-section at time *t_field*
        Returns the list of file paths written.
        """
        R        = float(self.physics.get("R", 0.5))
        V_max    = float(self.physics.get("V_max", 2.0))
        V_amp    = float(self.physics.get("V_amp", 1.0))
        T_period = float(self.physics.get("T_period", 1.0))
        x_lo     = float(self.physics.get("x_lo", -3.5))
        L        = float(self.physics.get("L", 7.0))
        Re       = float(self.physics.get("Re", 40.0))
        x_mid    = x_lo + 0.5 * L
        omega    = 2.0 * np.pi / T_period
        t_hi     = self.t_max
        saved: list[str] = []

        # 1) centreline velocity over the horizon vs inlet forcing
        ts   = np.linspace(0.0, t_hi, 400, endpoint=False)
        uc   = np.array([self.predict(float(t), np.array([[x_mid, 0.0, 0.0]]))[0, 0]
                         for t in ts])
        peak = V_max + V_amp * np.sin(omega * ts)
        fig, ax = plt.subplots(figsize=(10, 4))
        for k in self.available[1:]:
            ax.axvline(k * self.stride, color="0.7", lw=0.6, ls=":", alpha=0.7,
                       zorder=1)
        ax.plot(ts, peak, "--", color="#444444", lw=1.6,
                label="Inlet peak forcing", zorder=2)
        ax.plot(ts, uc, "-", color="#1f6fd6", lw=2.2,
                label=f"PINN centreline u @ x={x_mid:.1f}", zorder=3)
        ax.set_xlabel("t")
        ax.set_ylabel("u  (centreline)")
        ax.set_title(f"Pulsatile pipe — centreline velocity   (Re = {Re:g})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, os.path.join(
            self.out_dir, "predict_centreline_timeseries.png")))

        # 2) radial profiles at four snapshots
        Nr    = 80
        r_arr = np.linspace(0.0, R, Nr, dtype=np.float32)
        xyz_r = np.stack([np.full(Nr, x_mid, np.float32), r_arr,
                          np.zeros(Nr, np.float32)], axis=1)
        snaps = np.linspace(0.0, t_hi, 4, endpoint=False) + 0.25 * self.dT
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
        for ax, tt in zip(np.atleast_1d(axes), snaps):
            u_pred = self.predict(float(tt), xyz_r)[:, 0]
            pk     = V_max + V_amp * np.sin(omega * tt)
            u_qs   = pk * (1.0 - r_arr ** 2 / R ** 2)
            ax.plot(r_arr, u_qs, "--", color="#888888", lw=1.5,
                    label="Quasi-steady")
            ax.fill_between(r_arr, u_pred, color="#1f6fd6", alpha=0.10, zorder=1)
            ax.plot(r_arr, u_pred, "-", color="#1f6fd6", lw=2.2, label="PINN")
            ax.set_title(f"t = {tt:.2f}")
            ax.set_xlabel("r")
            ax.grid(True, alpha=0.25)
            ax.margins(x=0.02)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if tt == snaps[0]:
                ax.set_ylabel("u")
                ax.legend(loc="lower left")
        fig.suptitle(f"Radial velocity profiles at x = {x_mid:.1f}   (Re = {Re:g})",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, os.path.join(
            self.out_dir, "predict_radial_profiles.png")))

        # 3) optional cross-section of axial velocity at a chosen time
        if t_field is not None:
            N = 160
            yy = np.linspace(-R, R, N, dtype=np.float32)
            zz = np.linspace(-R, R, N, dtype=np.float32)
            YY, ZZ = np.meshgrid(yy, zz)
            xyz = np.stack([np.full(N * N, x_mid, np.float32),
                            YY.ravel(), ZZ.ravel()], axis=1)
            u_cs = self.predict(float(t_field), xyz)[:, 0].reshape(N, N)
            u_cs = np.where(YY ** 2 + ZZ ** 2 > R ** 2, np.nan, u_cs)
            fig, ax = plt.subplots(figsize=(5.4, 4.4))
            wall = plt.Circle((0, 0), R, fill=False, lw=1.6, edgecolor="#222222")
            ax.add_patch(wall)
            cf = _field(ax, yy, zz, u_cs, levels=120)
            try:                                  # clip the field exactly to the wall
                cf.set_clip_path(wall)
            except Exception:
                pass
            _cbar(fig, cf, ax, "u  (axial velocity)")
            ax.set_aspect("equal")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_title(f"Axial velocity at x = {x_mid:.1f},  t = {t_field:.2f}")
            fig.tight_layout()
            saved.append(_save(fig, os.path.join(
                self.out_dir, f"predict_crosssection_t{t_field:.2f}.png")))

        # 4) optional streamwise (axial) plane z=0:  u(x, y) + streamlines
        if t_field is not None:
            x_hi = x_lo + L
            Nx, Ny = 300, 70
            xg = np.linspace(x_lo, x_hi, Nx)      # float64 → equal spacing
            yg = np.linspace(-R, R, Ny)
            XX, YY = np.meshgrid(xg, yg)
            xyz = np.stack([XX.ravel(), YY.ravel(),
                            np.zeros(XX.size, np.float32)], axis=1).astype(np.float32)
            pred = self.predict(float(t_field), xyz)
            U = pred[:, 0].reshape(Ny, Nx)        # streamwise velocity
            V = pred[:, 1].reshape(Ny, Nx)        # transverse (y) velocity

            fig, ax = plt.subplots(figsize=(15, 3.6))
            cf = _field(ax, xg, yg, U, levels=120)
            _cbar(fig, cf, ax, "u  (streamwise)")
            # streamlines show the in-plane flow direction
            ax.streamplot(xg, yg, np.nan_to_num(U), np.nan_to_num(V),
                          density=(2.6, 0.6), color=(0, 0, 0, 0.55),
                          linewidth=0.6, arrowsize=0.8)
            # pipe walls
            ax.axhline(R,  color="#222222", lw=1.6)
            ax.axhline(-R, color="#222222", lw=1.6)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal")
            ax.set_xlabel("x  (streamwise)")
            ax.set_ylabel("y")
            ax.set_title(f"Streamwise plane z = 0 — u & streamlines   "
                         f"(t = {t_field:.2f},  Re = {Re:g})")
            fig.tight_layout()
            saved.append(_save(fig, os.path.join(
                self.out_dir, f"predict_axial_plane_t{t_field:.2f}.png")))

        return saved

    # ------------------------------------------------------------------
    def _phys(self) -> dict:
        p = self.physics
        return {
            "R":        float(p.get("R", 0.5)),
            "V_max":    float(p.get("V_max", 2.0)),
            "V_amp":    float(p.get("V_amp", 1.0)),
            "T_period": float(p.get("T_period", 1.0)),
            "x_lo":     float(p.get("x_lo", -3.5)),
            "L":        float(p.get("L", 7.0)),
            "Re":       float(p.get("Re", 40.0)),
        }

    def save_spacetime_centreline(
        self, n_x: int = 260, n_t: int = 260,
        filename: str = "predict_spacetime_centreline.png",
    ) -> str:
        """Static 'all-time' view: centreline axial velocity u(x, t) as a map."""
        ph   = self._phys()
        x_lo, L, Re = ph["x_lo"], ph["L"], ph["Re"]
        x_hi = x_lo + L
        xs   = np.linspace(x_lo, x_hi, n_x)
        ts   = np.linspace(0.0, self.t_max, n_t, endpoint=False)
        base = np.stack([xs, np.zeros(n_x), np.zeros(n_x)], axis=1).astype(np.float32)

        U = np.empty((n_t, n_x), dtype=np.float32)
        for j, t in enumerate(ts):
            U[j] = self.predict(float(t), base)[:, 0]

        fig, ax = plt.subplots(figsize=(11, 4))
        cf = _field(ax, ts, xs, U.T, levels=120)
        _cbar(fig, cf, ax, "u  (centreline, r = 0)")
        ax.set_xlabel("t")
        ax.set_ylabel("x  (streamwise)")
        ax.set_title(f"Centreline axial velocity — space–time   (Re = {Re:g})")
        fig.tight_layout()
        return _save(fig, os.path.join(self.out_dir, filename))

    def save_axial_animation(
        self, n_frames: int = 80, fps: int = 12,
        filename: str = "predict_axial_animation.gif",
    ) -> str:
        """Animate the streamwise plane z=0 (u contour) over the whole horizon."""
        ph   = self._phys()
        R, x_lo, L, Re = ph["R"], ph["x_lo"], ph["L"], ph["Re"]
        x_hi  = x_lo + L
        times = np.linspace(0.0, self.t_max, n_frames, endpoint=False)
        Nx, Ny = 260, 60
        xg = np.linspace(x_lo, x_hi, Nx)
        yg = np.linspace(-R, R, Ny)
        XX, YY = np.meshgrid(xg, yg)
        base = np.stack([XX.ravel(), YY.ravel(),
                         np.zeros(XX.size, np.float32)], axis=1).astype(np.float32)

        # Pre-compute every frame and a shared colour range
        frames = [self.predict(float(t), base)[:, 0].reshape(Ny, Nx) for t in times]
        vmin = float(min(U.min() for U in frames))
        vmax = float(max(U.max() for U in frames))
        levels = np.linspace(vmin, vmax, 120)

        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(14, 3.4))
        cf0 = _field(ax, xg, yg, frames[0], levels=levels)
        _cbar(fig, cf0, ax, "u  (streamwise)")

        def _draw(i):
            ax.clear()
            _field(ax, xg, yg, frames[i], levels=levels)
            ax.axhline(R,  color="#222222", lw=1.4)
            ax.axhline(-R, color="#222222", lw=1.4)
            ax.set_aspect("equal")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(-R, R)
            ax.set_xlabel("x  (streamwise)")
            ax.set_ylabel("y")
            ax.set_title(f"u(x, y)  z = 0     t = {times[i]:.3f}     (Re = {Re:g})")

        anim = FuncAnimation(fig, _draw, frames=len(times), interval=1000.0 / fps)
        out = os.path.join(self.out_dir, filename)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  saved animation ({len(times)} frames) → {out}")
        return out

    # ------------------------------------------------------------------
    def save_vtu_timeseries(
        self, n_frames: int = 60, n_axial: int = 120, n_radial: int = 12,
        n_theta: int = 24, t0: float = 0.0, t1: float | None = None,
        subdir: str = "vtu_series", name: str = "pulsatile_3d",
    ) -> str:
        """Write a 3-D time-dependent VTU series (+ a ``.pvd`` collection).

        The full 3-D velocity field (u, v, w), pressure and speed are sampled on
        a fixed cylindrical point cloud filling the pipe and evaluated at
        ``n_frames`` times.  Open the ``.pvd`` in ParaView (colour by ``speed``,
        glyph ``velocity``) and *File → Save Animation* to export a video.
        """
        from underPINN.utils.vtk_io import save_vtu_points, save_pvd

        ph = self._phys()
        R, x_lo, L = ph["R"], ph["x_lo"], ph["L"]
        x_hi = x_lo + L
        t1   = self.t_max if t1 is None else float(t1)
        times = np.linspace(t0, t1, n_frames, endpoint=False)

        # Fixed cylindrical point cloud (geometry constant; fields vary in time).
        xg = np.linspace(x_lo, x_hi, n_axial, dtype=np.float32)
        rg = np.linspace(0.0, R, n_radial, dtype=np.float32)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)
        yz = [(0.0, 0.0)]                              # axis (r = 0) once
        for r in rg[1:]:
            for t in th:
                yz.append((float(r * np.cos(t)), float(r * np.sin(t))))
        yz = np.asarray(yz, dtype=np.float32)         # (M, 2)
        M  = yz.shape[0]
        xyz = np.empty((n_axial * M, 3), dtype=np.float32)
        for i, x in enumerate(xg):
            xyz[i * M:(i + 1) * M, 0] = x
            xyz[i * M:(i + 1) * M, 1:] = yz

        series_dir = os.path.join(self.out_dir, subdir)
        os.makedirs(series_dir, exist_ok=True)
        entries = []
        for k, t in enumerate(times):
            uvwp  = self.predict(float(t), xyz)       # (N, 4)
            vel   = uvwp[:, :3]
            vtu = save_vtu_points(
                os.path.join(series_dir, f"{name}_{k:04d}.vtu"),
                xyz.astype(np.float64),
                {"velocity": vel, "pressure": uvwp[:, 3],
                 "speed": np.linalg.norm(vel, axis=1)},
            )
            entries.append((float(t), vtu))
        pvd = save_pvd(os.path.join(self.out_dir, f"{name}.pvd"), entries)
        print(f"  saved 3-D VTU series: {len(times)} frames × {xyz.shape[0]} pts "
              f"→ {pvd}")
        return pvd


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Predict pulsatile pipe flow at a given time.")
    ap.add_argument("out_dir", help="training output directory")
    ap.add_argument("--t", type=float, default=None, help="absolute time to evaluate")
    ap.add_argument("--x", type=float, default=0.0, help="x location (default 0)")
    ap.add_argument("--y", type=float, default=0.0, help="y location (default 0 = centreline)")
    ap.add_argument("--z", type=float, default=0.0, help="z location (default 0)")
    ap.add_argument("--plot", action="store_true",
                    help="save figures (centreline, radial, and cross-section if --t given)")
    ap.add_argument("--spacetime", action="store_true",
                    help="save the centreline u(x, t) space-time map (all times)")
    ap.add_argument("--animate", action="store_true",
                    help="save a GIF of the streamwise-plane u contour over all times")
    ap.add_argument("--vtu", action="store_true",
                    help="save a 3-D time-dependent VTU series + .pvd (ParaView → video)")
    ap.add_argument("--frames", type=int, default=80, help="animation frames (default 80)")
    ap.add_argument("--fps", type=int, default=12, help="animation frames per second")
    args = ap.parse_args()

    pred = PipePulsatilePredictor(args.out_dir)

    did_something = False

    if args.t is not None:
        uvwp = pred.predict(args.t, np.array([[args.x, args.y, args.z]]))[0]
        k = pred._window_for(args.t)
        print(f"t={args.t}  (window {k}, τ={args.t - k*pred.stride:.4f})  "
              f"at (x,y,z)=({args.x},{args.y},{args.z})")
        print(f"  u={uvwp[0]:+.5f}  v={uvwp[1]:+.5f}  w={uvwp[2]:+.5f}  p={uvwp[3]:+.5f}")
        did_something = True

    if args.plot:
        for p in pred.save_plots(t_field=args.t):
            print("Saved:", p)
        did_something = True

    if args.spacetime:
        print("Saved:", pred.save_spacetime_centreline())
        did_something = True

    if args.animate:
        print("Saved:", pred.save_axial_animation(n_frames=args.frames, fps=args.fps))
        did_something = True

    if args.vtu:
        print("Saved:", pred.save_vtu_timeseries(n_frames=args.frames))
        did_something = True

    if not did_something:
        ap.error("nothing to do — pass --t, --plot, --spacetime, --animate, and/or --vtu")
