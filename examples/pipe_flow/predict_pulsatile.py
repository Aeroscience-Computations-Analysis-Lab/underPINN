"""Load a trained pulsatile-pipe time-marching model and predict at any time.

The time-marching run saves one checkpoint per window under
``<out_dir>/windows/`` plus an index ``<out_dir>/windows_index.json``.  Each
window k starts at absolute time ``k·stride`` and covers a span of length
``dT`` (``stride == dT`` ⇒ non-overlapping, ``stride < dT`` ⇒ overlapping).
The network takes the **local** time ``τ = t − k·stride`` as its 4th input.
For overlapping windows multiple windows cover a given t — this module picks
the latest window whose start ≤ t (its weights are the freshest).

Usage (as a library)
--------------------
    from examples.pipe_flow.predict_pulsatile import PulsatilePredictor
    import numpy as np

    pred = PulsatilePredictor("outputs/pipe_flow_pulsatile_transfer")
    xyz  = np.array([[0.0, 0.0, 0.0]])        # pipe centreline at x=0
    uvwp = pred.predict(t_abs=2.7, xyz=xyz)   # (u, v, w, p) at t=2.7
    print(uvwp)

Usage (CLI)
-----------
    python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer --t 2.7
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.nn.mlp import MLP, GatedMLP
from underPINN.utils.checkpoint import load_checkpoint


class PulsatilePredictor:
    """Reconstruct the full-horizon solution from per-window checkpoints."""

    def __init__(self, out_dir: str):
        self.out_dir   = out_dir
        self.win_dir   = os.path.join(out_dir, "windows")
        index_path     = os.path.join(out_dir, "windows_index.json")

        net = dT = stride = n_windows = T_total = None
        physics = {}

        # Preferred: the index file written by training.
        if os.path.exists(index_path):
            with open(index_path) as f:
                idx = json.load(f)
            net       = idx["network"]
            dT        = float(idx["dT"])
            stride    = float(idx.get("stride", dT))    # legacy runs → stride=dT
            n_windows = int(idx["n_windows"])
            T_total   = float(idx.get("T_total", (n_windows - 1) * stride + dT))
            physics   = idx.get("physics", {})
        else:
            # Fallback: reconstruct from any per-window metadata sidecar.
            # (The index is only written at the very end of a run, but each
            #  window's *_meta.json carries dT / n_windows / network.)
            meta = self._first_window_meta()
            if meta is None:
                raise FileNotFoundError(
                    f"No windows_index.json and no window checkpoints found under "
                    f"{self.win_dir}.  Run the time-marching training first."
                )
            net       = meta["network"]
            tm        = meta.get("time_marching", {})
            dT        = float(tm["dT"])
            stride    = float(tm.get("stride", dT))     # legacy runs → stride=dT
            n_windows = int(tm["n_windows"])
            T_total   = float(tm.get("T_total", (n_windows - 1) * stride + dT))
            physics   = meta.get("physics", {})
            print(f"  [predict] windows_index.json missing — reconstructed from "
                  f"per-window metadata (dT={dT}, stride={stride}, "
                  f"n_windows={n_windows}).")

        self.dT        = dT
        self.stride    = stride
        self.T_total   = T_total
        self.n_windows = n_windows
        self.physics   = physics or {}

        layers  = list(net["layers"])
        net_cls = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net.get("type", "mlp"), MLP)
        self.model = net_cls(layers=layers)

        # Which window checkpoints actually exist on disk
        self.available = sorted(
            int(fn[len("params_window_"):-len(".msgpack")])
            for fn in os.listdir(self.win_dir)
            if fn.startswith("params_window_") and fn.endswith(".msgpack")
        ) if os.path.isdir(self.win_dir) else []
        if not self.available:
            raise FileNotFoundError(f"No window checkpoints found in {self.win_dir}.")
        print(f"  [predict] {len(self.available)} window checkpoint(s) available "
              f"(windows {self.available[0]}–{self.available[-1]}).")

        # Lazily load each window's params on first use
        self._params_cache: dict[int, object] = {}

    # ------------------------------------------------------------------

    def _first_window_meta(self):
        """Read any params_window_XXX_meta.json to recover run settings."""
        if not os.path.isdir(self.win_dir):
            return None
        for fn in sorted(os.listdir(self.win_dir)):
            if fn.startswith("params_window_") and fn.endswith("_meta.json"):
                with open(os.path.join(self.win_dir, fn)) as f:
                    return json.load(f)
        return None

    def _window_for(self, t_abs: float) -> int:
        # Latest window whose start ≤ t_abs (overlapping → freshest weights win)
        k = int(t_abs // self.stride)
        k = max(0, min(k, self.n_windows - 1))
        if k not in self.available:
            raise FileNotFoundError(
                f"t={t_abs} maps to window {k}, but its checkpoint is missing. "
                f"Available windows: {self.available[0]}–{self.available[-1]} "
                f"(t up to {self.available[-1] * self.stride + self.dT:.3f})."
            )
        return k

    def _params(self, k: int):
        if k not in self._params_cache:
            mp = os.path.join(self.out_dir, "windows", f"params_window_{k:03d}.msgpack")
            self._params_cache[k] = load_checkpoint(self.model, mp)
        return self._params_cache[k]

    def predict(self, t_abs: float, xyz: np.ndarray) -> np.ndarray:
        """(u, v, w, p) at absolute time *t_abs* for spatial points xyz (N, 3)."""
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        k   = self._window_for(t_abs)
        tau = float(t_abs - k * self.stride)
        xyzt = jnp.concatenate(
            [jnp.asarray(xyz), jnp.full((xyz.shape[0], 1), tau, dtype=jnp.float32)],
            axis=1)
        return np.array(self.model.apply(self._params(k), xyzt))

    # ------------------------------------------------------------------
    # Visualisation — regenerate figures from the saved windows
    # ------------------------------------------------------------------

    @property
    def t_max(self) -> float:
        """Largest absolute time covered by the available checkpoints."""
        return self.available[-1] * self.stride + self.dT

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
        ax.plot(ts, peak, "k--", lw=1.5, label="Inlet peak forcing")
        ax.plot(ts, uc, "b-", lw=1.8, label=f"PINN centreline u @ x={x_mid:.1f}")
        for k in self.available[1:]:
            ax.axvline(k * self.stride, color="grey", lw=0.5, ls=":", alpha=0.5)
        ax.set_xlabel("t")
        ax.set_ylabel("u (centreline)")
        ax.set_title(f"Pulsatile pipe — centreline velocity  (Re={Re})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(self.out_dir, "predict_centreline_timeseries.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        # 2) radial profiles at four snapshots
        Nr    = 80
        r_arr = np.linspace(0.0, R, Nr, dtype=np.float32)
        xyz_r = np.stack([np.full(Nr, x_mid, np.float32), r_arr,
                          np.zeros(Nr, np.float32)], axis=1)
        snaps = np.linspace(0.0, t_hi, 4, endpoint=False) + 0.25 * self.dT
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for ax, tt in zip(np.atleast_1d(axes), snaps):
            u_pred = self.predict(float(tt), xyz_r)[:, 0]
            pk     = V_max + V_amp * np.sin(omega * tt)
            u_qs   = pk * (1.0 - r_arr ** 2 / R ** 2)
            ax.plot(r_arr, u_qs, "k--", lw=1.3, label="Quasi-steady")
            ax.plot(r_arr, u_pred, "b-", lw=1.8, label="PINN")
            ax.set_title(f"t = {tt:.2f}")
            ax.set_xlabel("r")
            ax.set_ylabel("u")
            ax.grid(alpha=0.3)
            if tt == snaps[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"Radial velocity profiles at x={x_mid:.1f}  (Re={Re})")
        fig.tight_layout()
        p = os.path.join(self.out_dir, "predict_radial_profiles.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        # 3) optional cross-section of axial velocity at a chosen time
        if t_field is not None:
            N = 80
            yy = np.linspace(-R, R, N, dtype=np.float32)
            zz = np.linspace(-R, R, N, dtype=np.float32)
            YY, ZZ = np.meshgrid(yy, zz)
            xyz = np.stack([np.full(N * N, x_mid, np.float32),
                            YY.ravel(), ZZ.ravel()], axis=1)
            u_cs = self.predict(float(t_field), xyz)[:, 0].reshape(N, N)
            u_cs = np.where(YY ** 2 + ZZ ** 2 > R ** 2, np.nan, u_cs)
            fig, ax = plt.subplots(figsize=(5, 4))
            cf = ax.contourf(yy, zz, u_cs, levels=50, cmap="jet")
            plt.colorbar(cf, ax=ax, label="u")
            ax.set_aspect("equal")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_title(f"Axial velocity u at x={x_mid:.1f}, t={t_field:.2f}")
            fig.tight_layout()
            p = os.path.join(self.out_dir, f"predict_crosssection_t{t_field:.2f}.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

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

            fig, ax = plt.subplots(figsize=(15, 3.4))
            cf = ax.contourf(xg, yg, U, levels=60, cmap="jet")
            plt.colorbar(cf, ax=ax, label="u (streamwise)")
            # streamlines show the in-plane flow direction
            ax.streamplot(xg, yg, np.nan_to_num(U), np.nan_to_num(V),
                          density=(2.4, 0.6), color="k", linewidth=0.5,
                          arrowsize=0.7)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal")
            ax.set_xlabel("x  (streamwise)")
            ax.set_ylabel("y")
            ax.set_title(f"Streamwise plane z=0 — u & streamlines  "
                         f"(t={t_field:.2f}, Re={Re})")
            fig.tight_layout()
            p = os.path.join(self.out_dir, f"predict_axial_plane_t{t_field:.2f}.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        return saved

    # ------------------------------------------------------------------
    # Axial profile over ALL time
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
        """Static 'all-time' view: centreline axial velocity u(x, t) as a map.

        x runs along the pipe (streamwise), t along the full covered horizon.
        """
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
        cf = ax.contourf(ts, xs, U.T, levels=60, cmap="jet")
        fig.colorbar(cf, ax=ax, label="u (centreline, r=0)")
        ax.set_xlabel("t")
        ax.set_ylabel("x  (streamwise)")
        ax.set_title(f"Centreline axial velocity — space–time  (Re={Re})")
        fig.tight_layout()
        out = os.path.join(self.out_dir, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

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
        levels = np.linspace(vmin, vmax, 50)

        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(14, 3.2))
        cf0 = ax.contourf(xg, yg, frames[0], levels=levels, cmap="jet")
        fig.colorbar(cf0, ax=ax, label="u (streamwise)")

        def _draw(i):
            ax.clear()
            ax.contourf(xg, yg, frames[i], levels=levels, cmap="jet")
            ax.set_aspect("equal")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(-R, R)
            ax.set_xlabel("x  (streamwise)")
            ax.set_ylabel("y")
            ax.set_title(f"u(x, y)  z=0    t = {times[i]:.3f}    (Re={Re})")

        anim = FuncAnimation(fig, _draw, frames=len(times), interval=1000.0 / fps)
        out = os.path.join(self.out_dir, filename)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  saved animation ({len(times)} frames) → {out}")
        return out


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
    ap.add_argument("--frames", type=int, default=80, help="animation frames (default 80)")
    ap.add_argument("--fps", type=int, default=12, help="animation frames per second")
    args = ap.parse_args()

    pred = PulsatilePredictor(args.out_dir)

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

    if not did_something:
        ap.error("nothing to do — pass --t, --plot, --spacetime, and/or --animate")
