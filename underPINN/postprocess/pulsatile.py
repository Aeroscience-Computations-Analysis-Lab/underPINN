"""Reconstruct a windowed time-marching solution and predict at any time.

Geometry-agnostic core for the time-marching transfer-learning runs (pipe, AAA,
their Carreau variants).  The run saves one checkpoint per window under
``<out_dir>/windows/`` plus an index ``<out_dir>/windows_index.json``.  Window
``k`` starts at absolute time ``k·stride`` and spans ``dT`` (``stride == dT`` ⇒
non-overlapping, ``stride < dT`` ⇒ overlapping); the network takes the **local**
time ``τ = t − k·stride`` as its 4th input.  For overlapping windows several
windows cover a given ``t`` — the latest window whose start ≤ t wins (freshest
weights).

Example
-------
    from underPINN.postprocess import PulsatilePredictor
    import numpy as np

    pred = PulsatilePredictor("outputs/pipe_flow_pulsatile_transfer")
    uvwp = pred.predict(t_abs=2.7, xyz=np.array([[0.0, 0.0, 0.0]]))

Geometry-specific figures / videos build on top of this core (see the example
post-processors).
"""
from __future__ import annotations

import json
import os

import numpy as np
import jax.numpy as jnp

from underPINN.nn.factory import build_model
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

        # ``net`` is the full saved network config (carries ω / harmonics for
        # the temporal-Fourier net), so the factory rebuilds the exact model.
        self.model = build_model(net)

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

    @property
    def t_max(self) -> float:
        """Largest absolute time covered by the available checkpoints."""
        return self.available[-1] * self.stride + self.dT
