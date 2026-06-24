"""Predict the patient-specific Aneurysm solution on LARGE collocation clouds.

Loads a trained ``Aneurysm`` checkpoint, rebuilds the STL geometry, samples a
large interior + wall point cloud and evaluates the network — all in **memory-
safe chunks** so you can post-process hundreds of thousands to millions of
points without running the GPU out of memory.  The wall-shear-stress Jacobian
(the memory-heavy part) is chunked too.

Outputs (into ``out_dir``):
  predict_volume.vtu    interior cloud: velocity (u,v,w), pressure, speed
  predict_wall.vtu      wall cloud:     velocity, pressure, WSS magnitude
  predict_fields.npz    all arrays (coords, velocity, pressure, wss)

Usage (CLI)
-----------
    python examples/Aneurysm/predict_aneurysm.py outputs/Aneurysm \
        --n-volume 500000 --n-wall 100000 --batch 50000 --vtu --npz
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

from underPINN.geometry import STLVolumeGeometry
from underPINN.nn.factory import build_model
from underPINN.utils.checkpoint import load_checkpoint, read_metadata
from underPINN.utils.vtk_io import save_vtu_points

_HERE = pathlib.Path(__file__).parent
_DEFAULT_CENTER = [-18.40381048596882, -50.285383353981196, 12.848136936899031]
_DEFAULT_SCALE  = 0.4


def _chunked(fn, X: np.ndarray, batch: int) -> np.ndarray:
    """Apply a jitted ``fn`` to ``X`` (N, d) in fixed-size, padded chunks.

    Every chunk is padded to exactly ``batch`` rows so the jitted function is
    traced once (constant shape → no recompiles, no giant single allocation).
    """
    N = X.shape[0]
    out = []
    for i in range(0, N, batch):
        xb = X[i:i + batch]
        m  = xb.shape[0]
        if m < batch:                                   # pad the final short chunk
            pad = np.zeros((batch - m,) + xb.shape[1:], dtype=xb.dtype)
            xb  = np.concatenate([xb, pad], axis=0)
        yb = np.asarray(fn(jnp.asarray(xb)))
        out.append(yb[:m])
    return np.concatenate(out, axis=0)


class AneurysmPredictor:
    """Evaluate a trained Aneurysm model on large STL-sampled point clouds."""

    def __init__(self, out_dir: str, stl_dir: str | None = None,
                 cache_dir: str | None = None):
        self.out_dir = out_dir
        ckpt = os.path.join(out_dir, "params.msgpack")
        meta = read_metadata(ckpt) or {}
        net  = meta.get("network", {"type": "mlp", "layers": [3, 192, 192, 192, 192, 192, 4]})
        phys = meta.get("physics", {})
        self.Re     = float(phys.get("Re", 40.0))
        self.scale  = float(phys.get("scale", _DEFAULT_SCALE))
        self.center = np.asarray(phys.get("center", _DEFAULT_CENTER), dtype=np.float64)
        self.mu     = 1.0 / self.Re

        self.model  = build_model(net)
        # Initialise then overwrite with the trained params.
        self.params = self.model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
        self.params = load_checkpoint(self.params, ckpt)
        print(f"  [predict] loaded {net.get('type')} {net.get('layers')}  "
              f"Re={self.Re}  scale={self.scale}")

        # ── STL geometry (same surfaces as training) ──────────────────────────
        stl = pathlib.Path(stl_dir) if stl_dir else (_HERE / "stl")
        def _p(name):
            p = stl / name
            return str(p) if p.exists() else None
        self.geom = STLVolumeGeometry(
            closed_stl  =str(stl / "aneurysm_closed.stl"),
            inlet_stl   =_p("aneurysm_inlet.stl"),
            outlet_stl  =_p("aneurysm_outlet.stl"),
            noslip_stl  =_p("aneurysm_noslip.stl"),
            integral_stl=_p("aneurysm_integral.stl"),
            center=self.center, scale=self.scale,
            cache_dir=cache_dir or os.path.join(out_dir, "predict_cache"),
        )
        # JIT a batched forward once (constant-shape chunks reuse the compile).
        self._fwd = jax.jit(lambda x: self.model.apply(self.params, x))

    # ------------------------------------------------------------------
    def predict_points(self, xyz: np.ndarray, batch: int = 50000) -> np.ndarray:
        """Chunked (u, v, w, p) for an (N, 3) cloud — memory-safe at any N."""
        xyz = np.asarray(xyz, dtype=np.float32)
        return _chunked(self._fwd, xyz, batch)          # (N, 4)

    def predict_volume(self, n_points: int = 500_000, batch: int = 50000,
                       seed: int = 0):
        """Sample *n_points* in the lumen and evaluate the flow field."""
        print(f"  [predict] sampling {n_points} interior points "
              f"(rejection in mesh; cached on first run)…")
        xyz  = self.geom.sample_interior(n_points, seed=seed)
        uvwp = self.predict_points(xyz, batch=batch)
        return np.asarray(xyz), uvwp

    def predict_wall(self, n_points: int = 100_000, batch: int = 50000,
                     seed: int = 1):
        """Sample *n_points* on the wall and evaluate velocity, pressure, WSS."""
        print(f"  [predict] sampling {n_points} wall points + normals…")
        xyz, nrm = self.geom.sample_wall_with_normals(n_points, seed=seed)
        uvwp = self.predict_points(xyz, batch=batch)

        # WSS = | (τ·n) − ((τ·n)·n) n |,  τ = μ(∇u + ∇uᵀ).  Chunk the Jacobian.
        def _grad_u(xb):                                # (B,3) → (B,3,3)
            g = jax.vmap(lambda xi: jax.jacfwd(
                lambda x: self.model.apply(self.params, x.reshape(1, 3))[0, :3])(xi))
            return g(xb)
        gfun  = jax.jit(_grad_u)
        gradu = _chunked(gfun, np.asarray(xyz, np.float32), batch)   # (N,3,3)
        strain = gradu + np.transpose(gradu, (0, 2, 1))
        trac   = self.mu * np.einsum("nij,nj->ni", strain, nrm)
        wss_v  = trac - np.sum(trac * nrm, axis=1, keepdims=True) * nrm
        wss    = np.linalg.norm(wss_v, axis=1)
        return np.asarray(xyz), uvwp, np.asarray(nrm), wss

    # ------------------------------------------------------------------
    def save(self, n_volume: int = 500_000, n_wall: int = 100_000,
             batch: int = 50000, vtu: bool = True, npz: bool = True) -> list[str]:
        xyz_v, uvwp_v = self.predict_volume(n_volume, batch=batch)
        xyz_w, uvwp_w, nrm_w, wss = self.predict_wall(n_wall, batch=batch)
        vel_v = uvwp_v[:, :3]
        vel_w = uvwp_w[:, :3]
        saved: list[str] = []

        if vtu:
            saved.append(save_vtu_points(
                os.path.join(self.out_dir, "predict_volume.vtu"),
                xyz_v.astype(np.float64),
                {"velocity": vel_v, "pressure": uvwp_v[:, 3],
                 "speed": np.linalg.norm(vel_v, axis=1)}))
            saved.append(save_vtu_points(
                os.path.join(self.out_dir, "predict_wall.vtu"),
                xyz_w.astype(np.float64),
                {"velocity": vel_w, "pressure": uvwp_w[:, 3], "wss": wss}))
        if npz:
            p = os.path.join(self.out_dir, "predict_fields.npz")
            np.savez(p,
                     vol_xyz=xyz_v, vol_velocity=vel_v, vol_pressure=uvwp_v[:, 3],
                     wall_xyz=xyz_w, wall_velocity=vel_w, wall_pressure=uvwp_w[:, 3],
                     wall_normal=nrm_w, wall_wss=wss)
            saved.append(p)
        print(f"  [predict] volume={len(xyz_v)} pts, wall={len(xyz_w)} pts, "
              f"WSS range [{wss.min():.4g}, {wss.max():.4g}]")
        return saved


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Predict the Aneurysm flow on large collocation clouds.")
    ap.add_argument("out_dir", help="trained output dir (e.g. outputs/Aneurysm)")
    ap.add_argument("--n-volume", type=int, default=500_000,
                    help="interior points (default 500k)")
    ap.add_argument("--n-wall", type=int, default=100_000,
                    help="wall points (default 100k)")
    ap.add_argument("--batch", type=int, default=50_000,
                    help="evaluation chunk size — lower it if you hit OOM")
    ap.add_argument("--stl-dir", default=None, help="override STL directory")
    ap.add_argument("--no-vtu", action="store_true", help="skip VTU export")
    ap.add_argument("--no-npz", action="store_true", help="skip npz export")
    args = ap.parse_args()

    pred = AneurysmPredictor(args.out_dir, stl_dir=args.stl_dir)
    for p in pred.save(n_volume=args.n_volume, n_wall=args.n_wall,
                       batch=args.batch, vtu=not args.no_vtu, npz=not args.no_npz):
        print("Saved:", p)
