"""Steady 3-D flow through a patient-specific intracranial aneurysm.

Ported from a PyTorch+FBPINN reference (Nagargoje group) to underPINN's
JAX/Flax standard format.  Geometry is built ENTIRELY from the STL surfaces
under ``stl/`` at runtime via :class:`STLVolumeGeometry`, mirroring the NVIDIA
PhysicsNeMo aneurysm tutorial's surface decomposition (no pre-baked npz).

Run directly or via the CLI:

    python examples/Aneurysm/Aneurysm.py
    python examples/Aneurysm/Aneurysm.py myconfig.yaml
    python -m underPINN run examples/Aneurysm/config.yaml

PDE: steady 3-D incompressible Navier-Stokes (``SteadyNS3DPDE``)
Network: (x, y, z) → (u, v, w, p)

Boundary conditions
-------------------
* Inlet  — circular parabolic velocity profile aligned with the vessel axis,
           v = max_vel · max(1 − (r/R)², 0) · n̂  with R from the inlet area
* Outlet — pressure Dirichlet  p = 0
* Wall   — no-slip  u = v = w = 0
* Mass-flow — integral continuity on a mid-stream plane (∫ u·n dA = Q)

STL surfaces (under ``stl/``):
  aneurysm_closed.stl   — watertight interior (rejection-sampled collocation)
  aneurysm_inlet.stl    — inlet cap   (parabolic velocity BC)
  aneurysm_outlet.stl   — outlet cap  (zero-pressure BC)
  aneurysm_noslip.stl   — vessel wall (no-slip BC, sampled directly)
  aneurysm_integral.stl — mid-stream mass-flow plane (points/normal/area)

Outputs (under ``output.dir``):
  aneurysm_volume.vtu   — ParaView point cloud of the lumen interior with the
                          velocity vector, pressure and speed
  aneurysm_wall.vtu     — ParaView wall point cloud with velocity, pressure and
                          wall-shear-stress magnitude (μ = 1/Re)
  grid_predictions.npz  — fields on a structured bounding-box grid (legacy)
  loss.png, loss_hist.npy, params.msgpack, config.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, save_config
from underPINN.nn.mlp import MLP, GatedMLP
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry import STLVolumeGeometry
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.utils.checkpoint import save_checkpoint
from underPINN.utils.io import save_predictions
from underPINN.utils.restart import RestartManager
from underPINN.utils.sampling import safe_choice
from underPINN.utils.vtk_io import save_vtu_points

# Paper / reference geometry constants — physical → normalized coordinate frame
# (matches the original main.py exactly so the npz points map correctly).
_DEFAULT_CENTER = np.array(
    [-18.40381048596882, -50.285383353981196, 12.848136936899031], dtype=np.float64)
_DEFAULT_SCALE  = 0.4
_DEFAULT_INLET_NORMAL = np.array([0.8526, -0.428, 0.299], dtype=np.float64)
_DEFAULT_INLET_CENTER = np.array(
    [-4.24298030045776, 4.082857101816247, -4.637790193399717], dtype=np.float64)
_DEFAULT_INLET_AREA_PHYS = 21.1284     # in physical units; scaled below


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(arr, center, scale):
    """Affine map  x_norm = (x - center) * scale."""
    return (np.asarray(arr) - np.asarray(center)) * scale


def _circular_parabola(pts, center, normal, radius, max_vel):
    """Circular parabolic velocity profile at the inlet (matches original)."""
    d = np.linalg.norm(pts - center[None, :], axis=1)
    speed = max_vel * np.maximum(1.0 - (d / radius) ** 2, 0.0)
    return speed[:, None] * normal[None, :]


def _build_geometry(here: pathlib.Path, cfg, center, scale, cache_dir):
    """Build a STLVolumeGeometry and sample all collocation point clouds.

    Single source of truth for the case's geometry — points come exclusively
    from the STL surfaces (no pre-baked npz fallback).
    """
    d = cfg.data
    n_int    = int(cfg_get(d, "n_interior", default=50_000))
    n_in     = int(cfg_get(d, "n_inlet",    default=2_000))
    n_out    = int(cfg_get(d, "n_outlet",   default=2_000))
    n_wall   = int(cfg_get(d, "n_wall",     default=10_000))

    stl = here / "stl"
    # Optional dedicated wall / integral surfaces (NVIDIA-tutorial layout).
    noslip_path   = stl / "aneurysm_noslip.stl"
    integral_path = stl / "aneurysm_integral.stl"
    geom = STLVolumeGeometry(
        closed_stl  =str(stl / "aneurysm_closed.stl"),
        inlet_stl   =str(stl / "aneurysm_inlet.stl"),
        outlet_stl  =str(stl / "aneurysm_outlet.stl"),
        noslip_stl  =str(noslip_path)   if noslip_path.exists()   else None,
        integral_stl=str(integral_path) if integral_path.exists() else None,
        center=center, scale=scale, cache_dir=cache_dir,
    )
    print(f"  STL: vol={geom.mesh.volume:.2f}, faces={len(geom.mesh.faces)}, "
          f"watertight={geom.mesh.is_watertight}")
    print(f"        inlet  A={geom.inlet_area:.4f}  n_mean="
          f"{geom.inlet_normal.round(3).tolist()}")
    print(f"        outlet A={geom.outlet_area:.4f}  n_mean="
          f"{geom.outlet_normal.round(3).tolist()}")
    if geom.noslip_mesh is not None:
        print(f"        wall (noslip STL) faces={len(geom.noslip_mesh.faces)}")
    if geom.integral_mesh is not None:
        print(f"        integral A={geom.integral_area:.4f}  n_mean="
              f"{geom.integral_normal.round(3).tolist()}")
    print(f"  Sampling: {n_int} interior, {n_in} inlet, "
          f"{n_out} outlet, {n_wall} wall (cached on first run)")
    interior = geom.sample_interior(n_int, seed=0)
    inlet    = geom.sample_inlet(   n_in,  seed=1)
    outlet   = geom.sample_outlet(  n_out, seed=2)
    noslip   = geom.sample_wall(    n_wall, seed=3)
    # Structured eval grid for the final prediction NPZ (uses STL bounds).
    lo, hi = geom.bounds
    nx = ny = nz = 22
    xg = np.linspace(lo[0], hi[0], nx)
    yg = np.linspace(lo[1], hi[1], ny)
    zg = np.linspace(lo[2], hi[2], nz)
    XX, YY, ZZ = np.meshgrid(xg, yg, zg, indexing="ij")
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    return geom, interior, inlet, noslip, outlet, grid


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_Aneurysm(cfg) -> dict:
    """Train a PINN on steady 3-D flow through the patient-specific aneurysm."""
    ph  = cfg.physics
    tr  = cfg.training
    lw  = cfg.loss
    out = cfg_get(cfg, "output", default=None)
    out_dir = (cfg_get(out, "dir", default="outputs/Aneurysm")
               if out else "outputs/Aneurysm")
    os.makedirs(out_dir, exist_ok=True)

    Re      = float(cfg_get(ph, "Re",       default=100.0))
    max_vel = float(cfg_get(ph, "inlet_vel", default=1.5))
    scale   = float(cfg_get(ph, "scale",     default=_DEFAULT_SCALE))
    center  = np.asarray(cfg_get(ph, "center", default=_DEFAULT_CENTER.tolist()),
                         dtype=np.float64)
    inlet_normal = np.asarray(
        cfg_get(ph, "inlet_normal", default=_DEFAULT_INLET_NORMAL.tolist()),
        dtype=np.float64)
    inlet_center = np.asarray(
        cfg_get(ph, "inlet_center", default=_DEFAULT_INLET_CENTER.tolist()),
        dtype=np.float64)
    inlet_area = float(cfg_get(ph, "inlet_area_phys",
                               default=_DEFAULT_INLET_AREA_PHYS)) * scale ** 2
    inlet_radius = float(np.sqrt(inlet_area / np.pi))

    epochs    = int(tr.epochs)
    lr        = float(tr.lr)
    lr_alpha  = float(cfg_get(tr, "lr_alpha",  default=0.01))
    log_every = int(cfg_get(tr, "log_every",   default=500))
    seed      = int(cfg_get(tr, "seed",        default=0))
    batch_r   = int(cfg_get(tr, "batch_r",     default=10000))
    batch_bc  = int(cfg_get(tr, "batch_bc",    default=1024))

    W_PDE      = float(cfg_get(lw, "w_pde",      default=1.0))
    W_INLET    = float(cfg_get(lw, "w_inlet",    default=10.0))
    W_WALL     = float(cfg_get(lw, "w_wall",     default=10.0))
    W_OUTLET   = float(cfg_get(lw, "w_outlet",   default=10.0))
    W_INTEGRAL = float(cfg_get(lw, "w_integral", default=0.1))   # mass-flow term

    # ── Integral continuity (mass-flow conservation) ──────────────────────────
    # Original main.py uses Q = ±2.54 on outlet/mid-stream planes with areas
    # 1.932368 and 2.315767 (after the scale²=0.16 reduction).  Q comes from
    # mass conservation:  Q = (V_max/2) · A_inlet = 0.75 · 3.38 ≈ 2.54.
    ic = cfg_get(cfg, "integral_continuity", default=None)
    use_integral  = bool(cfg_get(ic, "enabled", default=True)) if ic else True
    target_Q      = float(cfg_get(ic, "target_Q",      default=2.54))   if ic else 2.54
    outlet_area   = float(cfg_get(ic, "outlet_area",   default=1.932368)) if ic else 1.932368
    integral_area = float(cfg_get(ic, "integral_area", default=2.315767)) if ic else 2.315767

    # ── Build the point clouds from STL surfaces (no npz fallback) ───────────
    here = pathlib.Path(__file__).parent
    cache_dir = os.path.join(out_dir, "stl_cache")
    geom, xyz_int, xyz_in, xyz_w, xyz_out, xyz_grid = _build_geometry(
        here, cfg, center, scale, cache_dir)
    u_in_target = _circular_parabola(xyz_in.astype(np.float64), inlet_center,
                                     inlet_normal, inlet_radius, max_vel
                                     ).astype(np.float32)

    print(f"Aneurysm (patient-specific):  Re={Re},  inlet U_max={max_vel}")
    print(f"  Inlet radius (normalised) = {inlet_radius:.3f}")
    print(f"  Points:  interior={len(xyz_int)},  inlet={len(xyz_in)},  "
          f"wall={len(xyz_w)},  outlet={len(xyz_out)}")

    # ── Model + PDE ───────────────────────────────────────────────────────────
    net_type = str(cfg_get(cfg.network, "type", default="mlp")).lower()
    net_cls  = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net_type, MLP)
    layers   = list(cfg.network.layers)
    model    = net_cls(layers=layers)
    pde      = SteadyNS3DPDE(model, Re=Re)
    print(f"  Network: {net_cls.__name__}  layers={layers}")

    key    = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 3)))

    lr_sched  = optax.cosine_decay_schedule(lr, decay_steps=epochs, alpha=lr_alpha)
    optimizer = optax.chain(optax.scale_by_adam(),
                            optax.scale_by_schedule(lr_sched),
                            optax.scale(-1.0))
    opt_state = optimizer.init(params)

    # Integral continuity (mass-flow conservation) — STL-only.
    # Outlet plane:   reuse the outlet cap STL sample + its STL-derived normal.
    # Mid-stream plane: prefer the dedicated ``aneurysm_integral.stl`` surface
    # (NVIDIA-tutorial layout); fall back to cutting a virtual cross-section
    # through the closed STL at the user-specified (origin, normal).
    if use_integral:
        ip = cfg_get(cfg, "integral_plane", default=None)
        ip_origin = np.asarray(cfg_get(ip, "origin",
                                       default=[0.3226, 1.2957, -0.3024])
                               if ip else [0.3226, 1.2957, -0.3024],
                               dtype=np.float64)
        ip_normal = np.asarray(cfg_get(ip, "normal",
                                       default=[0.0927, 0.1382, -0.9861])
                               if ip else [0.0927, 0.1382, -0.9861],
                               dtype=np.float64)
        n_ip = int(cfg_get(ip, "n_points", default=2000) if ip else 2000)

        # Outlet flux plane: the inlet/outlet samples already came from STL
        # caps; use the cap's mean face normal for u·n.
        n_out_vec = geom.outlet_normal           # (3,)
        xyz_out_ic = xyz_out                     # already sampled
        n_out_ic   = np.broadcast_to(n_out_vec.astype(np.float32),
                                     xyz_out_ic.shape).copy()

        if geom.integral_mesh is not None:
            # Mid-stream points / normal / area straight off the integral STL.
            # ``orient=ip_normal`` keeps the original downstream (+Q) sign.
            xyz_mid_ic, n_mid_ic, area_mid_from_stl = geom.sample_integral(
                n_ip, seed=42, orient=ip_normal)
            print(f"  Mid-stream plane from aneurysm_integral.stl  "
                  f"A={area_mid_from_stl:.4f}  n={n_mid_ic[0].round(3).tolist()}")
        else:
            # Fallback: cut a slice through the closed mesh
            xyz_mid_ic, n_mid_ic, area_mid_from_stl = geom.sample_plane(
                ip_origin, ip_normal, n=n_ip, seed=42)

        # Re-derive the target Q from the STL inlet area (mass conservation,
        # parabolic profile → U_mean = V_max / 2).  Overrides the config value.
        Q_derived = 0.5 * max_vel * geom.inlet_area
        if cfg_get(ic, "auto_Q", default=True) if ic else True:
            target_Q = Q_derived
        # Use STL-true areas for the two planes
        outlet_area_eff   = geom.outlet_area
        integral_area_eff = area_mid_from_stl
        print(f"  Integral continuity ENABLED  Q={target_Q:.4f}  "
              f"(outlet A={outlet_area_eff:.4f}, mid A={integral_area_eff:.4f})")

        xyz_out_ic_j = jnp.asarray(xyz_out_ic)
        n_out_ic_j   = jnp.asarray(n_out_ic)
        xyz_mid_ic_j = jnp.asarray(xyz_mid_ic)
        n_mid_ic_j   = jnp.asarray(n_mid_ic)
        outlet_area  = float(outlet_area_eff)
        integral_area = float(integral_area_eff)

    xyz_int_j  = jnp.asarray(xyz_int)
    xyz_in_j   = jnp.asarray(xyz_in)
    u_in_j     = jnp.asarray(u_in_target)
    xyz_w_j    = jnp.asarray(xyz_w)
    xyz_out_j  = jnp.asarray(xyz_out)

    # Integral-continuity term:  loss = (A · mean(u·n) − Q_target)²
    # Two planes (outlet → -Q, mid-stream → +Q) preserve mass through the vessel.
    # Whole-plane reduction (mean over all points), so points are NOT batched
    # — they're captured in the closure as full constants.
    def _integral_term(p):
        if not use_integral:
            return jnp.array(0.0)
        u_out  = model.apply(p, xyz_out_ic_j)[:, :3]
        flux_o = outlet_area   * jnp.mean(jnp.sum(u_out * n_out_ic_j, axis=1))
        u_mid  = model.apply(p, xyz_mid_ic_j)[:, :3]
        flux_m = integral_area * jnp.mean(jnp.sum(u_mid * n_mid_ic_j, axis=1))
        return (flux_o + target_Q) ** 2 + (flux_m - target_Q) ** 2

    @jax.jit
    def step(params, state, r_b, in_b, u_in_b, w_b, out_b):
        def loss_fn(p):
            res     = pde.residual(p, r_b)
            pde_l   = jnp.mean(jnp.sum(res ** 2, axis=-1))

            uvwp_in = model.apply(p, in_b)
            in_l    = (jnp.mean((uvwp_in[:, 0] - u_in_b[:, 0]) ** 2)
                       + jnp.mean((uvwp_in[:, 1] - u_in_b[:, 1]) ** 2)
                       + jnp.mean((uvwp_in[:, 2] - u_in_b[:, 2]) ** 2))

            uvwp_w  = model.apply(p, w_b)
            wall_l  = jnp.mean(uvwp_w[:, 0] ** 2 + uvwp_w[:, 1] ** 2 + uvwp_w[:, 2] ** 2)

            uvwp_o  = model.apply(p, out_b)
            outlet_l = jnp.mean(uvwp_o[:, 3] ** 2)

            integral_l = _integral_term(p)

            total = (W_PDE * pde_l + W_INLET * in_l
                     + W_WALL * wall_l + W_OUTLET * outlet_l
                     + W_INTEGRAL * integral_l)
            return total, (pde_l, in_l, wall_l, outlet_l, integral_l)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state, total, aux

    # ── Restart ───────────────────────────────────────────────────────────────
    save_restart = int(cfg_get(tr, "save_restart_every", default=500))
    restart = RestartManager(out_dir, save_every=save_restart, cfg=cfg)
    start_ep, params, opt_state, hists = restart.maybe_restore(params, opt_state)
    loss_hist = hists.get("loss_hist", [])

    logger = ConsoleLogger(log_every=log_every)
    N_r, N_in = xyz_int_j.shape[0], xyz_in_j.shape[0]
    N_w, N_out = xyz_w_j.shape[0], xyz_out_j.shape[0]
    key = jax.random.PRNGKey(seed + 99)

    try:
        for ep in range(start_ep, epochs):
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ir   = safe_choice(k1, N_r,   batch_r)
            iin  = safe_choice(k2, N_in,  min(batch_bc, N_in))
            iw   = safe_choice(k3, N_w,   min(batch_bc, N_w))
            iout = safe_choice(k4, N_out, min(batch_bc, N_out))

            params, opt_state, total, (pl, il, wl, ol, intl) = step(
                params, opt_state,
                xyz_int_j[ir], xyz_in_j[iin], u_in_j[iin],
                xyz_w_j[iw], xyz_out_j[iout])
            loss_hist.append(float(total))
            logger.on_epoch_end(ep, {"loss": float(total), "pde": float(pl),
                                     "inlet": float(il), "wall": float(wl),
                                     "outlet": float(ol),
                                     "integral": float(intl)})
            restart.maybe_save(ep, params, opt_state, {"loss_hist": loss_hist})
    except StopIteration:
        pass

    restart.done()
    logger.on_train_end({"loss": loss_hist[-1] if loss_hist else float("nan")})

    # ── Save predictions on the structured grid + interior pool ───────────────
    uvwp_grid = np.array(model.apply(params, jnp.asarray(xyz_grid)))
    save_predictions(
        out_dir,
        coords  = {"x": xyz_grid[:, 0], "y": xyz_grid[:, 1], "z": xyz_grid[:, 2]},
        outputs = {"u_pred": uvwp_grid[:, 0], "v_pred": uvwp_grid[:, 1],
                   "w_pred": uvwp_grid[:, 2], "p_pred": uvwp_grid[:, 3]},
        filename="grid_predictions.npz",
    )

    # ── ParaView VTU export — geometry-faithful flow field ────────────────────
    # The structured grid above samples the STL *bounding box* (many points lie
    # outside the lumen).  These VTU files instead carry the flow variables on
    # points that actually lie inside the vessel / on its wall, so they open
    # cleanly in ParaView for streamlines, glyphs and surface colouring.
    #   aneurysm_volume.vtu — interior points: velocity (u,v,w), pressure, speed
    #   aneurysm_wall.vtu   — wall points:    velocity, pressure, WSS magnitude
    # Coordinates and fields are in the (nondimensional) simulation frame.
    uvwp_int = np.array(model.apply(params, xyz_int_j))
    vel_int  = uvwp_int[:, :3]
    save_vtu_points(
        os.path.join(out_dir, "aneurysm_volume.vtu"),
        np.asarray(xyz_int, dtype=np.float64),
        {"velocity": vel_int, "pressure": uvwp_int[:, 3],
         "speed": np.linalg.norm(vel_int, axis=1)},
    )

    # Wall: velocity, pressure and wall-shear-stress magnitude.
    # WSS = | (τ·n) − ((τ·n)·n) n |,  τ = μ(∇u + ∇uᵀ),  μ = 1/Re (Newtonian).
    xyz_wn, n_wn = geom.sample_wall_with_normals(
        int(cfg_get(cfg.data, "n_wall", default=10_000)), seed=7)
    mu = 1.0 / Re

    def _grad_u(p, xi):
        return jax.jacfwd(lambda x: model.apply(p, x.reshape(1, 3))[0, :3])(xi)

    gradu  = jax.vmap(lambda xi: _grad_u(params, xi))(jnp.asarray(xyz_wn))
    strain = gradu + jnp.transpose(gradu, (0, 2, 1))
    n_j    = jnp.asarray(n_wn)
    trac   = mu * jnp.einsum("nij,nj->ni", strain, n_j)          # traction τ·n
    wss_v  = trac - jnp.sum(trac * n_j, axis=1, keepdims=True) * n_j
    wss    = np.array(jnp.linalg.norm(wss_v, axis=1))
    uvwp_w = np.array(model.apply(params, jnp.asarray(xyz_wn)))
    save_vtu_points(
        os.path.join(out_dir, "aneurysm_wall.vtu"),
        np.asarray(xyz_wn, dtype=np.float64),
        {"velocity": uvwp_w[:, :3], "pressure": uvwp_w[:, 3], "wss": wss},
    )
    print(f"  ParaView: aneurysm_volume.vtu ({len(xyz_int)} pts), "
          f"aneurysm_wall.vtu ({len(xyz_wn)} pts, WSS μ={mu:.4g})")

    # ── Loss plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(loss_hist, lw=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Aneurysm (patient-specific) — Re={Re}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    save_checkpoint(params, out_dir, metadata={
        "problem": "Aneurysm",
        "network": {"type": net_type, "layers": layers},
        "physics": {"Re": Re, "inlet_vel": max_vel, "inlet_radius": inlet_radius,
                    "scale": scale, "center": center.tolist(),
                    "inlet_normal": inlet_normal.tolist(),
                    "inlet_center": inlet_center.tolist()},
        "results": {"n_epochs": len(loss_hist),
                    "final_loss": loss_hist[-1] if loss_hist else float("nan")},
    })
    print(f"\nOutputs saved to: {out_dir}/")

    return {"params": params, "loss_hist": loss_hist,
            "n_epochs": len(loss_hist)}


if __name__ == "__main__":
    import sys
    _HERE = pathlib.Path(__file__).parent
    cfg_path = str(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else _HERE / "config.yaml")
    from underPINN.config.loader import load_config
    run_Aneurysm(load_config(cfg_path))
