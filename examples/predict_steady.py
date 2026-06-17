"""Predict / post-process the steady pipe-flow and AAA cases.

Works with all four steady internal-flow examples (it reads the ``config.yaml``
that training saved into the output directory and reconstructs everything):

    pipe_flow            — Newtonian pipe
    pipe_flow_rheology   — Carreau (shear-thinning blood) pipe
    AAA_flow             — Newtonian axisymmetric bulge
    AAA_rheology         — Carreau axisymmetric bulge

Usage
-----
    python examples/predict_steady.py outputs/pipe_flow
    python examples/predict_steady.py outputs/AAA_rheology

Outputs (written into the same directory)
-----------------------------------------
    predict_axial_velocity.png   u contour + streamlines on the axial plane z=0
    predict_axial_pressure.png   pressure contour on the same plane
    predict_pressure_line.png    centreline p(x) and wall p(x)
    predict_wss.png              wall shear stress along the wall (y-slice)
    predict_solution.npz         everything saved as arrays:
        2-D axial plane (z=0): x, y, u, v, w, p
        1-D pressure lines   : x_centreline, p_centreline, p_wall
        Wall slice           : wall_x, wall_R, wss [+ wss_Pa]
        3-D WALL distribution (parametrised by x, θ):
            wall3d_x, wall3d_theta, wall3d_xyz (Nx3, Nth, 3),
            wall3d_p (Nx3, Nth), wall3d_wss (Nx3, Nth) [+ wall3d_wss_Pa]
        3-D INTERIOR volume on a structured (x, y, z) grid:
            vol_x, vol_y, vol_z, vol_u, vol_v, vol_w, vol_p
        Metadata: Re, label

Wall shear stress
-----------------
WSS is the tangential viscous traction at the wall:

    τ_w = | (μ*/Re) S·n − [n·((μ*/Re) S·n)] n |,   S = ∇u + ∇uᵀ

with μ* = 1 for the Newtonian cases and the Carreau viscosity
μ* = 1 + (β−1)[1+(Cu γ̇)²]^((n−1)/2) for the rheology cases.  Values are in
the trained non-dimensional units; for the rheology cases a dimensional copy
``wss_Pa = wss · μ∞U/R`` is also stored.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from underPINN.config.loader import cfg_get, load_config
from underPINN.nn.mlp import MLP, GatedMLP
from underPINN.utils.checkpoint import load_checkpoint
from underPINN.geometry.aaa import BulgeGeometry


def _get2(ns, key_new, key_old, default=None):
    """Read a config key by its current name, falling back to the legacy one,
    then to *default* if neither is present."""
    val = cfg_get(ns, key_new, default=None)
    if val is None:
        val = cfg_get(ns, key_old, default=None)
    return default if val is None else val


# ---------------------------------------------------------------------------
# Case adapters — geometry/physics in the TRAINED coordinate system
# ---------------------------------------------------------------------------

def _case_setup(cfg) -> dict:
    prob = str(cfg.problem)
    ph   = cfg.physics

    if prob == "pipe_flow":                                   # Newtonian pipe
        R    = float(ph.R)
        L    = float(ph.L)
        x_lo = float(cfg_get(ph, "x_lo", default=0.0))
        return dict(label="Pipe flow (Newtonian)", newtonian=True,
                    Re=float(ph.Re), x_lo=x_lo, x_hi=x_lo + L,
                    radius=lambda x: np.full_like(np.asarray(x, float), R),
                    beta=None, Cu=None, n=None, dim=None)

    if prob == "pipe_flow_rheology":                          # Carreau pipe
        if cfg_get(ph, "Re", default=None) is not None:
            # New style: same domain/Re as the Newtonian pipe + (β, Cu, n).
            # Defaults make this work even on a partial config.
            R    = float(cfg_get(ph, "R",    default=0.5))
            L    = float(cfg_get(ph, "L",    default=7.0))
            x_lo = float(cfg_get(ph, "x_lo", default=-3.5))
            return dict(label="Pipe flow (Carreau)", newtonian=False,
                        Re=float(ph.Re), x_lo=x_lo, x_hi=x_lo + L,
                        radius=lambda x: np.full_like(np.asarray(x, float), R),
                        beta=float(cfg_get(ph, "beta", default=16.0)),
                        Cu  =float(cfg_get(ph, "Cu",   default=10.0)),
                        n   =float(cfg_get(ph, "n",    default=0.3568)),
                        dim=None)
        # Legacy style: dimensional blood inputs, trained non-dimensionally.
        # Every key is read via cfg_get so partial configs still resolve.
        rho    = float(cfg_get(ph, "rho",    default=1060.0))
        mu0    = float(cfg_get(ph, "mu0",    default=0.056))
        mu_inf = float(cfg_get(ph, "mu_inf", default=0.0035))
        lam    = float(cfg_get(ph, "lam",    default=3.131))
        n      = float(cfg_get(ph, "n",      default=0.3568))
        Rd     = float(cfg_get(ph, "R",      default=0.004))
        Ld     = float(cfg_get(ph, "L",      default=0.04))
        U      = float(cfg_get(ph, "U",      default=0.05))
        x_lo   = float(cfg_get(ph, "x_lo",   default=0.0)) / Rd
        return dict(label="Pipe flow (Carreau)", newtonian=False,
                    Re=rho * U * Rd / mu_inf, x_lo=x_lo, x_hi=x_lo + Ld / Rd,
                    radius=lambda x: np.full_like(np.asarray(x, float), 1.0),
                    beta=mu0 / mu_inf, Cu=lam * U / Rd, n=n,
                    dim=dict(mu_inf=mu_inf, U=U, R=Rd))

    if prob in ("AAA_flow", "aneurysm_flow"):                 # Newtonian bulge
        Rv   = float(cfg_get(ph, "R_vessel", default=0.5))
        Ra   = float(_get2(ph, "R_AAA", "R_aneurysm", default=1.0))
        La   = float(_get2(ph, "L_AAA", "L_aneurysm", default=1.5))
        L    = float(cfg_get(ph, "L",    default=7.0))
        x_lo = float(cfg_get(ph, "x_lo", default=-3.5))
        x0   = float(cfg_get(ph, "x0",   default=-2.0))
        geom = BulgeGeometry(R_vessel=Rv, R_AAA=Ra, L=L,
                             x_lo=x_lo, x0=x0, L_AAA=La)
        return dict(label="AAA (Newtonian)", newtonian=True,
                    Re=float(ph.Re), x_lo=x_lo, x_hi=x_lo + L,
                    radius=lambda x: np.asarray(geom.radius_at(x), float),
                    beta=None, Cu=None, n=None, dim=None)

    if prob in ("AAA_rheology", "aneurysm_rheology"):         # Carreau bulge
        Rv = float(cfg_get(ph, "R_vessel", default=0.5))
        Ra = float(_get2(ph, "R_AAA", "R_aneurysm", default=1.0))
        La = float(_get2(ph, "L_AAA", "L_aneurysm", default=1.5))
        if cfg_get(ph, "Re", default=None) is not None:
            # New style: same domain/Re as the Newtonian AAA + (β, Cu, n).
            # Defaults make this work even on a partial config.
            L    = float(cfg_get(ph, "L",    default=7.0))
            x_lo = float(cfg_get(ph, "x_lo", default=-3.5))
            x0   = float(cfg_get(ph, "x0",   default=-2.0))
            geom = BulgeGeometry(R_vessel=Rv, R_AAA=Ra, L=L,
                                 x_lo=x_lo, x0=x0, L_AAA=La)
            return dict(label="AAA (Carreau)", newtonian=False,
                        Re=float(ph.Re), x_lo=x_lo, x_hi=x_lo + L,
                        radius=lambda x: np.asarray(geom.radius_at(x), float),
                        beta=float(cfg_get(ph, "beta", default=16.0)),
                        Cu  =float(cfg_get(ph, "Cu",   default=10.0)),
                        n   =float(cfg_get(ph, "n",    default=0.3568)),
                        dim=None)
        # Legacy style: dimensional blood inputs, trained non-dimensionally.
        rho    = float(cfg_get(ph, "rho",    default=1060.0))
        mu0    = float(cfg_get(ph, "mu0",    default=0.056))
        mu_inf = float(cfg_get(ph, "mu_inf", default=0.0035))
        lam    = float(cfg_get(ph, "lam",    default=3.131))
        n      = float(cfg_get(ph, "n",      default=0.3568))
        Ld     = float(cfg_get(ph, "L",      default=0.04))
        U      = float(cfg_get(ph, "U",      default=0.05))
        x_lo   = float(cfg_get(ph, "x_lo",   default=-0.02)) / Rv
        x0     = float(cfg_get(ph, "x0",     default=0.0))   / Rv
        geom = BulgeGeometry(R_vessel=1.0, R_AAA=Ra / Rv, L=Ld / Rv,
                             x_lo=x_lo, x0=x0, L_AAA=La / Rv)
        return dict(label="AAA (Carreau)", newtonian=False,
                    Re=rho * U * Rv / mu_inf, x_lo=x_lo, x_hi=x_lo + Ld / Rv,
                    radius=lambda x: np.asarray(geom.radius_at(x), float),
                    beta=mu0 / mu_inf, Cu=lam * U / Rv, n=n,
                    dim=dict(mu_inf=mu_inf, U=U, R=Rv))

    raise ValueError(
        f"Unsupported problem '{prob}'. Expected one of: pipe_flow, "
        f"pipe_flow_rheology, AAA_flow, AAA_rheology."
    )


# ---------------------------------------------------------------------------
# Wall shear stress
# ---------------------------------------------------------------------------

def _mu_star(gdot, case):
    """Non-dimensional apparent viscosity (1 for Newtonian, Carreau otherwise)."""
    if case["newtonian"]:
        return jnp.ones_like(gdot)
    return 1.0 + (case["beta"] - 1.0) * (
        1.0 + (case["Cu"] * gdot) ** 2) ** ((case["n"] - 1.0) / 2.0)


def wall_shear_stress(model, params, xyz_wall, normals, case):
    """|tangential viscous traction| at wall points (non-dimensional)."""
    def _vel(p_in):
        return model.apply(params, p_in[None, :])[0, :3]

    J = jax.vmap(jax.jacfwd(_vel))(jnp.asarray(xyz_wall))     # (N, 3, 3)
    S = J + jnp.transpose(J, (0, 2, 1))                       # ∇u + ∇uᵀ
    gdot = jnp.sqrt(0.5 * jnp.sum(S * S, axis=(1, 2)) + 1e-12)
    mu   = _mu_star(gdot, case)

    nrm = jnp.asarray(normals)
    t   = (mu / case["Re"])[:, None] * jnp.einsum("nij,nj->ni", S, nrm)
    t_n = jnp.sum(t * nrm, axis=1, keepdims=True) * nrm       # normal part
    return np.array(jnp.linalg.norm(t - t_n, axis=1))         # tangential mag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict_steady(out_dir: str) -> dict:
    cfg_path = os.path.join(out_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} not found — run training first.")
    cfg  = load_config(cfg_path)
    case = _case_setup(cfg)
    x_lo, x_hi = case["x_lo"], case["x_hi"]
    radius     = case["radius"]

    # ── Model + checkpoint ────────────────────────────────────────────────────
    net_type = str(cfg_get(cfg.network, "type", default="mlp")).lower()
    net_cls  = {"mlp": MLP, "gated_mlp": GatedMLP}.get(net_type, MLP)
    model    = net_cls(layers=list(cfg.network.layers))
    params   = load_checkpoint(model, out_dir)
    print(f"[predict] {case['label']}   Re={case['Re']:.2f}   "
          f"x ∈ [{x_lo:.2f}, {x_hi:.2f}]")

    # ── Axial plane z = 0 ─────────────────────────────────────────────────────
    Nx, Ny = 400, 120
    xg = np.linspace(x_lo, x_hi, Nx)               # float64 → streamplot-safe
    R_of_x = radius(xg)
    R_max  = float(R_of_x.max())
    yg = np.linspace(-R_max, R_max, Ny)
    XX, YY = np.meshgrid(xg, yg)
    pts  = np.stack([XX.ravel(), YY.ravel(),
                     np.zeros(XX.size)], axis=1).astype(np.float32)
    pred = np.array(model.apply(params, jnp.asarray(pts)))    # (N, 4)
    U = pred[:, 0].reshape(Ny, Nx)
    V = pred[:, 1].reshape(Ny, Nx)
    W = pred[:, 2].reshape(Ny, Nx)
    P = pred[:, 3].reshape(Ny, Nx)
    outside = np.abs(YY) > R_of_x[None, :]
    for arr in (U, V, W, P):
        arr[outside] = np.nan

    # 1) u contour + streamlines (axial / streamwise direction)
    fig, ax = plt.subplots(figsize=(14, 3.6))
    cf = ax.contourf(xg, yg, U, levels=60, cmap="jet")
    plt.colorbar(cf, ax=ax, label="u (axial)")
    ax.streamplot(xg, yg, np.nan_to_num(U), np.nan_to_num(V),
                  density=(2.2, 0.6), color="k", linewidth=0.5, arrowsize=0.7)
    ax.plot(xg,  R_of_x, "k-", lw=1.5)
    ax.plot(xg, -R_of_x, "k-", lw=1.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-R_max, R_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("y")
    ax.set_title(f"{case['label']} — axial velocity & streamlines (z=0)")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "predict_axial_velocity.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) pressure contour on the same plane
    fig, ax = plt.subplots(figsize=(14, 3.6))
    cf = ax.contourf(xg, yg, P, levels=60, cmap="coolwarm")
    plt.colorbar(cf, ax=ax, label="p")
    ax.plot(xg,  R_of_x, "k-", lw=1.5)
    ax.plot(xg, -R_of_x, "k-", lw=1.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-R_max, R_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("y")
    ax.set_title(f"{case['label']} — pressure contour (z=0)")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "predict_axial_pressure.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Centreline + wall pressure lines ──────────────────────────────────────
    Nl   = 300
    x_l  = np.linspace(x_lo, x_hi, Nl).astype(np.float32)
    ctr  = np.stack([x_l, np.zeros(Nl, np.float32), np.zeros(Nl, np.float32)], axis=1)
    R_l  = radius(x_l).astype(np.float32)
    wallp = np.stack([x_l, R_l, np.zeros(Nl, np.float32)], axis=1)
    p_ctr  = np.array(model.apply(params, jnp.asarray(ctr))[:, 3])
    p_wall = np.array(model.apply(params, jnp.asarray(wallp))[:, 3])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_l, p_ctr,  "b-",  lw=1.8, label="centreline  p(x, 0, 0)")
    ax.plot(x_l, p_wall, "r--", lw=1.5, label="wall  p(x, R(x), 0)")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("p")
    ax.set_title(f"{case['label']} — pressure along the pipe")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "predict_pressure_line.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Wall shear stress along the wall (z=0, y=+R(x)) ───────────────────────
    # numeric dR/dx for the outward wall normal n ∝ (−R'(x), 1, 0)
    h    = 1e-4 * (x_hi - x_lo)
    dRdx = (radius(x_l + h) - radius(x_l - h)) / (2.0 * h)
    nrm  = np.stack([-dRdx, np.ones(Nl), np.zeros(Nl)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    wss  = wall_shear_stress(model, params, wallp, nrm.astype(np.float32), case)

    wss_Pa = None
    if case["dim"] is not None:
        dscale = case["dim"]["mu_inf"] * case["dim"]["U"] / case["dim"]["R"]
        wss_Pa = wss * dscale

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_l, wss, "b-", lw=1.8, label="WSS (non-dim)")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("τ_w  (non-dim)")
    if wss_Pa is not None:
        ax2 = ax.twinx()
        ax2.plot(x_l, wss_Pa, alpha=0.0)          # twin axis for the Pa scale
        ax2.set_ylabel("τ_w  [Pa]")
    ax.set_title(f"{case['label']} — wall shear stress")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(out_dir, "predict_wss.png")
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Full 3-D wall distribution: pressure + WSS at every wall point ────────
    # Parametrize the wall by (x, θ) so coverage is uniform around the
    # circumference, not just a single y-slice.  This gives a true wall map
    # of p(x, θ) and τ_w(x, θ) suitable for plotting or comparison with CFD.
    Nx3, Nth = 120, 64
    x_3d  = np.linspace(x_lo, x_hi, Nx3).astype(np.float32)
    th_3d = np.linspace(0.0, 2.0 * np.pi, Nth, endpoint=False).astype(np.float32)
    R_3d  = radius(x_3d).astype(np.float32)                # (Nx3,)
    XX3, TT3 = np.meshgrid(x_3d, th_3d, indexing="ij")     # (Nx3, Nth)
    RR3      = np.broadcast_to(R_3d[:, None], (Nx3, Nth))
    Y3       = RR3 * np.cos(TT3)
    Z3       = RR3 * np.sin(TT3)
    wall3d   = np.stack([XX3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    # Outward wall normal: ∂/∂x × ∂/∂θ on (x, R(x)cosθ, R(x)sinθ) gives
    #    n ∝ (-R'(x), cosθ, sinθ) / sqrt(R'² + 1)
    dRdx_3d = (radius(x_3d + h) - radius(x_3d - h)) / (2.0 * h)
    dRdx_full = np.broadcast_to(dRdx_3d[:, None], (Nx3, Nth))
    nrm3d = np.stack(
        [-dRdx_full.ravel(), np.cos(TT3).ravel(), np.sin(TT3).ravel()], axis=1)
    nrm3d /= np.linalg.norm(nrm3d, axis=1, keepdims=True)

    uvwp_wall = np.array(model.apply(
        params, jnp.asarray(wall3d, dtype=jnp.float32)))   # (N, 4)
    p_wall_3d = uvwp_wall[:, 3].reshape(Nx3, Nth)
    wss_wall_3d = wall_shear_stress(
        model, params, wall3d.astype(np.float32),
        nrm3d.astype(np.float32), case).reshape(Nx3, Nth)
    wss_wall_3d_Pa = (wss_wall_3d * (case["dim"]["mu_inf"] * case["dim"]["U"]
                                      / case["dim"]["R"])
                      if case["dim"] is not None else None)

    # ── 3-D interior volume on a structured (x, y, z) grid ────────────────────
    Nx_v, Ny_v, Nz_v = 80, 40, 40
    xg_v = np.linspace(x_lo, x_hi, Nx_v)
    yg_v = np.linspace(-R_max, R_max, Ny_v)
    zg_v = np.linspace(-R_max, R_max, Nz_v)
    XV, YV, ZV = np.meshgrid(xg_v, yg_v, zg_v, indexing="ij")
    R_at_xv = radius(xg_v)                                 # (Nx_v,)
    R_at_xv_full = np.broadcast_to(R_at_xv[:, None, None],
                                   (Nx_v, Ny_v, Nz_v))
    outside_v = YV ** 2 + ZV ** 2 > R_at_xv_full ** 2
    vol_pts = np.stack([XV.ravel(), YV.ravel(), ZV.ravel()],
                       axis=1).astype(np.float32)
    uvwp_vol = np.array(model.apply(params, jnp.asarray(vol_pts)))
    U_v = uvwp_vol[:, 0].reshape(Nx_v, Ny_v, Nz_v)
    V_v = uvwp_vol[:, 1].reshape(Nx_v, Ny_v, Nz_v)
    W_v = uvwp_vol[:, 2].reshape(Nx_v, Ny_v, Nz_v)
    P_v = uvwp_vol[:, 3].reshape(Nx_v, Ny_v, Nz_v)
    for arr in (U_v, V_v, W_v, P_v):
        arr[outside_v] = np.nan

    # ── Save the solution itself ──────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "predict_solution.npz")
    payload = dict(
        # 2-D axial-plane (z = 0) fields
        x=xg, y=yg, u=U, v=V, w=W, p=P,
        # 1-D wall line (y = +R(x), z = 0)  — back-compat
        wall_x=x_l, wall_R=R_l, wss=wss,
        # Centreline and wall pressure lines
        x_centreline=x_l, p_centreline=p_ctr, p_wall=p_wall,
        # Full 3-D WALL distribution  (Nx3, Nth) parametrised by (x, θ)
        wall3d_x=x_3d, wall3d_theta=th_3d,
        wall3d_xyz=wall3d.reshape(Nx3, Nth, 3),
        wall3d_p=p_wall_3d,
        wall3d_wss=wss_wall_3d,
        # Full 3-D INTERIOR volume on a structured grid
        vol_x=xg_v, vol_y=yg_v, vol_z=zg_v,
        vol_u=U_v, vol_v=V_v, vol_w=W_v, vol_p=P_v,
        # Metadata
        Re=case["Re"], label=case["label"],
    )
    if wss_Pa is not None:
        payload["wss_Pa"] = wss_Pa
    if wss_wall_3d_Pa is not None:
        payload["wall3d_wss_Pa"] = wss_wall_3d_Pa
    np.savez(npz_path, **payload)

    for f in (p1, p2, p3, p4, npz_path):
        print("Saved:", f)
    return {"npz": npz_path, "wss_max": float(np.max(wss))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Post-process a trained steady pipe/AAA run "
                    "(Newtonian or Carreau).")
    ap.add_argument("out_dir", help="training output directory "
                                    "(contains params.msgpack + config.yaml)")
    args = ap.parse_args()
    predict_steady(args.out_dir)
