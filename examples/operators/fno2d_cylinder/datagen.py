"""Chorin-projection incompressible Navier-Stokes reference solver for flow
past a cylinder — local to this example (not promoted to the shared library,
since no other example reuses it; same convention as e.g. the Aneurysm STL
ray-caster staying local to its own example).

Explicit projection method: an intermediate velocity is advanced with upwind
advection + central diffusion (no pressure term), a pressure Poisson equation
enforces incompressibility on that intermediate field via Jacobi iteration,
and the velocity is corrected by the pressure gradient. The cylinder is
imposed by a simple immersed-boundary mask: velocity is forced to zero at
every grid point inside the circle after every sub-step.
"""
from __future__ import annotations

import numpy as np


def make_obstacle_mask(Nx: int, Ny: int, dx: float, dy: float,
                       cx: float, cy: float, r: float) -> np.ndarray:
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def _poisson_solve(rhs, dx, dy, mask, n_iter=80):
    """Jacobi relaxation for ``lap(p) = rhs`` with Neumann walls/obstacle and
    a fixed reference pressure (p=0) at the outflow."""
    p = np.zeros_like(rhs)
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 * (dx2 + dy2)
    for _ in range(n_iter):
        p_xp = np.roll(p, -1, axis=0)
        p_xp[-1, :] = p_xp[-2, :]
        p_xm = np.roll(p, 1, axis=0)
        p_xm[0, :] = p_xm[1, :]
        p_yp = np.roll(p, -1, axis=1)
        p_yp[:, -1] = p_yp[:, -2]
        p_ym = np.roll(p, 1, axis=1)
        p_ym[:, 0] = p_ym[:, 1]

        p = ((p_xp + p_xm) * dy2 + (p_yp + p_ym) * dx2
            - rhs * dx2 * dy2) / denom
        p[mask] = 0.0
        p[-1, :] = 0.0     # outflow reference pressure
    return p


def solve_cylinder_flow(Re: float, T: float, Lx: float, Ly: float,
                        Nx: int, Ny: int, Nt: int, cx: float, cy: float,
                        r: float, U_in: float, seed: int,
                        poisson_iters: int = 80):
    """Explicit Chorin-projection solve on a channel with a circular obstacle.

    ``Re = U_in * (2r) / nu`` defines the kinematic viscosity.

    Returns ``(U, V, P, mask)`` — velocity/pressure snapshots each shaped
    ``(Nt + 1, Nx, Ny)``, and the ``(Nx, Ny)`` boolean obstacle mask.
    """
    dx, dy, dt = Lx / Nx, Ly / Ny, T / Nt
    nu = U_in * (2.0 * r) / Re

    cfl_adv = U_in * dt / min(dx, dy)
    cfl_diff = nu * dt / min(dx, dy) ** 2
    if cfl_adv > 0.4 or cfl_diff > 0.5:
        raise ValueError(
            f"CFL violated (adv={cfl_adv:.3f}>0.4 or diff={cfl_diff:.3f}>0.5); "
            f"increase Nt.")

    mask = make_obstacle_mask(Nx, Ny, dx, dy, cx, cy, r)

    def apply_bc(u, v):
        u[0, :] = U_in
        v[0, :] = 0.0                     # inflow
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]               # outflow (zero-gradient)
        u[:, 0] = 0.0
        v[:, 0] = 0.0                     # channel walls (no-slip)
        u[:, -1] = 0.0
        v[:, -1] = 0.0
        u[mask] = 0.0
        v[mask] = 0.0                     # obstacle (immersed-boundary mask)
        return u, v

    u = np.full((Nx, Ny), U_in, dtype=np.float64)
    v = np.zeros((Nx, Ny), dtype=np.float64)
    u, v = apply_bc(u, v)

    U = np.empty((Nt + 1, Nx, Ny), dtype=np.float32)
    V = np.empty((Nt + 1, Nx, Ny), dtype=np.float32)
    P = np.empty((Nt + 1, Nx, Ny), dtype=np.float32)
    p = np.zeros((Nx, Ny))
    U[0], V[0], P[0] = u, v, p

    for step in range(Nt):
        u_xp, u_xm = np.roll(u, -1, 0), np.roll(u, 1, 0)
        u_yp, u_ym = np.roll(u, -1, 1), np.roll(u, 1, 1)
        v_xp, v_xm = np.roll(v, -1, 0), np.roll(v, 1, 0)
        v_yp, v_ym = np.roll(v, -1, 1), np.roll(v, 1, 1)

        adv_u = (np.where(u >= 0, (u - u_xm) / dx, (u_xp - u) / dx) * u
                + np.where(v >= 0, (u - u_ym) / dy, (u_yp - u) / dy) * v)
        adv_v = (np.where(u >= 0, (v - v_xm) / dx, (v_xp - v) / dx) * u
                + np.where(v >= 0, (v - v_ym) / dy, (v_yp - v) / dy) * v)

        lap_u = (u_xp - 2.0 * u + u_xm) / dx ** 2 + (u_yp - 2.0 * u + u_ym) / dy ** 2
        lap_v = (v_xp - 2.0 * v + v_xm) / dx ** 2 + (v_yp - 2.0 * v + v_ym) / dy ** 2

        u_star = u + dt * (-adv_u + nu * lap_u)
        v_star = v + dt * (-adv_v + nu * lap_v)
        u_star, v_star = apply_bc(u_star, v_star)

        us_xp, us_xm = np.roll(u_star, -1, 0), np.roll(u_star, 1, 0)
        vs_yp, vs_ym = np.roll(v_star, -1, 1), np.roll(v_star, 1, 1)
        div = (us_xp - us_xm) / (2.0 * dx) + (vs_yp - vs_ym) / (2.0 * dy)
        rhs = div / dt

        p = _poisson_solve(rhs, dx, dy, mask, n_iter=poisson_iters)

        p_xp, p_xm = np.roll(p, -1, 0), np.roll(p, 1, 0)
        p_yp, p_ym = np.roll(p, -1, 1), np.roll(p, 1, 1)
        p_x = (p_xp - p_xm) / (2.0 * dx)
        p_y = (p_yp - p_ym) / (2.0 * dy)

        u = u_star - dt * p_x
        v = v_star - dt * p_y
        u, v = apply_bc(u, v)

        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            u = np.nan_to_num(u)
            v = np.nan_to_num(v)
            U[step + 1:], V[step + 1:], P[step + 1:] = 0.0, 0.0, 0.0
            break

        U[step + 1], V[step + 1], P[step + 1] = u, v, p

    return U, V, P, mask
