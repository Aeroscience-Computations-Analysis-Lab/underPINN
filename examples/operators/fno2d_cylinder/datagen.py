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

import json
import pathlib
import subprocess
import sys
import tempfile
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


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
        # Top/bottom: SLIP (v=0, du/dy=0), not no-slip. Verified against
        # Github/flowPastCylinder/final_flow/datagen.py's own solver (its
        # docstring: "top / bottom: slip (v = 0, du/dy = 0)"), run
        # unmodified at its own Re=100 config -- it sheds cleanly there
        # (flat windowed std, no residual transient, large amplitude)
        # while our channel with real no-slip walls needed Re=220-290/T=200
        # for comparable shedding. No-slip walls add both a genuine
        # viscous boundary layer and a 25%-blockage confinement effect,
        # both of which raise the effective critical Re for shedding well
        # above the textbook unbounded-cylinder value (~47) that slip walls
        # (no wall drag) stay close to.
        u[:, 0] = u[:, 1]
        v[:, 0] = 0.0                     # bottom: slip
        u[:, -1] = u[:, -2]
        v[:, -1] = 0.0                     # top: slip
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


# ---------------------------------------------------------------------------
# JAX/GPU-accelerated version of the exact same algorithm above.
# ---------------------------------------------------------------------------
# Measured ~7x faster per step than the NumPy version above at the same
# 128x128 resolution (this project's own JAX-jitted reference solver in
# Github/flowPastCylinder/final_flow/datagen.py showed a similar gap).
# Data generation (not training) dominates this example's wall-clock time
# (~10min for 8 trajectories vs ~2.5min training), so this is the single
# biggest remaining optimization opportunity in the whole example.
#
# IMPORTANT -- dtype: the NumPy solver above runs in float64. This flow is
# a genuine Hopf-bifurcation limit cycle (see config.yaml's comments) --
# its defining property is *amplifying* small perturbations, so silently
# running the same algorithm in JAX's float32 default would not just lose
# precision, it could plausibly change which trajectory comes out (a
# different, if still physically valid, point on/near the limit cycle).
# ``jax.config.update("jax_enable_x64", True)`` fixes this, but it is a
# GLOBAL, process-wide JAX setting -- enabling it in the same process that
# later trains the FNO (which this whole example's accuracy work has
# validated at float32 throughout) risks silently changing training
# precision too. These functions therefore assume the CALLER has already
# arranged for x64 to be enabled in an ISOLATED process (see
# ``run_datagen_jax_subprocess.py``, which sets it via env var in a
# subprocess and is the only intended caller) -- do not call
# ``solve_cylinder_flow_jax`` directly from the main training process.


def _apply_bc_jax(u, v, mask, U_in):
    u = u.at[0, :].set(U_in)
    v = v.at[0, :].set(0.0)                    # inflow
    u = u.at[-1, :].set(u[-2, :])
    v = v.at[-1, :].set(v[-2, :])               # outflow (zero-gradient)
    u = u.at[:, 0].set(u[:, 1])
    v = v.at[:, 0].set(0.0)                     # bottom: slip
    u = u.at[:, -1].set(u[:, -2])
    v = v.at[:, -1].set(0.0)                     # top: slip
    u = jnp.where(mask, 0.0, u)
    v = jnp.where(mask, 0.0, v)                  # obstacle
    return u, v


def _poisson_solve_jax(rhs, dx, dy, mask, n_iter):
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 * (dx2 + dy2)

    def body(_, p):
        p_xp = jnp.roll(p, -1, axis=0)
        p_xp = p_xp.at[-1, :].set(p_xp[-2, :])
        p_xm = jnp.roll(p, 1, axis=0)
        p_xm = p_xm.at[0, :].set(p_xm[1, :])
        p_yp = jnp.roll(p, -1, axis=1)
        p_yp = p_yp.at[:, -1].set(p_yp[:, -2])
        p_ym = jnp.roll(p, 1, axis=1)
        p_ym = p_ym.at[:, 0].set(p_ym[:, 1])
        p_new = ((p_xp + p_xm) * dy2 + (p_yp + p_ym) * dx2
                - rhs * dx2 * dy2) / denom
        p_new = jnp.where(mask, 0.0, p_new)
        p_new = p_new.at[-1, :].set(0.0)         # outflow reference pressure
        return p_new

    return jax.lax.fori_loop(0, n_iter, body, jnp.zeros_like(rhs))


@partial(jax.jit, static_argnames=("Nx", "Ny", "Nt", "poisson_iters"))
def _solve_core_jax(Re, T, Lx, Ly, Nx, Ny, Nt, cx, cy, r, U_in, poisson_iters):
    dx, dy, dt = Lx / Nx, Ly / Ny, T / Nt
    nu = U_in * (2.0 * r) / Re

    x = jnp.arange(Nx) * dx
    y = jnp.arange(Ny) * dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

    def apply_bc(u, v):
        return _apply_bc_jax(u, v, mask, U_in)

    u0 = jnp.full((Nx, Ny), U_in, dtype=jnp.float64)
    v0 = jnp.zeros((Nx, Ny), dtype=jnp.float64)
    u0, v0 = apply_bc(u0, v0)
    p0 = jnp.zeros((Nx, Ny), dtype=jnp.float64)

    def step(carry, _):
        u, v = carry
        u_xp, u_xm = jnp.roll(u, -1, 0), jnp.roll(u, 1, 0)
        u_yp, u_ym = jnp.roll(u, -1, 1), jnp.roll(u, 1, 1)
        v_xp, v_xm = jnp.roll(v, -1, 0), jnp.roll(v, 1, 0)
        v_yp, v_ym = jnp.roll(v, -1, 1), jnp.roll(v, 1, 1)

        adv_u = (jnp.where(u >= 0, (u - u_xm) / dx, (u_xp - u) / dx) * u
                + jnp.where(v >= 0, (u - u_ym) / dy, (u_yp - u) / dy) * v)
        adv_v = (jnp.where(u >= 0, (v - v_xm) / dx, (v_xp - v) / dx) * u
                + jnp.where(v >= 0, (v - v_ym) / dy, (v_yp - v) / dy) * v)

        lap_u = (u_xp - 2.0 * u + u_xm) / dx ** 2 + (u_yp - 2.0 * u + u_ym) / dy ** 2
        lap_v = (v_xp - 2.0 * v + v_xm) / dx ** 2 + (v_yp - 2.0 * v + v_ym) / dy ** 2

        u_star = u + dt * (-adv_u + nu * lap_u)
        v_star = v + dt * (-adv_v + nu * lap_v)
        u_star, v_star = apply_bc(u_star, v_star)

        us_xp, us_xm = jnp.roll(u_star, -1, 0), jnp.roll(u_star, 1, 0)
        vs_yp, vs_ym = jnp.roll(v_star, -1, 1), jnp.roll(v_star, 1, 1)
        div = (us_xp - us_xm) / (2.0 * dx) + (vs_yp - vs_ym) / (2.0 * dy)
        rhs = div / dt

        p = _poisson_solve_jax(rhs, dx, dy, mask, poisson_iters)

        p_xp, p_xm = jnp.roll(p, -1, 0), jnp.roll(p, 1, 0)
        p_yp, p_ym = jnp.roll(p, -1, 1), jnp.roll(p, 1, 1)
        p_x = (p_xp - p_xm) / (2.0 * dx)
        p_y = (p_yp - p_ym) / (2.0 * dy)

        u_new = u_star - dt * p_x
        v_new = v_star - dt * p_y
        u_new, v_new = apply_bc(u_new, v_new)

        return (u_new, v_new), (u_new, v_new, p)

    (_, _), (U_traj, V_traj, P_traj) = jax.lax.scan(step, (u0, v0), None, length=Nt)
    U_full = jnp.concatenate([u0[None], U_traj], axis=0)
    V_full = jnp.concatenate([v0[None], V_traj], axis=0)
    P_full = jnp.concatenate([p0[None], P_traj], axis=0)
    return U_full, V_full, P_full, mask


def solve_cylinder_flow_jax(Re: float, T: float, Lx: float, Ly: float,
                            Nx: int, Ny: int, Nt: int, cx: float, cy: float,
                            r: float, U_in: float, seed: int,
                            poisson_iters: int = 80):
    """JAX/GPU version of :func:`solve_cylinder_flow` -- identical algorithm,
    same signature and return format (``(U, V, P, mask)``, float32 arrays
    shaped ``(Nt+1, Nx, Ny)``). ``seed`` is accepted for signature parity
    only (unused, same as the NumPy version -- neither solver is stochastic).

    Requires the CALLING PROCESS to already have
    ``jax.config.update("jax_enable_x64", True)`` set -- see the module-
    level note above. Not NaN-safe mid-trajectory the way the NumPy version
    is (can't dynamically shorten a ``lax.scan``); instead, any non-finite
    values are detected and everything from the first bad step onward is
    zeroed out afterward, matching the NumPy version's end result exactly,
    just computed post-hoc rather than via an early Python break.
    """
    del seed
    dx, dy, dt = Lx / Nx, Ly / Ny, T / Nt
    nu = U_in * (2.0 * r) / Re
    cfl_adv = U_in * dt / min(dx, dy)
    cfl_diff = nu * dt / min(dx, dy) ** 2
    if cfl_adv > 0.4 or cfl_diff > 0.5:
        raise ValueError(
            f"CFL violated (adv={cfl_adv:.3f}>0.4 or diff={cfl_diff:.3f}>0.5); "
            f"increase Nt.")

    U, V, P, mask = _solve_core_jax(Re, T, Lx, Ly, Nx, Ny, Nt, cx, cy, r, U_in,
                                    poisson_iters)
    U, V, P = np.asarray(U), np.asarray(V), np.asarray(P)
    finite = np.all(np.isfinite(U.reshape(len(U), -1)), axis=1) & \
            np.all(np.isfinite(V.reshape(len(V), -1)), axis=1)
    if not finite.all():
        first_bad = int(np.argmax(~finite))
        U[first_bad:] = 0.0
        V[first_bad:] = 0.0
        P[first_bad:] = 0.0
    return U.astype(np.float32), V.astype(np.float32), P.astype(np.float32), \
        np.asarray(mask)


def solve_cylinder_flow_jax_isolated(Re: float, T: float, Lx: float, Ly: float,
                                     Nx: int, Ny: int, Nt: int, cx: float,
                                     cy: float, r: float, U_in: float,
                                     seed: int, poisson_iters: int = 80):
    """Drop-in, process-isolated replacement for :func:`solve_cylinder_flow`:
    same signature and return format, ~7x faster (JAX/GPU vs NumPy/CPU at
    128x128), by shelling out to ``run_datagen_jax_subprocess.py`` -- a
    dedicated subprocess with ``jax_enable_x64`` set only there, so the
    calling (training) process's float32 JAX config is never touched. See
    that script's and ``solve_cylinder_flow_jax``'s docstrings for why this
    isolation matters for this specific (dynamically unstable) flow.

    Safe to call repeatedly from the main process/loop over multiple Re
    values -- each call is a fresh subprocess (a few seconds of JAX/XLA
    startup + JIT-compile overhead per call, dominated by the ~10-60x
    faster solve itself for any non-trivial Nt).
    """
    worker = pathlib.Path(__file__).parent / "run_datagen_jax_subprocess.py"
    params = dict(Re=Re, T=T, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Nt=Nt, cx=cx, cy=cy,
                 r=r, U_in=U_in, seed=seed, poisson_iters=poisson_iters)
    with tempfile.TemporaryDirectory() as td:
        params_path = pathlib.Path(td) / "params.json"
        out_path = pathlib.Path(td) / "out.npz"
        with open(params_path, "w") as f:
            json.dump(params, f)
        result = subprocess.run(
            [sys.executable, str(worker), str(params_path), str(out_path)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"JAX datagen subprocess failed:\n{result.stdout}\n{result.stderr}")
        data = np.load(out_path)
        return data["U"], data["V"], data["P"], data["mask"]
