"""Reference-solution generators for neural-operator (PINO/DeepONet) training.

Neural operators learn a map between *functions* (e.g. initial condition →
solution), so training needs many reference trajectories rather than a single
solve.  This module holds the finite-difference solvers and random-field
generators shared across the Burgers operator examples (1-D periodic, 1-D
Dirichlet, 2-D periodic) plus the Cole–Hopf exact solution used to validate
against ground truth.

Random initial conditions are a small sum of 2–4 low-integer sine modes
(k ∈ {1,2,3}) rather than a full Gaussian random field — this keeps every
sampled IC smooth and, for the Dirichlet case, exactly zero at the walls.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1-D viscous Burgers:  u_t + u u_x = nu u_xx
# ---------------------------------------------------------------------------

def random_ic_1d(rng: np.random.default_rng, Nx: int, Lx: float,
                 periodic: bool) -> np.ndarray:
    """A random smooth IC: 2-4 low sine modes, k in {1,2,3}.

    Periodic domains get a random phase per mode; Dirichlet domains cannot
    (a phase would break ``u(walls) = 0``), so ``sin(k*pi*x)`` modes are used
    verbatim — every member of this family vanishes at ``x = ±Lx/2``.
    """
    if periodic:
        x = np.linspace(0.0, Lx, Nx, endpoint=False)
        n_modes = rng.integers(2, 5)
        u0 = np.zeros(Nx)
        for _ in range(n_modes):
            k = rng.integers(1, 4)
            amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
            phi = rng.uniform(0.0, 2.0 * np.pi)
            u0 += amp * np.sin(k * x / Lx * 2.0 * np.pi + phi)
        return u0.astype(np.float32)

    # Dirichlet: x in [-Lx/2, Lx/2], u(walls) = 0.
    x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx)
    n_modes = rng.integers(2, 5)
    u0 = np.zeros(Nx)
    for _ in range(n_modes):
        k = rng.integers(1, 4)
        amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        u0 += amp * np.sin(k * np.pi * x / (Lx / 2.0))
    peak = np.max(np.abs(u0)) + 1e-12
    u0 = u0 * (rng.uniform(0.5, 1.0) / peak)
    u0[0] = u0[-1] = 0.0
    return u0.astype(np.float32)


def solve_burgers1d(
    nu: np.ndarray, T: float, Lx: float, Nx: int, Nt: int,
    n_traj: int, periodic: bool, seed: int, u_max_cfl: float = 2.0,
) -> np.ndarray:
    """Batch-solve 1-D Burgers for *n_traj* random ICs, one nu per trajectory.

    Explicit RK2 (Heun) march; upwind advection on ``sign(u)``, central
    diffusion.  Periodic domains use ``np.roll`` for derivatives; Dirichlet
    domains zero-pad ghost cells and clamp the walls to 0 every step.

    Parameters
    ----------
    nu     : (n_traj,) viscosity per trajectory.
    Returns
    -------
    (n_traj, Nt+1, Nx) float32 snapshots, index 0 = the initial condition.
    """
    dx = Lx / Nx if periodic else Lx / (Nx - 1)
    dt = T / Nt
    cfl_adv = u_max_cfl * dt / dx
    cfl_diff = float(np.max(nu)) * dt / dx ** 2
    if cfl_adv > 0.4 or cfl_diff > 0.5:
        n_req = int(np.ceil(T / (0.4 * dx / u_max_cfl))) + 1
        raise ValueError(
            f"CFL violated (adv={cfl_adv:.3f}>0.4 or diff={cfl_diff:.3f}>0.5); "
            f"increase Nt to at least {n_req}.")

    rng = np.random.default_rng(seed)
    out = np.empty((n_traj, Nt + 1, Nx), dtype=np.float32)

    def _rhs(u, nu_i):
        if periodic:
            u_xp, u_xm = np.roll(u, -1), np.roll(u, 1)
        else:
            padded = np.concatenate([[0.0], u, [0.0]])
            u_xp, u_xm = padded[2:], padded[:-2]
        adv = np.where(u >= 0.0, (u - u_xm) / dx, (u_xp - u) / dx)
        diff = (u_xp - 2.0 * u + u_xm) / dx ** 2
        return -u * adv + nu_i * diff

    for i in range(n_traj):
        u = random_ic_1d(rng, Nx, Lx, periodic)
        nu_i = float(nu[i])
        traj = np.empty((Nt + 1, Nx), dtype=np.float32)
        traj[0] = u
        for step in range(Nt):
            k1 = _rhs(u, nu_i)
            u1 = u + dt * k1
            if not periodic:
                u1[0] = u1[-1] = 0.0
            k2 = _rhs(u1, nu_i)
            u = u + 0.5 * dt * (k1 + k2)
            if not periodic:
                u[0] = u[-1] = 0.0
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                traj[step + 1:] = 0.0
                break
            traj[step + 1] = u
        out[i] = traj
    return out


def burgers1d_exact(x, t, nu: float, u0_mode: int = 1, n_quad: int = 80):
    """Exact viscous-Burgers solution for ``u(x,0) = -sin(u0_mode*pi*x)`` via
    Cole-Hopf + Gauss-Hermite quadrature.

    Valid on the Dirichlet domain ``x in [-1, 1]`` with the k=1 IC (the
    classic textbook case); returns an ``(len(x), len(t))`` array.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y, w = np.polynomial.hermite.hermgauss(n_quad)
    u = np.empty((x.size, t.size), dtype=np.float64)
    for j, tj in enumerate(t):
        if tj <= 0.0:
            u[:, j] = -np.sin(u0_mode * np.pi * x)
            continue
        s = 2.0 * np.sqrt(nu * tj)
        arg = x[:, None] - s * y[None, :]
        e = -np.cos(u0_mode * np.pi * arg) / (2.0 * np.pi * nu)
        e -= e.max(axis=1, keepdims=True)
        h = np.exp(e)
        num = (h * (w * y)[None, :]).sum(axis=1)
        den = (h * w[None, :]).sum(axis=1)
        u[:, j] = 2.0 * np.sqrt(nu / tj) * num / den
    return u


# ---------------------------------------------------------------------------
# 2-D viscous Burgers:  u_t + u u_x + u u_y = nu (u_xx + u_yy)   (periodic)
# ---------------------------------------------------------------------------

def random_ic_2d(rng: np.random.default_rng, Nx: int, Lx: float, Ly: float
                ) -> np.ndarray:
    """Random smooth periodic 2-D IC: 2-4 sine modes with k_x, k_y in {1,2,3}."""
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Nx, endpoint=False)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    n_modes = rng.integers(2, 5)
    u0 = np.zeros((Nx, Nx))
    for _ in range(n_modes):
        kx, ky = rng.integers(1, 4), rng.integers(1, 4)
        amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        phi = rng.uniform(0.0, 2.0 * np.pi)
        u0 += amp * np.sin(kx * XX / Lx * 2.0 * np.pi
                          + ky * YY / Ly * 2.0 * np.pi + phi)
    return u0.astype(np.float32)


def solve_burgers2d(
    nu: np.ndarray, T: float, Lx: float, Ly: float, Nx: int, Nt: int,
    n_traj: int, seed: int, scheme: str = "central", u_max_cfl: float = 2.0,
) -> np.ndarray:
    """Batch-solve 2-D periodic Burgers for *n_traj* random ICs.

    Explicit RK2 (Heun); ``scheme="central"`` uses ``sign(u)`` upwind
    advection (matches the FNO-PINO reference stencil); ``scheme="upwind"``
    is an alias kept for API clarity (both variants use upwind advection —
    the "central"/"upwind" naming refers to which stencil the *residual loss*
    uses downstream, not this generator).

    Returns
    -------
    (n_traj, Nt+1, Nx, Nx) float32 snapshots, index 0 = the initial condition.
    """
    dx = Lx / Nx
    dy = Ly / Nx
    dt = T / Nt
    cfl_adv = u_max_cfl * dt / dx
    cfl_diff = float(np.max(nu)) * dt / dx ** 2
    if cfl_adv > 0.4 or cfl_diff > 0.5:
        raise ValueError(f"CFL violated: adv={cfl_adv:.3f} (>0.4?), "
                         f"diff={cfl_diff:.3f} (>0.5?). Increase Nt.")

    rng = np.random.default_rng(seed)
    out = np.empty((n_traj, Nt + 1, Nx, Nx), dtype=np.float32)

    def _rhs(u, nu_i):
        u_xp, u_xm = np.roll(u, -1, axis=0), np.roll(u, 1, axis=0)
        u_yp, u_ym = np.roll(u, -1, axis=1), np.roll(u, 1, axis=1)
        adv_x = np.where(u >= 0.0, (u - u_xm) / dx, (u_xp - u) / dx)
        adv_y = np.where(u >= 0.0, (u - u_ym) / dy, (u_yp - u) / dy)
        lap = (u_xp + u_xm + u_yp + u_ym - 4.0 * u) / dx ** 2
        return -u * (adv_x + adv_y) + nu_i * lap

    for i in range(n_traj):
        u = random_ic_2d(rng, Nx, Lx, Ly)
        nu_i = float(nu[i])
        traj = np.empty((Nt + 1, Nx, Nx), dtype=np.float32)
        traj[0] = u
        for step in range(Nt):
            k1 = _rhs(u, nu_i)
            k2 = _rhs(u + dt * k1, nu_i)
            u = u + 0.5 * dt * (k1 + k2)
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                traj[step + 1:] = 0.0
                break
            traj[step + 1] = u
        out[i] = traj
    return out


# ---------------------------------------------------------------------------
# Operator (window, target) pair builders
# ---------------------------------------------------------------------------

def build_pairs_1d(snapshots: np.ndarray, nu: np.ndarray, prev_steps: int,
                   pred_steps: int, pairs_per_traj: int, early_frac: float,
                   seed: int):
    """Slice ``(n_traj, Nt+1, Nx)`` snapshots into FNO1D (input, target) pairs.

    input  : (M, Nx, prev_steps + 1) — prev_steps history frames + a constant
             nu channel.
    target : (M, Nx, 1) — the frame ``pred_steps`` after the last history frame.
    """
    n_traj, n_frames, Nx = snapshots.shape
    last_start = int(early_frac * (n_frames - prev_steps - pred_steps))
    last_start = max(last_start, 1)
    rng = np.random.default_rng(seed)

    inputs, targets = [], []
    for i in range(n_traj):
        starts = rng.integers(0, last_start, size=pairs_per_traj)
        for s in starts:
            hist = snapshots[i, s:s + prev_steps]            # (prev_steps, Nx)
            tgt = snapshots[i, s + prev_steps + pred_steps - 1]  # (Nx,)
            nu_chan = np.full((1, Nx), nu[i], dtype=np.float32)
            inp = np.concatenate([hist, nu_chan], axis=0).T   # (Nx, prev_steps+1)
            inputs.append(inp)
            targets.append(tgt[:, None])
    return (np.stack(inputs).astype(np.float32),
            np.stack(targets).astype(np.float32))


def build_pairs_2d(snapshots: np.ndarray, nu: np.ndarray, prev_steps: int,
                   pred_steps: int, pairs_per_traj: int, early_frac: float,
                   seed: int):
    """Slice ``(n_traj, Nt+1, Nx, Nx)`` snapshots into FNO2D (input, target) pairs.

    input  : (M, Nx, Nx, prev_steps + 1)
    target : (M, Nx, Nx, 1)
    """
    n_traj, n_frames, Nx, _ = snapshots.shape
    last_start = int(early_frac * (n_frames - prev_steps - pred_steps))
    last_start = max(last_start, 1)
    rng = np.random.default_rng(seed)

    inputs, targets = [], []
    for i in range(n_traj):
        starts = rng.integers(0, last_start, size=pairs_per_traj)
        for s in starts:
            hist = snapshots[i, s:s + prev_steps]                # (prev, Nx, Nx)
            tgt = snapshots[i, s + prev_steps + pred_steps - 1]   # (Nx, Nx)
            hist = np.moveaxis(hist, 0, -1)                       # (Nx, Nx, prev)
            nu_chan = np.full((Nx, Nx, 1), nu[i], dtype=np.float32)
            inputs.append(np.concatenate([hist, nu_chan], axis=-1))
            targets.append(tgt[..., None])
    return (np.stack(inputs).astype(np.float32),
            np.stack(targets).astype(np.float32))
