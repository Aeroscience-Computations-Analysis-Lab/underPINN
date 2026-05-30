"""2-D Steady Compressible Euler PDE (primitive-variable form).

State vector:  (ρ, u, v, p)  — density, x-velocity, y-velocity, pressure.

Governing equations (steady, inviscid, calorically perfect gas):
  1. Continuity:  (ρ u)_x + (ρ v)_y = 0
  2. Momentum-x:  ρ(u u_x + v u_y) + p_x = 0
  3. Momentum-y:  ρ(u v_x + v v_y) + p_y = 0
  4. Energy:      u p_x + v p_y + γ p (u_x + v_y) = 0
     (isentropic form: entropy is constant along streamlines for smooth flow)

Non-dimensionalisation used throughout:
  ρ_∞ = 1,  a_∞ = 1  →  p_∞ = 1/γ,  u_∞ = M_∞ · a_∞

The network maps  (x, y) → raw outputs  (f_ρ, f_u, f_v, f_p).
Physical variables are recovered via::

    ρ = softplus(f_ρ) + ε        # ensures ρ > 0
    u = f_u
    v = f_v
    p = softplus(f_p) + ε        # ensures p > 0

This guarantees thermodynamic admissibility throughout training.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE


# ---------------------------------------------------------------------------
# Helper — θ-β-M solver (bisection on the weak-shock branch)
# ---------------------------------------------------------------------------

def _theta_from_beta(beta: float, M1: float, gamma: float) -> float:
    """Flow deflection angle θ (rad) for given shock angle β (rad) and M1.

    Standard θ-β-M relation (Anderson, eq. 4.17):
      tan θ = 2 cot β · (M1² sin²β − 1) / (M1²(γ + cos 2β) + 2)
    """
    sb = math.sin(beta)
    cb = math.cos(beta)
    num = M1 ** 2 * sb ** 2 - 1.0
    den = M1 ** 2 * (gamma + math.cos(2.0 * beta)) + 2.0   # ← correct denominator
    if num <= 0.0 or den <= 0.0:
        return 0.0
    return math.atan(2.0 * cb / sb * num / den)


def _solve_beta_weak(M1: float, theta: float, gamma: float = 1.4,
                     n_scan: int = 360, n_bisect: int = 120) -> float:
    """Find weak-shock angle β (rad) by bisection on the θ-β-M curve.

    Scans [μ, 80°] in 0.5° steps to locate the weak-shock bracket,
    then refines with bisection.
    """
    mu = math.asin(1.0 / M1)          # Mach angle (lower physical bound)

    # ── Coarse scan to find beta_max (peak deflection angle) ─────────────
    beta_max = mu
    theta_max = 0.0
    b = mu + math.radians(0.5)
    while b < math.radians(80.0):
        t = _theta_from_beta(b, M1, gamma)
        if t > theta_max:
            theta_max = t
            beta_max = b
        b += math.radians(0.5)

    if theta > theta_max:
        raise ValueError(
            f"Deflection angle {math.degrees(theta):.1f}° exceeds maximum "
            f"({math.degrees(theta_max):.1f}°) for M={M1}, γ={gamma}."
        )

    # ── Bisect on weak-shock branch: β ∈ (μ, β_max) ──────────────────────
    lo, hi = mu + 1e-9, beta_max
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _theta_from_beta(mid, M1, gamma) < theta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# PDE class
# ---------------------------------------------------------------------------

class CompressibleEulerPDE(BasePDE):
    """2-D steady compressible Euler equations in primitive-variable form.

    Parameters
    ----------
    model : Flax module
        Network  (x, y) → (f_ρ, f_u, f_v, f_p)  with 2-in / 4-out.
    gamma : float
        Ratio of specific heats (default 1.4, air).
    eps : float
        Small constant added after softplus for numerical safety.
    """

    def __init__(self, model, gamma: float = 1.4, eps: float = 1e-6):
        self.model = model
        self.gamma = float(gamma)
        self.eps   = float(eps)

    # ------------------------------------------------------------------
    # Physical forward pass (applies positivity transform)
    # ------------------------------------------------------------------

    def apply(self, params, xy):
        """Return physical state (ρ, u, v, p) as an (N, 4) array."""
        raw = self.model.apply(params, xy)          # (N, 4)
        rho = jax.nn.softplus(raw[:, 0]) + self.eps
        u   = raw[:, 1]
        v   = raw[:, 2]
        p   = jax.nn.softplus(raw[:, 3]) + self.eps
        return jnp.stack([rho, u, v, p], axis=1)   # (N, 4)

    # ------------------------------------------------------------------
    # BasePDE interface
    # ------------------------------------------------------------------

    def residual(self, params, xy):
        """Compute PDE residuals at collocation points.

        Parameters
        ----------
        params : Flax parameter pytree.
        xy     : (N, 2) collocation coordinates.

        Returns
        -------
        cont, mom_x, mom_y, energy — each of shape (N,)
        """
        gamma = self.gamma
        eps   = self.eps

        # Per-point function for jacfwd: (2,) → (4,) physical vars
        def _phys(xy_i):
            raw = self.model.apply(params, xy_i[None, :])[0]   # (4,)
            rho = jax.nn.softplus(raw[0]) + eps
            u   = raw[1]
            v   = raw[2]
            p   = jax.nn.softplus(raw[3]) + eps
            return jnp.stack([rho, u, v, p])

        # Jacobian of physical vars w.r.t. (x, y): shape (N, 4, 2)
        J = jax.vmap(jax.jacfwd(_phys))(xy)

        rho_x, rho_y = J[:, 0, 0], J[:, 0, 1]
        u_x,   u_y   = J[:, 1, 0], J[:, 1, 1]
        v_x,   v_y   = J[:, 2, 0], J[:, 2, 1]
        p_x,   p_y   = J[:, 3, 0], J[:, 3, 1]

        # Physical values at collocation points
        pv  = self.apply(params, xy)
        rho = pv[:, 0];  u = pv[:, 1];  v = pv[:, 2];  p = pv[:, 3]

        # 1. Continuity: (ρu)_x + (ρv)_y = 0
        cont  = rho_x * u + rho * u_x + rho_y * v + rho * v_y

        # 2. Momentum-x: ρ(u u_x + v u_y) + p_x = 0
        mom_x = rho * (u * u_x + v * u_y) + p_x

        # 3. Momentum-y: ρ(u v_x + v v_y) + p_y = 0
        mom_y = rho * (u * v_x + v * v_y) + p_y

        # 4. Energy (isentropic): u p_x + v p_y + γ p (u_x + v_y) = 0
        energy = u * p_x + v * p_y + gamma * p * (u_x + v_y)

        # Return shape (N, 4): [cont, mom_x, mom_y, energy]
        return jnp.stack([cont, mom_x, mom_y, energy], axis=-1)

    # ------------------------------------------------------------------
    # Freestream conditions (non-dimensional)
    # ------------------------------------------------------------------

    def freestream(self, M_inf: float):
        """Return freestream primitive variables (ρ_∞, u_∞, v_∞, p_∞).

        Non-dimensionalised with ρ_∞ = 1, a_∞ = 1:
          p_∞ = 1/γ,   u_∞ = M_∞,   v_∞ = 0
        """
        return (
            1.0,            # ρ_∞
            float(M_inf),   # u_∞  (flow aligned with +x)
            0.0,            # v_∞
            1.0 / self.gamma,  # p_∞
        )

    # ------------------------------------------------------------------
    # Oblique-shock analytical solution
    # ------------------------------------------------------------------

    def oblique_shock(self, M1: float, theta_deg: float):
        """Compute weak oblique-shock post-shock state analytically.

        Parameters
        ----------
        M1        : Upstream (freestream) Mach number.
        theta_deg : Wedge half-angle / ramp angle in degrees.

        Returns
        -------
        dict with keys:
          beta_deg  — shock wave angle from horizontal (degrees)
          M2        — post-shock Mach number
          rho2      — post-shock density   (non-dim, ρ_∞ = 1)
          u2, v2    — post-shock velocity components
          p2        — post-shock pressure  (non-dim, p_∞ = 1/γ)
        """
        gamma = self.gamma
        theta = math.radians(theta_deg)

        beta = _solve_beta_weak(M1, theta, gamma)

        # Normal component of upstream Mach
        Mn1 = M1 * math.sin(beta)

        # Normal shock relations (applied to normal component)
        rho_ratio = ((gamma + 1.0) * Mn1 ** 2
                     / ((gamma - 1.0) * Mn1 ** 2 + 2.0))
        p_ratio   = 1.0 + 2.0 * gamma * (Mn1 ** 2 - 1.0) / (gamma + 1.0)

        # Post-shock normal Mach
        Mn2_sq = (Mn1 ** 2 + 2.0 / (gamma - 1.0)) / (
                  2.0 * gamma * Mn1 ** 2 / (gamma - 1.0) - 1.0)
        Mn2 = math.sqrt(max(Mn2_sq, 0.0))

        # Post-shock total Mach (flow direction is at angle θ from x-axis)
        beta2 = beta - theta                     # angle between post-shock flow and shock
        M2    = Mn2 / math.sin(max(beta2, 1e-9))

        # Post-shock state in freestream non-dim units
        # (ρ_∞ = 1, a_∞ = 1, p_∞ = 1/γ)
        rho2 = float(rho_ratio)                  # ρ2/ρ_∞
        p2   = float(p_ratio) / gamma            # p2 in units where p_∞ = 1/γ

        # Speed of sound post-shock:  a2 = sqrt(γ p2/ρ2)  (in a_∞ units)
        a2 = math.sqrt(gamma * p2 / rho2)
        V2 = float(M2) * a2                      # total speed post-shock

        # Post-shock velocity: flow is deflected upward by θ (parallel to ramp)
        u2 =  V2 * math.cos(theta)
        v2 =  V2 * math.sin(theta)               # +y for a bottom ramp going up

        return {
            "beta_deg": math.degrees(beta),
            "M2":       M2,
            "rho2":     rho2,
            "u2":       u2,
            "v2":       v2,
            "p2":       p2,
        }
