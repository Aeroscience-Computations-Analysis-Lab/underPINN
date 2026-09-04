"""2-D Steady Compressible Euler PDE (conservative flux-divergence form).

The network outputs primitive variables (ρ, u, v, p), but the PDE residual is
evaluated in **conservative form** on the conserved variables
U = (ρ, ρu, ρv, E), which is the physically correct shock-capturing form:

  ∂F/∂x + ∂G/∂y = 0
    F = (ρu,  ρu²+p,  ρuv,    (E+p)u)        x-flux
    G = (ρv,  ρuv,    ρv²+p,  (E+p)v)        y-flux
    E = p/(γ−1) + ½ρ(u²+v²)                  total energy (perfect gas)

  1. Mass:      (ρu)_x + (ρv)_y = 0
  2. Momentum-x:(ρu²+p)_x + (ρuv)_y = 0
  3. Momentum-y:(ρuv)_x + (ρv²+p)_y = 0
  4. Energy:    ((E+p)u)_x + ((E+p)v)_y = 0

Optional global artificial viscosity subtracts ε ∇²U (Laplacian of the
conserved variables) from each equation to smooth shocks.

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
    art_visc : float
        Global artificial-viscosity coefficient ε (default 0.0 = pure Euler).
        When > 0 a Laplacian dissipation term ``−ε ∇²q`` is added to every
        equation (q = ρ, u, v, p respectively).  This smooths discontinuities
        (shocks) and stabilises training, at the cost of slightly smeared
        shock fronts — the standard artificial-viscosity regularisation.
    """

    def __init__(self, model, gamma: float = 1.4, eps: float = 1e-6,
                 art_visc: float = 0.0):
        self.model    = model
        self.gamma    = float(gamma)
        self.eps      = float(eps)
        self.art_visc = float(art_visc)

    # ------------------------------------------------------------------
    # Trainable-viscosity support
    # ------------------------------------------------------------------
    # The optimisable state may be either the plain network params, or a
    # combined pytree  {"net": net_params, "log_av": raw_scalar}  in which the
    # artificial viscosity is *learned* as  ε = softplus(log_av) ≥ 0.

    @staticmethod
    def _is_combined(params) -> bool:
        try:
            return ("net" in params) and ("log_av" in params)
        except TypeError:
            return False

    def _net(self, params):
        """Extract the network parameters from *params* (combined or plain)."""
        return params["net"] if self._is_combined(params) else params

    def viscosity(self, params=None):
        """Current artificial-viscosity coefficient ε.

        Returns the learned ``softplus(log_av)`` if *params* is the combined
        trainable pytree, otherwise the fixed ``art_visc`` float.
        """
        if params is not None and self._is_combined(params):
            return float(jax.nn.softplus(params["log_av"]))
        return self.art_visc

    @staticmethod
    def inverse_softplus(y: float) -> float:
        """Raw value x such that softplus(x) ≈ y  (for initialising log_av)."""
        return float(math.log(math.expm1(max(y, 1e-12))))

    # ------------------------------------------------------------------
    # Physical forward pass (applies positivity transform)
    # ------------------------------------------------------------------

    def apply(self, params, xy):
        """Return physical state (ρ, u, v, p) as an (N, 4) array."""
        raw = self.model.apply(self._net(params), xy)   # (N, 4)
        rho = jax.nn.softplus(raw[:, 0]) + self.eps
        u   = raw[:, 1]
        v   = raw[:, 2]
        p   = jax.nn.softplus(raw[:, 3]) + self.eps
        return jnp.stack([rho, u, v, p], axis=1)   # (N, 4)

    # ------------------------------------------------------------------
    # BasePDE interface
    # ------------------------------------------------------------------

    def residual(self, params, xy):
        """Compute PDE residuals in **conservative form** at collocation points.

        The steady 2-D Euler system is written as a flux divergence of the
        conserved variables  U = (ρ, ρu, ρv, E):

            ∂F/∂x + ∂G/∂y = 0

          F = (ρu,  ρu²+p,  ρuv,      (E+p)u)        x-flux
          G = (ρv,  ρuv,    ρv²+p,    (E+p)v)        y-flux
          E = p/(γ−1) + ½ρ(u²+v²)                    total energy

        This is the correct shock-capturing form (entropy may rise across a
        shock, unlike the isentropic primitive form).  Global artificial
        viscosity, when enabled, subtracts ε ∇²U from each equation — i.e. the
        Laplacian of the **conserved** variables.

        Parameters
        ----------
        params : Flax parameter pytree.
        xy     : (N, 2) collocation coordinates.

        Returns
        -------
        (N, 4) residual array — [mass, mom_x, mom_y, energy].
        """
        gamma = self.gamma
        eps   = self.eps
        net   = self._net(params)

        # Primitive variables (ρ, u, v, p) at a single point.
        def _prim(xy_i):
            raw = self.model.apply(net, xy_i[None, :])[0]   # (4,)
            rho = jax.nn.softplus(raw[0]) + eps
            u   = raw[1]
            v   = raw[2]
            p   = jax.nn.softplus(raw[3]) + eps
            return rho, u, v, p

        # Conserved variables  U = (ρ, ρu, ρv, E)  at a single point.
        def _cons(xy_i):
            rho, u, v, p = _prim(xy_i)
            E = p / (gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
            return jnp.stack([rho, rho * u, rho * v, E])

        # Flux pair stacked as (4, 2):  column 0 = F (x-flux), column 1 = G.
        def _flux(xy_i):
            rho, u, v, p = _prim(xy_i)
            E = p / (gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
            F = jnp.stack([rho * u, rho * u * u + p, rho * u * v, (E + p) * u])
            G = jnp.stack([rho * v, rho * u * v, rho * v * v + p, (E + p) * v])
            return jnp.stack([F, G], axis=1)               # (4, 2)

        # d(F,G)/d(x,y): shape (N, 4, 2, 2).  ∂F_k/∂x = J[k,0,0], ∂G_k/∂y = J[k,1,1].
        J    = jax.vmap(jax.jacfwd(_flux))(xy)
        res  = J[:, :, 0, 0] + J[:, :, 1, 1]               # (N, 4) flux divergence

        # ── Global artificial viscosity:  − ε ∇²U  (Laplacian of conserved U) ──
        # ε is either a learned softplus(log_av) (combined params, always on)
        # or the fixed float ``art_visc`` (only computed when > 0).
        if self._is_combined(params):
            av  = jax.nn.softplus(params["log_av"])        # learned ε ≥ 0
            Hc  = jax.vmap(jax.jacfwd(jax.jacfwd(_cons)))(xy)
            lap = Hc[:, :, 0, 0] + Hc[:, :, 1, 1]
            res = res - av * lap
        elif self.art_visc > 0.0:
            Hc  = jax.vmap(jax.jacfwd(jax.jacfwd(_cons)))(xy)
            lap = Hc[:, :, 0, 0] + Hc[:, :, 1, 1]
            res = res - self.art_visc * lap

        # Return shape (N, 4): [mass, mom_x, mom_y, energy]
        return res

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

    # ------------------------------------------------------------------
    # Prandtl-Meyer expansion-fan analytical solution
    # ------------------------------------------------------------------

    @staticmethod
    def _pm_nu(M: float, gamma: float) -> float:
        """Prandtl-Meyer function ν(M) in radians, M >= 1.

        ν(M) = sqrt((γ+1)/(γ-1)) atan(sqrt((γ-1)/(γ+1) (M²-1))) − atan(sqrt(M²-1))

        Verified against standard textbook values (γ=1.4): ν(1)=0°,
        ν(1.5)=11.91°, ν(2)=26.38°, ν(3)=49.76°, ν(M→∞)→130.45°.
        """
        if M <= 1.0:
            return 0.0
        a = math.sqrt((gamma + 1.0) / (gamma - 1.0))
        m = math.sqrt(M * M - 1.0)
        return a * math.atan(m / a) - math.atan(m)

    @classmethod
    def _pm_mach_from_nu(cls, nu_target: float, gamma: float,
                         lo: float = 1.0, hi: float = 50.0,
                         n_bisect: int = 100) -> float:
        """Invert ν(M) = nu_target (radians) by bisection -- ν is monotonic
        increasing in M for M >= 1, so this is well-posed."""
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if cls._pm_nu(mid, gamma) < nu_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def prandtl_meyer_expansion(self, M1: float, rho1: float, u1: float,
                                v1: float, p1: float, delta_deg: float):
        """Post-expansion-fan state for a convex (flow-turning-away) corner.

        An isentropic expansion fan: stagnation pressure/density are
        conserved across it (unlike a shock), so the post-expansion state
        follows from the Prandtl-Meyer angle shift ν(M2) = ν(M1) + Δθ and
        the standard isentropic p0/p, ρ0/ρ relations.

        Parameters
        ----------
        M1, rho1, u1, v1, p1 : upstream (pre-expansion) state; ``u1``/``v1``
            give the upstream flow *direction* the deflection is measured
            relative to.
        delta_deg : flow deflection angle (degrees, > 0), i.e. how far the
            wall (and hence the flow) turns *away* from itself at this
            corner.

        Returns
        -------
        dict with keys ``M2``, ``rho2``, ``u2``, ``v2``, ``p2`` -- the
        downstream primitive state, with velocity rotated by
        ``delta_deg`` relative to the upstream flow direction.
        """
        gamma = self.gamma
        nu1 = self._pm_nu(M1, gamma)
        nu2 = nu1 + math.radians(delta_deg)
        M2 = self._pm_mach_from_nu(nu2, gamma)

        def p0_over_p(M):
            return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))

        ratio = p0_over_p(M1) / p0_over_p(M2)      # = p2/p1 (shared p0)
        p2 = p1 * ratio
        rho2 = rho1 * ratio ** (1.0 / gamma)

        theta1 = math.atan2(v1, u1)                 # upstream flow angle
        a2 = math.sqrt(gamma * p2 / rho2)
        V2 = M2 * a2
        theta2 = theta1 - math.radians(delta_deg)    # turns away from the wall
        u2 = V2 * math.cos(theta2)
        v2 = V2 * math.sin(theta2)

        return {"M2": M2, "rho2": rho2, "u2": u2, "v2": v2, "p2": p2}
