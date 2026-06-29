"""2-D Steady Compressible Navier–Stokes PDE (conservative flux form).

Viscous counterpart of :class:`CompressibleEulerPDE` — adds the Newtonian
viscous stresses and Fourier heat conduction, for problems such as the Mach-3
compression-corner / shock-boundary-layer interaction (SBLI) over a ramp.

Non-dimensionalisation (freestream reference ρ∞, U∞, T∞, μ∞, length L)::

    ∂x(F − Fv/Re) + ∂y(G − Gv/Re) = 0
      U = (ρ, ρu, ρv, E),     E = p/(γ−1) + ½ρ(u²+v²)
      p = ρ T / (γ M²)                         (perfect-gas EOS, this scaling)

Inviscid fluxes::

    F = (ρu,  ρu²+p,  ρuv,    (E+p)u)
    G = (ρv,  ρuv,    ρv²+p,  (E+p)v)

Viscous fluxes (Newtonian, Stokes hypothesis λ = −⅔μ; Fourier law)::

    τxx = ⅔μ(2u_x − v_y),  τyy = ⅔μ(2v_y − u_x),  τxy = μ(u_y + v_x)
    κ   = μ / [(γ−1) Pr M²]                      (non-dim conductivity)
    Fv  = (0, τxx, τxy, uτxx + vτxy + κ T_x)
    Gv  = (0, τxy, τyy, uτxy + vτyy + κ T_y)

The network maps (x, y) → (f_ρ, f_u, f_v, f_T); ρ and T use a softplus
positivity transform so ρ > 0, T > 0 throughout training.  Viscosity is either
constant (μ* = 1) or Sutherland's law.

Freestream non-dimensional state:  ρ = 1, u = 1, v = 0, T = 1, p = 1/(γM²).
Total (stagnation) temperature:     T₀ = 1 + ½(γ−1)M².
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE


class CompressibleNS2DPDE(BasePDE):
    """2-D steady compressible Navier–Stokes in conservative flux-divergence form.

    Parameters
    ----------
    model   : Flax module  (x, y) → (f_ρ, f_u, f_v, f_T)  (2-in / 4-out).
    gamma   : ratio of specific heats (default 1.4).
    M_inf   : freestream Mach number (default 3.0).
    Re      : Reynolds number ρ∞U∞L/μ∞ (default 1e4).
    Pr      : Prandtl number (default 0.72).
    eps     : positivity floor added after softplus for ρ and T.
    mu_law  : ``"constant"`` (μ*=1, default) or ``"sutherland"``.
    suth_S  : non-dimensional Sutherland constant S/T∞ (default 0.383 ≈ 110.4/288).
    """

    def __init__(self, model, gamma: float = 1.4, M_inf: float = 3.0,
                 Re: float = 1.0e4, Pr: float = 0.72, eps: float = 1e-6,
                 mu_law: str = "constant", suth_S: float = 0.383,
                 art_visc: float = 0.0, av_sensor: str = "ducros",
                 av_s: float = 0.05):
        self.model    = model
        self.gamma    = float(gamma)
        self.M_inf    = float(M_inf)
        self.Re       = float(Re)
        self.Pr       = float(Pr)
        self.eps      = float(eps)
        self.mu_law   = str(mu_law).lower()
        self.suth_S   = float(suth_S)
        # Artificial viscosity ε on the conserved-variable Laplacian −ε∇²U.
        # At high Re the physical viscosity is tiny, so the shock is nearly
        # inviscid and a PINN develops Gibbs oscillations there; a small ε adds
        # numerical dissipation that stabilises the shock (shock capturing).
        #
        # ``av_sensor`` localises that dissipation so it does NOT smear the
        # boundary layer or smooth flow (which would lower |∇ρ| where you want
        # it sharp):
        #   "ducros" (default) — ε_local = ε · θ²/(θ²+ω²) · ½(1−tanh(θ/s)),
        #                        active only at compression shocks (θ=∇·u<0,
        #                        dilatation-dominated), ≈0 in the vorticity-
        #                        dominated boundary layer and in expansions.
        #   "global"           — uniform ε everywhere (smears ρ globally).
        self.art_visc  = float(art_visc)
        self.av_sensor = str(av_sensor).lower()
        self.av_s      = float(av_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def total_temperature(self) -> float:
        """Stagnation temperature  T₀/T∞ = 1 + ½(γ−1)M²."""
        return 1.0 + 0.5 * (self.gamma - 1.0) * self.M_inf ** 2

    def freestream(self):
        """Freestream non-dimensional primitives (ρ, u, v, T)."""
        return 1.0, 1.0, 0.0, 1.0

    def _mu(self, T):
        """Non-dimensional dynamic viscosity μ*(T*)."""
        if self.mu_law == "sutherland":
            S = self.suth_S
            return T ** 1.5 * (1.0 + S) / (T + S)
        return jnp.ones_like(T)

    def _prim_vec(self, net, xy):
        """Physical primitives (ρ, u, v, T) as (N, 4) for a batch of points."""
        raw = self.model.apply(net, xy)                 # (N, 4)
        rho = jax.nn.softplus(raw[:, 0]) + self.eps
        u   = raw[:, 1]
        v   = raw[:, 2]
        T   = jax.nn.softplus(raw[:, 3]) + self.eps
        return jnp.stack([rho, u, v, T], axis=1)

    @staticmethod
    def _is_combined(params):
        return False

    def _net(self, params):
        return params

    # ------------------------------------------------------------------
    # Physical forward pass
    # ------------------------------------------------------------------

    def apply(self, params, xy):
        """Return physical primitives (ρ, u, v, T) as an (N, 4) array."""
        return self._prim_vec(self._net(params), xy)

    def pressure(self, params, xy):
        prim = self.apply(params, xy)
        rho, T = prim[:, 0], prim[:, 3]
        return rho * T / (self.gamma * self.M_inf ** 2)

    def mach(self, params, xy):
        prim = self.apply(params, xy)
        u, v, T = prim[:, 1], prim[:, 2], prim[:, 3]
        a = jnp.sqrt(T) / self.M_inf                    # a*² = T*/M²
        return jnp.sqrt(u ** 2 + v ** 2) / a

    # ------------------------------------------------------------------
    # Residual (conservative, viscous)
    # ------------------------------------------------------------------

    def residual(self, params, xy):
        """Compressible NS residual at points xy (N, 2).  Returns (N, 4)."""
        gamma = self.gamma
        M2    = self.M_inf ** 2
        Re    = self.Re
        Pr    = self.Pr
        eps   = self.eps
        net   = self._net(params)

        def _prim(p_in):                                # (2,) → (4,)
            raw = self.model.apply(net, p_in[None, :])[0]
            rho = jax.nn.softplus(raw[0]) + eps
            u   = raw[1]
            v   = raw[2]
            T   = jax.nn.softplus(raw[3]) + eps
            return jnp.stack([rho, u, v, T])

        def _flux_pair(p_in):                           # (2,) → (4, 2)
            rho, u, v, T = _prim(p_in)
            p = rho * T / (gamma * M2)
            E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)

            # Inviscid fluxes
            F = jnp.stack([rho * u, rho * u * u + p, rho * u * v, (E + p) * u])
            G = jnp.stack([rho * v, rho * u * v, rho * v * v + p, (E + p) * v])

            # Primitive gradients (∂/∂x, ∂/∂y)
            Jp = jax.jacfwd(_prim)(p_in)                # (4, 2)
            u_x, u_y = Jp[1, 0], Jp[1, 1]
            v_x, v_y = Jp[2, 0], Jp[2, 1]
            T_x, T_y = Jp[3, 0], Jp[3, 1]

            mu  = self._mu(T)
            txx = (2.0 * mu / 3.0) * (2.0 * u_x - v_y)
            tyy = (2.0 * mu / 3.0) * (2.0 * v_y - u_x)
            txy = mu * (u_y + v_x)
            kap = mu / ((gamma - 1.0) * Pr * M2)        # κ*

            Fv = jnp.stack([jnp.zeros_like(rho), txx, txy,
                            u * txx + v * txy + kap * T_x])
            Gv = jnp.stack([jnp.zeros_like(rho), txy, tyy,
                            u * txy + v * tyy + kap * T_y])

            F_tot = F - Fv / Re
            G_tot = G - Gv / Re
            return jnp.stack([F_tot, G_tot], axis=1)    # (4, 2)

        def _cons(p_in):                                # conserved U = (ρ, ρu, ρv, E)
            rho, u, v, T = _prim(p_in)
            p = rho * T / (gamma * M2)
            E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
            return jnp.stack([rho, rho * u, rho * v, E])

        av       = self.art_visc
        ducros   = self.av_sensor == "ducros"
        av_s     = self.av_s

        def _point_res(p_in):
            # Divergence  ∂F_tot/∂x + ∂G_tot/∂y  (nested jacfwd ⇒ 2nd-order AD)
            Hf = jax.jacfwd(_flux_pair)(p_in)           # (4, 2, 2)
            r  = Hf[:, 0, 0] + Hf[:, 1, 1]              # (4,)
            if av > 0.0:
                # Artificial viscosity:  − ε_local ∇²U  (Laplacian of conserved U)
                Hc  = jax.jacfwd(jax.jacfwd(_cons))(p_in)   # (4, 2, 2)
                lap = Hc[:, 0, 0] + Hc[:, 1, 1]             # (4,)
                eps_local = av
                if ducros:
                    Jp    = jax.jacfwd(_prim)(p_in)         # (4, 2)
                    theta = Jp[1, 0] + Jp[2, 1]             # ∇·u (dilatation)
                    omega = Jp[2, 0] - Jp[1, 1]             # ∂x v − ∂y u (vorticity)
                    phi   = theta ** 2 / (theta ** 2 + omega ** 2 + 1e-8)
                    comp  = 0.5 * (1.0 - jnp.tanh(theta / av_s))   # ≈1 for θ<0
                    eps_local = av * phi * comp             # shock-localised
                r = r - eps_local * lap
            return r

        return jax.vmap(_point_res)(xy)                 # (N, 4)
