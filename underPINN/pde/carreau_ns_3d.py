"""3-D Steady Generalized-Newtonian (Carreau) Navier-Stokes PDE.

Models a shear-thinning fluid such as blood using the **Carreau** rheological
model (Nagargoje, Mishra & Gupta, *Phys. Fluids* 33, 071904 (2021), Eq. 6):

    μ(γ̇) = μ∞ + (μ0 − μ∞) [1 + (λ γ̇)²]^((n−1)/2)

with the shear rate  γ̇ = sqrt(2 E:E),  E = ½(∇u + ∇uᵀ).

The system is solved non-dimensionally.  Scaling with a length R, velocity U
and reference (high-shear) viscosity μ∞ gives:

    ∇·u = 0
    (u·∇)u = −∇p + (1/Re) ∇·[ μ*(γ̇*) (∇u + ∇uᵀ) ]

    Re   = ρ U R / μ∞                     Reynolds number
    μ*   = 1 + (β − 1) [1 + (Cu γ̇*)²]^((n−1)/2)
    β    = μ0 / μ∞                        zero/infinite-shear viscosity ratio
    Cu   = λ U / R                        Carreau number (dimensionless λ)

The high-shear limit μ* → 1 recovers a Newtonian fluid with viscosity μ∞;
β > 1 with n < 1 produces shear thinning (lower viscosity at high γ̇).

Network maps (x, y, z) → (u, v, w, p).  Build with layers[0]=3, layers[-1]=4.

Paper (Table II) blood parameters → β = 0.056/0.0035 = 16, n = 0.3568, and
Cu = λ U / R from λ = 3.131 s and the chosen U, R.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from underPINN.core.base import BasePDE


# ---------------------------------------------------------------------------
# Fully-developed Carreau pipe profile (non-dimensional reference, R* = 1)
# ---------------------------------------------------------------------------

def carreau_mu_star(a, beta: float, Cu: float, n: float):
    """Non-dimensional apparent viscosity μ* at shear-rate magnitude *a*."""
    return 1.0 + (beta - 1.0) * (1.0 + (Cu * a) ** 2) ** ((n - 1.0) / 2.0)


def carreau_shear_from_stress(tau_mag: float, beta: float, Cu: float, n: float) -> float:
    """Invert μ*(a)·a = tau_mag for the shear-rate magnitude a ≥ 0 (bisection).

    The shear stress μ*(a)·a is monotone increasing in a for a Carreau fluid,
    so the inverse is unique.
    """
    if tau_mag <= 0.0:
        return 0.0

    def f(a):
        return carreau_mu_star(a, beta, Cu, n) * a - tau_mag

    lo, hi = 0.0, 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def carreau_developed_profile(beta: float, Cu: float, n: float,
                              u_center: float = 1.0, n_r: int = 400):
    """Fully-developed Carreau velocity profile u*(r) on r ∈ [0, 1].

    Solves  μ*(|u*'|) u*' = ½ P r  with u*(1)=0, matching the centreline value
    ``u_center`` by adjusting the non-dimensional pressure-gradient constant P.

    Returns
    -------
    r : (n_r,) radial grid in [0, 1]
    u : (n_r,) velocity profile  (u[0] = u_center, u[-1] = 0)
    P : matched pressure-gradient constant  (= Re · dp*/dx*, negative)
    """
    r = np.linspace(0.0, 1.0, n_r)

    def profile_for_P(P):
        a = np.array([carreau_shear_from_stress(0.5 * abs(P) * ri, beta, Cu, n)
                      for ri in r])
        seg = 0.5 * (a[:-1] + a[1:]) * np.diff(r)        # trapezoid segments
        u = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])  # ∫_r^1 a dr'
        return u

    P_lo, P_hi = 1e-6, 1.0
    while profile_for_P(P_hi)[0] < u_center:
        P_hi *= 2.0
    for _ in range(80):
        P_mid = 0.5 * (P_lo + P_hi)
        if profile_for_P(P_mid)[0] < u_center:
            P_lo = P_mid
        else:
            P_hi = P_mid
    P = 0.5 * (P_lo + P_hi)
    return r, profile_for_P(P), -P


class CarreauNS3DPDE(BasePDE):
    """3-D steady incompressible Carreau (generalized-Newtonian) NS.

    Parameters
    ----------
    model : Flax module — (N, 3) → (N, 4).
    Re    : Reynolds number based on the infinite-shear viscosity μ∞.
    beta  : viscosity ratio μ0/μ∞ (≥ 1; Newtonian when = 1).
    Cu    : Carreau number λU/R (dimensionless relaxation time).
    n     : power-law index (< 1 → shear thinning).
    """

    def __init__(self, model, Re: float = 100.0, beta: float = 16.0,
                 Cu: float = 10.0, n: float = 0.3568):
        self.model = model
        self.Re    = float(Re)
        self.beta  = float(beta)
        self.Cu    = float(Cu)
        self.n     = float(n)

    # ------------------------------------------------------------------
    # Forward pass + viscosity
    # ------------------------------------------------------------------

    def uvwp(self, params, xyz):
        out = self.model.apply(params, xyz)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def _mu_star_from_gradu(self, J):
        """Non-dimensional apparent viscosity μ* from velocity gradient J (3,3)."""
        S    = J + J.T                       # ∇u + ∇uᵀ = 2E
        gdot = jnp.sqrt(0.5 * jnp.sum(S * S) + 1e-12)   # = sqrt(2 E:E)
        return 1.0 + (self.beta - 1.0) * (1.0 + (self.Cu * gdot) ** 2) ** ((self.n - 1.0) / 2.0)

    def apparent_viscosity(self, params, xyz):
        """μ*(γ̇*) at each point xyz (N, 3) → (N,)."""
        def _vel(p_in):
            return self.model.apply(params, p_in[None, :])[0, :3]

        def _mu(p_in):
            return self._mu_star_from_gradu(jax.jacfwd(_vel)(p_in))

        return jax.vmap(_mu)(xyz)

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def residual(self, params, xyz):
        """Compute the 4 PDE residuals at xyz (N, 3) → (N, 4)."""
        Re = self.Re

        def _vel(p_in):
            return self.model.apply(params, p_in[None, :])[0, :3]   # (3,)

        def _pre(p_in):
            return self.model.apply(params, p_in[None, :])[0, 3]    # scalar

        def _gradu(p_in):
            return jax.jacfwd(_vel)(p_in)                           # (3,3) [i,j]=∂u_i/∂x_j

        def _tau(p_in):
            J = _gradu(p_in)
            S = J + J.T
            return self._mu_star_from_gradu(J) * S                  # τ* (3,3)

        def _point(p_in):
            J     = _gradu(p_in)                                    # (3,3)
            u     = _vel(p_in)                                      # (3,)
            cont  = J[0, 0] + J[1, 1] + J[2, 2]
            conv  = J @ u                                           # (u·∇)u_i
            gp    = jax.jacfwd(_pre)(p_in)                          # (3,)
            dtau  = jax.jacfwd(_tau)(p_in)                          # (3,3,3) ∂τ_ij/∂x_k
            divt  = dtau[:, 0, 0] + dtau[:, 1, 1] + dtau[:, 2, 2]   # ∇·τ_i
            mom   = conv + gp - (1.0 / Re) * divt                   # (3,)
            return jnp.concatenate([cont[None], mom])              # (4,)

        return jax.vmap(_point)(xyz)
