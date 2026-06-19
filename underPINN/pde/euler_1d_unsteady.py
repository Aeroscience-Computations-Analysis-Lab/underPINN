"""1-D Unsteady Compressible Euler PDE (conservative form).

For Riemann problems such as the Sod shock tube.  The network maps
(x, t) → (f_ρ, f_u, f_p); physical variables use a positivity transform so
ρ > 0 and p > 0 throughout training.  Two transforms are available:

* ``softplus`` (default):  ρ = softplus(f_ρ) + ε,  p = softplus(f_p) + ε.
* ``exp`` (log-space):     ρ = exp(f_ρ),           p = exp(f_p).
  The exp / log parameterisation gives a *constant relative* gradient
  (d log ρ / d f_ρ = 1) across many decades, so it stays well-conditioned for
  severe Riemann problems like Toro test 3 (p spans 0.01 → 1000), where
  softplus' gradient collapses (∝ p) in the low-pressure region.

Conservative residual (ε = artificial viscosity):

    ∂U/∂t + ∂F/∂x − ε ∂²U/∂x² = 0
      U = (ρ, ρu, E)                      conserved variables
      F = (ρu, ρu²+p, (E+p)u)             flux
      E = p/(γ−1) + ½ρu²                  total energy (perfect gas)

The artificial viscosity may be a **fixed** float (``art_visc``) or **learned**
when the optimisable state is the combined pytree
``{"net": net_params, "log_av": raw_scalar}`` with ε = softplus(log_av) ≥ 0.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE


class Euler1DUnsteadyPDE(BasePDE):
    """1-D unsteady compressible Euler in conservative form.

    Parameters
    ----------
    model    : Flax module  (x, t) → (f_ρ, f_u, f_p)  with 2-in / 3-out.
    gamma    : ratio of specific heats (default 1.4).
    eps      : small constant added after softplus for positivity.
    art_visc : fixed artificial-viscosity coefficient ε (default 0.0).
               Ignored when the params carry a trainable ``log_av``.
    transform: positivity map for ρ and p — ``"softplus"`` (default) or
               ``"exp"`` (log-space; recommended for large dynamic range).
    """

    def __init__(self, model, gamma: float = 1.4, eps: float = 1e-6,
                 art_visc: float = 0.0, transform: str = "softplus"):
        self.model     = model
        self.gamma     = float(gamma)
        self.eps       = float(eps)
        self.art_visc  = float(art_visc)
        self.transform = str(transform).lower()

    def _pos(self, x):
        """Positivity transform applied to the ρ and p network outputs."""
        if self.transform == "exp":
            return jnp.exp(jnp.clip(x, -50.0, 50.0))
        return jax.nn.softplus(x) + self.eps

    # ------------------------------------------------------------------
    # Trainable-viscosity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_combined(params) -> bool:
        try:
            return ("net" in params) and ("log_av" in params)
        except TypeError:
            return False

    def _net(self, params):
        return params["net"] if self._is_combined(params) else params

    def viscosity(self, params=None):
        """Current ε — learned softplus(log_av) or the fixed float."""
        if params is not None and self._is_combined(params):
            return float(jax.nn.softplus(params["log_av"]))
        return self.art_visc

    @staticmethod
    def inverse_softplus(y: float) -> float:
        """Raw value x with softplus(x) ≈ y (to initialise log_av)."""
        return float(math.log(math.expm1(max(y, 1e-12))))

    # ------------------------------------------------------------------
    # Forward pass (physical variables)
    # ------------------------------------------------------------------

    def apply(self, params, xt):
        """Return physical state (ρ, u, p) as an (N, 3) array."""
        raw = self.model.apply(self._net(params), xt)   # (N, 3)
        rho = self._pos(raw[:, 0])
        u   = raw[:, 1]
        p   = self._pos(raw[:, 2])
        return jnp.stack([rho, u, p], axis=1)

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def residual(self, params, xt):
        """Conservative residual at points xt (N, 2) = (x, t).  Returns (N, 3)."""
        gamma = self.gamma
        net   = self._net(params)
        pos   = self._pos

        def _prim(p_in):
            raw = self.model.apply(net, p_in[None, :])[0]   # (3,)
            rho = pos(raw[0])
            u   = raw[1]
            p   = pos(raw[2])
            return rho, u, p

        def _cons(p_in):
            rho, u, p = _prim(p_in)
            E = p / (gamma - 1.0) + 0.5 * rho * u * u
            return jnp.stack([rho, rho * u, E])

        # Stack U (col 0) and F (col 1) → (3, 2); derivative index: 0=x, 1=t.
        def _UF(p_in):
            rho, u, p = _prim(p_in)
            E = p / (gamma - 1.0) + 0.5 * rho * u * u
            U = jnp.stack([rho, rho * u, E])
            F = jnp.stack([rho * u, rho * u * u + p, (E + p) * u])
            return jnp.stack([U, F], axis=1)               # (3, 2)

        J   = jax.vmap(jax.jacfwd(_UF))(xt)                # (N, 3, 2, 2)
        # ∂U_k/∂t = J[k, 0, 1] ;  ∂F_k/∂x = J[k, 1, 0]
        res = J[:, :, 0, 1] + J[:, :, 1, 0]                # (N, 3)

        # ── Artificial viscosity:  − ε ∂²U/∂x²  (Laplacian of conserved U in x) ──
        if self._is_combined(params):
            av  = jax.nn.softplus(params["log_av"])
            H   = jax.vmap(jax.jacfwd(jax.jacfwd(_cons)))(xt)   # (N, 3, 2, 2)
            res = res - av * H[:, :, 0, 0]
        elif self.art_visc > 0.0:
            H   = jax.vmap(jax.jacfwd(jax.jacfwd(_cons)))(xt)
            res = res - self.art_visc * H[:, :, 0, 0]

        return res
