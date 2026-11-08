"""1-D benchmark PDEs from the PINN failure-mode literature.

Krishnapriyan et al., "Characterizing possible failure modes in physics-informed
neural networks", NeurIPS 2021 — the convection and reaction problems used by
PINNsFormer and PINNMamba — and the Allen–Cahn problem used by PirateNets.

All take packed ``xt`` of shape ``(N, 2)`` (x, t) and return ``(N,)`` residuals.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE

__all__ = ["ConvectionPDE", "ReactionPDE", "AllenCahnPDE"]


def _u_and_grad(model, params):
    def u_single(xt_i):
        return model.apply(params, xt_i[None, :])[0, 0]

    def per_point(xt_i):
        u, vjp = jax.vjp(u_single, xt_i)
        (g,) = vjp(1.0)
        return u, g
    return u_single, per_point


class ConvectionPDE(BasePDE):
    """u_t + β u_x = 0 on x ∈ [0, 2π] (periodic), u(x, 0) = sin x.

    Exact: u = sin(x − β t). β = 30–50 is the classic failure regime.
    """

    def __init__(self, model, beta: float = 50.0):
        self.model, self.beta = model, beta

    def residual(self, params, xt):
        _, per_point = _u_and_grad(self.model, params)
        _, g = jax.vmap(per_point)(xt)
        return g[:, 1] + self.beta * g[:, 0]

    def exact(self, x, t):
        return jnp.sin(x - self.beta * t)

    @staticmethod
    def initial(x):
        return jnp.sin(x)


class ReactionPDE(BasePDE):
    """u_t − ρ u (1 − u) = 0 on x ∈ [0, 2π] (periodic),
    u(x, 0) = exp(−(x − π)² / (2 (π/4)²)).

    Exact (pointwise logistic ODE): u = h e^{ρt} / (h e^{ρt} + 1 − h), h = u(x, 0).
    """

    def __init__(self, model, rho: float = 5.0):
        self.model, self.rho = model, rho

    def residual(self, params, xt):
        _, per_point = _u_and_grad(self.model, params)
        u, g = jax.vmap(per_point)(xt)
        return g[:, 1] - self.rho * u * (1.0 - u)

    @staticmethod
    def initial(x):
        return jnp.exp(-((x - jnp.pi) ** 2) / (2.0 * (jnp.pi / 4.0) ** 2))

    def exact(self, x, t):
        h = self.initial(x)
        e = jnp.exp(self.rho * t)
        return h * e / (h * e + 1.0 - h)


class AllenCahnPDE(BasePDE):
    """u_t − ε u_xx + κ (u³ − u) = 0 on x ∈ [−1, 1] (periodic),
    u(x, 0) = x² cos(π x); defaults ε = 1e-4, κ = 5 (PirateNets / Raissi).

    No closed form: use a spectral reference (e.g. ETDRK4).
    """

    def __init__(self, model, eps: float = 1e-4, kappa: float = 5.0):
        self.model, self.eps, self.kappa = model, eps, kappa

    def residual(self, params, xt):
        u_single, per_point = _u_and_grad(self.model, params)

        def full(xt_i):
            u, g = per_point(xt_i)

            def grad_only(z):
                return jax.vjp(u_single, z)[1](1.0)[0]
            _, jv = jax.jvp(grad_only, (xt_i,), (jnp.array([1.0, 0.0], xt_i.dtype),))
            return u, g[1], jv[0]

        u, ut, uxx = jax.vmap(full)(xt)
        return ut - self.eps * uxx + self.kappa * (u ** 3 - u)

    @staticmethod
    def initial(x):
        return x ** 2 * jnp.cos(jnp.pi * x)
