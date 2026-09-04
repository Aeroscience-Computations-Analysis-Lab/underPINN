"""Autodiff Burgers residual for the DeepONet branch/trunk operator.

Unlike the grid-residual PDEs in :mod:`underPINN.pde.burgers_grid`, a
DeepONet's output ``s(u)(x, t)`` is a continuous function of the query
coordinate, so the residual is taken by autodiff through the *trunk* input —
exactly like the point-network PINN in :mod:`underPINN.pde.burgers`, except
the branch input ``u`` (the sampled initial condition) is held fixed per
sample rather than being a training variable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE


class DeepONetBurgersPDE(BasePDE):
    """Viscous Burgers residual ``s_t + s s_x - nu s_xx`` for a DeepONet.

    Parameters
    ----------
    model : a :class:`underPINN.nn.operators.DeepONet1D`.
    nu    : kinematic viscosity.
    """

    def __init__(self, model, nu: float = 0.01):
        self.model = model
        self.nu = nu

    def residual(self, params, u_batch, xt_batch):
        """``u_batch``: ``(batch, m)`` sensor values (branch input, fixed).

        ``xt_batch``: ``(batch, 2)`` packed query points, ``xt[:, 0] = x``,
        ``xt[:, 1] = t`` — the same packing convention as
        :class:`underPINN.pde.burgers.BurgersPDE`.

        Returns ``(batch,)`` residual values.
        """
        def s_of_xt(u_i, xt_i):
            return self.model.apply(params, u_i, xt_i)

        # Was: separate jax.jacfwd (both entries used, so no waste there)
        # and jax.hessian (full 2x2, only s_xx used) plus a third separate
        # forward pass for the value s -- same redundancy profiled and
        # fixed in underPINN/pde/burgers.py, adapted to keep the DeepONet
        # branch input u_i fixed (not differentiated) per sample.
        def per_point(u_i, xt_i):
            def scalar_fn(z):
                return s_of_xt(u_i, z)

            s_val, vjp_fn = jax.vjp(scalar_fn, xt_i)

            def grad_only(z):
                return jax.vjp(scalar_fn, z)[1](1.0)[0]

            grad_vec = vjp_fn(1.0)[0]
            _, jvp_out = jax.jvp(grad_only, (xt_i,), (jnp.array([1.0, 0.0]),))
            return s_val, grad_vec[0], grad_vec[1], jvp_out[0]

        s, s_x, s_t, s_xx = jax.vmap(per_point)(u_batch, xt_batch)
        return s_t + s * s_x - self.nu * s_xx
