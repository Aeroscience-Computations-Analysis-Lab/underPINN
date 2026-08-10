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

        def grad_single(u_i, xt_i):
            return jax.jacfwd(lambda z: s_of_xt(u_i, z))(xt_i)

        def hess_single(u_i, xt_i):
            return jax.hessian(lambda z: s_of_xt(u_i, z))(xt_i)

        J = jax.vmap(grad_single)(u_batch, xt_batch)      # (batch, 2): [s_x, s_t]
        H = jax.vmap(hess_single)(u_batch, xt_batch)      # (batch, 2, 2)
        s = jax.vmap(s_of_xt)(u_batch, xt_batch)

        s_x, s_t, s_xx = J[:, 0], J[:, 1], H[:, 0, 0]
        return s_t + s * s_x - self.nu * s_xx
