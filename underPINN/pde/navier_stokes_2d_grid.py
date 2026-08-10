"""Grid (finite-difference) incompressible Navier-Stokes residual, primitive
variables, for the flow-past-a-cylinder FNO2D operator.

Central differences throughout, matched to a Chorin-projection reference
solver (see the local ``datagen.py`` in the cylinder-flow example) so the
residual stencil agrees with the stencil that generated the training data —
the same central/upwind-matching concern documented in
:mod:`underPINN.pde.burgers_grid`.
"""
from __future__ import annotations

import jax.numpy as jnp

from underPINN.core.base import BasePDE


class CylinderNSGrid(BasePDE):
    r"""Incompressible NS grid residual on ``(u, v, p)``.

    .. math::
        u_t + u u_x + v u_y + p_x - \tfrac{1}{Re}(u_{xx}+u_{yy}) &= 0 \\
        v_t + u v_x + v v_y + p_y - \tfrac{1}{Re}(v_{xx}+v_{yy}) &= 0 \\
        u_x + v_y &= 0

    Parameters
    ----------
    model         : an :class:`underPINN.nn.operators.FNO2D` with
                    ``out_channels=3`` (u, v, p).
    dt, dx, dy    : grid spacing of the reference solution.
    pred_steps    : frames ahead of the last history frame the target is.
    obstacle_mask : optional ``(Nx, Ny)`` bool array, ``True`` where the grid
                    point sits inside the solid cylinder — those points have
                    no fluid PDE to satisfy and are excluded from the residual.

    Model input layout
    -------------------
    ``x_input``: ``(batch, Nx, Ny, 2*prev_steps + 1)`` — ``prev_steps``
    history frames of ``(u, v)`` interleaved (``u_1, v_1, ..., u_k, v_k``),
    followed by one constant Reynolds-number channel (read per-sample from
    the input, so a single batch can mix trajectories at different Re —
    exactly like the ``nu`` channel in :mod:`underPINN.pde.burgers_grid`).
    """

    def __init__(self, model, dt: float, dx: float, dy: float,
                pred_steps: int = 1, obstacle_mask=None):
        self.model = model
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.pred_steps = pred_steps
        self.obstacle_mask = obstacle_mask

    def residual(self, params, x_input):
        out = self.model.apply(params, x_input)     # (B, Nx, Ny, 3)
        return self.residual_from_pred(x_input, out)

    def residual_from_pred(self, x_input, out):
        """Same residual, reusing an already-computed ``(u, v, p)`` prediction
        (see :meth:`underPINN.pde.burgers_grid.BurgersGrid1D.residual_from_pred`)."""
        u, v, p = out[..., 0], out[..., 1], out[..., 2]

        u_prev = x_input[..., -3]
        v_prev = x_input[..., -2]
        Re = x_input[:, 0, 0, -1]

        u_t = (u - u_prev) / (self.dt * self.pred_steps)
        v_t = (v - v_prev) / (self.dt * self.pred_steps)

        u_xp, u_xm = jnp.roll(u, -1, axis=1), jnp.roll(u, 1, axis=1)
        u_yp, u_ym = jnp.roll(u, -1, axis=2), jnp.roll(u, 1, axis=2)
        v_xp, v_xm = jnp.roll(v, -1, axis=1), jnp.roll(v, 1, axis=1)
        v_yp, v_ym = jnp.roll(v, -1, axis=2), jnp.roll(v, 1, axis=2)
        p_xp, p_xm = jnp.roll(p, -1, axis=1), jnp.roll(p, 1, axis=1)
        p_yp, p_ym = jnp.roll(p, -1, axis=2), jnp.roll(p, 1, axis=2)

        u_x = (u_xp - u_xm) / (2.0 * self.dx)
        u_y = (u_yp - u_ym) / (2.0 * self.dy)
        v_x = (v_xp - v_xm) / (2.0 * self.dx)
        v_y = (v_yp - v_ym) / (2.0 * self.dy)
        p_x = (p_xp - p_xm) / (2.0 * self.dx)
        p_y = (p_yp - p_ym) / (2.0 * self.dy)

        u_xx = (u_xp - 2.0 * u + u_xm) / self.dx ** 2
        u_yy = (u_yp - 2.0 * u + u_ym) / self.dy ** 2
        v_xx = (v_xp - 2.0 * v + v_xm) / self.dx ** 2
        v_yy = (v_yp - 2.0 * v + v_ym) / self.dy ** 2

        inv_re = (1.0 / Re)[:, None, None]
        mom_x = u_t + u * u_x + v * u_y + p_x - inv_re * (u_xx + u_yy)
        mom_y = v_t + u * v_x + v * v_y + p_y - inv_re * (v_xx + v_yy)
        cont = u_x + v_y

        res = jnp.stack([mom_x, mom_y, cont], axis=-1)   # (B, Nx, Ny, 3)
        if self.obstacle_mask is not None:
            res = res * (~self.obstacle_mask)[None, :, :, None]
        return res
