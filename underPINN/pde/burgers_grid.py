"""Grid (finite-difference) Burgers residuals for the FNO / CViT operators.

Point-network PINNs (see :mod:`underPINN.pde.burgers`) evaluate the residual
by autodiff at scattered collocation points. Neural operators instead map a
whole grid to a whole grid, so the natural residual is a finite-difference
stencil applied to the predicted *field* — the PINO ("physics-informed neural
operator") approach of Li et al.

Both classes read a model input laid out as ``[history frames..., constant nu
channel]`` (built by :mod:`underPINN.utils.operator_datagen`), call the model
once to get the predicted next frame, then differentiate that prediction on
the grid. The time derivative uses backward Euler against the last history
frame.
"""
from __future__ import annotations

import jax.numpy as jnp

from underPINN.core.base import BasePDE


class BurgersGrid1D(BasePDE):
    """1-D viscous Burgers grid residual: ``u_t + u u_x - nu u_xx``.

    Parameters
    ----------
    model      : an :class:`underPINN.nn.operators.FNO1D` (or compatible).
    dt, dx     : grid spacing of the reference solution.
    pred_steps : how many ``dt`` the target frame is ahead of the last
                 history frame (matches the datagen ``pred_steps`` knob).
    periodic   : ``True`` wraps the space derivatives; ``False`` (Dirichlet)
                 drops the two boundary columns from the residual instead of
                 one-sided-differencing them (the walls are already pinned by
                 the data loss).
    """

    def __init__(self, model, dt: float, dx: float, pred_steps: int = 1,
                periodic: bool = True):
        self.model = model
        self.dt = dt
        self.dx = dx
        self.pred_steps = pred_steps
        self.periodic = periodic

    def residual(self, params, x_input):
        """``x_input``: ``(batch, Nx, prev_steps + 1)``, last channel = nu.

        Returns the residual field, same leading shape as the prediction
        (``(batch, Nx)`` if periodic, ``(batch, Nx - 2)`` if Dirichlet).
        """
        u_pred = self.model.apply(params, x_input)
        return self.residual_from_pred(x_input, u_pred)

    def residual_from_pred(self, x_input, u_pred):
        """Same residual, but reusing an already-computed prediction — lets a
        loss that also needs ``u_pred`` for a data term avoid a second
        forward pass. Accepts ``u_pred`` either as ``(batch, Nx)`` or with a
        trailing singleton channel axis (``(batch, Nx, 1)``, the model's raw
        output shape)."""
        if u_pred.ndim == x_input.ndim:
            u_pred = u_pred[..., 0]
        u_prev = x_input[..., -2]
        nu = x_input[:, 0, -1]                                  # (B,)

        u_t = (u_pred - u_prev) / (self.dt * self.pred_steps)

        if self.periodic:
            u_xp, u_xm = jnp.roll(u_pred, -1, axis=1), jnp.roll(u_pred, 1, axis=1)
        else:
            u_xp = jnp.concatenate([u_pred[:, 1:], u_pred[:, -1:]], axis=1)
            u_xm = jnp.concatenate([u_pred[:, :1], u_pred[:, :-1]], axis=1)

        u_x = (u_xp - u_xm) / (2.0 * self.dx)
        u_xx = (u_xp - 2.0 * u_pred + u_xm) / self.dx ** 2
        res = u_t + u_pred * u_x - nu[:, None] * u_xx

        if not self.periodic:
            res = res[:, 1:-1]      # boundary FD is one-sided/inaccurate here
        return res


class BurgersGrid2D(BasePDE):
    """2-D periodic viscous Burgers grid residual:
    ``u_t + u(u_x + u_y) - nu (u_xx + u_yy)``.

    Parameters
    ----------
    model      : an :class:`underPINN.nn.operators.FNO2D` (or compatible).
    dt, dx, dy : grid spacing of the reference solution.
    pred_steps : frames ahead of the last history frame the target is.
    scheme     : ``"central"`` or ``"upwind"`` for the advection term — must
                 match whatever stencil the *reference solver* used to
                 generate training data. A mismatched scheme (e.g. central
                 residual against an upwind-integrated dataset) measurably
                 hurts accuracy, so this is a required, not cosmetic, choice.
    """

    def __init__(self, model, dt: float, dx: float, dy: float,
                pred_steps: int = 1, scheme: str = "central"):
        if scheme not in ("central", "upwind"):
            raise ValueError(f"scheme must be 'central' or 'upwind', got {scheme!r}")
        self.model = model
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.pred_steps = pred_steps
        self.scheme = scheme

    def residual(self, params, x_input):
        """``x_input``: ``(batch, Nx, Ny, prev_steps + 1)``, last channel = nu."""
        u_pred = self.model.apply(params, x_input)
        return self.residual_from_pred(x_input, u_pred)

    def residual_from_pred(self, x_input, u_pred):
        """Same residual, reusing an already-computed prediction. Accepts
        ``u_pred`` either as ``(batch, Nx, Ny)`` or ``(batch, Nx, Ny, 1)``
        (see :meth:`BurgersGrid1D.residual_from_pred`)."""
        if u_pred.ndim == x_input.ndim:
            u_pred = u_pred[..., 0]
        u_prev = x_input[..., -2]
        nu = x_input[:, 0, 0, -1]                               # (B,)

        u_t = (u_pred - u_prev) / (self.dt * self.pred_steps)

        u_xp, u_xm = jnp.roll(u_pred, -1, axis=1), jnp.roll(u_pred, 1, axis=1)
        u_yp, u_ym = jnp.roll(u_pred, -1, axis=2), jnp.roll(u_pred, 1, axis=2)
        lap = ((u_xp - 2.0 * u_pred + u_xm) / self.dx ** 2
              + (u_yp - 2.0 * u_pred + u_ym) / self.dy ** 2)

        if self.scheme == "central":
            u_x = (u_xp - u_xm) / (2.0 * self.dx)
            u_y = (u_yp - u_ym) / (2.0 * self.dy)
        else:
            u_x = jnp.where(u_pred >= 0.0, (u_pred - u_xm) / self.dx,
                           (u_xp - u_pred) / self.dx)
            u_y = jnp.where(u_pred >= 0.0, (u_pred - u_ym) / self.dy,
                           (u_yp - u_pred) / self.dy)

        return u_t + u_pred * (u_x + u_y) - nu[:, None, None] * lap
