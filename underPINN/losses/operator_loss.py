"""Losses for neural-operator (PINO / DeepONet) training.

``OperatorLoss`` is the data + physics loss shared by the grid-based
operators (FNO1D, FNO2D, and CViT via
:func:`underPINN.nn.operators.cvit_grid_predict`) — all three ultimately
produce a full prediction *grid* and pair with a grid-residual PDE from
:mod:`underPINN.pde.burgers_grid` / :mod:`underPINN.pde.navier_stokes_2d_grid`.

``DeepONetLoss`` is purpose-built for the branch/trunk operator: unlike the
data-supervised grid losses, classic DeepONet training is self-supervised —
only the initial condition (exactly what the branch net's sensors already
sample) and, optionally, a boundary constraint are used as "data"; the
interior is fit purely by minimizing the autodiff PDE residual from
:mod:`underPINN.pde.burgers_deeponet`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BaseLoss
from underPINN.losses.loss import l1_loss, l2_loss


class OperatorLoss(BaseLoss):
    """Generic PINO loss: data MSE + weighted PDE grid residual.

    Parameters
    ----------
    predict_fn : ``(params, model_input) -> u_pred`` — a full forward pass
                 producing a prediction grid the same shape as the target
                 (e.g. ``model.apply`` for FNO1D/FNO2D, or a closure around
                 :func:`underPINN.nn.operators.cvit_grid_predict` for CViT).
    pde        : an object with ``residual_from_pred(pde_input, u_pred)``
                 (e.g. :class:`underPINN.pde.burgers_grid.BurgersGrid1D`).
    rba        : residual-based adaptivity — reweight the PDE term by each
                 point's relative residual magnitude (detached), matching
                 the RBA convention used by :class:`underPINN.losses.loss.PINNLoss`.
    data_mask  : optional array, spatial shape only (no batch/channel dims),
                 1 where a grid point counts toward the data loss and 0
                 where it shouldn't (e.g. inside a solid obstacle). None
                 (default): unmasked data loss, every existing example
                 unaffected.
    """

    def __init__(self, predict_fn, pde, loss_type: str = "l2",
                rba: bool = False, rba_eps: float = 1e-6, data_mask=None):
        self.predict_fn = predict_fn
        self.pde = pde
        self.rba = rba
        self.rba_eps = rba_eps
        # data_mask: optional array broadcastable against u_pred's spatial
        # dims (shape (Nx,) for FNO1D, (Nx, Ny) for FNO2D/CViT), 1 where a
        # grid point should count toward the data loss and 0 where it
        # shouldn't (e.g. inside a solid obstacle the target is only zero by
        # construction, not something the network should be scored against
        # or pushed to represent -- a low-Fourier-mode FNO structurally
        # cannot fit a sharp obstacle cutout via unmasked MSE, it can only
        # smear it; see examples/operators/fno2d_cylinder). None (default):
        # every existing FNO/CViT example is unaffected, unmasked data loss.
        self.data_mask = data_mask
        self._l1 = loss_type == "l1"
        if loss_type == "l2":
            self.norm = l2_loss
        elif loss_type == "l1":
            self.norm = l1_loss
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def __call__(self, params, x_input, u_target, pde_weight: float = 1.0,
                model_input=None):
        """
        x_input     : the PDE-facing array (supplies ``u_prev``/``nu`` to the
                      grid residual) — for FNO this is also the model input.
        u_target    : ground-truth next frame, same shape as the prediction.
        pde_weight  : PDE-term weight for this step (pass a warmup-ramped
                      value from the solver; kept as a per-call argument so
                      the ramp doesn't force re-tracing the jitted step).
        model_input : the model-facing input, if different from ``x_input``
                      (CViT: a ``(batch, T, Nx, Ny, C)`` window rather than
                      the FNO-style ``(batch, Nx, Ny, prev_steps+1)`` array).
                      Defaults to ``x_input``.
        """
        u_pred = self.predict_fn(params, x_input if model_input is None else model_input)
        diff = u_pred - u_target
        if self.data_mask is not None:
            # Broadcast the spatial mask over the leading batch dim and the
            # trailing channel dim, then normalise by the actual number of
            # unmasked (batch, spatial, channel) elements -- not the full
            # array size -- so masked-out points contribute neither to the
            # numerator nor inflate the denominator (matching a plain mean
            # over just the valid points, not a mean-with-zeros).
            mask_b = self.data_mask.reshape((1,) + self.data_mask.shape + (1,))
            n_valid = (jnp.sum(self.data_mask) * u_pred.shape[0] * u_pred.shape[-1]
                      + self.rba_eps)
            elementwise = jnp.abs(diff) if self._l1 else diff ** 2
            data_loss = jnp.sum(elementwise * mask_b) / n_valid
        else:
            data_loss = self.norm(diff)

        res = self.pde.residual_from_pred(x_input, u_pred)
        if self.rba:
            # Detached so RBA reweights step size per point, not gradient direction.
            w = jax.lax.stop_gradient(
                jnp.abs(res) / (jnp.mean(jnp.abs(res)) + self.rba_eps)
            )
            pde_loss = jnp.mean(w * res ** 2)
        else:
            pde_loss = self.norm(res)

        total = data_loss + pde_weight * pde_loss
        return total, (data_loss, pde_loss)


class DeepONetLoss(BaseLoss):
    """Physics-informed DeepONet loss: IC (+ optional BC) + PDE residual.

    No full-field ground truth is required: the initial condition is exactly
    what the branch net's sensors sample, so IC supervision is free, and the
    interior is fit purely by minimizing the PDE residual.
    """

    def __init__(self, model, pde, ics_weight: float = 20.0,
                bcs_weight: float = 1.0, loss_type: str = "l2",
                periodic: bool = True, rba: bool = False, rba_eps: float = 1e-6):
        self.model = model
        self.pde = pde
        self.ics_weight = ics_weight
        self.bcs_weight = bcs_weight
        self.periodic = periodic
        self.rba = rba
        self.rba_eps = rba_eps
        if loss_type == "l2":
            self.norm = l2_loss
        elif loss_type == "l1":
            self.norm = l1_loss
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def _apply_grid(self, params, u_batch, xt_grid):
        """``u_batch``: (B, m); ``xt_grid``: (Ng, 2) shared query points.
        Returns (B, Ng) — every trajectory queried at every grid point."""
        def per_traj(u_i):
            return jax.vmap(lambda xt_j: self.model.apply(params, u_i, xt_j))(xt_grid)
        return jax.vmap(per_traj)(u_batch)

    def __call__(self, params, u_batch, x_sensors, u_batch_r, xt_r,
                xt_bc_lo=None, xt_bc_hi=None, pde_weight: float = 1.0):
        """
        u_batch    : (B, m) sensor values — also the IC target, since the
                     sensors sample u(x, 0) by construction.
        x_sensors  : (m,) sensor x-locations (shared across the batch).
        u_batch_r  : (B_r, m) sensor values for the residual batch, paired
                     1:1 with ``xt_r`` (may differ in size from ``u_batch``).
        xt_r       : (B_r, 2) packed (x, t) collocation points, one per row.
        xt_bc_lo, xt_bc_hi : (N_bc, 2) shared boundary query points, both
                     required together to enable the BC term.
        """
        xt_ic = jnp.stack([x_sensors, jnp.zeros_like(x_sensors)], axis=1)  # (m, 2)
        u_ic_pred = self._apply_grid(params, u_batch, xt_ic)               # (B, m)
        ics_loss = self.norm(u_ic_pred - u_batch)

        if xt_bc_lo is not None and xt_bc_hi is not None:
            u_lo = self._apply_grid(params, u_batch, xt_bc_lo)
            u_hi = self._apply_grid(params, u_batch, xt_bc_hi)
            bcs_loss = (self.norm(u_lo - u_hi) if self.periodic
                       else self.norm(u_lo) + self.norm(u_hi))
        else:
            bcs_loss = 0.0

        res = self.pde.residual(params, u_batch_r, xt_r)
        if self.rba:
            w = jax.lax.stop_gradient(
                jnp.abs(res) / (jnp.mean(jnp.abs(res)) + self.rba_eps)
            )
            res_loss = jnp.mean(w * res ** 2)
        else:
            res_loss = self.norm(res)

        total = (self.ics_weight * ics_loss + self.bcs_weight * bcs_loss
                + pde_weight * res_loss)
        return total, (ics_loss, bcs_loss, res_loss)
