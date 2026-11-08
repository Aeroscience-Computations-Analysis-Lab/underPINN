"""Build a ``Problem`` from loss blocks, so ``loss``, ``rho`` and ``residual`` agree by construction.

A PINN loss is a weighted sum of mean-squared residual blocks:

    L(θ) = Σ_b  w_b · mean( r_b(θ; X_b)² )         r_b : PDE residual, IC misfit, BC misfit, data, ...

Describe each block once with a :class:`LossBlock` and :func:`problem_from_blocks` derives

    loss_terms_fn  {b: w_b · mean(r_b²)}
    rho_fn         concat_b  sqrt(w_b / n_b) · r_b      (so sum(rho²) == loss exactly)
    residual_fn    the block named ``residual_block`` (the PDE), for the NTK and off-grid checks

and keeps the blocks on ``Problem.blocks`` for the per-block diagnostics in
:mod:`underPINN.diagnostics.balance` (gradient conflict, per-term NTK, suggested weights).
Hand-writing ``rho_fn`` is the most common way to get a silently wrong curvature
diagnostic; with blocks it cannot drift from the loss.

:func:`problem_from_pinnloss` does this for underPINN's :class:`~underPINN.losses.loss.PINNLoss`
(PDE / IC / optional initial-velocity / Dirichlet-or-periodic BC), matching its loss value.

Block functions must be *point-separable* — row i of ``fn(params, X)`` may depend only on
``X[i]`` — which every PINN residual is. Points can carry their target values as extra
columns (e.g. ``[x, t, u_target]``) for the function to split off.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from .core import Problem

__all__ = ["LossBlock", "problem_from_blocks", "problem_from_pinnloss"]


@dataclass
class LossBlock:
    """One mean-squared loss block: ``weight * mean(fn(params, points)**2)``.

    fn     : ``(params, points) -> (N,)`` or ``(N, k)`` residual rows (target already subtracted).
    points : ``(N, D)`` array; row i of ``fn`` may depend only on ``points[i]``.
    weight : the block's weight in the loss.
    """
    fn: Callable
    points: Any
    weight: float = 1.0


def _rows(r):
    r = jnp.asarray(r)
    return r[:, None] if r.ndim == 1 else r.reshape(r.shape[0], -1)


def problem_from_blocks(blocks: Dict[str, LossBlock], residual_block: Optional[str] = "pde",
                        apply_fn: Optional[Callable] = None, hidden_fn: Optional[Callable] = None,
                        name: str = "problem", **kw) -> Problem:
    """A :class:`Problem` whose loss, rho and residual are all derived from ``blocks``.

    ``residual_block`` names the PDE block used by the NTK / off-grid diagnostics
    (``None`` if there is none). Extra keyword arguments go to :class:`Problem`.
    """
    if not blocks:
        raise ValueError("problem_from_blocks needs at least one block")
    if residual_block is not None and residual_block not in blocks:
        raise ValueError(f"residual_block={residual_block!r} is not one of {list(blocks)}")
    names = list(blocks)
    batch = {n: jnp.asarray(blocks[n].points) for n in names}

    def terms(params, b):
        return {n: blocks[n].weight * jnp.mean(_rows(blocks[n].fn(params, b[n])) ** 2) for n in names}

    def rho(params, b):
        out = []
        for n in names:
            r = _rows(blocks[n].fn(params, b[n]))
            out.append(jnp.sqrt(blocks[n].weight / r.size) * r.ravel())
        return jnp.concatenate(out)

    res_fn = x_res = None
    if residual_block is not None:
        blk = blocks[residual_block]
        res_fn = lambda params, x: _rows(blk.fn(params, x))           # noqa: E731
        x_res = batch[residual_block]
    return Problem(terms, batch, residual_fn=res_fn, x_res=x_res, rho_fn=rho, apply_fn=apply_fn,
                   hidden_fn=hidden_fn, name=name, blocks=blocks, **kw)


def problem_from_pinnloss(loss, x_r, t_r, x_i, u_i, x_b=None, t_b=None, u_b=None,
                          name: str = "pinn", **kw) -> Problem:
    """Diagnostic :class:`Problem` that reproduces a ``PINNLoss`` on fixed points.

    Same arguments as ``PINNLoss.__call__``. Handles ``ic_weight``, ``bc_weight``,
    ``ic_velocity_weight`` and ``periodic_length``. Residual-based adaptivity (``rba``) and
    weight regularisation (``reg_weight``) are not sums of fixed squares; they are left out
    with a warning, so the diagnostic loss is the plain weighted PINN loss.
    """
    model, pde = loss.model, loss.pde
    if getattr(loss, "rba", False):
        warnings.warn("problem_from_pinnloss: rba=True is not represented; diagnosing the "
                      "unweighted PDE residual", UserWarning, stacklevel=2)
    if getattr(loss, "reg_weight", 0.0):
        warnings.warn("problem_from_pinnloss: reg_weight is not represented", UserWarning, stacklevel=2)
    if getattr(loss, "norm", None) is not None and loss.norm.__name__ != "l2_loss":
        raise ValueError("problem_from_pinnloss supports loss_type='l2' only")

    def u(params, x, t):
        return model.apply(params, jnp.stack([x, t], 1))[:, 0]

    x_r2 = x_r if x_r.ndim == 2 else x_r[:, None]
    blocks = {
        "pde": LossBlock(lambda p, xt: pde.residual(p, xt), jnp.concatenate([x_r2, t_r[:, None]], 1), 1.0),
        "ic": LossBlock(lambda p, z: u(p, z[:, 0], jnp.zeros_like(z[:, 0])) - z[:, 1],
                        jnp.stack([x_i, u_i], 1), loss.ic_weight),
    }
    vel = getattr(loss, "ic_velocity_weight", 0.0)
    if vel:
        def ut0(p, z):
            g = jax.vmap(jax.grad(lambda xx, tt: model.apply(p, jnp.stack([xx, tt])[None])[0, 0], 1))
            return g(z[:, 0], jnp.zeros_like(z[:, 0]))
        blocks["ic_velocity"] = LossBlock(ut0, x_i[:, None], loss.ic_weight * vel)
    if x_b is not None:
        L = getattr(loss, "periodic_length", None)
        if L is not None:
            fn = lambda p, z: u(p, z[:, 0], z[:, 1]) - u(p, z[:, 0] + L, z[:, 1])   # noqa: E731
        else:
            fn = lambda p, z: u(p, z[:, 0], z[:, 1]) - z[:, 2]                       # noqa: E731
        blocks["bc"] = LossBlock(fn, jnp.stack([x_b, t_b, u_b if u_b is not None else jnp.zeros_like(x_b)], 1),
                                 loss.bc_weight)
    return problem_from_blocks(blocks, "pde", apply_fn=lambda p, x: model.apply(p, x)[:, 0], name=name, **kw)
