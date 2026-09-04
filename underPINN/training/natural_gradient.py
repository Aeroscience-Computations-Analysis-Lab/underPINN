"""Gauss-Newton / natural-gradient training for PINNs.

A reviewer noted the absence of second-order / natural-gradient training
(citing recent "D-NGD"-style work) alongside underPINN's Adam-based solvers.
Plain first-order Adam treats the PDE-residual loss landscape as if it were
well-conditioned; PINN loss surfaces are notoriously not (Wang et al. 2022's
"respecting causality" line of work, and the broader natural-gradient PINN
literature, motivate this), and a Gauss-Newton step -- which approximates the
loss's curvature from the residual's own Jacobian rather than from gradient
statistics -- is the classical way to correct for that.

This module is not a transcription of a specific published "D-NGD" algorithm
(the review comment does not pin one down precisely enough to reproduce
verbatim); it is a standard Levenberg-Marquardt-damped Gauss-Newton method,
in the same "natural-gradient-flavored second-order PINN training" spirit,
offered as a genuine alternative to Adam rather than a literal reproduction
of any one paper.

Algorithm
---------
Write the training loss as a sum of squared residuals ``L(theta) =
0.5 * ||r(theta)||^2`` (true of every underPINN loss: PDE residual plus
weighted IC/BC residuals, all concatenated into one vector). Gauss-Newton
approximates the loss's Hessian by ``J^T J`` (``J`` = the Jacobian of ``r``
w.r.t. the flattened parameters ``theta``) rather than requiring the true,
expensive Hessian, then solves the damped normal equations

    (J^T J + damping * I) @ delta = J^T r
    theta_new = theta - delta

Levenberg-Marquardt trust-region control adapts ``damping``: a step that
reduces the loss is accepted and damping is shrunk (trusting the curvature
model more, moving closer to pure Gauss-Newton); a step that increases the
loss is rejected (parameters unchanged) and damping is grown (falling back
towards gradient descent), then retried next iteration.

Cost and scope
---------------
Each step forms an explicit ``(n_params, n_params)`` matrix and solves a
dense linear system -- ``O(n_res * n_params^2 + n_params^3)`` per step. This
is the well-known limitation of exact Gauss-Newton/natural-gradient methods:
they are only tractable for small networks (low hundreds to a few thousand
parameters) and small-to-moderate residual counts, which is exactly the
regime the natural-gradient PINN literature demonstrates results in. This is
not intended to replace underPINN's Adam-based solvers on the large 3-D /
compressible-flow benchmarks elsewhere in the codebase.

Usage
-----
>>> from jax.flatten_util import ravel_pytree
>>> from underPINN.training.natural_gradient import train_gauss_newton
>>> def residual_fn(params):
...     # any pytree -> flat (or reshape-to-flat) residual vector
...     return pde.residual(params, x_r).reshape(-1)
>>> final_params, loss_hist, damping_hist = train_gauss_newton(
...     residual_fn, params0, epochs=200)
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def gauss_newton_step(
    flat_residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    flat_params: jnp.ndarray,
    damping: float,
) -> jnp.ndarray:
    """One Levenberg-Marquardt-damped Gauss-Newton step on flattened params.

    Parameters
    ----------
    flat_residual_fn : callable ``flat_theta -> r``, ``r`` a 1-D residual
                       vector (``L = 0.5 * ||r||^2``).
    flat_params       : current flattened parameters, shape ``(n_params,)``.
    damping           : Levenberg-Marquardt damping added to the
                        Gauss-Newton matrix's diagonal.

    Returns
    -------
    Trial (not yet accepted/rejected) flattened parameters after one step.

    Notes
    -----
    Two numerical precautions, both found necessary empirically (not just in
    theory) while validating this module on GPU -- Adam's first-order update
    tolerates the precision underPINN's other solvers run at by default, but
    Gauss-Newton's curvature estimate turned out not to:

    1. Solves the damped normal equations via an *augmented least-squares*
       form -- ``lstsq([J; sqrt(damping)*I], [r; 0])`` -- rather than
       explicitly forming ``J^T J + damping*I`` and calling ``solve``. The
       two are mathematically equivalent, but forming ``J^T J`` explicitly
       squares the Jacobian's condition number (the textbook pitfall of
       "naive" Gauss-Newton); the augmented-lstsq form never forms it, so
       the effective condition number is that of ``J`` itself.
    2. Callers (``train_gauss_newton`` below) run this under
       ``jax.default_matmul_precision("highest")``. JAX's default GPU matmul
       precision trades mantissa bits for throughput; that's harmless for
       Adam, but here it corrupts ``jacfwd``'s own internal matmuls (through
       every ``Dense`` layer) enough that the Jacobian itself is wrong in
       the low digits Newton's method depends on. *Precaution 1 alone did
       not fix this* -- a version with the augmented-lstsq solve still
       diverged on GPU at default matmul precision on this exact
       problem/seed, and only converged once "highest" precision was forced;
       we verified both precautions are independently necessary before
       settling on using both.
    """
    r = flat_residual_fn(flat_params)
    J = jax.jacfwd(flat_residual_fn)(flat_params)          # (n_res, n_params)
    n = flat_params.shape[0]
    sqrt_d = jnp.sqrt(damping).astype(flat_params.dtype)
    A = jnp.concatenate(
        [J, sqrt_d * jnp.eye(n, dtype=flat_params.dtype)], axis=0)
    b = jnp.concatenate([r, jnp.zeros(n, dtype=flat_params.dtype)])
    delta, _residuals, _rank, _sv = jnp.linalg.lstsq(A, b)
    return flat_params - delta


def train_gauss_newton(
    residual_fn: Callable,
    params0,
    epochs: int,
    damping0: float = 1e-3,
    damping_up: float = 3.0,
    damping_down: float = 0.5,
    damping_min: float = 1e-10,
    damping_max: float = 1e10,
    log_every: int = 0,
) -> Tuple[object, List[float], List[float]]:
    """Train *params0* for *epochs* Levenberg-Marquardt/Gauss-Newton steps.

    Parameters
    ----------
    residual_fn : callable ``params (pytree) -> r`` returning a 1-D (or
                  reshape-to-1-D) residual vector -- e.g.
                  ``lambda p: pde.residual(p, x_r).reshape(-1)``, optionally
                  concatenated with weighted IC/BC residual terms so the
                  whole loss is captured as one sum-of-squares.
    params0      : initial parameters, any JAX pytree.
    epochs       : number of accept/reject Gauss-Newton iterations.
    damping0     : initial Levenberg-Marquardt damping.
    damping_up   : multiplicative growth applied after a rejected step.
    damping_down : multiplicative shrink applied after an accepted step.
    damping_min, damping_max : clamp bounds on the damping.
    log_every    : if > 0, print progress every this many epochs.

    Returns
    -------
    (final_params, loss_hist, damping_hist) -- ``final_params`` has the same
    pytree structure as ``params0``; ``loss_hist``/``damping_hist`` are
    length ``epochs + 1`` (including the initial, pre-training values).
    """
    flat_params, unravel = ravel_pytree(params0)

    def flat_residual(fp):
        return jnp.asarray(residual_fn(unravel(fp))).reshape(-1)

    # Compile the step and the loss once and reuse the compiled executable
    # every epoch -- an un-jitted Python loop here would retrace the forward
    # pass + jacfwd + linear solve from scratch on every single iteration,
    # the exact per-epoch dispatch/retracing cost the rest of underPINN's
    # jax.jit-per-step design exists to eliminate (see
    # underPINN/solver/*.py). ``damping`` is a traced (not static) argument,
    # so it can change every epoch without triggering a recompile.
    #
    # The whole body below runs under forced "highest" matmul precision --
    # see gauss_newton_step's docstring for why this (and not just the
    # augmented-lstsq solve) turned out to be necessary on GPU. This has to
    # wrap tracing, not just execution: JAX bakes the active precision into
    # the compiled HLO at trace time, so entering the context only around
    # each *call* to an already-traced jit function would be a no-op.
    with jax.default_matmul_precision("highest"):
        _jit_step = jax.jit(
            lambda fp, d: gauss_newton_step(flat_residual, fp, d))
        _jit_loss = jax.jit(lambda fp: 0.5 * jnp.sum(flat_residual(fp) ** 2))

        def loss_of(fp) -> float:
            return float(_jit_loss(fp))

        damping = damping0
        cur_loss = loss_of(flat_params)
        loss_hist = [cur_loss]
        damping_hist = [damping]

        for ep in range(1, epochs + 1):
            trial = _jit_step(flat_params, damping)
            trial_loss = loss_of(trial)
            if trial_loss < cur_loss:
                flat_params, cur_loss = trial, trial_loss
                damping = max(damping * damping_down, damping_min)
            else:
                damping = min(damping * damping_up, damping_max)
            loss_hist.append(cur_loss)
            damping_hist.append(damping)
            if log_every and (ep % log_every == 0 or ep == epochs):
                print(f"[GaussNewton] epoch {ep:5d}  loss={cur_loss:.6e}  "
                     f"damping={damping:.3e}")

    return unravel(flat_params), loss_hist, damping_hist
