"""Residual-based Adaptive Distribution resampling (RAR-D).

RAR-D (Lu et al., 2021 — "DeepXDE") improves PINN training by periodically
replacing collocation points with new ones sampled in high-residual regions.

Workflow
--------
1.  Generate a large pool of *candidate* points  (5 × current batch by default).
2.  Evaluate the squared PDE residual at every candidate.
3.  Form a probability distribution  p_i ∝ |r_i|^k  (k=1 by default).
4.  Draw *n_keep* replacement points from the candidates using those weights.

The returned arrays have exactly the same shape as the inputs so the training
loop can drop them in without further adjustment.

Usage example
-------------
>>> from underPINN.training.resample import rar_d_resample
>>> x_r, t_r = rar_d_resample(
...     pde, params, x_r, t_r,
...     k=1.0,
...     key=jax.random.PRNGKey(42),
... )
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


def rar_d_resample(
    pde,
    params,
    x_r: jnp.ndarray,
    t_r: jnp.ndarray,
    *,
    k: float = 1.0,
    n_candidates: Optional[int] = None,
    candidate_sampler: Optional[Callable[[int, jax.random.KeyArray], Tuple]] = None,
    key: jax.random.KeyArray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Resample collocation points proportional to |residual|^k.

    Parameters
    ----------
    pde :
        Any object with a ``residual(params, x, t)`` method that returns an
        array of shape ``(n,)`` or ``(n, d)`` — one residual (or residual
        vector) per point.
    params :
        Current network parameters (JAX pytree).
    x_r, t_r :
        Current collocation spatial and time arrays, both shape ``(n_keep, …)``.
    k :
        Exponent in the weighting distribution  p ∝ |r|^k.
        *k = 1* (default) reproduces standard RAR-D.
        Larger k focuses more aggressively on high-residual regions.
    n_candidates :
        Size of the candidate pool evaluated before downsampling.
        Defaults to ``5 × x_r.shape[0]``.
    candidate_sampler :
        Optional callable ``(n: int, key) → (x_cand, t_cand)`` that draws
        fresh candidate points from the problem domain.
        When *None* the candidates are drawn by bootstrap resampling (with
        replacement) from the existing collocation set — useful when you
        don't have a domain sampler at hand.
    key :
        JAX PRNG key consumed by this call.

    Returns
    -------
    x_new, t_new :
        New collocation arrays of the same shape as ``x_r`` and ``t_r``.
    """
    n_keep = x_r.shape[0]
    if n_candidates is None:
        n_candidates = 5 * n_keep

    # ------------------------------------------------------------------ #
    # 1. Generate candidate pool                                           #
    # ------------------------------------------------------------------ #
    key, k1, k2 = jax.random.split(key, 3)

    if candidate_sampler is not None:
        x_cand, t_cand = candidate_sampler(n_candidates, k1)
        x_cand = jnp.asarray(x_cand)
        t_cand = jnp.asarray(t_cand)
    else:
        # Bootstrap: uniform resampling with replacement from current set
        idx_boot = jax.random.randint(k1, (n_candidates,), 0, n_keep)
        x_cand = jnp.asarray(x_r)[idx_boot]
        t_cand = jnp.asarray(t_r)[idx_boot]

    # ------------------------------------------------------------------ #
    # 2. Evaluate residual at every candidate                              #
    # ------------------------------------------------------------------ #
    res = pde.residual(params, x_cand, t_cand)           # (n_cand,) or (n_cand, d)

    # Reduce to a non-negative scalar per point
    if res.ndim > 1:
        res_mag = jnp.sqrt(jnp.sum(res ** 2, axis=-1))  # L2-norm, (n_cand,)
    else:
        res_mag = jnp.abs(res)                           # absolute value, (n_cand,)

    # ------------------------------------------------------------------ #
    # 3. Build sampling weights  p ∝ |r|^k                                #
    # ------------------------------------------------------------------ #
    weights = res_mag ** k
    total = weights.sum()
    # Guard against all-zero residual (fully converged region)
    weights = jnp.where(total > 0.0, weights / total, jnp.ones_like(weights) / n_candidates)

    # ------------------------------------------------------------------ #
    # 4. Draw n_keep points from the candidate pool                        #
    # ------------------------------------------------------------------ #
    idx_new = jax.random.choice(k2, n_candidates, shape=(n_keep,), replace=True, p=weights)

    return x_cand[idx_new], t_cand[idx_new]
