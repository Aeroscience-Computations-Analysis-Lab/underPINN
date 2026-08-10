"""Sampling utilities shared by runners, solvers, and example scripts.

The helpers here wrap ``jax.random.choice`` with safe defaults so that a
batch request larger than the pool never raises
``ValueError: Cannot take a larger sample than population when 'replace=False'``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def safe_choice(key, n: int, batch: int) -> jnp.ndarray:
    """Draw *batch* indices from ``[0, n)`` without crashing when batch > n.

    Equivalent to ``jax.random.choice(key, n, (batch,), replace=...)`` where
    ``replace`` is automatically set to ``True`` whenever ``batch > n``.

    Parameters
    ----------
    key:
        JAX PRNG key.
    n:
        Pool size (number of available samples).
    batch:
        Number of indices to draw.

    Returns
    -------
    jnp.ndarray
        Integer index array of shape ``(batch,)``.

    Examples
    --------
    >>> import jax
    >>> key = jax.random.PRNGKey(0)
    >>> safe_choice(key, n=100, batch=32)   # normal path  – replace=False
    >>> safe_choice(key, n=50,  batch=256)  # fallback path – replace=True
    """
    return jax.random.choice(key, n, (batch,), replace=(batch > n))


def rad_resample(
    pde,
    params,
    sampler,
    n_keep: int,
    n_candidates: int,
    k: float = 1.0,
    c: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Residual-based Adaptive (RAR / RAD) resampling of collocation points.

    Implements the RAD density of Wu et al. (2023): a fresh candidate pool is
    drawn, the PDE-residual magnitude ``r(x)`` is evaluated, and ``n_keep``
    points are sampled with probability proportional to

        p(x)  ∝  r(x)^k / E[r(x)^k]  +  c

    With ``k=1, c=1`` this concentrates points where the residual is large
    (e.g. shocks / contact discontinuities) while keeping uniform coverage
    everywhere.  Returning a fixed ``n_keep`` keeps array shapes constant, so
    a JIT-compiled training step is not retraced.

    Parameters
    ----------
    pde          : object with ``residual(params, X) -> (N, ...)``.
    params       : current parameters (plain or combined trainable pytree).
    sampler      : callable ``sampler(n, seed) -> (n, d) float array`` that
                   draws candidate points uniformly over the domain.
    n_keep       : number of points to return (the new pool size).
    n_candidates : size of the candidate pool (≥ a few × n_keep).
    k, c         : RAD exponent and additive constant.
    seed         : RNG seed for both the candidate draw and the selection.

    Returns
    -------
    (n_keep, d) float32 array of resampled collocation points.
    """
    cand = np.asarray(sampler(n_candidates, seed), dtype=np.float32)
    res  = np.asarray(pde.residual(params, jnp.asarray(cand)))
    if res.ndim > 1:
        mag = np.sqrt(np.sum(res ** 2, axis=-1))
    else:
        mag = np.abs(res)

    w = mag ** k
    mean = w.mean()
    w = (w / mean if mean > 0 else np.ones_like(w)) + c
    w = w / w.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), size=n_keep, replace=True, p=w)
    return cand[idx].astype(np.float32)
