"""Sampling utilities shared by runners, solvers, and example scripts.

The helpers here wrap ``jax.random.choice`` with safe defaults so that a
batch request larger than the pool never raises
``ValueError: Cannot take a larger sample than population when 'replace=False'``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
