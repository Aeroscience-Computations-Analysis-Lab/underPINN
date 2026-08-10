"""Reproducibility utilities.

A single :func:`set_seed` call seeds every RNG layer (Python, NumPy, JAX) so
experiments are reproducible without manual bookkeeping.

Usage::

    from underPINN.utils.seed import set_seed

    key = set_seed(42)          # seeds Python / NumPy / JAX, returns PRNGKey(42)
    params = model.init(key, x)
"""

from __future__ import annotations

import random

import jax
#import jax.numpy as jnp
import numpy as np


def set_seed(n: int) -> jax.Array:
    """Seed Python, NumPy, and JAX with the same integer *n*.

    JAX uses a pure-functional PRNG so there is no global mutable state to
    seed; instead this function returns a ``PRNGKey(n)`` that callers should
    use to initialise models and samplers.

    Parameters
    ----------
    n:
        Non-negative integer seed value.

    Returns
    -------
    jax.Array
        ``jax.random.PRNGKey(n)`` — a 2-element uint32 JAX array.

    Examples
    --------
    >>> key = set_seed(0)
    >>> params = model.init(key, jnp.ones((1, 2)))
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"seed must be a non-negative int, got {n!r}")
    random.seed(n)
    np.random.seed(n)
    return jax.random.PRNGKey(n)
