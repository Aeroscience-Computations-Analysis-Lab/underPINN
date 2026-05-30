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

Two interfaces are provided
---------------------------
:func:`rar_d_resample`
    **Preferred** — packed interface.  ``xy`` is an ``(N, D)`` array of all
    coordinates (spatial + time concatenated).  The optional
    ``candidate_sampler`` must return an ``(n, D)`` packed array.  Returns a
    new ``(N, D)`` array.

:func:`rar_d_resample_split`
    Compatibility shim for solvers that store spatial and temporal coordinates
    separately (``x_r`` + ``t_r``).  Accepts an optional ``candidate_sampler``
    that returns a ``(x_cand, t_cand)`` pair.  Packs internally and unpacks the
    result before returning ``(x_new, t_new)``.

Usage example — packed interface
---------------------------------
>>> from underPINN.training.resample import rar_d_resample
>>> xy_r = rar_d_resample(
...     pde, params, xy_r,
...     k=1.0,
...     key=jax.random.PRNGKey(42),
... )

Usage example — split interface (backward-compatible)
------------------------------------------------------
>>> from underPINN.training.resample import rar_d_resample_split
>>> x_r, t_r = rar_d_resample_split(
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
    xy: jnp.ndarray,
    *,
    k: float = 1.0,
    n_candidates: Optional[int] = None,
    candidate_sampler: Optional[Callable[[int, jax.random.KeyArray], jnp.ndarray]] = None,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Resample collocation points proportional to |residual|^k — packed API.

    Parameters
    ----------
    pde :
        Any object with a ``residual(params, xy)`` method that accepts a
        packed ``(N, D)`` coordinate array and returns shape ``(N,)`` or
        ``(N, d)`` — one residual (or residual vector) per point.
    params :
        Current network parameters (JAX pytree).
    xy :
        Current collocation array, shape ``(N, D)`` where D is the total
        number of coordinates (space + time packed together).
    k :
        Exponent in the weighting distribution  p ∝ |r|^k.
        *k = 1* (default) reproduces standard RAR-D.
        Larger k focuses more aggressively on high-residual regions.
    n_candidates :
        Size of the candidate pool evaluated before downsampling.
        Defaults to ``5 × N``.
    candidate_sampler :
        Optional callable ``(n: int, key) → xy_cand`` that draws
        ``n`` fresh candidate points from the problem domain and returns them
        as a packed ``(n, D)`` array.
        When *None* the candidates are drawn by bootstrap resampling (with
        replacement) from the existing collocation set.
    key :
        JAX PRNG key consumed by this call.

    Returns
    -------
    xy_new : jnp.ndarray, shape ``(N, D)``
        New collocation array of the same shape as ``xy``.
    """
    n_keep = xy.shape[0]
    if n_candidates is None:
        n_candidates = 5 * n_keep

    # ------------------------------------------------------------------ #
    # 1. Generate candidate pool                                           #
    # ------------------------------------------------------------------ #
    key, k1, k2 = jax.random.split(key, 3)

    if candidate_sampler is not None:
        xy_cand = jnp.asarray(candidate_sampler(n_candidates, k1))
    else:
        # Bootstrap: uniform resampling with replacement from current set
        idx_boot = jax.random.randint(k1, (n_candidates,), 0, n_keep)
        xy_cand = jnp.asarray(xy)[idx_boot]

    # ------------------------------------------------------------------ #
    # 2. Evaluate residual at every candidate — packed API                 #
    # ------------------------------------------------------------------ #
    res = pde.residual(params, xy_cand)       # (n_cand,) or (n_cand, d)

    # Reduce to a non-negative scalar per point
    if res.ndim > 1:
        res_mag = jnp.sqrt(jnp.sum(res ** 2, axis=-1))   # L2-norm, (n_cand,)
    else:
        res_mag = jnp.abs(res)                            # absolute value

    # ------------------------------------------------------------------ #
    # 3. Build sampling weights  p ∝ |r|^k                                #
    # ------------------------------------------------------------------ #
    weights = res_mag ** k
    total = weights.sum()
    # Guard against all-zero residual (fully converged region)
    weights = jnp.where(
        total > 0.0, weights / total, jnp.ones_like(weights) / n_candidates
    )

    # ------------------------------------------------------------------ #
    # 4. Draw n_keep points from the candidate pool                        #
    # ------------------------------------------------------------------ #
    idx_new = jax.random.choice(k2, n_candidates, shape=(n_keep,), replace=True, p=weights)
    return xy_cand[idx_new]


def rar_d_resample_split(
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
    """Compatibility shim — resample with separate spatial/time arrays.

    Wraps :func:`rar_d_resample` for solvers that store ``x_r`` and ``t_r``
    as separate arrays.  The arguments and semantics are identical to the
    packed interface except:

    * ``x_r`` : spatial coordinates, shape ``(N,)`` or ``(N, D_space)``
    * ``t_r`` : time coordinates, shape ``(N,)``
    * ``candidate_sampler`` (optional) : ``fn(n, key) → (x_cand, t_cand)``
      returning a *pair* of arrays rather than a single packed array.

    Returns ``(x_new, t_new)`` — same shapes as ``(x_r, t_r)``.

    See :func:`rar_d_resample` for full parameter documentation.
    """
    _x_r = jnp.asarray(x_r)
    _t_r = jnp.asarray(t_r)

    # Pack (x, t) → xy  (N, D_space + 1)
    _x_2d = _x_r if _x_r.ndim == 2 else _x_r[:, None]
    xy = jnp.concatenate([_x_2d, _t_r[:, None]], axis=1)

    # Adapt candidate_sampler: (x,t) pair → packed xy
    _packed_sampler = None
    if candidate_sampler is not None:
        def _packed_sampler(n: int, key: jax.random.KeyArray) -> jnp.ndarray:
            x_c, t_c = candidate_sampler(n, key)
            x_c = jnp.asarray(x_c)
            t_c = jnp.asarray(t_c)
            x_c_2d = x_c if x_c.ndim == 2 else x_c[:, None]
            return jnp.concatenate([x_c_2d, t_c[:, None]], axis=1)

    xy_new = rar_d_resample(
        pde, params, xy,
        k=k, n_candidates=n_candidates,
        candidate_sampler=_packed_sampler,
        key=key,
    )

    # Unpack
    if _x_r.ndim == 1:
        x_new = xy_new[:, 0]
    else:
        x_new = xy_new[:, :-1]
    t_new = xy_new[:, -1]
    return x_new, t_new
