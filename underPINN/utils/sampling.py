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


def qr_deim_resample(
    pde,
    params,
    sampler,
    n_keep: int,
    n_candidates: int,
    seed: int = 0,
    augment_coords: bool = True,
) -> np.ndarray:
    """QR-DEIM anchor points + leverage-score-weighted fill, for adaptive
    collocation resampling.

    ``rad_resample`` above (and RAR-D) select points by drawing *randomly*
    from a probability distribution ``p(x) ~ |r(x)|^k`` — points cluster
    wherever the residual is largest, with no mechanism to notice that many
    of those draws land on top of each other (e.g. dozens of points all
    within one grid cell of a single, narrow shock spike) instead of
    spreading across every high-residual region. The Discrete Empirical
    Interpolation Method (DEIM; Chaturantabut & Sorensen, 2010) and its
    numerically stabilized column-pivoted-QR selection rule (Drmac &
    Gugercin, 2016 — "QR-DEIM") were developed for exactly this kind of
    problem in reduced-order modelling: *deterministically* choosing a
    subset of rows/points that keeps a basis matrix as well-conditioned
    (mutually independent) as possible, rather than sampling proportional
    to magnitude alone.

    This function is not a transcription of a specific published "QR-DEIM-R"
    algorithm (we are not aware of one pinned down precisely enough by the
    review comment to reproduce verbatim); it is our own construction
    *inspired by* that selection philosophy. A collocation batch needs far
    more points (``n_keep`` in the thousands to tens of thousands) than a
    natural residual/coordinate feature basis has columns (``r0``, a
    handful), and plain QR-DEIM is capped at exactly one point per basis
    column — so it cannot fill a full batch on its own. The "R" (randomized)
    in the name is a leverage-score-weighted fill (Drineas, Mahoney & Muthu-
    krishnan, 2006 — the standard randomized-NLA way to draw a large,
    non-uniform subset that still respects a small basis's column space)
    used to reach the rest of ``n_keep``, rather than forming any matrix
    larger than ``(n_candidates, r0)``: an earlier version of this function
    sketched a dense ``(n_candidates, n_keep)`` matrix for a *second* pivoted
    QR to reach ``n_keep`` points directly, which is correct in principle but
    allocates ``O(n_candidates * n_keep)`` — tens of gigabytes at real
    collocation-batch sizes (e.g. 40,000 x 200,000). The construction below
    is ``O(n_candidates * r0)`` throughout. Concretely:

    1.  Draw ``n_candidates`` points and evaluate the (possibly vector-valued)
        PDE residual at each -- exactly as ``rad_resample`` does.
    2.  Build a small per-point feature basis ``U`` (shape
        ``(n_candidates, r0)``, ``r0`` a handful of columns): the residual
        component(s) themselves, plus (if ``augment_coords``) the point's
        coordinates scaled by its residual magnitude. The coordinate columns
        are what let this tell apart two points with similar residual
        magnitude but different locations -- pure magnitude has no way to
        do that.
    3.  **Anchors (deterministic, DEIM-exact):** column-pivoted QR of the
        *small* ``U.T`` (``r0 x n_candidates`` — cheap; this is the literal
        QR-DEIM selection rule, unchanged) gives up to ``r0`` pivot points
        that are, in the classical DEIM sense, the most independent/
        best-conditioned points for representing ``U``'s column space. These
        are always included.
    4.  **Fill (randomized, leverage-score-weighted):** a thin QR of ``U``
        itself gives an orthonormal basis ``Q`` (``n_candidates x r0``); each
        row's squared norm is its statistical leverage score — how much that
        point's feature vector is *not* explained by (i.e. independent of)
        the rest of the pool. The remaining ``n_keep - r0`` points are drawn
        with probability proportional to this score, so redundant points
        inside a cluster share and dilute one cluster's total leverage
        instead of each independently re-winning a magnitude-only lottery.

    This is meant for the same "call every few hundred/thousand epochs"
    cadence as RAR-D/RAD, not every step.

    Parameters
    ----------
    pde            : object with ``residual(params, X) -> (N, ...)``.
    params         : current parameters (plain or combined trainable pytree).
    sampler        : callable ``sampler(n, seed) -> (n, d)`` drawing candidate
                     points uniformly over the domain (same contract as
                     ``rad_resample``'s ``sampler``).
    n_keep         : number of points to return.
    n_candidates   : candidate pool size (must be >= n_keep; a few x n_keep
                     is typical, matching ``rad_resample``'s usage).
    seed           : RNG seed for the candidate draw and the leverage-score fill.
    augment_coords : include residual-weighted coordinate columns in the
                     basis (default True); set False to pivot on raw
                     residual value(s) alone.

    Returns
    -------
    (n_keep, d) float32 array of resampled collocation points.
    """
    import scipy.linalg

    n_candidates = max(n_candidates, n_keep)
    cand = np.asarray(sampler(n_candidates, seed), dtype=np.float32)
    res = np.asarray(pde.residual(params, jnp.asarray(cand)))
    res2d = res[:, None] if res.ndim == 1 else res.reshape(len(cand), -1)

    basis_cols = [res2d]
    if augment_coords:
        mag = np.linalg.norm(res2d, axis=1, keepdims=True)
        coords = cand if cand.ndim == 2 else cand[:, None]
        basis_cols.append(mag * coords)
    U = np.concatenate(basis_cols, axis=1).astype(np.float64)  # (n_cand, r0)
    r0 = U.shape[1]

    # Anchors: pivoted QR of the small (r0, n_cand) transpose.
    _, _, piv = scipy.linalg.qr(U.T, mode="economic", pivoting=True)
    n_anchors = min(r0, n_keep)
    anchor_idx = piv[:n_anchors]

    idx = anchor_idx
    n_fill = n_keep - n_anchors
    if n_fill > 0:
        # Leverage-score fill: thin QR of U (n_cand, r0) -> orthonormal Q;
        # row-norm^2 of Q is the standard leverage score for that row.
        Q, _ = np.linalg.qr(U)
        leverage = np.sum(Q ** 2, axis=1)
        total = leverage.sum()
        p = leverage / total if total > 0 else np.full(n_candidates, 1.0 / n_candidates)
        rng = np.random.default_rng(seed)
        fill_idx = rng.choice(n_candidates, size=n_fill, replace=True, p=p)
        idx = np.concatenate([anchor_idx, fill_idx])
    return cand[idx].astype(np.float32)
