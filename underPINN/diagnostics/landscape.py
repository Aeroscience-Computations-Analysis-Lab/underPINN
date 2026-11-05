"""Loss-landscape slices (Li et al., NeurIPS 2018) and a non-convexity count.

Filter normalisation: a random direction d with N(0, 1) entries is rescaled unit by unit
to the norm of the trained weights, so slices of different networks are comparable. A
"unit" is one neuron's incoming weights: a row of W in the u = W x convention, i.e.
one column of a flax Dense kernel (shape (in, out)) or of the Fourier matrix B (shape
(in, n_fourier)); each bias vector is rescaled as one block. A 1-D slice is
f(a) = L(theta* + a d) on a in [-span, span] (``report.Config`` defaults: 141 points, 7 directions); a 2-D
slice uses d1 and d2 with d2 made orthogonal to d1 and rescaled to ||d1||.

Non-convexity count: with D_j = f(a_{j+1}) - f(a_j), E = #{j : sgn D_j != sgn D_{j+1}}.
A slice through the trained point has E >= 1; E > 1 proves the slice is non-convex.
It is a LOWER bound on the number of non-convex slices (a slice with one minimum can
still be non-convex). ``rtol`` ignores differences below rtol*|f|, so single-precision
noise in a flat region is not counted as an extremum.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from .core import Problem


def filter_normalised_direction(params, key):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    out = []
    for k, w in zip(keys, leaves):
        d = jax.random.normal(k, w.shape, w.dtype)
        if w.ndim >= 2:                                   # one unit = one column (norm over all other axes)
            axes = tuple(range(w.ndim - 1))
            nw = jnp.sqrt(jnp.sum(w * w, axis=axes, keepdims=True))
            nd = jnp.sqrt(jnp.sum(d * d, axis=axes, keepdims=True))
        else:
            nw, nd = jnp.linalg.norm(w), jnp.linalg.norm(d)
        out.append(d * nw / jnp.maximum(nd, 1e-12))
    return jax.tree_util.tree_unflatten(treedef, out)


def _shift(params, d1, a, d2=None, b=0.0):
    return tree_map(lambda p, u, *v: p + a * u + (b * v[0] if v else 0.0), params, d1, *((d2,) if d2 is not None else ()))


def slice_1d(problem: Problem, params, n_dir: int = 7, n_alpha: int = 141, span: float = 1.0, seed: int = 0):
    """Loss along n_dir random filter-normalised directions. Returns (alphas, (n_dir, n_alpha))."""
    alphas = jnp.linspace(-span, span, n_alpha)
    f = jax.jit(lambda d: jax.lax.map(lambda a: problem.loss(_shift(params, d, a)), alphas))
    keys = jax.random.split(jax.random.PRNGKey(seed), n_dir)
    return np.asarray(alphas), np.stack([np.asarray(f(filter_normalised_direction(params, k))) for k in keys])


def slice_2d(problem: Problem, params, n: int = 31, span: float = 1.0, seed: int = 0):
    """Loss on the plane spanned by two filter-normalised directions (second made orthogonal to
    the first, then rescaled to the first's norm). Returns (grid, F) with
    F[i, j] = L(theta* + grid[j] d1 + grid[i] d2), i.e. alpha along columns, ready for
    ``contourf(grid, grid, F)``."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    d1, d2 = filter_normalised_direction(params, k1), filter_normalised_direction(params, k2)
    f1, f2 = (ravel_pytree(d)[0] for d in (d1, d2))
    f2 = f2 - (f1 @ f2) / (f1 @ f1) * f1
    f2 = f2 * jnp.linalg.norm(f1) / jnp.linalg.norm(f2)
    _, unravel = ravel_pytree(d1)
    d2 = unravel(f2)
    g = jnp.linspace(-span, span, n)
    A, B = jnp.meshgrid(g, g)
    pts = jnp.stack([A.ravel(), B.ravel()], 1)
    f = jax.jit(lambda: jax.lax.map(lambda ab: problem.loss(_shift(params, d1, ab[0], d2, ab[1])), pts))
    return np.asarray(g), np.asarray(f()).reshape(n, n)


def count_extrema(f, rtol: float = 1e-5) -> int:
    """Number of discrete local extrema of a sampled slice: sign changes of consecutive
    differences, ignoring differences below rtol * |f| (flat-region rounding noise)."""
    f = np.asarray(f, np.float64)
    d = np.diff(f)
    d = d[np.abs(d) > rtol * np.maximum(np.abs(f[1:]), np.abs(f[:-1]))]
    return int(np.sum(np.sign(d[:-1]) != np.sign(d[1:])))
