"""Kolmogorov–Arnold networks for PINNs (PIKANs).

Two drop-in point networks with the same contract as :class:`~underPINN.nn.mlp.MLP`
-- ``layers=[in, h1, …, out]``, ``apply(params, x: (N, in)) -> (N, out)`` -- so
every PDE, loss, solver, restart and data-parallel path works unchanged. Select
them from YAML via ``network.type`` (see :mod:`underPINN.nn.factory`).

Where an MLP layer is ``σ(W x + b)`` with a fixed activation, a KAN layer puts a
*learnable* 1-D function on every input→output edge and sums them::

    y_o = Σ_i φ_{i,o}(x_i)

The two classes differ in how each φ is parameterised.

ChebyKAN (cPIKAN)  -- ``network.type: chebykan``
    φ_{i,o}(x) = Σ_{k=0..K} c_{i,o,k} T_k(tanh x)

    Chebyshev polynomials of the first kind, built with the three-term
    recurrence ``T_k = 2 t T_{k-1} - T_{k-2}`` (never ``cos(k·arccos t)``,
    whose derivatives blow up at ±1). ``tanh`` maps every layer input into
    (-1, 1), the Chebyshev domain. Smooth (C^∞), fixed parameter shapes, and
    cheap: one ``einsum`` per layer. The recommended default for PINNs.

BSplineKAN (original KAN)  -- ``network.type: kan``
    φ_{i,o}(x) = w_{i,o} · silu(u) + Σ_m c_{i,o,m} B_m(u),   u = tanh x

    B_m are order-``spline_order`` B-splines (Cox–de Boor) on a uniform grid of
    ``grid_size`` intervals over ``grid_range`` (default (-1, 1), which ``tanh``
    fills). Cubic splines (order 3) are C², so second derivatives exist but are
    only piecewise linear -- fine for 2nd-order PDEs, poor beyond that.

    *Grid extension* (refining the grid during training) changes parameter
    shapes, which would invalidate the optimizer state and restart snapshots.
    It is therefore not done inside training: use :func:`refine_bspline_kan`
    on a checkpoint to fit the same network onto a finer grid, then start a
    new training run from the refined parameters.

Inputs
------
Both networks squash each layer's input with ``tanh``. Coordinates are best
pre-scaled to roughly [-1, 1] (``tanh`` of a [0, 1] input only spans
[0, 0.76], which wastes half of the basis); a coordinate range of [-2π, 2π]
saturates. This is a property of all PIKAN formulations, not of this code.

Cost
----
A KAN layer of width ``w`` costs roughly ``(K+1)×`` (Chebyshev) or
``(grid_size+spline_order+1)×`` (B-spline) an MLP layer of the same width.
PIKANs typically need much narrower layers than MLPs to reach the same error.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

__all__ = [
    "ChebyKAN",
    "BSplineKAN",
    "chebyshev_basis",
    "bspline_basis",
    "bspline_knots",
    "refine_bspline_kan",
]


# ── bases ────────────────────────────────────────────────────────────────────

def chebyshev_basis(t: jnp.ndarray, degree: int) -> jnp.ndarray:
    """``[T_0(t), …, T_degree(t)]`` stacked on a new last axis.

    Uses the three-term recurrence, which is exact polynomial arithmetic and
    therefore smooth to any derivative order (unlike ``cos(k·arccos t)``).
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    T = [jnp.ones_like(t)]
    if degree >= 1:
        T.append(t)
    for _ in range(2, degree + 1):
        T.append(2.0 * t * T[-1] - T[-2])
    return jnp.stack(T, axis=-1)


def bspline_knots(grid_size: int, spline_order: int,
                  grid_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Uniform knot vector extended by ``spline_order`` knots on each side,
    so that exactly ``grid_size + spline_order`` B-splines are non-zero on
    ``grid_range`` and they form a partition of unity there."""
    if grid_size < 1 or spline_order < 0:
        raise ValueError(f"need grid_size >= 1 and spline_order >= 0, "
                         f"got {grid_size}, {spline_order}")
    lo, hi = map(float, grid_range)
    if not hi > lo:
        raise ValueError(f"grid_range must be increasing, got {grid_range}")
    h = (hi - lo) / grid_size
    return lo + h * np.arange(-spline_order, grid_size + spline_order + 1,
                              dtype=np.float64)


def bspline_basis(x: jnp.ndarray, knots: np.ndarray, spline_order: int) -> jnp.ndarray:
    """Cox–de Boor B-spline basis, shape ``x.shape + (len(knots) - 1 - spline_order,)``.

    The order-0 indicators are piecewise constant, so autodiff differentiates
    each polynomial piece exactly; derivatives are correct everywhere except
    on the knots themselves (a measure-zero set).
    """
    g = jnp.asarray(knots, dtype=x.dtype)
    xe = x[..., None]
    B = ((xe >= g[:-1]) & (xe < g[1:])).astype(x.dtype)
    for p in range(1, spline_order + 1):
        left = (xe - g[:-(p + 1)]) / (g[p:-1] - g[:-(p + 1)]) * B[..., :-1]
        right = (g[p + 1:] - xe) / (g[p + 1:] - g[1:-p]) * B[..., 1:]
        B = left + right
    return B


# ── layers ───────────────────────────────────────────────────────────────────

class ChebyKANLayer(nn.Module):
    out_dim: int
    degree: int = 5

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        nb = self.degree + 1
        coef = self.param(
            "cheby_coef",
            nn.initializers.normal(stddev=1.0 / np.sqrt(in_dim * nb)),
            (in_dim, self.out_dim, nb))
        T = chebyshev_basis(jnp.tanh(x), self.degree)          # (N, in, nb)
        return jnp.einsum("nik,iok->no", T, coef)              # T_0 term acts as bias


class BSplineKANLayer(nn.Module):
    out_dim: int
    grid_size: int = 5
    spline_order: int = 3
    grid_range: Tuple[float, float] = (-1.0, 1.0)

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        knots = bspline_knots(self.grid_size, self.spline_order, self.grid_range)
        nb = self.grid_size + self.spline_order
        w_base = self.param("base_weight", nn.initializers.lecun_normal(),
                            (in_dim, self.out_dim))
        coef = self.param("spline_coef",
                          nn.initializers.normal(stddev=0.1 / np.sqrt(in_dim)),
                          (in_dim, self.out_dim, nb))
        bias = self.param("bias", nn.initializers.zeros, (self.out_dim,))
        u = jnp.tanh(x)
        B = bspline_basis(u, knots, self.spline_order)          # (N, in, nb)
        return nn.silu(u) @ w_base + jnp.einsum("nik,iok->no", B, coef) + bias


# ── networks ─────────────────────────────────────────────────────────────────

class ChebyKAN(nn.Module):
    """Chebyshev KAN (cPIKAN). ``layers=[in, h1, …, out]``; ``degree`` = K.

    Examples
    --------
    >>> net = ChebyKAN(layers=[2, 32, 32, 1], degree=5)   # doctest: +SKIP
    >>> u = net.apply(params, xy)        # (N, 2) -> (N, 1)  # doctest: +SKIP
    """
    layers: Sequence[int]
    degree: int = 5

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.layers[1:]):
            x = ChebyKANLayer(w, self.degree, name=f"layer_{i}")(x)
        return x


class BSplineKAN(nn.Module):
    """B-spline KAN (original KAN formulation).

    ``layers=[in, h1, …, out]``; ``grid_size`` intervals of order
    ``spline_order`` on ``grid_range``. See :func:`refine_bspline_kan` for
    grid extension between training runs.
    """
    layers: Sequence[int]
    grid_size: int = 5
    spline_order: int = 3
    grid_range: Tuple[float, float] = (-1.0, 1.0)

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.layers[1:]):
            x = BSplineKANLayer(w, self.grid_size, self.spline_order,
                                tuple(self.grid_range), name=f"layer_{i}")(x)
        return x


# ── grid extension (between runs, never inside one) ──────────────────────────

def refine_bspline_kan(params, grid_size: int, new_grid_size: int,
                       spline_order: int = 3,
                       grid_range: Tuple[float, float] = (-1.0, 1.0),
                       n_fit: int | None = None):
    """Refit a :class:`BSplineKAN`'s splines onto a finer grid.

    Every edge function φ_{i,o} is a 1-D spline on the fixed interval
    ``grid_range`` (layer inputs are ``tanh``-squashed into it), so it can be
    refitted by least squares on a dense sample of that interval -- no
    training data needed. When ``new_grid_size`` is a multiple of
    ``grid_size`` the old spline space is contained in the new one and the
    refit is exact to float precision; otherwise it is the L2-best
    approximation. Base weights and biases are copied unchanged.

    Returns a new params tree for
    ``BSplineKAN(layers, grid_size=new_grid_size, spline_order, grid_range)``.
    Re-initialise the optimizer for it (shapes changed) -- e.g. restore the
    refined params with ``solver.load_params`` / ``restore_checkpoint`` and
    train on.
    """
    if new_grid_size < 1:
        raise ValueError(f"new_grid_size must be >= 1, got {new_grid_size}")
    k_old = bspline_knots(grid_size, spline_order, grid_range)
    k_new = bspline_knots(new_grid_size, spline_order, grid_range)
    lo, hi = map(float, grid_range)
    m = n_fit or 8 * (new_grid_size + spline_order) + 64
    # Sample strictly inside [lo, hi): the basis is right-open at hi.
    xs = jnp.asarray(np.linspace(lo, hi, m, endpoint=False) + (hi - lo) / (2 * m))
    B_old = bspline_basis(xs, k_old, spline_order)              # (m, nb_old)
    B_new = bspline_basis(xs, k_new, spline_order)              # (m, nb_new)
    # Solve B_new @ C_new = B_old @ C_old for all (i, o) at once via the pseudo-inverse.
    P = jnp.linalg.pinv(B_new) @ B_old                          # (nb_new, nb_old)

    def refit(path, leaf):
        if getattr(path[-1], "key", None) == "spline_coef":
            if leaf.shape[-1] != B_old.shape[1]:
                raise ValueError(
                    f"spline_coef has {leaf.shape[-1]} bases, expected "
                    f"{B_old.shape[1]} for grid_size={grid_size}, "
                    f"spline_order={spline_order}")
            return jnp.einsum("ab,iob->ioa", P, leaf).astype(leaf.dtype)
        return leaf

    return jax.tree_util.tree_map_with_path(refit, params)
