"""Modal (sine-series) analysis for Dirichlet problems on an interval.

On [x0, x0 + L] with zero boundary values, sin(n pi (x - x0) / L) diagonalises -d^2/dx^2, so
the error of a 1-D network splits into modes with coefficients
c_n = (2 / L) int u(x) sin(n pi (x - x0) / L) dx.

    sine_coefficients       c_n of samples u(x), n = 1..n_modes (trapezoidal rule)
    modal_error             delta a_n = c_n(u_theta - u*): which modes carry the error
    modal_kernel            mu_n = ||grad_theta c_n||^2, the diagonal of the network's tangent kernel
                            in the sine basis: how fast the network can move mode n. Set by the
                            representation (and architecture and initialisation)
    laplacian_mode_weights  w_n = (n pi / L)^4 / S^2: how much the strong-form loss of -u'' = f,
                            residual scaled by S, weighs mode n. Set by the operator; Parseval gives
                            L_r ~ 1/2 sum_n w_n delta a_n^2

Under gradient flow with a diagonal, fixed kernel, delta a_n(t) ~ delta a_n(0) exp(-mu_n w_n t).
Adam and feature learning break both assumptions, so use the product mu_n w_n to order the
modes, not to predict iteration counts. A small error in a mode with tiny w_n (the slow
modes here: w_2 / w_50 = 2.6e-6) is nearly invisible to the loss, whatever the representation:
check such errors against a reference or the Green's-function bound.
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def _coords(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError("modal analysis needs 1-D points: shape (N,) or (N, 1)")
    if np.any(np.diff(x) <= 0):
        raise ValueError("points must be strictly increasing")
    return x


def sine_projection(x, n_modes: int, x0: Optional[float] = None, length: Optional[float] = None) -> np.ndarray:
    """Q (n_modes x N) with c = Q u for samples u(x); [x0, x0 + length] defaults to [x[0], x[-1]]."""
    x = _coords(x)
    x0 = float(x[0]) if x0 is None else float(x0)
    L = float(x[-1]) - x0 if length is None else float(length)
    w = np.zeros_like(x)
    d = np.diff(x)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    n = np.arange(1, n_modes + 1)
    return (2.0 / L) * np.sin(np.pi * np.outer(n, (x - x0) / L)) * w[None, :]


def sine_coefficients(u, x, n_modes: int, x0: Optional[float] = None, length: Optional[float] = None) -> np.ndarray:
    """c_n, n = 1..n_modes, of samples ``u`` (N,) or (N, k) on ``x``."""
    Q = sine_projection(x, n_modes, x0, length)
    U = np.asarray(u, np.float64).reshape(Q.shape[1], -1)
    c = Q @ U
    return c[:, 0] if c.shape[1] == 1 else c


def modal_error(u_theta, u_star, x, n_modes: int, x0: Optional[float] = None,
                length: Optional[float] = None) -> np.ndarray:
    """delta a_n = c_n(u_theta) - c_n(u*), n = 1..n_modes."""
    return sine_coefficients(np.asarray(u_theta, np.float64) - np.asarray(u_star, np.float64), x, n_modes, x0, length)


def modal_kernel(apply_fn: Callable, params, x, n_modes: int, full: bool = False,
                 x0: Optional[float] = None, length: Optional[float] = None):
    """mu_n = ||grad_theta c_n(theta)||^2 for c(theta) = Q u_theta(x), n = 1..n_modes.
    ``apply_fn(params, x)`` takes points (N, 1) and returns (N,) or (N, 1). With ``full`` also
    returns Theta_nm = grad c_n . grad c_m (n_modes x n_modes)."""
    xs = _coords(x)
    flat, unravel = ravel_pytree(params)
    Q = jnp.asarray(sine_projection(xs, n_modes, x0, length), flat.dtype)
    xin = jnp.asarray(xs[:, None], flat.dtype)

    def coeffs(v):
        u = jnp.reshape(apply_fn(unravel(v), xin), (-1,))
        return jnp.dot(Q, u, precision=jax.lax.Precision.HIGHEST)
    G = np.asarray(jax.jacrev(coeffs)(flat), np.float64)                  # (n_modes, P)
    mu = np.sum(G * G, axis=1)
    return (mu, G @ G.T) if full else mu


def laplacian_mode_weights(n_modes: int, length: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """w_n = (n pi / L)^4 / S^2, n = 1..n_modes: the strong-form weight of mode n for -u'' = f
    with the residual divided by S. By Parseval, the mean squared residual of an error with
    sine coefficients da_n is (1/2) sum_n w_n da_n^2."""
    n = np.arange(1, n_modes + 1)
    return (n * np.pi / length) ** 4 / scale ** 2
