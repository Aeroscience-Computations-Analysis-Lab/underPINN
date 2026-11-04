"""Residual neural tangent kernel (twolayer notes Sec. 3.5).

With J_ij = d r_theta(x_i) / d theta_j (rows: collocation points x equations) the residual
NTK is K = J J^T / N. Under gradient flow on the residual loss, r' = -(2/N) J J^T r, so the
component of r along eigenvector i of K decays at a rate proportional to lambda_i(K): a
fast-decaying spectrum means most residual patterns are learned very slowly. The drift
||K_k - K_0||_F / ||K_0||_F says whether the kernel (lazy) picture describes training
(small) or features are being learned (large).

The kernel uses ``problem.residual_fn`` at ``problem.x_res`` (so it sees the PDE residual
only, not the boundary or penalty terms) and at most ``n_max`` points, since J is stored
(rows x P floats). Compare kernels only on the same points (drift needs the same rows).
K is formed in float64 on the host: a float32 product on GPU may run in TF32 and would
corrupt the small eigenvalues the spectrum is about.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .core import Problem


def residual_jacobian(problem: Problem, params, n_max: int = 192, chunk: int = 16, residual_weights=None):
    """J (rows x P, point-major: row = point * n_eq + equation) of the residual at x_res[:n_max]."""
    if problem.residual_fn is None:
        raise ValueError("needs Problem.residual_fn")
    flat, unravel = ravel_pytree(params)
    x = jnp.asarray(problem.x_res)[:n_max]
    w = None if residual_weights is None else jnp.asarray(residual_weights, flat.dtype)

    def r1(v, xi):
        r = problem.residual_fn(unravel(v), xi[None])[0]
        return r if w is None else r * w
    jac = jax.jit(jax.vmap(jax.jacrev(r1), in_axes=(None, 0)))
    rows = [np.asarray(jac(flat, x[i:i + chunk])).reshape(-1, flat.size) for i in range(0, x.shape[0], chunk)]
    return np.concatenate(rows)


def ntk(problem: Problem, params, n_max: int = 192, residual_weights=None):
    """K = J J^T / N (N = number of rows) and its eigenvalues (descending)."""
    J = np.asarray(residual_jacobian(problem, params, n_max, residual_weights=residual_weights), np.float64)
    K = J @ J.T / J.shape[0]
    return K, np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))[::-1]


def ntk_drift(K, K0) -> float:
    """||K - K0||_F / ||K0||_F."""
    return float(np.linalg.norm(K - K0) / np.linalg.norm(K0))


def spectral_rank(eigs, rel: float = 1e-3) -> int:
    """Number of eigenvalues above rel * largest (``eigs`` descending)."""
    eigs = np.asarray(eigs)
    return int(np.sum(eigs > rel * eigs[0]))
