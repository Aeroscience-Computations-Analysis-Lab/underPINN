"""Hessian diagnostics (pinn_math_notes Secs. 6.2, 7; twolayer notes Secs. 3.4, 5.3).

    sharpness              lambda_max of H = grad^2 L (gradient descent is stable only for eta < 2/lambda_max)
    effective dimension    n_eps = #{i : lambda_i > eps lambda_max}, eps = 1e-3: directions with
                           non-negligible curvature
    top-ten condition      kappa_10 = lambda_1 / lambda_10. The classical lambda_max / lambda_min is not
                           usable: an over-parameterised H has many eigenvalues within rounding of zero
    Jacobi-scaled kappa_10 kappa_10 of D^-1/2 H D^-1/2 with D = diag(H): an idealised proxy for Adam's
                           diagonal preconditioning. It removes per-parameter scale differences, not
                           the correlations a poor representation puts into the Jacobian
    per-layer curvature    lambda_max of the diagonal block H_ll of each layer
    Gauss-Newton fidelity  ||H - 2 J^T J||_2 / ||H||_2 with loss = ||rho||^2 and J = d rho / d theta
                           (eq. 17 of the notes): small means the curvature IS the Gram matrix of the
                           residual Jacobian, which is what carries the input representation

Small P: exact (``full_hessian``). Large P: Hessian-vector products only (no P x P matrix),
Lanczos with full reorthogonalisation for extreme eigenvalues and spectral norms, and
stochastic Lanczos quadrature (SLQ) for the spectrum, whose spread over probes is reported.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .core import Problem, param_groups


def make_hvp(problem: Problem, params):
    """jit-compiled ``hvp(flat, v)`` of the problem loss, plus the flat parameters."""
    f, flat, _ = problem.flat_loss(params)
    g = jax.grad(f)

    @jax.jit
    def hvp(x, v):
        return jax.jvp(g, (x,), (v,))[1]
    return hvp, flat


def make_gnvp(problem: Problem, params):
    """jit-compiled ``gnvp(flat, v) = 2 J^T J v`` (J = d rho / d theta), plus the flat parameters."""
    flat, unravel = ravel_pytree(params)

    def rho(x):
        return problem.rho_fn(unravel(x), problem.batch)

    @jax.jit
    def gnvp(x, v):
        _, jv = jax.jvp(rho, (x,), (v,))
        return 2.0 * jax.vjp(rho, x)[1](jv)[0]
    return gnvp, flat


def lanczos(matvec: Callable, n: int, k: int, seed: int = 0, mask: Optional[np.ndarray] = None,
            rademacher: bool = False, dtype=jnp.float32):
    """k Lanczos steps with full reorthogonalisation. ``matvec`` maps a length-n vector to H v.
    With ``mask`` the iteration lives in the coordinate subspace (diagonal block of H).
    Returns (ritz values ascending, weights = squares of first components of Ritz vectors)."""
    rng = np.random.default_rng(seed)
    v = rng.choice([-1.0, 1.0], n) if rademacher else rng.standard_normal(n)
    m = None if mask is None else np.asarray(mask, np.float64)
    if m is not None:
        v = v * m
    v /= np.linalg.norm(v)
    Q = np.zeros((k, n))
    alphas, betas = [], []
    q = v
    for j in range(k):
        Q[j] = q
        w = np.array(matvec(jnp.asarray(q, dtype)), np.float64)
        if m is not None:
            w = w * m
        a = float(q @ w)
        alphas.append(a)
        for _ in range(2):
            w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        b = float(np.linalg.norm(w))
        if j == k - 1 or b < 1e-9 * max(1.0, abs(a)):
            break
        betas.append(b)
        q = w / b
    nk = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:nk - 1], 1) + np.diag(betas[:nk - 1], -1)
    ev, U = np.linalg.eigh(T)
    return ev, U[0] ** 2


def sharpness(problem: Problem, params, k: int = 40, seed: int = 0) -> float:
    """lambda_max of the Hessian (largest Ritz value of k Lanczos steps)."""
    hvp, flat = make_hvp(problem, params)
    ev, _ = lanczos(lambda v: hvp(flat, v), flat.size, min(k, flat.size), seed, dtype=flat.dtype)
    return float(ev[-1])


def top_eigenvalues(problem: Problem, params, n_top: int = 10, k: int = 60, seed: int = 0) -> np.ndarray:
    """The ``n_top`` largest Hessian eigenvalues (descending), as Ritz values of one k-step
    Lanczos run. Extreme eigenvalues converge first; raise k if the n_top-th one matters."""
    hvp, flat = make_hvp(problem, params)
    ev, _ = lanczos(lambda v: hvp(flat, v), flat.size, min(k, flat.size), seed, dtype=flat.dtype)
    return ev[::-1][:n_top]


def kappa_top(eigs, n: int = 10) -> float:
    """kappa_n = lambda_1 / lambda_n of a descending spectrum; nan if lambda_n <= 0 or missing."""
    e = np.asarray(eigs, np.float64)
    if e.size < n or e[n - 1] <= 0:
        return float("nan")
    return float(e[0] / e[n - 1])


def spectrum_slq(problem: Problem, params, k: int = 60, n_probes: int = 4, seed: int = 0):
    """SLQ spectral density: Ritz nodes and weights of ``n_probes`` Rademacher probes.
    The weights of each probe sum to 1, so the density of eigenvalues is n * mean(weights)."""
    hvp, flat = make_hvp(problem, params)
    nodes, weights = [], []
    for s in range(n_probes):
        ev, w = lanczos(lambda v: hvp(flat, v), flat.size, min(k, flat.size), seed + s, rademacher=True,
                        dtype=flat.dtype)
        nodes.append(ev)
        weights.append(w)
    return flat.size, nodes, weights


def effective_dimension(n: int, nodes, weights, eps: float = 1e-3, lam_max: Optional[float] = None):
    """SLQ estimate of #{lambda_i > eps lambda_max}, its spread over probes, and the lambda_max
    used: the largest Ritz value seen (a lower bound of the true one), or ``lam_max`` if that
    better estimate (e.g. from ``sharpness``) is larger."""
    lm = max(float(v[-1]) for v in nodes)
    if lam_max is not None:
        lm = max(lm, float(lam_max))
    counts = [n * float(np.sum(w[v > eps * lm])) for v, w in zip(nodes, weights)]
    return float(np.mean(counts)), float(np.std(counts)), lm


def full_hessian(problem: Problem, params) -> np.ndarray:
    """Exact symmetrised Hessian 1/2 (H + H^T) in float64; for small P only (P x P matrix)."""
    f, flat, _ = problem.flat_loss(params)
    H = np.asarray(jax.hessian(f)(flat), np.float64)
    return 0.5 * (H + H.T)


def full_hessian_spectrum(problem: Problem, params, H: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact eigenvalues (descending) of the symmetrised Hessian; for small P only."""
    H = full_hessian(problem, params) if H is None else H
    return np.sort(np.linalg.eigvalsh(H))[::-1]


def jacobi_scaled_spectrum(H: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of D^-1/2 H D^-1/2 with D = |diag(H)|. Parameters with no
    curvature at all (zero diagonal, e.g. dead units) are left out rather than divided by zero."""
    H = np.asarray(H, np.float64)
    d = np.abs(np.diag(H))
    keep = d > 0
    s = 1.0 / np.sqrt(d[keep])
    Hs = H[np.ix_(keep, keep)] * s[:, None] * s[None, :]
    return np.sort(np.linalg.eigvalsh(0.5 * (Hs + Hs.T)))[::-1]


def layer_sharpness(problem: Problem, params, k: int = 25, seed: int = 0) -> Dict[str, float]:
    """lambda_max of each layer's diagonal Hessian block."""
    hvp, flat = make_hvp(problem, params)
    out = {}
    for lab, mask in param_groups(params, problem.group_fn).items():
        ev, _ = lanczos(lambda v: hvp(flat, v), flat.size, min(k, int(mask.sum())), seed, mask=mask, dtype=flat.dtype)
        out[lab] = float(ev[-1])
    return out


def _gauss_newton_matrix(problem: Problem, params, chunk: int = 64) -> np.ndarray:
    """2 J^T J as a dense P x P float64 matrix, built from P Gauss-Newton-vector products
    (memory P^2 + chunk x rows, never the full rows x P Jacobian)."""
    gnvp, flat = make_gnvp(problem, params)
    eye = jnp.eye(flat.size, dtype=flat.dtype)
    cols = jax.jit(jax.vmap(lambda e: gnvp(flat, e)))
    G = np.concatenate([np.asarray(cols(eye[i:i + chunk]), np.float64) for i in range(0, flat.size, chunk)], 0)
    return 0.5 * (G + G.T)


def gauss_newton_fidelity(problem: Problem, params, exact: Optional[bool] = None, exact_max: int = 3000,
                          k: int = 40, seed: int = 0, H: Optional[np.ndarray] = None) -> Dict[str, float]:
    """||H - 2 J^T J||_2 / ||H||_2 with loss = sum(rho**2) and J = d rho / d theta.

    H = 2 J^T J + 2 sum_l rho_l grad^2 rho_l, so the ratio measures the second term: near a
    well-trained minimum the residuals are small and the curvature is the Gauss-Newton matrix.
    Exact for P <= ``exact_max`` (dense H, which can be passed in, and 2 J^T J); otherwise
    each spectral norm is the largest |Ritz value| of a k-step Lanczos run on the symmetric
    operator. Requires ``problem.rho_fn`` consistent with the loss (checked)."""
    problem.check_rho(params)
    flat, _ = ravel_pytree(params)
    P = int(flat.size)
    if exact is None:
        exact = P <= exact_max
    if exact:
        Hm = full_hessian(problem, params) if H is None else np.asarray(H, np.float64)
        E = Hm - _gauss_newton_matrix(problem, params)
        n_e, n_h = np.linalg.norm(E, 2), np.linalg.norm(Hm, 2)
        method = "exact"
    else:
        hvp, _ = make_hvp(problem, params)
        gnvp, _ = make_gnvp(problem, params)
        kk = min(k, P)
        ev_e, _ = lanczos(lambda v: hvp(flat, v) - gnvp(flat, v), P, kk, seed, dtype=flat.dtype)
        ev_h, _ = lanczos(lambda v: hvp(flat, v), P, kk, seed, dtype=flat.dtype)
        n_e = max(abs(ev_e[0]), abs(ev_e[-1]))
        n_h = max(abs(ev_h[0]), abs(ev_h[-1]))
        method = "lanczos"
    return {"gn_rel_err": float(n_e / n_h), "norm_H": float(n_h), "norm_H_minus_GN": float(n_e), "method": method}
