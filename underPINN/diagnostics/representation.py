"""What the network and its input representation do (twolayer notes Sec. 4; pinn_math_notes Sec. 6.1).

    feature_gram          kappa of the feature Gram matrix of an input representation
    erank                 exp(entropy of the normalised singular values) of centred activations
    saturation            fraction of pre-activations with |z| > 2 (tanh' < 0.071)
    linear_probe          1 - R^2 of a target on the top-k principal directions of a layer
    linear_cka            centred kernel alignment between two activation sets (rotation invariant)
    ablation_importance   loss increase when a hidden unit is replaced by its mean activation
                          (ranks the units to plot; to size a layer use design.refit_pruning)
"""
from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .core import Problem


def feature_gram(phi, standardise: bool = True):
    """G = Phi~^T Phi~ / N, its eigenvalues (ascending) and condition number kappa(G).

    With ``standardise`` each column is mapped to zero mean and unit variance over the
    grid (population std, notes eq. 8), so G is the correlation matrix of the features
    and kappa measures correlation, not scale.

    The operator-mapped Gram G_DPhi = (D Phi~)^T (D Phi~) / N (notes Sec. 6.1) is the one
    the physics loss sees: apply the differential operator to the STANDARDISED features
    (rows D phi~(t_i), including the operator acting on the constant shift) and pass the
    result with ``standardise=False``; re-standardising would remove exactly the
    operator's scaling of each feature."""
    P = np.asarray(phi, np.float64)
    if standardise:
        P = (P - P.mean(0)) / P.std(0)
    G = P.T @ P / P.shape[0]
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    return G, ev, float(ev[-1] / ev[0])


def _centred_sv(h) -> np.ndarray:
    H = np.asarray(h, np.float64)
    return np.linalg.svd(H - H.mean(0), compute_uv=False)


def erank(h) -> float:
    """exp(-sum p_i log p_i) with p_i = s_i / sum s_j over the singular values of the centred
    activations (N x width). Compare only between layers of the same width."""
    s = _centred_sv(h)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def singular_values(h) -> np.ndarray:
    """Singular values of the centred activations, normalised by the largest."""
    s = _centred_sv(h)
    return s / s[0] if s[0] > 0 else s


def saturation(z, thr: float = 2.0) -> float:
    return float(np.mean(np.abs(np.asarray(z)) > thr))


def linear_probe(h, y, ks: Sequence[int] = (1, 2, 4, 8, 16)) -> Dict[int, float]:
    """1 - R^2 for y ~ b0 + b^T (top-k principal components of h), a scalar or vector target
    (nan for a constant target, where R^2 is undefined)."""
    H = np.asarray(h, np.float64)
    H = H - H.mean(0)
    Y = np.asarray(y, np.float64).reshape(len(H), -1)
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    S = H @ Vt.T
    tot = np.sum((Y - Y.mean(0)) ** 2)
    out = {}
    for k in ks:
        if k > S.shape[1]:
            continue
        A = np.concatenate([np.ones((len(S), 1)), S[:, :k]], 1)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        out[int(k)] = float(np.sum((Y - A @ beta) ** 2) / tot) if tot > 0 else float("nan")
    return out


def linear_cka(X, Y) -> float:
    """||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on centred activations."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def hidden_stats(problem: Problem, params, x) -> Dict[str, Dict[str, float]]:
    if problem.hidden_fn is None:
        raise ValueError("needs Problem.hidden_fn")
    out = {}
    for name, (z, h) in problem.hidden_fn(params, jnp.asarray(x)).items():
        out[name] = {"erank": erank(h), "saturation": saturation(z), "width": int(np.shape(h)[1]),
                     "sv": singular_values(h)}
    return out


def ablation_importance(problem: Problem, params, x, layer: str, next_layer: str, max_units: int = 128):
    """Mean-ablation importance of each unit of hidden layer ``layer``: loss(ablated) / loss.
    A unit's constant output (its mean activation over ``x``) is folded into the next layer's
    bias and its outgoing kernel row is zeroed, so the unit contributes a constant and nothing
    to any derivative. ``next_layer`` is the Dense that consumes ``layer``."""
    if problem.hidden_fn is None:
        raise ValueError("needs Problem.hidden_fn")
    _, h = problem.hidden_fn(params, jnp.asarray(x))[layer]
    hm = jnp.mean(h, 0)
    base = float(problem.loss(params))
    W = params["params"][next_layer]["kernel"]
    b = params["params"][next_layer]["bias"]

    @jax.jit
    def ablated(j):
        e = jax.nn.one_hot(j, W.shape[0], dtype=W.dtype)
        W2 = W * (1.0 - e)[:, None]
        b2 = b + (e * hm) @ W
        p = {**params, "params": {**params["params"], next_layer: {**params["params"][next_layer],
                                                                  "kernel": W2, "bias": b2}}}
        return problem.loss(p)
    n = min(W.shape[0], max_units)
    return np.array([float(ablated(j)) for j in range(n)]) / base
