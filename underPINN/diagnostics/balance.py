"""Loss-term balance: which constraint does training actually serve?

A PINN optimises a weighted sum of terms (PDE residual, initial, boundary, data). Most PINN
failures are balance failures: one term's gradient dwarfs or opposes the others, so the
optimiser fits it and starves the rest. Two views:

gradient_conflict(problem, params)
    g_i = ∇θ (term i). Reports ‖g_i‖, the pairwise cosine cos(g_i, g_j) (negative = the
    terms pull against each other), each term's share of the actual step direction
    s_i = g_i·g / ‖g‖² (Σ s_i = 1), and learning-rate-annealing factors
    λ_i = max|g_pde| / mean|g_i| (Wang, Teng & Perdikaris 2021): multiply term i's weight by
    λ_i to equalise gradient magnitudes, and the cancellation ‖Σ g_i‖ / Σ‖g_i‖ (near 0 at a
    stationary point, where opposing term gradients are expected rather than pathological).
    Needs only ``loss_terms_fn``.

term_ntk(problem, params)
    Neural-tangent kernel restricted to each loss block (needs ``Problem.blocks``, i.e. a
    problem from :func:`~underPINN.diagnostics.blocks.problem_from_blocks`). Under gradient
    flow block b's residual decays at rates set by w_b·K_bb, so its mean diagonal
    m_b = tr(K_bb)/n_b is that block's convergence speed. Reports m_b, the effective rate
    w_b·m_b, the imbalance max/min of the effective rates, and NTK-balanced weights
    λ_b = Σ_c m_c / m_b (Wang, Yu & Perdikaris 2022), and each block's eigenvalues.
"""
from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .core import Problem
from .blocks import _rows

__all__ = ["gradient_conflict", "term_ntk", "block_jacobian"]


def gradient_conflict(problem: Problem, params, pde_term: Optional[str] = None) -> Dict:
    """Per-term gradient norms, pairwise cosines, step shares and annealing factors."""
    flat, unravel = ravel_pytree(params)
    names = list(problem.terms(params).keys())

    def vec(f):
        t = problem.terms(unravel(f))
        return jnp.stack([t[n] for n in names])
    G = np.asarray(jax.jacrev(vec)(flat), np.float64)                    # (T, P)
    norms = np.linalg.norm(G, axis=1)
    safe = np.maximum(norms, 1e-300)
    cos = (G @ G.T) / np.outer(safe, safe)
    g = G.sum(0)
    gg = float(g @ g)
    share = (G @ g) / max(gg, 1e-300)
    ref = pde_term if pde_term in names else names[0]
    gmax = np.abs(G[names.index(ref)]).max()
    lra = {n: float(gmax / max(np.abs(G[i]).mean(), 1e-300)) for i, n in enumerate(names)}
    off = cos[np.triu_indices(len(names), 1)] if len(names) > 1 else np.array([1.0])
    cancellation = float(np.sqrt(gg) / max(norms.sum(), 1e-300))   # 1: aligned, ~0: terms cancel
    return {"names": names, "grad_norm": {n: float(v) for n, v in zip(names, norms)},
            "cancellation": cancellation,
            "cos": cos, "min_cos": float(off.min()),
            "share": {n: float(v) for n, v in zip(names, share)},
            "norm_ratio": float(norms.max() / max(norms.min(), 1e-300)),
            "lra_factor": lra, "lra_reference": ref}


def block_jacobian(problem: Problem, params, block: str, n_max: int = 64, chunk: int = 16) -> np.ndarray:
    """Jacobian of block ``block``'s residual rows at its first ``n_max`` points: (rows, P)."""
    if not problem.blocks:
        raise ValueError("needs Problem.blocks (build the problem with problem_from_blocks)")
    blk = problem.blocks[block]
    flat, unravel = ravel_pytree(params)
    X = jnp.asarray(blk.points)[:n_max]

    def r1(v, xi):
        return _rows(blk.fn(unravel(v), xi[None]))[0]
    jac = jax.jit(jax.vmap(jax.jacrev(r1), in_axes=(None, 0)))
    rows = [np.asarray(jac(flat, X[i:i + chunk])).reshape(-1, flat.size) for i in range(0, X.shape[0], chunk)]
    return np.concatenate(rows).astype(np.float64)


def term_ntk(problem: Problem, params, n_max: int = 64) -> Dict:
    """Per-block NTK: convergence speed of each loss term and NTK-balanced weights."""
    if not problem.blocks:
        raise ValueError("needs Problem.blocks (build the problem with problem_from_blocks)")
    out = {"mean_diag": {}, "effective_rate": {}, "eigs": {}, "weight": {}}
    for name, blk in problem.blocks.items():
        J = block_jacobian(problem, params, name, n_max)
        K = J @ J.T
        out["mean_diag"][name] = float(np.trace(K) / K.shape[0])
        out["eigs"][name] = np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))[::-1]
        out["weight"][name] = float(blk.weight)
        out["effective_rate"][name] = float(blk.weight * out["mean_diag"][name])
    m = out["mean_diag"]
    tot = sum(m.values())
    out["ntk_weights"] = {n: tot / max(v, 1e-300) for n, v in m.items()}
    r = np.array(list(out["effective_rate"].values()))
    out["imbalance"] = float(r.max() / max(r.min(), 1e-300))
    out["slowest"] = min(out["effective_rate"], key=out["effective_rate"].get)
    return out
