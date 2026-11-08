"""Where do the residual and the error live? Coordinate profiles, causality, localisation.

residual_profile(problem, params, x, axis=-1)
    Mean squared PDE residual in bins along one input coordinate (time by default). With
    ``eps`` it also returns the causal weights of Wang, Sankaran & Perdikaris (2022),
    w_k = exp(-eps · Σ_{j<k} L_j): if w_k ≪ 1 for late bins, the network is fitting late
    times before early ones are resolved — the classic failure on transport problems
    (convection, wave), where an unresolved early state makes the late residual meaningless.
    ``growth`` = mean residual of the last quarter of bins / first quarter.

error_profile(problem, params, x, reference_fn, axis=-1)
    Same binning for the true error (needs ``apply_fn`` and a reference): error that grows
    along time while the residual does not is the signature of a solution that satisfies the
    PDE locally but has drifted (a "trivial" or phase-shifted solution).

residual_error_correlation(problem, params, x, reference_fn)
    Spearman rank correlation between |residual| and |error| point-wise. High: the residual is
    a usable error indicator, so residual-driven adaptive sampling (RAR-D) targets the right
    places. Low: it does not, and adding points where the residual is large will not fix
    the error.

residual_concentration(problem, params, x)
    Share of the total squared residual in the worst 10 % of points, and the Gini
    coefficient of r². Concentrated residual (shocks, layers, corners) is where adaptive
    sampling pays off; a uniform residual calls for capacity or optimisation instead.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np

from .core import Problem

__all__ = ["binned_mean", "residual_profile", "error_profile", "causal_weights",
           "residual_error_correlation", "residual_concentration"]


def binned_mean(values, coord, n_bins: int = 20, lo: Optional[float] = None, hi: Optional[float] = None):
    """(bin centres, mean of ``values`` per bin, count per bin) over equal-width bins of ``coord``."""
    v = np.asarray(values, np.float64).ravel()
    c = np.asarray(coord, np.float64).ravel()
    lo = c.min() if lo is None else lo
    hi = c.max() if hi is None else hi
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    s = np.bincount(idx, weights=v, minlength=n_bins)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), mean, cnt


def causal_weights(bin_losses, eps: float) -> np.ndarray:
    """w_k = exp(-eps · Σ_{j<k} L_j) for ordered bin losses (Wang, Sankaran & Perdikaris 2022)."""
    L = np.nan_to_num(np.asarray(bin_losses, np.float64))
    return np.exp(-eps * np.concatenate([[0.0], np.cumsum(L)[:-1]]))


def _growth(mean):
    m = np.asarray(mean, np.float64)
    q = max(1, len(m) // 4)
    return float(np.nanmean(m[-q:]) / max(np.nanmean(m[:q]), 1e-300))


def _sq_residual(problem: Problem, params, x, chunk: int = 4096):
    if problem.residual_fn is None:
        raise ValueError("needs Problem.residual_fn")
    x = jnp.asarray(x)
    out = [np.asarray(problem.residual_fn(params, x[i:i + chunk]), np.float64) for i in range(0, x.shape[0], chunk)]
    r = np.concatenate(out)
    return (r.reshape(r.shape[0], -1) ** 2).sum(1)


def _sq_error(problem: Problem, params, x, reference_fn: Callable, chunk: int = 4096):
    if problem.apply_fn is None:
        raise ValueError("needs Problem.apply_fn")
    x = jnp.asarray(x)
    u = np.concatenate([np.asarray(problem.apply_fn(params, x[i:i + chunk]), np.float64).reshape(min(chunk, x.shape[0] - i), -1)
                        for i in range(0, x.shape[0], chunk)])
    ref = np.asarray(reference_fn(x), np.float64).reshape(u.shape)
    return ((u - ref) ** 2).sum(1), (ref ** 2).sum(1)


def residual_profile(problem: Problem, params, x, axis: int = -1, n_bins: int = 20,
                     eps: Optional[float] = None) -> Dict:
    """Binned mean squared residual along coordinate ``axis`` of ``x`` (default: last = time)."""
    r2 = _sq_residual(problem, params, x)
    c, m, n = binned_mean(r2, np.asarray(x)[:, axis], n_bins)
    out = {"centres": c, "mean_sq_residual": m, "count": n, "growth": _growth(m)}
    if eps is not None:
        w = causal_weights(m, eps)
        out.update(causal_weights=w, min_causal_weight=float(w.min()), eps=eps)
    return out


def error_profile(problem: Problem, params, x, reference_fn: Callable, axis: int = -1,
                  n_bins: int = 20) -> Dict:
    """Binned relative squared error along coordinate ``axis`` (needs apply_fn + reference)."""
    e2, ref2 = _sq_error(problem, params, x, reference_fn)
    c, me, n = binned_mean(e2, np.asarray(x)[:, axis], n_bins)
    _, mr, _ = binned_mean(ref2, np.asarray(x)[:, axis], n_bins)
    rel = me / np.maximum(mr, 1e-300)
    q = max(1, n_bins // 4)
    return {"centres": c, "mean_sq_error": me, "rel_sq_error": rel, "count": n, "growth": _growth(rel),
            "early": float(np.sqrt(rel[np.isfinite(rel)][0])), "late": float(np.sqrt(np.nanmean(rel[-q:])))}


def _rank(a):
    a = np.asarray(a)
    r = np.empty(len(a))
    r[np.argsort(a, kind="mergesort")] = np.arange(len(a))
    return r


def residual_error_correlation(problem: Problem, params, x, reference_fn: Callable) -> Dict:
    """Spearman correlation of |residual| and |error| over the points ``x``."""
    r2 = _sq_residual(problem, params, x)
    e2, _ = _sq_error(problem, params, x, reference_fn)
    rr, re = _rank(r2), _rank(e2)
    rho = float(np.corrcoef(rr, re)[0, 1])
    return {"spearman": rho, "n": int(len(r2))}


def residual_concentration(problem: Problem, params, x, top: float = 0.10, n_worst: int = 5) -> Dict:
    """Share of Σr² in the worst ``top`` fraction of points, the Gini coefficient of r², the
    ratio max/median of |r|, and the coordinates of the ``n_worst`` worst points -- a residual
    spike at one location points to a singularity (network or data), not to under-sampling."""
    raw = _sq_residual(problem, params, x)
    order = np.argsort(raw)[::-1]
    worst = np.asarray(x)[order[:n_worst]]
    r2 = raw[order]
    tot = max(r2.sum(), 1e-300)
    k = max(1, int(round(top * len(r2))))
    a = np.sort(r2)
    n = len(a)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ a / (n * tot))
    return {"top_share": float(r2[:k].sum() / tot), "top": top, "gini": gini,
            "rms": float(np.sqrt(r2.mean())), "max": float(np.sqrt(r2[0])),
            "max_over_median": float(np.sqrt(r2[0] / max(np.median(r2), 1e-300))),
            "worst_points": worst.tolist(), "worst_values": np.sqrt(r2[:n_worst]).tolist()}
