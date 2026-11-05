"""Checks that inform network and training design.

output_layer_refit
    With the hidden layers fixed, the output u = W h + b is linear in the output-layer
    parameters s. For a linear PDE with linear boundary terms rho(s) = J s + c is affine,
    so min ||rho||^2 over s is an exact linear least-squares problem and one solve gives
    its optimum: a free check of whether training converged the last layer. For a
    nonlinear residual (Euler: positivity map and fluxes) the same solve is one
    Gauss-Newton step, so it is followed by a backtracking line search, and the loss the
    linear model predicted is reported next to the loss actually reached (they agree for
    a linear problem). The solve is an SVD least-squares solve in float64 on J itself,
    not on the normal equations, which would square its condition number.

refit_pruning
    Effective width: remove units of the last hidden layer one at a time, each time the
    one whose removal -- after refitting the output layer -- raises the loss least. A
    network that keeps its loss with far fewer units is wider than its representation
    needs. Plain mean-ablation without a refit is not a substitute: near a minimum,
    removing any unit raises the loss by orders of magnitude.

offgrid_residual_ratio
    RMS of the residual on a dense independent point set over its RMS on the collocation
    points. A ratio well above 1 means the network fits the points it was trained on at
    the expense of the gaps (too few points, or points that should follow the residual).
"""
from __future__ import annotations

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten_with_path, tree_unflatten

from .core import Problem, _key_name, param_groups


def _layer_jacobian(problem: Problem, params, layer: str, chunk: int = 64):
    """Jacobian of rho w.r.t. the parameters of ``layer`` (float64, rows x P_layer), by forward
    mode in column chunks, with everything needed to evaluate rho on modified layer values."""
    flat, unravel = ravel_pytree(params)
    groups = param_groups(params, problem.group_fn)
    if layer not in groups:
        raise KeyError(f"no parameter group {layer!r}; groups are {list(groups)}")
    idx = jnp.asarray(np.where(groups[layer])[0])
    sub0 = flat[idx]

    def rho_sub(s):
        return problem.rho_fn(unravel(flat.at[idx].set(s)), problem.batch)

    eye = jnp.eye(sub0.size, dtype=sub0.dtype)
    jvp = jax.jit(jax.vmap(lambda e: jax.jvp(rho_sub, (sub0,), (e,))[1]))
    J = np.concatenate([np.asarray(jvp(eye[i:i + chunk]), np.float64) for i in range(0, sub0.size, chunk)], 0).T
    rho0 = np.asarray(rho_sub(sub0), np.float64)
    return flat, unravel, idx, sub0, rho0, J, rho_sub


def _lstsq(J: np.ndarray, rhs: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """argmin_x ||J x - rhs||^2 (+ ridge * mean(diag J^T J) * ||x||^2), by SVD (minimum norm
    if J is rank deficient)."""
    if ridge > 0:
        lam = ridge * float(np.sum(J * J)) / J.shape[1]
        J = np.vstack([J, np.sqrt(lam) * np.eye(J.shape[1])])
        rhs = np.concatenate([rhs, np.zeros(J.shape[1])])
    return np.linalg.lstsq(J, rhs, rcond=None)[0]


def output_layer_refit(problem: Problem, params, layer: str, ridge: float = 0.0, max_halvings: int = 10):
    """Least-squares refit of the parameters of ``layer`` (normally the output Dense).
    Returns (new params, info) with the loss before and after, the loss the linear model
    predicted for the full step, the accepted step length and before/after."""
    problem.check_rho(params)
    flat, unravel, idx, sub0, rho0, J, rho_sub = _layer_jacobian(problem, params, layer)
    delta = _lstsq(J, -rho0, ridge)
    before = float(rho0 @ rho0)
    predicted = float(np.sum((rho0 + J @ delta) ** 2))
    d = jnp.asarray(delta, sub0.dtype)
    t, after, best = 1.0, before, 0.0
    for _ in range(max_halvings):
        trial = float(jnp.sum(rho_sub(sub0 + t * d) ** 2))
        if trial < before:
            after, best = trial, t
            break
        t *= 0.5
    new = unravel(flat.at[idx].set(sub0 + best * d))
    return new, {"loss_before": before, "loss_after": after, "loss_predicted_full_step": predicted,
                 "step": best, "ratio": before / after if after > 0 else float("inf")}


def _unit_of_layer_entry(problem: Problem, params, layer: str, idx) -> np.ndarray:
    """For each entry of the layer's flat parameter block: index of the input unit it belongs
    to (row j of the layer's kernel, shape (n_units, ...)), or -1 (bias and other leaves)."""
    leaves, treedef = tree_flatten_with_path(params)
    kpos = [i for i, (p, _) in enumerate(leaves) if problem.group_fn(p) == layer and _key_name(p[-1]) == "kernel"]
    if len(kpos) != 1:
        raise ValueError(f"layer {layer!r} must hold exactly one 'kernel' (a Dense layer), found {len(kpos)}")
    marked = [jnp.zeros(v.shape, jnp.float32) for _, v in leaves]
    W = leaves[kpos[0]][1]
    rows = jnp.arange(1, W.shape[0] + 1, dtype=jnp.float32).reshape((-1,) + (1,) * (W.ndim - 1))
    marked[kpos[0]] = jnp.broadcast_to(rows, W.shape)
    unit_flat, _ = ravel_pytree(tree_unflatten(treedef, marked))
    return np.rint(np.asarray(unit_flat)[np.asarray(idx)]).astype(int) - 1


def refit_pruning(problem: Problem, params, layer: str, min_units: int = 1, ridge: float = 1e-12) -> Dict[str, Any]:
    """Greedy backward elimination of the units feeding ``layer`` (the output Dense: one unit =
    one row of its kernel), refitting ``layer`` by least squares after every removal.

    Returns a dict of lists, one entry per width from all units down to ``min_units``:
      units      number of units kept
      loss       refitted loss with those units (linear least-squares value)
      loss_true  loss actually evaluated with those parameters (equals ``loss`` when the
                 residual is linear in the output layer; otherwise the linear model's error)
      removed    unit removed at each step (one fewer entry than ``units``)
    plus ``loss_trained``, the loss of the network as trained, before any refit.

    The unit to remove is chosen with the exact least-squares downdate: with A = J_S^T J_S
    over the active columns S and x the current solution, dropping the columns g of one unit
    raises the loss by x_g^T ([A^-1]_gg)^-1 x_g. ``ridge`` (relative to mean(diag A)) only
    keeps that inverse finite when units are exactly redundant; every recorded loss is
    re-solved directly. Cost: one (|S| x |S|) inverse per step, independent of the number
    of collocation points. Requires ``problem.rho_fn`` consistent with the loss (checked)."""
    problem.check_rho(params)
    flat, unravel, idx, sub0, rho0, J, rho_sub = _layer_jacobian(problem, params, layer)
    c = rho0 - J @ np.asarray(sub0, np.float64)           # rho(s) = J s + c for a linear problem
    unit = _unit_of_layer_entry(problem, params, layer, idx)
    n_units = int(unit.max()) + 1
    fixed = np.where(unit < 0)[0]
    cols = [np.where(unit == u)[0] for u in range(n_units)]
    A, b = J.T @ J, J.T @ c
    min_units = max(int(min_units), 0 if fixed.size else 1)   # a layer without bias keeps >= 1 unit

    def solve(active: List[int]):
        S = np.concatenate([fixed] + [cols[u] for u in active]).astype(int)
        s = np.zeros(J.shape[1])
        s[S] = _lstsq(J[:, S], -c)
        return s, float(np.sum((J @ s + c) ** 2))

    def true_loss(s):
        return float(jnp.sum(rho_sub(jnp.asarray(s, sub0.dtype)) ** 2))

    active = list(range(n_units))
    s, L = solve(active)
    out: Dict[str, Any] = {"units": [n_units], "loss": [L], "loss_true": [true_loss(s)], "removed": [],
                           "loss_trained": float(rho0 @ rho0)}
    while len(active) > min_units:
        S = np.concatenate([fixed] + [cols[u] for u in active]).astype(int)
        ASS = A[np.ix_(S, S)]
        Ainv = np.linalg.inv(ASS + ridge * np.trace(ASS) / len(S) * np.eye(len(S)))
        x = -Ainv @ b[S]
        start, inc = len(fixed), []
        for u in active:
            g = np.arange(start, start + len(cols[u]))
            start += len(cols[u])
            inc.append(float(x[g] @ np.linalg.solve(Ainv[np.ix_(g, g)], x[g])))
        drop = active[int(np.argmin(inc))]
        active.remove(drop)
        s, L = solve(active)
        out["units"].append(len(active))
        out["loss"].append(L)
        out["loss_true"].append(true_loss(s))
        out["removed"].append(drop)
    return out


def offgrid_residual_ratio(problem: Problem, params, x_dense, n_max: int = 8192):
    """RMS of the residual on ``x_dense`` over its RMS on the collocation points ``x_res``."""
    if problem.residual_fn is None:
        raise ValueError("needs Problem.residual_fn")

    def rms(x):
        return float(jnp.sqrt(jnp.mean(problem.residual_fn(params, jnp.asarray(x)) ** 2)))
    xc, xd = jnp.asarray(problem.x_res)[:n_max], jnp.asarray(x_dense)[:n_max]
    c, d = rms(xc), rms(xd)
    return {"rms_collocation": c, "rms_dense": d, "ratio": d / c}
