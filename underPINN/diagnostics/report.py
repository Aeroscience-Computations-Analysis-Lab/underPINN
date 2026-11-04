"""Run the checkpoint diagnostics on a trained (or part-trained) parameter set.

    from underPINN.diagnostics import Config, Problem, run_checkpoint
    res0 = run_checkpoint(problem, params0, step=0)
    res = run_checkpoint(problem, params, params0=params0, ntk0=res0["ntk"]["K"], out_dir="diag", step=50000)

Each diagnostic runs independently: if one needs something the Problem does not provide, or
fails, the reason goes into ``res["skipped"]`` and the others still run (a diagnostic must never
kill a training run). Costs (P parameters, one fixed batch): Lanczos/SLQ = k Hessian-vector
products each (about 3x the cost of a loss gradient through the PDE); the exact Hessian (P <=
``full_hessian_max``) is P of them and P^2 floats; NTK = n_ntk residual Jacobians (stored as
n_ntk*n_eq x P floats); slices = n_dir*n_alpha (+ n2d^2) loss evaluations.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import curvature, design, kernel, landscape, representation, spectral
from .core import Problem, param_groups


@dataclass
class Config:
    lanczos_k: int = 40
    slq_k: int = 60
    slq_probes: int = 4
    eps: float = 1e-3                 # effective dimension n_eps = #{lambda_i > eps lambda_max}
    n_top: int = 10                   # kappa_top = lambda_1 / lambda_{n_top}  (notes: kappa_10)
    layer_k: int = 25
    n_ntk: int = 192
    n_dir: int = 7
    n_alpha: int = 141
    n2d: int = 0                      # 2-D slice grid (0 = skip; n2d^2 loss evaluations)
    hidden_points: int = 4096
    probe_target: Any = None          # callable x -> target values (e.g. the exact solution), optional
    refit_layer: Optional[str] = None  # name of the output Dense layer, e.g. "Dense_5"
    prune: bool = False               # also refit-prune the units feeding refit_layer
    x_dense: Any = None               # independent dense points: off-grid residual check, solution error
    reference_fn: Any = None          # callable x -> exact solution: relative L2 error on x_dense (and its
                                      # spectrum when x_dense is a uniform 1-D grid); needs Problem.apply_fn
    residual_weights: Any = None
    full_hessian_max: int = 3000      # exact Hessian (spectrum, Jacobi scaling, Gauss-Newton) when P <= this


def _run(res, name, fn):
    t0 = time.time()
    try:
        res[name] = fn()
        res.setdefault("seconds", {})[name] = round(time.time() - t0, 2)
    except Exception as e:                                   # noqa: BLE001  (never kill training)
        res.setdefault("skipped", {})[name] = f"{type(e).__name__}: {str(e)[:200]}"
        res["skipped_trace_" + name] = traceback.format_exc()[-600:]


def _uniform_1d_step(x) -> Optional[float]:
    x = np.asarray(x, np.float64)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        return None
    try:
        return spectral.grid_step(x)
    except ValueError:
        return None


def run_checkpoint(problem: Problem, params, params0=None, ntk0=None, cfg: Optional[Config] = None,
                   out_dir: Optional[str] = None, step: int = 0) -> Dict[str, Any]:
    """All checkpoint diagnostics for ``params``. ``params0`` (the initial parameters) adds the
    per-layer distance from initialisation; ``ntk0`` (``res["ntk"]["K"]`` of the step-0 call, same
    points) adds the kernel drift."""
    cfg = cfg or Config()
    res: Dict[str, Any] = {"step": step, "problem": problem.name}
    res["loss_terms"] = {k: float(v) for k, v in problem.terms(params).items()}
    res["loss"] = float(sum(res["loss_terms"].values()))
    P = int(ravel_pytree(params)[0].size)
    res["n_params"] = P
    cache: Dict[str, Any] = {}

    def curv():
        out: Dict[str, Any] = {"eps": cfg.eps, "n_top": cfg.n_top}
        if P <= cfg.full_hessian_max:
            H = cache["H"] = curvature.full_hessian(problem, params)
            ev = curvature.full_hessian_spectrum(problem, params, H)
            out.update(method="exact", lambda_max=float(ev[0]), lambda_min=float(ev[-1]), eigs=ev,
                       n_eff=int(np.sum(ev > cfg.eps * ev[0])), n_eff_std=0.0,
                       kappa_top=curvature.kappa_top(ev, cfg.n_top),
                       kappa_top_jacobi=curvature.kappa_top(curvature.jacobi_scaled_spectrum(H), cfg.n_top))
        else:
            top = curvature.top_eigenvalues(problem, params, cfg.n_top, cfg.lanczos_k)
            n, nodes, w = curvature.spectrum_slq(problem, params, cfg.slq_k, cfg.slq_probes)
            ne, nstd, lm = curvature.effective_dimension(n, nodes, w, cfg.eps, lam_max=float(top[0]))
            out.update(method="slq", lambda_max=lm, top=top, kappa_top=curvature.kappa_top(top, cfg.n_top),
                       n_eff=ne, n_eff_std=nstd, ritz=np.concatenate(nodes),
                       ritz_w=np.concatenate([x / len(w) for x in w]))
        return out
    _run(res, "curvature", curv)
    _run(res, "layer_sharpness", lambda: curvature.layer_sharpness(problem, params, cfg.layer_k))
    _run(res, "gauss_newton", lambda: curvature.gauss_newton_fidelity(
        problem, params, exact_max=cfg.full_hessian_max, k=cfg.lanczos_k, H=cache.get("H")))

    if params0 is not None:
        def dist0():
            f = np.asarray(ravel_pytree(params)[0], np.float64)
            f0 = np.asarray(ravel_pytree(params0)[0], np.float64)
            return {g: float(np.linalg.norm((f - f0)[m]) / max(np.linalg.norm(f0[m]), 1e-30))
                    for g, m in param_groups(params, problem.group_fn).items()}
        _run(res, "dist0", dist0)

    def kern():
        K, ev = kernel.ntk(problem, params, cfg.n_ntk, cfg.residual_weights)
        out = {"eigs": ev, "rank_1e-3": kernel.spectral_rank(ev), "K": K}
        if ntk0 is not None:
            out["drift"] = kernel.ntk_drift(K, ntk0)
            out["trace_ratio"] = float(np.trace(K) / np.trace(ntk0))               # kernel size vs initialisation
            out["alignment"] = float(np.sum(K * ntk0) / (np.linalg.norm(K) * np.linalg.norm(ntk0)))  # cosine of K, K0
        return out
    _run(res, "ntk", kern)

    def slices():
        a, f = landscape.slice_1d(problem, params, cfg.n_dir, cfg.n_alpha)
        out = {"alpha": a, "f": f, "extrema": [landscape.count_extrema(r) for r in f]}
        out["n_nonconvex"] = int(sum(e > 1 for e in out["extrema"]))
        if cfg.n2d:
            g, F = landscape.slice_2d(problem, params, cfg.n2d)
            out.update(grid=g, f2d=F)
        return out
    _run(res, "slices", slices)

    def hidden():
        x = np.asarray(problem.x_res)[:cfg.hidden_points]
        st = representation.hidden_stats(problem, params, x)
        if cfg.probe_target is not None:
            y = np.asarray(cfg.probe_target(jnp.asarray(x)))
            for name, (z, h) in problem.hidden_fn(params, jnp.asarray(x)).items():
                st[name]["probe"] = representation.linear_probe(np.asarray(h), y)
        return st
    _run(res, "hidden", hidden)

    if cfg.refit_layer:
        def refit():
            _, info = design.output_layer_refit(problem, params, cfg.refit_layer)
            return info
        _run(res, "output_refit", refit)
        if cfg.prune:
            _run(res, "pruning", lambda: design.refit_pruning(problem, params, cfg.refit_layer))
    if cfg.x_dense is not None:
        _run(res, "offgrid", lambda: design.offgrid_residual_ratio(problem, params, cfg.x_dense))
    if cfg.x_dense is not None and cfg.reference_fn is not None:
        def err():
            if problem.apply_fn is None:
                raise ValueError("needs Problem.apply_fn")
            xd = jnp.asarray(cfg.x_dense)
            u = np.asarray(problem.apply_fn(params, xd), np.float64).reshape(xd.shape[0], -1)
            e = u - np.asarray(cfg.reference_fn(xd), np.float64).reshape(u.shape)
            out = {"rel_l2": float(np.linalg.norm(e) / np.linalg.norm(u - e)), "max_abs": float(np.abs(e).max())}
            dt = _uniform_1d_step(cfg.x_dense)
            if dt is not None:
                out["omega"], out["error_amp"] = spectral.amplitude_spectrum(e, dt)
            return out
        _run(res, "error", err)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, f"ckpt_{step:07d}.npz"), **_flatten(res))
        with open(os.path.join(out_dir, f"ckpt_{step:07d}.json"), "w") as fh:
            json.dump(_jsonable(res), fh, indent=1)
    return res


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        else:
            try:
                out[key] = np.asarray(v)
            except Exception:                                # noqa: BLE001
                pass
    return out


def _jsonable(d, max_len: int = 12):
    """Scalars and short lists only (big arrays live in the npz)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = _jsonable(v, max_len)
        elif isinstance(v, (np.ndarray, list, tuple)):
            a = np.asarray(v)
            if a.size <= max_len and a.dtype != object:
                out[str(k)] = a.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[str(k)] = v.item()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[str(k)] = v
    return out
