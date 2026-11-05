"""Training monitor: attach to ANY loop with one call per step.

    mon = TrainingMonitor(problem, params0, every=200)
    for step in range(n):
        mon.update(step, params)          # theta_k, before the update (L recorded is L(theta_k))
        params, ... = update(...)
    mon.save(path)

Every ``every`` steps it evaluates, on the problem's fixed diagnostic batch:
    terms        each weighted loss term: the term that stalls highest limits training; a term
                 that drops to round-off is over-weighted relative to what the others can reach
    grad_term    ||grad of each term|| (all of them in one backward pass): which term drives the
                 update; compare with ``terms`` before reweighting
    update       rho_l = ||theta_l(k) - theta_l(k')|| / ||theta_l(k')|| per layer, k' the previous
                 logged step (every=1 gives the per-step ratio): layers orders of magnitude
                 below the rest are effectively frozen; late values above ~1e-2 mean the steps are
                 too large
    dist0        delta_l = ||theta_l(k) - theta_l(0)|| / ||theta_l(0)|| per layer: lazy (<<1) vs
                 feature learning (~1)
    best         the logged iterate with the lowest total on the diagnostic batch (every=1 for the
                 true argmin over all iterates)
The diagnostic loss is whatever ``problem`` defines (normally the steady objective), whatever
objective the training loop is currently minimising (for example an implicit-Euler step).

``best_so_far`` and ``iterations_to`` turn a recorded loss history into efficiency measures:
the running best loss and the iterations needed to reach a target.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .core import Problem, param_groups


def best_so_far(losses) -> np.ndarray:
    """b_k = min_{j <= k} L(theta_j): unaffected by Adam's transient loss spikes."""
    return np.minimum.accumulate(np.asarray(losses, np.float64))


def iterations_to(losses, eps: float, steps=None) -> Optional[int]:
    """k(eps) = min{k : L(theta_k) <= eps}, or None if the level is never reached.
    ``steps`` gives the iteration number of each entry (default 0, 1, 2, ...)."""
    hit = np.nonzero(np.asarray(losses, np.float64) <= eps)[0]
    if hit.size == 0:
        return None
    return int(hit[0]) if steps is None else int(np.asarray(steps)[hit[0]])


def _masked_norm(v, m) -> float:
    return float(jnp.linalg.norm(jnp.where(m, v, 0.0)))


class TrainingMonitor:
    def __init__(self, problem: Problem, params0, every: int = 200, grad_terms: bool = True):
        self.problem, self.every, self.grad_terms = problem, every, grad_terms
        self.flat0, self.unravel = ravel_pytree(params0)
        self.groups = {k: jnp.asarray(v) for k, v in param_groups(params0, problem.group_fn).items()}
        self.names = list(problem.terms(params0).keys())

        def terms_vec(flat):
            t = problem.terms(self.unravel(flat))
            return jnp.stack([t[n] for n in self.names])
        self._terms = jax.jit(terms_vec)
        self._jac = jax.jit(jax.jacrev(terms_vec)) if grad_terms else None
        self._prev = self.flat0
        self.rec: Dict[str, List] = {"step": [], "wall": [], "terms": [], "grad_term": [], "update": [],
                                     "dist0": [], "grad_layer": [], "train_loss": []}
        self.best_loss, self.best_step, self.best_flat = float("inf"), -1, self.flat0
        self._t0 = time.time()

    def reset_best(self):
        """Call at a change of training stage (the best-so-far of the previous stage is not comparable)."""
        self.best_loss, self.best_step = float("inf"), -1

    def update(self, step: int, params, train_loss=None, force: bool = False):
        if not force and step % self.every:
            return
        flat, _ = ravel_pytree(params)
        t = np.asarray(self._terms(flat), np.float64)
        tot = float(t.sum())
        if tot < self.best_loss:
            self.best_loss, self.best_step, self.best_flat = tot, step, flat
        up = {g: _masked_norm(flat - self._prev, m) / max(_masked_norm(self._prev, m), 1e-30)
              for g, m in self.groups.items()}
        d0 = {g: _masked_norm(flat - self.flat0, m) / max(_masked_norm(self.flat0, m), 1e-30)
              for g, m in self.groups.items()}
        self.rec["step"].append(step)
        self.rec["wall"].append(time.time() - self._t0)
        self.rec["terms"].append(t)
        self.rec["update"].append([up[g] for g in self.groups])
        self.rec["dist0"].append([d0[g] for g in self.groups])
        self.rec["train_loss"].append(np.nan if train_loss is None else float(train_loss))
        if self._jac is not None:
            Jt = np.asarray(self._jac(flat), np.float64)                      # (n_terms, P)
            self.rec["grad_term"].append(np.linalg.norm(Jt, axis=1))
            g = Jt.sum(0)
            self.rec["grad_layer"].append([float(np.linalg.norm(g[np.asarray(m)])) for m in self.groups.values()])
        self._prev = flat

    def best_params(self):
        return self.unravel(self.best_flat)

    def total(self) -> np.ndarray:
        """Total diagnostic loss at each logged step."""
        return np.array([t.sum() for t in self.rec["terms"]])

    def save(self, path: str):
        d = {k: np.array(v) for k, v in self.rec.items() if len(v)}
        np.savez_compressed(path, term_names=np.array(self.names), layer_names=np.array(list(self.groups)),
                            best_loss=self.best_loss, best_step=self.best_step, **d)
