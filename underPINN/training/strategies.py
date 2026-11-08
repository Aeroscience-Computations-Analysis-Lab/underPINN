"""Standard PINN training strategies, usable with any network or loss.

causal_pde_loss(res, t, t_range, n_bins, eps)
    Causal weighting (Wang, Sankaran & Perdikaris, "Respecting causality is all you need
    for training physics-informed neural networks", 2022). The collocation points are
    split into ``n_bins`` time slices; slice k's mean squared residual L_k is weighted by
    w_k = exp(-eps · Σ_{j<k} L_j) (no gradient through w). Late slices only start to
    count once earlier ones are resolved, which prevents the network from fitting late
    times to a wrong early state (the "trivial solution" failure of transport problems).
    Works with randomly drawn points: each point is assigned to its slice by its time.

grad_norm_weights(terms_fn, params, names)
    Gradient-norm loss balancing (Wang, Teng & Perdikaris 2021; the "grad_norm" scheme of
    jaxpi): λ_i = Σ_j ‖∇L_j‖ / ‖∇L_i‖ for the *unweighted* terms, so every weighted term
    pulls with the same gradient magnitude. Combine with an EMA (``ema_update``) and
    recompute every few hundred to thousand steps.

lbfgs_refine(loss_fn, params, max_iters, time_budget)
    L-BFGS (optax, zoom line search) on a fixed, deterministic loss — the usual finishing
    phase after Adam. Stops at ``max_iters``, the time budget, a non-finite loss, or
    stagnation; returns the best iterate seen.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

__all__ = ["causal_pde_loss", "grad_norm_weights", "ema_update", "lbfgs_refine"]


def causal_pde_loss(res, t, t_range: Tuple[float, float], n_bins: int = 32, eps: float = 1.0):
    """Causally weighted mean squared residual; returns ``(loss, weights, bin_losses)``.

    ``res``: (N,) or (N, k) residual rows; ``t``: (N,) times of those points;
    ``t_range``: (t0, t1) of the time domain.
    """
    r2 = jnp.sum(jnp.reshape(res, (res.shape[0], -1)) ** 2, axis=1)
    t0, t1 = t_range
    idx = jnp.clip(jnp.floor((t - t0) / (t1 - t0) * n_bins).astype(jnp.int32), 0, n_bins - 1)
    s = jax.ops.segment_sum(r2, idx, num_segments=n_bins)
    c = jax.ops.segment_sum(jnp.ones_like(r2), idx, num_segments=n_bins)
    L = s / jnp.maximum(c, 1.0)
    cum = jnp.concatenate([jnp.zeros((1,), L.dtype), jnp.cumsum(L)[:-1]])
    w = jax.lax.stop_gradient(jnp.exp(-eps * cum))
    return jnp.mean(w * L), w, L


def grad_norm_weights(terms_fn: Callable, params, names: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """λ_i = Σ_j ‖∇L_j‖ / ‖∇L_i‖ for the unweighted terms returned by ``terms_fn(params)``."""
    def one(name):
        g = jax.grad(lambda p: terms_fn(p)[name])(params)
        return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    norms = {n: one(n) for n in names}
    tot = sum(norms.values())
    return {n: tot / jnp.maximum(v, 1e-12) for n, v in norms.items()}


def ema_update(old: Dict, new: Dict, momentum: float = 0.9) -> Dict:
    return {k: momentum * old[k] + (1.0 - momentum) * new[k] for k in old}


def lbfgs_refine(loss_fn: Callable, params, max_iters: int = 2000, time_budget: Optional[float] = None,
                 memory_size: int = 20, patience: int = 200, rtol: float = 1e-6, log_every: int = 0):
    """Minimise a deterministic ``loss_fn(params)`` with L-BFGS; returns (best params, history)."""
    opt = optax.lbfgs(memory_size=memory_size)

    def make_step(reuse: bool):
        # reuse=True re-uses the line search's value/gradient (optax helper). That helper
        # looks the value up by the *name* 'value' anywhere in the optimiser state, which
        # also holds a copy of the params -- so it fails for networks with a parameter
        # named 'value' (attention layers: query/key/value). Fall back to recomputing.
        vg = optax.value_and_grad_from_state(loss_fn) if reuse else None

        @jax.jit
        def step(p, s):
            v, g = vg(p, state=s) if reuse else jax.value_and_grad(loss_fn)(p)
            u, s = opt.update(g, s, p, value=v, grad=g, value_fn=loss_fn)
            return optax.apply_updates(p, u), s, v
        return step

    state = opt.init(params)
    best_p, best_v = params, float(loss_fn(params))
    hist, since = [best_v], 0
    # compile the step before the clock starts (the time budget is for iterations, not XLA)
    try:
        step = make_step(True)
        jax.block_until_ready(step.lower(params, state).compile()(params, state)[2])
    except KeyError:
        step = make_step(False)
        jax.block_until_ready(step.lower(params, state).compile()(params, state)[2])
    t0 = time.perf_counter()
    for it in range(max_iters):
        new_p, new_s, v = step(params, state)      # v = loss_fn(params): the loss BEFORE this step
        v = float(v)
        if not math.isfinite(v):
            break
        if it:                                     # it == 0 re-evaluates params0, already hist[0]
            hist.append(v)
        if v < best_v * (1 - rtol):
            best_v, best_p, since = v, params, 0   # pair v with the iterate it was measured at
        else:
            since += 1
        params, state = new_p, new_s
        if since >= patience or (time_budget is not None and time.perf_counter() - t0 > time_budget):
            break
        if log_every and it % log_every == 0:
            print(f"  L-BFGS {it:5d}  loss {v:.3e}", flush=True)
    final_v = float(loss_fn(params))
    if math.isfinite(final_v):
        hist.append(final_v)
        if final_v < best_v:
            best_p, best_v = params, final_v
    return best_p, {"loss": hist, "iters": len(hist) - 1, "best": best_v,
                    "seconds": time.perf_counter() - t0}
