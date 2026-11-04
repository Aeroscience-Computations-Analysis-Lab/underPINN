"""Problem adapter and parameter bookkeeping shared by all diagnostics.

A diagnostic never needs to know which PDE, optimiser or network produced the
parameters. It needs a ``Problem``:

    loss_terms_fn(params, batch) -> {name: scalar}    weighted terms, loss = sum
    batch                                              fixed diagnostic points
    residual_fn(params, x) -> (N, n_eq)                PDE residual rows (NTK, off-grid check)
    rho_fn(params, batch)  -> (M,)                     loss = sum(rho**2) exactly (Gauss-Newton,
                                                       output-layer refit, refit-pruning)
    apply_fn(params, x)    -> (N,) or (N, 1)           network output (error spectrum, modal analysis)
    hidden_fn(params, x)   -> {layer: (z, h)}          pre-/post-activations (representation)

Only ``loss_terms_fn`` and ``batch`` are required; each diagnostic that needs
more says so and is skipped (with the reason recorded) when it is missing.
The batch is FIXED, so every curvature or landscape number is a deterministic
function of the parameters, not of the stochastic mini-batch used in training.

``rho_fn`` must contain every loss term, scaled so that sum(rho**2) equals the
loss: for a mean-squared residual over N points use r_i / sqrt(N) (pinn_math_notes
Sec. 7.1). ``Problem.check_rho`` verifies this before any diagnostic relies on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten_with_path, tree_unflatten


def _key_name(k) -> str:
    return str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k))))


def default_group(path) -> str:
    """Layer label of a parameter: the first dict key below the optional 'params'
    (flax: 'Dense_0', 'Dense_1', ..., 'fourier_B')."""
    names = [_key_name(k) for k in path]
    if names and names[0] == "params":
        names = names[1:]
    return names[0] if names else "all"


def param_groups(params, group_fn: Callable = default_group) -> Dict[str, np.ndarray]:
    """Boolean mask over ``ravel_pytree(params)`` for each layer label (in tree order)."""
    leaves, treedef = tree_flatten_with_path(params)
    labels = [group_fn(p) for p, _ in leaves]
    out: Dict[str, np.ndarray] = {}
    for lab in dict.fromkeys(labels):
        marked = [jnp.ones_like(v) if lb == lab else jnp.zeros_like(v) for (_, v), lb in zip(leaves, labels)]
        flat, _ = ravel_pytree(tree_unflatten(treedef, marked))
        out[lab] = np.asarray(flat) > 0.5
    return out


@dataclass
class Problem:
    loss_terms_fn: Callable
    batch: Any
    residual_fn: Optional[Callable] = None
    x_res: Any = None
    rho_fn: Optional[Callable] = None
    apply_fn: Optional[Callable] = None
    hidden_fn: Optional[Callable] = None
    group_fn: Callable = default_group
    name: str = "problem"

    def terms(self, params) -> Dict[str, jnp.ndarray]:
        return self.loss_terms_fn(params, self.batch)

    def loss(self, params):
        return sum(self.terms(params).values())

    def flat_loss(self, params):
        """(loss as a function of the flat parameter vector, flat vector, unravel)."""
        flat, unravel = ravel_pytree(params)
        return (lambda v: self.loss(unravel(v))), flat, unravel

    def check_rho(self, params, rtol: Optional[float] = None) -> float:
        """Relative mismatch |sum(rho**2) - loss| / loss, raising ``ValueError`` above ``rtol``
        (default 1e4 x the machine epsilon of the parameters). Gauss-Newton fidelity, the
        output-layer refit and refit-pruning all assume loss == sum(rho**2); a rho_fn that
        drops a term or scales it differently would make them answer a different question."""
        if self.rho_fn is None:
            raise ValueError("needs Problem.rho_fn (a vector rho with loss == sum(rho**2))")
        flat, _ = ravel_pytree(params)
        loss = float(self.loss(params))
        ss = float(jnp.sum(jnp.asarray(self.rho_fn(params, self.batch)) ** 2))
        err = abs(ss - loss) / max(abs(loss), np.finfo(np.float64).tiny)
        tol = 1e4 * float(np.finfo(flat.dtype).eps) if rtol is None else rtol
        if err > tol:
            raise ValueError(
                f"Problem.rho_fn does not match the loss: sum(rho**2) = {ss:.6e} but loss = {loss:.6e} "
                f"(relative mismatch {err:.1e} > {tol:.0e}). rho must hold every loss term, scaled so "
                f"that sum(rho**2) == loss (e.g. r_i / sqrt(N) for a mean-squared residual).")
        return err


def flax_hidden_fn(model, act: Callable = jnp.tanh, input_transform: Optional[Callable] = None):
    """``hidden_fn`` for a flax chain of ``nn.Dense`` layers with activation ``act``
    between them (FourierMLP, MLP). Returns {Dense_k: (z, act(z))} for every hidden
    layer; the last Dense is the output layer and is excluded. Networks with learned
    activation slopes (LAAF, dual-tanh) are not covered: pass your own function."""
    import flax.linen as nn

    def fn(params, x):
        xin = x if input_transform is None else input_transform(x)
        _, state = model.apply(params, xin, capture_intermediates=lambda m, _n: isinstance(m, nn.Dense),
                               mutable=["intermediates"])
        inter = state["intermediates"]
        names = sorted((k for k in inter if k.startswith("Dense_")), key=lambda s: int(s.split("_")[1]))
        return {n: (inter[n]["__call__"][0], act(inter[n]["__call__"][0])) for n in names[:-1]}
    return fn
