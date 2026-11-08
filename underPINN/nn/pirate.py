"""PirateNets: physics-informed residual adaptive networks.

Wang, Li, Chen & Perdikaris, "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks", JMLR 2024 (arXiv:2402.00326). Follows the
reference implementation (jaxpi) in structure and initialisation:

    φ  = [cos(B x), sin(B x)]                      random Fourier features, B ~ N(0, s²)
    U  = σ(W_U φ),  V = σ(W_V φ)                    gates, computed once
    x⁰ = φ
    for each block l:
        f = σ(W₁ x + b₁);  z₁ = f ⊙ U + (1 − f) ⊙ V
        g = σ(W₂ z₁ + b₂); z₂ = g ⊙ U + (1 − g) ⊙ V
        h = σ(W₃ z₂ + b₃)
        x ← α_l h + (1 − α_l) x                     α_l trainable, **initialised to 0**
    u  = W_out x + b_out

Because every α starts at 0, the network is initially *linear in the Fourier
features*, u = W_out φ + b_out: depth is switched on gradually by training,
which is the paper's cure for deep PINNs training worse than shallow ones.
That also enables the *physics-informed initialisation*
(:func:`pirate_output_init`): fit W_out by least squares to the initial
condition before training starts.

All Dense layers use random weight factorisation (RWF, ``rwf=True``):
``W = diag(exp(s)) · V`` with ``s ~ N(1, 0.1)``, as in the paper.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

__all__ = ["PirateNet", "RWFDense", "pirate_output_init"]


class RWFDense(nn.Module):
    """Dense layer with random weight factorisation (Wang et al. 2022)."""
    features: int
    rwf: bool = True
    mean: float = 1.0
    stddev: float = 0.1

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        if not self.rwf:
            return nn.Dense(self.features)(x)

        def init_v(key, shape):
            k1, k2 = jax.random.split(key)
            w = nn.initializers.glorot_normal()(k1, shape)
            s = self.mean + self.stddev * jax.random.normal(k2, (shape[-1],))
            return jnp.concatenate([w / jnp.exp(s), s[None, :]], axis=0)

        vs = self.param("kernel_vs", init_v, (in_dim, self.features))
        v, s = vs[:-1], vs[-1]
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        return x @ (v * jnp.exp(s)) + b


class _FourierEmbedding(nn.Module):
    n_freq: int
    scale: float

    @nn.compact
    def __call__(self, x):
        B = self.param("B", nn.initializers.normal(self.scale), (x.shape[-1], self.n_freq))
        z = x @ B
        return jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)


class _PirateBlock(nn.Module):
    width: int
    rwf: bool

    @nn.compact
    def __call__(self, x, U, V):
        f = jnp.tanh(RWFDense(self.width, self.rwf)(x))
        z1 = f * U + (1.0 - f) * V
        g = jnp.tanh(RWFDense(self.width, self.rwf)(z1))
        z2 = g * U + (1.0 - g) * V
        h = jnp.tanh(RWFDense(self.width, self.rwf)(z2))
        alpha = self.param("alpha", nn.initializers.zeros, ())
        return alpha * h + (1.0 - alpha) * x


class PirateNet(nn.Module):
    """PirateNet. ``width`` = hidden width = Fourier-embedding dimension (even).

    Parameters
    ----------
    width : hidden width; the embedding has ``width // 2`` frequencies.
    n_blocks : number of residual adaptive blocks (3 Dense layers each).
    out_dim : outputs.
    fourier_scale : std of the Fourier frequency matrix B (paper: 1–2).
    rwf : random weight factorisation on every Dense layer.
    """
    width: int = 256
    n_blocks: int = 3
    out_dim: int = 1
    fourier_scale: float = 1.0
    rwf: bool = True

    def setup(self):
        if self.width % 2:
            raise ValueError(f"PirateNet width must be even, got {self.width}")
        self.embed = _FourierEmbedding(self.width // 2, self.fourier_scale)
        self.enc_U = RWFDense(self.width, self.rwf)
        self.enc_V = RWFDense(self.width, self.rwf)
        self.blocks = [_PirateBlock(self.width, self.rwf) for _ in range(self.n_blocks)]
        self.head = RWFDense(self.out_dim, self.rwf)

    def features(self, x):
        """Last hidden state (input to the output layer)."""
        phi = self.embed(x)
        U, V = jnp.tanh(self.enc_U(phi)), jnp.tanh(self.enc_V(phi))
        h = phi
        for blk in self.blocks:
            h = blk(h, U, V)
        return h

    def __call__(self, x):
        return self.head(self.features(x))


def pirate_output_init(params, feats: jnp.ndarray, targets: jnp.ndarray,
                       head_path: Sequence[str], ridge: float = 1e-6):
    """Physics-informed initialisation: least-squares fit of the output layer.

    At initialisation every α = 0, so the network output is linear in
    ``feats`` (the Fourier features). Solve
    ``min ‖feats · W + b − targets‖² + ridge ‖W‖²`` and write W, b into the
    output layer found at ``head_path`` (e.g. ``("params", "net", "head")``).

    ``feats``: ``(N, width)`` from ``model.apply(params, x_ic, method=...features)``;
    ``targets``: ``(N,)`` or ``(N, out_dim)`` initial-condition values.
    Works with and without RWF (with RWF the fitted W is re-factorised so the
    stored scale vector ``s`` is kept).
    """
    F = np.asarray(feats, np.float64)
    Y = np.asarray(targets, np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    A = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
    sol = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y)
    W, b = sol[:-1], sol[-1]

    def set_in(tree, path, fn):
        if not path:
            return fn(tree)
        out = dict(tree)
        out[path[0]] = set_in(tree[path[0]], path[1:], fn)
        return out

    def write(head):
        head = dict(head)
        dtype = head["bias"].dtype
        head["bias"] = jnp.asarray(b, dtype)
        if "kernel_vs" in head:
            s = head["kernel_vs"][-1]
            v = jnp.asarray(W, dtype) / jnp.exp(s)
            head["kernel_vs"] = jnp.concatenate([v, s[None, :]], axis=0)
        else:
            head["kernel"] = jnp.asarray(W, dtype)
        return head

    return set_in(params, list(head_path), write)
