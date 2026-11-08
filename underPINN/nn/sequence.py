"""Sequence-model PINNs over pseudo-time windows: PINNsFormer and PINNMamba.

Both turn every collocation point (x, t) into a short sequence of points
shifted in time, run a sequence model over it, and read one token out. Here
that happens *inside* the network, so the models keep the ordinary point
contract ``apply(params, (N, D)) -> (N, out)`` and every PDE / loss / solver /
data-parallel path works unchanged. Derivatives are exact derivatives of the
function u(x, t) the network defines (every token depends on t).

PINNsFormer
    Zhao, Ding & Prakash, "PINNsFormer: A Transformer-Based Framework For
    Physics-Informed Neural Networks", ICLR 2024 (arXiv:2307.11833). Follows
    the reference code: forward window ``[t, t+Δt, …, t+(k-1)Δt]`` (k=5,
    Δt=1e-4), linear spatio-temporal mixer, encoder–decoder with
    pre-activation residual blocks, wavelet activation
    ``ω₁ sin(x) + ω₂ cos(x)`` (learnable ω), MLP head. Attention is
    bidirectional, so token 0 (the point itself) sees the whole window and
    is the one read out.

    Deviation: the paper evaluates the residual at all k tokens and averages;
    here the residual is that of token 0's prediction u(x, t), which keeps
    u a single well-defined function of its inputs.

PINNMamba
    Xu et al., "Sub-Sequential Physics-Informed Learning with State Space
    Model", ICML 2025 (arXiv:2502.00318). Mamba (selective state-space)
    blocks over a time window, wavelet-activated head. The SSM is *causal*,
    so the window runs backwards ``[t-(k-1)Δt, …, t]`` and the **last**
    token (the point itself, having seen the whole window) is read out.

    Deviation: the paper's sub-sequence *alignment loss* (consistency of
    predictions across overlapping windows) is a training-loss term, not part
    of the architecture, and is not included.

Inputs are expected roughly in [-1, 1]; ``dt`` is in those (scaled) units.
``time_index`` selects the time column (default: last).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = ["WaveAct", "PINNsFormer", "PINNMamba", "pseudo_sequence"]


def pseudo_sequence(x, k: int, dt: float, time_index: int = -1, backward: bool = False):
    """``(N, D) -> (N, k, D)``: copies of each point shifted in time by j·dt."""
    D = x.shape[-1]
    ti = time_index % D
    steps = jnp.arange(k, dtype=x.dtype) * dt
    if backward:
        steps = steps - (k - 1) * dt                     # [-(k-1)dt, …, 0]
    shift = jnp.zeros((k, D), x.dtype).at[:, ti].set(steps)
    return x[:, None, :] + shift[None, :, :]


class WaveAct(nn.Module):
    """Wavelet activation ω₁ sin(x) + ω₂ cos(x), ω learnable (init 1)."""

    @nn.compact
    def __call__(self, x):
        w1 = self.param("w1", nn.initializers.ones, ())
        w2 = self.param("w2", nn.initializers.ones, ())
        return w1 * jnp.sin(x) + w2 * jnp.cos(x)


class _FeedForward(nn.Module):
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        x = WaveAct()(nn.Dense(self.d_ff)(x))
        x = WaveAct()(nn.Dense(self.d_ff)(x))
        return nn.Dense(self.d_model)(x)


class _EncoderLayer(nn.Module):
    d_model: int
    heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        x2 = WaveAct()(x)
        x = x + nn.MultiHeadDotProductAttention(self.heads, qkv_features=self.d_model)(x2, x2)
        x2 = WaveAct()(x)
        return x + _FeedForward(self.d_model, self.d_ff)(x2)


class _DecoderLayer(nn.Module):
    d_model: int
    heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x, enc):
        x2 = WaveAct()(x)
        x = x + nn.MultiHeadDotProductAttention(self.heads, qkv_features=self.d_model)(x2, enc)
        x2 = WaveAct()(x)
        return x + _FeedForward(self.d_model, self.d_ff)(x2)


def _head(x, d_hidden, out_dim):
    x = WaveAct()(nn.Dense(d_hidden)(x))
    x = WaveAct()(nn.Dense(d_hidden)(x))
    return nn.Dense(out_dim)(x)


class PINNsFormer(nn.Module):
    """PINNsFormer (reference defaults: d_model=32, heads=2, k=5, Δt=1e-4).

    ``d_ff`` / ``d_hidden`` are the feed-forward and head widths (reference
    code: 256 / 512, ≈450k parameters).
    """
    out_dim: int = 1
    d_model: int = 32
    heads: int = 2
    d_ff: int = 256
    d_hidden: int = 512
    n_layers: int = 1
    k: int = 5
    dt: float = 1e-4
    time_index: int = -1

    @nn.compact
    def __call__(self, x):
        seq = pseudo_sequence(x, self.k, self.dt, self.time_index)     # (N, k, D)
        src = nn.Dense(self.d_model, name="mixer")(seq)
        e = src
        for _ in range(self.n_layers):
            e = _EncoderLayer(self.d_model, self.heads, self.d_ff)(e)
        e = WaveAct()(e)
        d = src
        for _ in range(self.n_layers):
            d = _DecoderLayer(self.d_model, self.heads, self.d_ff)(d, e)
        d = WaveAct()(d)
        return _head(d[:, 0, :], self.d_hidden, self.out_dim)           # token 0 = (x, t)


class _RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps). Its derivatives scale like 1/sqrt(eps) where x ≈ 0, which a
    PINN feels through the PDE residual's second derivatives: with a tiny eps, any input that
    maps to a near-zero embedding (e.g. the origin of centred coordinates) becomes a residual
    singularity. ``PINNMamba.norm_eps`` controls it."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        g = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + self.eps) * g


def _inv_softplus(y):
    return y + jnp.log(-jnp.expm1(-y))


class MambaBlock(nn.Module):
    """Mamba-1 selective SSM block (Gu & Dao 2023), sequence axis = 1.

    in_proj → (x, z); causal depthwise conv over the sequence; SiLU;
    input-dependent (Δ, B, C); discretised diagonal SSM
    ``h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t``, ``y_t = C_t·h_t + D x_t``;
    gated by SiLU(z); out_proj. The scan is unrolled (windows are short).
    """
    d_model: int
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    @nn.compact
    def __call__(self, u):
        N, L, _ = u.shape
        E = self.expand * self.d_model
        R = max(1, math.ceil(self.d_model / 16))                       # dt rank
        xz = nn.Dense(2 * E, name="in_proj")(u)
        x, z = xz[..., :E], xz[..., E:]

        w = self.param("conv_w", nn.initializers.lecun_normal(), (self.d_conv, E))
        cb = self.param("conv_b", nn.initializers.zeros, (E,))
        xp = jnp.pad(x, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
        x = sum(w[i] * xp[:, self.d_conv - 1 - i: self.d_conv - 1 - i + L] for i in range(self.d_conv)) + cb
        x = nn.silu(x)

        dbc = nn.Dense(R + 2 * self.d_state, use_bias=False, name="x_proj")(x)
        dt_in, B, C = dbc[..., :R], dbc[..., R:R + self.d_state], dbc[..., R + self.d_state:]

        def dt_bias_init(key, shape, dtype=jnp.float32):
            dt = jnp.exp(jax.random.uniform(key, shape) * (math.log(self.dt_max) - math.log(self.dt_min))
                         + math.log(self.dt_min))
            return _inv_softplus(dt)

        delta = nn.softplus(nn.Dense(E, bias_init=dt_bias_init, name="dt_proj")(dt_in))   # (N,L,E)

        A_log = self.param("A_log", lambda k, s: jnp.log(jnp.tile(jnp.arange(1, s[1] + 1, dtype=jnp.float32), (s[0], 1))),
                           (E, self.d_state))
        A = -jnp.exp(A_log)                                                                # (E,S)
        Dskip = self.param("D", nn.initializers.ones, (E,))

        h = jnp.zeros((N, E, self.d_state), u.dtype)
        ys = []
        for t in range(L):
            dA = jnp.exp(delta[:, t, :, None] * A)                                         # (N,E,S)
            h = dA * h + (delta[:, t, :, None] * B[:, t, None, :]) * x[:, t, :, None]
            ys.append(jnp.einsum("nes,ns->ne", h, C[:, t]) + Dskip * x[:, t])
        y = jnp.stack(ys, axis=1) * nn.silu(z)
        return nn.Dense(self.d_model, name="out_proj")(y)


class PINNMamba(nn.Module):
    """PINNMamba: Mamba blocks over a backward pseudo-time window."""
    out_dim: int = 1
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    expand: int = 2
    d_hidden: int = 256
    k: int = 5
    dt: float = 2e-2
    time_index: int = -1
    norm_eps: float = 1e-6     # RMSNorm epsilon; see _RMSNorm for why PINNs may need it larger

    @nn.compact
    def __call__(self, x):
        seq = pseudo_sequence(x, self.k, self.dt, self.time_index, backward=True)
        h = nn.Dense(self.d_model, name="embed")(seq)
        for _ in range(self.n_layers):
            h = h + MambaBlock(self.d_model, self.d_state, self.expand)(_RMSNorm(self.norm_eps)(h))
        h = _RMSNorm(self.norm_eps)(h)
        return _head(h[:, -1, :], self.d_hidden, self.out_dim)          # last token = (x, t)
