"""Neural-operator architectures: FNO1D, FNO2D, DeepONet1D, CVit.

Unlike the point networks in :mod:`underPINN.nn.mlp` (which map a single
collocation point to a single output), these map an entire **function** —
sampled on a grid or at sensor locations — to another function.  They are the
building blocks for physics-informed neural operators (PINO): train once
across a distribution of inputs (initial conditions, viscosities, ...) and
then infer the solution for a new input in a single forward pass, instead of
re-solving a PDE from scratch.

* ``FNO1D`` / ``FNO2D`` — Fourier Neural Operator (Li et al., 2020).  Learns a
  convolution kernel in Fourier space, truncated to the lowest ``modes``
  frequencies, plus a pointwise (1×1 conv) skip path per stage.  Ported from
  the ``fourier-neural-operator-flax`` reference implementation.
* ``DeepONet1D`` — classic branch/trunk operator (Lu et al., 2021): a
  ``branch`` network encodes the input function (sampled at fixed sensors),
  a ``trunk`` network encodes the query coordinate, and the output is their
  dot product.  Reuses :class:`underPINN.nn.mlp.MLP` /
  :class:`underPINN.nn.mlp.GatedMLP` for both sub-networks.
* ``CVit`` — Continuous Vision Transformer for Operator Learning (Wang et al.,
  ICLR 2025): a ViT-style encoder (patch embed + latent time-aggregation +
  self-attention) over the input function, with a cross-attention decoder that
  queries the encoded latents at continuous coordinates.  Requires the
  ``einops`` package.
"""
from __future__ import annotations

from typing import Callable, Sequence

import einops
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import random
from jax.nn.initializers import normal as _jax_normal, xavier_uniform

from underPINN.nn.mlp import MLP, GatedMLP


def _normal(stddev: float = 1e-2, dtype=jnp.float32) -> Callable:
    """Normal initializer scaled by stddev (used by the FNO spectral kernels)."""
    def init(key, shape, dtype=dtype):
        k, _ = random.split(key)
        return random.normal(k, shape) * stddev
    return init


# =============================================================================
#  FNO1D — 1-D Fourier Neural Operator
# =============================================================================

class SpectralConv1d(nn.Module):
    """Truncated-spectrum convolution: keep the lowest ``modes1`` Fourier
    modes, apply a learned complex kernel, invert."""
    out_channels: int = 32
    modes1: int = 12

    @nn.compact
    def __call__(self, x):
        # x: (batch, length, in_channels)
        in_channels = x.shape[-1]
        length = x.shape[1]
        scale = 1.0 / (in_channels * self.out_channels)

        assert self.modes1 <= length // 2 + 1
        assert length % 2 == 0, "even-length inputs only"

        # rfft already drops the conjugate-symmetric half, so (unlike 2-D)
        # there is only one frequency band to learn a kernel for.
        kernel_r = self.param(
            "kernel_r", _normal(scale),
            (in_channels, self.out_channels, self.modes1), jnp.float32)
        kernel_i = self.param(
            "kernel_i", _normal(scale),
            (in_channels, self.out_channels, self.modes1), jnp.float32)

        x_ft = jnp.fft.rfft(x, axis=1)                  # (B, L//2+1, Cin) complex
        low = jnp.einsum(
            "bic,coi->bio",
            x_ft[:, :self.modes1, :],
            kernel_r + 1j * kernel_i,
        )
        out_ft = jnp.zeros(
            (x.shape[0], x_ft.shape[1], self.out_channels), dtype=x_ft.dtype)
        out_ft = out_ft.at[:, :self.modes1, :].set(low)
        return jnp.fft.irfft(out_ft, n=length, axis=1)  # (B, L, Cout)


class FourierStage1d(nn.Module):
    out_channels: int = 32
    modes1: int = 12
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        spectral = SpectralConv1d(self.out_channels, self.modes1)(x)
        local = nn.Conv(self.out_channels, (1,))(x)     # pointwise 1x1 conv
        return self.activation(spectral + local)


class FNO1D(nn.Module):
    """1-D Fourier Neural Operator.

    Parameters
    ----------
    modes1              retained Fourier modes (<= L//2 + 1)
    width               lifted channel width
    depth               number of Fourier stages
    channels_last_proj  hidden width of the final pointwise MLP
    out_channels        number of output channels
    padding             domain padding for non-periodic inputs (0 = periodic).
                        Non-zero for Dirichlet/Neumann domains: the spectral
                        conv assumes periodicity, so the input is zero-padded
                        before the FFT and trimmed after (standard FNO trick).
    """
    modes1: int = 12
    width: int = 32
    depth: int = 4
    channels_last_proj: int = 128
    activation: Callable = nn.gelu
    out_channels: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x):
        # Append a normalized coordinate channel (the 1-D grid).
        x = jnp.concatenate([x, self.get_grid(x)], axis=-1)
        x = nn.Dense(self.width)(x)

        if self.padding > 0:
            x = jnp.pad(x, ((0, 0), (0, self.padding), (0, 0)))

        # Last stage runs without an activation, matching the original FNO.
        for stage in range(self.depth):
            act = self.activation if stage < self.depth - 1 else (lambda z: z)
            x = FourierStage1d(self.width, self.modes1, act)(x)

        if self.padding > 0:
            x = x[:, :-self.padding, :]

        x = nn.Dense(self.channels_last_proj)(x)
        x = self.activation(x)
        return nn.Dense(self.out_channels)(x)

    @staticmethod
    def get_grid(x):
        coords = jnp.linspace(0, 1, x.shape[1])         # (L,)
        return jnp.repeat(coords[None, :, None], x.shape[0], axis=0)


# =============================================================================
#  FNO2D — 2-D Fourier Neural Operator
# =============================================================================

class SpectralConv2d(nn.Module):
    out_channels: int = 32
    modes1: int = 12
    modes2: int = 12

    @nn.compact
    def __call__(self, x):
        # x: (batch, height, width, in_channels)
        in_channels = x.shape[-1]
        height, width = x.shape[1], x.shape[2]
        scale = 1.0 / (in_channels * self.out_channels)

        assert self.modes1 <= height // 2 + 1
        assert self.modes2 <= width // 2 + 1
        assert height % 2 == 0 and width % 2 == 0  # only tested on even grids

        # Real input -> rfftn, so axis 2 is halved by conjugate symmetry, but
        # axis 1 still has its wrapped-around negative modes — hence two
        # kernels: one for the low-frequency corner, one for the negative band.
        kernel_1_r = self.param(
            "kernel_1_r", _normal(scale),
            (in_channels, self.out_channels, self.modes1, self.modes2), jnp.float32)
        kernel_1_i = self.param(
            "kernel_1_i", _normal(scale),
            (in_channels, self.out_channels, self.modes1, self.modes2), jnp.float32)
        kernel_2_r = self.param(
            "kernel_2_r", _normal(scale),
            (in_channels, self.out_channels, self.modes1, self.modes2), jnp.float32)
        kernel_2_i = self.param(
            "kernel_2_i", _normal(scale),
            (in_channels, self.out_channels, self.modes1, self.modes2), jnp.float32)

        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # zeros_like sizes the spectrum to in_channels, not out_channels — only
        # safe because every stage runs at uniform `width` channels.
        out_ft = jnp.zeros_like(x_ft)
        s1 = jnp.einsum("bijc,coij->bijo",
                        x_ft[:, :self.modes1, :self.modes2, :],
                        kernel_1_r + 1j * kernel_1_i)
        s2 = jnp.einsum("bijc,coij->bijo",
                        x_ft[:, -self.modes1:, :self.modes2, :],
                        kernel_2_r + 1j * kernel_2_i)
        out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(s1)
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(s2)
        return jnp.fft.irfftn(out_ft, axes=(1, 2))


class FourierStage(nn.Module):
    out_channels: int = 32
    modes1: int = 12
    modes2: int = 12
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        x_fourier = SpectralConv2d(self.out_channels, self.modes1, self.modes2)(x)
        x_local = nn.Conv(self.out_channels, (1, 1))(x)   # pointwise skip path
        return self.activation(x_fourier + x_local)


class FNO2D(nn.Module):
    r"""Fourier Neural Operator for 2-D signals.

    modes1, modes2     : Fourier modes kept on each axis
    width              : channels the input is lifted to
    depth              : number of Fourier stages
    channels_last_proj : hidden width of the final 2-layer channel-wise MLP
    out_channels       : >1 for non-scalar (vector) fields
    padding            : only for non-periodic inputs; 0 for periodic domains
    """
    modes1: int = 12
    modes2: int = 12
    width: int = 32
    depth: int = 4
    channels_last_proj: int = 128
    activation: Callable = nn.gelu
    out_channels: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        grid = self.get_grid(x)
        x = jnp.concatenate([x, grid], axis=-1)
        x = nn.Dense(self.width)(x)

        if self.padding > 0:
            x = jnp.pad(x, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                       mode="constant")

        # Last stage is linear — not in the paper, but matches the reference impl.
        for depthnum in range(self.depth):
            activation = self.activation if depthnum < self.depth - 1 else (lambda z: z)
            x = FourierStage(self.width, self.modes1, self.modes2, activation)(x)

        if self.padding > 0:
            x = x[:, :-self.padding, :-self.padding, :]

        x = nn.Dense(self.channels_last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)
        return x

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x1, x2 = jnp.meshgrid(x1, x2, indexing="ij")
        grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
        return jnp.repeat(grid, x.shape[0], axis=0)


# =============================================================================
#  DeepONet1D — branch/trunk operator
# =============================================================================

class DeepONet1D(nn.Module):
    """Classic DeepONet: ``s(u)(y) = branch(u) . trunk(y)``.

    Parameters
    ----------
    branch_layers : ``[m_in, h, h, ..., p]`` branch widths.  ``m_in`` is the
                    sensor count (the input function sampled at ``m`` fixed
                    locations; append e.g. a viscosity value as one extra
                    entry to condition on a PDE parameter).  ``p`` is the
                    latent (dot-product) dimension.
    trunk_layers  : ``[d_in, h, h, ..., p]`` trunk widths.  ``d_in`` is the
                    query-coordinate dimension (e.g. 2 for a packed ``(t, x)``
                    query).  ``p`` MUST equal the branch's.
    gated         : use :class:`underPINN.nn.mlp.GatedMLP` (the "modified
                    MLP" of Wang et al. — usually the strongest choice) for
                    both sub-networks, or the plain tanh
                    :class:`underPINN.nn.mlp.MLP` when ``False``.
    """

    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    gated: bool = True

    @nn.compact
    def __call__(self, u: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """u : (m_in,) sensor values;  y : (d_in,) query point. Returns scalar."""
        assert self.branch_layers[-1] == self.trunk_layers[-1], (
            "branch and trunk must share the same latent dim p "
            f"(got {self.branch_layers[-1]} vs {self.trunk_layers[-1]})")
        Net = GatedMLP if self.gated else MLP
        B = Net(layers=list(self.branch_layers), name="branch")(u)   # (p,)
        T = Net(layers=list(self.trunk_layers), name="trunk")(y)     # (p,)
        return jnp.sum(B * T)                                         # scalar


# =============================================================================
#  CVit — Continuous Vision Transformer for Operator Learning
#  Wang et al., ICLR 2025 (https://arxiv.org/abs/2405.13998)
# =============================================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega                        # (D/2,)

    pos = pos.reshape(-1)                               # (M,)
    out = jnp.einsum("m,d->md", pos, omega)             # (M, D/2)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)  # (M, D)


def _get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        _get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)),
        0)


def _get_2d_sincos_pos_embed(embed_dim, grid_size):
    def _from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="ij")
    grid = jnp.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = _from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


class _PatchEmbed(nn.Module):
    patch_size: tuple = (1, 16, 16)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape
        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            strides=(self.patch_size[0], self.patch_size[1], self.patch_size[2]),
            kernel_init=self.kernel_init, name="proj")(x)
        num_patches = (t // self.patch_size[0],
                      h // self.patch_size[1],
                      w // self.patch_size[2])
        x = jnp.reshape(x, (b, num_patches[0],
                            num_patches[1] * num_patches[2], self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=self.layer_norm_eps)(x)
        return x


class _MlpBlock(nn.Module):
    dim: int = 256
    out_dim: int = 256
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        return nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)


class _SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim)(x, x)
        x = x + inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = _MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class _CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim)(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = _MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class _TimeAggregation(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):                          # (B, T, S, D) → (B, T', S, D)
        latents = self.param("latents", _jax_normal(),
                             (self.num_latents, self.emb_dim))
        latents = repeat(latents, "t d -> b s t d", b=x.shape[0], s=x.shape[2])
        x = rearrange(x, "b t s d -> b s t d")
        for _ in range(self.depth):
            latents = _CrossAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio,
                self.layer_norm_eps)(latents, x)
        return rearrange(latents, "b s t d -> b t s d")


class _Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.num_layers):
            y = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(x)
            y = nn.gelu(y)
            x = x + y
            x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return nn.Dense(self.out_dim)(x)


class _Encoder(nn.Module):
    patch_size: tuple = (1, 16, 16)
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape
        x = _PatchEmbed(self.patch_size, self.emb_dim)(x)

        t_emb = self.variable("pos_emb", "enc_t_emb", _get_1d_sincos_pos_embed,
                              self.emb_dim, t // self.patch_size[0])
        s_emb = self.variable("pos_emb", "enc_s_emb", _get_2d_sincos_pos_embed,
                              self.emb_dim,
                              (h // self.patch_size[1], w // self.patch_size[2]))
        x = x + t_emb.value[:, :, jnp.newaxis, :] + s_emb.value[:, jnp.newaxis, :, :]

        x = _TimeAggregation(
            num_latents=1, emb_dim=self.emb_dim, depth=2,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps)(x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = rearrange(x, "b t s d -> b (t s) d")

        for _ in range(self.depth):
            x = _SelfAttnBlock(self.num_heads, self.emb_dim, self.mlp_ratio,
                              self.layer_norm_eps)(x)
        return x


class _FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", _jax_normal(self.embed_scale),
                            (x.shape[-1], self.embed_dim // 2))
        return jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)


class CVit(nn.Module):
    """Continuous Vision Transformer for Operator Learning.

    Encodes an input function sampled on a ``(T, H, W, C)`` space-time grid
    into latent tokens (patch embed → time-aggregation → self-attention),
    then decodes at arbitrary continuous query coordinates via a
    cross-attention decoder — so, unlike FNO, queries need not lie on a fixed
    grid.

    Parameters
    ----------
    patch_size     : (t, h, w) patch stride for the input grid.
    grid_size      : (n_x, n_y) reference grid used by the "grid" coordinate
                     embedding (a soft lookup into learned per-grid-point
                     latents; ignored for "fourier"/"mlp" embeddings).
    latent_dim     : width of the per-grid-point latents ("grid" embedding only).
    emb_dim        : encoder token width.
    depth          : number of self-attention encoder blocks.
    num_heads      : encoder attention heads.
    dec_emb_dim    : decoder (cross-attention) token width.
    dec_num_heads  : decoder attention heads.
    dec_depth      : number of cross-attention decoder blocks.
    num_mlp_layers : residual MLP layers in the final decode head.
    mlp_ratio      : hidden-width multiplier for the block MLPs.
    out_dim        : number of output channels.
    eps            : softmax temperature (inverse) for the "grid" embedding's
                     Gaussian-kernel lookup.
    embedding_type : "grid" | "fourier" | "mlp" — how the query coordinates
                     are embedded before the cross-attention decoder.
                     "fourier" is unconditionally stable; "grid" requires
                     query coordinates to lie on the reference grid closely
                     enough that the Gaussian lookup doesn't degenerate.
    """
    patch_size: tuple = (1, 16, 16)
    grid_size: tuple = (128, 128)
    latent_dim: int = 256
    emb_dim: int = 256
    depth: int = 3
    num_heads: int = 8
    dec_emb_dim: int = 256
    dec_num_heads: int = 8
    dec_depth: int = 1
    num_mlp_layers: int = 1
    mlp_ratio: int = 1
    out_dim: int = 1
    eps: float = 1e5
    layer_norm_eps: float = 1e-5
    embedding_type: str = "grid"

    def setup(self):
        if self.embedding_type == "grid":
            n_x, n_y = self.grid_size
            x = jnp.linspace(0, 1, n_x)
            y = jnp.linspace(0, 1, n_y)
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            self.grid = jnp.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
            self.latents = self.param("latents", _jax_normal(),
                                      (n_x * n_y, self.latent_dim))

    @nn.compact
    def __call__(self, x, coords):
        b, t, h, w, c = x.shape

        # ── coordinate embedding ──────────────────────────────────────────
        if self.embedding_type == "grid":
            d2 = ((coords[:, jnp.newaxis, :] - self.grid[jnp.newaxis, :, :]) ** 2
                 ).sum(axis=2)
            w_ = jnp.exp(-self.eps * d2)
            w_ = w_ / w_.sum(axis=1, keepdims=True)
            coords = jnp.einsum("ic,pi->pc", self.latents, w_)
            coords = nn.Dense(self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        elif self.embedding_type == "fourier":
            coords = _FourierEmbs(embed_scale=2 * jnp.pi,
                                  embed_dim=self.dec_emb_dim)(coords)
        elif self.embedding_type == "mlp":
            coords = _MlpBlock(self.dec_emb_dim, self.dec_emb_dim)(coords)
            coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type!r}")

        coords = einops.repeat(coords, "n d -> b n d", b=b)

        # ── encoder ─────────────────────────────────────────────────────────
        x = _Encoder(self.patch_size, self.emb_dim, self.depth,
                    self.num_heads, self.mlp_ratio, self.layer_norm_eps)(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.Dense(self.dec_emb_dim)(x)

        # ── cross-attention decoder ─────────────────────────────────────────
        for _ in range(self.dec_depth):
            coords = _CrossAttnBlock(
                self.dec_num_heads, self.dec_emb_dim, self.mlp_ratio,
                self.layer_norm_eps)(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        return _Mlp(self.num_mlp_layers, self.dec_emb_dim, self.out_dim,
                   layer_norm_eps=self.layer_norm_eps)(x)   # (B, N_query, out_dim)


def cvit_grid_predict(model: CVit, params, x_seq: jnp.ndarray,
                      u_prev: jnp.ndarray, delta_scale: float) -> jnp.ndarray:
    """Query a CVit model on its own input grid and reshape back to a grid.

    CVit decodes at arbitrary continuous coordinates, but a PINO-style grid
    finite-difference residual needs values *on* a grid — so this queries
    every grid point once and reshapes the flat output back to ``(Nx, Ny)``,
    letting the same :class:`underPINN.pde.burgers_grid.BurgersGrid2D` used
    for FNO2D also serve CVit.

    Also applies the scaled-increment trick (``u_pred = u_prev +
    delta_scale * raw``): predicting the *increment* from the last history
    frame rather than the raw field is what makes CViT rollout accurate in
    practice (the difference between a ~1% and a ~99% relative L2 error).

    Parameters
    ----------
    x_seq       : ``(batch, T, Nx, Ny, C)`` input history window.
    u_prev      : ``(batch, Nx, Ny)`` last history frame (the increment base).
    delta_scale : normalizing scale for the predicted increment, typically
                  ``std(u_next - u_prev)`` over the training set.

    Returns
    -------
    ``(batch, Nx, Ny, 1)`` prediction grid.
    """
    b, _t, nx, ny, _c = x_seq.shape
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 1.0, ny)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)      # (Nx*Ny, 2)
    raw = model.apply(params, x_seq, coords)                    # (b, Nx*Ny, out_dim)
    raw = raw.reshape(b, nx, ny, -1)
    return u_prev[..., None] + delta_scale * raw
