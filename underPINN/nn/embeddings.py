import jax.numpy as jnp
from flax import linen as nn

class FourierEmbed(nn.Module):
    dim: int
    num_freq: int = 3

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(1, self.num_freq + 1, dtype=x.dtype)
        angles = 2 * jnp.pi * x[..., None] * freqs
        sin = jnp.sin(angles).reshape(x.shape[0], -1)
        cos = jnp.cos(angles).reshape(x.shape[0], -1)
        return jnp.concatenate([x, sin, cos], axis=-1)
