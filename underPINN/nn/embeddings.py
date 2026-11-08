from typing import Any

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


class PeriodicInput(nn.Module):
    """Exact periodicity for any network: wrap it so that chosen inputs enter as (cos, sin).

    Coordinate ``i`` in ``periods`` (dict i -> period L_i, with origin ``lo[i]``) is replaced
    by cos(2π(x_i − lo_i)/L_i), sin(2π(x_i − lo_i)/L_i); every other coordinate is mapped
    affinely from [lo, hi] to [-1, 1]. Periodic coordinates come first, the rest keep their
    order (so a time coordinate that was last stays last -- PINNsFormer / PINNMamba read
    time from the last column). The wrapped function is then exactly L-periodic in those
    coordinates, with all derivatives, so periodic boundary conditions hold by construction.

    ``periods`` may be a dict ``{i: L_i}`` or a sequence of ``(i, L_i)`` pairs. It is stored
    as a sorted tuple of pairs: flax freezes dict attributes into a ``FrozenDict``, and on
    flax 0.8.x that fails at ``init`` for non-string keys.
    """
    net: nn.Module
    lo: tuple
    hi: tuple
    periods: Any

    def __post_init__(self):
        p = self.periods
        items = p.items() if hasattr(p, "items") else p
        object.__setattr__(self, "periods", tuple(sorted((int(i), float(L)) for i, L in items)))
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        lo, hi = jnp.asarray(self.lo, x.dtype), jnp.asarray(self.hi, x.dtype)
        d = x.shape[1]
        periods = {i % d: L for i, L in self.periods}
        feats = []
        for i, L in sorted(periods.items()):
            a = 2 * jnp.pi * (x[:, i] - lo[i]) / L
            feats += [jnp.cos(a), jnp.sin(a)]
        for i in range(d):
            if i not in periods:
                feats.append(2.0 * (x[:, i] - lo[i]) / (hi[i] - lo[i]) - 1.0)
        return self.net(jnp.stack(feats, axis=1))
