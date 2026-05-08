import jax
import jax.numpy as jnp
from flax import linen as nn
from .subdomain import SubdomainNetwork
from .attention import HybridAttention


def window_1d(x, xmin, xmax, smin, smax):
    return jnp.clip(jax.nn.sigmoid((x-xmin)/smin),1e-10,1.0) * \
           jnp.clip(jax.nn.sigmoid((xmax-x)/smax),1e-10,1.0)

def window_nd(x, xmin, xmax, smin, smax):
    w = 1.0
    for d in range(x.shape[1]):
        w *= window_1d(x[:, d], xmin[d], xmax[d], smin[d], smax[d])
    return w[:, None]

class FBPINN(nn.Module):
    layers: list
    shifts: jnp.ndarray
    xs_min: jnp.ndarray
    xs_max: jnp.ndarray
    smins: jnp.ndarray
    smaxs: jnp.ndarray
    attention_cls: callable = HybridAttention

    def setup(self):
        self.subnets = [
            SubdomainNetwork(self.layers, attention_cls=self.attention_cls)
            for _ in range(self.shifts.shape[0])
        ]

    def __call__(self, x):
        out = 0.0
        for i, net in enumerate(self.subnets):
            win = window_nd(x, self.xs_min[i], self.xs_max[i], self.smins[i], self.smaxs[i])
            out += net(x - self.shifts[i]) * win
        return out
