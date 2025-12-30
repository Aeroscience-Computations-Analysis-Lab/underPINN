import jax.numpy as jnp
from flax import linen as nn
from .attention import HybridAttention

class SubdomainNetwork(nn.Module):
    layers: list

    def setup(self):
        self.hidden = self.layers[1]
        self.encU = nn.Dense(self.hidden)
        self.encV = nn.Dense(self.hidden)
        self.linears = [nn.Dense(d) for d in self.layers[1:]]
        self.attns = [HybridAttention(self.hidden) for _ in range(len(self.layers)-2)]

    def __call__(self, x):
        U = nn.tanh(self.encU(x))
        V = nn.tanh(self.encV(x))
        h = x
        for lin, attn in zip(self.linears[:-1], self.attns):
            h = attn(nn.tanh(lin(h)), U, V)
        return self.linears[-1](h)
