import jax.numpy as jnp
from flax import linen as nn
from .embeddings import FourierEmbed

class AdditiveAttention(nn.Module):
    dim: int
    hidden: int = 32
    num_freq: int = 3

    def setup(self):
        self.embed = FourierEmbed(self.dim, self.num_freq)
        self.W1 = nn.Dense(self.hidden)
        self.W2 = nn.Dense(self.hidden)
        self.v  = nn.Dense(1)
        self.norm = nn.LayerNorm()

    def __call__(self, x, U, V):
        x_e, U_e, V_e = self.embed(x), self.embed(U), self.embed(V)
        s1 = self.v(nn.tanh(self.W1(x_e) + self.W2(U_e)))
        s2 = self.v(nn.tanh(self.W1(x_e) + self.W2(V_e)))
        alpha = nn.softmax(jnp.concatenate([s1, s2], axis=1), axis=1)
        out = alpha[:, :1] * U + alpha[:, 1:] * V
        return self.norm(out + x)


class DotProductAttention(nn.Module):
    dim: int

    def setup(self):
        self.scale = self.dim ** -0.5
        self.q = nn.Dense(self.dim)
        self.k = nn.Dense(self.dim)
        self.v = nn.Dense(self.dim)
        self.norm = nn.LayerNorm()

    def __call__(self, x, U, V):
        q = self.q(x)
        kv = jnp.stack([U, V], axis=1)
        k = self.k(kv)
        v = self.v(kv)
        scores = jnp.sum(q[:, None] * k, axis=-1) * self.scale
        alpha = nn.softmax(scores, axis=1)
        out = alpha[:, :1] * v[:, 0] + alpha[:, 1:] * v[:, 1]
        return self.norm(out + x)


class HybridAttention(nn.Module):
    dim: int
    hidden: int = 64

    def setup(self):
        self.A = AdditiveAttention(self.dim)
        self.D = DotProductAttention(self.dim)
        self.gate = nn.Sequential([
            nn.Dense(self.hidden),
            nn.LayerNorm(),
            nn.gelu,
            nn.Dense(2)
        ])

    def __call__(self, x, U, V):
        A, D = self.A(x, U, V), self.D(x, U, V)
        w = nn.softmax(self.gate(jnp.concatenate([x, A, D], axis=-1)), axis=1)
        return w[:, :1] * A + w[:, 1:] * D
