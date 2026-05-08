import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Vanilla multi-layer perceptron with tanh activations for PINNs."""
    layers: list

    @nn.compact
    def __call__(self, x):
        for width in self.layers[1:-1]:
            x = nn.tanh(nn.Dense(width)(x))
        return nn.Dense(self.layers[-1])(x)


class FourierMLP(nn.Module):
    """MLP with a trainable Random Fourier Feature (RFF) embedding at input.

    Encodes the input as [sin(Bx), cos(Bx)] before passing through a
    standard tanh MLP.  This gives the network a richer spectral basis
    and significantly improves accuracy on problems with oscillatory or
    multi-scale solutions (e.g. Burgers, wave equation).

    Parameters
    ----------
    layers : list
        Same format as MLP: [in_dim, h1, h2, ..., out_dim].
        ``layers[0]`` is the raw input dimension (e.g. 2 for (x, t)).
    n_fourier : int
        Number of frequency pairs.  The embedding dimension becomes
        2 * n_fourier.
    sigma : float
        Standard deviation of the random frequency initialisation.
        Larger σ → higher-frequency bias.  Tune to the expected solution
        frequency (e.g. σ~1 for O(1) spatial scales).
    """

    layers: list
    n_fourier: int = 16
    sigma: float = 1.0

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        # Trainable frequency matrix B: (in_dim, n_fourier)
        B = self.param(
            "fourier_B",
            nn.initializers.normal(self.sigma),
            (in_dim, self.n_fourier),
        )
        proj = x @ B                                          # (N, n_fourier)
        x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)  # (N, 2*n_fourier)
        for width in self.layers[1:-1]:
            x = nn.tanh(nn.Dense(width)(x))
        return nn.Dense(self.layers[-1])(x)
