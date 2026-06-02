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


class GatedMLP(nn.Module):
    """Modified MLP with U/V input encoders fused into every hidden layer.

    Two encoders are computed **once** from the raw input, then every hidden
    layer blends them using a gate derived from its own input state::

        U = tanh(W_U · x_raw + b_U)     # encoder 1 — from raw input
        V = tanh(W_V · x_raw + b_V)     # encoder 2 — from raw input

        H = x_raw
        for k in 1 … L:
            Z   = tanh(W_k · H + b_k)   # gate from current hidden state
            H   = Z ⊙ U  +  (1 − Z) ⊙ V

        output = W_out · H + b_out

    ``Z ∈ (−1, 1)`` acts as a learnable blend between the two encoder views
    of the original input, so each hidden layer can recover lost input
    information at any depth.  This cures the pathological gradient flow of
    plain tanh MLPs on stiff PDEs and was introduced for PINNs in Wang et al.
    (2022), *"Improved Architectures and Training Algorithms for Deep Operator
    Networks"*.

    Parameters
    ----------
    layers : list
        Same format as :class:`MLP`: ``[in_dim, h1, h2, …, out_dim]``.
        All hidden widths must be equal (``h1 == h2 == … == h_{L-1}``)
        because ``U`` and ``V`` are shared across layers.

    Notes
    -----
    Parameter count per hidden layer: ``width × width + width`` (same as MLP)
    plus the two encoder Dense layers ``2 × (in_dim × width + width)``
    amortised over all layers.

    Examples
    --------
    >>> net = GatedMLP(layers=[2, 128, 128, 128, 128, 3])
    >>> out = net.apply(params, x)   # x: (N, 2)  →  out: (N, 3)
    """

    layers: list

    @nn.compact
    def __call__(self, x):
        x_raw    = x
        hidden_w = self.layers[1]

        # ── Input encoders (computed once; shared across all hidden layers) ──
        U = nn.tanh(nn.Dense(hidden_w, name="enc_U")(x_raw))
        V = nn.tanh(nn.Dense(hidden_w, name="enc_V")(x_raw))

        # ── Hidden layers: gate-blend of the two encoders ────────────────────
        H = x_raw
        for width in self.layers[1:-1]:
            Z = nn.tanh(nn.Dense(width)(H))   # gate from current state
            H = Z * U + (1.0 - Z) * V         # convex blend of U and V

        # ── Linear output projection ─────────────────────────────────────────
        return nn.Dense(self.layers[-1])(H)


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
