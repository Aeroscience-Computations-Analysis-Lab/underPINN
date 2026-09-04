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


class SIREN(nn.Module):
    """Sinusoidal Representation Network (Sitzmann et al., 2020).

    A fully-connected network with ``sin`` activations everywhere.  Because
    ``sin`` and all its derivatives are smooth and bounded, SIREN represents
    oscillatory / high-frequency signals — and their derivatives — far better
    than tanh MLPs, which makes it a strong **general time-dependent** network
    when the temporal content is multi-frequency or its frequencies are *not*
    known a priori (unlike :class:`TemporalFourierMLP`, which hard-codes a known
    base frequency).  It applies to space and time jointly.

    Uses the SIREN initialisation scheme so activations stay well-scaled at
    depth:  the first layer draws weights from U(−1/n, 1/n) and is scaled by
    ``w0``; hidden layers draw from U(−√(6/n)/w0, √(6/n)/w0).

    Parameters
    ----------
    layers : ``[in_dim, h1, …, out_dim]`` (same convention as :class:`MLP`).
    w0     : first-layer frequency scale.  The SIREN paper uses 30 for images;
             for PINNs a smaller value (≈ 5–15) avoids excessively high
             frequencies.  Default 15.
    """

    layers: list
    w0: float = 15.0

    @nn.compact
    def __call__(self, x):
        import math as _math

        def _first_init(key, shape, dtype=jnp.float32):
            n = shape[0]
            return jax.random.uniform(key, shape, dtype, -1.0 / n, 1.0 / n)

        def _hidden_init(key, shape, dtype=jnp.float32):
            b = _math.sqrt(6.0 / shape[0]) / self.w0
            return jax.random.uniform(key, shape, dtype, -b, b)

        h = jnp.sin(self.w0 * nn.Dense(self.layers[1],
                                       kernel_init=_first_init)(x))
        for width in self.layers[2:-1]:
            h = jnp.sin(nn.Dense(width, kernel_init=_hidden_init)(h))
        return nn.Dense(self.layers[-1], kernel_init=_hidden_init)(h)


class TemporalFourierMLP(nn.Module):
    """MLP/GatedMLP with a deterministic **temporal-Fourier** embedding.

    Designed for time-dependent PINNs whose temporal dynamics is dominated by a
    *known* base frequency ``ω`` (e.g. pulsatile flow forced at ω = 2π/T_period).
    The time input ``τ`` is lifted to harmonics of ω before the network:

        γ(τ) = [ τ,  sin(ωτ), cos(ωτ),  …,  sin(Kωτ), cos(Kωτ) ]

    concatenated with the (raw) spatial coordinates.  Injecting the oscillatory
    basis directly means the network no longer has to synthesise sinusoids out
    of tanh units — it only learns the slowly-varying spatial amplitude/phase
    modulation.  In practice this overcomes the temporal spectral bias and
    converges to accurate periodic dynamics in **far fewer epochs per window**.

    Parameters
    ----------
    layers           : ``[in_dim, h1, …, out_dim]`` (``in_dim`` is informational;
                       the true input width is recomputed from the embedding).
                       Hidden widths must be equal when ``gated=True``.
    omega            : base angular frequency ω (rad).  ω = 2π/T_period.
    n_time_harmonics : number K of (sin, cos) harmonic pairs (default 4).
    time_index       : which input column is time (default -1 = last).
    gated            : use the GatedMLP U/V backbone (default True) else plain MLP.
    """

    layers: list
    omega: float
    n_time_harmonics: int = 4
    time_index: int = -1
    gated: bool = True

    @nn.compact
    def __call__(self, x):
        d  = x.shape[-1]
        ti = self.time_index % d
        t  = x[:, ti:ti + 1]
        if ti == 0:
            sp = x[:, 1:]
        elif ti == d - 1:
            sp = x[:, :d - 1]
        else:
            sp = jnp.concatenate([x[:, :ti], x[:, ti + 1:]], axis=-1)

        feats = [sp, t]
        for k in range(1, self.n_time_harmonics + 1):
            kw = float(k) * self.omega
            feats.append(jnp.sin(kw * t))
            feats.append(jnp.cos(kw * t))
        h0 = jnp.concatenate(feats, axis=-1)

        if self.gated:
            w1 = self.layers[1]
            U = nn.tanh(nn.Dense(w1, name="enc_U")(h0))
            V = nn.tanh(nn.Dense(w1, name="enc_V")(h0))
            H = h0
            for width in self.layers[1:-1]:
                Z = nn.tanh(nn.Dense(width)(H))
                H = Z * U + (1.0 - Z) * V
            return nn.Dense(self.layers[-1])(H)

        H = h0
        for width in self.layers[1:-1]:
            H = nn.tanh(nn.Dense(width)(H))
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
