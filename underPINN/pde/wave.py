import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class WavePDE(BasePDE):
    """1-D wave equation: u_tt - c² u_xx = 0

    Benchmark (exact solution known):
        Domain : x ∈ [0, 1],  t ∈ [0, T]
        IC     : u(x, 0)   = sin(πx),  u_t(x, 0) = 0
        BC     : u(0, t)   = u(1, t)  = 0
        Exact  : u(x, t)   = sin(πx) cos(c π t)

    Parameters
    ----------
    model : Flax module — input (N, 2) → output (N, 1)
    c     : wave speed (default 1.0)
    """

    def __init__(self, model, c: float = 1.0):
        self.model = model
        self.c = c

    def u(self, params, x, t):
        return self.model.apply(params, jnp.stack([x, t], axis=1))[:, 0]

    def u_t(self, params, x, t):
        """Time-derivative ∂u/∂t at arbitrary (x, t) points."""
        xy = jnp.stack([x, t], axis=1)

        def u_single(xy_i):
            return self.model.apply(params, xy_i[None, :])[0, 0]

        J = jax.vmap(jax.jacfwd(u_single))(xy)
        return J[:, 1]

    def residual(self, params, x, t):
        xy = jnp.stack([x, t], axis=1)

        def u_single(xy_i):
            return self.model.apply(params, xy_i[None, :])[0, 0]

        H = jax.vmap(jax.hessian(u_single))(xy)
        u_tt = H[:, 1, 1]
        u_xx = H[:, 0, 0]
        return u_tt - self.c ** 2 * u_xx

    def exact(self, x, t):
        return jnp.sin(jnp.pi * x) * jnp.cos(self.c * jnp.pi * t)
