import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class DiffusionPDE(BasePDE):
    """1-D diffusion (heat) equation: u_t = α u_xx

    Benchmark (exact solution known):
        Domain : x ∈ [0, 1],  t ∈ [0, T]
        IC     : u(x, 0) = sin(πx)
        BC     : u(0, t) = u(1, t) = 0
        Exact  : u(x, t) = sin(πx) exp(-α π² t)

    The diffusivity `alpha` may be a Python float (forward problem) or a JAX
    scalar included in the optimizer's parameter tree (inverse problem).  When
    passed explicitly to `residual` / `exact` it overrides ``self.alpha``.

    Parameters
    ----------
    model : Flax module — input (N, 2) → output (N, 1)
    alpha : thermal diffusivity (default 0.01)
    """

    def __init__(self, model, alpha: float = 0.01):
        self.model = model
        self.alpha = alpha

    def u(self, params, x, t):
        return self.model.apply(params, jnp.stack([x, t], axis=1))[:, 0]

    def residual(self, params, xt, alpha=None):
        """Compute u_t − α·u_xx at collocation points.

        Parameters
        ----------
        xt    : (N, 2) packed array — xt[:, 0] = x, xt[:, 1] = t.
        alpha : Overrides ``self.alpha`` when given (inverse-problem use).
        """
        a = self.alpha if alpha is None else alpha

        def u_single(xy_i):
            return self.model.apply(params, xy_i[None, :])[0, 0]

        J = jax.vmap(jax.jacfwd(u_single))(xt)
        H = jax.vmap(jax.hessian(u_single))(xt)
        u_t  = J[:, 1]
        u_xx = H[:, 0, 0]
        return u_t - a * u_xx

    def exact(self, x, t, alpha=None):
        a = self.alpha if alpha is None else alpha
        return jnp.sin(jnp.pi * x) * jnp.exp(-a * jnp.pi ** 2 * t)
