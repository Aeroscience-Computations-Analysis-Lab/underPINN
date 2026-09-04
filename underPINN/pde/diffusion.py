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

        # Was: separate jax.jacfwd (2 fwd passes, only J[:,1]=u_t used) and
        # jax.hessian (full 2x2, only H[:,0,0]=u_xx used) -- same redundancy
        # profiled and fixed in underPINN/pde/burgers.py: one jax.vjp gives
        # the full gradient (u_x, u_t) in a single reverse pass, and one
        # targeted jax.jvp of that gradient gives exactly u_xx.
        def per_point(xy_i):
            _, vjp_fn = jax.vjp(u_single, xy_i)

            def grad_only(z):
                return jax.vjp(u_single, z)[1](1.0)[0]

            grad_vec = vjp_fn(1.0)[0]
            _, jvp_out = jax.jvp(grad_only, (xy_i,), (jnp.array([1.0, 0.0]),))
            return grad_vec[1], jvp_out[0]   # u_t, u_xx

        u_t, u_xx = jax.vmap(per_point)(xt)
        return u_t - a * u_xx

    def exact(self, x, t, alpha=None):
        a = self.alpha if alpha is None else alpha
        return jnp.sin(jnp.pi * x) * jnp.exp(-a * jnp.pi ** 2 * t)
