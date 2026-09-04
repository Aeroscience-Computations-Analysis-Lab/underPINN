import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class UnsteadyHeat2DPDE(BasePDE):
    """2-D unsteady diffusion / heat equation: u_t = α (u_xx + u_yy)

    Benchmark (exact solution known):
        Domain : (x, y) ∈ [0, 1]²,  t ∈ [0, T]
        IC     : u(x, y, 0) = sin(πx) sin(πy)
        BC     : u = 0 on all four edges
        Exact  : u(x, y, t) = sin(πx) sin(πy) exp(−2α π² t)

    The network maps (x, y, t) → u, so the input dimension is 3.
    Build the model with ``layers[0] = 3``, e.g.
    ``MLP([3, 64, 64, 64, 1])``.

    The diffusivity ``alpha`` may be a plain float (forward problem) or a
    JAX scalar in the optimiser's parameter tree (inverse problem) — pass
    it explicitly to :meth:`residual` / :meth:`exact` in that case.

    Parameters
    ----------
    model : Flax module — input (N, 3) → output (N, 1)
    alpha : thermal diffusivity (default 0.01)
    """

    def __init__(self, model, alpha: float = 0.01):
        self.model = model
        self.alpha = alpha

    def u(self, params, xy, t):
        """Evaluate u at spatial points xy (N, 2) and times t (N,)."""
        xyt = jnp.concatenate([xy, t[:, None]], axis=1)   # (N, 3)
        return self.model.apply(params, xyt)[:, 0]

    def residual(self, params, xyt, alpha=None):
        """Compute u_t − α (u_xx + u_yy) at collocation points.

        Parameters
        ----------
        xyt   : (N, 3) packed array — xyt[:, 0:2] = (x, y), xyt[:, 2] = t.
        alpha : Overrides ``self.alpha`` when given (inverse-problem use).
        """
        a = self.alpha if alpha is None else alpha

        def u_single(xyt_i):
            return self.model.apply(params, xyt_i[None, :])[0, 0]

        # Was: separate jax.jacfwd (3 fwd passes for the full 3-vector
        # (u_x,u_y,u_t), only u_t used) and jax.hessian (full 3x3=9 entries,
        # only the 2 diagonal u_xx/u_yy used, u_tt and all 4 cross terms
        # discarded) -- same redundancy profiled and fixed in
        # underPINN/pde/burgers.py. One jax.vjp gives the full gradient
        # (hence u_t) in a single reverse pass; two targeted jax.jvp calls
        # of that gradient (x- and y-direction tangents only, skipping the
        # unused t-direction) give exactly u_xx and u_yy.
        def per_point(xyt_i):
            def grad_only(z):
                return jax.vjp(u_single, z)[1](1.0)[0]

            grad_vec = grad_only(xyt_i)
            _, jvp_x = jax.jvp(grad_only, (xyt_i,), (jnp.array([1.0, 0.0, 0.0]),))
            _, jvp_y = jax.jvp(grad_only, (xyt_i,), (jnp.array([0.0, 1.0, 0.0]),))
            return grad_vec[2], jvp_x[0], jvp_y[1]   # u_t, u_xx, u_yy

        u_t, u_xx, u_yy = jax.vmap(per_point)(xyt)
        return u_t - a * (u_xx + u_yy)

    def exact(self, xy, t, alpha=None):
        """Exact solution for the canonical IC sin(πx)sin(πy)."""
        a = self.alpha if alpha is None else alpha
        x, y = xy[:, 0], xy[:, 1]
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.exp(-2.0 * a * jnp.pi ** 2 * t)
