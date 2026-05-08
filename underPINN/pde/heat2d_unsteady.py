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

    def residual(self, params, xy, t, alpha=None):
        """Compute u_t − α (u_xx + u_yy) at collocation points.

        Parameters
        ----------
        xy    : (N, 2) spatial coordinates
        t     : (N,)  times
        alpha : override self.alpha (for inverse problems)
        """
        a   = self.alpha if alpha is None else alpha
        xyt = jnp.concatenate([xy, t[:, None]], axis=1)   # (N, 3)

        def u_single(xyt_i):
            return self.model.apply(params, xyt_i[None, :])[0, 0]

        # First-order: J[i] = (u_x, u_y, u_t)
        J = jax.vmap(jax.jacfwd(u_single))(xyt)           # (N, 3)
        # Second-order: H[i, j, k] = ∂²u / ∂x_j ∂x_k
        H = jax.vmap(jax.hessian(u_single))(xyt)          # (N, 3, 3)

        u_t  = J[:, 2]
        u_xx = H[:, 0, 0]
        u_yy = H[:, 1, 1]
        return u_t - a * (u_xx + u_yy)

    def exact(self, xy, t, alpha=None):
        """Exact solution for the canonical IC sin(πx)sin(πy)."""
        a = self.alpha if alpha is None else alpha
        x, y = xy[:, 0], xy[:, 1]
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.exp(-2.0 * a * jnp.pi ** 2 * t)
