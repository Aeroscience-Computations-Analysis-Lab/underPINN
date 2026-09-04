import jax
import jax.numpy as jnp
from typing import Callable, Optional
from underPINN.core.base import BasePDE


class SteadyHeatPDE(BasePDE):
    """2-D steady-state heat / Poisson equation.

        ∇²u = -f(x, y)     on Ω
        u   = g(x, y)     on ∂Ω

    When ``source_fn`` is ``None`` the equation reduces to the Laplace equation.

    Parameters
    ----------
    model :
        Flax module.  Input shape ``(N, 2)`` (x, y), output ``(N, 1)``.
    source_fn : callable or None
        ``f(x, y) -> (N,)`` JAX array.  Sign convention: ∇²u + f = 0.
        Pass ``None`` for the homogeneous Laplace equation.

    Canonical test case
    -------------------
    Domain  : [0, 1]²
    Source  : f(x, y) = 2π² sin(πx) sin(πy)
    BCs     : u = 0 on all four edges
    Exact   : u(x, y) = sin(πx) sin(πy)

    Verification:
        u_xx = -π² sin(πx) sin(πy)
        u_yy = -π² sin(πx) sin(πy)
        u_xx + u_yy = -2π² sin(πx) sin(πy) = -f   ✓
    """

    def __init__(self, model, source_fn: Optional[Callable] = None):
        self.model = model
        self.source_fn = source_fn

    def u(self, params, xy: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the network at spatial points.

        Parameters
        ----------
        xy : (N, 2)
        Returns
        -------
        (N,)
        """
        return self.model.apply(params, xy)[:, 0]

    def residual(self, params, xy: jnp.ndarray) -> jnp.ndarray:
        """Compute the PDE residual ∇²u + f at collocation points.

        Parameters
        ----------
        xy : (N, 2)
        Returns
        -------
        (N,)  — should be zero everywhere in the domain
        """
        def u_single(xy_i):
            return self.u(params, xy_i[None, :])[0]

        # H[i, j, k] = ∂²u/∂x_j ∂x_k evaluated at point i
        H = jax.vmap(jax.hessian(u_single))(xy)   # (N, 2, 2)
        laplacian = H[:, 0, 0] + H[:, 1, 1]       # u_xx + u_yy

        if self.source_fn is not None:
            f = self.source_fn(xy[:, 0], xy[:, 1])
            return laplacian + f                   # ∇²u + f = 0
        return laplacian                           # ∇²u = 0

    def exact(self, xy: jnp.ndarray) -> jnp.ndarray:
        """Exact solution for the canonical Poisson test case.

        Only valid when source_fn = 2π² sin(πx) sin(πy).
        """
        return jnp.sin(jnp.pi * xy[:, 0]) * jnp.sin(jnp.pi * xy[:, 1])
