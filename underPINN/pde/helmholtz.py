import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class HelmholtzPDE(BasePDE):
    """2-D Helmholtz equation: Δu + k² u = f(x, y)

    Benchmark (Dirichlet BCs, exact solution known):
        Domain  : (x, y) ∈ [0, 1]²
        Source  : f = -(2π² - k²) sin(πx) sin(πy)
        BCs     : u = 0 on all edges
        Exact   : u = sin(πx) sin(πy)

    For k > 1 the source and solution become increasingly oscillatory;
    FourierMLP is recommended over a plain MLP.

    Parameters
    ----------
    model : Flax module — input (N, 2) → output (N, 1)
    k     : wave number (default 1.0)
    """

    def __init__(self, model, k: float = 1.0):
        self.model = model
        self.k = k

    def u(self, params, xy):
        return self.model.apply(params, xy)[:, 0]

    def source(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return -(2.0 * jnp.pi ** 2 - self.k ** 2) * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    def residual(self, params, xy):
        def u_single(xy_i):
            return self.model.apply(params, xy_i[None, :])[0, 0]

        H = jax.vmap(jax.hessian(u_single))(xy)
        laplacian = H[:, 0, 0] + H[:, 1, 1]

        u = self.model.apply(params, xy)[:, 0]
        f = self.source(xy)
        return laplacian + self.k ** 2 * u - f

    def exact(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
