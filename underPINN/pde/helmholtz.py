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

        # jax.hessian's full 2x2 here is already near-minimal -- both
        # diagonal entries (u_xx, u_yy) are needed, matching the input
        # dimensionality, so there is no cheaper way to reach *them*
        # (unlike underPINN/pde/burgers.py, where only 1 of 4 entries was
        # used). What *was* wasteful, in isolation: a separate `model.apply`
        # call for the raw value `u`, duplicating a forward pass
        # jax.hessian's own internal computation already performs.
        #
        # A fused vjp+jvp version (one jax.vjp for value+gradient, two
        # targeted jax.jvp calls for the Hessian diagonal, mirroring
        # burgers.py) *was* tried here, and measured 1.35-1.58x faster than
        # this jax.hessian formulation -- but only as an isolated forward
        # residual call. Once wrapped in the outer
        # jax.value_and_grad(loss)(params, ...) an actual training step
        # needs -- differentiating the *whole* residual w.r.t. network
        # parameters, not just evaluating it -- it measured 0.84x (~16%
        # *slower*), reproducibly across repeated runs of
        # benchmarks/rebuttal/physicsnemo/compare_underpinn_multi.py
        # --problems helmholtz2d. Backpropagating through several
        # independently-traced nested vjp/jvp subgraphs costs more than
        # backpropagating through XLA's single fused jax.hessian primitive,
        # even though the forward evaluation alone is cheaper. Reverted to
        # this jax.hessian formulation for that reason -- see
        # benchmarks/rebuttal/README.md section 6 for the full measurement and
        # the general lesson (verify under the real outer parameter-
        # gradient, not a forward-only microbenchmark).
        H = jax.vmap(jax.hessian(u_single))(xy)
        u = self.model.apply(params, xy)[:, 0]
        laplacian = H[:, 0, 0] + H[:, 1, 1]
        f = self.source(xy)
        return laplacian + self.k ** 2 * u - f

    def exact(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
