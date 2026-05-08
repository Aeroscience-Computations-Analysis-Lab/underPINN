import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class BurgersPDE(BasePDE):
    """Viscous Burgers equation: u_t + u·u_x = ν·u_xx

    Previous implementation had a critical bug: u_xx was always zero because
    the inner lambda captured `ux` as a closed-over constant, making its
    gradient w.r.t. x identically zero — viscous dissipation was silently
    dropped and the equation reduced to inviscid Burgers.

    Fixed by computing derivatives with jax.jacfwd / jax.hessian via vmap,
    consistent with NavierStokesPDE and KEpsilonPDE.

    Parameters
    ----------
    nu : float
        Kinematic viscosity.  Default 0.01 (standard benchmark value).
    """

    def __init__(self, model, nu: float = 0.01):
        self.model = model
        self.nu = nu

    def u(self, params, x, t):
        return self.model.apply(params, jnp.stack([x, t], axis=1))[:, 0]

    def residual(self, params, x, t):
        xy = jnp.stack([x, t], axis=1)  # (N, 2)

        def u_single(xy_i):
            """Scalar network output at one (x, t) point."""
            return self.model.apply(params, xy_i[None, :])[0, 0]

        # First derivatives via forward-mode AD → (N, 2): [u_x, u_t]
        J = jax.vmap(jax.jacfwd(u_single))(xy)
        ux = J[:, 0]
        ut = J[:, 1]

        # Second derivative u_xx from Hessian diagonal → (N, 2, 2)
        H = jax.vmap(jax.hessian(u_single))(xy)
        uxx = H[:, 0, 0]

        u = self.model.apply(params, xy)[:, 0]
        return ut + u * ux - self.nu * uxx
