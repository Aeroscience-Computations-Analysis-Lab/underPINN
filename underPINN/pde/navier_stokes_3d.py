import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class SteadyNS3DPDE(BasePDE):
    """3-D steady incompressible Navier-Stokes equations.

    Network maps (x, y, z) → (u, v, w, p).
    Build model with ``layers[0] = 3``, ``layers[-1] = 4``.

    Non-dimensional PDE system (ρ = 1):
        ∇·u = 0
        (u·∇)u = −∇p + (1/Re) Δu

    Parameters
    ----------
    model : Flax module — input (N, 3) → output (N, 4)
    Re    : Reynolds number
    """

    def __init__(self, model, Re: float = 100.0):
        self.model = model
        self.Re = Re

    def uvwp(self, params, xyz):
        """Return (u, v, w, p) arrays at points xyz (N, 3)."""
        out = self.model.apply(params, xyz)  # (N, 4)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    def residual(self, params, xyz):
        """Compute 4 PDE residuals at collocation points xyz (N, 3).

        Returns
        -------
        cont, mom_x, mom_y, mom_z : each (N,)
        """
        nu = 1.0 / self.Re

        def net_single(xyz_i):
            return self.model.apply(params, xyz_i[None, :])[0, :]  # (4,)

        def jac_single(xyz_i):
            # J[i, j] = d f_i / d xyz_j,  shape (4, 3)
            return jax.jacfwd(net_single)(xyz_i)

        def compute(xyz_i):
            uvwp_i = net_single(xyz_i)                    # (4,)
            J      = jac_single(xyz_i)                    # (4, 3)
            # H[i, j, k] = d(J[i,j]) / d xyz_k,  shape (4, 3, 3)
            H      = jax.jacfwd(jac_single)(xyz_i)        # (4, 3, 3)
            return uvwp_i, J, H

        uvwp_b, J, H = jax.vmap(compute)(xyz)  # (N,4), (N,4,3), (N,4,3,3)

        u, v, w      = uvwp_b[:, 0], uvwp_b[:, 1], uvwp_b[:, 2]
        u_x, u_y, u_z = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
        v_x, v_y, v_z = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
        w_x, w_y, w_z = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
        p_x, p_y, p_z = J[:, 3, 0], J[:, 3, 1], J[:, 3, 2]

        # Laplacian of each velocity component: trace of spatial Hessian
        lap_u = H[:, 0, 0, 0] + H[:, 0, 1, 1] + H[:, 0, 2, 2]
        lap_v = H[:, 1, 0, 0] + H[:, 1, 1, 1] + H[:, 1, 2, 2]
        lap_w = H[:, 2, 0, 0] + H[:, 2, 1, 1] + H[:, 2, 2, 2]

        cont  = u_x + v_y + w_z
        mom_x = u * u_x + v * u_y + w * u_z + p_x - nu * lap_u
        mom_y = u * v_x + v * v_y + w * v_z + p_y - nu * lap_v
        mom_z = u * w_x + v * w_y + w * w_z + p_z - nu * lap_w

        # Return shape (N, 4): [cont, mom_x, mom_y, mom_z]
        return jnp.stack([cont, mom_x, mom_y, mom_z], axis=-1)

    def exact_poiseuille(self, xyz, R: float = 0.5, U_max: float = 1.0,
                         L: float = 2.0):
        """Hagen-Poiseuille exact solution.

        Assumes x is the pipe axis (x ∈ [0, L]), radial distance
        r² = y² + z².  Pressure is anchored to zero at the outlet x = L.
        Valid for any Re since the nonlinear term vanishes (∂u/∂x = 0).
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r2   = y ** 2 + z ** 2
        u    = U_max * (1.0 - r2 / R ** 2)
        v    = jnp.zeros_like(u)
        w    = jnp.zeros_like(u)
        nu   = 1.0 / self.Re
        dpdx = -4.0 * nu * U_max / R ** 2   # negative pressure gradient
        p    = dpdx * (x - L)               # p = 0 at x = L
        return u, v, w, p
