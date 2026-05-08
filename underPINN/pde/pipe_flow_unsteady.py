import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class UnsteadyPipeFlowPDE(BasePDE):
    """Unsteady pipe flow — cross-section formulation.

    Fully-developed, axially-uniform problem in the (y, z, t) domain:

        u_t = G + ν(u_yy + u_zz)   for  r = √(y²+z²) < R,  t > 0
        G   = 4 ν U_max / R²         (constant pressure-gradient body force)
        ν   = 1 / Re

    Initial condition : u(y, z, 0) = 0     (fluid at rest)
    Boundary condition: u = 0  at  r = R   (no-slip wall)
    Steady solution   : u_∞(r) = U_max (1 − r²/R²)

    Exact transient solution (Stokes starting-flow, Bessel series):

        u(r, t) = U_max(1 − r²/R²)
                  − 8 U_max  Σ  J₀(αₙ r/R) / (αₙ³ J₁(αₙ))  exp(−αₙ² ν t / R²)

    where αₙ are the positive zeros of J₀ (computed via scipy.special.jn_zeros).

    Network maps (y, z, t) → u (scalar).
    Build with ``layers[0] = 3``, ``layers[-1] = 1``.

    Parameters
    ----------
    model  : Flax module  (N, 3) → (N, 1)
    Re     : Reynolds number  (ν = 1/Re)
    R      : pipe radius
    U_max  : centreline velocity of the Poiseuille steady profile
    """

    def __init__(self, model, Re: float = 10.0, R: float = 0.5,
                 U_max: float = 1.0):
        self.model = model
        self.Re    = Re
        self.R     = R
        self.U_max = U_max

    def u(self, params, yz, t):
        """Evaluate u(y, z, t).  yz: (N, 2),  t: (N,)."""
        yzt = jnp.concatenate([yz, t[:, None]], axis=1)   # (N, 3)
        return self.model.apply(params, yzt)[:, 0]

    def residual(self, params, yz, t):
        """Compute  u_t − ν(u_yy + u_zz) − G  at collocation points.

        Parameters
        ----------
        yz : (N, 2)  spatial coordinates inside the disk
        t  : (N,)    times in [0, T]

        Returns
        -------
        (N,) residual array
        """
        nu  = 1.0 / self.Re
        G   = 4.0 * nu * self.U_max / self.R ** 2
        yzt = jnp.concatenate([yz, t[:, None]], axis=1)   # (N, 3)

        def u_single(yzt_i):
            return self.model.apply(params, yzt_i[None, :])[0, 0]

        # J[n, k] = du/d(yzt_k)  →  indices 0,1,2 = y, z, t
        J = jax.vmap(jax.jacfwd(u_single))(yzt)    # (N, 3)
        # H[n, j, k] = d²u / (d(yzt_j) d(yzt_k))
        H = jax.vmap(jax.hessian(u_single))(yzt)   # (N, 3, 3)

        u_t  = J[:, 2]
        u_yy = H[:, 0, 0]
        u_zz = H[:, 1, 1]
        return u_t - nu * (u_yy + u_zz) - G

    def exact(self, yz, t_val: float, N_terms: int = 30):
        """Bessel-series exact solution at a fixed scalar time t_val.

        Parameters
        ----------
        yz    : (N, 2) array of (y, z) coordinates inside the disk
        t_val : evaluation time (scalar float)

        Returns
        -------
        (N,) float32 array of u values
        """
        from scipy.special import j0, j1, jn_zeros
        import numpy as np

        alphas = jn_zeros(0, N_terms)                        # zeros of J₀
        yz_np  = np.asarray(yz, dtype=np.float64)
        r      = np.sqrt(yz_np[:, 0] ** 2 + yz_np[:, 1] ** 2)
        nu     = 1.0 / self.Re

        u = self.U_max * (1.0 - r ** 2 / self.R ** 2)
        for a in alphas:
            coeff = -8.0 * self.U_max / (a ** 3 * j1(a))
            u    += coeff * j0(a * r / self.R) * np.exp(
                        -a ** 2 * nu * t_val / self.R ** 2)
        return u.astype(np.float32)
