"""2-D Steady k-ε Turbulence RANS PDE.

Solves the incompressible RANS equations closed by the standard k-ε model on a
backward-facing step (BFS) geometry.

Network output: (u, v, p, k, ε) — 5 fields.
Positivity of k and ε is enforced via an output transform applied inside the
network (pass ``out_transform=k_eps_positivity`` to FBPINN).  The PDE then
operates directly on the already-transformed physical values.

Equations
---------
1. Continuity:     u_x + v_y = 0
2. Momentum-x:     u u_x + v u_y + p_x − (1/Re + μ_t)(u_xx + u_yy) = 0
3. Momentum-y:     u v_x + v v_y + p_y − (1/Re + μ_t)(v_xx + v_yy) = 0
4. k-transport:    u k_x + v k_y − ∇·((1/Re + μ_t/σ_k)∇k) − P_k + ε = 0
5. ε-transport:    u ε_x + v ε_y − ∇·((1/Re + μ_t/σ_ε)∇ε) − C1 ε/k P_k + C2 ε²/k = 0

where  μ_t = C_μ k²/ε,  P_k = 2μ_t(S_ij S_ij),  S_ij = ½(u_{i,j} + u_{j,i}).
Standard constants: C_μ=0.09, C1=1.44, C2=1.92, σ_k=1.0, σ_ε=1.3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from underPINN.core.base import BasePDE


class KEpsilonPDE(BasePDE):
    """Standard k-ε RANS PDE for 2-D steady incompressible flow.

    Parameters
    ----------
    model : Flax module
        Network  (x, y) → (u, v, p, k, ε).  k and ε **must be positive**;
        use ``out_transform=k_eps_positivity`` in the FBPINN constructor.
    Re : float
        Reynolds number (ν = 1/Re).
    """

    # Standard k-ε model constants
    C_mu = 0.09
    C1   = 1.44
    C2   = 1.92
    sigma_k = 1.0
    sigma_e = 1.3

    def __init__(self, model, Re: float = 10000.0):
        self.model = model
        self.Re    = float(Re)

    def u(self, params, x):
        """Forward pass — returns (u, v, p, k, ε) as (N, 5)."""
        return self.model.apply(params, x)

    def residual(self, params, x):
        """Compute all five PDE residuals at collocation points x (N, 2).

        Returns
        -------
        (N, 5) stacked array: [cont, mom_x, mom_y, T_k, T_ε]
        """
        Re    = self.Re
        C_mu  = self.C_mu
        C1    = self.C1
        C2    = self.C2
        s_k   = self.sigma_k
        s_e   = self.sigma_e

        # Per-point function for jacfwd
        def u_fn(x_i):  # (2,) → (5,)
            return self.u(params, x_i[None, :])[0]

        # Jacobian (N, 5, 2): forward-mode over a low (2-D) input is already
        # close to optimal (2 fwd passes give all 10 first derivatives used
        # below) -- left as-is, as is the plain-forward-pass value
        # extraction just below it.
        J = jax.vmap(jax.jacfwd(u_fn))(x)

        # Raw field values
        out = self.u(params, x)
        u_val, v_val, _ = out[:, 0], out[:, 1], out[:, 2]
        k_val, e_val        = out[:, 3], out[:, 4]

        # First derivatives  (spatial index: 0=x, 1=y)
        u_x, u_y = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y = J[:, 1, 0], J[:, 1, 1]
        p_x, p_y = J[:, 2, 0], J[:, 2, 1]
        k_x, k_y = J[:, 3, 0], J[:, 3, 1]
        e_x, e_y = J[:, 4, 0], J[:, 4, 1]

        # Second derivatives (Laplacians): jax.hessian(u_fn) computes the
        # full (5, 2, 2) tensor -- 20 entries -- for all 5 components
        # including pressure, whose second derivatives are never used, and
        # only the 2 diagonal entries of u/v/k/epsilon's 2x2 blocks are used
        # out of 4 each (8 of 20 total).
        #
        # A per-component, diagonal-only replacement (identical in
        # structure to underPINN/pde/navier_stokes.py's attempted fix, four
        # components' worth of vjp+2xjvp here instead of two) *was* tried
        # -- and, as an isolated forward residual call, measured faster.
        # But wrapped in the outer jax.value_and_grad(loss)(params, ...) an
        # actual training step needs, it measured 0.49x (~2x *slower*) at
        # the K-Epsilon config (layers=[2,96,96,96,96,96,5], batch=2048) --
        # the worst regression of the three files this happened to (see
        # helmholtz.py and navier_stokes.py for the other two), consistent
        # with backpropagating through *more* independently-traced
        # nested vjp/jvp subgraphs (8 here, vs. helmholtz's 2 and
        # navier_stokes's 4) costing correspondingly more. Reverted to
        # jax.hessian for that reason -- see
        # benchmarks/rebuttal/README.md section 6 for the full measurement.
        H = jax.vmap(jax.hessian(u_fn))(x)
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]
        k_xx, k_yy = H[:, 3, 0, 0], H[:, 3, 1, 1]
        e_xx, e_yy = H[:, 4, 0, 0], H[:, 4, 1, 1]

        # Turbulent eddy viscosity  μ_t = C_μ k² / ε
        mu_t = C_mu * k_val ** 2 / (e_val + 1e-8)

        # Turbulent kinetic energy production
        # P_k = 2 μ_t (u_x² + v_y² + ½(u_y + v_x)²)
        P_k = 2.0 * mu_t * (u_x ** 2 + v_y ** 2 + 0.5 * (u_y + v_x) ** 2)

        # 1. Continuity
        cont = u_x + v_y

        # 2. Momentum-x  (advection + pressure gradient − effective diffusion)
        mom_x = (u_val * u_x + v_val * u_y + p_x
                 - (1.0 / Re + mu_t) * (u_xx + u_yy))

        # 3. Momentum-y
        mom_y = (u_val * v_x + v_val * v_y + p_y
                 - (1.0 / Re + mu_t) * (v_xx + v_yy))

        # 4. k-transport
        diff_k = (1.0 / Re + mu_t / s_k) * (k_xx + k_yy)
        T_k = u_val * k_x + v_val * k_y - diff_k - P_k + e_val

        # 5. ε-transport
        diff_e  = (1.0 / Re + mu_t / s_e) * (e_xx + e_yy)
        prod_e  = C1 * (e_val / (k_val + 1e-8)) * P_k
        diss_e  = C2 * e_val ** 2 / (k_val + 1e-8)
        T_e = u_val * e_x + v_val * e_y - diff_e - prod_e + diss_e

        return jnp.stack([cont, mom_x, mom_y, T_k, T_e], axis=1)
