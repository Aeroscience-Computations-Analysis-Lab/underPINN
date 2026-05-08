import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class KEpsilonPDE(BasePDE):
    def __init__(self, model, Re=10000.0):
        self.model = model
        self.Re = Re
        self.c1 = 1.44
        self.c2 = 1.92
        self.s_k = 1.0
        self.s_e = 1.3
        self.u_in = 1.0
        self.L = 1.0

    def u(self, params, x):
        # Forward pass returning [u, v, p, k, eps]
        return self.model.apply(params, x)

    def residual(self, params, x):
        # We use jacfwd to get gradients w.r.t (x,y)
        # Output shape of self.u is (N, 5)
        # Jacobian shape will be (N, 5, 2)
        
        def get_vars(x_point):
            # Helper for single point evaluation to use with jacfwd/hessian
            # x_point shape: (2,)
            out = self.u(params, x_point[None, :])[0]
            return out # (5,)

        # Vectorized Jacobian and Hessian computation
        # Note: For efficiency in JAX, we often map over the batch
        
        # Function mapping: (2,) -> (5,)
        u_fn = lambda x_i: self.u(params, x_i[None, :])[0]
        
        # Jacobians: (N, 5, 2) -> [var_idx, spatial_idx]
        J = jax.vmap(jax.jacfwd(u_fn))(x)
        
        # Hessians: (N, 5, 2, 2)
        H = jax.vmap(jax.hessian(u_fn))(x)

        # Unpack Variables
        u   = J[:, 0, :] # u, v derivatives are not here, these are raw values? No, J is derivative.
        # We need raw values too.
        outputs = self.u(params, x)
        u_val, v_val, p_val, k_val, eps_val = outputs[:,0], outputs[:,1], outputs[:,2], outputs[:,3], outputs[:,4]

        # First Derivatives (x=0, y=1)
        u_x, u_y   = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y   = J[:, 1, 0], J[:, 1, 1]
        p_x, p_y   = J[:, 2, 0], J[:, 2, 1]
        k_x, k_y   = J[:, 3, 0], J[:, 3, 1]
        e_x, e_y   = J[:, 4, 0], J[:, 4, 1]

        # Second Derivatives
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]
        k_xx, k_yy = H[:, 3, 0, 0], H[:, 3, 1, 1]
        e_xx, e_yy = H[:, 4, 0, 0], H[:, 4, 1, 1]

        # Physics Definitions
        mu_t = (0.09 * k_val**2) / (eps_val + 1e-8)
        
        # Production Term P_k
        # P_k = 2 * mu_t * (u_x^2 + v_y^2 + 0.5 * (u_y + v_x)^2)
        P_k = 2 * mu_t * (u_x**2 + v_y**2 + 0.5 * (u_y + v_x)**2)

        # 1. Continuity
        cont = u_x + v_y

        # 2. Momentum X
        # (u u_x + v u_y) + p_x - (1/Re + mu_t)(u_xx + u_yy) ... Note: divergence of stress tensor implies derivatives of mu_t too? 
        # The user code implements simplified version: (1/Re + mu_t)*(laplacian). We follow user code.
        mom_x = (u_val * u_x + v_val * u_y) + p_x - (1.0/self.Re + mu_t) * (u_xx + u_yy)

        # 3. Momentum Y
        mom_y = (u_val * v_x + v_val * v_y) + p_y - (1.0/self.Re + mu_t) * (v_xx + v_yy)

        # 4. K-Transport
        # u k_x + v k_y - ...
        diff_k = (1.0/self.Re + mu_t/self.s_k) * (k_xx + k_yy)
        T_k = (u_val * k_x + v_val * k_y) - diff_k - P_k + eps_val

        # 5. Epsilon-Transport
        diff_e = (1.0/self.Re + mu_t/self.s_e) * (e_xx + e_yy)
        prod_e = self.c1 * (eps_val / (k_val + 1e-8)) * P_k
        diss_e = self.c2 * (eps_val**2 / (k_val + 1e-8))
        T_e = (u_val * e_x + v_val * e_y) - diff_e - prod_e + diss_e

        return jnp.stack([cont, mom_x, mom_y, T_k, T_e], axis=1)