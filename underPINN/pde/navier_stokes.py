import jax
import jax.numpy as jnp

class NavierStokesPDE:
    def __init__(self, model, Re=100.0):
        self.model = model
        self.Re = Re

    def u(self, params, x):
        return self.model.apply(params, x)

    def residual(self, params, x):
        # u_fn returns (u, v, p)
        u_fn = lambda x_i: self.u(params, x_i[None, :])[0]
        
        # Jacobian (1st derivatives)
        J = jax.vmap(jax.jacfwd(u_fn))(x)
        # Hessian (2nd derivatives)
        H = jax.vmap(jax.hessian(u_fn))(x)

        # Extract values
        out = self.u(params, x)
        u, v, p = out[:, 0], out[:, 1], out[:, 2]

        # First derivatives [0:x, 1:y]
        u_x, u_y = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y = J[:, 1, 0], J[:, 1, 1]
        p_x, p_y = J[:, 2, 0], J[:, 2, 1]

        # Second derivatives
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]

        # Physics Equations (Conservative Form matching PyTorch)
        # cont = u_x + v_y
        cont = u_x + v_y
        
        # mom_x = (u^2)_x + (uv)_y + p_x - (1/Re)(u_xx + u_yy)
        # (u^2)_x = 2*u*u_x
        # (uv)_y = u_y*v + u*v_y
        term_x1 = 2 * u * u_x
        term_x2 = u_y * v + u * v_y
        mom_x = term_x1 + term_x2 + p_x - (1.0/self.Re) * (u_xx + u_yy)

        # mom_y = (uv)_x + (v^2)_y + p_y - (1/Re)(v_xx + v_yy)
        # (uv)_x = u_x*v + u*v_x
        # (v^2)_y = 2*v*v_y
        term_y1 = u_x * v + u * v_x
        term_y2 = 2 * v * v_y
        mom_y = term_y1 + term_y2 + p_y - (1.0/self.Re) * (v_xx + v_yy)

        return jnp.stack([cont, mom_x, mom_y], axis=1)