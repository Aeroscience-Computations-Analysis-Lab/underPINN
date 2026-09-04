import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class NavierStokesPDE(BasePDE):
    def __init__(self, model, Re=100.0):
        self.model = model
        self.Re = Re

    def u(self, params, x):
        return self.model.apply(params, x)

    def residual(self, params, x):
        # u_fn returns (u, v, p)
        def u_fn(x_i):
            return self.u(params, x_i[None, :])[0]

        # Jacobian (1st derivatives): forward-mode over a low (2-D) input is
        # already close to optimal here (2 fwd passes give all 6 of u_x, u_y,
        # v_x, v_y, p_x, p_y, all of which are used below) -- left as-is.
        J = jax.vmap(jax.jacfwd(u_fn))(x)
        # Extract values: also already close to optimal (one plain forward
        # pass, cheaper than any AD-transform-based way to recover it).
        out = self.u(params, x)
        u, v, _ = out[:, 0], out[:, 1], out[:, 2]

        # First derivatives [0:x, 1:y]
        u_x, u_y = J[:, 0, 0], J[:, 0, 1]
        v_x, v_y = J[:, 1, 0], J[:, 1, 1]
        p_x, p_y = J[:, 2, 0], J[:, 2, 1]

        # Second derivatives: jax.hessian(u_fn) computes the full (3, 2, 2)
        # tensor -- all 4 entries for *every* output component, including
        # pressure, whose second derivatives are never used below -- and
        # only the 2 diagonal entries of u and v's 2x2 blocks are used out
        # of 4 each.
        #
        # A per-component, diagonal-only replacement (one jax.vjp for that
        # component's gradient, two targeted jax.jvp calls for its diagonal
        # Hessian entries, skipping pressure's Hessian and both components'
        # off-diagonal entries entirely) *was* tried here, mirroring
        # underPINN/pde/burgers.py's fix -- and measured faster as an
        # isolated forward residual call. But once wrapped in the outer
        # jax.value_and_grad(loss)(params, ...) an actual training step
        # needs, it measured 0.62x (~38% *slower*) at the LDC config
        # (layers=[2,64,64,64,64,3], batch=2048): backpropagating through
        # two independently-traced nested vjp/jvp subgraphs costs
        # substantially more than backpropagating through XLA's single
        # fused jax.hessian primitive, outweighing the forward-only saving.
        # Reverted to jax.hessian for that reason -- see
        # benchmarks/rebuttal/README.md section 6 for the full measurement,
        # and underPINN/pde/helmholtz.py for the same lesson on a simpler
        # single-output problem.
        H = jax.vmap(jax.hessian(u_fn))(x)
        u_xx, u_yy = H[:, 0, 0, 0], H[:, 0, 1, 1]
        v_xx, v_yy = H[:, 1, 0, 0], H[:, 1, 1, 1]

        # Physics equations (conservative form)
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