import jax.numpy as jnp
import jax
from underPINN.core.base import BaseLoss


def l2_loss(x):
    return jnp.mean(x ** 2)


def l1_loss(x):
    return jnp.mean(jnp.abs(x))


def weight_l2(params):
    """
    L2 regularization over all parameters.
    """
    leaves = jnp.concatenate([
        p.reshape(-1) for p in jax.tree_util.tree_leaves(params)
    ])
    return jnp.mean(leaves ** 2)

class PINNLoss(BaseLoss):
    def __init__(
        self,
        model,
        pde,
        loss_type="l2",
        ic_weight=10.0,
        bc_weight=1.0,          
        reg_weight=0.0,
        rba=False,
        rba_eps=1e-6,
    ):
        self.model = model
        self.pde = pde
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight   
        self.reg_weight = reg_weight
        self.rba = rba
        self.rba_eps = rba_eps


        if loss_type == "l2":
            self.norm = l2_loss
        elif loss_type == "l1":
            self.norm = l1_loss
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def __call__(self, params, x_r, t_r, x_i, u_i, x_b=None, t_b=None, u_b=None):
        # ---------- PDE residual ----------
        res = self.pde.residual(params, x_r, t_r) # NOTE: MULTI OUTPUT SUPPORT NEEDED

        if self.rba:
            # Residual-based adaptivity: weight each point by its relative
            # residual magnitude (detached so it doesn't affect the gradient
            # direction, only the step size per collocation point).
            # Bug fix: must weight individual squared residuals (element-wise),
            # not multiply the already-reduced scalar norm — that made RBA a no-op.
            w = jax.lax.stop_gradient(
                jnp.abs(res) / (jnp.mean(jnp.abs(res)) + self.rba_eps)
            )
            pde_loss = jnp.mean(w * res ** 2)
        else:
            pde_loss = self.norm(res)

        # ---------- Initial condition ----------
        u_pred = self.model.apply(
            params, jnp.stack([x_i, jnp.zeros_like(x_i)], axis=1)
        )[:, 0]
        ic_loss = self.norm(u_pred - u_i)

        # ---------- Boundary condition ----------
        if x_b is not None:
            u_b_pred = self.model.apply(
                params, jnp.stack([x_b, t_b], axis=1)
            )[:, 0]
            bc_loss = self.norm(u_b_pred - u_b)
        else:
            bc_loss = 0.0

        # ---------- Regularization ----------
        reg_loss = (
            self.reg_weight * weight_l2(params)
            if self.reg_weight > 0.0
            else 0.0
        )

        total_loss = (
            pde_loss
            + self.ic_weight * ic_loss
            + self.bc_weight * bc_loss
            + reg_loss
        )

        return total_loss, (pde_loss, ic_loss, bc_loss, reg_loss)
