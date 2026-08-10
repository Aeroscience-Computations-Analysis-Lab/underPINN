import jax
import jax.numpy as jnp
from .loss import l2_loss, l1_loss, weight_l2
from underPINN.core.base import BaseLoss


class SteadyLoss(BaseLoss):
    """Loss for time-independent PDEs (e.g. steady heat / Poisson).

    Enforces only the PDE residual on interior collocation points and
    Dirichlet boundary conditions — no initial condition is needed.

    Components
    ----------
    total = pde_loss + bc_weight * bc_loss + reg_weight * reg_loss

    Returns (total, (pde_loss, bc_loss, reg_loss)).
    """

    def __init__(
        self,
        model,
        pde,
        loss_type: str = "l2",
        bc_weight: float = 10.0,
        reg_weight: float = 0.0,
        rba: bool = False,
        rba_eps: float = 1e-6,
    ):
        self.model = model
        self.pde = pde
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

    def __call__(
        self,
        params,
        xy_r: jnp.ndarray,   # (N_r, 2) interior collocation points
        xy_b: jnp.ndarray,   # (N_b, 2) boundary points
        u_b: jnp.ndarray,    # (N_b,)   boundary values
    ):
        # ---- PDE residual ----
        res = self.pde.residual(params, xy_r)  # (N_r,)

        if self.rba:
            w = jax.lax.stop_gradient(
                jnp.abs(res) / (jnp.mean(jnp.abs(res)) + self.rba_eps)
            )
            pde_loss = jnp.mean(w * res ** 2)
        else:
            pde_loss = self.norm(res)

        # ---- Dirichlet BC ----
        u_b_pred = self.pde.u(params, xy_b)   # (N_b,)
        bc_loss = self.norm(u_b_pred - u_b)

        # ---- Regularization ----
        reg_loss = (
            self.reg_weight * weight_l2(params) if self.reg_weight > 0.0 else 0.0
        )

        total = pde_loss + self.bc_weight * bc_loss + reg_loss
        return total, (pde_loss, bc_loss, reg_loss)
