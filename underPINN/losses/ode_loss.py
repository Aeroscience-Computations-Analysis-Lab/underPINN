import jax.numpy as jnp
from .loss import l2_loss, l1_loss, weight_l2
from underPINN.core.base import BaseLoss


class ODELoss(BaseLoss):
    """
    Loss for first-order ODEs: pde residual + initial condition.

    Supports both first-order (ExponentialDecayODE) and second-order
    (HarmonicOscillatorODE) ODEs via the ic_derivative_weight argument.
    When ic_derivative_weight > 0 the solver expects u_ic_dot to be provided.
    """

    def __init__(
        self,
        model,
        pde,
        loss_type: str = "l2",
        ic_weight: float = 100.0,
        ic_derivative_weight: float = 0.0,
        reg_weight: float = 0.0,
    ):
        self.model = model
        self.pde = pde
        self.ic_weight = ic_weight
        self.ic_derivative_weight = ic_derivative_weight
        self.reg_weight = reg_weight

        if loss_type == "l2":
            self.norm = l2_loss
        elif loss_type == "l1":
            self.norm = l1_loss
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def __call__(self, params, t_r, t_ic, u_ic, u_ic_dot=None):
        pde_loss = self.norm(self.pde.residual(params, t_r))

        u_pred = self.pde.u(params, t_ic)
        ic_loss = self.norm(u_pred - u_ic)

        # Second-order ODE: also enforce u'(0)
        ic_dot_loss = 0.0
        if self.ic_derivative_weight > 0.0 and u_ic_dot is not None:
            ut_pred = self.pde.ut(params, t_ic)
            ic_dot_loss = self.norm(ut_pred - u_ic_dot)

        reg_loss = (
            self.reg_weight * weight_l2(params) if self.reg_weight > 0.0 else 0.0
        )

        total = (
            pde_loss
            + self.ic_weight * ic_loss
            + self.ic_derivative_weight * ic_dot_loss
            + reg_loss
        )
        return total, (pde_loss, ic_loss, ic_dot_loss, reg_loss)
