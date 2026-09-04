import jax
import jax.numpy as jnp
from underPINN.core.base import BasePDE


class ExponentialDecayODE(BasePDE):
    """
    du/dt + lam * u = 0
    Exact solution: u(t) = u0 * exp(-lam * t)
    """

    def __init__(self, model, lam: float = 2.0):
        self.model = model
        self.lam = lam

    def u(self, params, t):
        return self.model.apply(params, t[:, None])[:, 0]

    def residual(self, params, t):
        u = self.u(params, t)
        ut = jax.grad(lambda p, t: self.u(p, t).sum(), 1)(params, t)
        return ut + self.lam * u

    def exact(self, t, u0: float = 1.0):
        return u0 * jnp.exp(-self.lam * t)


class HarmonicOscillatorODE(BasePDE):
    """
    d²u/dt² + omega² * u = 0
    Exact solution: u(t) = A*cos(omega*t) + B*sin(omega*t)
    Default IC: u(0)=1, u'(0)=0  =>  A=1, B=0
    """

    def __init__(self, model, omega: float = 2.0):
        self.model = model
        self.omega = omega

    def u(self, params, t):
        return self.model.apply(params, t[:, None])[:, 0]

    def ut(self, params, t):
        return jax.grad(lambda p, t: self.u(p, t).sum(), 1)(params, t)

    def residual(self, params, t):
        def _u(p, t):
            return self.u(p, t).sum()

        def _ut(p, t):
            return jax.grad(_u, 1)(p, t).sum()

        utt = jax.grad(_ut, 1)(params, t)
        u = self.u(params, t)
        return utt + self.omega ** 2 * u

    def exact(self, t, A: float = 1.0, B: float = 0.0):
        return A * jnp.cos(self.omega * t) + B * jnp.sin(self.omega * t)
