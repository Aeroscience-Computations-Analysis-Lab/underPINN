import jax
import jax.numpy as jnp

class BurgersPDE:
    def __init__(self, model):
        self.model = model

    def u(self, params, x, t):
        return self.model.apply(params, jnp.stack([x, t], axis=1))[:, 0]

    def residual(self, params, x, t):
        u = self.u(params, x, t)
        ux = jax.grad(lambda p,x,t: self.u(p,x,t).sum(),1)(params,x,t)
        ut = jax.grad(lambda p,x,t: self.u(p,x,t).sum(),2)(params,x,t)
        uxx = jax.grad(lambda p,x,t: ux.sum(),1)(params,x,t)
        return ut + u*ux - 0.01*uxx
