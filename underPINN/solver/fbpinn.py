import time
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad


class FBPINNSolver:
    def __init__(self, model, pde, loss, lr=1e-3):
        self.model = model
        self.pde = pde
        self.loss = loss
        self.opt = optax.adam(lr)

        self.loss_hist = []
        self.pde_hist = []
        self.ic_hist = []
        self.bc_hist = []
        self.reg_hist = []

        self._step = self._build_step()

    def init(self, key):
        self.params = self.model.init(key, jnp.ones((1, 2)))
        self.state = self.opt.init(self.params)

    def _build_step(self):
        loss_fn = self.loss
        opt = self.opt

        @jax.jit
        def step(params, state, x_r, t_r, x_i, u_i, x_b, t_b, u_b):
            def objective(p):
                return loss_fn(p, x_r, t_r, x_i, u_i, x_b, t_b, u_b)

            (loss, (pde_l, ic_l, bc_l, reg_l)), grads = jax.value_and_grad(
                objective,
                has_aux=True
            )(params)

            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)

            return params, state, loss, pde_l, ic_l, bc_l, reg_l

        return step

    def train(
        self,
        x_r,
        t_r,
        x_i,
        u_i,
        x_b,
        t_b,
        u_b,
        epochs=1000,
        batch_r=4096,
        batch_i=512,
        batch_b=512,
        seed=0,
    ):
        key = jax.random.PRNGKey(seed)
        start = time.time()

        for ep in range(epochs):
            key, k1, k2, k3 = jax.random.split(key, 4)

            idx_r = jax.random.choice(k1, x_r.shape[0], (batch_r,), replace=False)
            idx_i = jax.random.choice(k2, x_i.shape[0], (batch_i,), replace=False)
            idx_b = jax.random.choice(k3, x_b.shape[0], (batch_b,), replace=False
        )

            self.params, self.state, loss, pde_l, ic_l, bc_l, reg_l = self._step(
                self.params,
                self.state,
                x_r[idx_r],
                t_r[idx_r],
                x_i[idx_i],
                u_i[idx_i],
                x_b[idx_b],
                t_b[idx_b],
                u_b[idx_b],
            )

            self.loss_hist.append(float(loss))
            self.pde_hist.append(float(pde_l))
            self.ic_hist.append(float(ic_l))
            self.bc_hist.append(float(bc_l))
            self.reg_hist.append(float(reg_l))

            if ep % 10 == 0:
                elapsed = time.time() - start
                print(
                    f"Epoch {ep:5d} | "
                    f"Loss {float(loss):.3e} | "
                    f"PDE {float(pde_l):.3e} | "
                    f"IC {float(ic_l):.3e} | "
                    f"BC {float(bc_l):.3e} | "
                    f"Time {elapsed:.2f}s"
                )

        print("Training complete")
