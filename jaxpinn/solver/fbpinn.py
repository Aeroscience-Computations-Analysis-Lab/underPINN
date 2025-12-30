import time
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad


class FBPINNSolver:
    def __init__(self, model, pde, lr=1e-3):
        self.model = model
        self.pde = pde
        self.opt = optax.adam(lr)

        self.loss_hist = []
        self.pde_hist = []
        self.ic_hist = []

        # build the jitted step ONCE
        self._step = self._build_step()

    def init(self, key):
        self.params = self.model.init(key, jnp.ones((1, 2)))
        self.state = self.opt.init(self.params)

    def _build_step(self):
        model = self.model
        pde = self.pde
        opt = self.opt

        @jit
        def step(params, state, x_r, t_r, x_i, u_i):
            def loss_fn(p):
                res = pde.residual(p, x_r, t_r)
                pde_loss = jnp.mean(res ** 2)

                u_pred = model.apply(
                    p, jnp.stack([x_i, jnp.zeros_like(x_i)], axis=1)
                )[:, 0]
                ic_loss = jnp.mean((u_pred - u_i) ** 2)

                return pde_loss + 10.0 * ic_loss, (pde_loss, ic_loss)

            (loss, (pde_l, ic_l)), grads = value_and_grad(
                loss_fn, has_aux=True
            )(params)

            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)

            return params, state, loss, pde_l, ic_l

        return step

    def train(
        self,
        x_r,
        t_r,
        x_i,
        u_i,
        epochs=1000,
        batch_r=4096,
        batch_i=512,
        seed=0,
    ):
        key = jax.random.PRNGKey(seed)
        start = time.time()

        for ep in range(epochs):
            key, k1, k2 = jax.random.split(key, 3)

            idx_r = jax.random.choice(k1, x_r.shape[0], (batch_r,), replace=False)
            idx_i = jax.random.choice(k2, x_i.shape[0], (batch_i,), replace=False)

            self.params, self.state, loss, pde_l, ic_l = self._step(
                self.params,
                self.state,
                x_r[idx_r],
                t_r[idx_r],
                x_i[idx_i],
                u_i[idx_i],
            )

            # Logging History
            self.loss_hist.append(float(loss))
            self.pde_hist.append(float(pde_l))
            self.ic_hist.append(float(ic_l))

            if ep % 10 == 0:
                elapsed = time.time() - start
                print(
                    f"Epoch {ep:5d} | "
                    f"Loss {float(loss):.3e} | "
                    f"PDE {float(pde_l):.3e} | "
                    f"IC {float(ic_l):.3e} | "
                    f"Time {elapsed:.2f}s"
                )

        print("Training complete")
