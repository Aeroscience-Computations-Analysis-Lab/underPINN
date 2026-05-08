import time
import jax
import jax.numpy as jnp
import optax

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig


class FBPINNSolver(BaseSolver):
    """Training-loop orchestrator for FBPINN / space-time PDE problems.

    Accepts an optional :class:`~underPINN.core.config.TrainingConfig` to
    unify hyperparameters and attach callbacks (logging, early stopping …).

    Backward-compatible: all legacy keyword arguments still work when
    ``config`` is not provided.
    """

    def __init__(self, model, pde, loss, lr: float = 1e-3, lr_schedule=None):
        self.model = model
        self.pde = pde
        self.loss = loss
        self._lr = lr
        self._lr_schedule = lr_schedule
        self.opt = self._make_opt(lr, lr_schedule)

        self.loss_hist: list = []
        self.pde_hist: list = []
        self.ic_hist: list = []
        self.bc_hist: list = []
        self.reg_hist: list = []

        self._step = self._build_step()

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def init(self, key, input_shape: tuple = (1, 2)) -> None:
        self.params = self.model.init(key, jnp.ones(input_shape))
        self.state = self.opt.init(self.params)

    def load_params(self, params) -> None:
        """Load pre-trained parameters for transfer learning.

        Replaces current parameters, resets the optimiser momentum
        buffer, and clears all loss histories so only the fine-tuning
        phase is recorded.  To fine-tune with a lower learning rate,
        pass a reduced ``lr`` / ``lr_schedule`` in the
        :class:`TrainingConfig` you hand to :meth:`train`.
        """
        self.params = params
        self.state  = self.opt.init(params)
        for h in (self.loss_hist, self.pde_hist, self.ic_hist,
                  self.bc_hist, self.reg_hist):
            h.clear()

    def train(
        self,
        x_r,
        t_r,
        x_i,
        u_i,
        x_b,
        t_b,
        u_b,
        epochs: int = 1000,
        batch_r: int = 4096,
        batch_i: int = 512,
        batch_b: int = 512,
        seed: int = 0,
        config: TrainingConfig = None,
    ) -> None:
        """Run the PDE training loop.

        Parameters
        ----------
        x_r, t_r : collocation points
        x_i, u_i : initial-condition points and values
        x_b, t_b, u_b : boundary-condition points, times and values
        epochs, batch_r, batch_i, batch_b, seed : legacy scalars
        config : :class:`TrainingConfig` — preferred production path
        """
        if config is not None:
            epochs = config.epochs
            batch_r = config.batch_r
            batch_i = config.batch_i
            batch_b = config.batch_b
            seed = config.seed
            callbacks = list(config.callbacks)
            if config.lr_schedule is not None:
                self.opt = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()
                self.state = self.opt.init(self.params)
        else:
            callbacks = []

        key = jax.random.PRNGKey(seed)
        start = time.time()

        try:
            for ep in range(epochs):
                key, k1, k2, k3 = jax.random.split(key, 4)

                idx_r = jax.random.choice(k1, x_r.shape[0], (batch_r,), replace=False)
                idx_i = jax.random.choice(k2, x_i.shape[0], (batch_i,), replace=False)
                idx_b = jax.random.choice(k3, x_b.shape[0], (batch_b,), replace=False)

                self.params, self.state, loss, pde_l, ic_l, bc_l, reg_l = self._step(
                    self.params,
                    self.state,
                    x_r[idx_r], t_r[idx_r],
                    x_i[idx_i], u_i[idx_i],
                    x_b[idx_b], t_b[idx_b], u_b[idx_b],
                )

                self.loss_hist.append(float(loss))
                self.pde_hist.append(float(pde_l))
                self.ic_hist.append(float(ic_l))
                self.bc_hist.append(float(bc_l))
                self.reg_hist.append(float(reg_l))

                logs = {
                    "loss": float(loss),
                    "pde": float(pde_l),
                    "ic": float(ic_l),
                    "bc": float(bc_l),
                }

                if not callbacks and ep % 10 == 0:
                    elapsed = time.time() - start
                    print(
                        f"Epoch {ep:5d} | "
                        f"Loss {float(loss):.3e} | "
                        f"PDE {float(pde_l):.3e} | "
                        f"IC {float(ic_l):.3e} | "
                        f"BC {float(bc_l):.3e} | "
                        f"Time {elapsed:.2f}s"
                    )

                for cb in callbacks:
                    cb.on_epoch_end(ep, logs)

        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "pde": self.pde_hist[-1] if self.pde_hist else float("nan"),
            "ic": self.ic_hist[-1] if self.ic_hist else float("nan"),
            "bc": self.bc_hist[-1] if self.bc_hist else float("nan"),
        }
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Training complete — final loss {final_logs['loss']:.3e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_opt(lr, lr_schedule):
        if lr_schedule is not None:
            return optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1.0),
            )
        return optax.adam(lr)

    def _build_step(self):
        loss_fn = self.loss
        opt = self.opt

        @jax.jit
        def step(params, state, x_r, t_r, x_i, u_i, x_b, t_b, u_b):
            def objective(p):
                return loss_fn(p, x_r, t_r, x_i, u_i, x_b, t_b, u_b)

            (loss, (pde_l, ic_l, bc_l, reg_l)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)

            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, loss, pde_l, ic_l, bc_l, reg_l

        return step
