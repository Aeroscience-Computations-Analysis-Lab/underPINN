import time
import jax
import jax.numpy as jnp
import optax

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig


class ODESolver(BaseSolver):
    """PINN solver for ODEs (no boundary conditions, only IC).

    Accepts an optional :class:`~underPINN.core.config.TrainingConfig` to
    unify hyperparameters and attach callbacks (logging, early stopping …).

    Backward-compatible: legacy keyword arguments still work when ``config``
    is not provided.
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
        self.ic_dot_hist: list = []

        self._step = self._build_step()

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def init(self, key) -> None:
        self.params = self.model.init(key, jnp.ones((1, 1)))
        self.state = self.opt.init(self.params)

    def train(
        self,
        t_r: jnp.ndarray,
        t_ic: jnp.ndarray,
        u_ic: jnp.ndarray,
        u_ic_dot: jnp.ndarray = None,
        epochs: int = 3000,
        log_every: int = 200,
        config: TrainingConfig = None,
    ) -> None:
        """Run the ODE training loop.

        Parameters
        ----------
        t_r : collocation time points
        t_ic, u_ic : initial-condition time and value
        u_ic_dot : initial derivative (required for 2nd-order ODEs)
        epochs, log_every : legacy scalars; ignored when *config* is given
        config : :class:`TrainingConfig` — preferred production path
        """
        if config is not None:
            epochs = config.epochs
            log_every = config.log_every
            callbacks = list(config.callbacks)
            if config.lr_schedule is not None:
                self.opt = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()
                self.state = self.opt.init(self.params)
        else:
            callbacks = []

        if u_ic_dot is None:
            u_ic_dot = jnp.zeros_like(u_ic)

        start = time.time()
        try:
            for ep in range(epochs):
                self.params, self.state, loss, pde_l, ic_l, ic_dot_l = self._step(
                    self.params, self.state, t_r, t_ic, u_ic, u_ic_dot
                )
                self.loss_hist.append(float(loss))
                self.pde_hist.append(float(pde_l))
                self.ic_hist.append(float(ic_l))
                self.ic_dot_hist.append(float(ic_dot_l))

                logs = {
                    "loss": float(loss),
                    "pde": float(pde_l),
                    "ic": float(ic_l),
                    "ic_dot": float(ic_dot_l),
                }

                if not callbacks and ep % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"Epoch {ep:5d} | Loss {float(loss):.3e} | "
                        f"PDE {float(pde_l):.3e} | IC {float(ic_l):.3e} | "
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
        def step(params, state, t_r, t_ic, u_ic, u_ic_dot):
            def objective(p):
                return loss_fn(p, t_r, t_ic, u_ic, u_ic_dot)

            (loss, (pde_l, ic_l, ic_dot_l, reg_l)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)

            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, loss, pde_l, ic_l, ic_dot_l

        return step
