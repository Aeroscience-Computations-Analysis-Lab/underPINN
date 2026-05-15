import time
import jax
import jax.numpy as jnp
import optax

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig
from underPINN.utils.sampling import safe_choice


class SteadySolver(BaseSolver):
    """PINN solver for time-independent 2-D PDEs (e.g. steady heat / Poisson).

    Training data:
        xy_r  — interior collocation points  (N_r, 2)
        xy_b  — boundary points              (N_b, 2)
        u_b   — Dirichlet boundary values    (N_b,)

    No initial condition is enforced (steady problem).
    Supports :class:`~underPINN.core.config.TrainingConfig` + callbacks.
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
        self.bc_hist: list = []

        self._step = self._build_step()

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def init(self, key) -> None:
        self.params = self.model.init(key, jnp.ones((1, 2)))
        self.state = self.opt.init(self.params)

    def train(
        self,
        xy_r: jnp.ndarray,
        xy_b: jnp.ndarray,
        u_b: jnp.ndarray,
        epochs: int = 5000,
        batch_r: int = 2048,
        batch_b: int = 256,
        seed: int = 0,
        log_every: int = 500,
        config: TrainingConfig = None,
    ) -> None:
        """Run the steady-state training loop.

        Parameters
        ----------
        xy_r : (N_r, 2) interior collocation points
        xy_b : (N_b, 2) boundary points
        u_b  : (N_b,)   Dirichlet boundary values
        config : :class:`TrainingConfig` — preferred production path
        """
        if config is not None:
            epochs = config.epochs
            batch_r = config.batch_r
            batch_b = config.batch_b
            seed = config.seed
            log_every = config.log_every
            callbacks = list(config.callbacks)
            self._attach_checkpoint_callbacks(callbacks)
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
                key, k1, k2 = jax.random.split(key, 3)

                idx_r = safe_choice(k1, xy_r.shape[0], batch_r)
                idx_b = safe_choice(k2, xy_b.shape[0], batch_b)

                self.params, self.state, loss, pde_l, bc_l = self._step(
                    self.params, self.state,
                    xy_r[idx_r], xy_b[idx_b], u_b[idx_b],
                )

                self.loss_hist.append(float(loss))
                self.pde_hist.append(float(pde_l))
                self.bc_hist.append(float(bc_l))

                logs = {
                    "loss": float(loss),
                    "pde": float(pde_l),
                    "bc": float(bc_l),
                }

                if not callbacks and ep % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"Epoch {ep:5d} | Loss {float(loss):.3e} | "
                        f"PDE {float(pde_l):.3e} | BC {float(bc_l):.3e} | "
                        f"Time {elapsed:.2f}s"
                    )

                for cb in callbacks:
                    cb.on_epoch_end(ep, logs)

        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "pde": self.pde_hist[-1] if self.pde_hist else float("nan"),
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
        def step(params, state, xy_r, xy_b, u_b):
            def objective(p):
                return loss_fn(p, xy_r, xy_b, u_b)

            (loss, (pde_l, bc_l, reg_l)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)

            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, loss, pde_l, bc_l

        return step
