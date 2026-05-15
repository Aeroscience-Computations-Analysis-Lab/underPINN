"""FBPINNSolver — space-time PINN solver with optional lax.scan acceleration.

Two training modes are supported:

*Python-loop mode* (``n_scan_steps=1``, default)
    One Python call to the JIT-compiled step function per epoch.  Callbacks
    fire every epoch.  Identical to the original behaviour.

*Scan mode* (``n_scan_steps > 1``)
    ``n_scan_steps`` gradient updates are fused into a single XLA kernel via
    ``jax.lax.scan``.  The Python interpreter is invoked only every
    ``n_scan_steps`` epochs — typically 50–500× less Python overhead on GPU.
    Callbacks fire once per outer step (= every ``n_scan_steps`` epochs).
    Early-stopping granularity is similarly coarser.

RAR-D adaptive resampling
--------------------------
When ``resample_period > 0`` in :class:`~underPINN.core.config.TrainingConfig`,
collocation points are replaced at that interval (measured in outer steps) by
new points sampled proportionally to ``|pde_residual|^k`` — focusing training
resources on high-error regions (Lu et al., 2021).
"""

import time
import jax
import jax.numpy as jnp
import optax

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig
from underPINN.training.resample import rar_d_resample
from underPINN.utils.sampling import safe_choice


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
        self.state = self.opt.init(params)
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
        # ------------------------------------------------------------------ #
        # Unpack config (or fall back to legacy kwargs)                        #
        # ------------------------------------------------------------------ #
        if config is not None:
            epochs = config.epochs
            batch_r = config.batch_r
            batch_i = config.batch_i
            batch_b = config.batch_b
            seed = config.seed
            callbacks = list(config.callbacks)
            self._attach_checkpoint_callbacks(callbacks)
            n_scan = max(1, config.n_scan_steps)
            resample_period = config.resample_period
            resample_candidates = config.resample_candidates
            resample_k = config.resample_k
            candidate_sampler = config.candidate_sampler
            if config.lr_schedule is not None:
                self.opt = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()  # rebuild with new opt
                self.state = self.opt.init(self.params)
        else:
            callbacks = []
            n_scan = 1
            resample_period = 0
            resample_candidates = 0
            resample_k = 1.0
            candidate_sampler = None

        # Convert to JAX arrays once (safe no-op if already JAX)
        x_r = jnp.asarray(x_r)
        t_r = jnp.asarray(t_r)
        x_i = jnp.asarray(x_i)
        u_i = jnp.asarray(u_i)
        x_b = jnp.asarray(x_b)
        t_b = jnp.asarray(t_b)
        u_b = jnp.asarray(u_b)

        N_r = x_r.shape[0]
        N_i = x_i.shape[0]
        N_b = x_b.shape[0]

        key = jax.random.PRNGKey(seed)
        start = time.time()

        # ------------------------------------------------------------------ #
        # SCAN MODE  (n_scan_steps > 1)                                       #
        # ------------------------------------------------------------------ #
        if n_scan > 1:
            scan_step = self._build_scan_step()
            n_outer = epochs // n_scan
            remainder = epochs % n_scan  # handled by fall-through to Python loop

            try:
                for outer in range(n_outer):
                    key, k1, k2, k3 = jax.random.split(key, 4)

                    # --- generate n_scan random batches in one vmap ---
                    # Using randint (with replacement) — faster than choice for
                    # large n_scan; statistically indistinguishable for typical
                    # dataset / batch size ratios.
                    scan_keys_r = jax.random.split(k1, n_scan)
                    scan_keys_i = jax.random.split(k2, n_scan)
                    scan_keys_b = jax.random.split(k3, n_scan)

                    idx_r = jax.vmap(
                        lambda k: jax.random.randint(k, (batch_r,), 0, N_r)
                    )(scan_keys_r)                             # (n_scan, batch_r)
                    idx_i = jax.vmap(
                        lambda k: jax.random.randint(k, (batch_i,), 0, N_i)
                    )(scan_keys_i)
                    idx_b = jax.vmap(
                        lambda k: jax.random.randint(k, (batch_b,), 0, N_b)
                    )(scan_keys_b)

                    batches = (
                        x_r[idx_r], t_r[idx_r],         # (n_scan, batch_r, …)
                        x_i[idx_i], u_i[idx_i],
                        x_b[idx_b], t_b[idx_b], u_b[idx_b],
                    )

                    self.params, self.state, (losses, pde_ls, ic_ls, bc_ls, reg_ls) = \
                        scan_step(self.params, self.state, batches)

                    # Extend per-epoch histories
                    self.loss_hist.extend(losses.tolist())
                    self.pde_hist.extend(pde_ls.tolist())
                    self.ic_hist.extend(ic_ls.tolist())
                    self.bc_hist.extend(bc_ls.tolist())
                    self.reg_hist.extend(reg_ls.tolist())

                    # Callbacks + logging fire at outer-step granularity
                    ep = (outer + 1) * n_scan - 1
                    logs = {
                        "loss": float(losses[-1]),
                        "pde":  float(pde_ls[-1]),
                        "ic":   float(ic_ls[-1]),
                        "bc":   float(bc_ls[-1]),
                    }

                    if not callbacks and outer % max(1, n_outer // 10) == 0:
                        elapsed = time.time() - start
                        print(
                            f"Epoch {ep:5d}/{epochs} | "
                            f"Loss {logs['loss']:.3e} | "
                            f"PDE {logs['pde']:.3e} | "
                            f"IC {logs['ic']:.3e} | "
                            f"BC {logs['bc']:.3e} | "
                            f"Time {elapsed:.2f}s"
                        )

                    for cb in callbacks:
                        cb.on_epoch_end(ep, logs)

                    # --- RAR-D adaptive resampling ---
                    if resample_period > 0 and (outer + 1) % resample_period == 0:
                        key, rkey = jax.random.split(key)
                        n_cands = (resample_candidates if resample_candidates > 0
                                   else 5 * N_r)
                        x_r, t_r = rar_d_resample(
                            self.pde, self.params, x_r, t_r,
                            k=resample_k,
                            n_candidates=n_cands,
                            candidate_sampler=candidate_sampler,
                            key=rkey,
                        )
                        # N_r stays the same (rar_d_resample preserves shape)

            except StopIteration:
                pass

            # Run any leftover epochs with the Python loop below
            epochs = remainder  # re-use the Python loop for the tail
            # If no remainder, skip to final_logs
            if remainder == 0:
                final_logs = {
                    "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
                    "pde":  self.pde_hist[-1] if self.pde_hist else float("nan"),
                    "ic":   self.ic_hist[-1] if self.ic_hist else float("nan"),
                    "bc":   self.bc_hist[-1] if self.bc_hist else float("nan"),
                }
                for cb in callbacks:
                    cb.on_train_end(final_logs)
                if not callbacks:
                    print(f"Training complete — final loss {final_logs['loss']:.3e}")
                return

        # ------------------------------------------------------------------ #
        # PYTHON-LOOP MODE  (n_scan == 1, or scan tail)                       #
        # ------------------------------------------------------------------ #
        ep_offset = len(self.loss_hist)  # so epoch numbers stay contiguous

        try:
            for ep in range(epochs):
                key, k1, k2, k3 = jax.random.split(key, 4)

                idx_r = safe_choice(k1, N_r, batch_r)
                idx_i = safe_choice(k2, N_i, batch_i)
                idx_b = safe_choice(k3, N_b, batch_b)

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
                    "pde":  float(pde_l),
                    "ic":   float(ic_l),
                    "bc":   float(bc_l),
                }

                if not callbacks and (ep_offset + ep) % 10 == 0:
                    elapsed = time.time() - start
                    print(
                        f"Epoch {ep_offset + ep:5d} | "
                        f"Loss {float(loss):.3e} | "
                        f"PDE {float(pde_l):.3e} | "
                        f"IC {float(ic_l):.3e} | "
                        f"BC {float(bc_l):.3e} | "
                        f"Time {elapsed:.2f}s"
                    )

                for cb in callbacks:
                    cb.on_epoch_end(ep_offset + ep, logs)

                # RAR-D at per-epoch granularity in Python-loop mode
                if (resample_period > 0
                        and (ep_offset + ep + 1) % resample_period == 0):
                    key, rkey = jax.random.split(key)
                    n_cands = (resample_candidates if resample_candidates > 0
                               else 5 * N_r)
                    x_r, t_r = rar_d_resample(
                        self.pde, self.params, x_r, t_r,
                        k=resample_k,
                        n_candidates=n_cands,
                        candidate_sampler=candidate_sampler,
                        key=rkey,
                    )

        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "pde":  self.pde_hist[-1] if self.pde_hist else float("nan"),
            "ic":   self.ic_hist[-1] if self.ic_hist else float("nan"),
            "bc":   self.bc_hist[-1] if self.bc_hist else float("nan"),
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
        """Single JIT-compiled gradient step (Python-loop mode)."""
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

    def _build_scan_step(self):
        """JIT-compiled function that runs N gradient steps via lax.scan.

        The scan body is *not* individually JIT-compiled; the outer
        ``@jax.jit`` compiles the entire scan including its unrolled body.

        Returns a function with signature::

            scan_step(params, state, batches)
              → (new_params, new_state, (losses, pde_ls, ic_ls, bc_ls, reg_ls))

        where each of ``losses`` etc. has shape ``(n_scan_steps,)``.
        """
        loss_fn = self.loss
        opt = self.opt

        def scan_body(carry, batch):
            params, state = carry
            x_r, t_r, x_i, u_i, x_b, t_b, u_b = batch

            def objective(p):
                return loss_fn(p, x_r, t_r, x_i, u_i, x_b, t_b, u_b)

            (loss, (pde_l, ic_l, bc_l, reg_l)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)

            updates, new_state = opt.update(grads, state)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_state), (loss, pde_l, ic_l, bc_l, reg_l)

        @jax.jit
        def scan_step(params, state, batches):
            (new_params, new_state), aux = jax.lax.scan(
                scan_body, (params, state), batches
            )
            return new_params, new_state, aux

        return scan_step
