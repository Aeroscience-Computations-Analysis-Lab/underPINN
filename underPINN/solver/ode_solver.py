"""ODESolver — PINN solver for ordinary differential equations.

Supports the same ``n_scan_steps`` acceleration as :class:`FBPINNSolver`:
setting ``n_scan_steps > 1`` in :class:`TrainingConfig` fuses that many
gradient steps into a single ``jax.lax.scan`` XLA kernel, reducing Python
overhead on GPU.

ODEs typically use the full collocation set each step (no mini-batching), so
the scan body simply carries ``(params, opt_state)`` with ``xs=None``.
"""

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
            self._attach_checkpoint_callbacks(callbacks)
            n_scan = max(1, config.n_scan_steps)
            if config.lr_schedule is not None:
                self.opt = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()
                self.state = self.opt.init(self.params)
        else:
            callbacks = []
            n_scan = 1

        # ── Restart ──────────────────────────────────────────────────────────────
        _restart = None
        if (config is not None
                and getattr(config, "out_dir", "")
                and getattr(config, "save_restart_every", 0) > 0):
            from underPINN.utils.restart import RestartManager
            _restart = RestartManager(
                config.out_dir,
                save_every=config.save_restart_every,
                cfg=None,   # hash check done by 'resume' CLI; solver uses done-flag only
            )
            _ep_resume, self.params, self.state, _hists = \
                _restart.maybe_restore(self.params, self.state)
            if _ep_resume > 0:
                self.loss_hist.extend(_hists.get("loss_hist", []))
                self.pde_hist.extend(_hists.get("pde_hist",   []))
                self.ic_hist.extend( _hists.get("ic_hist",    []))
                self.ic_dot_hist.extend(_hists.get("ic_dot_hist", []))
                epochs = max(0, epochs - _ep_resume)

        if u_ic_dot is None:
            u_ic_dot = jnp.zeros_like(u_ic)

        t_r = jnp.asarray(t_r)
        t_ic = jnp.asarray(t_ic)
        u_ic = jnp.asarray(u_ic)
        u_ic_dot = jnp.asarray(u_ic_dot)

        start = time.time()

        # ------------------------------------------------------------------ #
        # SCAN MODE                                                            #
        # ------------------------------------------------------------------ #
        if n_scan > 1:
            scan_step = self._build_scan_step(t_r, t_ic, u_ic, u_ic_dot)
            n_outer = epochs // n_scan
            remainder = epochs % n_scan

            dummy_xs = jnp.zeros(n_scan)  # shape drives lax.scan length

            try:
                for outer in range(n_outer):
                    self.params, self.state, (losses, pde_ls, ic_ls, ic_dot_ls) = \
                        scan_step(self.params, self.state, dummy_xs)

                    self.loss_hist.extend(losses.tolist())
                    self.pde_hist.extend(pde_ls.tolist())
                    self.ic_hist.extend(ic_ls.tolist())
                    self.ic_dot_hist.extend(ic_dot_ls.tolist())

                    # ── Restart snapshot (scan mode) ──────────────────────────
                    if _restart is not None:
                        _restart.maybe_save(
                            len(self.loss_hist) - 1,
                            self.params, self.state,
                            {"loss_hist":   self.loss_hist,
                             "pde_hist":    self.pde_hist,
                             "ic_hist":     self.ic_hist,
                             "ic_dot_hist": self.ic_dot_hist},
                        )

                    ep = (outer + 1) * n_scan - 1
                    logs = {
                        "loss":   float(losses[-1]),
                        "pde":    float(pde_ls[-1]),
                        "ic":     float(ic_ls[-1]),
                        "ic_dot": float(ic_dot_ls[-1]),
                    }

                    if not callbacks and outer % max(1, n_outer // 10) == 0:
                        elapsed = time.time() - start
                        print(
                            f"Epoch {ep:5d}/{epochs} | "
                            f"Loss {logs['loss']:.3e} | "
                            f"PDE {logs['pde']:.3e} | "
                            f"IC {logs['ic']:.3e} | "
                            f"Time {elapsed:.2f}s"
                        )

                    for cb in callbacks:
                        cb.on_epoch_end(ep, logs)

            except StopIteration:
                pass

            epochs = remainder
            if remainder == 0:
                final_logs = {
                    "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
                    "pde":  self.pde_hist[-1] if self.pde_hist else float("nan"),
                    "ic":   self.ic_hist[-1] if self.ic_hist else float("nan"),
                }
                for cb in callbacks:
                    cb.on_train_end(final_logs)
                if not callbacks:
                    print(f"Training complete — final loss {final_logs['loss']:.3e}")
                if _restart is not None:
                    _restart.done()
                return

        # ------------------------------------------------------------------ #
        # PYTHON-LOOP MODE  (n_scan == 1, or scan tail)                       #
        # ------------------------------------------------------------------ #
        ep_offset = len(self.loss_hist)

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
                    "loss":   float(loss),
                    "pde":    float(pde_l),
                    "ic":     float(ic_l),
                    "ic_dot": float(ic_dot_l),
                }

                if not callbacks and (ep_offset + ep) % log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"Epoch {ep_offset + ep:5d} | Loss {float(loss):.3e} | "
                        f"PDE {float(pde_l):.3e} | IC {float(ic_l):.3e} | "
                        f"Time {elapsed:.2f}s"
                    )

                for cb in callbacks:
                    cb.on_epoch_end(ep_offset + ep, logs)

                if _restart is not None:
                    _restart.maybe_save(
                        ep_offset + ep, self.params, self.state,
                        {"loss_hist": self.loss_hist, "pde_hist": self.pde_hist,
                         "ic_hist": self.ic_hist, "ic_dot_hist": self.ic_dot_hist})

        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "pde":  self.pde_hist[-1] if self.pde_hist else float("nan"),
            "ic":   self.ic_hist[-1] if self.ic_hist else float("nan"),
        }
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Training complete — final loss {final_logs['loss']:.3e}")

        if _restart is not None:
            _restart.done()

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

    def _build_scan_step(self, t_r, t_ic, u_ic, u_ic_dot):
        """JIT-compiled function that runs N steps via lax.scan.

        The ODE uses the full dataset each step (no mini-batching).
        A dummy zero-array of shape ``(n_steps,)`` is passed as ``xs`` so
        ``lax.scan`` can infer the iteration count from its leading dimension
        (the concrete shape is visible at JIT-trace time).

        Returns a function::

            scan_step(params, state, dummy_xs: jnp.ndarray[n_steps])
              → (new_params, new_state, (losses, pde_ls, ic_ls, ic_dot_ls))

        Call as::

            scan_step(params, state, jnp.zeros(n_scan))
        """
        loss_fn = self.loss
        opt = self.opt

        # Capture data in closure — arrays are static across all scan steps
        t_r_ = t_r
        t_ic_ = t_ic
        u_ic_ = u_ic
        u_ic_dot_ = u_ic_dot

        def scan_body(carry, _unused):
            params, state = carry

            def objective(p):
                return loss_fn(p, t_r_, t_ic_, u_ic_, u_ic_dot_)

            (loss, (pde_l, ic_l, ic_dot_l, _reg)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)

            updates, new_state = opt.update(grads, state)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_state), (loss, pde_l, ic_l, ic_dot_l)

        @jax.jit
        def scan_step(params, state, dummy_xs):
            # dummy_xs shape: (n_steps,) — values unused; shape drives iteration count
            (new_params, new_state), aux = jax.lax.scan(
                scan_body, (params, state), dummy_xs
            )
            return new_params, new_state, aux

        return scan_step
