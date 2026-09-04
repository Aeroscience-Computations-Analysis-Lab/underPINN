"""Training-loop orchestrators for neural operators (FNO / CViT / DeepONet).

``OperatorSolver`` is the generic solver for any grid-based operator trained
with :class:`underPINN.losses.operator_loss.OperatorLoss` — FNO1D, FNO2D, and
CViT (via :func:`underPINN.nn.operators.cvit_grid_predict`) all fit the same
"batch of (input window, target frame) pairs, fixed dataset held device-
resident" training loop, so one solver class covers all of them.

Supports the same ``n_scan_steps`` :func:`jax.lax.scan` acceleration as
:class:`underPINN.solver.fbpinn.FBPINNSolver` / :class:`underPINN.solver.ode_solver.ODESolver`,
plus an optional PDE-weight warmup ramp (``pde_warmup_epochs``): the physics
term is scaled up linearly from 0 over the first few epochs so the optimizer
fits the data term before the (initially very wrong) residual starts pulling
on the weights.

``DeepONetSolver`` is separate because DeepONet training samples three
distinct point sets each step (IC, BC, residual) rather than batching a
single fixed pool of (input, target) pairs.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig
from underPINN.utils.sampling import safe_choice
from underPINN.utils.timing import fmt_train_time


class OperatorSolver(BaseSolver):
    """Generic training-loop orchestrator for grid-based neural operators.

    Parameters
    ----------
    model              : the Flax module (only used by :meth:`init`; the
                         actual forward pass is whatever ``loss.predict_fn``
                         wraps — for CViT that's :func:`underPINN.nn.operators.cvit_grid_predict`).
    loss               : an :class:`underPINN.losses.operator_loss.OperatorLoss`.
    pde_weight         : target PDE-term weight (the λ in data + λ·physics).
    pde_warmup_epochs  : ramp λ from 0 to ``pde_weight`` over this many
                         epochs; 0 disables warmup (constant weight).
    """

    def __init__(self, model, loss, lr: float = 1e-3, lr_schedule=None,
                pde_weight: float = 1.0, pde_warmup_epochs: int = 0):
        self.model = model
        self.loss = loss
        self._lr = lr
        self._lr_schedule = lr_schedule
        self.opt = self._make_opt(lr, lr_schedule)
        self.pde_weight = pde_weight
        self.pde_warmup_epochs = pde_warmup_epochs

        self.loss_hist: list = []
        self.data_hist: list = []
        self.pde_hist: list = []

        self._step = self._build_step()

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def init(self, key, *init_args) -> None:
        """``*init_args`` are forwarded to ``model.init`` verbatim — a single
        dummy input array for FNO1D/FNO2D, or ``(x_seq, coords)`` for CViT."""
        self.params = self.model.init(key, *init_args)
        self.state = self.opt.init(self.params)

    def _warmup_weight(self, ep: int) -> float:
        if self.pde_warmup_epochs <= 0:
            return self.pde_weight
        return self.pde_weight * min(1.0, (ep + 1) / self.pde_warmup_epochs)

    def _warmup_weight_seq(self, start_ep: int, n: int) -> jnp.ndarray:
        eps = jnp.arange(start_ep, start_ep + n)
        if self.pde_warmup_epochs <= 0:
            return jnp.full((n,), self.pde_weight)
        return self.pde_weight * jnp.minimum(1.0, (eps + 1) / self.pde_warmup_epochs)

    def train(
        self,
        x_input,
        u_target,
        model_input=None,
        epochs: int = 1000,
        batch_size: int = 32,
        seed: int = 0,
        config: TrainingConfig = None,
    ) -> None:
        """Train on a fixed, device-resident pool of (input, target) pairs.

        Parameters
        ----------
        x_input    : ``(N, ...)`` PDE-facing input windows (supplies
                     ``u_prev``/``nu`` to the grid residual).
        u_target   : ``(N, ...)`` ground-truth next frames.
        model_input : ``(N, ...)`` model-facing input, if it differs from
                     ``x_input`` (CViT's ``(T, Nx, Ny, C)`` window). Defaults
                     to ``x_input``.
        config     : :class:`TrainingConfig` — preferred production path.
        """
        if config is not None:
            epochs = config.epochs
            batch_size = config.batch_r
            seed = config.seed
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

        x_input = jnp.asarray(x_input)
        u_target = jnp.asarray(u_target)
        model_input = x_input if model_input is None else jnp.asarray(model_input)
        N = x_input.shape[0]

        _restart = None
        _ep_resume = 0
        if (config is not None and getattr(config, "out_dir", "")
                and getattr(config, "save_restart_every", 0) > 0):
            from underPINN.utils.restart import RestartManager
            _restart = RestartManager(config.out_dir,
                                      save_every=config.save_restart_every, cfg=None)
            _ep_resume, self.params, self.state, hists = \
                _restart.maybe_restore(self.params, self.state)
            if _ep_resume > 0:
                for attr, hk in (("loss_hist", "loss_hist"),
                                ("data_hist", "data_hist"),
                                ("pde_hist", "pde_hist")):
                    saved = hists.get(hk, [])
                    if saved:
                        getattr(self, attr).extend(saved)

        key = jax.random.PRNGKey(seed)
        start = time.time()

        # ------------------------------------------------------------------
        # SCAN MODE  (n_scan_steps > 1)
        # ------------------------------------------------------------------
        if n_scan > 1:
            scan_step = self._build_scan_step()
            n_outer = epochs // n_scan
            remainder = epochs % n_scan

            try:
                for outer in range(n_outer):
                    key, k = jax.random.split(key)
                    scan_keys = jax.random.split(k, n_scan)
                    idx = jax.vmap(
                        lambda kk: jax.random.randint(kk, (batch_size,), 0, N)
                    )(scan_keys)                                     # (n_scan, batch)

                    w_seq = self._warmup_weight_seq(outer * n_scan, n_scan)
                    batches = (x_input[idx], model_input[idx], u_target[idx], w_seq)

                    self.params, self.state, (losses, dls, pls) = scan_step(
                        self.params, self.state, batches)

                    self.loss_hist.extend(losses.tolist())
                    self.data_hist.extend(dls.tolist())
                    self.pde_hist.extend(pls.tolist())

                    ep = (outer + 1) * n_scan - 1
                    logs = {"loss": float(losses[-1]), "data": float(dls[-1]),
                           "pde": float(pls[-1])}

                    if not callbacks and outer % max(1, n_outer // 10) == 0:
                        elapsed = time.time() - start
                        print(f"Epoch {ep:5d}/{epochs} | Loss {logs['loss']:.3e} | "
                             f"Data {logs['data']:.3e} | PDE {logs['pde']:.3e} | "
                             f"Time {elapsed:.2f}s")

                    for cb in callbacks:
                        cb.on_epoch_end(ep, logs)

                    if _restart is not None:
                        _restart.maybe_save(
                            ep, self.params, self.state,
                            {"loss_hist": self.loss_hist, "data_hist": self.data_hist,
                             "pde_hist": self.pde_hist})
            except StopIteration:
                pass

            epochs = remainder
            if remainder == 0:
                final_logs = {
                    "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
                    "data": self.data_hist[-1] if self.data_hist else float("nan"),
                    "pde": self.pde_hist[-1] if self.pde_hist else float("nan"),
                }
                for cb in callbacks:
                    cb.on_train_end(final_logs)
                if not callbacks:
                    elapsed = time.time() - start
                    print(f"Training complete — final loss {final_logs['loss']:.3e} | "
                         f"{elapsed:.1f}s")
                if _restart is not None:
                    _restart.done()
                return

        # ------------------------------------------------------------------
        # PYTHON-LOOP MODE  (n_scan == 1, or scan tail)
        # ------------------------------------------------------------------
        ep_offset = len(self.loss_hist)
        if n_scan == 1 and _ep_resume > 0:
            epochs = epochs - _ep_resume

        _t_first = None
        _n_start = len(self.loss_hist)

        try:
            for ep in range(epochs):
                _t0 = time.time()
                key, k = jax.random.split(key)
                idx = safe_choice(k, N, batch_size)
                w = self._warmup_weight(ep_offset + ep)

                self.params, self.state, loss, dl, pl = self._step(
                    self.params, self.state,
                    x_input[idx], model_input[idx], u_target[idx], w)

                if _t_first is None:
                    _t_first = time.time() - _t0

                self.loss_hist.append(float(loss))
                self.data_hist.append(float(dl))
                self.pde_hist.append(float(pl))

                logs = {"loss": float(loss), "data": float(dl), "pde": float(pl)}

                if not callbacks and (ep_offset + ep) % 10 == 0:
                    elapsed = time.time() - start
                    print(f"Epoch {ep_offset + ep:5d} | Loss {float(loss):.3e} | "
                         f"Data {float(dl):.3e} | PDE {float(pl):.3e} | "
                         f"Time {elapsed:.2f}s")

                for cb in callbacks:
                    cb.on_epoch_end(ep_offset + ep, logs)

                if _restart is not None:
                    _restart.maybe_save(
                        ep_offset + ep, self.params, self.state,
                        {"loss_hist": self.loss_hist, "data_hist": self.data_hist,
                         "pde_hist": self.pde_hist})
        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "data": self.data_hist[-1] if self.data_hist else float("nan"),
            "pde": self.pde_hist[-1] if self.pde_hist else float("nan"),
        }
        elapsed = time.time() - start
        _n_ep = len(self.loss_hist) - _n_start
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Training complete — final loss {final_logs['loss']:.3e} | "
                 f"{fmt_train_time(elapsed, _t_first, _n_ep)}")

        if _restart is not None:
            _restart.done()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_step(self):
        loss_fn = self.loss
        opt = self.opt

        @jax.jit
        def step(params, state, x_input, model_input, u_target, pde_weight):
            def objective(p):
                return loss_fn(p, x_input, u_target, pde_weight, model_input)

            (loss, (dl, pl)), grads = jax.value_and_grad(
                objective, has_aux=True)(params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, loss, dl, pl

        return step

    def _build_scan_step(self):
        loss_fn = self.loss
        opt = self.opt

        def scan_body(carry, batch):
            params, state = carry
            x_input, model_input, u_target, pde_weight = batch

            def objective(p):
                return loss_fn(p, x_input, u_target, pde_weight, model_input)

            (loss, (dl, pl)), grads = jax.value_and_grad(
                objective, has_aux=True)(params)
            updates, new_state = opt.update(grads, state)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_state), (loss, dl, pl)

        @jax.jit
        def scan_step(params, state, batches):
            (new_params, new_state), aux = jax.lax.scan(
                scan_body, (params, state), batches)
            return new_params, new_state, aux

        return scan_step


class DeepONetSolver(BaseSolver):
    """Training-loop orchestrator for the physics-informed DeepONet.

    Unlike :class:`OperatorSolver`, each step draws three independent point
    sets (IC, BC, residual) rather than batching one fixed pool, so training
    resamples fresh collocation/IC/BC points every epoch instead of indexing
    into a pre-built array (Python-loop only — no ``n_scan_steps`` mode).
    """

    def __init__(self, model, loss, lr: float = 1e-3, lr_schedule=None,
                pde_weight: float = 1.0, pde_warmup_epochs: int = 0):
        self.model = model
        self.loss = loss
        self.opt = self._make_opt(lr, lr_schedule)
        self.pde_weight = pde_weight
        self.pde_warmup_epochs = pde_warmup_epochs

        self.loss_hist: list = []
        self.ics_hist: list = []
        self.bcs_hist: list = []
        self.res_hist: list = []

        self._step = self._build_step()

    def init(self, key, u_dummy, xt_dummy) -> None:
        self.params = self.model.init(key, u_dummy, xt_dummy)
        self.state = self.opt.init(self.params)

    def _warmup_weight(self, ep: int) -> float:
        if self.pde_warmup_epochs <= 0:
            return self.pde_weight
        return self.pde_weight * min(1.0, (ep + 1) / self.pde_warmup_epochs)

    def train(
        self,
        u_pool,
        x_sensors,
        xt_bc_lo=None,
        xt_bc_hi=None,
        n_r: int = 2000,
        x_range: tuple = (-1.0, 1.0),
        t_range: tuple = (0.0, 1.0),
        epochs: int = 20000,
        batch_size: int = 64,
        seed: int = 0,
        config: TrainingConfig = None,
    ) -> None:
        """
        u_pool     : ``(N, m)`` sensor-value pool, one row per sampled IC.
        x_sensors  : ``(m,)`` shared sensor x-locations.
        xt_bc_lo, xt_bc_hi : ``(N_bc, 2)`` shared boundary query points, or
                     ``None`` to skip the BC term (e.g. Dirichlet handled via
                     ``xt_bc_hi=None`` and the loss's own zero-Dirichlet path).
        n_r        : residual points sampled fresh each step.
        x_range, t_range : sampling ranges for the residual collocation points.
        """
        if config is not None:
            epochs = config.epochs
            batch_size = config.batch_i
            seed = config.seed
            callbacks = list(config.callbacks)
            self._attach_checkpoint_callbacks(callbacks)
            if config.lr_schedule is not None:
                self.opt = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()
                self.state = self.opt.init(self.params)
        else:
            callbacks = []

        u_pool = jnp.asarray(u_pool)
        x_sensors = jnp.asarray(x_sensors)
        N = u_pool.shape[0]
        key = jax.random.PRNGKey(seed)
        start = time.time()
        _t_first = None
        _n_start = len(self.loss_hist)

        try:
            for ep in range(epochs):
                _t0 = time.time()
                key, k1, k2, k3, k4 = jax.random.split(key, 5)

                idx_ic = safe_choice(k1, N, batch_size)
                idx_r = safe_choice(k2, N, batch_size)
                x_r = jax.random.uniform(k3, (batch_size,), minval=x_range[0],
                                        maxval=x_range[1])
                t_r = jax.random.uniform(k4, (batch_size,), minval=t_range[0],
                                        maxval=t_range[1])
                xt_r = jnp.stack([x_r, t_r], axis=1)
                w = self._warmup_weight(ep)

                self.params, self.state, loss, ics_l, bcs_l, res_l = self._step(
                    self.params, self.state,
                    u_pool[idx_ic], x_sensors, u_pool[idx_r], xt_r,
                    xt_bc_lo, xt_bc_hi, w)

                if _t_first is None:
                    _t_first = time.time() - _t0

                self.loss_hist.append(float(loss))
                self.ics_hist.append(float(ics_l))
                self.bcs_hist.append(float(bcs_l))
                self.res_hist.append(float(res_l))

                logs = {"loss": float(loss), "ics": float(ics_l),
                       "bcs": float(bcs_l), "res": float(res_l)}

                if not callbacks and ep % 500 == 0:
                    elapsed = time.time() - start
                    print(f"Epoch {ep:5d} | Loss {float(loss):.3e} | "
                         f"ICS {float(ics_l):.3e} | BCS {float(bcs_l):.3e} | "
                         f"RES {float(res_l):.3e} | Time {elapsed:.2f}s")

                for cb in callbacks:
                    cb.on_epoch_end(ep, logs)
        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "ics": self.ics_hist[-1] if self.ics_hist else float("nan"),
            "bcs": self.bcs_hist[-1] if self.bcs_hist else float("nan"),
            "res": self.res_hist[-1] if self.res_hist else float("nan"),
        }
        elapsed = time.time() - start
        _n_ep = len(self.loss_hist) - _n_start
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Training complete — final loss {final_logs['loss']:.3e} | "
                 f"{fmt_train_time(elapsed, _t_first, _n_ep)}")

    def _build_step(self):
        loss_fn = self.loss
        opt = self.opt

        @jax.jit
        def step(params, state, u_ic, x_sensors, u_r, xt_r, xt_bc_lo, xt_bc_hi,
                pde_weight):
            def objective(p):
                return loss_fn(p, u_ic, x_sensors, u_r, xt_r, xt_bc_lo, xt_bc_hi,
                              pde_weight)

            (loss, (ics_l, bcs_l, res_l)), grads = jax.value_and_grad(
                objective, has_aux=True)(params)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state, loss, ics_l, bcs_l, res_l

        return step
