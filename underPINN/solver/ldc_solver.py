import jax
import jax.numpy as jnp
import optax
import time
from jax.tree_util import register_pytree_node_class

from underPINN.core.base import BaseSolver
from underPINN.utils.timing import fmt_train_time
from underPINN.core.config import TrainingConfig


@register_pytree_node_class
class LDCInputWrapper:
    def __init__(self, col, inlet, noslip):
        self.col = col
        self.inlet = inlet
        self.noslip = noslip

    def tree_flatten(self):
        return ((self.col, self.inlet, self.noslip), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class LDCSolver(BaseSolver):
    """Training-loop orchestrator for the Lid-Driven Cavity (2-D steady NS).

    Supports both the legacy keyword API and the production :class:`TrainingConfig`
    path with callbacks (logging, early stopping, …).

    RBA (Residual-Based Adaptivity) is always active for the physics residuals.
    """

    # EMA decay and initial bias for the per-point running sums
    _GAMMA = 0.999
    _ETA   = 0.01
    _R0    = 2.0          # initial running-sum value (avoids cold-start division)

    def __init__(self, model, pde, optimizer=None):
        self.model = model
        self.pde   = pde
        self.opt   = optimizer

        self.loss_hist: list = []
        self.phys_hist: list = []
        self.bc_hist:   list = []

        self._step = self._build_step()

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def init(self, key) -> None:
        self.params = self.model.init(key, jnp.ones((1, 2)))
        self.state  = self.opt.init(self.params)

    def train(
        self,
        inputs: LDCInputWrapper,
        epochs:     int  = 5000,
        batch_size: int  = 2000,
        seed:       int  = 0,
        config: TrainingConfig = None,
    ) -> None:
        """Run the LDC training loop.

        Parameters
        ----------
        inputs     : collocation + boundary point wrapper
        epochs, batch_size, seed : legacy scalars
        config     : :class:`TrainingConfig` — preferred production path
        """
        if config is not None:
            epochs     = config.epochs
            batch_size = config.batch_r
            seed       = config.seed
            callbacks  = list(config.callbacks)
            self._attach_checkpoint_callbacks(callbacks)
            if config.lr_schedule is not None:
                self.opt   = self._make_opt(config.lr, config.lr_schedule)
                self._step = self._build_step()
                self.state = self.opt.init(self.params)
        else:
            callbacks = []

        # ── Restart / resume ──────────────────────────────────────────────────
        _restart = None
        start_ep = 0
        if (config is not None
                and getattr(config, "out_dir", "")
                and getattr(config, "save_restart_every", 0) > 0):
            from underPINN.utils.restart import RestartManager
            _restart = RestartManager(
                config.out_dir,
                save_every=config.save_restart_every,
                cfg=None,   # hash check done by 'resume' CLI; solver uses done-flag only
            )
            start_ep, self.params, self.state, _hists = \
                _restart.maybe_restore(self.params, self.state)
            if start_ep > 0:
                for _attr, _hk in (
                    ("loss_hist", "loss_hist"),
                    ("phys_hist", "phys_hist"),
                    ("bc_hist",   "bc_hist"),
                ):
                    _saved = _hists.get(_hk, [])
                    if _saved:
                        getattr(self, _attr).extend(_saved)

        n_col    = inputs.col.shape[0]
        n_inlet  = inputs.inlet.shape[0]
        n_noslip = inputs.noslip.shape[0]
        batch_size = min(batch_size, n_col)
        steps = n_col // batch_size

        # Per-point EMA running sums for RBA (one per residual component)
        rsum1 = jnp.zeros(n_col)
        rsum2 = jnp.zeros(n_col)
        rsum3 = jnp.zeros(n_col)

        key   = jax.random.PRNGKey(seed)
        start = time.time()

        print(
            f"LDCSolver: {epochs} epochs | batch={batch_size} | "
            f"col={n_col} inlet={n_inlet} noslip={n_noslip}"
        )

        _t_first: float | None = None   # first-step time for JIT detection
        _n_start = len(self.loss_hist)  # history length before this run

        try:
            for ep in range(start_ep, epochs):
                _t0 = time.time()
                # True for the first epoch of any run — fresh start or resumed.
                # Ensures RBA running sums are properly seeded even after restart
                # (rsum1/2/3 are zeroed above; is_init=True sets eta=1.0).
                is_init = ep == start_ep
                key, k1, k2, k3 = jax.random.split(key, 4)

                perms = jax.random.permutation(k1, n_col)
                col_s          = inputs.col[perms]
                r1_s, r2_s, r3_s = rsum1[perms], rsum2[perms], rsum3[perms]

                ep_loss = 0.0
                r1_ups, r2_ups, r3_ups = [], [], []

                for i in range(steps):
                    s, e = i * batch_size, (i + 1) * batch_size

                    col_batch = col_s[s:e]
                    r_batch   = (r1_s[s:e], r2_s[s:e], r3_s[s:e])

                    idx_in = jax.random.randint(k2, (batch_size,), 0, n_inlet)
                    idx_no = jax.random.randint(k3, (batch_size,), 0, n_noslip)

                    self.params, self.state, new_r, loss, aux = self._step(
                        self.params, self.state, r_batch,
                        col_batch, inputs.inlet[idx_in], inputs.noslip[idx_no],
                        is_init,
                    )

                    ep_loss += float(loss)
                    r1_ups.append(new_r[0])
                    r2_ups.append(new_r[1])
                    r3_ups.append(new_r[2])

                rsum1 = jnp.concatenate(r1_ups)
                rsum2 = jnp.concatenate(r2_ups)
                rsum3 = jnp.concatenate(r3_ups)

                if _t_first is None:
                    _t_first = time.time() - _t0

                avg_loss = ep_loss / steps
                phys_l, lin, lno = aux
                bc_l = float(lin) + float(lno)

                self.loss_hist.append(avg_loss)
                self.phys_hist.append(float(phys_l))
                self.bc_hist.append(bc_l)

                # ── Restart snapshot ──────────────────────────────────────────
                if _restart is not None:
                    _restart.maybe_save(
                        ep,
                        self.params, self.state,
                        {"loss_hist": self.loss_hist,
                         "phys_hist": self.phys_hist,
                         "bc_hist":   self.bc_hist},
                    )

                logs = {"loss": avg_loss, "pde": float(phys_l), "bc": bc_l}

                if not callbacks and ep % 100 == 0:
                    elapsed = time.time() - start
                    print(
                        f"Ep {ep:5d} | Loss {avg_loss:.4e} | "
                        f"Phys {float(phys_l):.4e} | BC {bc_l:.4e} | "
                        f"Time {elapsed:.2f}s"
                    )

                for cb in callbacks:
                    cb.on_epoch_end(ep, logs)

        except StopIteration:
            pass

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
            "pde":  self.phys_hist[-1] if self.phys_hist else float("nan"),
            "bc":   self.bc_hist[-1]   if self.bc_hist   else float("nan"),
        }
        elapsed = time.time() - start
        _n_ep   = len(self.loss_hist) - _n_start
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Training complete — final loss {final_logs['loss']:.3e} | "
                  f"{fmt_train_time(elapsed, _t_first, _n_ep)}")

        # Mark snapshot as done so the next identical run starts fresh.
        if _restart is not None:
            _restart.done()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def loss_fn(self, params, rsum_state, col, inlet, noslip, is_init):
        res = self.pde.residual(params, col)
        r_cont, r_mom_x, r_mom_y = res[:, 0], res[:, 1], res[:, 2]

        eta = jax.lax.select(is_init, 1.0, self._ETA)

        def norm(r):
            return eta * jnp.abs(r) / (jnp.max(jnp.abs(r)) + 1e-8)

        r1, r2, r3 = rsum_state
        r1_new = jax.lax.stop_gradient(r1 * self._GAMMA + norm(r_cont))
        r2_new = jax.lax.stop_gradient(r2 * self._GAMMA + norm(r_mom_x))
        r3_new = jax.lax.stop_gradient(r3 * self._GAMMA + norm(r_mom_y))

        l_cont  = jnp.mean(((r1_new + self._R0) * r_cont)  ** 2)
        l_mom_x = jnp.mean(((r2_new + self._R0) * r_mom_x) ** 2)
        l_mom_y = jnp.mean(((r3_new + self._R0) * r_mom_y) ** 2)
        loss_phys = l_cont + l_mom_x + l_mom_y

        out_in = self.model.apply(params, inlet)
        l_in   = jnp.mean((out_in[:, 0] - 1.0) ** 2) + jnp.mean(out_in[:, 1] ** 2)

        out_no = self.model.apply(params, noslip)
        l_no   = jnp.mean(out_no[:, 0] ** 2) + jnp.mean(out_no[:, 1] ** 2)

        total_loss = loss_phys + 10.0 * (l_in + l_no)

        return total_loss, ((r1_new, r2_new, r3_new), (loss_phys, l_in, l_no))

    def _build_step(self):
        @jax.jit
        def step(params, opt_state, rsum, col, inlet, noslip, is_init):
            def loss_w(p):
                return self.loss_fn(p, rsum, col, inlet, noslip, is_init)

            (loss, (new_rsum, aux)), grads = jax.value_and_grad(
                loss_w, has_aux=True
            )(params)

            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, new_rsum, loss, aux

        return step
