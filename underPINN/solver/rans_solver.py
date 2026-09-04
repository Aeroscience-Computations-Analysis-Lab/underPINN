import jax
import jax.numpy as jnp
import optax
import time
from jax.tree_util import register_pytree_node_class

from underPINN.core.base import BaseSolver
from underPINN.core.config import TrainingConfig
from underPINN.utils.timing import fmt_train_time

@register_pytree_node_class
class RANSInputWrapper:
    """Struct to hold all data arrays."""
    def __init__(self, col, inlet, noslip, outlet, data_x, data_u):
        self.col = col
        self.inlet = inlet
        self.noslip = noslip
        self.outlet = outlet
        self.data_x = data_x
        self.data_u = data_u

    def tree_flatten(self):
        return ((self.col, self.inlet, self.noslip, self.outlet, self.data_x, self.data_u), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class RANSSolver(BaseSolver):
    def __init__(self, model, pde, optimizer=None):
        self.model = model
        self.pde   = pde

        if optimizer is None:
            schedule = optax.cosine_decay_schedule(
                init_value=1e-3, decay_steps=5000, alpha=0.1)
            self.opt = optax.adam(learning_rate=schedule)
        else:
            self.opt = optimizer

        self.step_fn = self._build_step()

        # RBA Hyperparameters
        self.gamma = 0.999
        self.eta = 0.01
        self.rsum_val = 2.0

        # Loss histories (populated during train())
        self.loss_hist:  list = []
        self.phys_hist:  list = []
        self.bc_hist:    list = []
        self.data_hist:  list = []

    def loss_fn_rba(self, params, rsum_state, col, inlet, noslip, outlet, data_x, data_u, is_init_step):
        """
        Computes loss using Residual-Based Adaptivity (RBA).
        rsum_state: Tuple of (rsum1, rsum2, rsum3, rsum4, rsum5) for the CURRENT batch.
        """
        # 1. Compute Physics Residuals
        # residuals shape: (Batch, 5) -> [cont, mom_x, mom_y, T_k, T_e]
        res = self.pde.residual(params, col)

        r_cont  = res[:, 0]
        r_u_mom = res[:, 1]
        r_v_mom = res[:, 2]
        r_Tk    = res[:, 3]
        r_Te    = res[:, 4]

        # 2. RBA Normalization & Update
        eta = jax.lax.select(is_init_step, 1.0, self.eta)

        def normalize(r):
            return eta * jnp.abs(r) / (jnp.max(jnp.abs(r)) + 1e-8)

        n_cont  = normalize(r_cont)
        n_u_mom = normalize(r_u_mom)
        n_v_mom = normalize(r_v_mom)
        n_Tk    = normalize(r_Tk)
        n_Te    = normalize(r_Te)

        # Update running sums (Exponential Moving Average)
        r1_old, r2_old, r3_old, r4_old, r5_old = rsum_state

        r1_new = jax.lax.stop_gradient(r1_old * self.gamma + n_u_mom)
        r2_new = jax.lax.stop_gradient(r2_old * self.gamma + n_v_mom)
        r3_new = jax.lax.stop_gradient(r3_old * self.gamma + n_cont)
        r4_new = jax.lax.stop_gradient(r4_old * self.gamma + n_Tk)
        r5_new = jax.lax.stop_gradient(r5_old * self.gamma + n_Te)

        # 3. Compute Adaptive Physics Loss
        l_u_mom = jnp.mean(((r1_new + self.rsum_val) * r_u_mom) ** 2)
        l_v_mom = jnp.mean(((r2_new + self.rsum_val) * r_v_mom) ** 2)
        l_cont  = jnp.mean(((r3_new + self.rsum_val) * r_cont) ** 2)
        l_Tk    = jnp.mean(((r4_new + self.rsum_val) * r_Tk) ** 2)
        l_Te    = jnp.mean(((r5_new + self.rsum_val) * r_Te) ** 2)

        loss_phys = l_u_mom + l_v_mom + l_cont + l_Tk + l_Te

        # 4. Boundary & Data Losses (Standard)
        out_in = self.model.apply(params, inlet)
        loss_inlet = jnp.mean((out_in[:, 0] - 1.0)**2) + jnp.mean(out_in[:, 1]**2)

        out_noslip = self.model.apply(params, noslip)
        loss_noslip = jnp.mean(out_noslip[:, 0]**2) + jnp.mean(out_noslip[:, 1]**2)

        out_outlet = self.model.apply(params, outlet)
        loss_pressure = jnp.mean(out_outlet[:, 2]**2)

        out_data = self.pde.u(params, data_x)
        loss_data = jnp.mean((out_data[:,0]-data_u[:,0])**2 +
                             (out_data[:,1]-data_u[:,1])**2 +
                             (out_data[:,2]-data_u[:,2])**2 +
                             2.0*(out_data[:,3]-data_u[:,3])**2 +
                             (out_data[:,4]-data_u[:,4])**2)

        total_loss = loss_phys + 10.0 * (loss_inlet + loss_noslip + loss_pressure + loss_data)

        new_rsum_state = (r1_new, r2_new, r3_new, r4_new, r5_new)
        aux_logs = (loss_phys, loss_inlet, loss_noslip, loss_pressure, loss_data)

        return total_loss, (new_rsum_state, aux_logs)

    def _build_step(self):
        @jax.jit
        def step(params, opt_state, rsum_state, col, inlet, noslip, outlet, data_x, data_u, is_init_step):

            def loss_wrapper(p):
                return self.loss_fn_rba(p, rsum_state, col, inlet, noslip, outlet, data_x, data_u, is_init_step)

            (loss, (new_rsum, aux)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, new_rsum, loss, aux
        return step

    def init(self, key) -> None:
        """Initialise model parameters and optimizer state."""
        self.params = self.model.init(key, jnp.ones((1, 2)))
        self.state  = self.opt.init(self.params)

    def train(
        self,
        params,
        inputs: RANSInputWrapper,
        epochs: int = 5000,
        batch_size: int = 2000,
        seed: int = 0,
        out_dir: str = "",
        save_restart_every: int = 500,
        config: TrainingConfig = None,
    ):
        """Run the RANS RBA training loop.

        Parameters
        ----------
        params :
            Initial network parameters (JAX pytree).  Also updates
            ``self.params`` at the end of training.
        inputs :
            :class:`RANSInputWrapper` containing all boundary/data arrays.
        epochs, batch_size, seed :
            Legacy scalars; ignored when *config* is provided.
        out_dir, save_restart_every :
            Legacy restart args; ignored when *config* provides ``out_dir``
            and ``save_restart_every``.
        config :
            :class:`~underPINN.core.config.TrainingConfig` — preferred
            production path.  When given, all scalar kwargs above are
            overridden by the config fields.
        """
        # ── Unpack config or fall back to legacy kwargs ───────────────────────
        if config is not None:
            epochs             = config.epochs
            batch_size         = config.batch_r
            seed               = config.seed
            out_dir            = config.out_dir
            save_restart_every = config.save_restart_every
            callbacks          = list(config.callbacks)
            self._attach_checkpoint_callbacks(callbacks)
            if config.lr_schedule is not None:
                self.opt     = self._make_opt(config.lr, config.lr_schedule)
                self.step_fn = self._build_step()
                state        = self.opt.init(params)
            else:
                state = self.opt.init(params)
        else:
            callbacks = []
            state = self.opt.init(params)

        key = jax.random.PRNGKey(seed)

        # ── Restart / resume ──────────────────────────────────────────────────
        from underPINN.utils.restart import RestartManager
        _restart = None
        if out_dir and save_restart_every > 0:
            _restart = RestartManager(
                out_dir,
                save_every=save_restart_every,
                cfg=None,
            )
            start_ep, params, state, _hists = _restart.maybe_restore(params, state)
            _saved_loss = _hists.get("loss_hist", [])
            if _saved_loss:
                self.loss_hist.extend(_saved_loss)
        else:
            start_ep = 0

        n_col = inputs.col.shape[0]
        batch_size = min(batch_size, n_col)
        steps_per_epoch = n_col // batch_size

        # Initialise RBA running sums
        rsum1 = jnp.zeros(n_col)
        rsum2 = jnp.zeros(n_col)
        rsum3 = jnp.zeros(n_col)
        rsum4 = jnp.zeros(n_col)
        rsum5 = jnp.zeros(n_col)

        print(f"RBA Training on {n_col} points. Batch: {batch_size}. Steps/ep: {steps_per_epoch}")
        start_time = time.time()
        _t_first: float | None = None

        try:
            for ep in range(start_ep, epochs):
                _t0 = time.time()
                is_init_step = (ep == start_ep)

                # Shuffle indices
                key, subkey = jax.random.split(key)
                perms = jax.random.permutation(subkey, n_col)

                col_shuffled = inputs.col[perms]
                rsum1_s = rsum1[perms]
                rsum2_s = rsum2[perms]
                rsum3_s = rsum3[perms]
                rsum4_s = rsum4[perms]
                rsum5_s = rsum5[perms]

                epoch_loss = 0.0
                r1_updates, r2_updates = [], []
                r3_updates, r4_updates, r5_updates = [], [], []
                last_aux = None

                for i in range(steps_per_epoch):
                    idx_s = i * batch_size
                    idx_e = idx_s + batch_size

                    col_batch = col_shuffled[idx_s:idx_e]
                    rsum_batch = (
                        rsum1_s[idx_s:idx_e],
                        rsum2_s[idx_s:idx_e],
                        rsum3_s[idx_s:idx_e],
                        rsum4_s[idx_s:idx_e],
                        rsum5_s[idx_s:idx_e],
                    )

                    params, state, new_rsum_batch, loss, aux = self.step_fn(
                        params, state, rsum_batch,
                        col_batch, inputs.inlet, inputs.noslip, inputs.outlet,
                        inputs.data_x, inputs.data_u,
                        is_init_step,
                    )

                    epoch_loss += loss
                    last_aux = aux

                    r1_updates.append(new_rsum_batch[0])
                    r2_updates.append(new_rsum_batch[1])
                    r3_updates.append(new_rsum_batch[2])
                    r4_updates.append(new_rsum_batch[3])
                    r5_updates.append(new_rsum_batch[4])

                # Rebuild global RBA state
                rsum1 = jnp.concatenate(r1_updates, axis=0)
                rsum2 = jnp.concatenate(r2_updates, axis=0)
                rsum3 = jnp.concatenate(r3_updates, axis=0)
                rsum4 = jnp.concatenate(r4_updates, axis=0)
                rsum5 = jnp.concatenate(r5_updates, axis=0)

                if _t_first is None:
                    _t_first = time.time() - _t0

                avg = epoch_loss / steps_per_epoch
                self.loss_hist.append(float(avg))

                phys, inl, nos, press, dat = last_aux
                self.phys_hist.append(float(phys))
                self.bc_hist.append(float(inl + nos + press))
                self.data_hist.append(float(dat))

                logs = {
                    "loss": float(avg),
                    "phys": float(phys),
                    "bc":   float(inl + nos + press),
                    "data": float(dat),
                }

                if not callbacks and ep % 10 == 0:
                    print(f"Ep {ep:4d} | Tot: {avg:.3e} | Phys: {phys:.3e} "
                          f"| BC: {inl+nos:.3e} | Data: {dat:.3e}")

                for cb in callbacks:
                    cb.on_epoch_end(ep, logs)

                if _restart is not None:
                    _restart.maybe_save(ep, params, state, {"loss_hist": self.loss_hist})

        except StopIteration:
            pass

        if _restart is not None:
            _restart.done()

        final_logs = {
            "loss": self.loss_hist[-1] if self.loss_hist else float("nan"),
        }
        elapsed = time.time() - start_time
        _n_ep = len(self.loss_hist) - start_ep
        for cb in callbacks:
            cb.on_train_end(final_logs)
        if not callbacks:
            print(f"Finished — final loss {final_logs['loss']:.3e} | "
                  f"{fmt_train_time(elapsed, _t_first, _n_ep)}")

        # Persist final state
        self.params = params
        self.state  = state
        return params, self.loss_hist
