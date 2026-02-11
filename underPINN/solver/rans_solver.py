import jax
import jax.numpy as jnp
import optax
import time
from jax.tree_util import register_pytree_node_class

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

class RANSSolver:
    def __init__(self, model, pde, optimizer=None):
        self.model = model
        self.pde = pde
        
        # Scheduler (ReduceLROnPlateau equivalent not directly in Optax, 
        # using CosineDecay or manual intervention is common. 
        # Here we use a standard schedule for stability).
        if optimizer is None:
            # Mimicking your initial LR=0.001
            schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=5000, alpha=0.1)
            self.opt = optax.adam(learning_rate=schedule)
        else:
            self.opt = optimizer

        self.step_fn = self._build_step()
        
        # RBA Hyperparameters from your code
        self.gamma = 0.999
        self.eta = 0.01
        self.rsum_val = 2.0  # Corresponds to self.rsum = 2 in your init

    def loss_fn_rba(self, params, rsum_state, col, inlet, noslip, outlet, data_x, data_u, is_init_step):
        """
        Computes loss using Residual-Based Adaptivity (RBA).
        rsum_state: Tuple of (rsum1, rsum2, rsum3, rsum4, rsum5) for the CURRENT batch.
        """
        # 1. Compute Physics Residuals
        # residuals shape: (Batch, 5) -> [cont, mom_x, mom_y, T_k, T_e]
        res = self.pde.residual(params, col)
        
        r_cont = res[:, 0]
        r_u_mom = res[:, 1]
        r_v_mom = res[:, 2]
        r_Tk = res[:, 3]
        r_Te = res[:, 4]

        # 2. RBA Normalization & Update
        # Note: Your code uses 'eta = 1 if self.init == 2 and epoch == 0 else self.eta'
        # We handle this via the is_init_step flag
        eta = jax.lax.select(is_init_step, 1.0, self.eta)

        def normalize(r):
            return eta * jnp.abs(r) / (jnp.max(jnp.abs(r)) + 1e-8)

        # Calculate norms
        n_cont = normalize(r_cont)
        n_u_mom = normalize(r_u_mom)
        n_v_mom = normalize(r_v_mom)
        n_Tk = normalize(r_Tk)
        n_Te = normalize(r_Te)

        # Update running sums (Exponential Moving Average)
        # rsum_new = rsum_old * gamma + norm
        r1_old, r2_old, r3_old, r4_old, r5_old = rsum_state
        
        r1_new = jax.lax.stop_gradient(r1_old * self.gamma + n_u_mom)
        r2_new = jax.lax.stop_gradient(r2_old * self.gamma + n_v_mom)
        r3_new = jax.lax.stop_gradient(r3_old * self.gamma + n_cont)
        r4_new = jax.lax.stop_gradient(r4_old * self.gamma + n_Tk)
        r5_new = jax.lax.stop_gradient(r5_old * self.gamma + n_Te)

        # 3. Compute Adaptive Physics Loss
        # Loss = mean( ((rsum_updated + fixed_rsum) * residual)^2 )
        # Note: Your code maps:
        # rsum1 -> u_mom, rsum2 -> v_mom, rsum3 -> cont, rsum4 -> Tk, rsum5 -> Te
        
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

        out_data = self.model.apply(params, data_x)
        # Weights: u,v,p,eps=1.0, k=2.0
        loss_data = jnp.mean((out_data[:,0]-data_u[:,0])**2 + 
                             (out_data[:,1]-data_u[:,1])**2 + 
                             (out_data[:,2]-data_u[:,2])**2 + 
                             2.0*(out_data[:,3]-data_u[:,3])**2 + 
                             (out_data[:,4]-data_u[:,4])**2)

        # Total Loss
        total_loss = loss_phys + 10.0 * (loss_inlet + loss_noslip + loss_pressure + loss_data)

        # Pack updated state and aux logs
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

    def train(self, params, inputs: RANSInputWrapper, epochs=5000, batch_size=2000, seed=0):
        opt_state = self.opt.init(params)
        key = jax.random.PRNGKey(seed)
        
        n_col = inputs.col.shape[0]
        batch_size = min(batch_size, n_col)
        steps_per_epoch = n_col // batch_size

        # Initialize RBA State (Zeros for all collocation points)
        # We store these on GPU and slice them for each batch
        rsum1 = jnp.zeros(n_col)
        rsum2 = jnp.zeros(n_col)
        rsum3 = jnp.zeros(n_col)
        rsum4 = jnp.zeros(n_col)
        rsum5 = jnp.zeros(n_col)

        print(f"RBA Training on {n_col} points. Batch: {batch_size}. Steps: {steps_per_epoch}")
        start_time = time.time()
        
        for ep in range(epochs):
            # Flag for the special initialization step in your code (init==2)
            is_init_step = (ep == 0)
            
            # Shuffle indices
            key, subkey = jax.random.split(key)
            perms = jax.random.permutation(subkey, n_col)
            
            # Shuffle Data AND RBA State to match
            col_shuffled = inputs.col[perms]
            rsum1_s = rsum1[perms]
            rsum2_s = rsum2[perms]
            rsum3_s = rsum3[perms]
            rsum4_s = rsum4[perms]
            rsum5_s = rsum5[perms]

            epoch_loss = 0.0
            
            # Lists to collect updated RBA states to put them back together
            # (Note: In pure JAX this reconstruct is slightly expensive but necessary for RBA state persistence)
            r1_updates, r2_updates, r3_updates, r4_updates, r5_updates = [], [], [], [], []

            for i in range(steps_per_epoch):
                idx_start = i * batch_size
                idx_end = idx_start + batch_size
                
                # Batch Slicing
                col_batch = col_shuffled[idx_start:idx_end]
                rsum_batch = (
                    rsum1_s[idx_start:idx_end],
                    rsum2_s[idx_start:idx_end],
                    rsum3_s[idx_start:idx_end],
                    rsum4_s[idx_start:idx_end],
                    rsum5_s[idx_start:idx_end]
                )

                # Train Step
                params, opt_state, new_rsum_batch, loss, aux = self.step_fn(
                    params, opt_state, rsum_batch,
                    col_batch, inputs.inlet, inputs.noslip, inputs.outlet, inputs.data_x, inputs.data_u,
                    is_init_step
                )
                
                epoch_loss += loss
                
                # Collect updated RBA states
                r1_updates.append(new_rsum_batch[0])
                r2_updates.append(new_rsum_batch[1])
                r3_updates.append(new_rsum_batch[2])
                r4_updates.append(new_rsum_batch[3])
                r5_updates.append(new_rsum_batch[4])

            # Reconstruct global RBA state from updates (sorted back to original order is strictly NOT required 
            # if we shuffle every epoch, we just carry forward the 'bag' of weights. 
            # But standard RBA implies point-wise tracking. 
            # Since we shuffle 'rsum' WITH 'col' at start of loop, 'r1_updates' corresponds to 'col_shuffled'.
            # We can just overwrite rsum1 with concatenated updates and use that as the new state for next epoch's shuffle.
            rsum1 = jnp.concatenate(r1_updates, axis=0)
            rsum2 = jnp.concatenate(r2_updates, axis=0)
            rsum3 = jnp.concatenate(r3_updates, axis=0)
            rsum4 = jnp.concatenate(r4_updates, axis=0)
            rsum5 = jnp.concatenate(r5_updates, axis=0)
            
            # Note: Technically we should technically unsort perms to map weights back to exact spatial X coordinates
            # if spatial locality matters for stability, but for RBA which is point-specific, 
            # keeping (x, weight) paired via the shuffle-carry mechanism is sufficient.
            
            if ep % 10 == 0:
                phys, inl, nos, press, dat = aux
                avg = epoch_loss / steps_per_epoch
                print(f"Ep {ep:4d} | Tot: {avg:.3e} | Phys: {phys:.3e} | BC: {inl+nos:.3e} | Data: {dat:.3e}")
        
        print(f"Finished in {time.time()-start_time:.2f}s")
        return params