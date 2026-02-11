import jax
import jax.numpy as jnp
import optax
import time
from jax.tree_util import register_pytree_node_class

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

class LDCSolver:
    def __init__(self, model, pde, optimizer=None):
        self.model = model
        self.pde = pde
        self.opt = optimizer
        self.step_fn = self._build_step()
        
        # RBA Parameters
        self.gamma = 0.999
        self.eta = 0.01
        self.rsum_val = 2.0

    def loss_fn(self, params, rsum_state, col, inlet, noslip, is_init):
        # 1. Physics (on mini-batch)
        res = self.pde.residual(params, col)
        r_cont, r_mom_x, r_mom_y = res[:, 0], res[:, 1], res[:, 2]

        # 2. RBA Update
        eta = jax.lax.select(is_init, 1.0, self.eta)
        def norm(r): return eta * jnp.abs(r) / (jnp.max(jnp.abs(r)) + 1e-8)
        
        r1, r2, r3 = rsum_state
        r1_new = jax.lax.stop_gradient(r1 * self.gamma + norm(r_cont))
        r2_new = jax.lax.stop_gradient(r2 * self.gamma + norm(r_mom_x))
        r3_new = jax.lax.stop_gradient(r3 * self.gamma + norm(r_mom_y))

        # Weighted Physics Loss
        l_cont  = jnp.mean(((r1_new + self.rsum_val) * r_cont)**2)
        l_mom_x = jnp.mean(((r2_new + self.rsum_val) * r_mom_x)**2)
        l_mom_y = jnp.mean(((r3_new + self.rsum_val) * r_mom_y)**2)
        loss_phys = l_cont + l_mom_x + l_mom_y

        # 3. Boundaries (on mini-batch)
        # Inlet (Top Lid): u=1, v=0
        out_in = self.model.apply(params, inlet)
        l_in = jnp.mean((out_in[:, 0] - 1.0)**2) + jnp.mean(out_in[:, 1]**2)

        # No Slip (Walls): u=0, v=0
        out_no = self.model.apply(params, noslip)
        l_no = jnp.mean(out_no[:, 0]**2) + jnp.mean(out_no[:, 1]**2)

        total_loss = loss_phys + 10.0 * (l_in + l_no)
        
        return total_loss, ((r1_new, r2_new, r3_new), (loss_phys, l_in, l_no))

    def _build_step(self):
        @jax.jit
        def step(params, opt_state, rsum, col, inlet, noslip, is_init):
            def loss_w(p): return self.loss_fn(p, rsum, col, inlet, noslip, is_init)
            (loss, (new_rsum, aux)), grads = jax.value_and_grad(loss_w, has_aux=True)(params)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, new_rsum, loss, aux
        return step

    def train(self, params, inputs, epochs=20000, batch_size=2000):
        opt_state = self.opt.init(params)
        key = jax.random.PRNGKey(0)
        
        n_col = inputs.col.shape[0]
        n_inlet = inputs.inlet.shape[0]
        n_noslip = inputs.noslip.shape[0]
        
        # Ensure batch size isn't larger than the dataset
        batch_size = min(batch_size, n_col)
        steps = n_col // batch_size
        
        # RBA State (one for each collocation point)
        rsum1 = jnp.zeros(n_col)
        rsum2 = jnp.zeros(n_col)
        rsum3 = jnp.zeros(n_col)

        print(f"Starting LDC Training: {epochs} epochs, {batch_size} batch size.")
        print(f"Total Points - Col: {n_col}, Inlet: {n_inlet}, NoSlip: {n_noslip}")
        
        start = time.time()

        for ep in range(epochs):
            is_init = (ep == 0)
            
            # Shuffle Collocation Points
            key, k1, k2, k3 = jax.random.split(key, 4)
            perms = jax.random.permutation(k1, n_col)
            
            # Shuffle RBA state with Collocation
            col_s = inputs.col[perms]
            r1_s, r2_s, r3_s = rsum1[perms], rsum2[perms], rsum3[perms]

            ep_loss = 0
            r1_ups, r2_ups, r3_ups = [], [], []

            for i in range(steps):
                s, e = i*batch_size, (i+1)*batch_size
                
                # 1. Physics Batch
                col_batch = col_s[s:e]
                r_batch = (r1_s[s:e], r2_s[s:e], r3_s[s:e])
                
                # 2. Boundary Batches (Random Sample)
                # Note: We take a random batch of boundary points matching the physics batch size
                # or a fixed size to ensure memory stability.
                idx_in = jax.random.randint(k2, (batch_size,), 0, n_inlet)
                idx_no = jax.random.randint(k3, (batch_size,), 0, n_noslip)
                
                inlet_batch = inputs.inlet[idx_in]
                noslip_batch = inputs.noslip[idx_no]
                
                # Update Step
                params, opt_state, new_r, loss, aux = self.step_fn(
                    params, opt_state, r_batch, 
                    col_batch, inlet_batch, noslip_batch, 
                    is_init
                )
                
                ep_loss += loss
                r1_ups.append(new_r[0]); r2_ups.append(new_r[1]); r3_ups.append(new_r[2])

            # Reconstruct RBA state
            rsum1 = jnp.concatenate(r1_ups)
            rsum2 = jnp.concatenate(r2_ups)
            rsum3 = jnp.concatenate(r3_ups)

            if ep % 100 == 0:
                phys, lin, lno = aux
                print(f"Ep {ep} | Loss: {ep_loss/steps:.4e} | Phys: {phys:.4e} | BC: {lin+lno:.4e}")

        print(f"Done in {time.time()-start:.2f}s")
        return params