import numpy as np
import jax.numpy as jnp

def save_prediction_npz(model, params, x, t, filename):
    """
    Save model prediction u(x,t) to NPZ file.
    """
    inp = jnp.stack([x, t], axis=1)
    u_pred = model.apply(params, inp)[:, 0]

    np.savez(
        filename,
        u=np.array(u_pred),
        x=np.array(x),
        t=np.array(t),
    )
