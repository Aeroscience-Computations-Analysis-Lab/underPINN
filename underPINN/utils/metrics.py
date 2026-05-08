import jax.numpy as jnp
import numpy as np


def relative_l2_error(u_pred, u_exact, eps: float = 1e-12):
    """Relative L2 error: ||u_pred - u_exact||_2 / ||u_exact||_2."""
    return jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)) / (
        jnp.sqrt(jnp.mean(u_exact ** 2)) + eps
    )


def max_absolute_error(u_pred, u_exact):
    """Maximum absolute pointwise error."""
    return jnp.max(jnp.abs(u_pred - u_exact))


def mean_absolute_error(u_pred, u_exact):
    """Mean absolute error."""
    return jnp.mean(jnp.abs(u_pred - u_exact))


def print_errors(u_pred, u_exact, label: str = ""):
    prefix = f"[{label}] " if label else ""
    rel = float(relative_l2_error(u_pred, u_exact))
    mae = float(mean_absolute_error(u_pred, u_exact))
    maxe = float(max_absolute_error(u_pred, u_exact))
    print(
        f"{prefix}Rel-L2: {rel:.4e}  |  MAE: {mae:.4e}  |  Max-AE: {maxe:.4e}"
    )
    return {"rel_l2": rel, "mae": mae, "max_ae": maxe}
