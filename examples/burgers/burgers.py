import numpy as np
import jax
import jax.numpy as jnp

from jaxpinn.nn.fbpinn import FBPINN
from jaxpinn.pde.burgers import BurgersPDE
from jaxpinn.solver.fbpinn import FBPINNSolver
from jaxpinn.losses.loss import PINNLoss

from jaxpinn.utils.plotting import (
    make_prediction_grid,
    plot_solution,
    plot_losses,
    plot_difference
)

from jaxpinn.utils.serialization import save_prediction_npz


def make_data():
    N = 80000
    rng = np.random.default_rng(42)

    x_r = rng.random(N) * (4 * np.pi) - 2 * np.pi
    t_r = rng.random(N) * 5.0

    x = np.linspace(0, 1, 1000) * (4 * np.pi) - 2 * np.pi
    u = (
        2.0 * np.exp(-(x + 2.0) ** 2 / 0.5)
        + 1.5 * np.exp(-(x) ** 2 / 0.3)
        + 1.0 * np.exp(-(x - 2.0) ** 2 / 0.4)
        + 0.3 * np.sin(2 * x) * np.exp(-x ** 2 / 8.0)
    )

    return (
        jnp.array(x_r, dtype=jnp.float32),
        jnp.array(t_r, dtype=jnp.float32),
        jnp.array(x, dtype=jnp.float32),
        jnp.array(u, dtype=jnp.float32),
    )

def make_boundary_data(n=1000):
    t = np.random.rand(n) * 5.0

    x_left = np.full_like(t, -2 * np.pi)
    x_right = np.full_like(t, 2 * np.pi)

    x_b = np.concatenate([x_left, x_right])
    t_b = np.concatenate([t, t])

    u_b = np.zeros_like(x_b)   # u = 0 at boundaries

    return (
        jnp.array(x_b, dtype=jnp.float32),
        jnp.array(t_b, dtype=jnp.float32),
        jnp.array(u_b, dtype=jnp.float32),
    )

def main():
    print("JAX devices:", jax.devices())

    layers = [2, 64, 64, 64, 64, 64, 1]

    shifts = jnp.array([
        [-2.0, 0.0],
        [ 0.0, 0.0],
        [ 2.0, 0.0],
    ])

    xs_min = jnp.array([
        [-2*np.pi, 0.0],
        [-2*np.pi/3, 0.0],
        [0.0, 0.0],
    ])

    xs_max = jnp.array([
        [0.0, 5.0],
        [2*np.pi/3, 5.0],
        [2*np.pi, 5.0],
    ])

    smins = jnp.ones_like(xs_min) * 0.5
    smaxs = jnp.ones_like(xs_max) * 0.5

    

    model = FBPINN(layers, shifts, xs_min, xs_max, smins, smaxs)
    pde = BurgersPDE(model)

    loss = PINNLoss(
        model=model,
        pde=pde,
        loss_type="l2",
        bc_weight=1.0,
        reg_weight=0.0,
        ic_weight=10.0,
    )

    solver = FBPINNSolver(model, pde, loss=loss, lr=1e-3)

    key = jax.random.PRNGKey(0)
    solver.init(key)

    x_r, t_r, x_i, u_i = make_data()

    x_b, t_b, u_b = make_boundary_data()    

    solver.train(
        x_r,
        t_r,
        x_i,
        u_i,
        x_b,
        t_b,
        u_b,
        epochs=1000,
        batch_r=4096,
        batch_i=512,
        batch_b=512,
    )

    # Create prediction grid
    x_pred, t_pred, _, _ = make_prediction_grid()

    # Save predictions (THIS CREATES pinn_bfs2.npz)
    save_prediction_npz(
        model=model,
        params=solver.params,
        x=x_pred,
        t=t_pred,
        filename="pinn_bfs2.npz",
    )

    # ----- Plot solution -----
    x_pred, t_pred, x_grid, t_grid = make_prediction_grid()

    plot_solution(
        model,
        solver.params,
        x_pred,
        t_pred,
        x_grid,
        t_grid,
        filename="burgers_solution.png",
    )

    # ----- Plot losses -----
    plot_losses(
        solver.loss_hist,
        solver.pde_hist,
        solver.ic_hist,
        filename="training_loss.png",
    )

    plot_difference(
    npz_pred="pinn_bfs2.npz",
    npz_ref="burgers_complex.npz",
    nx=400,
    ny=400,
    filename="diff_jax.png",
)


if __name__ == "__main__":
    main()
