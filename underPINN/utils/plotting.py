import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def make_prediction_grid(nx=400, nt=400):
    x = np.linspace(0, 1, nx) * (4 * np.pi) - 2 * np.pi
    t = np.linspace(0, 1, nt) * 5.0

    xx, tt = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)

    return (
        jnp.array(xt[:, 0], dtype=jnp.float32),
        jnp.array(xt[:, 1], dtype=jnp.float32),
        x,
        t,
    )


def plot_solution(
    model,
    params,
    x_pred,
    t_pred,
    x_grid,
    t_grid,
    filename="solution.png",
):
    inp = jnp.stack([x_pred, t_pred], axis=1)
    u_pred = model.apply(params, inp)[:, 0]

    u = np.array(u_pred).reshape(len(x_grid), len(t_grid))

    plt.figure(figsize=(6, 4))
    plt.contourf(x_grid, t_grid, u, levels=100, cmap="jet")
    plt.colorbar(label="u(x, t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Burgers Equation Solution")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_losses(
    loss_hist,
    pde_hist,
    ic_hist,
    filename="loss_curve.png",
):
    plt.figure(figsize=(7, 5))

    plt.semilogy(loss_hist, label="Total Loss")
    plt.semilogy(pde_hist, label="PDE Loss")
    plt.semilogy(ic_hist, label="IC Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_difference(
    npz_pred,
    npz_ref,
    nx=400,
    ny=400,
    filename="diff_jax.png",
    cmap="jet",
):
    """
    Plot absolute difference between PINN prediction and reference solution.

    npz_pred : str
        Path to PINN prediction .npz (must contain u, x, t)
    npz_ref : str
        Path to reference solution .npz (must contain u)
    """

    data = np.load(npz_pred)
    u = data["u"]
    x = data["x"]
    t = data["t"]

    u = u.reshape(nx, ny)
    x = x.reshape(nx, ny)
    t = t.reshape(nx, ny)

    data_ref = np.load(npz_ref)
    u_ref = data_ref["u"]

    # Reference usually stored as (ny, nx)
    diff = np.abs(u - u_ref.T)

    plt.figure(figsize=(10, 6), dpi=150)

    cont = plt.contourf(
        x,
        t,
        diff,
        levels=200,
        cmap=cmap,
        extend="both",
    )

    cbar = plt.colorbar(cont, pad=0.02)
    cbar.set_label("PINN − Reference", fontsize=12)

    plt.xlabel("x", fontsize=12)
    plt.ylabel("t", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
