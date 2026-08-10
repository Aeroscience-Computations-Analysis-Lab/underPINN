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
    bc_hist=None,
    reg_hist=None,
    filename="loss_curve.png",
):
    plt.figure(figsize=(7, 5))

    plt.semilogy(loss_hist, label="Total Loss")
    plt.semilogy(pde_hist, label="PDE Loss")
    plt.semilogy(ic_hist, label="IC Loss")
    if bc_hist is not None and any(v > 0 for v in bc_hist):
        plt.semilogy(bc_hist, label="BC Loss")
    if reg_hist is not None and any(v > 0 for v in reg_hist):
        plt.semilogy(reg_hist, label="Reg Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_ode_result(
    t_test,
    u_pred,
    u_exact,
    loss_hist,
    pde_hist,
    ic_hist,
    title: str = "ODE Solution",
    filename: str = "ode_result.png",
):
    """Two-panel plot: solution comparison + training loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    t = np.array(t_test)
    axes[0].plot(t, np.array(u_exact), "k-", lw=2, label="Exact")
    axes[0].plot(t, np.array(u_pred), "r--", lw=2, label="PINN")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u(t)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(loss_hist, label="Total")
    axes[1].semilogy(pde_hist, label="PDE")
    axes[1].semilogy(ic_hist, label="IC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")


def plot_2d_comparison(
    xy: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    loss_hist: list,
    pde_hist: list,
    bc_hist: list,
    nx: int = 100,
    ny: int = 100,
    title: str = "2-D Steady PDE",
    filename: str = "heat_result.png",
    cmap: str = "hot",
):
    """Four-panel figure for 2-D steady-state PINN results.

    Panels: PINN prediction | Exact solution | Absolute error | Training loss.

    Parameters
    ----------
    xy : (N, 2)   evaluation points (x, y)
    u_pred, u_exact : (N,)
    loss_hist, pde_hist, bc_hist : per-epoch loss lists
    nx, ny : grid resolution for the pcolormesh
    """
    x_vals = xy[:, 0].reshape(nx, ny)
    y_vals = xy[:, 1].reshape(nx, ny)
    u_p = np.array(u_pred).reshape(nx, ny)
    u_e = np.array(u_exact).reshape(nx, ny)
    err = np.abs(u_p - u_e)

    vmin = min(u_p.min(), u_e.min())
    vmax = max(u_p.max(), u_e.max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, data, label in zip(
        axes[:3],
        [u_p, u_e, err],
        ["PINN", "Exact", "|Error|"],
    ):
        vlo, vhi = (vmin, vmax) if label != "|Error|" else (0, err.max())
        cm = cmap if label != "|Error|" else "Reds"
        pcm = ax.pcolormesh(x_vals, y_vals, data, cmap=cm, vmin=vlo, vmax=vhi, shading="auto")
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    axes[3].semilogy(loss_hist, label="Total")
    axes[3].semilogy(pde_hist, label="PDE")
    axes[3].semilogy(bc_hist, label="BC")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Loss")
    axes[3].set_title("Training Loss")
    axes[3].legend()
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


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
