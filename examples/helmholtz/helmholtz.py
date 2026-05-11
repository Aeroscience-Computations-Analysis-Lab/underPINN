"""
Helmholtz Equation — underPINN Example
========================================

Solves the 2-D Helmholtz equation with a manufactured (exact) solution:

    Δu + k² u = f(x, y),    (x, y) ∈ [0, 1]²

    f     = -(2π² - k²) sin(πx) sin(πy)
    BC    : u = 0 on all four edges
    Exact : u(x, y) = sin(πx) sin(πy)

Key features demonstrated
--------------------------
- Steady elliptic PDE with oscillatory solution
- FourierMLP: Fourier-feature embedding improves resolution for high k
- SteadySolver + SteadyLoss with RBA (residual-based adaptive weighting)
- TrainingConfig / ConsoleLogger / EarlyStopping
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from underPINN.nn.mlp import FourierMLP
from underPINN.pde.helmholtz import HelmholtzPDE
from underPINN.losses.steady_loss import SteadyLoss
from underPINN.solver.steady_solver import SteadySolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.io import save_predictions


# ---- Hyper-parameters ----
K      = 4.0    # wave number — increase for a harder oscillatory problem
EPOCHS = 8000


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_data(N_r=10000, N_per_edge=200, seed=0):
    rng = np.random.default_rng(seed)

    # Interior collocation (uniform random in [0,1]²)
    xy_r = rng.uniform(0.0, 1.0, (N_r, 2)).astype(np.float32)

    # Boundary: all four edges, u = 0 everywhere
    t = np.linspace(0.0, 1.0, N_per_edge, dtype=np.float32)
    bottom = np.stack([t,              np.zeros_like(t)], axis=1)
    top    = np.stack([t,              np.ones_like(t)],  axis=1)
    left   = np.stack([np.zeros_like(t), t],              axis=1)
    right  = np.stack([np.ones_like(t),  t],              axis=1)
    xy_b = np.concatenate([bottom, top, left, right], axis=0)
    u_b  = np.zeros(len(xy_b), dtype=np.float32)

    return jnp.array(xy_r), jnp.array(xy_b), jnp.array(u_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())
    print(f"Helmholtz: k={K}, epochs={EPOCHS}")

    # FourierMLP: sigma ≈ k to seed frequencies near the solution's wavenumber
    model  = FourierMLP(layers=[2, 128, 128, 128, 1], n_fourier=32, sigma=float(K))
    pde    = HelmholtzPDE(model, k=K)
    loss   = SteadyLoss(model, pde, bc_weight=20.0, rba=True)
    solver = SteadySolver(model, pde, loss)

    solver.init(jax.random.PRNGKey(0))

    xy_r, xy_b, u_b = make_data()

    config = TrainingConfig(
        epochs=EPOCHS,
        lr=1e-3,
        lr_schedule=optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2),
        batch_r=2048,
        batch_b=256,
        log_every=500,
        callbacks=[
            ConsoleLogger(log_every=500),
            EarlyStopping(patience=600),
        ],
    )

    solver.train(xy_r, xy_b, u_b, config=config)

    # ---- Evaluate on grid ----
    N = 200
    x = jnp.linspace(0.0, 1.0, N)
    y = jnp.linspace(0.0, 1.0, N)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

    u_pred  = model.apply(solver.params, grid)[:, 0].reshape(N, N)
    u_exact = pde.exact(grid).reshape(N, N)

    rel_l2 = float(
        jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
        / jnp.sqrt(jnp.mean(u_exact ** 2))
    )
    max_err = float(jnp.max(jnp.abs(u_pred - u_exact)))
    print(f"\nRelative L2 error : {rel_l2:.4e}")
    print(f"Max absolute error: {max_err:.4e}")

    # ---- Plots ----
    x_np = np.array(x)
    y_np = np.array(y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes,
        [u_exact, u_pred, jnp.abs(u_pred - u_exact)],
        ["Exact  u(x,y)", "PINN  u(x,y)", f"|Error|  (Rel-L2={rel_l2:.2e})"],
    ):
        cf = ax.contourf(x_np, y_np, np.array(field), 50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"Helmholtz  Δu + k²u = f   (k={K})")
    fig.tight_layout()
    fig.savefig("helmholtz_solution.png", dpi=150)
    plt.close(fig)

    # Loss history
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(solver.loss_hist, label="Total")
    ax2.semilogy(solver.pde_hist,  label="PDE",  alpha=0.7)
    ax2.semilogy(solver.bc_hist,   label="BC",   alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Helmholtz (k={K}) — training loss")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("helmholtz_loss.png", dpi=150)
    plt.close(fig2)

    print("Plots saved: helmholtz_solution.png, helmholtz_loss.png")

    # ---- Save predictions at collocation points ----
    u_pred_r  = model.apply(solver.params, xy_r)[:, 0]
    u_exact_r = pde.exact(xy_r)
    save_predictions(
        ".",
        coords  = {"x": np.array(xy_r[:, 0]), "y": np.array(xy_r[:, 1])},
        outputs = {"u_pred": u_pred_r},
        exact   = {"u_exact": u_exact_r},
    )


if __name__ == "__main__":
    main()
