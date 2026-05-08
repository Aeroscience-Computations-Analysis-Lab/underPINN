"""
2-D Steady-State Heat Equation (Poisson Problem)
=================================================

Problem
-------
    ∇²u = -f(x, y)     on Ω = [0,1]²
    u   = 0             on ∂Ω  (all four edges)

Source term
-----------
    f(x, y) = 2π² sin(πx) sin(πy)

Exact solution
--------------
    u(x, y) = sin(πx) sin(πy)

Verification:
    u_xx = -π² sin(πx) sin(πy)
    u_yy = -π² sin(πx) sin(πy)
    u_xx + u_yy = -2π² sin(πx) sin(πy) = -f  ✓
    u = 0 on x=0, x=1, y=0, y=1  ✓
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from underPINN.nn.mlp import MLP
from underPINN.pde.heat import SteadyHeatPDE
from underPINN.losses.steady_loss import SteadyLoss
from underPINN.solver.steady_solver import SteadySolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.utils.metrics import print_errors
from underPINN.utils.plotting import plot_2d_comparison


# ---------------------------------------------------------------------------
# Source term: f(x, y) = 2π² sin(πx) sin(πy)
# ---------------------------------------------------------------------------

def source(x, y):
    return 2.0 * jnp.pi ** 2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_interior(n: int = 8000, seed: int = 0) -> jnp.ndarray:
    """Latin-Hypercube-style uniform interior collocation points."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    return jnp.array(xy)


def make_boundary(n_per_edge: int = 200) -> tuple:
    """Sample n_per_edge points on each of the four edges of [0,1]².

    All Dirichlet values are zero for this problem.
    Returns (xy_b, u_b) with shapes (4*n_per_edge, 2) and (4*n_per_edge,).
    """
    t = np.linspace(0.0, 1.0, n_per_edge, dtype=np.float32)

    bottom = np.stack([t,             np.zeros_like(t)], axis=1)  # y = 0
    top    = np.stack([t,             np.ones_like(t)],  axis=1)  # y = 1
    left   = np.stack([np.zeros_like(t), t],             axis=1)  # x = 0
    right  = np.stack([np.ones_like(t),  t],             axis=1)  # x = 1

    xy_b = jnp.array(np.concatenate([bottom, top, left, right], axis=0))
    u_b  = jnp.zeros(xy_b.shape[0])
    return xy_b, u_b


def make_eval_grid(nx: int = 100, ny: int = 100) -> jnp.ndarray:
    """Regular grid over [0,1]² for evaluation and plotting."""
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return jnp.array(np.stack([xx.ravel(), yy.ravel()], axis=1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("JAX devices:", jax.devices())

    EPOCHS = 8000
    NX = NY = 100

    # -- Data ----------------------------------------------------------------
    xy_r        = make_interior(n=8000)
    xy_b, u_b   = make_boundary(n_per_edge=200)
    xy_eval     = make_eval_grid(NX, NY)

    # -- Model ---------------------------------------------------------------
    # Input: (N, 2) [x, y]  →  Output: (N, 1)
    model = MLP(layers=[2, 64, 64, 64, 64, 1])

    # -- Physics + Loss ------------------------------------------------------
    pde  = SteadyHeatPDE(model, source_fn=source)
    loss = SteadyLoss(model, pde, bc_weight=20.0)

    # -- Solver + Config -----------------------------------------------------
    config = TrainingConfig(
        epochs=EPOCHS,
        lr=1e-3,
        lr_schedule=optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=5e-3),
        batch_r=2048,
        batch_b=256,
        log_every=1000,
        callbacks=[
            ConsoleLogger(log_every=1000),
            EarlyStopping(patience=500),
        ],
    )

    solver = SteadySolver(model, pde, loss)
    solver.init(jax.random.PRNGKey(0))
    solver.train(xy_r, xy_b, u_b, config=config)

    # -- Evaluate ------------------------------------------------------------
    u_pred  = pde.u(solver.params, xy_eval)
    u_exact = pde.exact(xy_eval)

    print_errors(u_pred, u_exact, label="Poisson 2D")

    # -- Plot ----------------------------------------------------------------
    plot_2d_comparison(
        xy=np.array(xy_eval),
        u_pred=np.array(u_pred),
        u_exact=np.array(u_exact),
        loss_hist=solver.loss_hist,
        pde_hist=solver.pde_hist,
        bc_hist=solver.bc_hist,
        nx=NX,
        ny=NY,
        title="2-D Steady Heat (Poisson): ∇²u = -2π²sin(πx)sin(πy)",
        filename="heat_poisson_result.png",
    )

    # -- Pass/Fail -----------------------------------------------------------
    rel_err = float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)) /
                    jnp.sqrt(jnp.mean(u_exact ** 2)))
    assert rel_err < 5e-2, f"Relative L2 error too large: {rel_err:.4e}"
    print(f"\n2-D Steady Heat test PASSED  (Rel-L2 = {rel_err:.4e})")


if __name__ == "__main__":
    main()
