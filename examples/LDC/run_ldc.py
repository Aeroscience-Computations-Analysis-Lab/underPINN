import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from shapely.geometry import Polygon

from underPINN.geometry.shapely_geom import ShapelyPolygon
from underPINN.nn.fbpinn import FBPINN
from underPINN.nn.attention import SimpleGate
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.solver.ldc_solver import LDCSolver, LDCInputWrapper
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping

from underPINN.benchmark_utils.benchmark_suite import BenchmarkTracker
from underPINN.utils.io import save_predictions


def generate_ldc_geometry(n_col=40000, n_per_edge=1000):
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = ShapelyPolygon(vertices)

    x_col = poly.sample(n_col, seed=0)

    # Lid (top, y=1): u=1, v=0
    x_lid = np.linspace(0, 1, n_per_edge)
    x_inlet = np.stack([x_lid, np.ones_like(x_lid)], axis=1)

    # No-slip walls: left (x=0), right (x=1), bottom (y=0)
    t = np.linspace(0, 1, n_per_edge)
    w_left   = np.stack([np.zeros_like(t), t], axis=1)
    w_right  = np.stack([np.ones_like(t),  t], axis=1)
    w_bottom = np.stack([t, np.zeros_like(t)], axis=1)
    x_noslip = np.concatenate([w_left, w_right, w_bottom], axis=0)

    return x_col, x_inlet, x_noslip


def save_and_plot_results(model, params, filename="pinn_ldc.npz"):
    x = jnp.linspace(0, 1, 201)
    y = jnp.linspace(0, 1, 201)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

    pred = model.apply(params, grid)
    u = pred[:, 0].reshape(201, 201)
    v = pred[:, 1].reshape(201, 201)
    p = pred[:, 2].reshape(201, 201)

    print(f"Saving results to {filename}...")
    np.savez(filename, u=np.array(u), v=np.array(v), p=np.array(p),
             x=np.array(XX), y=np.array(YY))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, label in zip(axes, [u, v, p], ["u", "v", "p"]):
        cf = ax.contourf(x, y, field, levels=50, cmap="jet")
        plt.colorbar(cf, ax=ax)
        ax.set_title(f"PINN: {label}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig("ldc_solution.png", dpi=150)
    plt.close(fig)
    print("Plot saved to ldc_solution.png")


def main():
    print("JAX devices:", jax.devices())

    EPOCHS = 5000

    # ---- Smaller network: [2, 64, 64, 64, 64, 3] ----
    # Previous: [2, 224, 224, 224, 224, 224, 3] — ~5× more parameters per layer
    layers = [2, 64, 64, 64, 64, 3]

    # Single subdomain covering the full [0,1]² domain
    shifts = jnp.array([[0.5, 0.5]])
    xs_min = jnp.array([[0.0, 0.0]])
    xs_max = jnp.array([[1.0, 1.0]])
    smins  = jnp.array([[0.4, 0.4]])
    smaxs  = jnp.array([[0.4, 0.4]])

    model = FBPINN(
        layers=layers,
        shifts=shifts,
        xs_min=xs_min,
        xs_max=xs_max,
        smins=smins,
        smaxs=smaxs,
        attention_cls=SimpleGate,   # pass at construction, not via class mutation
    )

    pde    = NavierStokesPDE(model, Re=100.0)
    schedule = optax.cosine_decay_schedule(1e-3, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    solver = LDCSolver(model, pde, optimizer=optimizer)
    solver.init(jax.random.PRNGKey(123))

    config = TrainingConfig(
        epochs=EPOCHS,
        lr=1e-3,
        lr_schedule=schedule,
        batch_r=2000,
        log_every=500,
        callbacks=[
            ConsoleLogger(log_every=500),
            EarlyStopping(patience=500),
        ],
    )

    print("Generating geometry...")
    x_col, x_inlet, x_noslip = generate_ldc_geometry(n_col=40000, n_per_edge=1000)

    inputs = LDCInputWrapper(
        col    = jnp.array(x_col,    dtype=jnp.float32),
        inlet  = jnp.array(x_inlet,  dtype=jnp.float32),
        noslip = jnp.array(x_noslip, dtype=jnp.float32),
    )

    tracker = BenchmarkTracker()
    tracker.start()

    solver.train(inputs, config=config)

    tracker.stop()
    tracker.log("epochs", len(solver.loss_hist))
    tracker.save(case_name="LDC", framework="JAX")

    save_and_plot_results(model, solver.params, filename="pinn_ldc.npz")

    # Save predictions at collocation (interior residual) points
    pred_col = np.array(model.apply(solver.params, inputs.col))
    save_predictions(
        ".",
        coords  = {"x": np.array(inputs.col[:, 0]),
                   "y": np.array(inputs.col[:, 1])},
        outputs = {"u_pred": pred_col[:, 0],
                   "v_pred": pred_col[:, 1],
                   "p_pred": pred_col[:, 2]},
    )

    from flax import serialization
    with open("ldc_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(solver.params))
    print("Parameters saved to ldc_params.msgpack")


if __name__ == "__main__":
    main()
