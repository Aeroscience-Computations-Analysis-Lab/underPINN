import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from shapely.geometry import Polygon

# Import framework components
from underPINN.geometry.shapely_geom import ShapelyPolygon
from underPINN.nn.fbpinn import FBPINN, window_nd
from underPINN.nn.attention import SimpleGate
from underPINN.nn.subdomain import SubdomainNetwork
from underPINN.pde.navier_stokes import NavierStokesPDE
from underPINN.solver.ldc_solver import LDCSolver, LDCInputWrapper

from underPINN.benchmark_utils.benchmark_suite import BenchmarkTracker

def generate_ldc_geometry():
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = ShapelyPolygon(vertices)
    
    # 1. Collocation Points
    x_col = poly.sample(80000, seed=0)

    # 2. Boundary Points
    x_lid = np.linspace(0, 1, 3333) 
    x_inlet = np.stack([x_lid, np.ones_like(x_lid)], axis=1)

    y_wall = np.linspace(0, 1, 3333)
    x_wall = np.linspace(0, 1, 3334)
    w1 = np.stack([np.zeros_like(y_wall), y_wall], axis=1)
    w2 = np.stack([np.ones_like(y_wall), y_wall], axis=1)
    w3 = np.stack([x_wall, np.zeros_like(x_wall)], axis=1)
    x_noslip = np.concatenate([w1, w2, w3], axis=0)
    
    return x_col, x_inlet, x_noslip

def save_and_plot_results(model, params, filename='pinn_bfs1.npz'):
    x = jnp.linspace(0, 1, 201)
    y = jnp.linspace(0, 1, 201)
    XX, YY = jnp.meshgrid(x, y, indexing='ij')
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    
    pred = model.apply(params, grid)
    u = pred[:, 0]
    v = pred[:, 1]
    p = pred[:, 2]
    
    print(f"Saving results to {filename}...")
    np.savez(filename, u=u.reshape(201, 201), v=v.reshape(201, 201), p=p.reshape(201, 201), x=XX, y=YY)

    u_plot = u.reshape(201, 201)
    v_plot = v.reshape(201, 201)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, u_plot, levels=100, cmap='jet')
    plt.colorbar(label='u(x,y)')
    plt.title('PINN Prediction (u)')
    plt.savefig('pinn_prediction_u_bfs.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, v_plot, levels=100, cmap='jet')
    plt.colorbar(label='v(x,y)')
    plt.title('PINN Prediction (v)')
    plt.savefig('pinn_prediction_v_bfs.png')
    plt.close()
    print("Plots saved.")

def main():
    # 2. Initialize and Start
    tracker = BenchmarkTracker()
    tracker.start()
    
    print("Generating geometry...")
    x_col, x_inlet, x_noslip = generate_ldc_geometry()
    
    inputs = LDCInputWrapper(
        col=jnp.array(x_col, dtype=jnp.float32),
        inlet=jnp.array(x_inlet, dtype=jnp.float32),
        noslip=jnp.array(x_noslip, dtype=jnp.float32)
    )
    
    # FBPINN Config
    shifts = jnp.array([[0.5, 0.5]])
    xs_min = jnp.array([[0.0, 0.0]])
    xs_max = jnp.array([[1.0, 1.0]])
    smins = jnp.array([[0.4, 0.4]])
    smaxs = jnp.array([[0.4, 0.4]])
    layers = [2] + [224]*5 + [3]
    
    SubdomainNetwork.attention_cls = SimpleGate

    model = FBPINN(
        layers=layers,
        shifts=shifts,
        xs_min=xs_min,
        xs_max=xs_max,
        smins=smins,
        smaxs=smaxs
    )
    
    key = jax.random.PRNGKey(123)
    params = model.init(key, jnp.ones((1, 2)))
    
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=20000, alpha=0.1)
    optimizer = optax.adam(learning_rate=schedule)
    
    pde = NavierStokesPDE(model, Re=100.0)
    solver = LDCSolver(model, pde, optimizer)
    
    # Reduced batch size for safety (1000 instead of 2000)
    # The solver now also batches boundary points, so total memory is significantly lower.
    params = solver.train(params, inputs, epochs=10, batch_size=1000)

    tracker.stop()
    tracker.log("epochs", 10)

    tracker.save(case_name="LDC", framework="JAX")
    
    save_and_plot_results(model, params, filename='pinn_bfs1.npz')
    
    from flax import serialization
    with open("ldc_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(params))

    

if __name__ == "__main__":
    main()