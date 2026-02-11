import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Import framework modules
from underPINN.geometry.shapely_geom import ShapelyPolygon
from underPINN.nn.fbpinn import FBPINN, window_nd
from underPINN.pde.k_epsilon import KEpsilonPDE
from underPINN.solver.rans_solver import RANSSolver, RANSInputWrapper

from underPINN.benchmark_utils.benchmark_suite import BenchmarkTracker

# -------------------------------------------------------------------
# 1. Custom Geometry Helper 
# -------------------------------------------------------------------
def generate_bfs_points(n_col=80000):
    vertices = [
        (0, 1.9423), (35, 1.9423), (35, 0), 
        (5, 0), (5, 0.9423), (0, 0.9423)
    ]
    poly = ShapelyPolygon(vertices)
    
    # Interior points
    # Using underPINN's sample method (uniform) or sample_near_boundary
    x_col = poly.sample_near_boundary(n_col, decay=2.0, seed=42)
    
    # Boundary definitions (Manually defining edges for BCs)
    # Inlet: x=0, 0.9423 <= y <= 1.9423
    y_in = np.linspace(0.9423, 1.9423, 200)
    x_inlet = np.stack([np.zeros_like(y_in), y_in], axis=1)
    
    # Outlet: x=35, 0 <= y <= 1.9423
    y_out = np.linspace(0, 1.9423, 200)
    x_outlet = np.stack([np.full_like(y_out, 35.0), y_out], axis=1)
    
    # No-slip (Walls): The rest of the boundary
    # Top wall
    x_top = np.linspace(0, 35, 400)
    w1 = np.stack([x_top, np.full_like(x_top, 1.9423)], axis=1)
    # Bottom Outlet part
    x_bot = np.linspace(5, 35, 300)
    w2 = np.stack([x_bot, np.zeros_like(x_bot)], axis=1)
    # Step vertical
    y_step = np.linspace(0, 0.9423, 100)
    w3 = np.stack([np.full_like(y_step, 5.0), y_step], axis=1)
    # Step horizontal
    x_step = np.linspace(0, 5, 100)
    w4 = np.stack([x_step, np.full_like(x_step, 0.9423)], axis=1)
    
    x_noslip = np.concatenate([w1, w2, w3, w4], axis=0)
    
    return x_col, x_inlet, x_outlet, x_noslip

from flax import serialization

def save_params(params, filename):
    """Save model parameters to a binary file."""
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Parameters saved to {filename}")

def load_params(params_template, filename):
    """Load parameters from a file (requires a template of the same structure)."""
    with open(filename, "rb") as f:
        return serialization.from_bytes(params_template, f.read())


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_global_predictions(model, params, x_grid, shape=(350, 50), save_prefix='pinn_prediction'):
    """
    Plots global predictions for u, v, k, epsilon.
    x_grid: Flat JAX array of shape (N, 2)
    shape: Tuple (Nx, Ny) for reshaping grid
    """
    # Predict
    output = model.apply(params, x_grid)
    
    # Unpack and Reshape
    # Output indices: 0:u, 1:v, 2:p, 3:k, 4:eps
    u = output[:, 0].reshape(shape)
    v = output[:, 1].reshape(shape)
    k = output[:, 3].reshape(shape)
    eps = output[:, 4].reshape(shape)
    
    # Grid for plotting
    x_np = x_grid[:, 0].reshape(shape)
    y_np = x_grid[:, 1].reshape(shape)

    # Helper for consistent plotting
    def _plot_field(field, name, filename):
        plt.figure(figsize=(15, 5))
        plt.contourf(x_np, y_np, field, levels=100, cmap='jet')
        plt.colorbar(label=f"{name}(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"PINN Prediction - {name}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    _plot_field(u, "u", f"{save_prefix}_u_bfs.png")
    _plot_field(v, "v", f"{save_prefix}_v_bfs.png")
    _plot_field(k, "k", f"{save_prefix}_k_bfs.png")
    _plot_field(eps, "eps", f"{save_prefix}_eps_bfs.png")
    print(f"Global plots saved with prefix '{save_prefix}'")

# -------------------------------------------------------------------
# 2. Main Execution
# -------------------------------------------------------------------
def main():
    # A. Load Data
    print("Generating geometry...")
    x_col, x_inlet, x_noslip, x_outlet = generate_bfs_points()
    
    print("Loading data from CSV...")
    
    # 1. Read the CSV file (Adjust path if needed)
    data_result_file = "Re10000" 
    df = pd.read_csv(data_result_file, header=0)
    
    # 2. Clean column names (remove whitespace)
    df.columns = df.columns.str.strip()

    # 3. Randomly select 'n' rows (n=500 as per your original code)
    n = 500
    selected_rows = df.sample(n=n, random_state=42)

    # 4. Extract Spatial Coordinates (x, y)
    # Note: Using .values to ensure we get numpy arrays directly
    x_val = selected_rows['x-coordinate'].values[:, None]
    y_val = selected_rows['y-coordinate'].values[:, None]
    x_data_np = np.hstack((x_val, y_val))

    # 5. Extract Physics Fields [u, v, p, k, eps]
    # Order MUST match the model output: u, v, p, k, eps
    u_data_np = selected_rows[[
        'x-velocity', 
        'y-velocity', 
        'pressure', 
        'turb-kinetic-energy', 
        'turb-diss-rate'
    ]].values

    # 6. Convert to JAX arrays (float32)
    x_data = jnp.array(x_data_np, dtype=jnp.float32)
    u_data = jnp.array(u_data_np, dtype=jnp.float32)

    print(f"Loaded {len(x_data)} data points from {data_result_file}")

    
    # Convert to JAX arrays
    inputs = RANSInputWrapper(
        col=jnp.array(x_col),
        inlet=jnp.array(x_inlet),
        noslip=jnp.array(x_noslip),
        outlet=jnp.array(x_outlet),
        data_x = x_data,
        data_u = u_data
    )

    # B. Define Model Architecture
    # Transform to enforce k, eps > 0
    def k_eps_positivity(x):
        # x is (..., 5) -> [u, v, p, k, eps]
        uvp = x[..., :3]
        ke = jnp.exp(x[..., 3:]) # Enforce positive
        return jnp.concatenate([uvp, ke], axis=-1)

    # Define Windows (using sigmoid logic from fbpinn.py)
    # Shifts and Scales must be arrays
    shifts = jnp.array([
        [6.0, 1.0], 
        [18.0, 1.0], 
        [30.0, 1.0]
    ])
    
    # Define bounds for 3 subdomains
    # Domain 1: x in [0, 12]
    # Domain 2: x in [12, 24]
    # Domain 3: x in [24, 35]
    xs_min = jnp.array([[0.0, 0.0],  [12.0, 0.0], [24.0, 0.0]])
    xs_max = jnp.array([[12.0, 2.0], [24.0, 2.0], [35.0, 2.0]])
    # Smoothness factors
    smins = jnp.ones_like(xs_min)
    smaxs = jnp.ones_like(xs_max)

    # Configure Layers: Input=2, Output=5
    layers = [2] + [96]*5 + [5]

    # Initialize FBPINN
    # We pass the custom output transform to the FBPINN -> Subnets
    # Note: You need to modify FBPINN in nn/fbpinn.py to pass kwargs to SubdomainNetwork
    # OR, we define the subnets manually here if FBPINN is too rigid.
    # Assuming FBPINN is modified or we monkey-patch.
    # Ideally, FBPINN should accept `subnet_cls` or `subnet_kwargs`.
    
    # For now, let's assume we modified SubdomainNetwork class variable or default
    # A cleaner way using Flax:
    model = FBPINN(
        layers=layers,
        shifts=shifts,
        xs_min=xs_min,
        xs_max=xs_max,
        smins=smins,
        smaxs=smaxs
    )
    
    # Hack to inject the transform into the SubdomainNetwork definition
    # In a real scenario, update FBPINN code to pass this down.
    from underPINN.nn.subdomain import SubdomainNetwork
    SubdomainNetwork.out_transform = staticmethod(k_eps_positivity)

    # C. Initialize
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 2)))

    # D. Setup Solver
    pde = KEpsilonPDE(model, Re=10000.0)

    schedule = optax.piecewise_constant_schedule(
        init_value=1e-3,
        boundaries_and_scales={2000: 0.5, 4000: 0.5} # decays at 2k and 4k
    )

    optimizer = optax.adam(learning_rate=schedule) #learning_rate=1e-3
    
    solver = RANSSolver(model, pde, optimizer)

    tracker = BenchmarkTracker()
    tracker.start()

    # E. Train
    print("Starting training...")
    final_params = solver.train(params, inputs, epochs=10, batch_size=2000)

    tracker.stop()
    tracker.log("epochs", 10)
    tracker.save(case_name="Turbulence", framework="JAX")

"""     # F. Prediction & Plotting
    # 1. Save Parameters
    save_params(final_params, "rans_params.msgpack")

    # 2. Global Plotting
    print("Generating Global Plots...")
    # Define plotting grid
    x = jnp.linspace(0, 35, 350)
    y = jnp.linspace(0, 1.9423, 50)
    XX, YY = jnp.meshgrid(x, y, indexing='ij')
    grid_flat = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    
    plot_global_predictions(model, final_params, grid_flat, shape=(350, 50))
    
    pred = model.apply(final_params, grid_flat) """
    
    # Save .npz
"""     np.savez("pinn_results.npz", 
             x=XX, y=YY, 
             u=pred[:,0].reshape(XX.shape),
             v=pred[:,1].reshape(XX.shape),
             p=pred[:,2].reshape(XX.shape),
             k=pred[:,3].reshape(XX.shape),
             eps=pred[:,4].reshape(XX.shape))
    print("Done.") """

if __name__ == "__main__":
    main()