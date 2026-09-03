# API Reference

Three abstract base classes form the backbone of underPINN. Every PDE, loss, and solver
conforms via inheritance — no rewrites required when adding a new physics case.

## Core abstractions

```{code-block} python
:caption: underPINN/core/base.py

class BasePDE(ABC):
    # Every PDE implements residual()
    @abstractmethod
    def residual(self, params, *args): ...

class BaseLoss(ABC):
    # Returns (total_loss, aux_tuple)
    @abstractmethod
    def __call__(self, params, *args): ...

class BaseSolver(ABC):
    @abstractmethod
    def init(self, key): ...
    @abstractmethod
    def train(self, *args, **kwargs): ...

    # Concrete helpers, available on every solver:
    def save_checkpoint(self, out_dir, stem="params", metadata=None): ...
    def restore_checkpoint(self, path): ...
    def load_params(self, params): ...  # transfer-learning warm-start
```

## PDE + Geometry convention

```{code-block} python
:caption: Every PDE

class BurgersPDE(BasePDE):
    def residual(self, params, x, t):
        # returns |u_t + u*u_x - nu*u_xx|
        ...
```

```{code-block} python
:caption: Every geometry

class Pipe:
    def sample_interior(self, n, key): ...
    def sample_wall(self, n, key): ...
    def sample_inlet(self, n, key): ...
    def sample_outlet(self, n, key): ...
```

## Package layout

```{list-table}
:header-rows: 1
:widths: 20 80

* - Package
  - Contents
* - `core/`
  - `BasePDE`, `BaseLoss`, `BaseSolver`, `TrainingConfig`
* - `nn/`
  - `MLP`, `GatedMLP`, `FourierMLP`, `FBPINN`, `HybridAttention`, `SimpleGate`,
    `FNO1D`, `FNO2D`, `DeepONet1D`, `CViT`, `factory.py` (single model-building path)
* - `pde/`
  - Burgers, Wave, Helmholtz, Heat, N-S 2-D/3-D (steady + unsteady), Carreau N-S,
    k-ε, Euler (2-D ramp + 1-D Sod/Toro3), ODE, operator-grid residuals
* - `solver/`
  - `FBPINNSolver`, `SteadySolver`, `ODESolver`, `LDCSolver`, `RANSSolver`,
    `OperatorSolver`, `DeepONetSolver`
* - `losses/`
  - `PINNLoss` (with RBA), `ODELoss`, `SteadyLoss`, `OperatorLoss`, `DeepONetLoss`
* - `callbacks/`
  - `ConsoleLogger`, `EarlyStopping`, `ModelCheckpoint`
* - `geometry/`
  - `Interval`, `Rectangle`, `NACAAirfoil`, `Cylinder2D`, `Pipe`, `BulgeGeometry`,
    `Ramp`, `Composite`, `ShapelyGeom`
* - `training/`
  - `rar_d_resample` (RAR-D adaptive collocation)
* - `config/`
  - `load_config`, `generate_sweep_configs`, `cfg_get`
* - `runner/`
  - `dispatch.py` path-registry + `importlib` loader; CLI dispatch only
* - `utils/`
  - `save_predictions`, `checkpoint`, `restart`, `ModelPredictor`, `timing`, `metrics`,
    `plotting`, `operator_datagen`
* - `postprocess/`
  - `plotting` (shared matplotlib style), `PulsatilePredictor`, operator plotting
    helpers
* - `benchmark_utils/`
  - `BenchmarkRunner`, evaluators, report generation
```

## Full repository tree

```text
underPINN/
├── core/
│   ├── base.py            # BasePDE, BaseLoss, BaseSolver (+ save/restore_checkpoint)
│   └── config.py          # TrainingConfig dataclass with validation
│
├── nn/
│   ├── mlp.py             # MLP, FourierMLP
│   ├── fbpinn.py          # FBPINN (domain-decomposed network)
│   ├── attention.py       # HybridAttention, SimpleGate
│   ├── embeddings.py      # Fourier / positional embeddings
│   ├── subdomain.py       # SubdomainNetwork
│   ├── operators.py       # FNO1D, FNO2D, DeepONet1D, CVit, cvit_grid_predict
│   └── factory.py         # build_model / network_config — single model-building path
│
├── pde/
│   ├── burgers.py             # 1-D Burgers equation
│   ├── burgers_grid.py        # BurgersGrid1D/2D — FD residual for FNO1D/FNO2D/CViT
│   ├── burgers_deeponet.py    # DeepONetBurgersPDE — autodiff residual for DeepONet
│   ├── navier_stokes_2d_grid.py # CylinderNSGrid — FD residual for the cylinder FNO2D
│   ├── diffusion.py           # 1-D unsteady diffusion / heat inverse
│   ├── heat.py                # 2-D steady heat (Poisson)
│   ├── heat2d_unsteady.py     # 2-D unsteady heat  (x, y, t) → u
│   ├── helmholtz.py           # 2-D Helmholtz  Δu + k²u = f
│   ├── wave.py                # 1-D wave equation  u_tt = c²u_xx
│   ├── navier_stokes.py       # 2-D steady incompressible N-S
│   ├── navier_stokes_3d.py    # 3-D steady + UNSTEADY incompressible N-S
│   ├── carreau_ns_3d.py       # 3-D Carreau (shear-thinning) N-S + 1-D exact profile
│   ├── compressible_euler.py  # 2-D steady Euler — conservative form + artificial viscosity
│   ├── euler_1d_unsteady.py   # 1-D unsteady Euler (Sod) — learnable artificial viscosity
│   ├── pipe_flow_unsteady.py  # Unsteady pipe cross-section  (y, z, t) → u
│   ├── k_epsilon.py           # RANS k-ε turbulence model
│   └── ode.py                 # Exponential decay, Harmonic oscillator
│
├── geometry/
│   ├── interval.py         # 1-D interval sampler
│   ├── rectangle.py        # 2-D rectangle sampler
│   ├── airfoil.py          # NACA 4-digit (sym/cambered) + AoA rotation + SDF sampling
│   ├── cylinder.py         # 2-D circular cylinder (cross-flow exterior)
│   ├── pipe.py             # Cylindrical pipe (interior, wall, inlet, outlet)
│   ├── aaa.py               # BulgeGeometry — axisymmetric AAA bulge R(x)
│   ├── ramp.py               # Trapezoidal ramp domain above a wedge (compressible Euler)
│   ├── composite.py          # Boolean combination of geometries
│   └── shapely_geom.py       # Shapely-backed arbitrary polygon sampler
│
├── solver/
│   ├── fbpinn.py           # FBPINNSolver  (space-time PDE, lax.scan, RAR-D)
│   ├── ode_solver.py       # ODESolver
│   ├── steady_solver.py    # SteadySolver  (no time dimension)
│   ├── ldc_solver.py       # LDCSolver     (lid-driven cavity / FBPINN)
│   ├── rans_solver.py      # RANSSolver    (k-ε turbulence)
│   └── operator.py         # OperatorSolver (FNO/CViT), DeepONetSolver
│
├── losses/
│   ├── loss.py              # PINNLoss  (with optional RBA)
│   ├── ode_loss.py          # ODELoss
│   ├── steady_loss.py       # SteadyLoss
│   └── operator_loss.py     # OperatorLoss (data+PDE+warmup+RBA), DeepONetLoss
│
├── callbacks/
│   ├── base.py               # Callback ABC
│   ├── logging.py            # ConsoleLogger
│   ├── early_stopping.py     # EarlyStopping
│   └── checkpoint.py         # ModelCheckpoint  (save best model during training)
│
├── runner/                   # CLI dispatch only — runner logic lives in examples/
│   ├── dispatch.py           # _REGISTRY: problem → (script path, fn name)
│   ├── pipe_flow.py          # pipe_flow runner helper
│   ├── wave.py               # wave runner helper
│   └── heat_forward.py       # heat_forward runner helper
│
├── training/
│   └── resample.py           # rar_d_resample  (RAR-D adaptive collocation)
│
├── config/
│   └── loader.py             # load_config, generate_sweep_configs, cfg_get
│
├── benchmark_utils/
│   ├── evaluators.py         # per-problem evaluators with exact solutions
│   ├── benchmark_suite.py    # BenchmarkResult, BenchmarkRunner
│   └── report.py             # plots, CSV, Markdown report generation
│
├── utils/
│   ├── io.py                 # save_predictions (NPZ archives)
│   ├── sampling.py           # safe_choice (replace-safe mini-batching)
│   ├── seed.py                # set_seed (Python + NumPy + JAX)
│   ├── checkpoint.py          # save_checkpoint, load_checkpoint, ModelPredictor
│   ├── restart.py             # RestartManager (snapshot + resume + done marker)
│   ├── timing.py              # fmt_train_time (JIT-aware training time reporting)
│   ├── metrics.py             # rel_l2, mse helpers
│   ├── plotting.py            # plot_losses, plot_ode_result
│   └── operator_datagen.py    # FD reference solvers + random ICs for operator examples
│
├── postprocess/
│   ├── plotting.py            # field / cbar / save_fig (shared matplotlib style)
│   ├── pulsatile.py           # PulsatilePredictor
│   └── operators.py           # plot_operator_loss, plot_prediction_1d/2d
│
└── __main__.py                # CLI entry point (python -m underPINN)
                                # sets XLA_PYTHON_CLIENT_PREALLOCATE=false before import jax
```

```{seealso}
{doc}`cli` documents the three-step recipe for registering a new problem in
`dispatch.py`.
```
