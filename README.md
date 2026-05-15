# underPINN

> A modular, GPU-accelerated Physics-Informed Neural Network framework built on JAX + Flax

![Static Badge](https://img.shields.io/badge/repo%20status-Active-95eb34) ![Static Badge](https://img.shields.io/badge/license-GPL--3.0-green)

underPINN is a research-grade PINN engine that combines classical collocation-based PINNs with Finite Basis decomposition (FBPINN), attention-augmented networks, residual-based adaptive weighting, transfer learning, and inverse problems — all JIT-compiled and differentiable via XLA.

---

## Features

- **Domain decomposition** — FBPINN with overlapping subdomains and sigmoid partition-of-unity windows
- **Attention networks** — Hybrid attention and gated residual blocks inside each subdomain
- **FourierMLP** — Trainable random Fourier feature embeddings for oscillatory solutions (Helmholtz, wave)
- **Residual-based adaptivity (RBA)** — Element-wise loss weighting that focuses training on high-residual regions
- **RAR-D adaptive resampling** — Periodically replaces collocation points proportionally to `|residual|^k`, focusing compute on high-error regions (Lu et al., 2021)
- **`lax.scan` training loop** — Fuses N gradient steps into a single XLA kernel, eliminating Python overhead between epochs (50–500× less dispatch overhead on GPU)
- **Transfer learning** — `load_params` warm-start for parameter transfer (different Re / ν) and temporal transfer (extended time horizon)
- **Inverse problems** — Joint optimisation of network weights + physics parameters (e.g. recover thermal diffusivity from sparse noisy observations)
- **Model checkpointing & inference** — Every runner auto-saves `params.msgpack` + `params_meta.json`; `load_checkpoint` / `ModelPredictor` reload a trained model in one line; `ModelCheckpoint` callback saves the best model during training
- **Callbacks** — `ConsoleLogger`, `EarlyStopping`, `ModelCheckpoint` — pluggable via `TrainingConfig`
- **12 CLI-registered runners** — `burgers`, `wave`, `pipe_flow`, `helmholtz`, `heat_forward`, `ode`, `ldc`, `airfoil`, `heat_inverse`, `burgers_transfer`, `pipe_flow_unsteady_transfer`, `inverse_diffusion`
- **3-D problems** — Full 3-D Navier-Stokes with double-`jacfwd` Hessians, cylindrical pipe geometry
- **Benchmark suite** — systematic accuracy vs. epoch budget comparisons across all examples with one command
- **YAML-driven experiments** — every hyperparameter lives in a config file; no code changes needed to sweep
- **GPU / multi-GPU** — Pure JAX/XLA; runs on CPU, single GPU, or multi-GPU with no code changes

---

## Running experiments

Each example folder is **self-contained** — script + YAML live together. Run a problem two ways:

```bash
# ── Option A: directly (script auto-finds its YAML) ──────────────────────
python examples/burgers/burgers.py
python examples/wave/wave.py
python examples/LDC/run_ldc.py
python examples/airfoil/airfoil_flow.py
python examples/heat/forward.py
python examples/heat/inverse.py           # heat inverse (recover α)
python examples/ode/ode_test.py
python examples/pipe_flow/pipe_flow.py
python examples/transfer/burgers_transfer.py
python examples/pipe_flow/pipe_flow_unsteady_transfer.py

# Pass a custom config as the first argument:
python examples/burgers/burgers.py my_custom.yaml

# ── Option B: via CLI (same YAML, dispatched dynamically) ────────────────
python -m underPINN run  examples/burgers/config.yaml
python -m underPINN run  examples/LDC/config.yaml
python -m underPINN run  examples/airfoil/config.yaml
python -m underPINN run  examples/heat/heat_inverse.yaml

# Hyperparameter sweep (Cartesian product)
python -m underPINN sweep examples/burgers/burgers_nu_sweep.yaml
python -m underPINN sweep examples/pipe_flow/pipe_flow_re_sweep.yaml

# Inspect a config (no training)
python -m underPINN show examples/wave/config.yaml

# List all 12 registered runners
python -m underPINN list
```

### Adding a new case

1. Create `examples/<mycase>/` with your script (containing `run_<mycase>(cfg)`) and a YAML config
2. Add one line to `underPINN/runner/dispatch.py`'s `_REGISTRY`:
   ```python
   "mycase": ("examples/mycase/mycase.py", "run_mycase"),
   ```
That's it — no other underPINN files need to change.

### Config file anatomy

```yaml
problem: burgers        # selects the runner  (one of 12 registered problems)

network:
  type  : mlp           # mlp | fourier_mlp
  layers: [2, 64, 64, 64, 1]

physics:
  nu: 0.01              # PDE parameters

data:
  T: 2.0
  n_collocation: 6000

training:
  epochs                  : 5000
  lr                      : 1.0e-3
  early_stopping_patience : 400   # omit to disable

loss:
  ic_weight: 100.0
  rba      : true

output:
  dir        : outputs/burgers   # predictions, loss, config, and model saved here
  save_params: true              # write params.msgpack + params_meta.json (default: true)
```

### Sweep file anatomy

```yaml
base:                           # shared config for all runs
  problem: burgers
  ...

sweep:                          # dot-separated key → list of values
  physics.nu       : [0.1, 0.05, 0.025, 0.01]
  training.epochs  : [3000, 5000]
```

Each run gets its own sub-directory (`outputs/…/run_000`, `run_001`, …) with a saved `config.yaml` for full reproducibility.

---

## Repository Structure

```
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
│   └── subdomain.py       # SubdomainNetwork
│
├── pde/
│   ├── burgers.py         # 1-D Burgers equation
│   ├── diffusion.py       # 1-D unsteady diffusion / heat inverse
│   ├── heat.py            # 2-D steady heat (Poisson)
│   ├── heat2d_unsteady.py # 2-D unsteady heat  (x, y, t) → u
│   ├── helmholtz.py       # 2-D Helmholtz  Δu + k²u = f
│   ├── wave.py            # 1-D wave equation  u_tt = c²u_xx
│   ├── navier_stokes.py   # 2-D steady incompressible N-S
│   ├── navier_stokes_3d.py# 3-D steady incompressible N-S
│   ├── pipe_flow_unsteady.py # Unsteady pipe cross-section  (y, z, t) → u
│   ├── k_epsilon.py       # RANS k-ε turbulence model
│   └── ode.py             # Exponential decay, Harmonic oscillator
│
├── geometry/
│   ├── interval.py        # 1-D interval sampler
│   ├── rectangle.py       # 2-D rectangle sampler
│   ├── airfoil.py         # NACA 4-digit profile + exterior/surface sampling
│   ├── pipe.py            # Cylindrical pipe (interior, wall, inlet, outlet)
│   ├── composite.py       # Boolean combination of geometries
│   └── shapely_geom.py    # Shapely-backed arbitrary polygon sampler
│
├── solver/
│   ├── fbpinn.py          # FBPINNSolver  (space-time PDE, lax.scan, RAR-D)
│   ├── ode_solver.py      # ODESolver
│   ├── steady_solver.py   # SteadySolver  (no time dimension)
│   ├── ldc_solver.py      # LDCSolver     (lid-driven cavity / FBPINN)
│   └── rans_solver.py     # RANSSolver    (k-ε turbulence)
│
├── losses/
│   ├── loss.py            # PINNLoss  (with optional RBA)
│   ├── ode_loss.py        # ODELoss
│   └── steady_loss.py     # SteadyLoss
│
├── callbacks/
│   ├── base.py            # Callback ABC
│   ├── logging.py         # ConsoleLogger
│   ├── early_stopping.py  # EarlyStopping
│   └── checkpoint.py      # ModelCheckpoint  (save best model during training)
│
├── runner/                # CLI dispatch only — runner logic lives in examples/
│   └── dispatch.py        # _REGISTRY: problem → (script path, fn name)
│                          # _load_runner: importlib dynamic loader
│                          # Adding a case = 1 line in _REGISTRY, no other changes
│
├── training/
│   └── resample.py        # rar_d_resample  (RAR-D adaptive collocation)
│
├── config/
│   └── loader.py          # load_config, generate_sweep_configs, cfg_get
│
├── benchmark_utils/
│   ├── evaluators.py      # per-problem evaluators with exact solutions
│   ├── benchmark_suite.py # BenchmarkResult, BenchmarkRunner
│   └── report.py          # plots, CSV, Markdown report generation
│
├── utils/
│   ├── io.py              # save_predictions (NPZ archives)
│   ├── sampling.py        # safe_choice (replace-safe mini-batching)
│   ├── seed.py            # set_seed (Python + NumPy + JAX)
│   ├── checkpoint.py      # save_checkpoint, load_checkpoint, ModelPredictor
│   ├── metrics.py         # rel_l2, mse helpers
│   └── plotting.py        # plot_losses, plot_ode_result
│
└── __main__.py            # CLI entry point (python -m underPINN)

examples/                  # self-contained: each folder holds script + YAML
│                          # Adding a new case = create folder + add 1 line to dispatch.py
├── burgers/               burgers.py  +  config.yaml            (1-D Burgers FBPINN + RBA)
├── wave/                  wave.py     +  config.yaml            (1-D wave FourierMLP)
├── heat/                  forward.py  +  heat_forward.yaml      (2-D steady heat / Poisson)
│                          inverse.py  +  heat_inverse.yaml      (recover α from noisy data)
├── helmholtz/             helmholtz.py + config.yaml            (2-D Helmholtz FourierMLP)
├── ode/                   ode_test.py +  config.yaml            (exp decay + harmonic osc.)
├── inverse/               inverse_diffusion.py + config.yaml    (2-D diffusion inverse)
├── LDC/                   run_ldc.py  +  config.yaml            (2-D Lid-Driven Cavity Re=100)
├── K-Epsilon/             (k-ε RANS turbulent channel)
├── airfoil/               airfoil_flow.py + config.yaml         (NACA 0012 Re=200)
├── pipe_flow/             pipe_flow.py + pipe_flow.yaml         (3-D Hagen-Poiseuille)
│                          pipe_flow_unsteady_transfer.py + yaml  (Re + temporal transfer)
└── transfer/              burgers_transfer.py + yaml            (Burgers param + temp. TL)

docs/
└── index.html             # Static framework documentation website
```

---

## Examples

| Problem | PDE | Highlights | Config |
|---|---|---|---|
| Exponential Decay | du/dt + λu = 0 | `ODESolver`, `TrainingConfig`, callbacks | `examples/ode/config.yaml` |
| Harmonic Oscillator | d²u/dt² + ω²u = 0 | `ODESolver`, IC derivative | `examples/ode/config.yaml` |
| 1-D Burgers | u_t + uu_x = νu_xx | FBPINN, RBA, cosine LR | `examples/burgers/config.yaml` |
| 1-D Heat — Forward | u_t = αu_xx | `FBPINNSolver`, exact Gaussian | `examples/heat/config.yaml` |
| 1-D Heat — Inverse | u_t = αu_xx | Recover α from noisy observations | `examples/heat/config.yaml` |
| 1-D Wave | u_tt = c²u_xx | `FourierMLP`, dual IC (u and u_t) | `examples/wave/config.yaml` |
| 2-D Helmholtz | Δu + k²u = f | `FourierMLP`, k = 4, manufactured source | `examples/helmholtz/config.yaml` |
| 2-D Diffusion Inverse | u_t = α∇²u | Log-param joint optimisation | `examples/inverse/config.yaml` |
| 2-D Lid-Driven Cavity | Steady N-S | FBPINN + `SimpleGate` attention, Re=100 | `examples/LDC/config.yaml` |
| 2-D Turbulent Channel | RANS k-ε | k-ε PDE, `RANSSolver`, Re=10000 | `examples/K-Epsilon/config.yaml` |
| Steady NACA 0012 Airfoil | Steady N-S | Exterior geometry, Cp curve, CL estimate | `examples/airfoil/config.yaml` |
| 3-D Hagen-Poiseuille | Steady 3-D N-S | Double-jacfwd Hessian, `Pipe` geometry | `examples/pipe_flow/pipe_flow.yaml` |
| 3-D Unsteady Pipe — Transfer | u_t = G + ν∇²u | Re-transfer + temporal transfer | `examples/pipe_flow/pipe_flow_unsteady_transfer.yaml` |
| 1-D Burgers Transfer | Burgers | Parameter transfer (ν) + temporal transfer | `examples/transfer/burgers_transfer.yaml` |

---

## Framework Design

### Core abstractions

```python
# Every PDE implements one method
class BasePDE(ABC):
    @abstractmethod
    def residual(self, params, *args): ...

# Every loss is callable and returns (total, aux_tuple)
class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, params, *args, **kwargs): ...

# Every solver has init + train + checkpoint helpers (inherited)
class BaseSolver(ABC):
    @abstractmethod
    def init(self, key): ...
    @abstractmethod
    def train(self, *args, **kwargs): ...

    # Concrete — available on every solver:
    def save_checkpoint(self, out_dir, stem="params", metadata=None): ...
    def restore_checkpoint(self, path): ...
```

### TrainingConfig + callbacks

```python
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.callbacks.checkpoint import ModelCheckpoint
import optax

config = TrainingConfig(
    epochs      = 5000,
    lr          = 1e-3,
    lr_schedule = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    batch_r     = 2048,
    log_every   = 500,
    callbacks   = [
        ConsoleLogger(log_every=500),
        EarlyStopping(patience=400),
        ModelCheckpoint("outputs/burgers/", monitor="loss"),  # saves best model
    ],
)
solver.train(*data, config=config)
```

### Model saving & inference

Every runner writes two files to the output directory after training:

```
outputs/burgers/
  params.msgpack       ← exact Flax/msgpack serialization of all weights
  params_meta.json     ← {"problem": "burgers", "network": {"type": "mlp", "layers": [...]}, ...}
  predictions.npz      ← collocation-point predictions
  config.yaml          ← resolved training config (reproducibility)
  loss_hist.npy
  loss.png
```

**Reload and predict on new inputs:**

```python
from underPINN.utils.checkpoint import ModelPredictor
import jax.numpy as jnp

# Option A — auto-build model from saved metadata (zero boilerplate)
predictor = ModelPredictor.from_meta("outputs/burgers/")

# Option B — provide model explicitly
from underPINN.nn.mlp import MLP
predictor = ModelPredictor.from_checkpoint(
    MLP(layers=[2, 64, 64, 64, 1]),
    "outputs/burgers/",
)

# Run inference
x_new = jnp.linspace(-1.0, 1.0, 500)
t_new = jnp.full(500, 0.8)
u = predictor.predict(jnp.stack([x_new, t_new], axis=1))
```

**Lower-level API:**

```python
from underPINN.utils.checkpoint import save_checkpoint, load_checkpoint

# Save any param pytree
save_checkpoint(params, "my_dir/", metadata={"problem": "wave", "network": {"layers": [...]}})

# Load (model used as template for structure)
params = load_checkpoint(model, "my_dir/")
```

### Transfer learning

```python
# Train source model
solver_src.train(*data_src, config=cfg_src)

# Save source checkpoint
solver_src.save_checkpoint("outputs/source/")

# Warm-start target from source weights, then fine-tune
solver_tgt.load_params(solver_src.params)    # or: solver_tgt.restore_checkpoint("outputs/source/")
solver_tgt.train(*data_tgt, config=cfg_tgt)  # lower lr recommended
```

---

## Benchmark Suite

```bash
# Run all fast problems with default epoch budgets [500, 1000, 2000, 5000]
python -m underPINN bench

# Select specific problems and budgets
python -m underPINN bench \
    --problems burgers wave helmholtz heat_steady ode_exp ode_harmonic \
    --epochs 500 1000 2000 5000 \
    --output outputs/bench

# Include slow problems (3-D pipe flow)
python -m underPINN bench --all

# Regenerate plots from a previous run without re-training
python -m underPINN bench --from-json outputs/bench/results.json
```

### Outputs written to `outputs/bench/`

| File | Description |
|---|---|
| `accuracy_vs_epochs.png` | Log-log rel-L² vs epoch budget, one line per problem |
| `accuracy_summary_bar.png` | Grouped bar chart of rel-L² at each epoch budget |
| `wall_time_vs_epochs.png` | Training time vs epoch budget |
| `ms_per_epoch.png` | Bar chart of training throughput per problem |
| `loss_grid.png` | Convergence curves for each problem |
| `benchmark_results.csv` | Full raw data table |
| `benchmark_summary.md` | Markdown table (one row per problem at max epochs) |
| `results.json` | Reusable JSON for `--from-json` replays |

---

## Performance

### `lax.scan` — eliminating Python loop overhead

```python
config = TrainingConfig(
    epochs       = 5000,
    lr           = 1e-3,
    n_scan_steps = 100,   # 50 outer Python calls instead of 5000
    callbacks    = [ConsoleLogger(log_every=500)],
)
solver.train(*data, config=config)
```

| `n_scan_steps` | Python calls | Callback granularity |
|:-:|:-:|:-:|
| 1 (default) | 5000 | every epoch |
| 100 | 50 | every 100 epochs |
| 500 | 10 | every 500 epochs |

### RAR-D — adaptive collocation resampling

```python
config = TrainingConfig(
    epochs              = 5000,
    n_scan_steps        = 100,
    resample_period     = 5,    # resample every 5 outer steps (= 500 epochs)
    resample_k          = 1.0,  # p ∝ |residual|^k
)
solver.train(*data, config=config)
```

---

## Quick Start

### Install

```bash
# CPU (development / testing)
pip install jax flax optax matplotlib scipy

# GPU (CUDA 12)
pip install -U "jax[cuda12]" flax optax matplotlib scipy
```

### Run an example via CLI

```bash
# edit examples/burgers/config.yaml, then:
python -m underPINN run examples/burgers/config.yaml
```

### Programmatic usage

```python
import jax, optax
import jax.numpy as jnp
from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
from underPINN.callbacks.checkpoint import ModelCheckpoint

model  = MLP(layers=[2, 64, 64, 64, 1])
pde    = BurgersPDE(model, nu=0.01)
loss   = PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0, rba=True)
solver = FBPINNSolver(model, pde, loss=loss)
solver.init(jax.random.PRNGKey(0))

config = TrainingConfig(
    epochs      = 5000,
    lr          = 1e-3,
    lr_schedule = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    callbacks   = [
        ConsoleLogger(log_every=500),
        EarlyStopping(patience=400),
        ModelCheckpoint("outputs/burgers/", monitor="loss"),
    ],
)
solver.train(*data, config=config)

# Checkpoint is saved automatically; reload for inference:
from underPINN.utils.checkpoint import ModelPredictor
predictor = ModelPredictor.from_meta("outputs/burgers/")
u = predictor.predict(jnp.stack([x_test, t_test], axis=1))
```

### 3-D pipe flow

```python
from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.pipe import Pipe

pipe  = Pipe(R=0.5, L=2.0)
model = MLP(layers=[3, 64, 64, 64, 64, 4])   # (x,y,z) → (u,v,w,p)
pde   = SteadyNS3DPDE(model, Re=10.0)

xyz = jnp.array(pipe.sample_interior(1000))
u_exact, v_exact, w_exact, p_exact = pde.exact_poiseuille(xyz, R=0.5, U_max=1.0, L=2.0)
```

### FourierMLP for oscillatory PDEs

```python
from underPINN.nn.mlp import FourierMLP

# Recommended for Helmholtz, wave, and high-frequency solutions
model = FourierMLP(layers=[2, 128, 128, 128, 1], n_fourier=32, sigma=4.0)
```

---

## Cite underPINN

If you use underPINN in research or publications, please cite:

```bibtex
@software{underPINN,
  author = {},
  title  = {underPINN: A Modular JAX Framework for Physics-Informed Neural Networks},
  year   = {},
  note   = {https://github.com/Prashantiitk23/PINN}
}
```

## License

underPINN is released under the GPL-3.0 License. See [LICENSE.txt](https://github.com/lohithgsk/PINN/blob/main/LICENSE) for the full text.
