# underPINN-v2605
> A modular, GPU-accelerated Physics-Informed Neural Network framework built on JAX + Flax + Optax

![Static Badge](https://img.shields.io/badge/version-v2605-blue) ![Static Badge](https://img.shields.io/badge/repo%20status-Active-95eb34) ![Static Badge](https://img.shields.io/badge/license-GPL--3.0-green) ![Static Badge](https://img.shields.io/badge/python-%3E%3D3.9-blue) ![Static Badge](https://img.shields.io/badge/jax-%3E%3D0.4.26-orange)

underPINN is a research-grade PINN engine that combines classical collocation-based PINNs with Finite Basis decomposition (FBPINN), attention-augmented networks, residual-based adaptive weighting, transfer learning, inverse problems, and a full restart/resume system — all JIT-compiled and differentiable via XLA.

---

## Features

### Network Architectures
- **MLP** — standard multi-layer perceptron with tanh activations; configurable depth and width via a layer list
- **FourierMLP** — random Fourier feature embeddings (trainable σ) prepended to a standard MLP; essential for oscillatory solutions (Helmholtz, wave, high-Re flows) where plain MLPs fail to represent high spatial frequencies
- **FBPINN** — overlapping subdomain decomposition with sigmoid partition-of-unity windows; each subdomain gets its own network so training is never dominated by one region
- **HybridAttention + SimpleGate** — gated residual blocks inside each FBPINN subdomain; SimpleGate multiplies the hidden state element-wise by a learnable gate for compact, expressive feature modulation

### Training
- **`lax.scan` fused kernels** — fuse N gradient steps into a single XLA kernel, eliminating Python dispatch between epochs; delivers 50–500× less overhead on GPU compared to a Python for-loop
- **Cosine LR decay** — via `optax.cosine_decay_schedule`; integrates seamlessly with `TrainingConfig`
- **RAR-D adaptive collocation resampling** — periodically replaces a fraction of collocation points with samples drawn proportional to `|residual|^k` (Lu et al., 2021); focuses compute on high-error regions without changing the total batch size
- **RBA element-wise loss weighting** — residual-based adaptivity assigns per-point weights so that boundary and collocation losses are automatically balanced during training
- **EarlyStopping** — monitors a metric (default: total loss) and halts training after `patience` epochs without improvement
- **`TrainingConfig` dataclass** — centralises all hyperparameters with runtime validation; a single object is passed to every solver
- **Callbacks** — `ConsoleLogger` (prints loss every N epochs), `EarlyStopping` (halts on plateau), `ModelCheckpoint` (saves best model during training); all callbacks fire correctly even inside `lax.scan` loops

### Restart / Resume
- **`RestartManager`** saves `params.msgpack`, `opt_state.msgpack`, loss histories, and a `meta.json` to `<out_dir>/restart/` every `save_restart_every` epochs
- On re-run with the same config file, training resumes exactly from the last snapshot — epoch counter, optimizer state, and loss histories are all restored
- **Config-hash check** — the MD5 of the JSON-serialised YAML is stored in `meta.json`; if _any_ field changes (epochs, lr, layers, …), the hash differs and a fresh run starts automatically; it is safe to leave `save_restart_every` on permanently
- **`done()` marker** — once training completes normally or via early stopping, the snapshot is marked `"done": true`; the next run with the same config starts fresh instead of re-resuming a completed run

### GPU Memory
- JAX's XLA BFC allocator pre-reserves ~90% of all free VRAM the moment `import jax` executes; on an 80 GB A100 this shows as ~73 GB reserved even for a 3-layer MLP
- **underPINN sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` automatically** — in `underPINN/__main__.py` for CLI runs and at the top of every example script for direct `python examples/…` runs — so on-demand GPU memory growth is the default behaviour out of the box

### PDE Library
- 1-D Burgers equation (`u_t + uu_x = νu_xx`)
- 1-D / 2-D diffusion / heat — forward and inverse (recover thermal diffusivity α)
- 1-D wave equation (`u_tt = c²u_xx`)
- 2-D Helmholtz (`Δu + k²u = f`, manufactured source)
- 2-D steady incompressible Navier-Stokes (lid-driven cavity, Re=100)
- 2-D RANS k-ε turbulence model (turbulent channel, Re=10 000)
- 3-D steady incompressible Navier-Stokes (Hagen-Poiseuille pipe flow)
- 2-D steady compressible Euler (oblique-shock ramp, Mach 3, θ=10°)
- Unsteady pipe cross-section (`(y, z, t) → u`)
- Harmonic oscillator (`d²u/dt² + ω²u = 0`)
- Exponential decay ODE (`du/dt + λu = 0`)

### Geometry
- **Interval** — 1-D uniform / stratified sampler
- **Rectangle** — 2-D interior + boundary-aware sampling
- **NACA 4-digit airfoil** — exterior domain, near-surface, and farfield boundary sampling; exact profile coordinates
- **Cylindrical Pipe** — interior, wall, inlet, and outlet face samplers
- **Ramp** — trapezoidal domain above a wedge surface for compressible shock problems
- **Composite** — boolean combinations of any geometry objects
- **Shapely-backed polygon** — arbitrary 2-D polygon sampler backed by Shapely 2.x

### Solvers
- **`FBPINNSolver`** — space-time PDE training with `lax.scan`, RAR-D, and RestartManager integration
- **`ODESolver`** — lightweight ODE training with callbacks and checkpointing
- **`SteadySolver`** — stationary (no time dimension) PDE training
- **`LDCSolver`** — lid-driven cavity / FBPINN variant
- **`RANSSolver`** — k-ε turbulence model with RBA loss weighting

### Checkpointing & Inference
- Every runner saves `params.msgpack` + `params_meta.json` to the output directory after training
- `ModelPredictor.from_meta(path)` rebuilds the exact model architecture from the JSON sidecar and loads weights — zero boilerplate, no need to re-specify layers
- `ModelCheckpoint` callback saves the best model (by monitored metric) during training

### Transfer Learning
- **Parameter transfer** — warm-start from a trained model when changing ν, Re, or diffusivity; converges 2–3× faster than training from scratch
- **Temporal transfer** — extend the time horizon by fine-tuning on a new time interval starting from a previously trained checkpoint
- Both modes use `solver.load_params(src_params)` or `solver.restore_checkpoint(path)`

### Inverse Problems
- Joint optimisation of network weights + physics parameters (e.g. recover thermal diffusivity α from 50 sparse noisy observations)
- Log-parameterisation (`log_alpha = log(α)`) ensures positivity without constraints
- Gradient flows simultaneously through the PDE residual and the observation loss

### Benchmark Suite
- One command (`python -m underPINN bench`) runs all registered problems across multiple epoch budgets
- Outputs include PNG accuracy plots, convergence grids, CSV tables, wall-time charts, and a Markdown summary report
- `--from-json` replays plotting from a previous JSON result without re-training

### CLI
- **`run`** — run a single problem from a YAML config
- **`sweep`** — Cartesian product hyperparameter sweep; each combination gets its own sub-directory
- **`bench`** — full benchmark suite
- **`list`** — list all registered runners
- **`show`** — print the resolved config without training
- **`version`** — print the framework version

### Versioning
Calendar versioning (CalVer YYMM) — May 2026 → **underPINN-v2605**

---

## Installation

```bash
# CPU / development
pip install jax flax optax matplotlib scipy shapely pandas pyyaml
```

```bash
# GPU (CUDA 12)
pip install -U "jax[cuda12]" && pip install -r requirements-gpu.txt
```

```bash
# From source (editable install — recommended)
git clone https://github.com/Prashantiitk23/underPINN-v2605
cd underPINN-v2605
pip install -e .
```

### Verify GPU is visible

```bash
python -c "import jax; print(jax.devices())"
# Expected on GPU: [CudaDevice(id=0)]
```

---

## GPU Memory Management

### Why does `nvidia-smi` show 73 GB immediately after import?

JAX's XLA BFC (Best-Fit with Coalescing) allocator pre-reserves approximately 90% of all free VRAM the moment `import jax` executes — before any tensor is created — to avoid memory fragmentation during training. On an 80 GB A100 this appears as ~73 GB reserved even for a tiny 3-layer MLP that actually uses only ~200 MB of active arrays.

This is a deliberate XLA design choice: by owning the memory pool upfront, it can coalesce and reuse buffers without ever calling `cudaMalloc` again during training. The downside is that two JAX processes cannot share a GPU gracefully unless you set explicit limits.

### underPINN disables this automatically

The environment variable is set in `underPINN/__main__.py` (for CLI runs) and at the top of every example script (for direct `python examples/…` runs) **before** `import jax`, so you get on-demand allocation out of the box:

```python
# This is already done for you — shown here for transparency
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # now allocates only what it actually needs
```

### Manual control

```bash
# On-demand growth (default in underPINN) — frees all unreserved VRAM for other jobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Hard cap — useful when sharing a node; limits to e.g. 20% of VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20

# Platform allocator — no XLA pool at all (slowest, minimal fragmentation)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Multi-GPU: restrict to a single device (e.g. GPU 1)
export CUDA_VISIBLE_DEVICES=1
```

### Programmatic override (must be BEFORE `import jax`)

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
import jax  # now uses at most 15% of VRAM
```

### Typical actual VRAM usage per problem (with preallocation disabled)

| Problem | Network | VRAM (approx) |
|---|---|---|
| Burgers 1-D | [2,64,64,64,1] | ~200 MB |
| Wave 1-D | FourierMLP | ~300 MB |
| Helmholtz 2-D | FourierMLP | ~400 MB |
| LDC 2-D | FBPINN | ~800 MB |
| Airfoil 2-D | [2,128,128,128,3] | ~1.2 GB |
| Pipe Flow 3-D | [3,64,64,64,64,4] | ~2.0 GB |
| Compressible Ramp | [2,80,80,80,80,80,4] | ~1.8 GB |
| k-ε Turbulence | FBPINN | ~3.0 GB |

---

## Quick Start

### CLI (zero Python)

```bash
python -m underPINN run  examples/burgers/config.yaml
python -m underPINN run  examples/wave/config.yaml
python -m underPINN run  examples/pipe_flow/pipe_flow.yaml
python -m underPINN run  examples/ramp/config.yaml
```

### Programmatic

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, optax, jax.numpy as jnp
from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping

model  = MLP(layers=[2, 64, 64, 64, 1])
pde    = BurgersPDE(model, nu=0.01)
loss   = PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0, rba=True)
solver = FBPINNSolver(model, pde, loss=loss)
solver.init(jax.random.PRNGKey(0))

config = TrainingConfig(
    epochs      = 5000,
    lr          = 1e-3,
    lr_schedule = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    batch_r     = 2048,
    log_every   = 500,
    out_dir     = "outputs/burgers",   # enables auto-restart
    save_restart_every = 500,
    callbacks   = [
        ConsoleLogger(log_every=500),
        EarlyStopping(patience=400),
    ],
)
solver.train(*data, config=config)
```

---

## Running Experiments

### Direct script

Each example folder is **self-contained** — script + YAML live together. Run any problem directly:

```bash
python examples/burgers/burgers.py
python examples/wave/wave.py
python examples/helmholtz/helmholtz.py
python examples/heat/forward.py
python examples/heat/inverse.py
python examples/LDC/run_ldc.py
python examples/airfoil/airfoil_flow.py
python examples/ode/ode_test.py
python examples/pipe_flow/pipe_flow.py
python examples/pipe_flow/pipe_flow_unsteady_transfer.py
python examples/transfer/burgers_transfer.py
python examples/inverse/inverse_diffusion.py

# Pass a custom config as the first argument:
python examples/burgers/burgers.py my_custom.yaml
```

### CLI commands

```bash
# Single run
python -m underPINN run   examples/burgers/config.yaml

# Hyperparameter sweep (Cartesian product)
python -m underPINN sweep examples/burgers/burgers_nu_sweep.yaml
python -m underPINN sweep examples/pipe_flow/pipe_flow_re_sweep.yaml

# Benchmark all problems
python -m underPINN bench
python -m underPINN bench --problems burgers wave helmholtz --epochs 500 1000 2000 5000
python -m underPINN bench --all
python -m underPINN bench --from-json outputs/bench/results.json

# Utilities
python -m underPINN list                          # list all registered runners
python -m underPINN show examples/wave/config.yaml   # inspect resolved config
python -m underPINN version                       # print version string
```

### Config anatomy

```yaml
problem: burgers        # selects the runner (one of the registered problems)

network:
  type  : mlp           # mlp | fourier_mlp
  layers: [2, 64, 64, 64, 1]

physics:
  nu: 0.01              # PDE parameters (problem-specific)

data:
  T: 2.0                # time horizon
  n_collocation: 6000   # interior collocation points
  n_ic: 200             # initial-condition points
  n_bc: 200             # boundary-condition points

training:
  epochs                  : 5000
  lr                      : 1.0e-3
  early_stopping_patience : 400    # omit to disable
  save_restart_every      : 500    # snapshot every 500 epochs (0 to disable)

loss:
  ic_weight: 100.0       # weight on IC loss term
  bc_weight: 10.0        # weight on BC loss term
  rba      : true        # residual-based adaptivity

output:
  dir        : outputs/burgers   # predictions, loss, config, model saved here
  save_params: true              # write params.msgpack + params_meta.json
```

### Sweep anatomy

```yaml
base:                           # shared config for all runs
  problem: burgers
  network:
    type: mlp
    layers: [2, 64, 64, 64, 1]
  training:
    epochs: 5000

sweep:                          # dot-separated key → list of values
  physics.nu       : [0.1, 0.05, 0.025, 0.01]
  training.epochs  : [3000, 5000]
```

Each run gets its own sub-directory (`outputs/…/run_000`, `run_001`, …) with a saved `config.yaml` for full reproducibility.

### Adding a new case

```
1. Create examples/<mycase>/mycase.py  — define run_mycase(cfg) -> dict
2. Create examples/<mycase>/config.yaml  — set problem: mycase
3. Add ONE line to underPINN/runner/dispatch.py:
   "mycase": ("examples/mycase/mycase.py", "run_mycase"),
```

No other files need to change.

---

## Training System

### TrainingConfig — full field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | 1000 | Total training epochs |
| `lr` | float | 1e-3 | Base learning rate |
| `lr_schedule` | optax schedule | None | Overrides `lr` when set; use `optax.cosine_decay_schedule` |
| `batch_r` | int | 4096 | Collocation mini-batch size |
| `batch_i` | int | 512 | Initial-condition mini-batch |
| `batch_b` | int | 512 | Boundary-condition mini-batch |
| `log_every` | int | 100 | Print interval (used by ConsoleLogger) |
| `seed` | int | 0 | PRNG seed |
| `callbacks` | list | [] | List of Callback objects |
| `n_scan_steps` | int | 1 | Fuse N steps into one XLA kernel (1 = Python loop) |
| `resample_period` | int | 0 | RAR-D resampling every N outer steps (0 = off) |
| `resample_candidates` | int | 0 | Candidate pool size (0 → 5 × batch_r) |
| `resample_k` | float | 1.0 | Exponent in p ∝ \|residual\|^k |
| `out_dir` | str | "" | Output directory; enables auto-restart when non-empty |
| `save_restart_every` | int | 500 | Snapshot interval in epochs (0 = off) |

### Callbacks

**ConsoleLogger**
```python
ConsoleLogger(log_every=500)
# Prints: [epoch / total]  loss=X.XXe-04  pde=X.XXe-04  ic=X.XXe-03 ...
```

**EarlyStopping**
```python
EarlyStopping(patience=400, monitor="loss", min_delta=1e-8)
# Raises StopIteration (caught by the solver) after `patience` epochs without improvement.
# Works correctly inside lax.scan loops — fires at the outer-step boundary.
```

**ModelCheckpoint**
```python
ModelCheckpoint(
    out_dir="outputs/burgers/",
    monitor="loss",      # metric key from the loss aux dict
    mode="min",          # "min" or "max"
    save_best_only=True, # skip non-improving epochs
    metadata={"problem": "burgers", "network": {"type": "mlp", "layers": [2,64,64,64,1]}},
)
# Writes params.msgpack + params_meta.json whenever a new best is reached.
```

### `lax.scan` acceleration

Instead of a Python `for` loop that calls back into Python every epoch, `lax.scan` unrolls N gradient steps into a single compiled XLA program. The Python interpreter only touches the computation once per `n_scan_steps` iterations, dramatically reducing dispatch overhead:

```python
config = TrainingConfig(
    epochs       = 5000,
    lr           = 1e-3,
    n_scan_steps = 100,   # 50 outer Python calls instead of 5000
    callbacks    = [ConsoleLogger(log_every=500)],
)
solver.train(*data, config=config)
```

| `n_scan_steps` | Python calls / 5 000 epochs | Callback granularity | Use case |
|:-:|:-:|:-:|:-:|
| 1 (default) | 5 000 | every epoch | Development / debugging |
| 100 | 50 | every 100 epochs | GPU training, medium runs |
| 500 | 10 | every 500 epochs | Long GPU runs, production |

### RAR-D adaptive collocation resampling

At every `resample_period` outer steps, the solver:
1. Evaluates the PDE residual `r(x)` at a pool of `resample_candidates` candidate points
2. Computes sampling probabilities `p(x) ∝ |r(x)|^k`
3. Replaces the lowest-residual collocation points with new draws from this distribution

This concentrates compute on high-error regions without changing total batch size or requiring any geometry change.

```yaml
training:
  n_scan_steps    : 100
  resample_period : 5      # every 5 outer steps = every 500 epochs
  resample_k      : 1.0    # linear in |residual|
```

---

## Restart / Resume System

The restart system lets you safely interrupt and resume any training run without losing progress. It is fully automatic — just set `save_restart_every` in your config.

### How it works

1. Every `save_restart_every` epochs, a snapshot is written to `<out_dir>/restart/`:
   - `params.msgpack` — Flax-serialised model parameters at that epoch
   - `opt_state.msgpack` — Flax-serialised optimizer state (Adam moments, step count)
   - `hists.npz` — all loss history arrays accumulated so far (`loss_hist`, `pde_hist`, etc.)
   - `meta.json` — `{"epoch": N, "cfg_hash": "...", "done": false}`

2. On re-run with the same config file, `RestartManager` detects the snapshot directory, verifies the config MD5 hash, and resumes training from the saved epoch. The loss histories are stitched together so plots are continuous.

3. If the YAML config changed between runs (different `lr`, `epochs`, `layers`, or any field), the stored hash differs from the current config's hash → the snapshot is silently ignored and a fresh run starts.

4. After training finishes — either normally or via early stopping — `done()` marks the snapshot with `"done": true`. The next run with the same config file starts fresh rather than re-resuming a completed run.

### YAML config (the only change needed)

```yaml
training:
  save_restart_every: 500   # snapshot every 500 epochs; 0 to disable
```

### TrainingConfig (FBPINNSolver / ODESolver)

```python
config = TrainingConfig(
    epochs             = 10000,
    out_dir            = "outputs/burgers",
    save_restart_every = 500,
)
solver.train(*data, config=config)
```

That's all. If the process is killed at epoch 3 700, the next run resumes from epoch 3 500 (the last snapshot) automatically.

### Snapshot contents

| File | Contents |
|---|---|
| `params.msgpack` | Flax-serialised model parameters |
| `opt_state.msgpack` | Flax-serialised optimizer state |
| `hists.npz` | Loss history arrays (`loss_hist`, `pde_hist`, etc.) |
| `meta.json` | `{"epoch": N, "cfg_hash": "...", "done": false}` |

### Config change detection

The config hash is the MD5 of the JSON-serialised YAML after loading. It covers every field — `epochs`, `lr`, `layers`, physics constants, loss weights. Changing any value produces a different hash → clean start. This means you can keep `save_restart_every` enabled permanently: sweeps and updated configs always get fresh runs, while interrupted runs of the same config always resume.

---

## Model Checkpointing & Inference

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

### Save during training (ModelCheckpoint callback)

```python
from underPINN.callbacks.checkpoint import ModelCheckpoint

ModelCheckpoint(
    out_dir="outputs/burgers/",
    monitor="loss",
    mode="min",
    save_best_only=True,
    metadata={"problem": "burgers", "network": {"type": "mlp", "layers": [2, 64, 64, 64, 1]}},
)
```

### Reload and predict on new inputs

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

### Lower-level API

```python
from underPINN.utils.checkpoint import save_checkpoint, load_checkpoint

# Save any param pytree
save_checkpoint(params, "my_dir/", metadata={"problem": "wave", "network": {"layers": [...]}})

# Load (model used as template for structure)
params = load_checkpoint(model, "my_dir/")
```

---

## Transfer Learning

underPINN supports two transfer learning modes, both using the same warm-start API.

### Parameter transfer (different ν / Re / diffusivity)

```python
# Phase 1: train source model (e.g. Burgers ν=0.1)
solver_src.train(*data_src, config=cfg_src)
solver_src.save_checkpoint("outputs/source/")

# Phase 2: warm-start target from source weights, then fine-tune (e.g. ν=0.01)
solver_tgt.load_params(solver_src.params)        # or restore_checkpoint("outputs/source/")
solver_tgt.train(*data_tgt, config=cfg_tgt)      # lower lr recommended (3e-4 instead of 1e-3)
# Converges 2-3× faster than training from scratch
```

### Temporal transfer (extended time horizon)

```python
# Phase 1: train on t ∈ [0, T_1]
solver_phase1.train(*data_t1, config=cfg_phase1)

# Phase 2: extend to t ∈ [0, T_2], T_2 > T_1, warm-start from Phase 1
solver_phase2.load_params(solver_phase1.params)
solver_phase2.train(*data_t2, config=cfg_phase2)
```

Both modes are demonstrated in `examples/transfer/burgers_transfer.py` and `examples/pipe_flow/pipe_flow_unsteady_transfer.py`.

---

## Inverse Problems

The heat inverse problem (`examples/heat/inverse.py`) recovers the unknown thermal diffusivity α from 50 sparse noisy observations:

- **Joint optimisation**: the optimizer simultaneously updates network weights `θ` and the physics parameter `log_α = log(α)` via a single `jax.grad` call
- **Log-parameterisation**: optimising `log_α` instead of `α` directly guarantees positivity without any constraints or projections; the true α is recovered as `exp(log_α)` after training
- **Observation loss**: a separate MSE term penalises the discrepancy between model predictions at the 50 observation locations and the noisy measurements; the PDE residual loss is the regulariser

```python
# Simplified view of the inverse problem setup
from underPINN.pde.diffusion import DiffusionInversePDE

pde = DiffusionInversePDE(model, log_alpha_init=jnp.log(0.5))
# pde.log_alpha is a trainable parameter alongside model weights
# After training: alpha_recovered = jnp.exp(pde.log_alpha)
```

The 2-D diffusion inverse (`examples/inverse/inverse_diffusion.py`) follows the same pattern for a 2-D domain.

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
│   ├── ramp.py            # Trapezoidal ramp domain above a wedge (compressible Euler)
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
│   ├── dispatch.py        # _REGISTRY: problem → (script path, fn name)
│   ├── pipe_flow.py       # pipe_flow runner helper
│   ├── wave.py            # wave runner helper
│   └── heat_forward.py    # heat_forward runner helper
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
│   ├── restart.py         # RestartManager (snapshot + resume + done marker)
│   ├── metrics.py         # rel_l2, mse helpers
│   └── plotting.py        # plot_losses, plot_ode_result
│
└── __main__.py            # CLI entry point (python -m underPINN)
                           # sets XLA_PYTHON_CLIENT_PREALLOCATE=false before import jax

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
├── K-Epsilon/             run_kepsilon.py + config.yaml         (k-ε RANS turbulent channel)
├── airfoil/               airfoil_flow.py + config.yaml         (NACA 0012 Re=200)
├── pipe_flow/             pipe_flow.py + pipe_flow.yaml         (3-D Hagen-Poiseuille)
│                          pipe_flow_unsteady_transfer.py + yaml  (Re + temporal transfer)
├── ramp/                  ramp.py     +  config.yaml            (2-D compressible Euler, M=3)
└── transfer/              burgers_transfer.py + yaml            (Burgers param + temp. TL)
                           heat2d_transfer.py  + yaml            (2-D heat transfer)

docs/
└── index.html             # Static framework documentation website
```

---

## Examples

| Problem | PDE | Network | Key Features | Config |
|---|---|---|---|---|
| Exponential Decay | du/dt + λu = 0 | MLP [1,32,32,1] | `ODESolver`, TrainingConfig, callbacks | `examples/ode/config.yaml` |
| Harmonic Oscillator | d²u/dt² + ω²u = 0 | MLP [1,32,32,1] | `ODESolver`, IC derivative | `examples/ode/config.yaml` |
| 1-D Burgers | u_t + uu_x = νu_xx | MLP [2,64,64,64,1] | FBPINN, RBA, cosine LR | `examples/burgers/config.yaml` |
| 1-D Heat — Forward | u_t = αu_xx | MLP [2,64,64,64,1] | `FBPINNSolver`, exact Gaussian IC | `examples/heat/heat_forward.yaml` |
| 1-D Heat — Inverse | u_t = αu_xx | MLP [2,64,64,64,1] | Recover α from 50 noisy observations | `examples/heat/heat_inverse.yaml` |
| 1-D Wave | u_tt = c²u_xx | FourierMLP [2,128,128,1] | Dual IC (u and u_t), n_fourier=32 | `examples/wave/config.yaml` |
| 2-D Helmholtz | Δu + k²u = f | FourierMLP [2,128,128,1] | k=4, manufactured source term | `examples/helmholtz/config.yaml` |
| 2-D Diffusion Inverse | u_t = α∇²u | MLP [3,64,64,64,1] | Log-param joint optimisation | `examples/inverse/config.yaml` |
| 2-D Lid-Driven Cavity | Steady N-S, Re=100 | FBPINN + SimpleGate | `LDCSolver`, attention, Re=100 | `examples/LDC/config.yaml` |
| 2-D RANS k-ε | Turbulent channel | FBPINN | `RANSSolver`, RBA, Re=10000 | `examples/K-Epsilon/config.yaml` |
| 2-D Compressible Ramp | Steady Euler, M=3 | MLP [2,80,80,80,80,80,4] | Oblique shock θ=10°, ramp geometry | `examples/ramp/config.yaml` |
| NACA 0012 Airfoil | Steady N-S, Re=200 | MLP [2,128,128,128,3] | Exterior geometry, Cp curve, CL | `examples/airfoil/config.yaml` |
| 3-D Pipe Flow | Steady 3-D N-S | MLP [3,64,64,64,64,4] | Double-jacfwd Hessian, Pipe geometry | `examples/pipe_flow/pipe_flow.yaml` |
| 3-D Unsteady Pipe Transfer | u_t = G + ν∇²u | MLP [3,64,64,64,64,1] | Bessel exact solution, Re + temporal TL | `examples/pipe_flow/pipe_flow_unsteady_transfer.yaml` |
| Burgers Transfer | Burgers | MLP [2,64,64,64,1] | Parameter transfer (ν) + temporal transfer | `examples/transfer/burgers_transfer.yaml` |
| Heat 2-D Transfer | 2-D heat | MLP [3,64,64,64,1] | Cross-diffusivity transfer + temporal | `examples/transfer/heat2d_transfer.yaml` |

---

## PDE Reference

| PDE | Equation | Key method | Used in |
|---|---|---|---|
| Burgers (1-D) | u_t + uu_x = νu_xx | `BurgersPDE.residual` | `examples/burgers/`, `examples/transfer/` |
| Diffusion / Heat (1-D) | u_t = αu_xx | `DiffusionPDE.residual` | `examples/heat/` |
| Heat (2-D unsteady) | u_t = α(u_xx + u_yy) | `Heat2DPDE.residual` | `examples/inverse/`, `examples/transfer/` |
| Wave (1-D) | u_tt = c²u_xx | `WavePDE.residual` | `examples/wave/` |
| Helmholtz (2-D) | Δu + k²u = f | `HelmholtzPDE.residual` | `examples/helmholtz/` |
| Navier-Stokes (2-D steady) | ∇·u=0, u·∇u = -∇p + ν∇²u | `SteadyNSPDE.residual` | `examples/LDC/`, `examples/airfoil/` |
| Navier-Stokes (3-D steady) | Same + z-momentum | `SteadyNS3DPDE.residual` | `examples/pipe_flow/` |
| Pipe unsteady | u_t = G + ν(u_yy + u_zz) | `PipeUnsteadyPDE.residual` | `examples/pipe_flow/` |
| RANS k-ε | N-S + k + ε transport | `KEpsilonPDE.residual` | `examples/K-Epsilon/` |
| Compressible Euler (2-D) | ∂_t U + ∇·F = 0 (steady) | `EulerRampPDE.residual` | `examples/ramp/` |
| Exponential Decay | du/dt + λu = 0 | `ExpDecayODE.residual` | `examples/ode/` |
| Harmonic Oscillator | d²u/dt² + ω²u = 0 | `HarmonicODE.residual` | `examples/ode/` |

---

## Geometry Reference

| Class | What it samples | Example uses |
|---|---|---|
| `Interval` | 1-D uniform or Sobol interior + boundary points | Burgers, wave, heat (1-D) |
| `Rectangle` | 2-D interior (LHS / Sobol) + all four boundary edges | Helmholtz, LDC, diffusion inverse |
| `Airfoil` | NACA 4-digit exterior domain, near-surface strip, farfield arc | `examples/airfoil/` |
| `Pipe` | 3-D cylindrical interior, lateral wall, circular inlet, circular outlet | `examples/pipe_flow/` |
| `Ramp` | Trapezoidal domain above a wedge surface at angle θ | `examples/ramp/` |
| `Composite` | Boolean union / intersection / difference of any two geometry objects | LDC (cavity minus any obstacle) |
| `ShapelyGeom` | Arbitrary 2-D polygon backed by Shapely 2.x; rejection-samples interior | Custom geometries |

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

# Include slow problems (3-D pipe flow, k-ε)
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

## Performance Tips

### GPU Memory — XLA preallocation
Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX. underPINN does this automatically in every entry point, but if you are writing a new script, add it at the top before `import jax`.

### `lax.scan` — use `n_scan_steps=100` for GPU training
On GPU, each Python→XLA dispatch has ~1 ms of overhead. With 5 000 epochs and `n_scan_steps=1` that is ~5 s of pure dispatch. With `n_scan_steps=100` it drops to ~50 ms. For long runs, use `n_scan_steps=500`.

### RAR-D — enable for difficult solutions
Enable RAR-D (`resample_period > 0`) when the solution has sharp gradients or shocks (Burgers at low ν, Euler ramp). A typical setting is `resample_period=5`, `resample_k=1.0`.

### Early stopping — tune patience to the problem
- Fast ODEs: `patience=200`
- Medium PDEs (Burgers, wave, Helmholtz): `patience=400–800`
- Complex PDEs (LDC, airfoil, 3-D pipe): `patience=1000–2000`

### Float32 — do not use float64
JAX defaults to float32, which is optimal on all GPUs. Do not enable `jax.config.update("jax_enable_x64", True)` unless you have a specific reason; it halves throughput on CUDA devices.

### Multi-GPU
Use `CUDA_VISIBLE_DEVICES=1` to restrict to a specific GPU. Full multi-GPU `pmap` training is not currently implemented; single-device training on the fastest GPU is the recommended approach.

### Cosine LR decay
Always prefer `optax.cosine_decay_schedule` over a fixed learning rate for runs longer than 2 000 epochs. It provides free accuracy improvement at no cost by reducing the LR smoothly toward a small `alpha` value (recommended: `alpha=1e-2`).

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

---

## Cite underPINN

If you use underPINN in research or publications, please cite:

```bibtex
@software{underPINN_v2605,
  author  = {},
  title   = {underPINN-v2605: A Modular JAX Framework for Physics-Informed Neural Networks},
  year    = {2026},
  version = {v2605},
  url     = {https://github.com/Prashantiitk23/underPINN-v2605}
}
```

## License

underPINN is released under the GPL-3.0 License. See [LICENSE.txt](https://github.com/Prashantiitk23/underPINN-v2605/blob/main/LICENSE) for the full text.
