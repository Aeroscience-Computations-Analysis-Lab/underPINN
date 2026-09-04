# underPINN-v2605
> A modular, GPU-accelerated Physics-Informed Neural Network framework built on JAX + Flax + Optax

![Static Badge](https://img.shields.io/badge/version-v2605-blue) ![Static Badge](https://img.shields.io/badge/repo%20status-Active-95eb34) ![Static Badge](https://img.shields.io/badge/license-GPL--3.0-green) ![Static Badge](https://img.shields.io/badge/python-%3E%3D3.9-blue) ![Static Badge](https://img.shields.io/badge/jax-%3E%3D0.4.26-orange)

underPINN is a research-grade PINN engine that combines classical collocation-based PINNs with Finite Basis decomposition (FBPINN), attention-augmented networks, residual-based adaptive weighting/resampling, transfer learning (including windowed time-marching for long-horizon unsteady flows), shock capturing with learnable artificial viscosity, non-Newtonian (Carreau) blood rheology, inverse problems, and a full restart/resume system — all JIT-compiled and differentiable via XLA on CPU, GPU, and TPU.

---

## Features

### Network Architectures
- **MLP** — standard multi-layer perceptron with tanh activations; configurable depth and width via a layer list
- **GatedMLP** — modified MLP (Wang et al. 2022): two input encoders U/V are gate-blended into every hidden layer, curing the pathological gradient flow of plain tanh MLPs on stiff PDEs; selectable in any flow example via `network.type: gated_mlp`
- **FourierMLP** — random Fourier feature embeddings (trainable σ) prepended to a standard MLP; essential for oscillatory solutions (Helmholtz, wave, high-Re flows) where plain MLPs fail to represent high spatial frequencies
- **FBPINN** — overlapping subdomain decomposition with sigmoid partition-of-unity windows; each subdomain gets its own network so training is never dominated by one region
- **HybridAttention + SimpleGate** — gated residual blocks inside each FBPINN subdomain; SimpleGate multiplies the hidden state element-wise by a learnable gate for compact, expressive feature modulation

### Neural Operators (PINO / DeepONet / CViT)
Unlike the point networks above (one collocation point in, one value out), these map an entire **function** — sampled on a grid or at sensors — to another function, so one trained model generalizes across a *distribution* of ICs / PDE parameters instead of re-solving from scratch each time.
- **FNO1D / FNO2D** — Fourier Neural Operator (Li et al., 2020): truncated-spectrum convolution (`SpectralConv1d`/`SpectralConv2d`) + a pointwise skip path per stage; `padding` zero-pads non-periodic (Dirichlet) domains before the FFT and trims after
- **DeepONet1D** — branch/trunk operator (Lu et al., 2021): `s(u)(y) = branch(u) · trunk(y)`; reuses `MLP`/`GatedMLP` for both sub-networks
- **CViT** — Continuous Vision Transformer for Operator Learning (Wang et al., ICLR 2025): patch embed → learned sincos positions → latent time-aggregation → self-attention encoder → cross-attention decoder that queries continuous coordinates; `cvit_grid_predict` queries it on a regular grid (with a scaled-increment prediction trick) so it can pair with the same finite-difference residual used by FNO2D
- Selectable via `network.type: fno1d | fno2d | deeponet | cvit` in any case's YAML — registered once in `underPINN/nn/factory.py`
- Trained with `OperatorLoss` (data MSE + weighted PDE grid-residual, optional warmup + RBA) or `DeepONetLoss` (IC/BC + autodiff residual, no full-field ground truth needed) via `OperatorSolver` / `DeepONetSolver` — see the six examples under `examples/operators/`

### Training
- **`lax.scan` fused kernels** — fuse N gradient steps into a single XLA kernel, eliminating Python dispatch between epochs; delivers 50–500× less overhead on GPU compared to a Python for-loop
- **Cosine LR decay** — via `optax.cosine_decay_schedule`; integrates seamlessly with `TrainingConfig`
- **RAR-D adaptive collocation resampling** — periodically replaces a fraction of collocation points with samples drawn proportional to `|residual|^k` (Lu et al., 2021); focuses compute on high-error regions without changing the total batch size
- **RAR/RAD shock-focused resampling** — `rad_resample` (Wu et al., 2023; `p ∝ r^k/E[r^k] + c`) refreshes the interior pool toward shocks/contacts in the compressible cases (`ramp`, `sod_shock`, `toro3`); config knobs `rar_period`, `rar_candidates`, `rar_k`, `rar_c`
- **QR-DEIM-R adaptive collocation resampling** — `qr_deim_resample` (`underPINN/utils/sampling.py`): column-pivoted-QR DEIM anchors (Chaturantabut & Sorensen, 2010; Drmac & Gugercin, 2016) plus a leverage-score-weighted randomized fill (Drineas, Mahoney & Muthukrishnan, 2006) — selects points *deterministically well-spread* across every high-residual region instead of RAD's magnitude-proportional random draw, which can cluster redundantly on one narrow shock spike; `O(n_candidates × r0)` throughout, no dense `(n_candidates, n_keep)` matrix ever formed
- **Artificial viscosity for shocks** — global Laplacian dissipation `−ε∇²U` on the conserved variables in the compressible Euler cases; ε can be **fixed** (`art_visc`) or **learned** as `ε = softplus(log_av)` jointly with the network (`trainable_visc: true`). When learned, the parameters become a `{"net", "log_av"}` pytree optimized by a single `optax` chain, so `log_av` currently shares the network's optimizer, learning rate and cosine schedule
- **Gauss-Newton / natural-gradient training** — `underPINN.training.train_gauss_newton`: Levenberg-Marquardt-damped Gauss-Newton second-order optimizer (`(JᵀJ + λI)Δθ = Jᵀr`, trust-region damping adapted step-to-step) as an alternative to Adam; explicit dense solve is `O(n_params³)` per step, so it is scoped to small networks (low hundreds to a few thousand parameters) — not a replacement for underPINN's Adam-based solvers on the large 3-D / compressible-flow benchmarks, where `optax.lbfgs` limited-memory quasi-Newton refinement after Adam is the scalable second-order option instead
- **Time-marching transfer learning** — long-horizon unsteady problems split into windows; each window warm-starts from the previous one and chains its end-state as the next initial condition (pulsatile pipe flow), with per-window checkpoints and window-level restart
- **RBA element-wise loss weighting** — residual-based adaptivity assigns per-point weights so that boundary and collocation losses are automatically balanced during training
- **EarlyStopping** — monitors a metric (default: total loss) and halts training after `patience` epochs without improvement
- **`TrainingConfig` dataclass** — centralises all hyperparameters with runtime validation; a single object is passed to every solver
- **Callbacks** — `ConsoleLogger` (prints loss every N epochs), `EarlyStopping` (halts on plateau), `ModelCheckpoint` (saves best model during training); all callbacks fire correctly even inside `lax.scan` loops

### Restart / Resume
- **`RestartManager`** saves `params.msgpack`, `opt_state.msgpack`, loss histories, and a `meta.json` to `<out_dir>/restart/` every `save_restart_every` epochs
- On re-run, `RestartManager` checks the `"done"` flag; if `done: false` (interrupted run), training resumes exactly from the last snapshot — epoch counter, optimizer state, and loss histories are all restored
- **`done()` marker** — once training completes normally or via early stopping, the snapshot is marked `"done": true`; the next `run` starts fresh. An interrupted run leaves `done: false` and auto-resumes on the next `run`
- **`resume` CLI command** — `python -m underPINN resume <config.yaml>` verifies the MD5 config hash and resets `done` so a completed run can be continued; warns if any field (lr, epochs, layers, …) changed since the last snapshot

### GPU Memory
- JAX's XLA BFC allocator pre-reserves ~90% of all free VRAM the moment `import jax` executes; on an 80 GB A100 this shows as ~73 GB reserved even for a 3-layer MLP
- **underPINN sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` automatically** — in `underPINN/__main__.py` for CLI runs and at the top of every example script for direct `python examples/…` runs — so on-demand GPU memory growth is the default behaviour out of the box

### PDE Library
- 1-D Burgers equation (`u_t + uu_x = νu_xx`)
- 1-D / 2-D diffusion / heat — forward and inverse (recover thermal diffusivity α)
- 1-D wave equation (`u_tt = c²u_xx`)
- 2-D Helmholtz (`Δu + k²u = f`, manufactured source)
- 2-D steady incompressible Navier-Stokes (lid-driven cavity Re=100, NACA airfoil, circular cylinder Re=40)
- 2-D RANS k-ε turbulence model (turbulent channel, Re=10 000)
- 3-D steady incompressible Navier-Stokes (Hagen-Poiseuille pipe flow, AAA bulge)
- 3-D **unsteady** incompressible Navier-Stokes (`(x,y,z,t) → (u,v,w,p)`; pulsatile pipe via time-marching transfer)
- 3-D **generalized-Newtonian (Carreau)** Navier-Stokes — shear-thinning blood rheology (`μ(γ̇) = μ∞ + (μ0−μ∞)[1+(λγ̇)²]^((n−1)/2)`); pipe and AAA cases
- 2-D steady compressible Euler — **conservative flux-divergence form** with optional artificial viscosity (oblique-shock compression ramp, Mach 3, θ=10°; two-corner **compression-expansion ramp** combining the oblique shock with an exact **Prandtl-Meyer expansion fan** reference state via `prandtl_meyer_expansion`)
- 2-D steady compressible **Navier–Stokes** — viscous Mach-3 compression ramp / **shock–boundary-layer interaction** (no-slip + isothermal `T=T₀` walls, Newtonian stress + Fourier conduction, Re=10⁴, Pr=0.72)
- 1-D **unsteady** compressible Euler — Sod shock tube with learnable artificial viscosity + exact Riemann reference
- 1-D **unsteady** compressible Euler — **Toro test 3** (Woodward–Colella blast wave, 5-decade pressure jump) with **exp/log positivity** + reference-state **non-dimensionalisation** for the extreme dynamic range
- Unsteady pipe cross-section (`(y, z, t) → u`)
- Harmonic oscillator (`d²u/dt² + ω²u = 0`)
- Exponential decay ODE (`du/dt + λu = 0`)
- **FBPINN ODE** (`du/dx = cos(ω x)`) — domain decomposition into overlapping subdomains; defeats the spectral bias that stalls a single PINN at high ω
- 1-D periodic / Dirichlet viscous Burgers — grid (finite-difference) residual for FNO1D (`BurgersGrid1D`)
- 2-D periodic viscous Burgers — grid residual for FNO2D / CViT, selectable central or upwind advection stencil (`BurgersGrid2D`)
- 1-D viscous Burgers — autodiff residual for DeepONet (`DeepONetBurgersPDE`)
- 2-D incompressible Navier-Stokes, primitive variables `(u,v,p)` — grid residual for the cylinder-flow FNO2D operator, with an obstacle mask (`CylinderNSGrid`)

### Geometry
- **Interval** — 1-D uniform / stratified sampler
- **Rectangle** — 2-D interior + boundary-aware sampling
- **NACA 4-digit airfoil** — symmetric **and cambered** profiles; angle of attack imposed by **rotating the airfoil** about the quarter-chord (inflow stays horizontal); SDF-weighted near-surface sampling; camber-aware upper/lower surface split
- **Circular cylinder (2-D)** — exterior cross-flow domain with analytic SDF (`Cylinder2D`)
- **Cylindrical Pipe** — interior, wall, inlet, and outlet face samplers
- **AAA bulge** (`BulgeGeometry`) — axisymmetric vessel with a cosine-squared AAA bulge `R(x)`; interior / curved-wall / inlet / outlet samplers
- **Ramp** — trapezoidal domain above a wedge surface for compressible shock problems
- **Compression-Expansion Ramp** (`CompressionExpansionRampGeometry`) — two-corner wall (compression corner into an expansion corner), piecewise-linear with per-segment normals, for combined oblique-shock + Prandtl-Meyer-expansion problems
- **Composite** — boolean combinations of any geometry objects
- **Shapely-backed polygon** — arbitrary 2-D polygon sampler backed by Shapely 2.x

### Solvers
- **`FBPINNSolver`** — space-time PDE training with `lax.scan`, RAR-D, and RestartManager integration
- **`ODESolver`** — lightweight ODE training with callbacks and checkpointing
- **`SteadySolver`** — stationary (no time dimension) PDE training
- **`LDCSolver`** — lid-driven cavity / FBPINN variant
- **`RANSSolver`** — k-ε turbulence model with RBA loss weighting
- **`OperatorSolver`** — generic training loop for grid-based neural operators (FNO1D, FNO2D, CViT); same `n_scan_steps` acceleration as `FBPINNSolver`, plus a PDE-weight warmup ramp
- **`DeepONetSolver`** — physics-informed DeepONet training (fresh IC/BC/residual points sampled each step)

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
- Two-tier epoch budgets: smooth PDEs (Burgers, wave, Helmholtz, …) use the default `[500, 1000, 2000, 5000]`; problems marked `complex=True` on their evaluator (shocks / viscous SBLI / 3-D N-S — `ramp`, `toro3`, `pipe_flow`, `ramp_ns`) automatically get a much larger `[2000, 5000, 15000, 40000]` — they converge far slower and the shared small budget was under-training them. Both are overridable (`--epochs`, `--complex-epochs`)
- Outputs include PNG accuracy plots, convergence grids, CSV tables, wall-time charts, and a Markdown summary report
- `--from-json` replays plotting from a previous JSON result without re-training

### CLI
- **`run`** — run a single problem from a YAML config
- **`resume`** — verify the config MD5 hash against a stored snapshot and reset `done` for continuation; warns if any config field changed
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
pip install jax flax optax matplotlib scipy shapely trimesh pandas einops pyyaml
```

```bash
# GPU (CUDA 12)
pip install -U "jax[cuda12]" && pip install -r requirements-gpu.txt
```

```bash
# TPU (Google Colab TPU runtime / Cloud TPU VM)
pip install -r requirements-tpu.txt
pip install -e . --no-deps     # --no-deps so pip can't replace the TPU jax with the cpu pin
# underPINN auto-detects the TPU and forces full-float32 matmuls
# (the bf16 MXU default corrupts second-order PDE residuals)
```

```bash
# From source (editable install — recommended)
git clone https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git
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
python -m underPINN run  examples/cylinder/config.yaml                          # flow over a cylinder (Re=40)
python -m underPINN run  examples/sod_shock/config.yaml                         # Sod tube, learnable ε
python -m underPINN run  examples/AAA/config.yaml                               # 3-D AAA bulge
python -m underPINN run  examples/pipe_flow_rheology/config.yaml                # Carreau blood, pipe
python -m underPINN run  examples/AAA_rheology/config.yaml                      # Carreau blood, AAA
python -m underPINN run  examples/pipe_flow/pipe_flow_pulsatile_transfer.yaml   # pulsatile, time-marching TL
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
python examples/cylinder/cylinder_flow.py
python examples/ode/ode_test.py
python examples/pipe_flow/pipe_flow.py
python examples/pipe_flow/pipe_flow_unsteady_transfer.py
python examples/pipe_flow/pipe_flow_pulsatile_transfer.py
python examples/pipe_flow_rheology/pipe_flow_rheology.py
python examples/AAA/AAA_flow.py
python examples/AAA_rheology/AAA_rheology.py
python examples/ramp/ramp.py
python examples/sod_shock/sod_shock.py
python examples/transfer/burgers_transfer.py
python examples/inverse/inverse_diffusion.py

# Pass a custom config as the first argument:
python examples/burgers/burgers.py my_custom.yaml
```

### Post-processing / prediction (after training)

```bash
# Steady pipe & AAA (Newtonian or Carreau) — axial-plane u contour + streamlines,
# pressure contour & line plots, wall shear stress, and an NPZ of the solution:
python examples/predict_steady.py outputs/pipe_flow
python examples/predict_steady.py outputs/AAA_rheology

# Pulsatile pipe (time-marching) — point queries, snapshot/spacetime plots, GIF:
python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer --t 2.7 --plot
python examples/pipe_flow/predict_pulsatile.py outputs/pipe_flow_pulsatile_transfer --spacetime --animate
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

2. On re-run, `RestartManager` detects the snapshot directory and reads `meta.json`. If `done: false` (the run was interrupted), params, optimizer state, and loss histories are restored and training continues from the saved epoch. Plots are continuous across restarts.

3. **Config-change safety** — solvers do not hash-check configs automatically. An interrupted run will resume even if you changed `lr`, `epochs`, or other fields. Use `python -m underPINN resume <config.yaml>` to verify the config hash before resuming. To force a fresh start without the command, delete `<out_dir>/restart/` manually.

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
| `meta.json` | `{"epoch": N, "cfg_hash": null, "done": false}` — `cfg_hash` is `null` when run via solver; populated by the `resume` CLI |

### Config change detection

Solvers gate resumption on the `done` flag only — not on a config hash. This means an interrupted run will auto-resume even if you changed `lr`, `epochs`, `layers`, or physics parameters. To check for config changes before resuming, use the `resume` CLI command:

```bash
python -m underPINN resume examples/burgers/config.yaml
```

`resume` computes the MD5 of the current YAML, compares it against the hash stored in `meta.json`, and warns you if any field changed since the last snapshot. If everything is consistent, it resets `done` to `false` so the next `run` will resume.

To force a fresh start without the command, simply delete `<out_dir>/restart/` or set `save_restart_every: 0` temporarily.

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
│   ├── subdomain.py       # SubdomainNetwork
│   ├── operators.py       # FNO1D, FNO2D, DeepONet1D, CVit, cvit_grid_predict
│   └── factory.py         # build_model / network_config — single model-building path
│
├── pde/
│   ├── burgers.py         # 1-D Burgers equation
│   ├── burgers_grid.py    # BurgersGrid1D/2D — FD residual for FNO1D/FNO2D/CViT
│   ├── burgers_deeponet.py# DeepONetBurgersPDE — autodiff residual for DeepONet
│   ├── navier_stokes_2d_grid.py # CylinderNSGrid — FD residual for the cylinder FNO2D
│   ├── diffusion.py       # 1-D unsteady diffusion / heat inverse
│   ├── heat.py            # 2-D steady heat (Poisson)
│   ├── heat2d_unsteady.py # 2-D unsteady heat  (x, y, t) → u
│   ├── helmholtz.py       # 2-D Helmholtz  Δu + k²u = f
│   ├── wave.py            # 1-D wave equation  u_tt = c²u_xx
│   ├── navier_stokes.py   # 2-D steady incompressible N-S
│   ├── navier_stokes_3d.py# 3-D steady + UNSTEADY incompressible N-S
│   ├── carreau_ns_3d.py   # 3-D Carreau (shear-thinning) N-S + 1-D exact profile
│   ├── compressible_euler.py # 2-D steady Euler — conservative form + artificial viscosity; oblique_shock + prandtl_meyer_expansion reference states
│   ├── euler_1d_unsteady.py  # 1-D unsteady Euler (Sod) — learnable artificial viscosity
│   ├── pipe_flow_unsteady.py # Unsteady pipe cross-section  (y, z, t) → u
│   ├── k_epsilon.py       # RANS k-ε turbulence model
│   └── ode.py             # Exponential decay, Harmonic oscillator
│
├── geometry/
│   ├── interval.py        # 1-D interval sampler
│   ├── rectangle.py       # 2-D rectangle sampler
│   ├── airfoil.py         # NACA 4-digit (sym/cambered) + AoA rotation + SDF sampling
│   ├── cylinder.py        # 2-D circular cylinder (cross-flow exterior)
│   ├── pipe.py            # Cylindrical pipe (interior, wall, inlet, outlet)
│   ├── aaa.py             # BulgeGeometry — axisymmetric AAA bulge R(x)
│   ├── ramp.py            # RampGeometry (single wedge) + CompressionExpansionRampGeometry (two-corner)
│   ├── composite.py       # Boolean combination of geometries
│   └── shapely_geom.py    # Shapely-backed arbitrary polygon sampler
│
├── solver/
│   ├── fbpinn.py          # FBPINNSolver  (space-time PDE, lax.scan, RAR-D)
│   ├── ode_solver.py      # ODESolver
│   ├── steady_solver.py   # SteadySolver  (no time dimension)
│   ├── ldc_solver.py      # LDCSolver     (lid-driven cavity / FBPINN)
│   ├── rans_solver.py     # RANSSolver    (k-ε turbulence)
│   └── operator.py        # OperatorSolver (FNO/CViT), DeepONetSolver
│
├── losses/
│   ├── loss.py            # PINNLoss  (with optional RBA)
│   ├── ode_loss.py        # ODELoss
│   ├── steady_loss.py     # SteadyLoss
│   └── operator_loss.py   # OperatorLoss (data+PDE+warmup+RBA), DeepONetLoss
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
│   ├── resample.py        # rar_d_resample, rar_d_resample_split  (RAR-D adaptive collocation)
│   └── natural_gradient.py # train_gauss_newton — L-M-damped Gauss-Newton / natural-gradient training
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
│   ├── sampling.py        # safe_choice (mini-batching); rad_resample, qr_deim_resample (adaptive collocation)
│   ├── seed.py            # set_seed (Python + NumPy + JAX)
│   ├── checkpoint.py      # save_checkpoint, load_checkpoint, ModelPredictor
│   ├── restart.py         # RestartManager (snapshot + resume + done marker)
│   ├── timing.py          # fmt_train_time (JIT-aware training time reporting)
│   ├── metrics.py         # rel_l2, mse helpers
│   ├── plotting.py        # plot_losses, plot_ode_result
│   └── operator_datagen.py # FD reference solvers + random ICs for the operator examples
│
├── postprocess/
│   ├── plotting.py        # field / cbar / save_fig (shared matplotlib style)
│   ├── pulsatile.py       # PulsatilePredictor
│   └── operators.py       # plot_operator_loss, plot_prediction_1d/2d
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
├── fbpinn_ode/            fbpinn_ode.py + config.yaml           (FBPINN subdomains, du/dx=cos ωx)
├── inverse/               inverse_diffusion.py + config.yaml    (2-D diffusion inverse)
├── LDC/                   run_ldc.py  +  config.yaml            (2-D Lid-Driven Cavity Re=100)
├── K-Epsilon/             run_kepsilon.py + config.yaml         (k-ε RANS turbulent channel)
├── airfoil/               airfoil_flow.py + config.yaml         (NACA airfoil, AoA via rotation)
├── cylinder/              cylinder_flow.py + config.yaml        (cylinder cross-flow, Re=40)
├── pipe_flow/             pipe_flow.py + pipe_flow.yaml         (3-D Hagen-Poiseuille)
│                          pipe_flow_unsteady_transfer.py + yaml  (Re + temporal transfer)
│                          pipe_flow_pulsatile_transfer.py + yaml (3-D pulsatile, time-marching TL)
│                          predict_pulsatile.py                   (window predictor: plots, GIF, spacetime)
├── pipe_flow_rheology/    pipe_flow_rheology.py + config.yaml   (Carreau blood, pipe)
├── AAA/                   AAA_flow.py + config.yaml             (3-D AAA bulge, Newtonian)
├── AAA_rheology/          AAA_rheology.py + config.yaml         (Carreau blood, AAA bulge)
├── ramp/                  ramp.py     +  config.yaml            (2-D compressible Euler, M=3, AV + RAR)
├── ramp_ns/               ramp_ns.py  +  config.yaml            (2-D compressible NS, viscous SBLI, no-slip + isothermal)
├── sod_shock/             sod_shock.py + config.yaml            (Sod tube, learnable ε + RAR)
├── toro3/                  toro3.py + config.yaml                (Toro-3 blast wave, exp positivity + non-dim)
├── predict_steady.py                                            (post-process steady pipe/AAA: WSS, contours, NPZ)
├── transfer/              burgers_transfer.py + yaml            (Burgers param + temp. TL)
│                          heat2d_transfer.py  + yaml            (2-D heat transfer)
└── operators/             fno1d_periodic/      + config.yaml    (1-D periodic Burgers, FNO1D PINO)
                           fno1d_dirichlet/     + config.yaml    (1-D Dirichlet Burgers, FNO1D PINO)
                           fno2d_burgers/       + config.yaml    (2-D periodic Burgers, FNO2D PINO)
                           deeponet1d_burgers/  + config.yaml    (1-D Burgers, physics-informed DeepONet)
                           cvit2d_burgers/      + config.yaml    (2-D periodic Burgers, CViT PINO)
                           fno2d_cylinder/      + config.yaml + datagen.py  (cylinder-flow FNO2D PINO)

docs/
└── index.html             # Static framework documentation website
```

---

## Examples

| Problem | PDE | Network | Key Features | Config |
|---|---|---|---|---|
| Exponential Decay | du/dt + λu = 0 | MLP [1,32,32,1] | `ODESolver`, TrainingConfig, callbacks | `examples/ode/config.yaml` |
| Harmonic Oscillator | d²u/dt² + ω²u = 0 | MLP [1,32,32,1] | `ODESolver`, IC derivative | `examples/ode/config.yaml` |
| FBPINN ODE | du/dx = cos(ω x) | FBPINN — 15 subnets [1,16,16,1] | Overlapping subdomains + partition-of-unity windows, hard IC constraint | `examples/fbpinn_ode/config.yaml` |
| 1-D Burgers | u_t + uu_x = νu_xx | MLP [2,64,64,64,1] | FBPINN, RBA, cosine LR | `examples/burgers/config.yaml` |
| 1-D Heat — Forward | u_t = αu_xx | MLP [2,64,64,64,1] | `FBPINNSolver`, exact Gaussian IC | `examples/heat/heat_forward.yaml` |
| 1-D Heat — Inverse | u_t = αu_xx | MLP [2,64,64,64,1] | Recover α from 50 noisy observations | `examples/heat/heat_inverse.yaml` |
| 1-D Wave | u_tt = c²u_xx | FourierMLP [2,128,128,1] | Dual IC (u and u_t), n_fourier=32 | `examples/wave/config.yaml` |
| 2-D Helmholtz | Δu + k²u = f | FourierMLP [2,128,128,1] | k=4, manufactured source term | `examples/helmholtz/config.yaml` |
| 2-D Diffusion Inverse | u_t = α∇²u | MLP [3,64,64,64,1] | Log-param joint optimisation | `examples/inverse/config.yaml` |
| 2-D Lid-Driven Cavity | Steady N-S, Re=100 | FBPINN + SimpleGate | `LDCSolver`, attention, Re=100 | `examples/LDC/config.yaml` |
| 2-D RANS k-ε | Turbulent channel | FBPINN | `RANSSolver`, RBA, Re=10000 | `examples/K-Epsilon/config.yaml` |
| 2-D Compressible Ramp | Steady Euler (conservative), M=3 | MLP [2,80,80,80,80,80,4] | Oblique shock θ=10°, artificial viscosity (fixed/learnable), RAR | `examples/ramp/config.yaml` |
| 2-D Compressible NS Ramp (SBLI) | Steady Navier–Stokes (conservative), M=3 | MLP [2,128×4,4] | Viscous shock–boundary-layer interaction, **no-slip + isothermal `T=T₀`** walls, Re=10⁴, Pr=0.72, RAR | `examples/ramp_ns/config.yaml` |
| 1-D Sod Shock Tube | Unsteady Euler (conservative) | MLP [2,80×5,3] | **Learnable ε = softplus(log_av)**, exact Riemann reference, RAR | `examples/sod_shock/config.yaml` |
| 1-D Toro Test 3 (blast wave) | Unsteady Euler (conservative) | MLP [2,128×4,3] | **exp/log positivity**, reference-state **non-dimensionalisation**, learnable ε, RAR | `examples/toro3/config.yaml` |
| NACA Airfoil | Steady N-S, Re=100 | MLP / GatedMLP [2,128×6,3] | Cambered profiles, AoA via airfoil rotation, surface pressure & Cp | `examples/airfoil/config.yaml` |
| Cylinder Cross-flow | Steady N-S, Re=40 | MLP [2,128×6,3] | Pure-PINN recipe, Cp(θ) vs inviscid reference, wake pool | `examples/cylinder/config.yaml` |
| 3-D Pipe Flow | Steady 3-D N-S | MLP / GatedMLP [3,…,4] | Double-jacfwd Hessian, Hagen-Poiseuille exact | `examples/pipe_flow/pipe_flow.yaml` |
| 3-D AAA Bulge | Steady 3-D N-S | GatedMLP [3,192×5,4] | Cosine² bulge `R(x)`, flow-rate balance check | `examples/AAA/config.yaml` |
| Carreau Pipe (blood) | Steady Carreau N-S | GatedMLP [3,128×4,4] | Shear-thinning μ(γ̇), 1-D Carreau exact profile, β=16; same domain/Re as Newtonian pipe | `examples/pipe_flow_rheology/config.yaml` |
| Carreau AAA (blood) | Steady Carreau N-S | GatedMLP [3,192×5,4] | Blood rheology in the bulge, apparent-viscosity maps | `examples/AAA_rheology/config.yaml` |
| 3-D Pulsatile Pipe | Unsteady 3-D N-S | GatedMLP [4,…,4] | **Time-marching transfer** (windowed), per-window ckpts, window restart | `examples/pipe_flow/pipe_flow_pulsatile_transfer.yaml` |
| 3-D Unsteady Pipe Transfer | u_t = G + ν∇²u | MLP [3,64,64,64,64,1] | Bessel exact solution, Re + temporal TL | `examples/pipe_flow/pipe_flow_unsteady_transfer.yaml` |
| Burgers Transfer | Burgers | MLP [2,64,64,64,1] | Parameter transfer (ν) + temporal transfer | `examples/transfer/burgers_transfer.yaml` |
| Heat 2-D Transfer | 2-D heat | MLP [3,64,64,64,1] | Cross-diffusivity transfer + temporal | `examples/transfer/heat2d_transfer.yaml` |
| FNO1D Periodic Burgers | Burgers (PINO, grid residual) | FNO1D | Generalizes across ν, `pde_weight` warmup | `examples/operators/fno1d_periodic/config.yaml` |
| FNO1D Dirichlet Burgers | Burgers (PINO, grid residual) | FNO1D | Zero-wall BCs, FNO domain-padding trick | `examples/operators/fno1d_dirichlet/config.yaml` |
| FNO2D Burgers | 2-D Burgers (PINO, grid residual) | FNO2D | Central/upwind stencil selectable | `examples/operators/fno2d_burgers/config.yaml` |
| DeepONet Burgers | Burgers (self-supervised, autodiff residual) | DeepONet1D | IC/BC + residual only, no full-field data | `examples/operators/deeponet1d_burgers/config.yaml` |
| CViT Burgers | 2-D Burgers (PINO, grid residual) | CViT | Scaled-increment prediction, matched-upwind residual | `examples/operators/cvit2d_burgers/config.yaml` |
| FNO2D Cylinder Flow | 2-D incompressible N-S, `(u,v,p)` | FNO2D | Chorin-projection reference data, obstacle mask | `examples/operators/fno2d_cylinder/config.yaml` |

---

## PDE Reference

| PDE | Equation | Key method | Used in |
|---|---|---|---|
| Burgers (1-D) | u_t + uu_x = νu_xx | `BurgersPDE.residual` | `examples/burgers/`, `examples/transfer/` |
| Diffusion / Heat (1-D) | u_t = αu_xx | `DiffusionPDE.residual` | `examples/heat/` |
| Heat (2-D unsteady) | u_t = α(u_xx + u_yy) | `Heat2DPDE.residual` | `examples/inverse/`, `examples/transfer/` |
| Wave (1-D) | u_tt = c²u_xx | `WavePDE.residual` | `examples/wave/` |
| Helmholtz (2-D) | Δu + k²u = f | `HelmholtzPDE.residual` | `examples/helmholtz/` |
| Navier-Stokes (2-D steady) | ∇·u=0, u·∇u = -∇p + ν∇²u | `NavierStokesPDE.residual` | `examples/LDC/`, `examples/airfoil/`, `examples/cylinder/` |
| Navier-Stokes (3-D steady) | Same + z-momentum | `SteadyNS3DPDE.residual` | `examples/pipe_flow/`, `examples/AAA/` |
| Navier-Stokes (3-D unsteady) | u_t + (u·∇)u = −∇p + ν∇²u | `UnsteadyNS3DPDE.residual` | `examples/pipe_flow/` (pulsatile) |
| Carreau N-S (3-D steady) | ∇·[μ*(γ̇)(∇u+∇uᵀ)] stress | `CarreauNS3DPDE.residual` | `examples/pipe_flow_rheology/`, `examples/AAA_rheology/` |
| Pipe unsteady | u_t = G + ν(u_yy + u_zz) | `PipeUnsteadyPDE.residual` | `examples/pipe_flow/` |
| RANS k-ε | N-S + k + ε transport | `KEpsilonPDE.residual` | `examples/K-Epsilon/` |
| Compressible Euler (2-D steady) | ∂F/∂x + ∂G/∂y = ε∇²U (conservative) | `CompressibleEulerPDE.residual` (+ `oblique_shock`, `prandtl_meyer_expansion` reference states) | `examples/ramp/`, compression-expansion ramp studies |
| Compressible Navier–Stokes (2-D steady) | ∂x(F−Fv/Re) + ∂y(G−Gv/Re) = 0 | `CompressibleNS2DPDE.residual` | `examples/ramp_ns/` |
| Compressible Euler (1-D unsteady) | ∂U/∂t + ∂F/∂x = ε∂²U/∂x² | `Euler1DUnsteadyPDE.residual` | `examples/sod_shock/` |
| Exponential Decay | du/dt + λu = 0 | `ExpDecayODE.residual` | `examples/ode/` |
| Harmonic Oscillator | d²u/dt² + ω²u = 0 | `HarmonicODE.residual` | `examples/ode/` |
| Burgers grid (1-D, FNO) | u_t + uu_x = νu_xx (FD residual) | `BurgersGrid1D.residual` | `examples/operators/fno1d_periodic/`, `fno1d_dirichlet/` |
| Burgers grid (2-D, FNO/CViT) | u_t + u(u_x+u_y) = ν(u_xx+u_yy) (FD residual) | `BurgersGrid2D.residual` | `examples/operators/fno2d_burgers/`, `cvit2d_burgers/` |
| Burgers (DeepONet) | u_t + uu_x = νu_xx (autodiff residual) | `DeepONetBurgersPDE.residual` | `examples/operators/deeponet1d_burgers/` |
| Navier-Stokes grid (2-D, FNO) | ∇·u=0, u·∇u = -∇p + ν∇²u (FD residual) | `CylinderNSGrid.residual` | `examples/operators/fno2d_cylinder/` |

---

## Geometry Reference

| Class | What it samples | Example uses |
|---|---|---|
| `Interval` | 1-D uniform or Sobol interior + boundary points | Burgers, wave, heat (1-D) |
| `Rectangle` | 2-D interior (LHS / Sobol) + all four boundary edges | Helmholtz, LDC, diffusion inverse |
| `NACAAirfoil` | NACA 4-digit (symmetric & cambered) exterior domain, SDF-weighted near-surface, AoA via quarter-chord rotation | `examples/airfoil/` |
| `Cylinder2D` | Circular cylinder exterior cross-flow domain, analytic SDF, surface points | `examples/cylinder/` |
| `Pipe` | 3-D cylindrical interior, lateral wall, circular inlet, circular outlet | `examples/pipe_flow/`, `examples/pipe_flow_rheology/` |
| `BulgeGeometry` | Axisymmetric AAA bulge `R(x)` (cosine²): interior, curved wall, inlet, outlet | `examples/AAA/`, `examples/AAA_rheology/` |
| `Ramp` | Trapezoidal domain above a wedge surface at angle θ | `examples/ramp/` |
| `CompressionExpansionRampGeometry` | Two-corner wall (compression + expansion), piecewise-linear, per-segment normals | Compression-expansion ramp studies |
| `Composite` | Boolean union / intersection / difference of any two geometry objects | LDC (cavity minus any obstacle) |
| `ShapelyGeom` | Arbitrary 2-D polygon backed by Shapely 2.x; rejection-samples interior | Custom geometries |

---

## Benchmark Suite

```bash
# Run all fast problems — smooth PDEs get [500, 1000, 2000, 5000] epochs;
# problems marked complex=True (ramp, toro3) get [2000, 5000, 15000, 40000]
python -m underPINN bench

# Select specific problems and a custom budget for the simple ones
python -m underPINN bench \
    --problems burgers wave helmholtz heat_steady ode_harmonic ramp toro3 \
    --epochs 500 1000 2000 5000 \
    --output outputs/bench

# Include slow problems (3-D pipe flow, viscous ramp NS — also complex=True)
python -m underPINN bench --all

# Override the complex-problem budget directly (shocks / viscous SBLI / 3-D
# N-S converge far slower than smooth PDEs — the default 40k-epoch top budget
# reflects that; drop it for a quicker, coarser pass)
python -m underPINN bench --all --complex-epochs 2000 5000 15000

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
Use `CUDA_VISIBLE_DEVICES=1` to restrict to a specific GPU. Full multi-GPU sharded training is not currently implemented; single-device training on the fastest GPU is the recommended approach. To use several GPUs, launch one run per device (e.g. a parameter sweep with a different `CUDA_VISIBLE_DEVICES` per process).

### TPU
Install with `requirements-tpu.txt` (see Installation). On import, underPINN detects the TPU backend and sets `jax_default_matmul_precision = "float32"` automatically — the TPU's default bfloat16 matmuls corrupt second-order PDE residuals (Hessians), so full-float32 matmuls are required for PINN accuracy. Override via the `JAX_DEFAULT_MATMUL_PRECISION` env var. (Set it to the canonical `"float32"`, not the `"highest"` alias — JAX's validator accepts only `'bfloat16'`/`'tensorfloat32'`/`'float32'` and rejects the aliases; `"float32"` is `Precision.HIGHEST`.)

### Cosine LR decay
Always prefer `optax.cosine_decay_schedule` over a fixed learning rate for runs longer than 2 000 epochs. It provides free accuracy improvement at no cost by reducing the LR smoothly toward a small `alpha` value (recommended: `alpha=1e-2`).

### Training time reporting
Every solver prints a timing summary at the end of training:
```
Training complete — final loss 1.23e-04 | 45.2s  [JIT≈12s + 3.3ms/ep]
```
The `JIT≈…` component appears when the first epoch is ≥ 3 s and at least 4× slower than the average of subsequent epochs, separating XLA compilation overhead from actual training time. The `ms/ep` figure is the mean wall-clock cost per epoch after JIT warm-up — useful for benchmarking solver configurations.

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
  author  = {Kumar Prashant, Senthilkumar Lohith, Ranjan Rajesh},
  title   = {underPINN-v2605: A Modular JAX Framework for Physics-Informed Neural Networks},
  year    = {2026},
  version = {v2605},
  url     = {https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git}
}
```

## License

underPINN is released under the GPL-3.0 License. See [LICENSE.txt](https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN/blob/main/LICENSE) for the full text.
