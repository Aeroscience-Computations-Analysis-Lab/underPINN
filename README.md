# underPINN

> A modular, GPU-accelerated Physics-Informed Neural Network framework built on JAX + Flax

![Static Badge](https://img.shields.io/badge/repo%20status-Active-95eb34) ![Static Badge](https://img.shields.io/badge/license-GPL--3.0-green)

underPINN is a research-grade PINN engine that combines classical collocation-based PINNs with Finite Basis decomposition (FBPINN), attention-augmented networks, residual-based adaptive weighting, and transfer learning — all JIT-compiled and differentiable via XLA.

---

## Features

- **Domain decomposition** — FBPINN with overlapping subdomains and sigmoid partition-of-unity windows
- **Attention networks** — Hybrid attention and gated residual blocks inside each subdomain
- **FourierMLP** — Trainable random Fourier feature embeddings for oscillatory solutions (Helmholtz, wave)
- **Residual-based adaptivity (RBA)** — Element-wise loss weighting that focuses training on high-residual regions
- **Transfer learning** — `load_params` warm-start for parameter transfer (different Re) and temporal transfer (extended time horizon)
- **Inverse problems** — Joint optimisation of network weights + physics parameters (e.g. recover thermal diffusivity from sparse noisy observations)
- **Callbacks** — `ConsoleLogger`, `EarlyStopping`, pluggable via `TrainingConfig`
- **3-D problems** — Full 3-D Navier-Stokes with double-`jacfwd` Hessians, cylindrical pipe geometry
- **GPU / multi-GPU** — Pure JAX/XLA; runs on CPU, single GPU, or multi-GPU with no code changes

---

## Repository Structure

```
underPINN/
├── core/
│   ├── base.py            # BasePDE, BaseLoss, BaseSolver abstract classes
│   └── config.py          # TrainingConfig dataclass
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
│   ├── diffusion.py       # 1-D unsteady diffusion
│   ├── heat.py            # 1-D / 2-D steady heat
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
│   ├── airfoil.py         # NACA 4-digit profile + exterior sampling
│   ├── pipe.py            # Cylindrical pipe (interior, wall, inlet, outlet)
│   ├── composite.py       # Boolean combination of geometries
│   └── shapely_geom.py    # Shapely-backed arbitrary polygon sampler
│
├── solver/
│   ├── fbpinn.py          # FBPINNSolver  (space-time PDE, TrainingConfig)
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
│   └── early_stopping.py  # EarlyStopping
│
└── utils/
    ├── plotting.py        # plot_losses, plot_ode_result
    ├── metrics.py         # rel_l2, mse helpers
    └── serialization.py   # save / load params

examples/
├── ode/                   # Exponential decay + Harmonic oscillator
├── burgers/               # 1-D Burgers  (FBPINN + RBA)
├── heat/                  # 1-D heat: forward and inverse
├── wave/                  # 1-D wave  (FourierMLP)
├── helmholtz/             # 2-D Helmholtz  (FourierMLP)
├── inverse/               # 2-D diffusion inverse  (recover α)
├── Lid Driven Cavity/     # 2-D LDC  (Re = 100, FBPINN)
├── K-Epsilon/             # 2-D turbulent channel  (k-ε RANS)
├── airfoil/               # Steady NACA 0012 airfoil flow  (Re = 200)
├── pipe_flow/             # 3-D Hagen-Poiseuille + unsteady transfer learning
└── transfer/              # Burgers and 2-D heat transfer learning
```

---

## Examples

| Problem | PDE | Highlights | Script |
|---|---|---|---|
| Exponential decay & harmonic oscillator | ODE | `ODESolver`, `TrainingConfig`, `EarlyStopping` | `examples/ode/ode_test.py` |
| 1-D Burgers equation | u_t + uu_x = νu_xx | FBPINN, RBA, cosine LR, IC/BC weights | `examples/burgers/burgers.py` |
| 1-D heat — forward | u_t = αu_xx | `FBPINNSolver`, exact Gaussian solution | `examples/heat/forward.py` |
| 1-D heat — inverse | u_t = αu_xx | Recover α from 50 noisy observations | `examples/heat/inverse.py` |
| 1-D wave equation | u_tt = c²u_xx | `FourierMLP`, dual IC (u and u_t) | `examples/wave/wave.py` |
| 2-D Helmholtz | Δu + k²u = f | `FourierMLP`, k = 4, manufactured source | `examples/helmholtz/helmholtz.py` |
| 2-D diffusion — inverse | u_t = α(u_xx+u_yy) | Log-parameterised joint optimisation | `examples/inverse/inverse_diffusion.py` |
| 2-D Lid-driven cavity | Steady N-S | FBPINN + `SimpleGate` attention, Re = 100 | `examples/Lid Driven Cavity/run_ldc.py` |
| 2-D turbulent channel | RANS k-ε | k-ε PDE, `RANSSolver` | `examples/K-Epsilon/turbulence.py` |
| Steady NACA 0012 airfoil | Steady N-S | Exterior geometry, Cp curve, CL estimate | `examples/airfoil/airfoil_flow.py` |
| 3-D Hagen-Poiseuille | Steady 3-D N-S | Double-jacfwd Hessian, `Pipe` geometry | `examples/pipe_flow/pipe_flow.py` |
| 3-D unsteady pipe — transfer | u_t = G + ν∇²u | Bessel-series exact, Re & temporal transfer | `examples/pipe_flow/pipe_flow_unsteady_transfer.py` |
| 1-D Burgers transfer learning | Burgers | Parameter transfer (ν) + temporal transfer | `examples/transfer/burgers_transfer.py` |
| 2-D unsteady heat transfer | u_t = α∇²u | Parameter transfer (α) + temporal transfer | `examples/transfer/heat2d_transfer.py` |

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

# Every solver has init + train
class BaseSolver(ABC):
    @abstractmethod
    def init(self, key): ...
    @abstractmethod
    def train(self, *args, **kwargs): ...
```

### TrainingConfig + callbacks

```python
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger
from underPINN.callbacks.early_stopping import EarlyStopping
import optax

config = TrainingConfig(
    epochs      = 5000,
    lr          = 1e-3,
    lr_schedule = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    batch_r     = 2048,
    log_every   = 500,
    callbacks   = [ConsoleLogger(log_every=500), EarlyStopping(patience=400)],
)
```

### Transfer learning

```python
# Train source model
solver_src.train(*data_src, config=cfg_src)

# Warm-start target model from source weights
solver_tgt.load_params(solver_src.params)   # resets optimiser state
solver_tgt.train(*data_tgt, config=cfg_tgt) # fine-tune at lower lr
```

---

## Quick Start

### 1-D Burgers equation

```python
import jax, optax
from underPINN.nn.mlp import MLP
from underPINN.pde.burgers import BurgersPDE
from underPINN.losses.loss import PINNLoss
from underPINN.solver.fbpinn import FBPINNSolver
from underPINN.core.config import TrainingConfig
from underPINN.callbacks.logging import ConsoleLogger

model  = MLP(layers=[2, 64, 64, 64, 1])
pde    = BurgersPDE(model, nu=0.01)
loss   = PINNLoss(model, pde, ic_weight=100.0, bc_weight=10.0, rba=True)
solver = FBPINNSolver(model, pde, loss=loss)
solver.init(jax.random.PRNGKey(0))

config = TrainingConfig(
    epochs      = 5000,
    lr          = 1e-3,
    lr_schedule = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    callbacks   = [ConsoleLogger(log_every=500)],
)
solver.train(*data, config=config)
```

### 3-D pipe flow

```python
from underPINN.nn.mlp import MLP
from underPINN.pde.navier_stokes_3d import SteadyNS3DPDE
from underPINN.geometry.pipe import Pipe

pipe  = Pipe(R=0.5, L=2.0)
model = MLP(layers=[3, 64, 64, 64, 64, 4])   # (x,y,z) → (u,v,w,p)
pde   = SteadyNS3DPDE(model, Re=10.0)

# Exact Hagen-Poiseuille solution for validation
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

## Installation

```bash
# CPU (development / testing)
pip install jax flax optax matplotlib scipy

# GPU (CUDA 12)
pip install -U "jax[cuda12]" flax optax matplotlib scipy
```

Verify GPU support:

```python
import jax
print(jax.devices())   # [CudaDevice(id=0), ...]
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
