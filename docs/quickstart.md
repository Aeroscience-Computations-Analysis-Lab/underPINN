# Quick Start

Six entry points into underPINN — pick the one that fits your workflow. All examples set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` automatically; see {doc}`gpu_memory`.

## 1. CLI — zero Python

```bash
# Single run — point at any registered YAML config
python -m underPINN run  examples/burgers/config.yaml
python -m underPINN run  examples/wave/config.yaml
python -m underPINN run  examples/pipe_flow/pipe_flow.yaml
python -m underPINN run  examples/ramp/config.yaml

# Hyperparameter sweep (Cartesian product)
python -m underPINN sweep examples/burgers/burgers_nu_sweep.yaml

# Benchmark all problems
python -m underPINN bench

# List all registered runners
python -m underPINN list

# Print resolved config without training
python -m underPINN show examples/wave/config.yaml

# Print framework version
python -m underPINN version
```

See {doc}`cli` for the full command reference.

## 2. Python API

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, optax
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
    epochs             = 5000,
    lr                 = 1e-3,
    lr_schedule        = optax.cosine_decay_schedule(1e-3, 5000, alpha=1e-2),
    batch_r            = 2048,
    log_every          = 500,
    out_dir            = "outputs/burgers",   # enables auto-restart
    save_restart_every = 500,
    callbacks = [
        ConsoleLogger(log_every=500),
        EarlyStopping(patience=400),
    ],
)
solver.train(*data, config=config)
```

## 3. YAML Config

```yaml
problem: burgers       # selects the runner

network:
  type  : mlp
  layers: [2, 64, 64, 64, 1]

physics:
  nu: 0.01

data:
  T: 2.0
  n_collocation: 6000
  n_ic: 200
  n_bc: 200

training:
  epochs                  : 5000
  lr                      : 1.0e-3
  early_stopping_patience : 400
  save_restart_every      : 500  # snapshot every 500 epochs

loss:
  ic_weight: 100.0
  bc_weight: 10.0
  rba      : true

output:
  dir        : outputs/burgers
  save_params: true
```

```bash
python -m underPINN run examples/burgers/config.yaml
```

## 4. Checkpoint & Inference

```python
from underPINN.utils.checkpoint import ModelPredictor
import jax.numpy as jnp

# Option A — auto-rebuild model from saved metadata (zero boilerplate)
predictor = ModelPredictor.from_meta("outputs/burgers/")

# Option B — supply model explicitly
from underPINN.nn.mlp import MLP
predictor = ModelPredictor.from_checkpoint(
    MLP(layers=[2, 64, 64, 64, 1]),
    "outputs/burgers/",
)

# Inference
x_test = jnp.linspace(-1.0, 1.0, 500)
t_test = jnp.full(500, 0.8)
u = predictor.predict(jnp.stack([x_test, t_test], axis=1))
```

Full reference: {doc}`checkpointing`.

## 5. Transfer Learning

```python
# Phase 1: train source model (e.g. Burgers nu=0.1)
solver_src.train(*data_src, config=cfg_src)
solver_src.save_checkpoint("outputs/source/")

# Phase 2: warm-start target from source weights (e.g. nu=0.01)
solver_tgt.load_params(solver_src.params)       # or restore_checkpoint(...)
solver_tgt.train(*data_tgt, config=cfg_tgt)     # lower lr recommended (3e-4)
# Converges 2-3x faster than training from scratch
```

Full reference: {doc}`transfer_learning`.

## 6. GPU Acceleration — `lax.scan` + RAR-D

```python
# Fuse 100 gradient steps per XLA kernel + adaptive resampling
config = TrainingConfig(
    epochs          = 5000,
    lr              = 1e-3,
    n_scan_steps    = 100,   # 50 Python calls instead of 5000
    resample_period = 5,     # RAR-D every 5 outer steps (= 500 epochs)
    resample_k      = 1.0,   # probability proportional to |residual|^1
    callbacks       = [ConsoleLogger(log_every=500)],
)
solver.train(*data, config=config)
```

Full reference: {doc}`training` and {doc}`performance`.

## Running examples directly

Every example folder is self-contained — a script plus a YAML config side by side:

```bash
python examples/burgers/burgers.py
python examples/wave/wave.py
python examples/helmholtz/helmholtz.py
python examples/LDC/run_ldc.py
python examples/pipe_flow/pipe_flow.py

# Pass a custom config as the first argument
python examples/burgers/burgers.py my_custom.yaml
```

```{admonition} What's next?
:class: seealso
- {doc}`examples` — browse all 22 worked physics examples
- {doc}`training` — the full `TrainingConfig` field reference and callback system
- {doc}`restart` — never lose progress to a killed job again
```
