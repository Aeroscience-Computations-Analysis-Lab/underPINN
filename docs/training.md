# Training System

A single `TrainingConfig` dataclass centralises every hyperparameter with runtime
validation, and is passed to any solver's `train()` method — no kwargs scattered across
multiple calls.

## `TrainingConfig` — full field reference

```{list-table}
:header-rows: 1
:widths: 22 16 12 50

* - Field
  - Type
  - Default
  - Description
* - `epochs`
  - `int`
  - `1000`
  - Total training epochs
* - `lr`
  - `float`
  - `1e-3`
  - Base learning rate
* - `lr_schedule`
  - optax schedule
  - `None`
  - Overrides `lr` when set; use `optax.cosine_decay_schedule`
* - `batch_r`
  - `int`
  - `4096`
  - Collocation mini-batch size
* - `batch_i`
  - `int`
  - `512`
  - Initial-condition mini-batch size
* - `batch_b`
  - `int`
  - `512`
  - Boundary-condition mini-batch size
* - `log_every`
  - `int`
  - `100`
  - Print interval (used by `ConsoleLogger`)
* - `seed`
  - `int`
  - `0`
  - PRNG seed
* - `callbacks`
  - `list`
  - `[]`
  - List of `Callback` objects
* - `n_scan_steps`
  - `int`
  - `1`
  - Fuse N steps into one XLA kernel (`1` = plain Python loop)
* - `resample_period`
  - `int`
  - `0`
  - RAR-D resampling every N outer steps (`0` = off)
* - `resample_candidates`
  - `int`
  - `0`
  - Candidate pool size (`0` → `5 × batch_r`)
* - `resample_k`
  - `float`
  - `1.0`
  - Exponent in `p ∝ |residual|^k`
* - `out_dir`
  - `str`
  - `""`
  - Output directory; enables auto-restart when non-empty
* - `save_restart_every`
  - `int`
  - `500`
  - Snapshot interval in epochs (`0` = off)
```

## Network architectures

```{list-table}
:header-rows: 1
:widths: 20 80

* - Architecture
  - Description
* - **MLP**
  - Standard multi-layer perceptron with tanh activations. Configurable depth and
    width via a simple layer list, e.g. `[2, 64, 64, 64, 1]` for a space-time Burgers
    network.
* - **GatedMLP**
  - Modified MLP (Wang et al., 2022): two input encoders U/V are gate-blended into
    every hidden layer. Cures pathological gradient flow on stiff PDEs. Select with
    `network.type: gated_mlp`.
* - **FourierMLP**
  - Trainable random Fourier feature embeddings prepended to a standard MLP. Essential
    for oscillatory solutions — Helmholtz, wave, high-Re flows — where plain MLPs
    exhibit spectral bias.
* - **FBPINN + SimpleGate**
  - Overlapping subdomain decomposition with sigmoid partition-of-unity windows.
    `HybridAttention` and `SimpleGate` gated residual blocks live inside each
    subdomain for complex geometries.
```

```{seealso}
Neural operators (FNO, DeepONet, CViT) live in a separate family — see
{doc}`neural_operators`.
```

## `lax.scan` XLA fusion

Instead of a Python `for` loop that calls back into Python every epoch, `lax.scan`
unrolls N gradient steps into a single compiled XLA program. The interpreter only
touches the computation once per `n_scan_steps` iterations, dramatically reducing
dispatch overhead.

```python
config = TrainingConfig(
    epochs       = 5000,
    lr           = 1e-3,
    n_scan_steps = 100,   # 50 outer Python calls instead of 5000
    callbacks    = [ConsoleLogger(log_every=500)],
)
solver.train(*data, config=config)
```

```{list-table}
:header-rows: 1
:widths: 20 30 25 25

* - `n_scan_steps`
  - Python calls / 5 000 epochs
  - Callback granularity
  - Use case
* - **1** (default)
  - 5 000
  - every epoch
  - Development / debugging
* - **100**
  - 50
  - every 100 epochs
  - GPU training, medium runs
* - **500**
  - 10
  - every 500 epochs
  - Long GPU runs, production
```

## RAR-D adaptive collocation resampling

At every `resample_period` outer steps, the solver:

1. Evaluates the PDE residual `r(x)` at a pool of `resample_candidates` candidate points
2. Computes sampling probabilities `p(x) ∝ |r(x)|^k`
3. Replaces the lowest-residual collocation points with new draws from this distribution

This concentrates compute on high-error regions without changing the total batch size or
requiring any geometry change (Lu et al., 2021).

```yaml
training:
  n_scan_steps    : 100
  resample_period : 5      # every 5 outer steps = every 500 epochs
  resample_k      : 1.0    # linear in |residual|
```

```{note}
For shock-dominated problems (`ramp`, `sod_shock`, `toro3`) underPINN instead uses
**RAR/RAD shock-focused resampling** (`rad_resample`, Wu et al. 2023),
`p ∝ r^k / E[r^k] + c`, tuned via `rar_period`, `rar_candidates`, `rar_k`, `rar_c`.
```

## QR-DEIM-R adaptive collocation resampling

`underPINN.utils.sampling.qr_deim_resample` is an alternative selection rule for the
same resampling slot, addressing a specific weakness of magnitude-proportional
sampling: because RAR-D/RAD draw *randomly* from `p(x) ∝ |r(x)|^k`, nothing stops
dozens of draws landing on top of each other on one narrow shock spike while other
high-residual regions go unsampled.

QR-DEIM instead selects points **deterministically well-spread** across the residual
field, using the column-pivoted-QR selection rule from the Discrete Empirical
Interpolation Method (Chaturantabut & Sorensen, 2010; Drmač & Gugercin, 2016) to keep
the chosen subset as mutually independent as possible.

```python
from underPINN.utils.sampling import qr_deim_resample

new_points = qr_deim_resample(
    pde, params, sampler,
    n_keep=20000,          # collocation batch size
    n_candidates=200000,   # candidate pool to select from
    augment_coords=True,   # add residual-scaled coordinates to the feature basis
)
```

Plain QR-DEIM is capped at one point per basis column, which cannot fill a collocation
batch of thousands. The **R** (randomized) is a leverage-score-weighted fill (Drineas,
Mahoney & Muthukrishnan, 2006) that reaches the remaining `n_keep` points while
respecting the same small basis.

```{note}
This is not a transcription of a specific published "QR-DEIM-R" algorithm — it is a
construction inspired by that selection philosophy. Cost is `O(n_candidates × r0)`
throughout (`r0` = a handful of basis columns); no dense
`(n_candidates, n_keep)` matrix is ever formed, which at real batch sizes
(e.g. 40,000 × 200,000) would allocate tens of gigabytes.
```

## Shock capturing — artificial viscosity

The compressible Euler cases add global Laplacian dissipation `−ε∇²U` on the conserved
variables. `ε` can be:

- **Fixed** — set directly via `art_visc`
- **Learned** — jointly optimised as `ε = softplus(log_av)` alongside the network
  weights (`trainable_visc: true`)

When learned, the parameters become a `{"net", "log_av"}` pytree optimized by a
single `optax` chain, so `log_av` shares the network's optimizer, learning rate and
cosine schedule rather than having its own.

```{note}
The shipped shock configs (`examples/toro3`, `examples/ramp`, `examples/sod_shock`)
all set `trainable_visc: false` and run with a fixed `art_visc`. If you want the
learned coefficient, you must opt in explicitly.
```

## Time-marching transfer learning

Long-horizon unsteady flows (e.g. the 3-D pulsatile pipe) are split into time windows.
Each window warm-starts from the end-state of the previous window, with per-window
checkpoints and window-level restart. See {doc}`transfer_learning`.

## RBA — residual-based adaptive weighting

Residual-based adaptivity assigns **per-point loss weights** so boundary and collocation
losses are automatically balanced during training, especially effective for stiff
boundary conditions. Enable with `loss.rba: true`.

## Gauss-Newton / natural-gradient training

An optional **second-order** alternative to Adam, for cases where the PDE-residual
loss surface is too ill-conditioned for a first-order method to make progress.

Every underPINN loss is a sum of squared residuals,
$L(\theta) = \tfrac{1}{2}\lVert r(\theta) \rVert^2$ (PDE residual plus weighted
IC/BC residuals, concatenated). Gauss-Newton approximates the Hessian by
$J^\mathsf{T}J$ — curvature taken from the residual's own Jacobian rather than from
gradient statistics — and solves the Levenberg-Marquardt-damped normal equations

$$(J^\mathsf{T}J + \lambda I)\,\Delta\theta = J^\mathsf{T} r$$

with the damping $\lambda$ adapted step-to-step: each step is accepted only if it
lowers the loss, damping shrinks after an accepted step and grows after a rejected
one (a trust-region scheme). Loss is therefore **monotonically non-increasing**.

```python
from underPINN.training.natural_gradient import train_gauss_newton

# residual_fn: params pytree -> 1-D residual vector
def residual_fn(params):
    return pde.residual(params, x_r).reshape(-1)

final_params, loss_hist, damping_hist = train_gauss_newton(
    residual_fn, params0,
    epochs=200,
    damping0=1e-3,      # initial LM damping
    damping_up=3.0,     # grow after a rejected step
    damping_down=0.5,   # shrink after an accepted step
    log_every=20,
)
```

`final_params` keeps the same pytree structure as `params0`; `loss_hist` and
`damping_hist` have length `epochs + 1` (the initial values are included).

```{warning}
**Scoped to small networks.** The step performs an explicit dense solve costing
$O(n_\text{params}^3)$, so it is practical only for networks in the low hundreds to
a few thousand parameters. It is *not* a replacement for the Adam-based solvers on
the large 3-D or compressible-flow benchmarks — for those, `optax.lbfgs`
limited-memory quasi-Newton refinement *after* Adam is the scalable second-order
option.
```

```{note}
Two numerical precautions are load-bearing and were each verified necessary:
the step solves an **augmented least-squares** system rather than forming
$J^\mathsf{T}J$ explicitly (forming it squares the Jacobian's condition number),
and the whole traced body runs under forced `jax.default_matmul_precision("float32")`
— JAX's default GPU matmul precision corrupts `jacfwd`'s internal matmuls enough to
make the Jacobian wrong in exactly the low-order digits Newton's method depends on.
See {doc}`performance` for the precision naming caveat.
```

## Callbacks

**ConsoleLogger**

```python
ConsoleLogger(log_every=500)
# Prints: [epoch / total]  loss=X.XXe-04  pde=X.XXe-04  ic=X.XXe-03 ...
```

**EarlyStopping**

```python
EarlyStopping(patience=400, monitor="loss", min_delta=1e-8)
```

Monitors a metric (default: total loss) and halts training after `patience` epochs
without improvement. Correctly fires at the outer-step boundary even inside `lax.scan`
loops.

**ModelCheckpoint**

```python
from underPINN.callbacks.checkpoint import ModelCheckpoint

ModelCheckpoint(
    out_dir="outputs/burgers/",
    monitor="loss",       # metric key from the loss aux dict
    mode="min",            # "min" or "max"
    save_best_only=True,   # skip non-improving epochs
    metadata={
        "problem": "burgers",
        "network": {"type": "mlp", "layers": [2, 64, 64, 64, 1]},
    },
)
```

Writes `params.msgpack` + `params_meta.json` whenever a new best is reached. Full
reference: {doc}`checkpointing`.

## Performance tips

```{list-table}
:header-rows: 1
:widths: 30 70

* - Problem class
  - Recommended `EarlyStopping.patience`
* - Fast ODEs
  - 200
* - Medium PDEs (Burgers, wave, Helmholtz)
  - 400 – 800
* - Complex PDEs (LDC, airfoil, 3-D)
  - 1 000 – 2 000
```

Combine early stopping with **cosine LR decay** (`optax.cosine_decay_schedule`) for
runs longer than 2 000 epochs — it delivers free accuracy improvement at no extra cost.

See {doc}`performance` for a full GPU throughput tuning guide.
