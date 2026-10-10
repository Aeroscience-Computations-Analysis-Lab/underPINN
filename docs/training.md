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
* - `data_parallel`
  - `bool`
  - `False`
  - Split each mini-batch across all visible devices — see {ref}`data-parallelism`
* - `data_parallel_devices`
  - `int` / `None`
  - `None`
  - Use only the first N visible devices for `data_parallel` (`None` = all)
* - `model_parallel`
  - `bool`
  - `False`
  - Split an FBPINN's subdomain networks across devices — see {ref}`model-parallelism`
* - `model_parallel_devices`
  - `int` / `None`
  - `None`
  - Use only the first N visible devices for `model_parallel` (`None` = all)
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

(data-parallelism)=
## Data parallelism (multi-GPU)

Split every mini-batch across all visible devices along its point axis, with the
network parameters and optimizer state replicated on each device. **Every example
and every solver supports it**, switched on with one line of YAML — no code change:

```yaml
training:
  data_parallel: true
  data_parallel_devices: 4    # optional: use only the first 4 visible devices
```

Omit `data_parallel_devices` to use every visible device; **any count works**
(2, 3, 5, 7, …), with any batch size. A smaller count leaves the remaining GPUs free
for other jobs.
Asking for more devices than are visible is an error rather than a silent cap; to
choose *which* GPUs are used, restrict visibility with `CUDA_VISIBLE_DEVICES`. A
count set while `data_parallel` is off is ignored with a warning.

Programmatically, pass it to any solver (`FBPINNSolver`, `SteadySolver`,
`LDCSolver`, `ODESolver`, `RANSSolver`, `OperatorSolver`, `DeepONetSolver`):

```python
config = TrainingConfig(epochs=20000, batch_r=8192, batch_b=512,
                        data_parallel=True, data_parallel_devices=4)
solver.train(*data, config=config)
```

`examples/PINNs/transfer/heat2d_transfer.py` has no YAML; use
`python heat2d_transfer.py --data-parallel [--devices N]`.

**It does not change the optimisation.** `jax.jit` over `NamedSharding`-annotated
arrays is a *global-view* computation: the program means exactly what it would
unsharded, and the sharding only decides where work runs. Every underPINN loss term
is a mean over its own point set, which the partitioner lowers to a cross-device
all-reduce divided by the *global* point count — so the gradient equals the
single-device full-batch gradient, up to float32 reduction-order noise.

**Batches that do not divide the device count are padded.** Each device gets an
equal slice, so a batch of N points on D devices is padded to the next multiple of
D by repeating its first `(-N) % D` rows — at most D−1 points. Runs are reported at
start-up:

```text
[data_parallel] each batch split across 3 gpu devices; params replicated
[data_parallel] batches not divisible by 3 are padded by repeating up to 2 of their own points:
    batch_r = 2048 -> 2049  (+1; use 2046 or 2049 for an exact split)
```

The only effect is that those repeated points count twice in that step's mean
(here 1 point in 2049). For randomly drawn mini-batches the repeated points are
themselves random, so the gradient stays an unbiased estimate; for a fixed
full-batch set it is a fixed reweighting of at most D−1 points. **When every batch
divides the device count nothing is padded**, and training is exactly equivalent to
one device. Points and their paired targets / RBA state are padded with the same
rows, so they stay aligned; solvers that return per-point state (RBA in
`LDCSolver` / `RANSSolver`) drop the padding again, and saved predictions use the
original, unpadded points.

```{note}
Why pad instead of splitting unevenly: JAX can split an uneven array only *inside*
a single compiled program. An unevenly split array passed between compiled steps
comes back fully **replicated** — correct, but nothing is split and nothing warns
you. Padding keeps every device's slice equal at the cost of ≤ D−1 repeated points.
```

```{list-table}
:header-rows: 1
:widths: 26 74

* - Behaviour
  - Detail
* - Batch sizes
  - Any size, on any device count. Batches that do not divide are padded (above);
    the start-up report lists each one — including derived sizes such as
    `min(batch_bc, N_inlet)` and Ramp-NS's uniform / boundary-layer / adaptive
    sub-batches — with the nearest sizes that avoid padding. For a run that must
    be exactly equivalent to one device, choose those, or call
    `dp.require_divisible(...)` to make any padding an error.
* - What is split
  - The mini-batches — collocation points and their per-point targets / RBA state,
    and for operators the batch of samples.
* - What stays replicated
  - Small or shared sets evaluated whole every step: boundary sets in the
    full-batch flows (airfoil, cylinder, heat-inverse), the initial-condition term
    in time-marching, RANS's inlet / wall / outlet / observation sets, DeepONet's
    sensors and BC points. Correct, but not divided — they bound the speedup.
* - Training paths
  - Python loop and `lax.scan` (`n_scan_steps > 1`); scan stacks are split along
    their point axis, not their scan-step axis.
* - Restart / resume
  - Parameters restored from a snapshot are re-replicated, so a resumed run stays
    data-parallel. Checkpoints of replicated parameters save normally.
* - Resampling
  - RAR-D / RAD / QR-DEIM-R work unchanged: the point pools are not sharded, only
    each gathered mini-batch is.
* - Time-marching / transfer
  - Set up per window / per run, since each warm-starts from the previous params
    with a fresh optimizer state.
* - Single device
  - A no-op — no padding, no behaviour change.
```

```{warning}
Data parallelism scales **throughput, not model size**. Parameters are replicated, so
a network that does not fit in one device's memory still will not fit. Splitting the
parameters themselves is model parallelism — available for FBPINN subdomain networks,
see {ref}`model-parallelism`.
```

### In your own training loop

The examples all use the same three calls; a custom loop can do the same:

```python
from underPINN.training import data_parallel_from_cfg

dp = data_parallel_from_cfg(cfg.training)        # inert unless data_parallel: true;
                                                 # honours data_parallel_devices
# after any restart-restore, so restored params get replicated too:
params, opt_state = dp.prepare(params, opt_state,
                               {"batch_r": batch_r, "batch_bc": batch_bc})
for ep in range(epochs):
    ...
    params, opt_state, loss = step(params, opt_state,
                                   *dp.shard_all((x_r[ir], x_bc[ib])))
```

An inert instance makes every call a no-op, so the loop needs no
`if data_parallel:` branches. For a `lax.scan` stack shaped
`(n_scan, batch, ...)`, use `dp.shard_all(..., axis=1)`.

```{important}
Pass the data into the jitted step **as arguments**. `jax.jit` bakes a closed-over
array into the compiled program as a full constant on *every* device — so a step
that closes over its collocation points runs replicated, with no all-reduce and no
speedup, and gives no error. (`ODESolver`'s `lax.scan` path and
`examples/PINNs/heat/inverse.py` were restructured for exactly this reason.)
```

```{note}
Correctness is CI-tested on 8 *virtual* CPU devices. `tests/test_parallel.py`
checks the sharded gradient, every solver, and that the compiled `ODESolver` scan
step contains a real all-reduce. `tests/test_parallel_examples.py` trains **every
example** with the flag off and on and requires the loss trajectories to match —
while counting the sharding each run performs, so an example that silently ignored
the flag would fail rather than pass on identical numbers. Virtual devices prove the
numerics; they cannot measure speedup, which needs real accelerators.
```

(model-parallelism)=
## Model parallelism (FBPINN subdomains)

Data parallelism copies the whole network to every device and splits the *points*.
Model parallelism splits the *network*: an FBPINN's subdomain networks are divided
across devices, so with S subdomains on D devices each device holds and evaluates
only S/D of them (on every point), and the contributions are summed. Per-device
parameter memory, activation memory and compute all scale with S/D — this is what
lets an FBPINN grow to many subdomains.

```yaml
training:
  model_parallel: true
  model_parallel_devices: 4    # optional: use only the first 4 visible devices
```

The FBPINN examples (`fbpinn_ode`, `LDC`, `K-Epsilon`) take this switch directly. In
your own code, wrap the model **before** building the PDE (the PDE keeps a
reference to it):

```python
from underPINN.training import model_parallel_from_cfg

mp    = model_parallel_from_cfg(cfg.training)     # inert unless model_parallel: true
model = mp.wrap(FBPINN(...))                       # -> StackedFBPINN, padded for D
pde   = MyPDE(model, ...)
params = model.init(key, x0)
params, opt_state = mp.prepare(params, opt.init(params), model=model)  # after restore
...
save_checkpoint(mp.export(model, params), out_dir)  # ordinary FBPINN layout
```

With a solver, wrap the model the same way and pass
`TrainingConfig(model_parallel=True)`; the solver does the `prepare` step.

**It does not change the optimisation.** `mp.wrap` turns the FBPINN into a
`StackedFBPINN` — the same networks with their weights stacked along a "subdomain"
axis instead of kept in a Python list, which is what lets them be split. It is the
same function (outputs, derivatives and gradients agree to float32 rounding),
`init` reproduces FBPINN's random initialisation **bit-for-bit**, and splitting the
stacked axis is a layout choice inside a global-view `jit`. Training follows the
single-device loss trajectory exactly.

```{list-table}
:header-rows: 1
:widths: 26 74

* - Behaviour
  - Detail
* - Any device count
  - If S does not divide by D, the model is padded with up to D−1 *dummy*
    subdomains multiplied by an exact 0 mask: they add nothing to the output and
    receive exactly zero gradient, so the optimiser never moves them.
* - Checkpoints
  - `mp.export(model, params)` returns the ordinary FBPINN layout
    (`subnets_0 … subnets_{S-1}`, padding dropped), so a checkpoint from a
    model-parallel run loads into a plain `FBPINN` and vice versa. The FBPINN
    examples save this way.
* - Restart / resume
  - Restored parameters are re-split across devices.
* - Solver coverage
  - All solvers accept the flag. A non-FBPINN model is refused with an error
    rather than silently trained on one device.
* - Too few subdomains
  - With fewer real subdomains than devices, some devices hold only padding and
    do no useful work; a warning says so. `LDC` ships a **single** subdomain, so
    model parallelism cannot speed it up — use data parallelism there.
* - With data parallelism
  - Not yet combined: enable one or the other. Doing both needs a 2-D device
    mesh (points × subdomains).
```

```{note}
Why only FBPINN: underPINN's other networks are far too small to split usefully
(an MLP is ~0.2–0.6 MB; the largest shipped FNO ~38 MB). Splitting one across
devices adds communication inside every layer, multiplied by the nested
`jacfwd` / `hessian` passes a PDE residual needs — communication would outweigh
compute. FBPINN's subdomains are independent networks joined only by a weighted
sum, so they split cleanly. Use data parallelism for everything else.
```

```{note}
Correctness is CI-tested on virtual CPU devices: exact initialisation and
function equivalence, gradients on 2–8 devices including padded splits, a real
all-reduce in the compiled step, dummy subdomains that never move under Adam,
end-to-end training through `FBPINNSolver`, `LDCSolver` and `RANSSolver`, and the
FBPINN examples with plain-layout checkpoints. Speedup needs real accelerators.
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
The shipped shock configs (`examples/PINNs/toro3`, `examples/PINNs/ramp`, `examples/PINNs/sod_shock`)
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
from underPINN.training import train_gauss_newton

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
