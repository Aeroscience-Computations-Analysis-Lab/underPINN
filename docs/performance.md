# Performance

Engineered for GPU throughput. Two orthogonal optimisations stack cleanly:
`lax.scan`-based XLA fusion eliminates Python overhead, and RAR-D concentrates compute
on hard regions. Both are composable with every solver.

## Headline numbers

```{list-table}
:header-rows: 0
:widths: 20 80

* - **500×**
  - Less Python dispatch overhead on GPU with `n_scan_steps=500`
* - **\|r\|^k**
  - RAR-D resampling probability proportional to residual magnitude
* - **float32**
  - All arrays cast to float32 — optimal throughput on all GPUs; do not enable x64
* - **0 MB**
  - Wasted VRAM — on-demand XLA allocation by default via
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`
```

## `lax.scan` — `n_scan_steps` reference

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

On GPU, each Python→XLA dispatch has roughly 1 ms of overhead. With 5 000 epochs and
`n_scan_steps=1`, that's ~5 s of pure dispatch. With `n_scan_steps=100` it drops to
~50 ms. For long runs, use `n_scan_steps=500`.

## Tuning checklist

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GPU Memory
Already handled — underPINN sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` in every entry
point before `import jax`. If writing a new script, add this line yourself at the
very top. See {doc}`gpu_memory`.
:::

:::{grid-item-card} `lax.scan` on GPU
Use `n_scan_steps=100` for medium GPU runs. For long runs (>5 000 epochs), use
`500`. On CPU, leave it at `1` for full callback granularity.
:::

:::{grid-item-card} RAR-D for sharp solutions
Enable RAR-D (`resample_period=5`, `resample_k=1.0`) when the solution has sharp
gradients or shocks — Burgers at low ν, Euler ramp, wave at high frequency.
:::

:::{grid-item-card} Early stopping patience
Fast ODEs: `patience=200`. Medium PDEs: `400–800`. Complex PDEs (LDC, airfoil,
3-D): `1000–2000`. Combine with cosine LR decay for best results.
:::

:::{grid-item-card} Float32 — do not use float64
Do not call `jax.config.update("jax_enable_x64", True)`. Float64 halves throughput
on CUDA devices and is not needed for PINN training.
:::

:::{grid-item-card} Multi-GPU
Use `CUDA_VISIBLE_DEVICES=1` to restrict to a specific GPU. Full multi-GPU `pmap`
training is not currently implemented — launch one run per device instead.
:::
::::

## TPU

Installing via `requirements-tpu.txt` (see {doc}`installation`) causes underPINN to
detect the TPU backend on import and set `jax_default_matmul_precision = "highest"`
automatically.

```{warning}
The TPU's default bfloat16 MXU matmuls corrupt second-order PDE residuals (Hessians),
so full-float32 matmuls are required for PINN accuracy. Override via the
`JAX_DEFAULT_MATMUL_PRECISION` environment variable if you have a specific reason to.
```

## Cosine learning-rate decay

Always prefer `optax.cosine_decay_schedule` over a fixed learning rate for runs longer
than 2 000 epochs. It provides free accuracy improvement at no extra cost by reducing
the LR smoothly toward a small `alpha` value (recommended: `alpha=1e-2`).

## Training time reporting

Every solver prints a timing summary at the end of training:

```text
Training complete — final loss 1.23e-04 | 45.2s  [JIT≈12s + 3.3ms/ep]
```

The `JIT≈…` component appears when the first epoch is ≥ 3 s **and** at least 4× slower
than the average of subsequent epochs — cleanly separating XLA compilation overhead
from actual per-epoch training cost. The `ms/ep` figure is the mean wall-clock cost per
epoch after JIT warm-up, useful for benchmarking solver configurations.

```{seealso}
{doc}`training` for `TrainingConfig` fields, {doc}`gpu_memory` for VRAM management, and
{doc}`benchmark` for systematically measuring these numbers across every problem.
```
