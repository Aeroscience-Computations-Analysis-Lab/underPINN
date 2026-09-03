# GPU Memory Management

## Why does `nvidia-smi` show 73 GB right after import?

JAX's XLA **BFC (Best-Fit with Coalescing)** allocator pre-reserves approximately 90% of
all free VRAM the moment `import jax` executes — before a single tensor is created — in
order to avoid memory fragmentation during training.

```{warning}
**Default JAX behaviour (not what underPINN uses).** On an 80 GB A100, `import jax`
immediately reserves ~73 GB of VRAM even if your model only needs 200 MB. This blocks
other processes from using the GPU and makes it look like your job consumed the entire
card.
```

This is a deliberate XLA design choice: by owning the memory pool upfront, it can
coalesce and reuse buffers without ever calling `cudaMalloc` again during training. The
downside is that two JAX processes cannot gracefully share a GPU unless explicit limits
are set.

## underPINN disables this automatically

```{tip}
**`XLA_PYTHON_CLIENT_PREALLOCATE=false` is set for you.** This happens in
`underPINN/__main__.py` for CLI runs, and at the top of every example script for direct
`python examples/...` runs — always **before** `import jax`. You get on-demand GPU
memory growth out of the box, with no configuration needed.
```

```python
# This is already done for you — shown here for transparency
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # now allocates only what it actually needs
```

## Manual control via environment variables

```bash
# On-demand growth (default in underPINN) — frees unreserved VRAM for other jobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Hard cap — useful when sharing a node; limits to e.g. 20% of VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20

# Platform allocator — no XLA pool at all (slowest, minimal fragmentation)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Multi-GPU: restrict to a single device (e.g. GPU 1)
export CUDA_VISIBLE_DEVICES=1
```

## Programmatic override

```{important}
Any programmatic override **must** run before `import jax` — JAX reads these
environment variables once, at import time.
```

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
import jax  # now uses at most 15% of VRAM
```

## Typical VRAM usage (preallocation disabled)

```{list-table}
:header-rows: 1
:widths: 30 35 15

* - Problem
  - Network
  - VRAM (approx.)
* - Burgers 1-D
  - `[2, 64, 64, 64, 1]`
  - ~200 MB
* - Wave 1-D
  - FourierMLP
  - ~300 MB
* - Helmholtz 2-D
  - FourierMLP
  - ~400 MB
* - Lid-Driven Cavity 2-D
  - FBPINN
  - ~800 MB
* - Airfoil 2-D
  - `[2, 128, 128, 128, 3]`
  - ~1.2 GB
* - Pipe Flow 3-D
  - `[3, 64, 64, 64, 64, 4]`
  - ~2.0 GB
* - Compressible Ramp
  - `[2, 80, 80, 80, 80, 80, 4]`
  - ~1.8 GB
* - k-ε Turbulence
  - FBPINN
  - ~3.0 GB
```

```{seealso}
{doc}`performance` for further GPU throughput tuning (`lax.scan` fusion, float32
precision, multi-GPU device selection).
```
