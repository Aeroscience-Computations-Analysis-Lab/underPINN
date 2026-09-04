# Installation

underPINN is pure Python. Install it in **editable mode** from the repository root so
every example and the CLI resolve imports automatically.

## Requirements

```{list-table}
:header-rows: 1
:widths: 30 70

* - Package
  - Purpose
* - `jax[cpu] >= 0.4.26`
  - JIT compilation, autodiff, PRNG
* - `flax >= 0.8.0`
  - Neural network layers and parameter trees
* - `optax >= 0.2.0`
  - Adam, cosine decay, gradient clipping
* - `numpy`, `scipy`, `matplotlib`
  - Numerics, exact solutions, plotting
* - `shapely >= 2.0`
  - Arbitrary polygon geometry support
* - `pyyaml >= 6.0`
  - YAML config loading and merging
* - `pandas`, `trimesh`, `einops`
  - Reporting utilities and operator-learning helpers
```

## CPU / Development

```bash
pip install jax flax optax matplotlib scipy shapely trimesh pandas einops pyyaml
```

## GPU (CUDA 12)

```bash
pip install -U "jax[cuda12]"
pip install -r requirements-gpu.txt
```

## TPU (Colab / Cloud TPU VM)

```bash
pip install -r requirements-tpu.txt
pip install -e . --no-deps
```

```{note}
`--no-deps` is required so pip cannot silently replace the TPU-provisioned JAX build
with the CPU pin from `setup.py`. underPINN auto-detects the TPU backend and forces
full-float32 matmuls — the default bfloat16 MXU precision corrupts second-order PDE
Hessians used throughout the framework.
```

## From Source (recommended)

```bash
git clone https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git
cd underPINN
pip install -e .
```

## Verify your installation

```bash
python -c "import jax; print(jax.devices())"
```

Expected output on a GPU-enabled machine:

```text
[CudaDevice(id=0)]
```

```{seealso}
Once installed, head to {doc}`quickstart` to train your first PINN, or read
{doc}`gpu_memory` to understand how underPINN manages VRAM on shared GPU nodes.
```
