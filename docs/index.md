# underPINN

*A modular, GPU-accelerated Physics-Informed Neural Network framework built on JAX + Flax + Optax*

```{image} https://img.shields.io/badge/version-v2605-blue
:alt: version
```
```{image} https://img.shields.io/badge/repo%20status-Active-95eb34
:alt: repo status
```
```{image} https://img.shields.io/badge/license-GPL--3.0-green
:alt: license
```
```{image} https://img.shields.io/badge/python-%3E%3D3.9-blue
:alt: python
```
```{image} https://img.shields.io/badge/jax-%3E%3D0.4.26-orange
:alt: jax
```

underPINN is a research-grade PINN engine that combines classical collocation-based PINNs
with Finite Basis decomposition (**FBPINN**), attention-augmented networks, residual-based
adaptive weighting/resampling, transfer learning (including windowed time-marching for
long-horizon unsteady flows), shock capturing with learnable artificial viscosity,
non-Newtonian (Carreau) blood rheology, neural operators (**FNO**, **DeepONet**, **CViT**),
inverse problems, and a full restart/resume system — all JIT-compiled and differentiable
via XLA on CPU, GPU, and TPU.

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Get Started
:link: quickstart
:link-type: doc
Install underPINN and train your first PINN in five minutes.
:::

:::{grid-item-card} 🧪 Physics Examples
:link: examples
:link-type: doc
22 worked examples across 8 physics domains, from ODEs to 3-D turbulence.
:::

:::{grid-item-card} ⚙️ Training System
:link: training
:link-type: doc
`TrainingConfig`, callbacks, `lax.scan` fusion, and RAR-D adaptive resampling.
:::

:::{grid-item-card} 💾 Restart & Checkpoints
:link: restart
:link-type: doc
Fault-tolerant training that resumes exactly where it left off.
:::
::::

## Key numbers

```{list-table}
:header-rows: 0
:widths: 20 80

* - **22**
  - Physics examples
* - **5**
  - PDE solver classes
* - **19+**
  - CLI-registered runners
* - **500×**
  - Less GPU dispatch overhead with `lax.scan` fusion
* - **3-D**
  - Unsteady Navier–Stokes support (pulsatile pipe flow)
* - **Auto**
  - Restart / resume on any interruption
```

## Why underPINN?

```{admonition} Zero-boilerplate GPU memory management
:class: tip
JAX's XLA allocator reserves ~90% of free VRAM the instant `import jax` runs. underPINN
sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` automatically in every entry point, so a
3-layer MLP uses ~200 MB instead of reserving 73 GB on an 80 GB A100. See
{doc}`gpu_memory` for details.
```

```{admonition} YAML-driven, zero code changes
:class: note
Every hyperparameter — network architecture, physics parameters, training schedule,
loss weights — lives in a YAML file. Adding a brand-new physics case requires **one
script, one YAML file, and one line** in the runner registry. See {doc}`cli` and
{doc}`examples`.
```

## Documentation contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
gpu_memory
quickstart
cli
```

```{toctree}
:maxdepth: 2
:caption: Core Systems

training
restart
checkpointing
transfer_learning
inverse_problems
neural_operators
```

```{toctree}
:maxdepth: 2
:caption: Reference

examples
pde_reference
geometry_reference
api_reference
benchmark
performance
```

## Citing underPINN

If you use underPINN in research or publications, please cite:

```bibtex
@software{underPINN,
  author  = {Kumar Prashant, Senthilkumar Lohith, Ranjan Rajesh},
  title   = {underPINN: A Modular JAX Framework for Physics-Informed Neural Networks},
  year    = {2026},
  version = {v2608},
  url     = {https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN.git}
}
```

underPINN is released under the **GPL-3.0** License. See
[LICENSE.txt](https://github.com/Aeroscience-Computations-Analysis-Lab/underPINN/blob/main/LICENSE)
for the full text.
