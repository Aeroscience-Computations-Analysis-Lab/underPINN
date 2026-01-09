# JAXPINN
> A Modular, GPU-Accelerated Physics-Informed Neural Network Framework with Attention and Domain Decomposition

![Static Badge](https://img.shields.io/badge/repo%20status-Active-95eb34) ![Static Badge](https://img.shields.io/badge/license-GPL--3.0-green)

JAXPINN-FBPINN is a high-performance, modular framework for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs) in JAX + Flax.
## Overview
The framework supports:
- Classical PINNs
- Domain-decomposed PINNs (FBPINNs / XPINNs style)
- Attention-augmented neural operators
- GPU / multi-GPU acceleration via XLA
- Flexible geometry definition and sampling
- Modular PDEs, losses, solvers, and constraints
- Scientific reproducibility and extensibility

The goal is to provide a research-grade yet production-ready PINN engine suitable for:
- Scientific computing
- Physics simulation
- Inverse problems
- Data-driven PDE discovery
- Large-scale GPU experiments

## Repository Structure
```
jaxpinn/
│
├── geometry/
│   ├── base.py
│   ├── composite.py
│   ├── interval.py
│   ├── rectangle.py
│   ├── sampler.py
│   ├── shapely_geom.py
│
├── losses/
│   ├── loss.py
├── nn/
│   ├── attention.py
│   ├── embeddings.py
│   ├── fbpinn.py
│   ├── subdomain.py
├── pde/
│   ├── burgers.py
├── solver/
│   ├── fbpinn.py
├── utils/
│   ├── plotting.py
│   ├── serialization.py
```

## Installation

Install dependencies
```bash
pip install -U jax[cuda] flax optax shapely matplotlib
```

Verify GPU Support:
```bash
import jax
print(jax.devices())
```


## Cite JAXPINN

If you use JAXPINN in your research, academic work, or publications, please cite it as follows:

```
@article{jaxpinn,
  author  = {},
  title   = {},
  journal = {},
  year    = {},
  note    = {Software library},
  doi     = {}
}
```
## License

JAXPINN is provided under the GPL-3.0 License. Please see [LICENSE.txt](https://github.com/lohithgsk/PINN/blob/main/LICENSE) for the full license text. 
