# Inverse Problems

underPINN supports **joint optimisation** of network weights and physics parameters,
recovering unknown PDE coefficients directly from sparse, noisy observations.

`examples/heat/inverse.py` recovers the unknown thermal diffusivity `α` from 50 sparse
noisy observations of a 1-D diffusion field.

## How it works

```{list-table}
:header-rows: 1
:widths: 30 70

* - Mechanism
  - Description
* - **Joint optimisation**
  - The optimizer simultaneously updates network weights `θ` and the physics
    parameter `log_α = log(α)` via a single `jax.grad` call.
* - **Log-parameterisation**
  - Optimising `log_α` instead of `α` directly guarantees positivity without any
    constraints or projections; the true `α` is recovered as `exp(log_α)` after
    training.
* - **Observation loss**
  - A separate MSE term penalises the discrepancy between model predictions at the
    50 observation locations and the noisy measurements; the PDE residual loss acts
    as the regulariser.
```

```python
# Simplified view of the inverse problem setup
from underPINN.pde.diffusion import DiffusionInversePDE

pde = DiffusionInversePDE(model, log_alpha_init=jnp.log(0.5))
# pde.log_alpha is a trainable parameter alongside model weights
# After training: alpha_recovered = jnp.exp(pde.log_alpha)
```

The 2-D diffusion inverse case (`examples/inverse/inverse_diffusion.py`) follows the
same pattern for a full 2-D domain.

```{list-table} Inverse-problem examples
:header-rows: 1
:widths: 30 20 50

* - Example
  - Recovers
  - Config
* - 1-D Heat — Inverse
  - Thermal diffusivity `α` from 50 noisy observations
  - `examples/heat/heat_inverse.yaml`
* - 2-D Diffusion Inverse
  - `α` via log-parameterised joint optimisation
  - `examples/inverse/config.yaml`
```

```{admonition} Gradient flow
:class: tip
Because `log_α` is just another leaf in the same parameter pytree as the network
weights, gradients flow through **both** the PDE residual and the observation loss
simultaneously — no alternating-optimisation scheme or bi-level loop is required.
```

```{seealso}
{doc}`pde_reference` for the full list of PDE residual classes, including
`DiffusionInversePDE`.
```
