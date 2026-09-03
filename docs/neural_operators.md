# Neural Operators (PINO / DeepONet / CViT)

Unlike the point networks used elsewhere in underPINN (one collocation point in, one
value out), neural operators map an entire **function** — sampled on a grid or at
sensors — to another function. A single trained model generalizes across a
*distribution* of initial conditions or PDE parameters, instead of re-solving from
scratch for every new case.

## Architectures

```{list-table}
:header-rows: 1
:widths: 20 80

* - Architecture
  - Description
* - **FNO1D / FNO2D**
  - Fourier Neural Operator (Li et al., 2020): truncated-spectrum convolution
    (`SpectralConv1d` / `SpectralConv2d`) plus a pointwise skip path per stage.
    `padding` zero-pads non-periodic (Dirichlet) domains before the FFT and trims
    after.
* - **DeepONet1D**
  - Branch/trunk operator (Lu et al., 2021): `s(u)(y) = branch(u) · trunk(y)`. Reuses
    the standard `MLP` / `GatedMLP` for both sub-networks.
* - **CViT**
  - Continuous Vision Transformer for Operator Learning (Wang et al., ICLR 2025):
    patch embed → learned sin/cos positions → latent time-aggregation →
    self-attention encoder → cross-attention decoder that queries continuous
    coordinates. `cvit_grid_predict` queries it on a regular grid (with a
    scaled-increment prediction trick) so it can pair with the same finite-difference
    residual used by FNO2D.
```

Select an operator architecture the same way as any other network — via YAML:

```yaml
network:
  type: fno1d   # fno1d | fno2d | deeponet | cvit
```

All four are registered once in `underPINN/nn/factory.py`.

## Training loops

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Solver
  - Loss
  - Used for
* - `OperatorSolver`
  - `OperatorLoss` — data MSE + weighted PDE grid-residual, optional warmup + RBA
  - FNO1D, FNO2D, CViT
* - `DeepONetSolver`
  - `DeepONetLoss` — IC/BC + autodiff residual, no full-field ground truth needed
  - DeepONet1D
```

`OperatorSolver` shares the same `n_scan_steps` GPU acceleration as `FBPINNSolver`
(see {doc}`training`), plus a PDE-weight warmup ramp.

## Operator-specific PDE residuals

```{list-table}
:header-rows: 1
:widths: 35 45 20

* - Residual class
  - Equation
  - Discretisation
* - `BurgersGrid1D`
  - `u_t + uu_x = νu_xx`
  - Finite-difference, periodic or Dirichlet
* - `BurgersGrid2D`
  - `u_t + u(u_x+u_y) = ν(u_xx+u_yy)`
  - Finite-difference, central or upwind stencil (selectable)
* - `DeepONetBurgersPDE`
  - `u_t + uu_x = νu_xx`
  - Autodiff (no grid required)
* - `CylinderNSGrid`
  - `∇·u = 0`, `u·∇u = -∇p + ν∇²u`
  - Finite-difference, with an obstacle mask
```

## Worked examples

```{list-table}
:header-rows: 1
:widths: 25 30 25 20

* - Example
  - Architecture
  - Highlights
  - Config
* - FNO1D Periodic Burgers
  - FNO1D
  - Generalizes across ν, `pde_weight` warmup
  - `examples/operators/fno1d_periodic/config.yaml`
* - FNO1D Dirichlet Burgers
  - FNO1D
  - Zero-wall BCs, FNO domain-padding trick
  - `examples/operators/fno1d_dirichlet/config.yaml`
* - FNO2D Burgers
  - FNO2D
  - Central/upwind stencil selectable
  - `examples/operators/fno2d_burgers/config.yaml`
* - DeepONet Burgers
  - DeepONet1D
  - IC/BC + residual only, no full-field data
  - `examples/operators/deeponet1d_burgers/config.yaml`
* - CViT Burgers
  - CViT
  - Scaled-increment prediction, matched-upwind residual
  - `examples/operators/cvit2d_burgers/config.yaml`
* - FNO2D Cylinder Flow
  - FNO2D
  - Chorin-projection reference data, obstacle mask
  - `examples/operators/fno2d_cylinder/config.yaml`
```

```{tip}
The FNO2D cylinder-flow example ships its own `datagen.py`, which generates
Chorin-projection reference solutions used as supervised targets for the operator's
data-fit loss term.
```

```{seealso}
{doc}`pde_reference` for the full residual-class table and {doc}`examples` for the
complete catalogue of 22 worked physics examples.
```
