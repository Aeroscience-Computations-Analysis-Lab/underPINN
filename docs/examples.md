# Physics Examples

22 worked examples across 8 physics domains. Each example folder is **self-contained**:
one script + one YAML config living side by side. Run directly
(`python examples/burgers/burgers.py`) or via the {doc}`cli` — both save predictions,
plots, and a `params.msgpack` checkpoint automatically.

## Core PINN examples

```{list-table}
:header-rows: 1
:widths: 18 22 18 27 15

* - Problem
  - PDE
  - Network
  - Key Features
  - Config
* - Exponential Decay
  - `du/dt + λu = 0`
  - MLP `[1,32,32,1]`
  - `ODESolver`, `TrainingConfig`
  - `examples/ode/config.yaml`
* - Harmonic Oscillator
  - `d²u/dt² + ω²u = 0`
  - MLP `[1,32,32,1]`
  - `ODESolver`, IC derivative
  - `examples/ode/config.yaml`
* - FBPINN ODE
  - `du/dx = cos(ωx)`
  - FBPINN — 15 subnets `[1,16,16,1]`
  - Overlapping subdomains, partition-of-unity windows, hard IC constraint
  - `examples/fbpinn_ode/config.yaml`
* - 1-D Burgers
  - `u_t + uu_x = νu_xx`
  - MLP `[2,64,64,64,1]`
  - FBPINN, RBA, cosine LR
  - `examples/burgers/config.yaml`
* - 1-D Heat — Forward
  - `u_t = αu_xx`
  - MLP `[2,64,64,64,1]`
  - `FBPINNSolver`, exact Gaussian IC
  - `examples/heat/heat_forward.yaml`
* - 1-D Heat — Inverse
  - `u_t = αu_xx`
  - MLP `[2,64,64,64,1]`
  - Recover `α` from 50 noisy observations
  - `examples/heat/heat_inverse.yaml`
* - 1-D Wave
  - `u_tt = c²u_xx`
  - FourierMLP `[2,128,128,1]`
  - Dual IC (`u`, `u_t`), `n_fourier=32`
  - `examples/wave/config.yaml`
* - 2-D Helmholtz
  - `Δu + k²u = f`
  - FourierMLP `[2,128,128,1]`
  - `k=4`, manufactured source term
  - `examples/helmholtz/config.yaml`
* - 2-D Diffusion Inverse
  - `u_t = α∇²u`
  - MLP `[3,64,64,64,1]`
  - Log-param joint optimisation
  - `examples/inverse/config.yaml`
```

## Fluid dynamics

```{list-table}
:header-rows: 1
:widths: 18 22 18 27 15

* - Problem
  - PDE
  - Network
  - Key Features
  - Config
* - 2-D Lid-Driven Cavity
  - Steady N-S, Re=100
  - FBPINN + SimpleGate
  - `LDCSolver`, attention
  - `examples/LDC/config.yaml`
* - 2-D RANS k-ε
  - Turbulent channel, Re=10 000
  - FBPINN
  - `RANSSolver`, RBA
  - `examples/K-Epsilon/config.yaml`
* - NACA Airfoil
  - Steady N-S, Re=100
  - MLP / GatedMLP `[2,128×6,3]`
  - Cambered profiles, AoA via rotation, surface `Cp`
  - `examples/airfoil/config.yaml`
* - Cylinder Cross-flow
  - Steady N-S, Re=40
  - MLP `[2,128×6,3]`
  - Pure-PINN recipe, `Cp(θ)` vs inviscid, wake pool
  - `examples/cylinder/config.yaml`
* - 3-D Pipe Flow
  - Steady 3-D N-S
  - MLP / GatedMLP `[3,…,4]`
  - Double-`jacfwd` Hessian, Hagen–Poiseuille exact
  - `examples/pipe_flow/pipe_flow.yaml`
* - 3-D AAA Bulge
  - Steady 3-D N-S
  - GatedMLP `[3,192×5,4]`
  - Cosine² bulge `R(x)`, flow-rate balance
  - `examples/AAA/config.yaml`
* - Carreau Pipe (blood)
  - Steady Carreau N-S
  - GatedMLP `[3,128×4,4]`
  - Shear-thinning `μ(γ̇)`, 1-D Carreau exact
  - `examples/pipe_flow_rheology/config.yaml`
* - Carreau AAA (blood)
  - Steady Carreau N-S
  - GatedMLP `[3,192×5,4]`
  - Blood rheology in the bulge, apparent-viscosity maps
  - `examples/AAA_rheology/config.yaml`
* - 3-D Pulsatile Pipe
  - Unsteady 3-D N-S
  - GatedMLP `[4,…,4]`
  - Time-marching transfer, per-window ckpts, window restart
  - `examples/pipe_flow/pipe_flow_pulsatile_transfer.yaml`
* - 3-D Unsteady Pipe — Transfer
  - `u_t = G + ν∇²u`
  - MLP `[3,64,64,64,64,1]`
  - Bessel exact, Re & temporal transfer
  - `examples/pipe_flow/pipe_flow_unsteady_transfer.yaml`
```

## Compressible flow (shock capturing)

```{list-table}
:header-rows: 1
:widths: 18 22 18 27 15

* - Problem
  - PDE
  - Network
  - Key Features
  - Config
* - 2-D Compressible Ramp
  - Steady Euler (conservative), M=3
  - MLP `[2,80,80,80,80,80,4]`
  - Oblique shock θ=10°, artificial viscosity (fixed/learnable), RAR
  - `examples/ramp/config.yaml`
* - 2-D Compressible NS Ramp (SBLI)
  - Steady N-S (conservative), M=3
  - MLP `[2,128×4,4]`
  - Viscous shock–boundary-layer interaction, no-slip + isothermal walls,
    Re=10⁴, Pr=0.72
  - `examples/ramp_ns/config.yaml`
* - 1-D Sod Shock Tube
  - Unsteady Euler (conservative)
  - MLP `[2,80×5,3]`
  - Learnable `ε = softplus(log_av)`, exact Riemann reference, RAR
  - `examples/sod_shock/config.yaml`
* - 1-D Toro Test 3 (blast wave)
  - Unsteady Euler (conservative)
  - MLP `[2,128×4,3]`
  - exp/log positivity, non-dimensionalisation, learnable `ε`, RAR
  - `examples/toro3/config.yaml`
```

## Transfer-learning examples

```{list-table}
:header-rows: 1
:widths: 22 30 33 15

* - Example
  - Network
  - Key Features
  - Config
* - Burgers Transfer
  - MLP `[2,64,64,64,1]`
  - Parameter transfer (`ν`) + temporal transfer
  - `examples/transfer/burgers_transfer.yaml`
* - Heat 2-D Transfer
  - MLP `[3,64,64,64,1]`
  - Cross-diffusivity transfer + temporal
  - `examples/transfer/heat2d_transfer.yaml`
```

## Neural operator examples

```{list-table}
:header-rows: 1
:widths: 22 15 43 20

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

```{seealso}
{doc}`pde_reference` for the underlying residual classes and
{doc}`geometry_reference` for the domain samplers used across these examples.
```
