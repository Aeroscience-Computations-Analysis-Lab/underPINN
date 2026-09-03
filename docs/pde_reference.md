# PDE Library Reference

Every PDE in underPINN implements a single `residual()` method conforming to
`BasePDE` (see {doc}`api_reference`). This page lists every registered residual class,
its governing equation, and the examples that use it.

```{list-table}
:header-rows: 1
:widths: 26 30 24 20

* - PDE
  - Equation
  - Key method
  - Used in
* - Burgers (1-D)
  - `u_t + uu_x = νu_xx`
  - `BurgersPDE.residual`
  - `examples/burgers/`, `examples/transfer/`
* - Diffusion / Heat (1-D)
  - `u_t = αu_xx`
  - `DiffusionPDE.residual`
  - `examples/heat/`
* - Heat (2-D unsteady)
  - `u_t = α(u_xx + u_yy)`
  - `Heat2DPDE.residual`
  - `examples/inverse/`, `examples/transfer/`
* - Wave (1-D)
  - `u_tt = c²u_xx`
  - `WavePDE.residual`
  - `examples/wave/`
* - Helmholtz (2-D)
  - `Δu + k²u = f`
  - `HelmholtzPDE.residual`
  - `examples/helmholtz/`
* - Navier-Stokes (2-D steady)
  - `∇·u=0`, `u·∇u = -∇p + ν∇²u`
  - `NavierStokesPDE.residual`
  - `examples/LDC/`, `examples/airfoil/`, `examples/cylinder/`
* - Navier-Stokes (3-D steady)
  - Same + z-momentum
  - `SteadyNS3DPDE.residual`
  - `examples/pipe_flow/`, `examples/AAA/`
* - Navier-Stokes (3-D unsteady)
  - `u_t + (u·∇)u = −∇p + ν∇²u`
  - `UnsteadyNS3DPDE.residual`
  - `examples/pipe_flow/` (pulsatile)
* - Carreau N-S (3-D steady)
  - `∇·[μ*(γ̇)(∇u+∇uᵀ)]` stress
  - `CarreauNS3DPDE.residual`
  - `examples/pipe_flow_rheology/`, `examples/AAA_rheology/`
* - Pipe unsteady
  - `u_t = G + ν(u_yy + u_zz)`
  - `PipeUnsteadyPDE.residual`
  - `examples/pipe_flow/`
* - RANS k-ε
  - N-S + `k` + `ε` transport
  - `KEpsilonPDE.residual`
  - `examples/K-Epsilon/`
* - Compressible Euler (2-D steady)
  - `∂F/∂x + ∂G/∂y = ε∇²U` (conservative)
  - `CompressibleEulerPDE.residual`
  - `examples/ramp/`
* - Compressible Navier–Stokes (2-D steady)
  - `∂x(F−Fv/Re) + ∂y(G−Gv/Re) = 0`
  - `CompressibleNS2DPDE.residual`
  - `examples/ramp_ns/`
* - Compressible Euler (1-D unsteady)
  - `∂U/∂t + ∂F/∂x = ε∂²U/∂x²`
  - `Euler1DUnsteadyPDE.residual`
  - `examples/sod_shock/`
* - Exponential Decay
  - `du/dt + λu = 0`
  - `ExpDecayODE.residual`
  - `examples/ode/`
* - Harmonic Oscillator
  - `d²u/dt² + ω²u = 0`
  - `HarmonicODE.residual`
  - `examples/ode/`
* - Burgers grid (1-D, FNO)
  - `u_t + uu_x = νu_xx` (FD residual)
  - `BurgersGrid1D.residual`
  - `examples/operators/fno1d_periodic/`, `fno1d_dirichlet/`
* - Burgers grid (2-D, FNO/CViT)
  - `u_t + u(u_x+u_y) = ν(u_xx+u_yy)` (FD residual)
  - `BurgersGrid2D.residual`
  - `examples/operators/fno2d_burgers/`, `cvit2d_burgers/`
* - Burgers (DeepONet)
  - `u_t + uu_x = νu_xx` (autodiff residual)
  - `DeepONetBurgersPDE.residual`
  - `examples/operators/deeponet1d_burgers/`
* - Navier-Stokes grid (2-D, FNO)
  - `∇·u=0`, `u·∇u = -∇p + ν∇²u` (FD residual)
  - `CylinderNSGrid.residual`
  - `examples/operators/fno2d_cylinder/`
```

```{note}
Module paths live under `underPINN/pde/` — e.g. `underPINN.pde.burgers.BurgersPDE`,
`underPINN.pde.navier_stokes_3d.SteadyNS3DPDE`. See {doc}`api_reference` for the
package layout.
```
