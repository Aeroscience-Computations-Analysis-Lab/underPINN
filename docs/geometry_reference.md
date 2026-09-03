# Geometry Reference

```{list-table}
:header-rows: 1
:widths: 20 45 35

* - Class
  - What it samples
  - Example uses
* - `Interval`
  - 1-D uniform or Sobol interior + boundary points
  - Burgers, wave, heat (1-D)
* - `Rectangle`
  - 2-D interior (LHS / Sobol) + all four boundary edges
  - Helmholtz, LDC, diffusion inverse
* - `NACAAirfoil`
  - NACA 4-digit (symmetric & cambered) exterior domain, SDF-weighted
    near-surface, AoA via quarter-chord rotation
  - `examples/airfoil/`
* - `Cylinder2D`
  - Circular cylinder exterior cross-flow domain, analytic SDF, surface points
  - `examples/cylinder/`
* - `Pipe`
  - 3-D cylindrical interior, lateral wall, circular inlet, circular outlet
  - `examples/pipe_flow/`, `examples/pipe_flow_rheology/`
* - `BulgeGeometry`
  - Axisymmetric AAA bulge `R(x)` (cosine²): interior, curved wall, inlet, outlet
  - `examples/AAA/`, `examples/AAA_rheology/`
* - `Ramp`
  - Trapezoidal domain above a wedge surface at angle θ
  - `examples/ramp/`
* - `Composite`
  - Boolean union / intersection / difference of any two geometry objects
  - LDC (cavity minus any obstacle)
* - `ShapelyGeom`
  - Arbitrary 2-D polygon backed by Shapely 2.x; rejection-samples interior
  - Custom geometries
```

## Notable design details

```{admonition} Airfoil angle of attack via rotation, not inflow tilt
:class: note
`NACAAirfoil` imposes angle of attack (AoA) by **rotating the airfoil geometry**
about the quarter-chord point, rather than tilting the free-stream inflow direction.
This keeps the inflow boundary condition horizontal and simple, while still producing
the correct relative flow angle.
```

```{admonition} SDF-weighted near-surface sampling
:class: tip
Both `NACAAirfoil` and `Cylinder2D` expose analytic signed-distance functions (SDFs)
that bias collocation-point density toward the body surface — where gradients are
steepest — without hand-tuned refinement zones.
```

## Every geometry implements a common sampler interface

```python
class Pipe:
    def sample_interior(self, n, key): ...
    def sample_wall(self, n, key): ...
    def sample_inlet(self, n, key): ...
    def sample_outlet(self, n, key): ...
```

The exact method set varies slightly by dimensionality (e.g. `Rectangle` exposes four
boundary-edge samplers instead of `wall`/`inlet`/`outlet`), but every geometry class
follows the same `sample_<region>(n, key) -> array` convention, making it
straightforward to swap geometries within an existing PDE/loss/solver stack.

```{seealso}
{doc}`pde_reference` for the PDE residual classes each geometry pairs with, and
{doc}`examples` for the full worked-example catalogue.
```
