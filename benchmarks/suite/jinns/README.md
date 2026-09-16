# jinns — a genuine third-party JAX-native PINN comparison

Closes the "missing discussion of recent JAX-native PINN frameworks (e.g.
jinns)" reviewer gap with a real install and a real, working run on this
machine's GPU, matching the standard the rest of `benchmarks/suite/`
holds itself to (see `../physicsnemo/README.md` for the analogous NVIDIA
PhysicsNeMo work).

**Scope of what's reported: training throughput (ms/epoch) only.** Both
sides were scored against exact/manufactured reference solutions during
development, to confirm each implementation is training a physically
sensible model rather than converging to something wrong -- that
correctness checking is documented inline below where it produced a real
finding worth keeping (e.g. the Pipe Flow residual's exact-zero check
against the analytic solution). But per-problem accuracy numbers and
cross-framework accuracy comparisons are not the claim this file makes;
every table below reports ms/epoch only.

## Install

jinns (`pip install jinns`, real PyPI package, not to be confused with any
similarly-named project) lives in its own isolated venv here
(`.venv`, `--system-site-packages`), exactly like `../physicsnemo/.venv`.
Installed cleanly with no build issues on this GB10 machine -- no GPU/CUDA
compatibility problems to document here, unlike PhysicsNeMo's six.

```bash
cd benchmarks/suite/jinns
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install jinns
```

The real project lives on GitLab, not GitHub
(`https://gitlab.com/mia_jinns/jinns`, docs at
`https://mia_jinns.gitlab.io/jinns/`) -- we had first guessed a GitHub URL
for the paper's bibliography entry before installing it and checking `pip
show jinns`'s `Project-URL` field directly; fixed before it became a wrong
citation in the actual paper.

## Six problems, one settings audit each

Every comparison below matches physics, network depth/width, collocation
counts, per-step batch sizes, boundary-condition semantics, and optimizer
(plain Adam with a cosine schedule -- jinns' own example notebook for
Burgers uses its `vanilla_ngd` natural-gradient optimizer instead; Adam is
used here specifically to avoid confounding the comparison with an
optimizer choice) as closely as jinns' data-generator/loss API allows,
built by reading jinns' own source rather than assuming API behavior from
its docstrings or argument names.

### 1. Burgers 1D (`burgers_jinns.py`)

Matches `../baselines/burgers_baselines.py` /
`../physicsnemo/compare_underpinn.py`: `u_t + u u_x = nu u_xx`, nu=0.01,
domain x in [-1,1], t in [0,1.5], IC `u(x,0)=-sin(pi x)`, BC `u(+-1,t)=0`,
5 hidden layers x 64 tanh, full-batch every step (`domain_batch_size=None`
-- confirmed from source this means the *same* n-point pool every
iteration, not per-step resampling), N_r/N_ic/N_bc = 20000/200/300(x2
sides), IC/BC loss weight 100/10.

**One API detail that would have silently broken the physics if we had
trusted the docstring's plain-English description over the actual
source:** jinns normalizes the PINN's time input to `[0, 1]` and takes
`Tmax` as a separate scalar that the dynamic-loss residual is multiplied
by (`jinns.loss.BurgersEquation.equation`'s literal source:
`du_dtx[0:1] + Tmax * (u * du_dx - nu * d2u_dx2)`), rather than feeding
physical time directly. Confirmed `jinns.loss.Dirichlet` is a soft penalty
(squared network output at the boundary, added to the loss), not a hard
architectural constraint, by reading `Dirichlet.equation_u`'s source
directly.

### 2. Heat 2D steady / Poisson (`heat2d_jinns.py`)

Matches `../physicsnemo/compare_underpinn_multi.py::run_heat`:
$\nabla^2 u + f = 0$, $f=2\pi^2\sin(\pi x)\sin(\pi y)$, domain
$(x,y)\in[0,1]^2$, $u=0$ on all four edges. **Plain MLP**, 3 hidden layers
x 64 -- chosen deliberately: this is the one problem in the whole
comparison set where underPINN itself already uses a plain MLP rather
than FourierMLP, so there is no architecture gap to reason about on
either side. Minibatched every step (`batch_r=2048`/5000-point pool,
`batch_b=256`/1200-point boundary pool), unlike Burgers' full-batch
convention, via jinns' `omega_batch_size`/`omega_border_batch_size`.

jinns has no built-in Poisson/Helmholtz dynamic loss, so this uses its
documented extension point: a `jinns.loss.PDEStatio` subclass implementing
`equation(x, u, params)`, built on jinns' own `laplacian_rev` operator --
the same operator jinns' built-in `FisherKPP` dynamic loss uses
internally, confirmed by reading its source.

**A minibatching-semantics difference, found by reading
`jinns.data._CubicMeshPDEStatio.inside_batch`'s source rather than
assuming `omega_batch_size` behaves like underPINN's own minibatching:**
jinns walks the shuffled n-point interior pool in sequential,
non-overlapping chunks, reshuffling once the pool is exhausted --
standard epoch-based SGD. underPINN's `safe_choice` instead draws a fresh
independent random index set from the full pool on *every* step. Both are
minibatch SGD over the same candidate pool and batch size, so per-step
compute cost is identical either way -- this affects what each step
*sees*, not how expensive it is, so it does not affect the throughput
comparison below, but is flagged rather than glossed over.

### 3. Diffusion 1D (`diffusion_jinns.py`)

No pre-existing example script existed for this problem (unlike Burgers/
Heat/Wave/Helmholtz/ODE), so this config -- and the matching underPINN-side
`run_diffusion` in `compare_underpinn_multi.py` -- were built directly
from `underPINN.pde.diffusion.DiffusionPDE`'s own docstring's canonical
test case: `u_t = alpha*u_xx`, alpha=0.01, domain x in [0,1], t in [0,1],
IC `u(x,0)=sin(pi x)`, BC `u(0,t)=u(1,t)=0`. Plain MLP, 3 hidden layers x
64. jinns has no built-in pure-diffusion loss, so this defines an
explicit `DiffusionEquation(PDENonStatio)` (same Tmax-scaling convention
as `BurgersEquation`, using `laplacian_rev`).

**A real bug caught before either side's numbers could be trusted:** the
first version of both this script and its underPINN-side counterpart used
`N_IC=200 < BATCH_I=256` -- jinns' sequential-chunk minibatching cannot
draw a batch larger than its pool and errors outright
(`slice_sizes must be less than or equal to operand shape`), which is what
caught it; the underPINN side would have silently fallen back to
with-replacement oversampling instead (`safe_choice`'s documented
behavior) rather than erroring, so this would have gone unnoticed on that
side alone. Fixed by raising `N_IC` to 300 on both sides.

### 4. Heat 2D unsteady (`heat2d_unsteady_jinns.py`)

Matches the new `compare_underpinn_multi.py::run_heat2d_unsteady` (built
fresh from `UnsteadyHeat2DPDE`'s own docstring, `layers=[3,64,64,64,64,1]`
taken from `examples/transfer/heat2d_transfer.py`'s `LAYERS` constant):
`u_t = alpha*(u_xx+u_yy)`, domain `(x,y) in [0,1]^2`, `t in [0,1]`, IC
`u(x,y,0)=sin(pi x)sin(pi y)`, BC `u=0` on all four edges. This one reuses
jinns' own **built-in** `FisherKPP` dynamic loss directly, with its
reaction coefficients zeroed (`r=0, g=0`) -- confirmed from
`FisherKPP.equation`'s source that this reduces exactly to
`u_t = D*laplacian(u)`, not assumed from the class name. No new
`DynamicLoss` subclass needed; `FisherKPP` already supports the 2-D
spatial dimension via its `dim_x` field.

### 5. ODE harmonic oscillator (`ode_harmonic_jinns.py`)

Matches `run_ode_harmonic` (`omega=2`, `T_MAX=5`, IC `u(0)=1, u'(0)=0`,
exact `u(t)=cos(omega t)`). Plain MLP, 3 hidden layers x 64.

**A real API limitation, found rather than worked around silently:**
`jinns.loss.LossODE`'s `initial_condition` only supports a *value*
constraint `u(t0)=u0` (confirmed from `_LossODE.py`'s source) -- there is
no built-in mechanism for this problem's derivative IC, `u'(0)=0`. Rather
than skip the problem or drop that constraint, this script bypasses
`LossODE`/`jinns.solve` and writes a small composite loss directly: a
genuine `jinns.loss.ODE` subclass (`HarmonicOscillatorEquation`, built the
same way jinns' own `GeneralizedLotkaVolterra` is -- jinns ships no
harmonic-oscillator equation itself) for the interior residual, plus the
derivative-IC term added with a plain `jax.grad` call mirroring
`underPINN.pde.ode.HarmonicOscillatorODE.ut`'s own formula exactly. Only
the top-level training loop is hand-rolled (`jax.value_and_grad` +
`optax`, not `jinns.solve`); the PINN class, the ODE dynamic-loss base
class, and the collocation sampler (`jinns.data.DataGeneratorODE`) are all
genuine jinns machinery.

The Tmax-scaling for a *second*-order-in-time equation was hand-derived
(t_tilde = t/Tmax implies d^2/dt_physical^2 = (1/Tmax^2) d^2/dt_tilde^2,
so the whole equation scales by Tmax^2, not Tmax as in the first-order
cases above) rather than copied from an existing jinns example, so it was
checked empirically rather than trusted on its own: the trained network's
predictions were confirmed to track `cos(2t)` closely, not just to have a
low loss value -- a sign or scaling error in the IC-derivative or Tmax
terms would have prevented convergence to *any* sensible periodic
function, so this convergence itself is direct evidence the derivation is
right.

### 6. Pipe Flow 3D / Navier-Stokes (`pipe_flow_jinns.py`,
### `../physicsnemo/pipe_flow_matched_jinns.py`)

The most physically interesting of the eight PhysicsNeMo-comparison
problems, attempted despite GatedMLP having no jinns equivalent (plain MLP
used on both sides, disclosed rather than silently substituted).

**Two real jinns API gaps, found rather than assumed away:**

1. jinns' convective-term utility, `_u_dot_nabla_times_u_rev`, is
   hard-coded to 2-D inputs (`assert x.shape[0] == 2` in its own source)
   -- unusable for this 3-D problem. `(u.grad)u` is hand-written here with
   `jax.jacfwd` instead. The diffusion term still uses jinns' own
   `vectorial_laplacian_rev` (dimension-general, confirmed from source)
   and continuity uses jinns' own `divergence_rev` (also
   dimension-general) -- genuine jinns machinery for the two terms that
   fit, hand-written only for the one that doesn't.
2. This problem's geometry (a cylinder, with a different boundary
   condition per region -- wall/inlet/outlet) does not fit jinns'
   `CubicMeshPDEStatio`/`Dirichlet` machinery (an axis-aligned box, one
   condition per facet). Like `ode_harmonic_jinns.py`, this bypasses
   `jinns.solve` for a manual composite loss + training loop.

**Verified before trusting a long run, not assumed correct:** the custom
`pipe_ns_residual` function was checked against the exact analytic
Hagen-Poiseuille solution (a genuine ground-truth test available here
specifically because the nonlinear convective term vanishes exactly for
fully-developed flow, noted in `SteadyNS3DPDE.exact_poiseuille`'s own
docstring) -- residual came back **exactly zero** at five random interior
points, not just close to it.

Also disclosed: both this comparison's scale (`N_interior=20,000`,
`epochs=30,000`) and network (plain MLP) are reduced/adjusted from
`compare_underpinn_multi.py::run_pipe_flow`'s full PhysicsNeMo-matched
config (100,000 points, GatedMLP, 70,000 epochs) -- verifying a
from-scratch jinns port at the full scale was not tractable in this pass.
`pipe_flow_matched_jinns.py` reruns underPINN's own side at this same
reduced scale/architecture so the two throughput numbers below are
genuinely comparable to each other. Both sides reuse underPINN's own
`underPINN.geometry.pipe.Pipe` sampler directly (same class, same seeds),
so interior/wall/inlet/outlet points are statistically identical on both
sides, not just matched in count.

## Settings audit: is this actually apples-to-apples for throughput?

Asked directly whether "the settings are the same," so we checked rather
than assumed, the same way the PhysicsNeMo `weight_norm` confound was
found -- reading jinns' actual source for each candidate rather than
trusting argument names or docstrings, across all six problems:

| aspect | jinns (verified from source) | underPINN | matched? |
|---|---|---|---|
| numerical precision | float32 (`jax_enable_x64` stays `False` even after `import jinns`) | float32 | **yes** |
| loss-term aggregation | `jnp.mean(...)` in `LossPDENonStatio`/`LossPDEStatio` (`mean_sum_reduction` computes `jnp.mean(jnp.sum(residuals**2, axis=-1))`) | `jnp.mean(res**2)` | **yes** |
| per-step batch sizes | matched per problem (see each problem's section above) | matched | **yes** |
| interior-pool minibatching scheme | sequential chunks with periodic reshuffle | fresh random draw every step | **no** -- real difference, but same per-step cost either way (Heat 2D steady section above) |
| boundary-facet loss aggregation (Heat 2D steady only) | mean-per-facet, summed across 4 facets | single pooled mean | **no** -- real difference, but a loss-weight effect, not a compute-cost one |
| boundary condition | soft penalty (`Dirichlet.equation_u` returns raw `u(inputs, params)`, squared and weighted) | soft penalty | **yes** |
| Adam hyperparameters | `optax.adam(sched)` (standard defaults) | `optax.chain(scale_by_adam(), scale_by_schedule(sched), scale(-1.0))` (same standard defaults) | **yes**, mathematically equivalent |
| weight/bias initialization | `eqx.nn.Linear` default: `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))` for both weight and bias | Flax `nn.Dense` default: `lecun_normal()` weight, zero bias | **no** -- real difference, but does not change per-step compute cost |

The two confirmed differences (minibatching scheme, boundary-facet
aggregation) both change what the network *sees* or how the loss is
*weighted*, not how expensive a training step is -- neither is a
confound for the throughput numbers reported below. Weight/bias
initialization only affects starting values, not per-step cost, for the
same reason.

## Results (this GPU, single run each, ms/epoch only)

| problem | epochs | underPINN | jinns | jinns faster? |
|---|---|---|---|---|
| Burgers 1D | 5,000 | **8.219** | 10.267 | no, underPINN 1.25x faster |
| Heat 2D steady | 5,000 | **0.396** | 0.995 | no, underPINN 2.51x faster |
| Diffusion 1D | 5,000 | **0.427** | 0.935 | no, underPINN 2.19x faster |
| Heat 2D unsteady | 5,000 | **0.944** | 1.852 | no, underPINN 1.96x faster |
| ODE Harmonic | 3,000 | **0.480** | 0.500 | no, essentially tied (underPINN 1.04x faster) |
| Pipe Flow 3D (reduced scale) | 30,000 | **8.163** | 17.143 | no, underPINN 2.10x faster |

**underPINN is faster than jinns on every problem tested**, by a margin
ranging from essentially tied (ODE Harmonic, 1.04x) to 2.51x (Heat 2D
steady). Burgers' number is itself a rerun: the original comparison had
jinns measuring faster (underPINN at 14.617ms/epoch, before a library-wide
fix removed a real redundant-autodiff-transform bottleneck in
`underPINN/pde/burgers.py`, documented in
`benchmarks/suite/README.md` section 6) -- `compare_underpinn.py`'s
Burgers residual is a self-contained duplicate of that library code, not
an import of it, so it needed the identical fix applied separately before
this number reflected the actual library state. jinns is in turn faster
than PhysicsNeMo Sym on every problem where a PhysicsNeMo number exists
(Burgers: 26.773ms; Heat 2D steady: 5.822ms; ODE Harmonic: 6.705ms),
consistent with this suite's own dispatch-overhead story -- PhysicsNeMo's
eager-mode PyTorch dispatch is the common factor both JAX-native
frameworks beat.

**Scope note:** six problems, one seed each, one jinns version (1.10.0).
The PhysicsNeMo comparison covers eight matched problems (seven of which
now overlap with this jinns comparison: Burgers, Heat 2D, ODE Harmonic,
Pipe Flow here, plus Wave 1D and Helmholtz 2D not yet attempted -- both
use FourierMLP with no jinns equivalent and, for Wave, the same
derivative-IC complication as ODE Harmonic); extending the jinns
comparison to that same scope, and to Pipe Flow's full 100,000-point/
GatedMLP/70,000-epoch configuration specifically, remains future work.
