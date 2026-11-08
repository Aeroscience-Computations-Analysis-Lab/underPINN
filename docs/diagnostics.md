(diagnostics)=
# Training & Network Diagnostics

A loss curve says *that* a PINN trained, not *why* it trained the way it did.
`underPINN.diagnostics` turns a trained (or part-trained) network into explanations you
can act on. It answers questions like these:

- Which loss term limits accuracy?
- Is the landscape sharp or ill-conditioned, and why?
- Does the input representation suit the physics?
- Is the output layer converged?
- How wide does the network really need to be?
- Where does the remaining error live?

The diagnostics are **method-agnostic**: they do not depend on which solver, optimiser or
network produced the parameters. You describe the problem once with a small `Problem`
adapter. Then you call one function per training step and one per checkpoint.

Every diagnostic is checked in `tests/test_diagnostics.py`, both against exact identities
and on two problems with known answers: the forced, damped spring of the quick start and a
multiscale Poisson equation.

## Quick start

The forced, damped spring {math}`m\ddot u + c\dot u + ku = F_0\cos\Omega t` with a
standard `MLP`:

```python
import jax
import jax.numpy as jnp
import optax

from underPINN import diagnostics as dg
from underPINN.nn import MLP

# m u'' + c u' + k u = F0 cos(Om t),  u(0) = 1,  u'(0) = 0
m, c, k, F0, Om, T = 1.0, 1.2, 40.0, 20.0, 4.0, 3.0
model = MLP(layers=[1, 32, 32, 1])
params0 = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)))

def u(p, t):
    return model.apply(p, jnp.array([[t / T]]))[0, 0]
u_t = jax.grad(u, 1)
u_tt = jax.grad(u_t, 1)

def residual_fn(p, x):          # (N, 1) points -> (N, 1) residual rows
    def r(t):
        lhs = m * u_tt(p, t) + c * u_t(p, t) + k * u(p, t)
        return (lhs - F0 * jnp.cos(Om * t)) / k
    return jax.vmap(r)(x[:, 0])[:, None]

def ic(p):                      # velocity error scaled by omega_0
    return u(p, 0.0) - 1.0, u_t(p, 0.0) / jnp.sqrt(k / m)

def loss_terms(p, x):           # weighted terms; the loss is their sum
    a, b = ic(p)
    r = residual_fn(p, x)
    return {"residual": jnp.mean(r ** 2), "ic": a ** 2 + b ** 2}

def rho_fn(p, x):               # the same loss as a vector: sum(rho**2)
    a, b = ic(p)
    r = residual_fn(p, x)[:, 0] / jnp.sqrt(x.shape[0])
    return jnp.concatenate([r, jnp.stack([a, b])])

x = jnp.linspace(0.0, T, 256)[:, None]
problem = dg.Problem(
    loss_terms, x,
    residual_fn=residual_fn, x_res=x, rho_fn=rho_fn,
    apply_fn=lambda p, xx: jax.vmap(lambda t: u(p, t))(xx[:, 0]),
    hidden_fn=dg.flax_hidden_fn(model, input_transform=lambda xx: xx / T),
)
```

Attach the monitor to any training loop. This one is a plain Adam loop, but the same
`mon.update` line works inside a solver callback or a custom loop:

```python
opt = optax.adam(5e-3)
params, opt_state = params0, opt.init(params0)

@jax.jit
def step(p, s):
    loss, g = jax.value_and_grad(problem.loss)(p)
    upd, s = opt.update(g, s)
    return optax.apply_updates(p, upd), s, loss

mon = dg.TrainingMonitor(problem, params0, every=100)
for it in range(3000):
    mon.update(it, params)      # log theta_k before the step
    params, opt_state, _ = step(params, opt_state)
mon.save("monitor.npz")

best = mon.best_params()        # report the best iterate, not the last
k_hit = dg.iterations_to(mon.total(), 1e-3, steps=mon.rec["step"])
print("k(1e-3) =", k_hit)
```

Then run the checkpoint diagnostics on the trained network:

```python
x_dense = jnp.linspace(0.0, T, 4096)[:, None]
cfg = dg.Config(refit_layer="Dense_2", prune=True, x_dense=x_dense)
res = dg.run_checkpoint(problem, best, params0=params0, cfg=cfg,
                        out_dir="diag", step=mon.best_step)

curv = res["curvature"]
print(curv["lambda_max"], curv["n_eff"], curv["kappa_top"])
print(res["gauss_newton"]["gn_rel_err"], res["output_refit"]["ratio"])
print(res["offgrid"]["ratio"])
print({name: h["erank"] for name, h in res["hidden"].items()})
print(res.get("skipped", "nothing skipped"))
```

`run_checkpoint` writes `diag/ckpt_<step>.json`, which holds the scalars and short lists,
and `diag/ckpt_<step>.npz`, which holds every array: spectra, kernels, slices and singular
values. Call it at several steps to follow the diagnostics through training.

## The `Problem` contract

```{list-table}
:header-rows: 1
:widths: 26 34 40

* - Field
  - Signature
  - Needed by
* - `loss_terms_fn`, `batch`
  - `(params, batch) -> {name: scalar}`; the loss is the sum of the terms
  - everything (**required**)
* - `residual_fn`, `x_res`
  - `(params, x) -> (N, n_eq)` PDE residual rows; `x_res` = collocation points
  - NTK, off-grid check, representation (`x_res` is also where activations are sampled)
* - `rho_fn`
  - `(params, batch) -> (M,)` with **`loss == sum(rho**2)`**
  - Gauss–Newton fidelity, output-layer refit, refit-pruning
* - `apply_fn`
  - `(params, x) -> (N,)` or `(N, 1)`: the network output
  - solution error and its spectrum, modal kernel
* - `hidden_fn`
  - `(params, x) -> {layer: (z, h)}`: pre- and post-activations
  - effective rank, saturation, probes, CKA, ablation
* - `group_fn`
  - parameter path → layer label (default: the flax module name, e.g. `Dense_0`)
  - every per-layer quantity
```

Only `loss_terms_fn` and `batch` are required. If a diagnostic needs a missing field, it is
skipped, and the reason is recorded in `res["skipped"]`. A diagnostic never stops a training
run. The batch is **fixed**, so every curvature and landscape number is a deterministic function
of the parameters, not of a stochastic mini-batch.

`dg.flax_hidden_fn(model, input_transform=...)` builds `hidden_fn` for any flax chain of
`nn.Dense` layers (`MLP`, `FourierMLP`). It does not cover networks with learned activation
slopes; pass your own function for those.

```{admonition} rho must reproduce the loss exactly
:class: warning
The Gauss–Newton and least-squares diagnostics all assume the loss is
{math}`\mathcal L(\theta)=\lVert\rho(\theta)\rVert^2`. Every loss term must appear in `rho`,
scaled to match: a mean-squared residual over N points contributes {math}`r_i/\sqrt N`, and
a weighted term {math}`\lambda\,e^2` contributes {math}`\sqrt\lambda\,e`. `Problem.check_rho`
compares the two before any diagnostic relies on them. A mismatch raises a `ValueError`
naming both values, instead of returning a silently wrong answer.
```

### Building a `Problem` from loss blocks

Writing `loss_terms_fn` and `rho_fn` by hand is the easiest way to get a silently wrong
curvature number. Describe the loss as weighted mean-squared **blocks** instead, and the
loss, `rho` and the PDE residual are derived from one definition:

```python
from underPINN import diagnostics as dg

def u(p, xt):                   # (N, 2) points -> (N,) network output
    return model.apply(p, xt)[:, 0]

def misfit(p, z):               # z = [x, t, target]
    return u(p, z[:, :2]) - z[:, 2]

blocks = {
    "pde": dg.LossBlock(pde.residual, xt_r, weight=1.0),
    "ic": dg.LossBlock(misfit, ic_points, weight=100.0),
    "bc": dg.LossBlock(misfit, bc_points, weight=10.0),
}
problem = dg.problem_from_blocks(blocks, residual_block="pde", apply_fn=u)
```

Each block contributes `weight · mean(fn(params, points)**2)`. Row *i* of `fn` must depend
only on `points[i]`, which holds for every PINN residual. Targets can ride along as extra
columns of `points`. For underPINN's `PINNLoss`, one call reproduces the training loss:

```python
problem = dg.problem_from_pinnloss(loss, x_r, t_r, x_i, u_i, x_b, t_b, u_b)
```

This handles the IC, BC, initial-velocity (`ic_velocity_weight`) and periodic
(`periodic_length`) terms. RBA and weight regularisation are not sums of fixed squares; they
are left out with a warning.

Blocks also enable the per-term NTK (`term_ntk`, see *Loss balance, causality and
localisation* below).

For networks that are not plain Dense chains (PirateNet, KANs, PINNsFormer, PINNMamba, or
your own), use `dg.generic_hidden_fn(model)`. It captures every submodule's output through
flax `capture_intermediates`, leaving out the layer that produces the network output, so
the representation diagnostics work for any architecture.

## Diagnostics checklist

In the "Where" column, ★ marks quantities logged during training by `TrainingMonitor`. The
other rows come from `run_checkpoint` or the standalone functions.

```{list-table}
:header-rows: 1
:widths: 22 22 28 28

* - Diagnostic
  - Where
  - What to look for
  - Design decision it informs
* - Loss-term balance ★
  - `rec["terms"]`, `rec["grad_term"]`
  - the term that stalls highest limits training
  - loss weights, adaptive weighting
* - Update ratio per layer ★
  - `rec["update"]`
  - late values > 1e-2: steps too large; a layer far below the others is frozen
  - learning rate, schedule, per-layer rates
* - Distance from initialisation ★
  - `rec["dist0"]`, `res["dist0"]`
  - ≪ 1 lazy (kernel) regime; ~1 feature learning
  - input features, initialisation scale, width
* - Best vs final iterate ★
  - `mon.best_params()`, `best_so_far`
  - final iterate much worse than the best
  - keep the best iterate; decay the learning rate
* - Sharpness, effective dimension
  - `res["curvature"]`
  - rising λ_max tightens the step limit; small n_ε means ill-conditioning
  - step size, warm-up, decay; representation
* - κ₁₀ and Jacobi-scaled κ₁₀
  - `res["curvature"]`
  - κ₁₀ that stays large after Jacobi scaling: correlations, not scale
  - representation (a diagonal preconditioner cannot fix it)
* - Per-layer curvature
  - `res["layer_sharpness"]`
  - which block is stiff
  - per-layer learning rates; output-layer refit
* - Gauss–Newton fidelity
  - `res["gauss_newton"]`
  - small: curvature = Gram matrix of the residual Jacobian
  - whether Jacobian-based reasoning applies
* - Residual NTK spectrum and drift
  - `res["ntk"]`
  - fast spectral decay; large drift = features being learned
  - representation, width
* - Loss slices, non-convex count
  - `res["slices"]`
  - slices with more than one extremum
  - representation; initialisation
* - Effective rank, saturation
  - `res["hidden"]`
  - erank ≪ width: redundant units; rising saturation
  - width; initialisation scale, normalisation
* - Linear probes, CKA, ablation
  - `linear_probe`, `linear_cka`, `ablation_importance`
  - where the solution becomes linearly available; similarity across seeds and networks
  - where information forms; ensembling; feature choice
* - Output-layer refit
  - `res["output_refit"]`
  - loss drops on refit: Adam had not converged the last layer
  - hybrid least-squares + Adam
* - Refit-pruning
  - `res["pruning"]`
  - loss holds with far fewer units
  - width
* - Off-grid residual
  - `res["offgrid"]`
  - dense/collocation RMS well above 1
  - number and placement of collocation points
* - Error spectrum
  - `res["error"]`, `amplitude_spectrum`
  - error lingering at frequencies known from the physics
  - Fourier or physics features
* - Modal error and modal kernel
  - `modal_error`, `modal_kernel`
  - is a mode's slow convergence due to the representation (small μ_n) or the loss (small w_n)?
  - representation vs loss formulation
* - Term-gradient conflict ★
  - `rec["grad_cos_min"]`, `res["gradient_conflict"]`
  - gradient norm ratio ≫ 1; negative cosines while the error is large
  - loss weights (LRA), gradient surgery, hard constraints
* - Per-term NTK
  - `res["term_ntk"]` (needs blocks)
  - effective rates that differ by orders of magnitude
  - NTK-balanced weights
* - Residual and error along time
  - `res["residual_profile"]`, `res["error_profile"]`
  - late-time residual or error far above early; small causal weights
  - causal training, time marching
* - Residual concentration and residual–error correlation
  - `res["residual_concentration"]`, `res["residual_error_correlation"]`
  - residual in a few points; does it track the error?
  - adaptive sampling (RAR-D) or not; singularities
* - N-D spectral error
  - `res["spectral_error"]`
  - resolved bandwidth far below the solution's band
  - Fourier features, multiscale networks, FBPINN
* - Findings
  - `res["findings"]`, `res["findings_md"]`
  - the rules' verdict, with evidence
  - where to start
```

## Definitions

Each family below gives the definition, what a value means, and the function that
computes it.

### Training dynamics: `TrainingMonitor`

`mon.update(k, params)` returns immediately unless `k % every == 0`. On logged steps it
evaluates the following on the fixed diagnostic batch:

- **terms**: each weighted loss term. The term that stalls highest is the one limiting
  accuracy. A term that drops to round-off is over-weighted relative to what the others can
  reach.
- **grad_term**: {math}`\lVert\nabla_\theta \mathcal L_j\rVert` for each term, all from one
  backward pass. It shows which term actually drives the update; compare it with `terms`
  before reweighting.
- **update**: {math}`\rho_\ell(k)=\lVert\theta_\ell(k)-\theta_\ell(k')\rVert/\lVert\theta_\ell(k')\rVert`
  per layer, where k′ is the previous logged step. With `every=1` this is the per-step ratio.
- **dist0**: {math}`\delta_\ell(k)=\lVert\theta_\ell(k)-\theta_\ell(0)\rVert/\lVert\theta_\ell(0)\rVert`.
  Large movement of the hidden layers means the representation is doing work the input
  features should have done.
- **best**: the logged iterate with the lowest total. `mon.best_params()` returns it, at no
  extra cost. Use `every=1` to get the true argmin over all iterates.

Call `update` with {math}`\theta_k` **before** the optimiser step, so the recorded loss is
{math}`\mathcal L(\theta_k)`. Call `mon.reset_best()` when the objective changes, for example
at a new training stage or time window.

Two efficiency measures from a loss history:

```python
dg.best_so_far(losses)                       # b_k = min_{j<=k} L(theta_j)
dg.iterations_to(losses, 1e-4, steps=steps)  # k(eps), or None if never reached
```

### Curvature

```{list-table}
:header-rows: 1
:widths: 30 70

* - Quantity
  - Definition
* - Sharpness
  - {math}`\lambda_{\max}(H)`, {math}`H=\nabla^2\mathcal L`. Gradient descent is stable only
    for {math}`\eta<2/\lambda_{\max}`.
* - Effective dimension
  - {math}`n_\epsilon=\#\{i:\lambda_i>\epsilon\lambda_{\max}\}` with
    {math}`\epsilon=10^{-3}` (`Config.eps`)
* - Top-ten condition number
  - {math}`\kappa_{10}=\lambda_1/\lambda_{10}` (`kappa_top`, `Config.n_top`). The classical
    {math}`\lambda_{\max}/\lambda_{\min}` is unusable: an over-parameterised network has many
    eigenvalues within rounding of zero.
* - Jacobi-scaled κ₁₀
  - {math}`\kappa_{10}` of {math}`D^{-1/2}HD^{-1/2}`, {math}`D=\operatorname{diag}(H)`. It is
    an idealised proxy for Adam's diagonal preconditioning: per-parameter scale is removed,
    while correlations inherited from the representation remain.
* - Per-layer curvature
  - {math}`\lambda_{\max}(H_{\ell\ell})` of each layer's diagonal block
* - Gauss–Newton fidelity
  - {math}`\lVert H-2J^\top J\rVert_2/\lVert H\rVert_2` with {math}`J=\partial\rho/\partial\theta`.
    Since {math}`H=2J^\top J+2\sum_\ell\rho_\ell\nabla^2\rho_\ell`, a small value means the
    curvature is the Gram matrix of the residual Jacobian. That Jacobian is where the input
    representation enters.
```

For **P ≤ `full_hessian_max`** (default 3000), everything is exact: the dense Hessian, its
spectrum, the Jacobi scaling and {math}`2J^\top J`. Above that limit only Hessian-vector
products are used, never a P×P matrix:

- Lanczos with full reorthogonalisation gives the top eigenvalues and spectral norms.
- Stochastic Lanczos quadrature gives {math}`n_\epsilon`, and its spread over probes is
  reported as `n_eff_std`.

The functions are also available individually: `sharpness`, `top_eigenvalues`,
`spectrum_slq`, `effective_dimension`, `full_hessian`, `jacobi_scaled_spectrum`,
`layer_sharpness` and `gauss_newton_fidelity`.

### Residual neural tangent kernel

With {math}`J_{ij}=\partial r_\theta(x_i)/\partial\theta_j`, the residual NTK is
{math}`K=JJ^\top/N`. Under gradient flow on the residual loss,
{math}`\dot r=-(2/N)JJ^\top r`. The component of the residual along eigenvector *i* therefore
decays at a rate proportional to {math}`\lambda_i(K)`.

A fast-decaying spectrum means most residual patterns are learned very slowly. To measure
drift, pass `ntk0=res0["ntk"]["K"]` from the step-0 call (computed on the same points). The
drift {math}`\lVert K_k-K_0\rVert_F/\lVert K_0\rVert_F` says whether the lazy (kernel) picture
describes training.

`K` is formed in float64 on the host. A float32 product on a GPU may run in TF32, which would
corrupt exactly the small eigenvalues the spectrum is about. The kernel uses at most `n_ntk`
points, because J is stored as rows × P floats.

### Loss landscape

`slice_1d` evaluates {math}`f(\alpha)=\mathcal L(\theta^*+\alpha d)` along random
**filter-normalised** directions (Li et al., 2018). Each neuron's incoming weights in *d*, a
column of a flax kernel, are rescaled to the norm of the trained neuron's weights, and each
bias is rescaled as one block. The result is that slices of different networks are
comparable. The defaults are seven directions of 141 points.

`slice_2d` adds a second direction, orthogonalised against the first and rescaled to its
norm. It returns `F[i, j]` = loss at {math}`(\alpha_j,\beta_i)`, ready for
`contourf(grid, grid, F)`.

`count_extrema` counts sign changes of consecutive differences. Every slice through the trained
point has at least one extremum; **more than one proves the slice is non-convex**. The count is
a lower bound, because a slice with a single minimum can still be non-convex. Differences below
`rtol·|f|` are ignored, so rounding noise in a flat basin is not counted.

### Representation

`feature_gram(phi)` computes the correlation matrix {math}`G_\Phi=\tilde\Phi^\top\tilde\Phi/N`
of standardised features, its eigenvalues and {math}`\kappa(G_\Phi)`. For the quick-start
spring on 256 points of {math}`[0,3]`, with {math}`\tau=2t/T-1`, the two input
representations give very different values:

- polynomial features {math}`(\tau,\tau^2,\dots,\tau^5)` give {math}`\kappa=567.7`;
- frequency-matched features {math}`(\tau,\sin\omega_d t,\cos\omega_d t,\sin\Omega t,\cos\Omega t)`
  give {math}`\kappa=2.7`, where {math}`\omega_d=\sqrt{k/m-(c/2m)^2}` is the damped
  natural frequency.

The test suite checks both values.

```{admonition} The operator-mapped Gram matrix
:class: tip
The physics loss sees {math}`\mathcal D u_\theta`, so the relevant conditioning is that of
the *operator-mapped* features {math}`G_{\mathcal D\Phi}=(\mathcal D\tilde\Phi)^\top(\mathcal D\tilde\Phi)/N`.
Apply the differential operator to the standardised features, including its action on the
constant shift, and call `feature_gram(D_phi, standardise=False)`. Re-standardising would
remove exactly the operator's scaling of each feature. For the spring this gives 990.0
(polynomial) vs 33.0 (frequency-matched); re-standardised, it would give 990 vs 2.7.
```

On hidden activations:

- **`erank`**: {math}`\exp(-\sum_i p_i\log p_i)` with {math}`p_i=\sigma_i/\sum_j\sigma_j`
  over the centred activations. A value far below the width means many units are redundant,
  so width is not the bottleneck. Compare only layers of the same width.
- **`saturation`**: the fraction of pre-activations with {math}`|z|>2`, where
  {math}`\tanh'<0.071`. In a PINN, saturated units also flatten {math}`\dot u` and
  {math}`\ddot u`.
- **`linear_probe(h, y)`**: {math}`1-R^2` of a target (for example {math}`u^*`,
  {math}`\dot u^*`) regressed on the top-k principal directions of a layer. A probe on the
  *input* features is a cheap way to screen candidate representations. Set
  `Config.probe_target` to probe every hidden layer in `run_checkpoint`.
- **`linear_cka(X, Y)`**: {math}`\lVert Y^\top X\rVert_F^2/(\lVert X^\top X\rVert_F\lVert Y^\top Y\rVert_F)`
  on centred activations. High CKA across seeds means ensembling adds little; low CKA between
  two networks means the representation, not chance, decides the internal solution.
- **`ablation_importance`**: loss after replacing one unit by its mean activation, divided
  by the trained loss. It ranks units for plotting. To size a layer, use refit-pruning
  instead: near a minimum, removing any unit raises the loss by orders of magnitude.

### Network design

**Output-layer refit** (`output_layer_refit`, `Config.refit_layer`). With the hidden layers
fixed, {math}`u=Wh+b` is linear in the output layer. For a linear PDE with linear boundary
terms, {math}`\min\lVert\rho\rVert^2` over that layer is therefore an exact linear
least-squares problem. The solve is an SVD least-squares solve in float64 on J itself, not
on the normal equations, which would square its condition number. A large `ratio`
(loss before / after) means Adam had not converged the last layer.

For a nonlinear residual, such as Euler's positivity map and fluxes, the solve is one
Gauss–Newton step followed by a backtracking line search. The loss the linear model predicted
is reported next to the loss actually reached.

**Refit-pruning** (`refit_pruning`, `Config.prune=True`). Units of the last hidden layer are
removed one at a time. Each step removes the unit whose removal raises the loss least after an
exact least-squares refit of the output layer. The result lists the loss at every width from
the full layer down to one unit.

A network that keeps its loss with far fewer units is wider than its representation needs.
`loss_true` re-evaluates the actual loss at each width; it equals the least-squares value
whenever the residual is linear in the output layer.

**Off-grid residual** (`offgrid_residual_ratio`, `Config.x_dense`) is the residual RMS on a
dense independent grid divided by its RMS on the collocation points. A ratio well above 1
means the network fits the training points at the expense of the gaps.

### Spectral analysis of error and residual

On a uniform 1-D grid:

```python
# Hann window; omega is the angular frequency
omega, amp = dg.amplitude_spectrum(e, dt)
# share of the residual's energy below 2 omega_0
frac = dg.energy_fraction_below(r, dt, 2 * omega_0)
```

Applied to the error {math}`e=u_\theta-u^*` at each checkpoint, the spectrum shows spectral
bias: which frequencies of the error are removed first and which linger. If the error lingers
at frequencies you can predict from the physics, put those frequencies into the
representation.

The residual's spectrum matters as much as its size. For the damped oscillator, the error is
the residual filtered through the operator's response {math}`\hat G(\omega)`. Residual near
resonance is amplified, while high-frequency residual is suppressed. This is why two
networks with similar losses can have very different errors.

Set `Config.reference_fn` (the exact solution) together with `x_dense` to get the relative
L² error in `run_checkpoint`. On a uniform 1-D grid the report also includes the error
spectrum.

### Modal analysis (Dirichlet problems)

On {math}`[x_0,x_0+L]` with zero boundary values, the error splits into sine modes with
coefficients {math}`c_n=\tfrac2L\int u\sin\tfrac{n\pi(x-x_0)}{L}\,dx`:

```python
# delta a_n: which modes carry the error
da = dg.modal_error(u_theta, u_star, x, n_modes=80)
# mu_n = ||grad_theta c_n||^2: how fast the network can move mode n
mu = dg.modal_kernel(problem.apply_fn, params, x, n_modes=80)
# w_n = (n pi / L)^4 / S^2: how much the loss of -u'' = f weighs mode n
w = dg.laplacian_mode_weights(80, scale=S)
```

Under gradient flow with a diagonal kernel,
{math}`\delta a_n(t)\approx\delta a_n(0)\,e^{-\mu_n w_n t}`. The per-mode rate factorises into
two parts:

- {math}`\mu_n` is set by the **representation**: how easily the network can move mode *n*.
- {math}`w_n` is set by the **operator and loss**: how much the loss cares about mode *n*.

In the strong form of {math}`-u''=f`, {math}`w_n\propto(n\pi)^4`. Take
{math}`u^*=\sin 2\pi x+0.1\sin 50\pi x` on {math}`[0,1]`: the slow mode (*n* = 2) counts
{math}`(2/50)^4\approx2.6\times10^{-6}` times as much in the loss as the fast one (*n* = 50).
No input representation can fix that. A small loss does not mean a small error on
multiscale problems, so check the error against a reference.

Adam and feature learning break the diagonal-kernel assumption. Use {math}`\mu_n w_n` to
order the modes, not to predict iteration counts.

### Loss balance, causality and localisation

These diagnostics answer the questions behind most PINN failures. All of them are
architecture-agnostic.

```{list-table}
:header-rows: 1
:widths: 24 36 40

* - Function
  - What it measures
  - What to look for
* - `gradient_conflict`
  - ‖∇Lᵢ‖ per term, pairwise cos(∇Lᵢ, ∇Lⱼ), each term's share of the update direction,
    learning-rate-annealing factors (Wang, Teng & Perdikaris 2021)
  - norm ratio ≫ 1: one term starves the others. Negative cosine: the terms fight; use
    re-weighting, gradient surgery or a hard constraint.
* - `term_ntk` *(needs blocks)*
  - NTK restricted to each block: mean diagonal = that term's convergence speed, the
    effective rate wᵢ·mᵢ, imbalance, NTK-balanced weights (Wang, Yu & Perdikaris 2022)
  - imbalance ≫ 1: the slowest term will be the last satisfied, if ever.
* - `residual_profile`
  - mean r² binned along a coordinate (time by default), late/early growth, causal weights
    {math}`w_k=\exp(-\varepsilon\sum_{j<k}L_j)` (Wang, Sankaran & Perdikaris 2022)
  - growth ≫ 1 or wₖ ≪ 1: late times are trained before early times are resolved.
* - `error_profile`
  - relative error per bin, `early` / `late` quarter values
  - error that accumulates while the residual *falls* is a trivial (collapsed) solution.
* - `residual_error_correlation`
  - Spearman rank correlation of |r| and |e| point-wise
  - high: residual-adaptive sampling targets the right places. Low or negative: it will not.
* - `residual_concentration`
  - share of Σr² in the worst 10 % of points; Gini of r²
  - concentrated residual (shocks, layers) is where adaptive sampling pays off.
* - `radial_spectrum`, `spectral_error`
  - radially averaged spectra on N-D grids; per-wavenumber error ratio |E(k)|/|U(k)| and the
    resolved bandwidth
  - a resolved bandwidth far below the solution's band means spectral bias limits accuracy.
```

`TrainingMonitor` also logs `rec["grad_cos_min"]`, the most negative pairwise cosine
between term gradients at each logged step, which flags term conflict during training.

### Automatic findings

`dg.interpret(res, lr=...)` reads any diagnostics dict and returns findings. Each finding
has a severity, the evidence, and an action; `dg.to_markdown(findings)` formats them. The
rules and thresholds are listed in `underPINN/diagnostics/advice.py`, and every finding
quotes the numbers it is based on. `run_checkpoint` runs the new diagnostics through these
`Config` fields and attaches `res["findings"]` and `res["findings_md"]` (also written to
`findings_<step>.md`):

```python
cfg = dg.Config(
    x_dense=xt_grid, reference_fn=u_exact,
    term_ntk=64,                # per-term NTK, points per block
    profile_axis=-1,            # profile residual / error along time
    causal_eps=1.0,             # add causal weights
    grid_shape=(nx, nt),        # x_dense is this C-ordered grid,
    grid_spacings=(dx, dt),     #   for the N-D spectral error
    lr=1e-3,                    # enables the stability rule
)
res = dg.run_checkpoint(problem, params, params0=params0, cfg=cfg,
                        out_dir="diag")
print(res["findings_md"])
```

Example output (abridged and wrapped) for an MLP that fails the convection benchmark
(β = 50, relative error 0.84):

```text
- [high] loss balance: 'bc' is 85% of the loss
- [high] loss balance: loss terms pull in opposing directions
  (min cosine -0.27)
- [high] causality: error accumulates along time (0.12 → 1) while the
  residual falls (rank corr -0.83)
  - action: Trivial / collapsed solution: the network satisfies the PDE
    where it has given up on the true solution ... Train causally (causal
    weights, time marching, curriculum on the horizon) and enforce IC/BC
    harder or exactly; more collocation points will not help.
- [medium] sampling: the residual does not indicate the error
  (Spearman -0.31)
```

On models that train well (the same problem solved by PINNsFormer, or Burgers at 1e-3
to 5e-5 error), the same rules return only `info` notes. Opposing term gradients and a
dominant PDE term are expected near convergence, so they are escalated only when the
error is large or an IC/BC term dominates. Pass the overall error as `res["rel_l2"]` (or
let `run_checkpoint` compute `res["error"]` from `reference_fn`) so the rules can tell
the two situations apart.

## Configuration, cost and caveats

### `Config` fields

```{list-table}
:header-rows: 1
:widths: 24 14 62

* - Field
  - Default
  - Meaning
* - `full_hessian_max`
  - `3000`
  - exact Hessian, Jacobi scaling and Gauss–Newton when P ≤ this; Lanczos/SLQ above it
* - `lanczos_k`, `slq_k`, `slq_probes`
  - `40`, `60`, `4`
  - Lanczos steps (top eigenvalues, norms); SLQ steps and probes (spectrum)
* - `eps`, `n_top`
  - `1e-3`, `10`
  - effective dimension threshold; κ index (κ₁₀)
* - `layer_k`
  - `25`
  - Lanczos steps per layer block
* - `n_ntk`
  - `192`
  - collocation points in the NTK
* - `n_dir`, `n_alpha`, `n2d`
  - `7`, `141`, `0`
  - 1-D slice directions and points; 2-D slice grid (0 = skip)
* - `hidden_points`, `probe_target`
  - `4096`, `None`
  - points for activation statistics; optional probe target `x -> y`
* - `refit_layer`, `prune`
  - `None`, `False`
  - output Dense layer to refit (e.g. `"Dense_2"`); also refit-prune its inputs
* - `x_dense`, `reference_fn`
  - `None`, `None`
  - dense points for the off-grid check, profiles and localisation; exact solution for the
    error, its spectrum and the error profile
* - `residual_weights`
  - `None`
  - per-equation weights in the NTK
* - `grad_conflict`
  - `True`
  - per-term gradient norms, cosines, step shares and LRA factors
* - `term_ntk`
  - `0`
  - points per block for the per-term NTK (needs `Problem.blocks`; 0 = skip)
* - `profile_axis`, `profile_x`, `profile_bins`
  - `None`, `None`, `20`
  - coordinate to profile along (e.g. `-1` = time; `None` = skip); points (default
    `x_dense`); number of bins
* - `causal_eps`
  - `None`
  - adds causal weights {math}`\exp(-\varepsilon\sum_{j<k}L_j)` to the residual profile
* - `grid_shape`, `grid_spacings`
  - `None`, `None`
  - `x_dense` is a C-ordered grid of this shape and spacing: enables the N-D spectral error
* - `lr`
  - `None`
  - learning rate, for the λ_max·lr stability rule in the findings
* - `interpret`
  - `True`
  - attach `res["findings"]` and `res["findings_md"]`
```

### Cost and precision

```{list-table}
:header-rows: 1
:widths: 30 70

* - Diagnostic
  - Cost (P parameters)
* - Lanczos / SLQ
  - k Hessian-vector products each, roughly 3× a loss gradient through the PDE
* - Exact Hessian
  - P Hessian-vector products and P² floats. For P in the low thousands this takes minutes
    on a CPU; lower `full_hessian_max` to use Lanczos instead.
* - NTK
  - `n_ntk` residual Jacobians, stored as `n_ntk·n_eq × P` floats
* - Slices
  - `n_dir·n_alpha` (+ `n2d²`) loss evaluations
* - Refit / pruning
  - one forward-mode Jacobian of the output layer; then small dense solves whose cost does
    not depend on the number of collocation points
* - Gradient conflict
  - one reverse pass per loss term
* - Per-term NTK
  - `term_ntk` residual Jacobians per block
* - Profiles, localisation, spectral error
  - one residual (and output) evaluation on the profile points
```

```{admonition} Run post-processing in float64 when you need small eigenvalues
:class: note
Spectra that span many decades need double precision. In float32, Hessian or NTK
eigenvalues below about {math}`10^{-6}\lambda_{\max}`, and loss differences near a converged
minimum, are rounding noise. For spectra, small `eps` or slice counts, enable
`jax_enable_x64` for the **post-processing run only** and cast the trained parameters to
float64. Keep training itself in float32 (see {doc}`performance`).
```

### Caveats

- **Local view.** Random 1-D and 2-D slices are low-dimensional views of a high-dimensional
  surface; the Hessian spectrum is the full local picture.
- **Gradient-flow models.** The NTK flow equation and the modal rate assume gradient flow
  and a fixed kernel. Adam and feature learning modify both, so use them to order events,
  not to predict iteration counts.
- **Fixed diagnostic batch.** Every number refers to the fixed `Problem.batch`, which may
  differ from the stochastic mini-batches used in training.
- **Seed-to-seed spread.** Single-seed checkpoint diagnostics can be unrepresentative. For
  claims, report medians over seeds.

```{seealso}
{doc}`training` for the solvers and `TrainingConfig` these diagnostics attach to, and
{doc}`api_reference` for the package layout.
```
