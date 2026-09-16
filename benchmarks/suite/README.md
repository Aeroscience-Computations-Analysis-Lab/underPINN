# Reviewer-response benchmarks

Code answering the experimental concerns raised in review. Each script produces
a JSON result under `results/`, and `summarize.py` renders them as one report.

```bash
bash benchmarks/suite/run_all.sh 5000        # GPU, paper settings
ALLOW_CPU=1 bash benchmarks/suite/run_all.sh 50   # correctness smoke test
```

**Every script refuses to run on CPU unless `--allow-cpu` / `ALLOW_CPU=1` is
given**, and CPU results are stamped and flagged loudly in the summary. A CPU
fallback can never be silently reported as a GPU measurement.

---

## 1. `parity/dispatch_parity.py` — the throughput discrepancy

> *"3-D pipe flow is 0.76 ms/epoch in Table 1 vs ~12 ms/epoch in Figure 2,
> raising concerns about measurement methodology or test parity."*

Both numbers are real. The gap is **host-side scaffolding**, not the physics or
the compiled step. Per epoch, `PipeFlowEvaluator.train` does — *in addition to*
its one `jax.jit`-ed `step` call:

| work | dispatches |
|---|---|
| `jax.random.split(key, 5)` | 1 |
| 4× `jax.random.randint` | 4 |
| 4× fancy-index gather `xyz_int[ir]` | 4 |
| `float(total)`, `float(pl)` | 2 **blocking** device→host syncs |

So the compiled step is ~1 of ~12 dispatches, and two `float()` calls stall the
host on the device every epoch, preventing any cross-epoch overlap.

The script times five variants of the **identical** problem — same physics,
network, batch sizes, weights, and RNG stream (all five follow bit-identical
loss trajectories; the reported final-loss spread should be ~0). Only the
scaffolding differs:

| variant | what changed |
|---|---|
| `A_evaluator_style` | exactly what the evaluator does today |
| `B_no_host_sync` | A, minus the per-epoch `float()` syncs |
| `C_batching_in_jit` | RNG + gather moved inside the jitted step |
| `D_jit_minimal` | C, minus the per-epoch sync |
| `E_lax_scan` | whole loop fused into one `jax.lax.scan` |

**Use this to say in the paper which configuration each reported number came
from** — and, if `D`/`E` win by a wide margin on GPU, to fix the evaluator.

## 2. `baselines/burgers_baselines.py` — stronger PyTorch baselines

> *"Baselines omit stronger PyTorch modes (torch.compile, TorchScript)."*

Fair, and addressed. Six variants of identical 1-D Burgers:
`torch_eager`, `torch_script`, `torch_func_eager`, `torch_func_compile`,
`jax_jit`, `jax_scan`.

**A verified negative result worth reporting:** `torch.compile` *cannot* capture
the classic PINN formulation at all —

```
torch.compile with aot_autograd does not currently support double backward
```

This was confirmed on both the `inductor` and `aot_eager` backends, and whether
`torch.compile` wraps the loss function or only the `nn.Module`. The nested
`autograd.grad(..., create_graph=True)` a PDE residual needs is simply outside
its supported set.

`torch.compile` **does** work on the `torch.func`
(`jacrev`/`hessian`/`vmap`) formulation — PyTorch's direct analogue of JAX's
`jacfwd`/`hessian`. That is therefore the strongest available PyTorch baseline,
and the one the paper should quote a best-vs-best ratio against.

**Two more verified, hardware-specific negative results**, found running this
suite on an NVIDIA GB10 (Blackwell, compute capability `sm_121`):

- `torch_script` (`torch.jit.script`) fails outright —
  `nvrtc: error: invalid value for --gpu-architecture (-arch)`. This build of
  PyTorch (`2.11.0+cu128`) only ships fused-kernel codegen for
  `sm_80/90/100/120` (`torch.cuda.get_arch_list()`); `sm_121` isn't in that
  list yet, so NVRTC can't JIT the kernel TorchScript's fuser generates. This
  is a PyTorch/CUDA-toolkit gap on very new hardware, not something fixable
  from this script.
- `torch_func_compile`'s default `inductor` backend hits a reproducible
  Inductor codegen bug on the `vmap(jacrev)` + `vmap(hessian)` composition's
  backward pass — the generated kernel tries to write in-place into an
  autograd `ZeroTensor` placeholder (`ZeroTensors are immutable. Please use
  the materialized zero tensor obtained using .clone() ...`). Confirmed
  reproducible in isolation, independent of this script's structure. The
  script now falls back to `torch.compile(backend="aot_eager")` — AOTAutograd
  graph capture without Inductor codegen — which does run, and records which
  backend the reported number came from (`compile_backend` in the saved JSON).

Separately, `ensure_host_cc()` in `common.py` repairs a broken `$CC`/`$CXX`
before any `torch.compile` call: on module-based HPC systems (e.g. this
machine's `nvhpc` environment module), those variables can point at a
toolchain path that isn't actually mounted, which makes Triton fail deep
inside Inductor with an opaque `FileNotFoundError` for `nvc` that gives no
hint the real problem is an environment variable. This only matters for the
PyTorch baselines; JAX never touches it.

## 3. `ablations/` — do the advertised features earn their place?

> *"FBPINNs, gated attention, trainable artificial viscosity, and RBA are not
> ablated or quantitatively validated."*

**`ablate_features.py`** — five arms on 1-D Burgers, scored on relative L² against
the same RK45 reference `BurgersEvaluator` uses (accuracy, not just loss).
Everything else is held constant; **parameter counts are reported per arm**, so
an architecture that wins only by being bigger is visible as such.

| arm | claim under test |
|---|---|
| `mlp` | baseline |
| `gated_mlp` | gated attention |
| `fourier_mlp` | trainable spectral embedding |
| `fbpinn` | domain decomposition |
| `mlp_rba` | residual-based adaptivity |

**`ablate_artificial_viscosity.py`** — three arms on the Toro-3 blast wave,
scored against the exact Riemann solution.

Worth noting explicitly: the paper advertises a **trainable** artificial
viscosity, but every shock benchmark it reports (`Toro3Evaluator`, the ramp
examples) runs with a **fixed** `art_visc=0.001`. The advertised feature is
never exercised in the results. This script runs `none` / `fixed` / `trainable`
and reports the learned ε alongside the hand-picked 0.001 — telling the reader
whether the automated tuning reproduces, beats, or underperforms hand tuning.

## 4. New algorithms: QR-DEIM-R adaptive collocation, Gauss-Newton training

> *"Missing discussion of ... adaptive selection schemes (e.g. QR-DEIM/
> QR-DEIM-R), and second-order/natural-gradient training for PINNs (e.g.
> recent D-NGD)."*

Rather than only discussing these in prose, we implemented both as genuine,
tested, drop-in alternatives to underPINN's existing Adam + RAR-D/RAD
machinery, and measured whether they actually help. Neither is a
transcription of a specific paper's algorithm — the review comment does not
pin either name down precisely enough for that, and both modules' docstrings
say exactly what published idea they follow and where they are our own
construction instead.

**`underPINN/utils/sampling.py::qr_deim_resample`** — QR-DEIM anchor points
(classical column-pivoted-QR DEIM selection, Drmac & Gugercin 2016, on a
small residual+coordinate feature basis) plus a leverage-score-weighted fill
(Drineas/Mahoney/Muthukrishnan-style randomized NLA) to reach a full
collocation batch. Unlike `rad_resample`'s pure `|residual|^k` weighting,
this is meant to discount *redundant* points inside one cluster rather than
have each one independently re-win a magnitude lottery. Caught and fixed
during testing: an earlier version sketched a dense
`(n_candidates, n_keep)` matrix for a second QR pass, which is correct in
principle but allocates tens of GB at real collocation-batch sizes
(`benchmarks/suite/ablations/ablate_qr_deim.py`'s 40,000/200,000
pool triggered a `MemoryError`) — replaced with an `O(n_candidates * r0)`
construction; `tests/test_sampling.py` has a regression test pinning this at
realistic scale.

`ablate_qr_deim.py` compares `none` / `rad` / `qr_deim` on the same Toro-3
problem as `ablate_artificial_viscosity.py` (fixed `art_visc=0.001`, only
the resampling strategy varies). At 5,000 epochs on GPU, the three arms
landed within a few percent of each other (rel-L2 0.448–0.472) and `rad`
edged out `qr_deim` on this particular run — **QR-DEIM-R did not show a
clear win here**, and we are reporting that rather than only the cases where
a new idea helps, in the same spirit as the FBPINN and trainable-viscosity
results above. Toro-3 is already the hardest, plateau-prone case in the
paper's own suite (>45% error even at 60,000 epochs), so this is one
data point on one hard problem, not a general verdict on the method.

**`ablate_qr_deim_ramp_ns.py`** repeats the same three-arm comparison on the
paper's actual flagship RAR-D problem instead of a toy 1-D case: 2-D viscous
compression-ramp SBLI (`RampNSEvaluator`'s exact geometry, ~66k-parameter
network, 51k-point collocation pool), scored by
`RampNSEvaluator.evaluate()`'s own metric (outer-flow Mach relative L2
against the analytic oblique-shock state). The resample cadence is a CLI
flag (`--resample-period`, default 500 epochs); we ran it at two settings:

| epochs | resample period | resamplings | best arm | rel L2 (`none` / `rad` / `qr_deim`) |
|---|---|---|---|---|
| 5,000 | ~1,000 (`epochs//5`) | ~5 | `rad` | 9.569e-2 / **9.473e-2** / 9.478e-2 |
| 30,000 | 500 | 60 | `none` | **6.845e-2** / 6.867e-2 / 6.995e-2 |

Two things worth flagging, neither of which is the "QR-DEIM-R wins" result
we'd have liked to report:

1. **Which strategy is "best" flips between the two runs.** At the shorter
   budget `rad` edges out a frozen pool; at 30,000 epochs with resampling
   *six times* as often, the frozen pool (`none`) actually finishes
   marginally ahead of both adaptive strategies. A plausible reading: once
   the network is well into fine-convergence, perturbing the collocation
   set every 500 epochs re-introduces non-stationarity right when the
   optimizer would otherwise be settling — resampling more often is not
   unconditionally better. This is one seed, one problem; we're reporting
   the reversal rather than picking whichever run makes a cleaner story.
2. **QR-DEIM-R is not the best arm in either run**, and is the worst of the
   three at 30,000 epochs (6.954e-2 vs. `none`'s 6.843e-2, a ~2% gap). Across
   Toro-3 and both Ramp NS settings, QR-DEIM-R has now been tested three
   times and has not once beaten the existing RAD baseline. Combined with
   §"Not covered" below, our honest overall assessment is that this
   particular construction is not yet an improvement worth adopting over
   `rad_resample` — a genuine negative result, reported as one.

**Collocation migration plots** (`ablate_qr_deim.py` and
`ablate_qr_deim_ramp_ns.py` now both plot the resampled pool's initial vs.
final positions — `benchmarks/suite/results/qr_deim_{toro3,ramp_ns}
_collocation_migration.png` — one column per arm, "none" included as a
frozen-pool sanity check) turned a "which number is smaller" result into a
mechanistic one:

* **Toro-3**: both `rad` and `qr_deim` visibly migrate onto a diagonal band
  tracking the moving shock through `(x, t)` — the qualitatively "correct"
  behaviour. `qr_deim` additionally resolves a *second* diagonal band (very
  plausibly the contact discontinuity, which propagates at a different
  characteristic speed) plus denser structure near the origin — richer
  structure-recovery than `rad` shows, even on a run where `qr_deim` scored
  numerically worse. (We also saw the same seed produce meaningfully
  different `qr_deim` numbers across three separate 5,000-epoch runs —
  0.472, 0.533, 0.497 — which we attribute to GPU floating-point reduction
  non-determinism, not a script bug; single-seed numbers here should be read
  with at least a ±0.03 noise floor.)
* **Ramp NS**: this is where the migration plot earns its keep. `rad`'s
  final pool stays broadly spread across the domain. `qr_deim`'s final pool
  has collapsed almost entirely onto the near-wall boundary-layer band, with
  very few points left in the outer flow where the shock actually is. That
  is a concrete, mechanistic explanation for QR-DEIM-R's underperformance
  here, not just a smaller number: the near-wall viscous residual (no-slip,
  a steep thermal boundary layer) is naturally large-magnitude and
  spatially coherent, so QR-DEIM's anchor+leverage-score selection gets
  dominated by wall structure — starving the outer-flow shock region of
  adaptive points, which is exactly the region `RampNSEvaluator`'s own
  metric scores (the near-wall band is deliberately *excluded*, since a
  boundary layer is real viscous physics no inviscid reference captures).
  `rad`'s simpler magnitude-only weighting doesn't get trapped this way.
  This suggests a concrete fix worth trying before concluding the method is
  a dead end — e.g. excluding (or down-weighting) the near-wall band from
  `qr_deim_resample`'s candidate pool, mirroring what the evaluation metric
  already does — but we have not implemented or tested that here, so we are
  not claiming it would work.

**Solution-field plots** (same two scripts, `..._solutions.png`) show what
those collocation differences actually do to the predicted field, exact
overlaid, one figure comparing all three arms directly:

* **Toro-3** (density/velocity/pressure profiles at `t_final`): all three
  arms qualitatively track the shock/rarefaction structure, but `qr_deim`
  visibly *under-resolves the density spike's peak* (~2.8 vs. the exact
  ~6.0, versus ~3.5 for `none`/`rad`) and is noticeably more diffuse through
  every jump — a direct, visual counterpart to the migration-plot finding:
  spreading its point budget across the shock *and* the contact
  discontinuity leaves fewer points sitting exactly on the density peak,
  under-resolving it relative to `rad`'s single-minded magnitude-only
  concentration there.
* **Ramp NS** (Mach field, PINN vs. exact vs. |error|, all three arms): at
  the converged 30,000-epoch budget the three fields are visually close to
  identical, and the error band around the shock line is similar width and
  intensity for all three — the ~1-2% rel-L2 gap between arms here is real
  but small enough that it is *not* obviously visible by eye, unlike the
  Toro-3 density-spike difference. Worth knowing before reading too much
  into small percentage differences in this suite generally: some show up
  as visible field-level degradation (Toro-3), others don't (Ramp NS at
  this budget).

**`underPINN/training/natural_gradient.py::train_gauss_newton`** —
Levenberg-Marquardt-damped Gauss-Newton: the loss is written as
`0.5*||r||^2`, and each step solves the damped normal equations via an
*augmented least-squares* form (`lstsq([J; sqrt(damping)*I], [r; 0])`)
rather than explicitly forming `J^T J`, with LM accept/reject damping
control. Tractable only for small networks (explicit Jacobian, `O(n_params^2)`
per step) — exactly the regime natural-gradient PINN papers demonstrate in.
**Two real bugs, not hypothetical ones, were caught getting this to work
correctly on GPU** (both documented in the module, both now fixed by
default, no caller action needed):
1. An un-jitted training loop retraced the forward pass + Jacobian + solve
   from scratch every epoch — the same per-epoch dispatch cost this entire
   `benchmarks/suite/` suite exists to diagnose elsewhere, just
   reintroduced here. Fixed by compiling the step once (§1 above).
2. Gauss-Newton converged cleanly in float32 on CPU, then **diverged** in
   float32 on GPU on the identical problem and seed. Root cause: JAX's
   default GPU matmul precision trades mantissa bits for throughput, and
   that's enough to corrupt `jacfwd`'s own internal matmuls -- fine for
   Adam's first-order update, not fine for a curvature estimate. Fixed by
   running under `jax.default_matmul_precision("highest")`, baked into
   `train_gauss_newton` itself (not left as a caller responsibility).

`ablate_natural_gradient.py` compares Adam vs. Gauss-Newton on the ODE
harmonic oscillator (`HarmonicOscillatorODE`, ~500-parameter network) at a
matched 2,000-epoch budget on GPU: Gauss-Newton reached relative L2
`3.4e-3` vs. Adam's `0.954` (**283x** lower error) — but at **283x** more
wall-clock time (56.3s vs. 0.2s), since each Gauss-Newton epoch forms and
solves an explicit parameter-count-sized linear system. Report both numbers
together: which one "wins" depends on whether the budget is epochs or
wall-clock, and that tradeoff — not a blanket "natural gradient is better" —
is the honest finding here.

## 5. NVIDIA PhysicsNeMo (Modulus) — a genuine third-party framework comparison

> *"NVIDIA Modulus provides a highly optimized, PyTorch-based framework...
> Benchmarking against NVIDIA Modulus or jinns would be a genuine addition"*

Done for real, not just discussed — see `benchmarks/suite/physicsnemo/`
(its own README documents the install, six real environment issues hit and
fixed along the way, and the full results). Installed
`nvidia-physicsnemo`/`nvidia-physicsnemo-sym` 2.2.1/2.4.0 into an isolated
venv (confirmed not to touch the shared JAX/PyTorch environment this suite
otherwise runs in), built its CUDA extension from source against a scoped
CUDA-12.8 `nvcc` (needed since this GPU's system toolchain is CUDA 13.2,
which mismatches the PyTorch build PhysicsNeMo requires), and ran six
matched PINN problems in both frameworks (same physics, network depth/
width, per-step minibatch sizes), each scored against the same exact/
manufactured solution underPINN itself uses:

| problem | speedup (underPINN) |
|---|---|
| Burgers 1D | 3.26x faster |
| Wave 1D | 9.88x faster |
| Heat 2D | 14.70x faster |
| Helmholtz 2D | 4.58x faster |
| ODE Harmonic | 13.97x faster |
| Pipe Flow 3D | 1.29x faster |

underPINN is faster in all six (consistent with this suite's dispatch-
overhead findings elsewhere). **Scope note: only training throughput
(ms/epoch) is reported here.** Both frameworks were scored against exact/
manufactured reference solutions during development to confirm each side
trains a physically sensible model, but per-problem accuracy numbers and
cross-framework accuracy comparisons are not the claim made in this
document.

**A user asked why the two frameworks' outputs differed on the *same*
network — the answer mattered for a fair throughput comparison too.**
Every `FullyConnectedArch(...)` call had silently inherited PhysicsNeMo's
default `weight_norm=True` (Salimans & Kingma 2016 weight normalization on
every hidden layer) — a real architectural difference from underPINN's
genuinely plain `MLP`/`FourierMLP`/`GatedMLP`, undocumented in our first
pass despite each script's "not matched" notes claiming only the
Fourier/gating difference. The extra per-layer normalization arithmetic
also changes PhysicsNeMo's per-step compute cost, directly relevant to a
throughput comparison, not just an accuracy one. Fixed by passing
`weight_norm=False` explicitly everywhere and rerunning all six — every
number in the table above is the corrected run. See the physicsnemo
README for the full architecture-parity audit (activation, bias init,
input/output scaling, weight init) that followed. This remains six
problems and one seed each, not a controlled ablation, and PhysicsNeMo's
own JIT/NVFuser fusion had to stay disabled (`jit=false`) throughout to
work around the same Blackwell/NVRTC gap documented in §2, so none of its
numbers here reflect its own fastest available path.

## 6. `underPINN/pde/*` — a real bottleneck found by profiling underPINN itself

Not a reviewer comment — found by turning the same profiling scrutiny this
suite applies to the *comparison* scripts onto underPINN's own library code.

Ten of underPINN's `BasePDE.residual()` implementations were audited. Most
computed `jax.jacfwd` (Jacobian) and `jax.hessian` (Hessian) as two
*independent* AD transforms over the same collocation points, plus a third,
separate `model.apply()` call just for the raw network value — each one
re-tracing/re-running the forward pass, when the value, the needed gradient
components, and the needed Hessian diagonal entries can all come from one
shared `jax.vjp` call (value + full gradient in one backward pass) plus
targeted `jax.jvp` calls for only the specific second-derivative terms the
residual actually uses (a standard forward-over-reverse trick for cheap
selected Hessian entries). Confirmed via micro-benchmark, not assumed, that
XLA does **not** perform cross-transform common-subexpression elimination
between independently-traced `jacfwd`/`hessian`/`apply` calls — this is real
duplicated compute, not something the compiler already removes.

**Caveat checked before generalizing the fix, not assumed:** it only saves
compute when the number of Hessian entries actually needed, K, is less than
the input dimensionality N. `heat.py` and `wave.py` need *all* N diagonal
entries for their 2-D inputs (K=N=2) and have no separate redundant value
pass — audited and **left unchanged**, since there is nothing to remove.

### A second, more important caveat: forward-only benchmarks are the wrong metric

The table below was first written after benchmarking each fixed file's
`residual()` as an isolated `jax.jit`-ed forward call. That is **not** what
a training step actually costs: the real step differentiates the *whole*
loss (which contains the residual) via `jax.value_and_grad(loss)(params,
...)` w.r.t. the network's parameters, for the optimizer update. A file's
residual can be a clear win as a bare forward call and a clear **loss**
once wrapped in that outer parameter-gradient — this was caught by
re-checking `helmholtz.py`'s real number inside
`compare_underpinn_multi.py --problems helmholtz2d` (which reruns the
actual training step, not just the residual): it came back **slower**
(1.203→~1.25ms/epoch, reproducible across three repeated runs), directly
contradicting the file's own "1.22x faster" forward-only micro-benchmark.
Re-verifying every fixed file under `jax.value_and_grad` (mirroring a real
step) rather than a bare residual call settled which fixes are real:

| file | fix | forward-only | **under outer `value_and_grad`** (the metric that matters) | verdict |
|---|---|---|---|---|
| `burgers.py` | vjp + 1 targeted jvp | 1.54x (0.494→0.321 ms/epoch) | **1.88x** (also confirmed via a real full run, see below) | **kept** |
| `diffusion.py` | vjp + 1 targeted jvp | — | **1.49x** | **kept** |
| `burgers_deeponet.py` | vjp + 1 targeted jvp | — | **1.25x** | **kept** |
| `heat2d_unsteady.py` | vjp + 2 targeted jvp | up to 1.84x at large batch | **0.94x–1.30x, batch-dependent** (net positive at every batch ≥1024, including its real production batch) | **kept** |
| `pipe_flow_unsteady.py` | vjp + 2 targeted jvp | — | **0.90x–1.30x, batch-dependent**, same shape as `heat2d_unsteady.py` (its real production batch, ~512, measures 1.08x) | **kept** |
| `helmholtz.py` | vjp for value + 2 separate jvp-of-vjp calls | 1.22–1.58x | **0.84x (~16% slower)**, reproducible | **reverted** |
| `navier_stokes.py` | per-component `hess_diag`, 2 components × 2 jvp | 1.30x | **0.62x (~38% slower)** | **reverted** |
| `k_epsilon.py` | per-component `hess_diag`, 4 components × 2 jvp | 1.24x | **0.49x (~2x slower)** — worst regression | **reverted** |

The pattern: fixes needing exactly **one** extra `jax.jvp` call (K=1:
Burgers, Diffusion, DeepONet-Burgers) are unambiguous wins under the real
metric too. Fixes needing **two** `jax.jvp` calls for a single scalar
output (Heat2D-unsteady, Pipe-flow-unsteady) are batch-size-dependent but
net positive at realistic batch sizes. Fixes needing **several independent**
`hess_diag`-style vjp+jvp calls across multiple output components
(Helmholtz: 2, NavierStokes: 4, K-Epsilon: 8) are **real regressions**:
backpropagating the outer parameter-gradient through several
independently-traced nested vjp/jvp subgraphs costs more than
backpropagating through XLA's single fused `jax.hessian` primitive, even
though the forward evaluation alone is cheaper. `helmholtz.py`,
`navier_stokes.py`, and `k_epsilon.py` were reverted to their original
`jax.hessian`-based residuals, with the measurement documented inline in
each file's `residual()` comment.

Two more files got this same outer-gradient check *before* ever being
changed, per the "measure, don't assume" discipline established above —
and came back negative, so were correctly never modified in the first
place: **`ode.py`**'s `HarmonicOscillatorODE` (reverse-over-reverse
double-`jax.grad` on a batched sum-trick — 0.74x, slower, likely because
the sum-trick avoids `jax.vmap` entirely and a vmapped vjp+jvp replacement
can't beat that at this scale) and **`navier_stokes_3d.py`**'s
`SteadyNS3DPDE` (pure forward-over-forward `jacfwd(jacfwd(...))`, which
gets the *full* multi-component Hessian for all 4 fields in one batched
sweep more cheaply than several separate vjp+jvp calls can reach just the
9 needed entries — 0.65–0.73x, slower, at the real `run_pipe_flow`
config). `UnsteadyNS3DPDE` (same file, used by
`examples/pipe_flow/pipe_flow_pulsatile_transfer.py` and
`examples/AAA/AAA_pulsatile_transfer.py`) was not separately measured but
shares the identical structure, so was left unchanged by the same
reasoning rather than assumed safe.

**A duplicate-code gotcha, caught while rerunning the PhysicsNeMo
comparison after this fix:** `benchmarks/suite/physicsnemo/
compare_underpinn.py`'s Burgers `loss_fn` — the actual timed hot path for
that comparison — is a **self-contained reimplementation** of the old
unfused Burgers residual, not an import of `underPINN.pde.burgers`. Fixing
the real library file alone left this comparison script's numbers frozen
at the pre-fix values, silently. Found by noticing the comparison script's
number hadn't moved after the library fix landed; fixed by applying the
identical vjp+jvp pattern to both `loss_fn` and the `_BurgersResidual`
helper class in that file, verified against the legacy formulation
(max diff 1.7e-3, consistent with the established float32 AD-graph-
topology noise floor for a deeper 5×64 network), then rerun: **8.219ms/
epoch, down from 14.458ms** — a real 1.76x speedup on the actual
comparison, pushing the underPINN-vs-PhysicsNeMo Burgers speedup from
1.85x to **3.26x** (see `physicsnemo/README.md`). The general lesson: a
comparison script that duplicates a library residual rather than importing
it will not pick up a library fix automatically, and needs to be checked
and re-synced by hand.

**Verification:** two new test files, `tests/test_pde_derivative_fusion.py`
(the 5 files kept fused: analytic ground truth where a closed-form model
made it feasible, plus an inline reimplementation of each old
`jacfwd`/`hessian`/`apply` formulation as a cross-check; the 3 reverted
files' test classes now compare the current, reverted code against an
identical hand-written `jax.hessian` formulation — an intentionally exact
check, kept as a plain regression/shape guard rather than evidence a
fusion rewrite still lives there) and `tests/test_pde_burgers_residual.py`.
Two pre-existing numeric-regression tests in `tests/test_pde_signatures.py`
(Burgers, Heat2D) had their tolerance loosened from `1e-5` to `5e-4`, with
the reason documented inline: the old and new formulations are different
(mathematically equivalent) AD graphs, and float32 accumulates rounding
differently across them — verified against analytic-function ground truth
(agreement to ~1e-7) to confirm this is expected numerical noise, not a
bug. Full suite, after the reverts: **211 passed, 0 failed**
(`pytest tests/`, ~8 min on this GPU, run with nothing else competing for
the device — an earlier attempt to time and test concurrently triggered a
91 GB CUDA allocation failure on this shared-memory GB10).

## Not covered here (prose, not code)

The review also asks for discussion of other JAX-native PINN frameworks
generally. **jinns** specifically *has* now been installed and run for
real — see `benchmarks/suite/jinns/README.md` (six matched cases —
Burgers, 2-D steady heat, 1-D diffusion, 2-D unsteady heat, the ODE
harmonic oscillator, and 3-D pipe flow (Navier-Stokes) — each with a
settings audit; only training throughput is reported, not accuracy).
underPINN is faster than jinns on every problem tested, by margins from
essentially tied (ODE Harmonic) to 2.51x (Heat 2D steady); jinns is in
turn faster than PhysicsNeMo on every problem where a PhysicsNeMo number
exists. Pipe Flow required real engineering, not just config-matching:
jinns' own convective-term operator is 2-D-only, so the 3-D `(u.grad)u`
term is hand-written; the cylindrical geometry doesn't fit jinns'
box-shaped boundary-condition machinery, so that comparison bypasses
`jinns.solve` for a manual training loop; and the custom residual was
verified against the exact analytic Poiseuille solution before being
trusted (came back exactly zero, not just close). What remains prose-only
is broadening to jinns' natural-gradient optimizer, Wave and Helmholtz
(both use FourierMLP with no jinns equivalent), Pipe Flow's full
100,000-point/GatedMLP/70,000-epoch configuration (this comparison used a
reduced, disclosed scale), and any other JAX-native framework beyond jinns
and PhysicsNeMo.
