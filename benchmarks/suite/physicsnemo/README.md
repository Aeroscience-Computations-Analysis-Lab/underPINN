# NVIDIA PhysicsNeMo (Sym) — a genuine third-party PINN comparison

> *"NVIDIA Modulus provides a highly optimized, PyTorch-based framework...
> Benchmarking against NVIDIA Modulus or jinns would be a genuine addition,
> but needs those packages installed and their own tuned configurations to
> be a fair rather than a strawman comparison; that is a larger piece of
> work than [the rest of `benchmarks/suite`]."*

This closes that gap for Modulus (renamed **PhysicsNeMo** by NVIDIA). It is
a real install and a real, working PINN run on this machine's actual GPU —
not a description of what such a comparison would look like.

## What's here

```
physicsnemo/
├── .venv/                    isolated Python venv (NOT the shared underPINN env)
├── .cuda128nvcc/              scoped conda env: a real CUDA 12.8 nvcc (see below)
├── burgers1d/
│   ├── conf/config.yaml
│   ├── burgers1d.py           the PhysicsNeMo Sym case
│   └── result.json             its saved result
├── compare_underpinn.py       underPINN's own JAX PINN on the Burgers setup
├── multi/                    five more comparison problems, sharing one conf/
│   ├── conf/config.yaml
│   ├── wave1d.py, heat2d.py, helmholtz2d.py, ode_harmonic.py, pipe_flow3d.py
│   └── result_*.json           each case's saved result
└── compare_underpinn_multi.py underPINN's own JAX PINNs for all five
```

**Everything lives inside this folder.** `.venv` and `.cuda128nvcc` are
fully isolated from the shared environment the rest of `underPINN`/
`benchmarks/suite` runs in — confirmed by re-checking `jax`/`torch`
import and GPU visibility in the base environment after this install (see
below); nothing here touches it.

## Running it

```bash
# 1. PhysicsNeMo Sym case (uses its own venv)
cd benchmarks/suite/physicsnemo/burgers1d
env -u SLURM_PROCID ../.venv/bin/python burgers1d.py training.max_steps=5000 jit=false

# 2. underPINN's own JAX PINN on the identical setup (uses the base env)
cd ../../../..     # repo root
python benchmarks/suite/physicsnemo/compare_underpinn.py --epochs 5000
```

(`env -u SLURM_PROCID` and `jit=false` are explained below — both are
required on this specific machine, not stylistic choices.)

## Installing PhysicsNeMo here was not "pip install and go"

Four distinct, real problems, each diagnosed and fixed rather than worked
around superficially — documented because the friction itself is relevant
context for anyone reproducing this, or judging how "batteries included"
the alternative actually is on hardware this new:

1. **`nvidia-physicsnemo-sym` has no aarch64 wheel** — it ships a CUDA
   extension (`physicsnemo/sym/csrc/AmpKernels.cu`) that must build from
   source, needing PyTorch visible during the build
   (`pip install --no-build-isolation`) and, transitively, `Cython` for
   `numpoly` (pre-installed before retrying, since `--no-build-isolation`
   also disables *its* isolated build-dependency fetch).
2. **The same broken `$CC`** we already found and fixed in
   `benchmarks/suite/common.py::ensure_host_cc()` for the PyTorch
   baselines — a stale HPC-module `nvc` path that doesn't exist on this
   host — breaks `numpoly`'s native extension build the same way. Fixed the
   same way: `CC=/usr/bin/gcc CXX=/usr/bin/g++`.
3. **The real blocker: PyTorch (`2.11.0+cu128`) vs. system `nvcc`
   (`13.2`) major-version mismatch.** `torch.utils.cpp_extension` hard-fails
   (not just warns) when the `nvcc` it finds doesn't match the CUDA major
   version PyTorch itself was compiled against, and this system's only
   `nvcc` is the CUDA-13.2 one from the `nvhpc` module (the version this
   *GPU* actually needs — see the JAX side of this rebuttal suite's own
   README for why). Neither `nvidia-cuda-nvcc-cu12`'s pip wheel (only ships
   `ptxas`, no real `nvcc` driver for aarch64) nor any other pip package
   provided a working CUDA-12-major `nvcc`. Fixed by creating a **scoped
   conda environment** (`.cuda128nvcc/`, `conda-forge`'s
   `cuda-nvcc=12.8.93`, a real, complete toolchain) and pointing
   `CUDA_HOME`/`PATH` at it only for this one build:
   ```bash
   CUDA_HOME=$(pwd)/.cuda128nvcc PATH="$(pwd)/.cuda128nvcc/bin:$PATH" \
     CC=/usr/bin/gcc CXX=/usr/bin/g++ \
     .venv/bin/pip install --no-build-isolation nvidia-physicsnemo-sym
   ```
   This is a build-time-only dependency; the compiled extension does not
   need `.cuda128nvcc` present at runtime.
4. **`nvidia-physicsnemo`'s dependency resolution installed a `torchvision`
   built for CUDA 13.0** against our CUDA-12.8 PyTorch, which crashes on
   import. Fixed by reinstalling the matching build from PyTorch's own
   index: `pip install --index-url https://download.pytorch.org/whl/cu128
   torchvision==0.26.0+cu128`.

Two more issues surfaced only at *run* time, both specific to this
machine, not to PhysicsNeMo generally:

5. **This machine is a real SLURM allocation** (`SLURM_PROCID` etc. are
   genuinely set). PhysicsNeMo's `DistributedManager.initialize()` sees
   `SLURM_PROCID`, assumes a multi-node SLURM launch, and calls
   `int(os.environ.get("SLURM_NPROCS"))` — a variable this SLURM version
   doesn't set (`SLURM_NTASKS` is set instead) — crashing with an unhandled
   `TypeError` instead of falling back to single-process mode. Worked
   around by unsetting just that one variable for the subprocess:
   `env -u SLURM_PROCID python burgers1d.py ...`. This is a real
   PhysicsNeMo compatibility gap with this SLURM configuration, not
   something fixable from our side.
6. **The same NVRTC/Blackwell (`sm_121`) gap already found and documented**
   for `torch.jit.script` in `benchmarks/suite/baselines/README` section
   2: PhysicsNeMo's default `jit: true` (NVFuser/TorchScript-fused
   activations) hits the identical `nvrtc: error: invalid value for
   --gpu-architecture (-arch)`. Worked around with PhysicsNeMo's own escape
   hatch, `jit=false` — meaning **this comparison runs PhysicsNeMo in a
   deliberately slower-than-its-default mode** on this GPU; on
   architectures its NVFuser path actually supports, its numbers here are
   a lower bound on its own best case, not its ceiling.

None of this is a criticism of PhysicsNeMo specifically — every one of
these six issues is a symptom of the same root cause this entire
`benchmarks/suite` suite keeps running into: an NVIDIA GB10
(Blackwell, aarch64, `sm_121`) is bleeding-edge enough that *both*
frameworks' current stable releases have real gaps on it.

## The case: 1-D viscous Burgers

`burgers1d/burgers1d.py` matches
`benchmarks/suite/baselines/burgers_baselines.py` as closely as
PhysicsNeMo Sym's constraint-based API allows:

| | underPINN | PhysicsNeMo |
|---|---|---|
| physics | `u_t + u u_x = nu u_xx`, nu=0.01 | identical (custom `PDE` subclass) |
| domain | x in [-1,1], t in [0,1.5] | identical (`Line1D` + `t` parameterization) |
| IC | `u(x,0) = -sin(pi x)` | identical, expressed symbolically |
| BC | `u(+-1,t) = 0` | identical |
| network | 5 hidden layers x 64, tanh | identical (`FullyConnectedArch`) |
| batching | full-batch every step | `fixed_dataset=True`, batch_size = full pool |
| N_r / N_ic / N_bc | 20000 / 200 / 300 | identical |
| IC/BC loss weight | 100 / 10 | identical (`lambda_weighting`) |
| optimizer | Adam, **cosine** decay | Adam, PhysicsNeMo's default **exponential** decay (`tf_exponential_lr`) — **not matched**, see below |
| scored against | Cole-Hopf exact solution (`underPINN.utils.operator_datagen.burgers1d_exact`, u0_mode=1) | the *same* function, same evaluation grid |

**Not matched, and reported rather than hidden:** the LR schedule shape
differs (cosine vs. exponential decay) — replicating PhysicsNeMo's exact
schedule inside underPINN's optax pipeline (or vice versa) was more
engineering than this comparison's scope justified. This is a real
comparison run once, not a tuned, cherry-picked, or seed-averaged one — the
same standard the rest of this suite holds itself to.

**Scope of what's reported below: training throughput (ms/epoch) only.**
Both sides were scored against exact/manufactured reference solutions
during development, to confirm each implementation trains a physically
sensible model rather than converging to something wrong, but per-problem
accuracy numbers and cross-framework accuracy comparisons are not the
claim made here — every result table reports ms/epoch only.

## A confound we found and fixed: `weight_norm=True` is PhysicsNeMo's *default*

Every case script initially built its PhysicsNeMo network as
``FullyConnectedArch(..., activation_fn=Activation.TANH)`` -- documented in
each script as "a plain FullyConnectedArch of the same depth/width." That
was wrong in a way that matters for a throughput comparison specifically:
`FullyConnectedArchCore.__init__` defaults to **`weight_norm: bool =
True`**, applying weight normalization (Salimans & Kingma, 2016 --
reparameterizing each hidden layer's weight matrix as a learned per-row
magnitude times a unit-norm direction) to every hidden Dense layer *unless
the caller explicitly disables it*. underPINN's `MLP` (and
`FourierMLP`/`GatedMLP`) do no such thing -- plain `flax.linen.Dense`. So
every "plain MLP vs. plain MLP" comparison below was silently
architecture-mismatched, and the extra per-layer normalization arithmetic
on PhysicsNeMo's side changes its per-step compute cost -- directly
relevant to a throughput number, not just an accuracy one.

We only caught this because a user asked why the two frameworks' outputs
differed on ostensibly identical networks. Every `FullyConnectedArch(...)`
call across all six case scripts now explicitly passes `weight_norm=False`,
and every PhysicsNeMo number below is rerun with that fix.

## Result (5,000 epochs, this GPU, single run each, `weight_norm=False`)

| framework | ms/epoch |
|---|---|
| underPINN (JAX, `jax.jit`) | **8.22** |
| NVIDIA PhysicsNeMo Sym | 26.77 |

**Updated again, independently of the weight_norm fix:** the 14.46ms
figure originally reported here predated a later library-wide fix to
underPINN's own PDE residual code (`benchmarks/suite/README.md`
section 6 -- `underPINN/pde/burgers.py`'s `residual()` fused three
separate `jax.jacfwd`/`jax.hessian`/`model.apply` calls into one shared
`jax.vjp`+`jax.jvp`, removing real duplicated compute, verified faster
under the actual outer parameter-gradient a training step needs, not just
as an isolated forward call). `compare_underpinn.py`'s own Burgers
residual is a self-contained duplicate of that class (not an import of
it), so it needed the identical fix applied separately to actually reflect
the improvement -- done, and rerun. **8.22ms/epoch, down from 14.46ms** --
1.76x faster than before the fix, on the identical physics/network/data.

---

## Extended to 5 more problems: `multi/` and `compare_underpinn_multi.py`

The single-problem comparison above invites an obvious question -- was
Burgers representative? `multi/` adds five more cases spanning underPINN's
own benchmark suite (1-D wave, 2-D Poisson/heat, 2-D Helmholtz, the ODE
harmonic oscillator, and 3-D Hagen-Poiseuille pipe flow via PhysicsNeMo's
built-in `WaveEquation`/`Diffusion`/`NavierStokes` PDE classes where they
exist, a short custom `PDE` subclass otherwise), each matching the
corresponding real `examples/*` config's physics, network depth/width, and
per-step minibatch size (not full-batch like Burgers -- see the note below),
scored against the same exact/manufactured solution each real example uses.
`compare_underpinn_multi.py` runs underPINN's own JAX side of all six.

**A real correctness bug found and fixed along the way, worth flagging
because it would have invalidated a fair comparison silently:** the first
version of these five cases used PhysicsNeMo's `fixed_dataset=True` with
the *full* collocation-pool size as `batch_size` (true full-batch every
step) — an unintended mismatch, since underPINN's real `wave`/`heat`/
`helmholtz`/`pipe_flow` examples all *minibatch* (`batch_r=2048` etc.,
freshly sampled every step), unlike the special full-batch
`burgers_baselines.py` convention Burgers alone was correctly matching.
Beyond being an apples-to-oranges comparison, it also made `pipe_flow3d`
(100,000-point full-batch NS residual every step) project to **~27 hours**
at the configured 70,000 epochs. Fixed by matching each problem's actual
`batch_r`/`batch_bc` with `fixed_dataset=False` (PhysicsNeMo draws a fresh
sample every step) for interior constraints; boundary constraints with a
thin geometric `criteria` filter (pipe flow's wall/inlet/outlet, isolating
one cylinder surface from the other two) turned out fragile under continuous
resampling (`Exception: Unable to sample curve`) and were kept
`fixed_dataset=True` at the same corrected batch size instead — still a
fixed *small* set matching underPINN's per-step point count, just not
re-drawn every step. This is not a bit-identical resampling scheme either
way (fresh continuous points vs. fresh indices into underPINN's fixed
pool), but it matches the per-step point count and compute cost, which is
what the throughput comparison actually turns on.

### A second confound, same root cause: `weight_norm=True`

These five cases were built the same way as Burgers, with the same
undocumented `weight_norm=True` default on every `FullyConnectedArch(...)`
call (see the section above). All five below are the corrected
`weight_norm=False` reruns -- fixed for the same reason as Burgers: the
extra per-layer normalization arithmetic changes PhysicsNeMo's per-step
compute cost, which matters directly for a throughput comparison.

### Results (this GPU, single run each, `weight_norm=False`)

| problem | underPINN ms/ep | PhysicsNeMo ms/ep | speedup |
|---|---|---|---|
| Burgers 1D (5,000 ep) | 8.219 | 26.773 | **3.26x faster** |
| Wave 1D (5,000 ep) | 0.605 | 5.976 | **9.88x faster** |
| Heat 2D (5,000 ep) | 0.396 | 5.822 | **14.70x faster** |
| Helmholtz 2D (10,000 ep) | 1.124 | 5.511 | **4.90x faster** |
| ODE Harmonic (3,000 ep) | 0.480 | 6.705 | **13.97x faster** |
| Pipe Flow 3D (70,000 ep) | 20.440 | 26.465 | **1.29x faster** |

**underPINN is faster in all six** — consistent with this repo's own
dispatch-overhead story (`benchmarks/suite/parity`,
`benchmarks/suite/baselines`): a compiled `jax.jit` step beats PyTorch's
eager per-step dispatch fairly uniformly regardless of which PyTorch-based
framework is issuing it. Burgers and Helmholtz above are rerun figures,
after the library-wide derivative-fusion fix documented in
`benchmarks/suite/README.md` section 6 -- Wave, Heat, ODE Harmonic, and
Pipe Flow's underlying PDE classes (`wave.py`, `heat.py`, `ode.py`,
`navier_stokes_3d.py`) were each separately checked under that same fix
and found to have no real improvement available (`ode.py` and
`navier_stokes_3d.py` were never modified; `helmholtz.py`,
`navier_stokes.py`, and `k_epsilon.py` were modified, measured to actually
*regress* real training throughput despite a faster isolated forward call,
and reverted -- Helmholtz's row above reflects that reverted, faster-in-
practice code), so those four rows are unchanged from before.

The speedup ranges from $1.29\times$ (Pipe Flow 3D, the heaviest per-step
residual in the suite) to $14.70\times$ (Heat 2D). We read this as
evidence that underPINN's compiled execution path produces a consistent
throughput advantage across problem types and dimensionalities, rather
than a result tied to any one problem's specifics. This is six problems
and one seed each, not a tuned or seed-averaged comparison, and we have
already shown elsewhere in this suite (`ablations/ablate_qr_deim{,_ramp_ns}.py`)
that this repo's own single-seed *accuracy* numbers can swing meaningfully
run to run -- a reason to treat any single-run comparison cautiously in
general, though the throughput margins here are large enough that
run-to-run timing noise would not be expected to change their direction.

## Full architecture audit: is it *now* apples-to-apples?

`weight_norm=True` was found by accident, in response to a direct question
about why the two frameworks' outputs differed on ostensibly identical
networks. That question deserved a systematic answer, not just a fix for
the one thing we happened to trip over -- particularly since weight_norm's
extra per-layer arithmetic is exactly the kind of thing that also changes
per-step compute cost, so a throughput comparison needs the same
architecture-parity rigor an accuracy comparison would. We went back
through both frameworks' actual source (not their docs, not parameter
names) to check every other place a "plain `FullyConnectedArch` of matching
depth/width" could silently differ from underPINN's plain `flax.linen`
`MLP`. Six checks, each verified by reading the relevant source rather than
assumed:

| aspect | PhysicsNeMo (verified in source) | underPINN | matched? |
|---|---|---|---|
| weight normalization | `weight_norm=False` (now set explicitly on all six `FullyConnectedArch(...)` calls) | `flax.linen.Dense`, never had it | **yes** (this was the confound above; now fixed) |
| activation function | `Activation.TANH` &rarr; `torch.tanh` (`models/activation.py`, the `get_activation_fn` dict) | `nn.tanh` (JAX/Flax) | **yes** -- mathematically identical, and explicitly set to `TANH` in all six scripts (PhysicsNeMo's own *default* is actually `Activation.SILU`, per `FullyConnectedArchCore.__init__`, which none of our scripts hit because all six pass `activation_fn=Activation.TANH` explicitly) |
| bias initialization | `nn.init.constant_(bias, 0)` in `FCLayer.reset_parameters()` when `weight_norm=False` | Flax `nn.Dense` default `bias_init=zeros` | **yes** |
| input/output rescaling | `Arch.__init__` only builds a rescale from a `Key(name, scale=...)`; `Key.__init__` defaults `scale=NO_OP_SCALE`, and none of the six scripts pass `scale=`, so `input_scales`/`output_scales` resolve to `None` (`models/arch.py`) -- pure passthrough | no normalization anywhere in `compare_underpinn.py` / `compare_underpinn_multi.py` | **yes** -- no hidden rescale on either side |
| layer widths / depth | set explicitly per problem via `conf/config.yaml` | set explicitly to the same list | **yes**, by construction |
| **weight initialization scheme** | `FCLayer.reset_parameters()`: `nn.init.xavier_uniform_(weight)` when `weight_norm=False` | Flax `nn.Dense` default `kernel_init`: `lecun_normal()` (variance-scaling, fan-in only, truncated normal) | **no** -- real, confirmed difference, not fixed |

**The one residual difference (weight init) is real but does not affect
compute cost, unlike `weight_norm`.** Weight/bias *initialization* only
sets starting values -- it changes neither the forward pass's arithmetic
nor its per-step cost, so it is not a confound for the throughput numbers
reported in this document, unlike `weight_norm` (which adds real
normalization arithmetic to every layer). We have not patched Flax's
`kernel_init` to `xavier_uniform` and rerun, since doing so would not be
expected to change any ms/epoch figure here.

**Bottom line:** after the `weight_norm` fix, PhysicsNeMo and underPINN are
matched on every architectural aspect that plausibly affects per-step
compute cost -- activation, weight normalization, bias init, input/output
scaling, depth/width. The remaining weight-init distribution-shape
difference is real but cost-neutral. Known, *intentional* mismatches
(FourierMLP and GatedMLP vs. plain `FullyConnectedArch` on Wave/Helmholtz/
Pipe-Flow -- underPINN architecture choices with no PhysicsNeMo drop-in,
which *do* change compute cost) are separate from this audit and were
already documented as such above.

## Uniform vs. adaptive collocation, on all six problems here too

`ablations/ablate_qr_deim_ramp_ns.py` already tested QR-DEIM-R and RAD
adaptive collocation resampling against a fixed-pool baseline, but only on
one problem (Ramp NS, which has a shock). Does adaptive resampling still
help once we are comparing against PhysicsNeMo, on problems with smooth
solutions and no localized feature to chase? `compare_underpinn.py` and
`compare_underpinn_multi.py` now take a `--sampling {uniform,qr_deim,rad}`
flag (`uniform` is the default and reproduces the original comparison
exactly -- confirmed by rerunning it and matching the original numbers to
within normal run-to-run noise). `qr_deim`/`rad` periodically (every
`--resample-period` epochs, default 500) replace the interior collocation
pool with a fresh draw from `underPINN.utils.sampling.qr_deim_resample` /
`rad_resample`, weighted by the *live* PDE residual under the
currently-training network -- the same mechanism the Ramp NS ablation uses,
applied here to Burgers/Wave/Heat/Helmholtz/ODE-Harmonic/Pipe-Flow. This is
purely an underPINN-side axis: PhysicsNeMo's own numbers in every table
above are unaffected and still describe its default (effectively uniform,
`fixed_dataset=True`/random-per-step) sampling.

### Two designs: full replacement vs. a hybrid stable/adaptive split

The first pass here (`--adaptive-frac` defaulted to `1.0`) replaced the
*entire* interior pool every cycle -- a real design difference from
`ablations/ablate_qr_deim_ramp_ns.py`, which instead keeps a majority
*uniform* portion of its pool permanently fixed and only ever resamples a
smaller *adaptive* fraction (`xy_uniform + xy_bl + xy_adapt`). Full
replacement turned out to actively hurt QR-DEIM-R across the board (see the
"full replacement" columns below) -- plausibly because, unlike Ramp NS,
none of these problems has a shock for the pivoted selection to lock onto,
so replacing 100% of the pool with an aggressive, deterministic,
independence-pivoted subset can leave whole regions of a *smooth* domain
uncovered, destabilizing training rather than refining it.

`--adaptive-frac` now reproduces the Ramp NS split directly: at
`adaptive_frac < 1.0`, the first `N*(1-adaptive_frac)` points of the pool
are drawn once and **frozen for the entire run** -- uniform coverage, for
training stability -- and only the remaining `N*adaptive_frac` points are
ever replaced by `qr_deim_resample`/`rad_resample` -- residual-driven
refinement, for accuracy -- concatenated back into the same fixed-shape pool
every cycle (so `step` still compiles once; a resample only ever changes
values, never shape). The runs below use **`adaptive_frac=0.2`**: 80% of
the pool stays uniform-fixed, 20% is adaptively refreshed every 500 epochs.
This is implemented identically in both `compare_underpinn.py` (Burgers,
full-batch -- the whole pool is used every step, "hybrid" here means 80% of
that full pool is frozen) and `compare_underpinn_multi.py` (the other five,
minibatched from the pool as before).

### Results (this GPU, single run each, `weight_norm=False`, timed with no other GPU jobs running concurrently)

| problem | uniform | QR-DEIM-R, full replace | RAD, full replace | QR-DEIM-R, hybrid 80/20 | RAD, hybrid 80/20 |
|---|---|---|---|---|---|
| Burgers 1D (5,000 ep) | 8.22ms / **0.2604** | 8.96ms / 0.5658 (**2.17x worse**) | 8.79ms / **0.0842** (**0.32x, 3.1x better**) | 9.04ms / 0.1853 (0.71x better) | 8.74ms / 0.1750 (0.67x better) |
| Wave 1D (5,000 ep) | **0.63ms** / **0.01056** | 2.02ms / 0.01358 (1.29x worse) | 1.70ms / 0.01091 (1.03x worse) | 2.01ms / 0.01085 (1.03x worse) | 1.63ms / 0.01080 (1.02x worse) |
| Heat 2D (5,000 ep) | **0.41ms** / 0.00236 | 1.24ms / 0.00305 (1.29x worse) | 1.06ms / **0.00186** (0.79x better) | 1.05ms / 0.00196 (0.83x better) | 1.02ms / 0.00211 (0.89x better) |
| Helmholtz 2D (10,000 ep) | **1.15ms** / 0.000607 | 2.23ms / 0.002673 (4.40x worse) | 2.07ms / 0.000639 (1.05x worse) | 2.01ms / 0.000598 (0.98x better) | 1.96ms / **0.000590** (**0.97x, best**) |
| ODE Harmonic (3,000 ep) | **0.48ms** / 0.8282 | 1.01ms / 0.8980 (1.08x worse) | 0.85ms / **0.7928** (0.96x better) | 0.92ms / 0.8121 (0.98x better) | 0.88ms / 0.8161 (0.99x better) |
| Pipe Flow 3D (70,000 ep, velocity) | 24.80ms / 0.00866 | 23.24ms / 0.01073 (1.24x worse) | 22.96ms / **0.00823** (0.95x better) | 21.32ms / 0.00868 (1.00x, tied) | 21.06ms / 0.00888 (1.03x worse) |

(Pipe Flow's pressure-field L2 follows the same ranking each column: 0.01618
uniform -> 0.01919 QR-DEIM-R-full (worse) -> **0.01515** RAD-full (best) ->
0.01582 QR-DEIM-R-hybrid -> 0.01602 RAD-hybrid.)

**QR-DEIM-R full-replacement is worse than uniform on all six problems**
(1.03x-4.40x), confirming this is not noise. **The hybrid 80/20 split fixes
this almost completely: hybrid QR-DEIM-R now matches or beats uniform on
five of six** (Burgers 0.71x, Heat 0.83x, Helmholtz 0.98x, ODE Harmonic
0.98x, Pipe Flow tied at 1.00x), with only Wave still slightly worse (1.03x,
down from full replacement's 1.29x). This is exactly the mechanism your
"uniform maintains stability, adaptive improves the solution" framing
predicts, and it is the single largest effect in this whole ablation: going
from full replacement to an 80%-fixed/20%-adaptive split turned QR-DEIM-R
from a method that hurt every problem into one that helps or ties on
five of six, without changing the resampling rule itself at all.

**RAD tells a different, and slightly more surprising, story: it barely
needs the stability crutch.** RAD full-replacement was *already* at or
better than uniform on four of six problems (Burgers 0.32x, Heat 0.79x, ODE
Harmonic 0.96x, Pipe Flow 0.95x), roughly tied on the other two (Wave 1.03x,
Helmholtz 1.05x) -- and on **Burgers and Pipe Flow specifically, full
replacement clearly beats the hybrid version** (Burgers: 0.32x full vs.
0.67x hybrid; Pipe Flow: 0.95x full vs. 1.00x/1.03x hybrid). Helmholtz is
the one problem where hybrid RAD is the best arm in the whole table
(0.000590, edging out RAD-full's 0.000639). Our reading: RAD's own
`p(x) ~ |r(x)|^k + c` sampling rule already keeps a uniform floor (the `+ c`
term, `c=1.0` here) even when replacing the whole pool, so it never fully
abandons broad coverage the way QR-DEIM-R's deterministic pivoted selection
can -- RAD does not need an external 80%-fixed scaffold to stay stable,
because a version of that scaffold is already baked into its own sampling
density. QR-DEIM-R has no equivalent built-in floor, which is plausibly
exactly why it needed the hybrid split to become competitive.

**Practical read, updated from the full-replacement-only pass above:**
adaptive collocation is not a free accuracy win in general (uniform still
wins outright on Wave, in every arm), but with the right pool-split design
it stops being a *liability* too. QR-DEIM-R should be used with a
majority-fixed/minority-adaptive split like Ramp NS's (and this section's
0.2 fraction) rather than full-pool replacement -- see below for why we no
longer state this as conditional on "outside of problems with a strong
localized residual feature (shocks, interfaces)": Toro-3 (a real shock
problem) tests that specific claim directly, and it does not hold up. RAD is
more forgiving of full replacement and, on this evidence, is the one case
here where full replacement can beat the hybrid split rather than the other
way around -- so "hybrid is always safer" is not a universal rule, it is
specifically what fixes QR-DEIM-R's failure mode.

---

## Extending the hybrid test to the paper's two other shock problems, and to PhysicsNeMo

The six problems above are all smooth (or, for Burgers, mildly steep but not
genuinely discontinuous). `benchmarks/suite/ablations/ablate_qr_deim.py`
(1-D Toro-3 blast wave, a real Riemann shock) and
`ablate_qr_deim_ramp_ns.py` (2-D compression-ramp SBLI, a real oblique
shock) are the two places in this repo that test QR-DEIM-R/RAD on problems
that actually have the kind of localized residual feature the "hybrid split
fixes QR-DEIM-R, mostly because full replacement has nothing to lock onto on
smooth solutions" story above would predict behaves differently. Both
ablation scripts now take the same hybrid design as
`compare_underpinn_multi.py`'s `--adaptive-frac`: a new `ARM_CONFIG` maps
each arm to a resampling method and an adaptive fraction, adding
`qr_deim_hybrid`/`rad_hybrid` arms (80% of the pool frozen for the whole
run, 20% resampled every `resample_period` epochs) alongside the existing
`none`/`rad`/`qr_deim` full-replacement arms -- run here at each script's
own established convention (Toro-3: 5,000 epochs; Ramp NS: 30,000 epochs,
resample every 500, this repo's established floor for that problem).

Neither problem has ever been run against PhysicsNeMo before this. Unlike
the six problems above, PhysicsNeMo Sym ships **no built-in compressible
Euler or Navier-Stokes equations** (its `eq/pdes/` only has incompressible
`NavierStokes`, `Diffusion`, `WaveEquation`, `AdvectionDiffusion`,
`LinearElasticity`, `Electromagnetic` -- nothing for compressible/shock
flow), so both `physicsnemo/toro3/toro3.py` and
`physicsnemo/ramp_ns/ramp_ns.py` are new, from-scratch sympy `PDE`
subclasses -- a materially larger lift than reusing a built-in class the way
Wave/Heat/Pipe-Flow did:

* **`Euler1DUnsteady`** -- a direct sympy transcription of
  `underPINN.pde.euler_1d_unsteady.Euler1DUnsteadyPDE`'s conservative form
  (mass/momentum/energy, `transform="exp"` positivity, fixed artificial
  viscosity on `d^2U/dx^2`), matching `ablate_qr_deim.py`'s exact setup
  (same non-dimensionalisation, network, `art_visc=0.001`, batch sizes).
  IC/BC targets are a single sympy `Piecewise` per constraint (selects
  LEFT vs. RIGHT by x-position) rather than underPINN's per-point NumPy
  array -- exact, not an approximation.
* **`CompressibleNS2D`** -- a direct sympy transcription of
  `underPINN.pde.compressible_ns_2d.CompressibleNS2DPDE`'s conservative
  viscous form *including* its Ducros-sensor-localised artificial viscosity
  (`eps_local = art_visc * theta^2/(theta^2+omega^2) * 0.5(1-tanh(theta/s))`,
  the nonlinear shock sensor that keeps dissipation off the boundary layer)
  -- not simplified away despite being the most complex piece, since it is
  the actual mechanism the ablation's own default (`av_sensor="ducros"`)
  uses. The ramp domain is a 5-vertex PhysicsNeMo `Polygon`; inlet/upper/
  no-slip-wall/slip-wall regions are isolated from its combined boundary via
  position `criteria` -- including the same `y <= y_wall(x) + eps` test
  `RampGeometry` itself uses to tell wall points from interior/farfield ones,
  which is robust to the x-range overlap between the flat and inclined wall
  segments that a naive x-only or y-only threshold would mishandle.
  **Not matched:** PhysicsNeMo's interior draw is a single plain-uniform
  sample over the whole domain, while the ablation's baseline pool blends
  three differently-biased sub-pools (40,000 plain-uniform + 5,000
  boundary-layer-clustered, geometrically stretched toward the wall + 6,000
  uniform-but-x>=0.25-restricted) -- replicating that exact blend inside
  PhysicsNeMo's geometry API was judged not worth the engineering cost for
  what is, on the PhysicsNeMo side, a *uniform-sampling-only* reference
  point regardless (PhysicsNeMo has no adaptive-resampling analogue to
  compare against here). This gives underPINN's baseline a denser
  near-wall/downstream sample than PhysicsNeMo's -- a plausible accuracy
  advantage for underPINN not attributable to the network or solver alone.

Both scripts ran cleanly end-to-end on the first real (non-toy) attempt --
no debugging cycle to report here, unlike several of the six problems above.

### Results (this GPU, single run each, `weight_norm=False`)

**Toro-3** (5,000 epochs) -- **two independent runs, same seed=0, same
code**, shown side by side rather than picking one, because they disagree
enough to be the finding themselves:

| arm | ms/ep (run 2) | rel L2 (run 1) | rel L2 (run 2) | vs. uniform (run 1) | vs. uniform (run 2) |
|---|---|---|---|---|---|
| uniform (`none`) | 10.648 | 0.4507 | 0.4501 | -- | -- |
| QR-DEIM-R, full replace | 7.964 | 0.5352 | 0.4998 | 1.19x worse | 1.11x worse |
| RAD, full replace | 10.532 | 0.4539 | 0.4482 | 1.01x worse | 1.00x, tied |
| QR-DEIM-R, hybrid 80/20 | 9.975 | 0.4548 | **0.5314** | 1.01x worse | **1.18x worse** |
| RAD, hybrid 80/20 | 8.917 | 0.4667 | 0.4561 | 1.04x worse | 1.01x worse |
| **NVIDIA PhysicsNeMo Sym** | 19.369 | -- | 0.6435 | -- | (PhysicsNeMo's own baseline) |

**Run 1 said the hybrid split rescues QR-DEIM-R on Toro-3 (1.01x worse,
essentially tied). Run 2, same seed, same code, says the opposite: hybrid
QR-DEIM-R is the single *worst* arm (1.18x worse), worse even than full
replacement (1.11x worse).** `none` itself is stable across the two runs
(0.4507 vs. 0.4501, 0.1% apart -- no resampling, so nothing to diverge), but
every arm that resamples moved measurably, and QR-DEIM-R hybrid moved enough
to flip which story it tells. This is not a code bug -- the two runs use
identical training/resampling logic (verified: the only change between them
was adding non-functional metadata fields to the returned dict, confirmed
by `git diff`-equivalent inspection of `run_arm`), same seed, same
hyperparameters. Our best explanation, offered as a plausible mechanism
rather than a verified one: JAX/XLA GPU ops (matmul, parallel reductions)
are not guaranteed bit-reproducible run to run even at a fixed seed, and
`qr_deim_resample`/`rad_resample` read the network's current residual back
into NumPy every `resample_period` epochs -- a tiny float difference in
that residual at the *first* resample event selects measurably different
points, and PINN training is nonlinear enough that this can compound into a
qualitatively different trajectory by epoch 5,000, rather than staying a
small perturbation. `none` never hits this because it never reads params
back into a host-side resampling decision.

**Practical consequence: no single-run ranking among resampling arms on
Toro-3 should be trusted, including every ranking claim in this section
before this rerun.** We are leaving both runs in the table rather than
re-running until we like one, or quietly averaging them into a single
number that would hide exactly the instability that matters here. A
properly-powered version of this ablation needs multiple seeds *and*
multiple same-seed reruns per arm to separate "different starting point"
noise from this GPU-non-determinism noise -- out of scope for what we ran
here, flagged rather than done.

**Ramp NS** (30,000 epochs, resample every 500) -- **also two independent
runs, same seed=0, same code**, given what Toro-3's rerun found. Unlike
Toro-3, Ramp NS turns out to be reproducible: every arm's rel L2 moves by
well under 1% between runs, and the ranking (uniform best, QR-DEIM-R hybrid
worst) is identical in both:

| arm | ms/ep (run 2) | rel L2 (run 1) | rel L2 (run 2) | vs. uniform (run 2) |
|---|---|---|---|---|
| uniform (`none`) | 11.200 | 0.06873 | 0.06863 | -- |
| QR-DEIM-R, full replace (11.8% adaptive) | 11.925 | 0.06941 | 0.06925 | 1.01x worse |
| RAD, full replace (11.8% adaptive) | 11.263 | 0.06879 | 0.06882 | 1.00x, tied |
| QR-DEIM-R, hybrid 20% adaptive | 11.659 | 0.06974 | 0.06944 | 1.01x worse |
| RAD, hybrid 20% adaptive | 11.494 | **0.06845** | 0.06876 | 1.00x, tied |
| **NVIDIA PhysicsNeMo Sym** | 27.975 | -- | 0.07161 | (PhysicsNeMo's own baseline) |

So the GPU-non-determinism effect that scrambled Toro-3's ranking is real
but not universal -- Ramp NS's much larger, much longer (30,000 vs. 5,000
epochs) run apparently damps it out rather than amplifying it, the opposite
of what "more steps = more chances to diverge" would naively predict. We do
not have a confident explanation for that difference and are not
speculating past what the two problems' own numbers show.

**A second, independent metric tells a different story than rel L2 does.**
`run_arm` now also computes the peak density-gradient magnitude
`|grad(rho)|` in a band around the analytic shock line -- a schlieren-like
measure of how sharp a jump the trained network actually produces there, as
opposed to rel L2's domain-averaged error (diluted by the much larger
uniform-flow region far from the shock, where every arm does equally well
regardless of resampling strategy). Ranked by peak `|grad(rho)|` (run 2,
higher = sharper shock, not scored against a reference -- see
`plot_solutions` in `ablate_qr_deim_ramp_ns.py` for why there isn't one):

| arm | peak `|grad(rho)|` near shock | rel L2 rank |
|---|---|---|
| QR-DEIM-R, hybrid 20% adaptive | **20.649 (sharpest)** | worst |
| QR-DEIM-R, full replace | 20.471 | 2nd-worst |
| RAD, hybrid 20% adaptive | 20.357 | tied-best |
| RAD, full replace | 20.312 | tied-best |
| uniform (`none`) | 19.757 (least sharp) | **best** |

**This is close to a complete inversion of the rel-L2 ranking.** The arm
with the lowest domain-averaged error (`none`) produces the *least* sharp
shock of all five; the arm with the worst domain-averaged error
(`qr_deim_hybrid`) produces the *sharpest*. Both metrics are measuring
something real and are not in conflict so much as answering different
questions: rel L2 (evaluated on a 140x110 grid dominated by uniform
upstream/downstream flow) rewards not perturbing the easy 95% of the
domain, while peak `|grad(rho)|` specifically asks how well-resolved the
one feature that actually matters -- the shock itself -- is. Every arm that
ever touches the adaptive pool (all four resampling arms) resolves the
shock more sharply than the never-resampled baseline; QR-DEIM-R in
particular, despite being the method that most consistently loses on rel
L2 across this entire suite, produces the sharpest shock of all five arms
here, in both its full-replacement and hybrid forms. Whether that sharper
gradient is closer to the true (near-discontinuous) jump or is overshoot
is not something rel L2 or this peak-magnitude metric alone can
distinguish -- the full field plot (`results/qr_deim_ramp_ns_solutions.png`)
is worth reading directly rather than only the two summary numbers.

**Two follow-ups, both prompted by direct feedback on the plot above.**
First, a rendering fix: the shared `|grad(rho)|` colour scale was originally
set from the field's raw max, which is dominated by a genuine geometric
singularity at the inlet/wall corner `(x, y) = (0, 0)` -- the BC jumps
discontinuously there between the inlet's freestream condition and the
wall's no-slip condition (the same corner `RampGeometry.sample_interior`'s
`x_min` argument already routes adaptive sampling around, for the same
reason). Uncapped, that singularity's value is several times larger than
anything at the actual shock, so it silently washed out the shock band's
own contrast. `plot_solutions` now clips the colour scale to `1.3x` the max
*near-shock* peak across arms (`extend="max"` marks points above that
value rather than hiding the clip) -- the shock band is now the brightest,
clearly-visible feature in every panel, which it was not before.

Second: **does the artificial-viscosity coefficient, not the resampling
method, explain why every arm's peak `|grad(rho)|` clustered so closely
together?** `ART_VISC=2e-3` directly damps `d^2U/dx^2` -- it is a
numerical-stability term, and it caps how sharp *any* arm's captured shock
can get regardless of collocation strategy. `run_arm`/`main` now take an
`--art-visc` override; swept on the `none` arm (isolating the effect from
any resampling confound), 30,000 epochs each:

| `art_visc` | rel L2 | peak `|grad(rho)|` |
|---|---|---|
| 2e-3 (original default) | 0.06863 | 19.76 |
| 1e-3 | 0.06644 | 20.78 |
| 5e-4 | 0.06150 | 21.43 |
| 2e-4 | 0.05986 | 21.90 |
| 1e-4 | 0.05988 | 22.08 |
| 5e-5 | 0.05996 | 22.21 |
| 2e-5 | **0.05965** | 22.30 |
| 1e-5 | 0.05977 | **22.33** |

**Confirmed: the default was a real, unnecessary ceiling.** Dropping it 10x
(to `2e-4`) improves both metrics substantially and monotonically -- 12.8%
lower error, 10.8% sharper shock -- with the field plot still clean. Pushed
a further 20x past that (down to `1e-5`, three orders of magnitude below
the original default), the trend **plateaus rather than breaking down**:
rel L2 sits in a tight, non-monotonic ~0.5%-wide band (0.05965-0.05996) the
rest of the way, and peak `|grad(rho)|` keeps creeping up but with
strongly diminishing returns (+10.8% from 2e-3 to 2e-4; only +1.9% more
from 2e-4 all the way to 1e-5). **We swept three orders of magnitude and
never found an instability threshold** -- worth stating plainly rather than
implying we found a cliff we didn't.

We did check the field plots at the lowest values for Gibbs-type ringing
the scalar metrics might miss, and initially flagged a faint mottled
texture near the domain's *other* corner (`x~0, y~1`, the inlet/upper
farfield corner) in the lowest-viscosity runs as a possible early
degradation signal -- **that turned out to be wrong**: the same texture is
present at the *original* `art_visc=2e-3` plot too (visible in the `qr_deim`
and `rad_hybrid` rows of `results/qr_deim_ramp_ns_solutions.png`), so it
predates and is independent of the viscosity sweep, not a low-viscosity
artifact. Flagging the correction rather than leaving the wrong claim
standing.

**Practical recommendation:** given the plateau, `art_visc=2e-4` (not the
most extreme value tested) is the sensible choice -- it captures nearly all
of the available gain in both metrics, and there is no evidence pushing
lower buys anything further. We reran all five resampling arms at
`art_visc=2e-4` to check whether relaxing the ceiling changes the earlier
comparison; **it complicates the story rather than resolving it**:

| arm | rel L2 @ 2e-4 | peak `|grad(rho)|` @ 2e-4 |
|---|---|---|
| uniform (`none`) | **0.05986** (best) | 21.90 |
| RAD, full replace | 0.06012 | 21.53 |
| QR-DEIM-R, full replace | 0.06020 | 21.66 |
| QR-DEIM-R, hybrid | 0.06039 | **22.20** (sharpest) |
| RAD, hybrid | 0.06089 (worst) | 21.93 |

The inter-arm spread did **not** widen once the viscosity ceiling was
relaxed -- if anything it is slightly *tighter* than at the default
viscosity (rel L2: ~1.7% here vs. ~1.9% at 2e-3; peak `|grad(rho)|`: ~3.1%
vs. ~4.5%), the opposite of what "the viscosity ceiling was suppressing the
algorithm differences" would have predicted. QR-DEIM-R hybrid stays the
sharpest-shock arm at both viscosities (a reproducible finding across this
whole exercise), but the rel-L2-*worst* arm changes from `qr_deim_hybrid`
(at 2e-3) to `rad_hybrid` (at 2e-4) -- so even the "which arm is worst"
ranking is not stable across viscosity settings, on top of not being stable
across reruns at fixed viscosity (the Toro-3 finding earlier in this
section). We are reporting this as a genuine result, not the confirmation
of a tidier hypothesis: lowering artificial viscosity is a real, free
improvement worth making on its own merits (sharper shock, lower error, no
instability found), but it does not make the resampling-method comparison
more decisive, and we do not have a good explanation for why not.

**Does the improved viscosity change the underPINN-vs-PhysicsNeMo picture
too?** Reran PhysicsNeMo's own `ramp_ns.py` at `art_visc=2e-4` to check --
via a new `ART_VISC` env var (PhysicsNeMo's hydra config is a strict
structured schema that rejects an arbitrary new `art_visc=X` CLI key even
when declared in `config.yaml`, so an env var was used instead of a hydra
override for the value itself).

**Caught a real bug doing this, not just a config inconvenience: the first
`art_visc=2e-4` attempt silently trained on nothing.** PhysicsNeMo's
checkpoint/restart directory is named from its *hydra* CLI overrides only
(`outputs/jit=false,training.max_steps=30000/ramp_ns`); since `ART_VISC` is
an env var, not a hydra key, it isn't part of that name, so the `2e-4` run
landed on the exact same directory the earlier `2e-3` run had already
finished in, "restored" that already-`step=30000` checkpoint, and exited
immediately having done zero additional training. The giveaway was the
reported timing, not the accuracy number (which looked superficially
plausible): `train=3.82s` and `ms/epoch=0.127`, versus every real run on
this problem taking 500-900s. Fixed by adding a distinguishing
`+av_tag=avX` hydra override (a second `+`-prefixed key, purely so it shows
up in the auto-generated directory name) alongside any non-default
`ART_VISC` -- both documented directly in `ramp_ns.py` now. The corrected,
verified-real run: `train=541s`, `ms/epoch=18.0` -- back in the expected
range.

| | art_visc=2e-3 (original) | art_visc=2e-4 (improved) |
|---|---|---|
| underPINN ms/ep (best arm) | 11.20 | 11.21 |
| PhysicsNeMo ms/ep | 27.975 | **18.042** |
| speed ratio (PN/underPINN) | 2.50x | **1.61x** |

**PhysicsNeMo's own throughput improved at the lower viscosity.**
PhysicsNeMo's ms/epoch dropped 35% (27.975 -> 18.042) at the lower
viscosity while underPINN's stayed flat (11.20 -> 11.21) -- narrowing the
speed gap from 2.50x down to 1.61x, PhysicsNeMo's best showing anywhere in
this whole comparison suite. We do not have a mechanistic explanation for
why a PDE coefficient value would change PhysicsNeMo's per-step wall-clock
this much and are not speculating past what the two numbers show -- it may
be ordinary run-to-run GPU timing variance of the kind documented earlier
in this section (Toro-3's non-reproducible reruns), just landing in
PhysicsNeMo's favour this time. This remains a real, if narrowed,
throughput advantage for underPINN at the improved setting.

**underPINN is faster on both** (1.87x on Toro-3, 2.55x on Ramp NS), though
both speedups are smaller than most of the six smoother problems above. We
do not have a confident mechanistic explanation for why -- it may be that
PhysicsNeMo Sym, built with exactly this kind of engineering-CFD
shock/compressible-flow use case in mind, has a genuine per-step-cost
advantage on this problem type that the six smoother problems don't
exercise, or simply a difference in how each framework's per-step cost
scales with this problem's heavier residual. Flagging the uncertainty
rather than picking whichever story sounds better.

**Toro-3 revises the "QR-DEIM-R only helps on shocks" reading from the
six-problem section above -- and its own two runs revise each other.**
Toro-3 has a genuine Riemann shock -- exactly the kind of concentrated,
low-dimensional residual structure that section speculated QR-DEIM-R's
pivoted selection needs to pay off -- and full-replacement QR-DEIM-R
underperforms uniform in *both* runs (1.19x and 1.11x worse), just by less
than on the smooth problems (up to 4.40x worse on Helmholtz) rather than by
winning outright. Whether the hybrid split helps beyond that, though, is
exactly the part that flipped between runs (1.01x worse in run 1, 1.18x
worse -- the single worst arm -- in run 2): "hybrid fixes QR-DEIM-R because
full replacement only fails without a shock to lock onto" is not a claim
either run alone can support with any confidence now that we know the same
seed can produce either outcome. **Ramp NS, by contrast, is reproducible --
confirmed with a second run, unlike Toro-3 -- and it consistently does *not*
support the "hybrid fixes it" story: QR-DEIM-R hybrid is the single *worst*
arm by rel L2 in both runs**, not rescued at all, a reversal of the pattern
on most of the six smooth problems. Every arm (including both
full-replacement ones) sits within a ~1.9% rel-L2 band of uniform on Ramp
NS, the smallest spread of any problem tested here -- we think that small
spread is best explained not by the shock but by Ramp NS's baseline already
not being plain uniform (see the "not matched" pool-composition note
above): its `none` arm already blends a boundary-layer-clustered sub-pool
and a frozen, x-restricted "adaptive" sub-pool on top of uniform sampling,
leaving far less headroom for any resampling *strategy* choice to move the
number. **The peak-`|grad(rho)|` metric above complicates even that,
though**: by that measure QR-DEIM-R hybrid -- rel L2's worst arm on Ramp
NS -- produces the *sharpest* resolved shock of all five, and `none` --
rel L2's best arm -- the *least* sharp. "Worse" and "better" here depend
entirely on which question you're asking (domain-averaged accuracy vs.
shock sharpness), which is itself a reason to distrust any single-scalar
ranking claim in this whole ablation, on any problem, without also looking
at the field plot. Net: the shock/no-shock framing from the six-problem
section does not survive contact with two more shock problems tested
directly, no specific per-arm rel-L2 ranking on Toro-3 survives contact
with a second run of the *same* problem, and even Ramp NS's reproducible
rel-L2 ranking does not agree with its own shock-sharpness ranking. The
honest updated picture is that we do not have a reliable predictor of when
QR-DEIM-R full replacement will or won't hurt beyond "it usually does" --
and we no longer have a reliable claim
that the hybrid split fixes it *consistently*, only that it did in most of
the runs we happened to look at.

### Five fixes/additions, all prompted by direct questions

Five things changed in `ablations/ablate_qr_deim.py` and
`ablate_qr_deim_ramp_ns.py` after this section was first written, each in
response to being asked directly rather than caught independently:

1. **"Why do the uniform points look like they moved in the migration
   plot?"** They don't -- verified with `np.array_equal` on the actual
   arrays (not just re-read from the plot), `max abs diff = 0.0` between the
   fixed portion's initial and final snapshots for every hybrid arm. The
   *plot* was the bug: it drew the fixed and adaptive points in one colour,
   so a newly-visible structure (the adaptive points tracing the shock) read
   as if the whole 40,000-point cloud had changed. `plot_migration` in both
   scripts now colour-splits fixed (grey) from adaptive (red) points in
   every panel -- see the regenerated
   `results/qr_deim_toro3_collocation_migration.png` -- which makes the
   "grey never moves" claim visible rather than just asserted.
2. **Results are now cached for replotting without retraining.** Both
   scripts' large `_`-prefixed plotting arrays (previously computed, used
   for the PNGs, and discarded) are now also written to
   `results/ablation_qr_deim_{toro3,ramp_ns}_raw.npz` via
   `common.save_raw_arrays`; a new `--replot` flag reloads that npz plus the
   existing JSON and regenerates both plots in a few seconds with no GPU
   device, no training loop, no seed-dependent rerun risk of the kind the
   section above just documented.
3. **"Plot the density gradient instead of the exact solution, to see which
   arm resolves the shock sharpest."** Only applies to
   `ablate_qr_deim_ramp_ns.py` (Toro-3 is 1-D and already plots the field
   directly, no second spatial dimension to take a gradient across). The
   `plot_solutions` middle panel (previously the analytic Mach reference,
   redundant with the same field already needed for the error panel) is now
   each arm's own `|grad(rho)|` -- see the results table and field plot
   above.
4. **"The third column isn't necessary."** The `|error|`-vs-Mach-reference
   panel is dropped; `plot_solutions` is now a 2-column [Mach] [`|grad(rho)|`]
   layout per arm, and rel L2 (which the error panel was a spatial breakdown
   of) already sits in the Mach panel's own title.
5. **"Clip the colormap so the shock is visible, and sweep the artificial
   viscosity further."** Both addressed above: the colour-scale clip fix and
   the full viscosity sweep (2e-3 down to 1e-5, including the corrected
   "the corner artifact isn't viscosity-related" finding and the
   `art_visc=2e-4` resampling-method recheck) are documented in the
   preceding subsection, not repeated here.
