# Reviewer-response benchmarks

Code answering the experimental concerns raised in review. Each script produces
a JSON result under `results/`, and `summarize.py` renders them as one report.

```bash
bash benchmarks/rebuttal/run_all.sh 5000        # GPU, paper settings
ALLOW_CPU=1 bash benchmarks/rebuttal/run_all.sh 50   # correctness smoke test
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

---

## Not covered here (prose, not code)

The review also asks for discussion of **jinns** and other JAX-native PINN
frameworks, **QR-DEIM / QR-DEIM-R** adaptive selection, and **second-order /
natural-gradient (D-NGD)** training. Those are related-work and positioning
changes to the manuscript, not experiments — no code in this directory
addresses them. Benchmarking against NVIDIA Modulus or jinns would be a
genuine addition, but needs those packages installed and their own tuned
configurations to be a fair rather than a strawman comparison; that is a larger
piece of work than this suite.
