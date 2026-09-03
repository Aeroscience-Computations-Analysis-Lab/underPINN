# Transfer Learning

underPINN supports two transfer-learning modes, both using the same warm-start API:
`solver.load_params(...)` or `solver.restore_checkpoint(...)`.

## Parameter transfer (different ν / Re / diffusivity)

Warm-start from a trained model when changing a physics parameter — converges **2–3×
faster** than training from scratch.

```python
# Phase 1: train source model (e.g. Burgers ν=0.1)
solver_src.train(*data_src, config=cfg_src)
solver_src.save_checkpoint("outputs/source/")

# Phase 2: warm-start target from source weights, then fine-tune (e.g. ν=0.01)
solver_tgt.load_params(solver_src.params)        # or restore_checkpoint("outputs/source/")
solver_tgt.train(*data_tgt, config=cfg_tgt)      # lower lr recommended (3e-4 instead of 1e-3)
```

```{tip}
Use a lower learning rate for the fine-tuning phase (e.g. `3e-4` instead of `1e-3`) to
avoid destroying the warm-started weights in the first few epochs.
```

## Temporal transfer (extended time horizon)

Extend the trained time horizon by fine-tuning on a new interval, starting from a
previously trained checkpoint.

```python
# Phase 1: train on t ∈ [0, T_1]
solver_phase1.train(*data_t1, config=cfg_phase1)

# Phase 2: extend to t ∈ [0, T_2], T_2 > T_1, warm-start from Phase 1
solver_phase2.load_params(solver_phase1.params)
solver_phase2.train(*data_t2, config=cfg_phase2)
```

Both modes are demonstrated in `examples/transfer/burgers_transfer.py` and
`examples/pipe_flow/pipe_flow_unsteady_transfer.py`.

## Time-marching transfer (windowed, long-horizon unsteady flows)

For problems where the time horizon is too long to train in one shot — the **3-D
pulsatile pipe flow** case, for instance — underPINN splits the horizon into windows:

- Each window **warm-starts** from the previous window's trained weights
- The previous window's **end-state** is chained in as the next window's initial
  condition
- **Per-window checkpoints** are written, and restart is tracked **at the window level**
  (not just the epoch level) — see {doc}`restart`

```bash
python -m underPINN run examples/pipe_flow/pipe_flow_pulsatile_transfer.yaml
```

```{admonition} Why window instead of training end-to-end?
:class: note
Directly training a PINN across a long unsteady horizon suffers from causality
violation and vanishing-gradient-like effects as the temporal domain grows. Windowed
time-marching keeps each sub-problem well-posed and numerically tractable while
still producing one continuous solution across the full horizon.
```

```{seealso}
{doc}`training` for the general transfer/warm-start mechanics, and {doc}`checkpointing`
for the `ModelPredictor` API used to query trained windows.
```
