# Restart / Resume System

Fault-tolerant training that resumes exactly where it left off. Set
`save_restart_every: 500` in your YAML (or `TrainingConfig`) and an interrupted run
automatically resumes from the last snapshot the next time it's launched — no code
changes needed.

```{admonition} How resumption is gated
:class: important
Solvers check the `done` flag in `meta.json`. If `done: false` (the run was
interrupted), the snapshot is restored automatically — **regardless of config
changes**. Use `python -m underPINN resume config.yaml` to verify config integrity
before resuming a completed run. To force a fresh start, delete `<out_dir>/restart/`
manually.
```

## How it works

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1. Snapshot written every N epochs
Every `save_restart_every` epochs, `RestartManager` writes `params.msgpack`,
`opt_state.msgpack`, `hists.npz`, and `meta.json` to `<out_dir>/restart/`.
:::

:::{grid-item-card} 2. Re-run checks `done` and resumes
On re-run, `RestartManager` reads `meta.json`. If `done: false`, params, optimizer
state, and loss histories are restored — training continues from the saved epoch.
Plots stay continuous across restarts.
:::

:::{grid-item-card} 3. Config-change safety via `resume`
Solvers do **not** hash-check configs by default — an interrupted run resumes even if
you changed `lr`, `epochs`, etc. Run `python -m underPINN resume config.yaml` to
detect config drift before resuming.
:::

:::{grid-item-card} 4. Completion marks the snapshot done
After training finishes — normally or via early stopping — `done()` writes
`"done": true` to `meta.json`. The next run with the same config starts fresh
instead of re-resuming a completed run.
:::
::::

## Snapshot directory contents

```{list-table}
:header-rows: 1
:widths: 25 75

* - File
  - Contents
* - `params.msgpack`
  - Flax-serialised model parameters at the snapshot epoch
* - `opt_state.msgpack`
  - Flax-serialised optimizer state (Adam moments, step count)
* - `hists.npz`
  - All loss history arrays accumulated so far (`loss_hist`, `pde_hist`, etc.)
* - `meta.json`
  - `{"epoch": N, "cfg_hash": null, "done": false}` — `cfg_hash` is `null` when
    written by a solver directly; populated by the `resume` CLI command
```

## Configuration

`````{tab-set}
````{tab-item} YAML
```yaml
training:
  save_restart_every: 500   # 0 to disable
```
````
````{tab-item} Python API
```python
config = TrainingConfig(
    epochs             = 10000,
    out_dir            = "outputs/burgers",
    save_restart_every = 500,
)
solver.train(*data, config=config)
# If killed at epoch 3700, the next run resumes
# from epoch 3500 (last snapshot) automatically.
```
````
`````

## Verifying config changes before resuming

```bash
python -m underPINN resume examples/burgers/config.yaml
```

`resume` computes the MD5 of the current YAML, compares it against the hash stored in
`meta.json`, and warns you if any field changed since the last snapshot (learning rate,
epoch count, network layers, physics parameters, …). If everything is consistent, it
resets `done` to `false` so the next `run` continues training.

```{tip}
To force a fresh start without the `resume` command, simply delete
`<out_dir>/restart/`, or set `save_restart_every: 0` temporarily.
```

```{seealso}
{doc}`checkpointing` covers the separate, longer-lived `params.msgpack` artifact that
every runner writes on **successful completion** — distinct from the in-progress
restart snapshot described here.
```
