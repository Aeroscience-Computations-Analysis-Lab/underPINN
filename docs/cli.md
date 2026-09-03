# Command-Line Interface

underPINN ships a YAML-driven CLI so experiments require **zero code changes**. Run
scripts directly, or point the CLI at the same YAML — both work identically.

```{list-table}
:header-rows: 1
:widths: 18 82

* - Command
  - Description
* - `run`
  - Run a single problem from a YAML config
* - `resume`
  - Verify the config MD5 hash against a stored snapshot and reset `done` for
    continuation; warns if any field changed
* - `sweep`
  - Cartesian-product hyperparameter sweep; each combination gets its own sub-directory
* - `bench`
  - Full benchmark suite across all registered problems
* - `list`
  - List all registered runners
* - `show`
  - Print the resolved config without training
* - `version`
  - Print the framework version string
```

## Single run

```bash
python -m underPINN run examples/burgers/config.yaml
python -m underPINN run examples/wave/config.yaml
python -m underPINN run examples/helmholtz/config.yaml
python -m underPINN run examples/ramp/config.yaml
python -m underPINN run examples/airfoil/config.yaml
python -m underPINN run examples/cylinder/config.yaml
python -m underPINN run examples/sod_shock/config.yaml
python -m underPINN run examples/AAA/config.yaml
python -m underPINN run examples/pipe_flow_rheology/config.yaml
python -m underPINN run examples/pipe_flow/pipe_flow.yaml
```

## Hyperparameter sweep

Cartesian product across any dot-separated config key. Each run gets its own
sub-directory with a saved `config.yaml` for full reproducibility.

```bash
python -m underPINN sweep examples/burgers/burgers_nu_sweep.yaml
```

```yaml
# sweep YAML anatomy
base:                           # shared config for all runs
  problem: burgers
  network:
    type: mlp
    layers: [2, 64, 64, 64, 1]
  training:
    epochs: 5000

sweep:                          # dot-separated key → list of values
  physics.nu       : [0.1, 0.05, 0.025, 0.01]
  training.epochs  : [3000, 5000]
```

Each run lands in `outputs/…/run_000`, `run_001`, …

## Inspect & list

```bash
python -m underPINN show examples/wave/config.yaml     # print resolved config
python -m underPINN resume examples/burgers/config.yaml # verify config hash, allow resume
python -m underPINN list                                # list registered runners
python -m underPINN version                              # print version string
```

```text
Registered runners: burgers, wave, helmholtz, heat_forward, heat_inverse,
ode, ldc, airfoil, pipe_flow, ramp, burgers_transfer,
pipe_flow_unsteady_transfer, inverse_diffusion, ...
```

## Benchmark suite

```bash
python -m underPINN bench
python -m underPINN bench \
    --problems burgers wave ode_exp \
    --epochs 500 2000 5000 \
    --output outputs/bench
python -m underPINN bench --all
python -m underPINN bench --from-json outputs/bench/results.json
```

See {doc}`benchmark` for the full options and output file reference.

## Adding a new physics case

Registering a new problem only requires touching one dispatch table:

```{code-block} python
:caption: underPINN/runner/dispatch.py

_REGISTRY = {
    "burgers"  : ("examples/burgers/burgers.py",  "run_burgers"),
    "wave"     : ("examples/wave/wave.py",        "run_wave"),
    "mycase"   : ("examples/mycase/mycase.py",    "run_mycase"),  # ← add this
    # ... no other files need to change
}
```

```{admonition} Three-step recipe
:class: note
1. Create `examples/mycase/mycase.py` — define `run_mycase(cfg) -> dict`
2. Create `examples/mycase/config.yaml` — set `problem: mycase`
3. Add **one line** to `underPINN/runner/dispatch.py`

No other files need to change.
```
