# Benchmark Suite

One command trains every registered problem across multiple epoch budgets and produces
a complete analysis package — plots, CSV, Markdown, and reusable JSON.

## CLI usage

```bash
# Run all fast problems — smooth PDEs get [500, 1000, 2000, 5000] epochs;
# problems marked complex=True (ramp, toro3) get [2000, 5000, 15000, 40000]
python -m underPINN bench

# Select specific problems and a custom budget for the simple ones
python -m underPINN bench \
    --problems burgers wave helmholtz heat_steady ode_harmonic ramp toro3 \
    --epochs 500 1000 2000 5000 \
    --output outputs/bench

# Include slow problems (3-D pipe flow, viscous ramp NS — also complex=True)
python -m underPINN bench --all

# Override the complex-problem budget directly
python -m underPINN bench --all --complex-epochs 2000 5000 15000

# Regenerate plots from a previous run without re-training
python -m underPINN bench --from-json outputs/bench/results.json
```

```{admonition} Two-tier epoch budgets
:class: note
Smooth PDEs (Burgers, wave, Helmholtz, …) use the default `[500, 1000, 2000, 5000]`.
Problems marked `complex=True` on their evaluator — shocks, viscous SBLI, 3-D N-S
(`ramp`, `toro3`, `pipe_flow`, `ramp_ns`) — automatically get a much larger
`[2000, 5000, 15000, 40000]` budget, because they converge far more slowly and the
shared small budget was under-training them. Both tiers are overridable via
`--epochs` / `--complex-epochs`.
```

## Programmatic usage

```python
from underPINN.benchmark_utils import BenchmarkRunner, generate_report

runner = BenchmarkRunner(
    problems      = ["burgers", "wave", "ode_exp", "helmholtz"],
    epoch_budgets = [500, 1000, 2000, 5000],
    seed          = 0, fast_only=True, verbose=True,
)
results = runner.run(out_dir="outputs/bench")
runner.save_json("outputs/bench/results.json")
generate_report(results, runner, out_dir="outputs/bench")
```

## Outputs written to `outputs/bench/`

```{list-table}
:header-rows: 1
:widths: 32 68

* - File
  - Description
* - `accuracy_vs_epochs.png`
  - Log-log rel-L² vs epoch budget, one curve per problem
* - `accuracy_summary_bar.png`
  - Grouped bar chart of rel-L² at each epoch budget
* - `wall_time_vs_epochs.png`
  - Training time vs epoch budget
* - `ms_per_epoch.png`
  - Bar chart of training throughput per problem
* - `loss_grid.png`
  - Convergence curves for every problem
* - `benchmark_results.csv`
  - Full raw data table — importable into pandas
* - `benchmark_summary.md`
  - Markdown table, one row per problem at max epochs
* - `results.json`
  - Reusable for `--from-json` replays without re-training
```

```{tip}
`results.json` is a complete, self-contained snapshot of a benchmark run — check it
into version control alongside a paper or report so figures can be regenerated
byte-for-byte without re-running any GPU training.
```
