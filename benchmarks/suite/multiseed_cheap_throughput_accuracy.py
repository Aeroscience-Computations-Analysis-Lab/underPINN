"""Multi-seed throughput + accuracy for the five cheap benchmark problems
(Burgers, Wave, Helmholtz, Steady Heat, ODE Harmonic), at their own default
BenchmarkRunner budget (5,000 epochs -- the "largest budget tested" already
cited in the paper for these problems' own convergence curves, matching
`BenchmarkRunner`'s default `epoch_budgets=[500,1000,2000,5000]`). Extends
the existing 3-seed check already done for these problems (whose per-seed
raw data was never saved) with saved per-seed JSON, and complements
multiseed_heavy_throughput_accuracy.py's 4 heavy-problem results so both
summary figures can be regenerated with real error bars.

Usage:
    python benchmarks/suite/multiseed_cheap_throughput_accuracy.py
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from underPINN.benchmark_utils.evaluators import EVALUATOR_REGISTRY

PROBLEMS = ["burgers", "wave", "helmholtz", "heat_steady", "ode_harmonic"]
EPOCHS = 5000
SEEDS = [0, 1, 2]

OUT_PATH = "benchmarks/suite/results/multiseed_cheap_throughput_accuracy.json"


def main() -> None:
    results: dict[str, list[dict]] = {}
    t_start = time.time()
    for prob in PROBLEMS:
        cls = EVALUATOR_REGISTRY[prob]
        results[prob] = []
        for seed in SEEDS:
            print(f"\n=== {prob}  epochs={EPOCHS}  seed={seed} ===", flush=True)
            ev = cls()
            t0 = time.time()
            wall = ev.train(epochs=EPOCHS, seed=seed)
            metrics = ev.evaluate()
            elapsed = time.time() - t0
            row = {
                "problem": prob,
                "epochs": EPOCHS,
                "seed": seed,
                "wall_s": wall,
                "ms_per_epoch": 1000.0 * wall / EPOCHS,
                **metrics,
            }
            results[prob].append(row)
            print(f"  wall={wall:.1f}s  ms/ep={row['ms_per_epoch']:.3f}  "
                  f"metrics={metrics}  (elapsed this run: {elapsed:.1f}s, "
                  f"total so far: {time.time()-t_start:.1f}s)", flush=True)

            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Total wall: {time.time()-t_start:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
