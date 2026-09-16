"""Multi-seed throughput + accuracy for the four heavy benchmark problems
(Ramp, Toro Test 3, Pipe Flow, Ramp NS) -- extends the existing 3-seed check
already done for the five cheap problems (Burgers/Wave/Helmholtz/Steady
Heat/ODE Harmonic) to the shock/3-D problems that were still single-run.

Epoch budgets match each problem's existing single-run headline number
elsewhere in the paper:
    ramp      20,000  ("reaching ~1% relative L2 by 20,000 epochs")
    toro3     60,000  ("stays above 45% relative L2 even at 60,000 epochs")
    pipe_flow 60,000  ("7.5e-4 at 60,000 epochs")
    ramp_ns   30,000  (matches the existing 5-seed QR-DEIM-R ablation budget)

Usage:
    python benchmarks/suite/multiseed_heavy_throughput_accuracy.py
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from underPINN.benchmark_utils.evaluators import EVALUATOR_REGISTRY

BUDGETS = {
    "ramp":      20000,
    "toro3":     60000,
    "pipe_flow": 60000,
    "ramp_ns":   30000,
}
SEEDS = [0, 1, 2]

OUT_PATH = "benchmarks/suite/results/multiseed_heavy_throughput_accuracy.json"


def main() -> None:
    results: dict[str, list[dict]] = {}
    t_start = time.time()
    for prob, epochs in BUDGETS.items():
        cls = EVALUATOR_REGISTRY[prob]
        results[prob] = []
        for seed in SEEDS:
            print(f"\n=== {prob}  epochs={epochs}  seed={seed} ===", flush=True)
            ev = cls()
            t0 = time.time()
            wall = ev.train(epochs=epochs, seed=seed)
            metrics = ev.evaluate()
            elapsed = time.time() - t0
            row = {
                "problem": prob,
                "epochs": epochs,
                "seed": seed,
                "wall_s": wall,
                "ms_per_epoch": 1000.0 * wall / epochs,
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
