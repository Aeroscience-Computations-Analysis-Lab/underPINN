"""BenchmarkResult + BenchmarkRunner — orchestrates multi-problem, multi-epoch runs.

Usage
-----
::

    from underPINN.benchmark_utils import BenchmarkRunner, EVALUATOR_REGISTRY

    runner = BenchmarkRunner(
        problems=["burgers", "wave", "ode_exp"],
        epoch_budgets=[500, 1000, 2000, 5000],
        seed=0,
        verbose=True,
    )
    results = runner.run()
    runner.save_json("outputs/bench/results.json")
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional


# =============================================================================
#  Data record
# =============================================================================

@dataclass
class BenchmarkResult:
    """One measurement: a single (problem, epochs) data point."""

    problem: str                #: evaluator key, e.g. "burgers"
    label: str                  #: human label, e.g. "1-D Burgers (ν=0.01)"
    epochs: int
    rel_l2: float               #: relative L2 error vs exact (NaN if unavailable)
    max_ae: float               #: maximum absolute error (NaN if unavailable)
    loss_final: float           #: total loss at last training step
    pde_loss_final: float       #: PDE component of loss at last training step
    wall_time_s: float          #: training wall-clock time
    ms_per_epoch: float         #: wall_time_s × 1000 / epochs
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers

    @property
    def log10_rel_l2(self) -> float:
        return math.log10(self.rel_l2) if self.rel_l2 > 0 else float("nan")

    def as_row(self) -> dict:
        """Return flat dict suitable for CSV / tabular display."""
        return {
            "problem":         self.problem,
            "label":           self.label,
            "epochs":          self.epochs,
            "rel_l2":          self.rel_l2,
            "max_ae":          self.max_ae,
            "loss_final":      self.loss_final,
            "pde_loss_final":  self.pde_loss_final,
            "wall_time_s":     self.wall_time_s,
            "ms_per_epoch":    self.ms_per_epoch,
        }


# =============================================================================
#  Runner
# =============================================================================

class BenchmarkRunner:
    """Runs each (problem, epoch_budget) combination and collects results.

    Parameters
    ----------
    problems :
        List of evaluator keys from :data:`EVALUATOR_REGISTRY`.
        Pass ``None`` to use all registered problems.
    epoch_budgets :
        List of epoch counts to test per problem.
    seed :
        Base PRNG seed forwarded to every evaluator.
    fast_only :
        When ``True`` (default), skip evaluators marked ``fast=False``
        (e.g. 3-D pipe flow with expensive Hessians).
    verbose :
        Print per-run progress lines.
    """

    def __init__(
        self,
        problems: Optional[List[str]] = None,
        epoch_budgets: Optional[List[int]] = None,
        seed: int = 0,
        fast_only: bool = True,
        verbose: bool = True,
    ):
        from underPINN.benchmark_utils.evaluators import (
            EVALUATOR_REGISTRY, SLOW_PROBLEMS)

        self.epoch_budgets = epoch_budgets or [500, 1000, 2000, 5000]
        self.seed = seed
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self._loss_snapshots: dict = {}   # problem → list[list] keyed by epochs

        if problems is None:
            problems = list(EVALUATOR_REGISTRY.keys())
        if fast_only:
            problems = [p for p in problems if p not in SLOW_PROBLEMS]

        self._problems   = problems
        self._registry   = EVALUATOR_REGISTRY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[BenchmarkResult]:
        """Run all (problem, epochs) combinations.

        Each evaluator is constructed fresh per problem so state does not
        leak across epoch budgets for the same problem.

        Returns
        -------
        List[BenchmarkResult]
            Also stored in ``self.results``.
        """
        n_total = len(self._problems) * len(self.epoch_budgets)
        counter = 0

        for prob in self._problems:
            cls = self._registry[prob]
            self._loss_snapshots[prob] = {}

            for epochs in self.epoch_budgets:
                counter += 1
                if self.verbose:
                    print(
                        f"\n[{counter}/{n_total}]  {prob:<20s}  epochs={epochs:>6d}",
                        flush=True,
                    )

                ev = cls()
                try:
                    _ctx = (contextlib.redirect_stdout(io.StringIO())
                            if not self.verbose else contextlib.nullcontext())
                    with _ctx:
                        wall = ev.train(epochs=epochs, seed=self.seed)
                        metrics = ev.evaluate()
                    loss_final = ev.loss_hist[-1] if ev.loss_hist else float("nan")
                    pde_final  = ev.pde_hist[-1]  if ev.pde_hist  else float("nan")
                    self._loss_snapshots[prob][epochs] = list(ev.loss_hist)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR: {exc}")
                    wall, metrics = 0.0, {}
                    loss_final = pde_final = float("nan")

                rel_l2 = metrics.get("rel_l2", float("nan"))
                max_ae = metrics.get("max_ae", float("nan"))
                ms_per = 1000.0 * wall / max(epochs, 1)

                result = BenchmarkResult(
                    problem=prob,
                    label=ev.label,
                    epochs=epochs,
                    rel_l2=rel_l2,
                    max_ae=max_ae,
                    loss_final=loss_final,
                    pde_loss_final=pde_final,
                    wall_time_s=wall,
                    ms_per_epoch=ms_per,
                )
                self.results.append(result)

                if self.verbose:
                    _fmt = lambda v: f"{v:.3e}" if not math.isnan(v) else "  N/A  "
                    print(
                        f"  rel-L2={_fmt(rel_l2)}  "
                        f"loss={_fmt(loss_final)}  "
                        f"time={wall:.1f}s  "
                        f"({ms_per:.2f} ms/epoch)"
                    )

        if self.verbose:
            print(f"\nBenchmark complete — {len(self.results)} results.")
        return self.results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(self, path: str) -> None:
        """Serialise all results to a JSON file (append if exists)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        existing: list = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass

        new_entries = [asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(existing + new_entries, f, indent=2)
        print(f"Results saved → {path}")

    def save_loss_npz(self, path: str) -> None:
        """Save per-problem loss histories to a .npz archive."""
        import numpy as np
        arrays = {}
        for prob, snap in self._loss_snapshots.items():
            for ep, hist in snap.items():
                arrays[f"{prob}__ep{ep}"] = np.array(hist)
        np.savez(path, **arrays)
        print(f"Loss histories saved → {path}")

    @classmethod
    def load_json(cls, path: str) -> List[BenchmarkResult]:
        """Load a previously saved results JSON file."""
        with open(path) as f:
            raw = json.load(f)
        results = []
        for entry in raw:
            extra = {k: v for k, v in entry.items()
                     if k not in BenchmarkResult.__dataclass_fields__}
            base  = {k: v for k, v in entry.items()
                     if k in BenchmarkResult.__dataclass_fields__}
            base["extra"] = base.get("extra", extra)
            results.append(BenchmarkResult(**base))
        return results
