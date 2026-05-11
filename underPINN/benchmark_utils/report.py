"""Reporting utilities: plots, CSV table, Markdown summary.

All functions accept a list of :class:`BenchmarkResult` objects and an output
directory.  Outputs are written to ``{out_dir}/``:

* ``accuracy_vs_epochs.png``   — rel-L2 vs epoch budget for every problem
* ``wall_time_vs_epochs.png``  — training time vs epoch budget
* ``loss_grid.png``            — convergence curves for the max-epoch run
* ``benchmark_results.csv``    — full raw data table
* ``benchmark_summary.md``     — markdown table (one row per max-epoch run)

Usage
-----
::

    from underPINN.benchmark_utils import BenchmarkRunner
    from underPINN.benchmark_utils.report import generate_report

    runner = BenchmarkRunner(problems=["burgers", "wave"], epoch_budgets=[1000, 5000])
    results = runner.run()
    generate_report(results, runner, out_dir="outputs/bench")
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from underPINN.benchmark_utils.benchmark_suite import BenchmarkResult


# colour palette (colour-blind-friendly)
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#999999",
]


# =============================================================================
#  Individual plots
# =============================================================================

def plot_accuracy_vs_epochs(
    results: List[BenchmarkResult],
    out_dir: str,
    *,
    filename: str = "accuracy_vs_epochs.png",
    ylim: Optional[tuple] = None,
) -> str:
    """Log-log accuracy (rel-L2) vs epoch budget, one line per problem."""
    # Group by problem
    groups: Dict[str, list] = {}
    for r in results:
        if not math.isnan(r.rel_l2):
            groups.setdefault(r.label, []).append(r)

    if not groups:
        print("[report] No rel-L2 data found — skipping accuracy plot.")
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, rlist) in enumerate(sorted(groups.items())):
        rlist = sorted(rlist, key=lambda r: r.epochs)
        xs = [r.epochs   for r in rlist]
        ys = [r.rel_l2   for r in rlist]
        col = _PALETTE[i % len(_PALETTE)]
        ax.loglog(xs, ys, "o-", color=col, label=label, lw=1.8,
                  markersize=6, markeredgewidth=0.7, markeredgecolor="white")

    ax.set_xlabel("Epoch budget", fontsize=12)
    ax.set_ylabel("Relative L² error", fontsize=12)
    ax.set_title("Accuracy vs. Epoch Budget", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, framealpha=0.85)
    ax.grid(True, which="both", alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_wall_time_vs_epochs(
    results: List[BenchmarkResult],
    out_dir: str,
    *,
    filename: str = "wall_time_vs_epochs.png",
) -> str:
    """Training wall time vs epoch budget, one line per problem."""
    groups: Dict[str, list] = {}
    for r in results:
        groups.setdefault(r.label, []).append(r)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, rlist) in enumerate(sorted(groups.items())):
        rlist = sorted(rlist, key=lambda r: r.epochs)
        xs = [r.epochs       for r in rlist]
        ys = [r.wall_time_s  for r in rlist]
        col = _PALETTE[i % len(_PALETTE)]
        ax.plot(xs, ys, "o-", color=col, label=label, lw=1.8,
                markersize=6, markeredgewidth=0.7, markeredgecolor="white")

    ax.set_xlabel("Epoch budget", fontsize=12)
    ax.set_ylabel("Training time (s)", fontsize=12)
    ax.set_title("Training Wall Time vs. Epoch Budget", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_ms_per_epoch(
    results: List[BenchmarkResult],
    out_dir: str,
    *,
    filename: str = "ms_per_epoch.png",
) -> str:
    """Bar chart: ms per epoch at the *largest* tested epoch budget."""
    # Keep only the largest epoch run per problem
    best: Dict[str, BenchmarkResult] = {}
    for r in results:
        if r.label not in best or r.epochs > best[r.label].epochs:
            best[r.label] = r

    labels = sorted(best.keys())
    times  = [best[l].ms_per_epoch for l in labels]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
    bars = ax.bar(range(len(labels)), times,
                  color=[_PALETTE[i % len(_PALETTE)] for i in range(len(labels))],
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("ms / epoch", fontsize=11)
    ax.set_title("Training Throughput (lower = faster)", fontsize=12, fontweight="bold")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_loss_grid(
    runner,
    out_dir: str,
    *,
    filename: str = "loss_grid.png",
    max_epochs_only: bool = True,
) -> str:
    """Grid of convergence plots — one subplot per problem.

    Reads loss histories from ``runner._loss_snapshots``.
    """
    snapshots = getattr(runner, "_loss_snapshots", {})
    if not snapshots:
        print("[report] No loss snapshots in runner — skipping loss grid.")
        return ""

    probs = sorted(snapshots.keys())
    n = len(probs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)

    for idx, prob in enumerate(probs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        snap = snapshots[prob]

        epoch_keys = sorted(snap.keys())
        if max_epochs_only:
            epoch_keys = [epoch_keys[-1]]

        for ep_key in epoch_keys:
            hist = snap[ep_key]
            xs   = np.arange(1, len(hist) + 1)
            ax.semilogy(xs, hist, lw=1.2, alpha=0.85, label=f"{ep_key} ep")

        ax.set_title(prob, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", alpha=0.25)
        if not max_epochs_only and len(epoch_keys) > 1:
            ax.legend(fontsize=7)

    # Hide empty subplots
    for idx in range(len(probs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Loss Convergence (max epoch budget per problem)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_summary_bar(
    results: List[BenchmarkResult],
    out_dir: str,
    *,
    filename: str = "accuracy_summary_bar.png",
) -> str:
    """Grouped bar chart: rel-L2 at each epoch budget for each problem."""
    groups: Dict[str, list] = {}
    for r in results:
        if not math.isnan(r.rel_l2):
            groups.setdefault(r.label, []).append(r)
    if not groups:
        return ""

    labels = sorted(groups.keys())
    epoch_budgets = sorted({r.epochs for r in results})
    x = np.arange(len(labels))
    width = 0.8 / len(epoch_budgets)

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    for j, ep in enumerate(epoch_budgets):
        ys = []
        for label in labels:
            hit = [r for r in groups[label] if r.epochs == ep]
            ys.append(hit[0].rel_l2 if hit else float("nan"))
        offset = (j - len(epoch_budgets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, ys, width * 0.9,
                      label=f"{ep} epochs",
                      color=_PALETTE[j % len(_PALETTE)],
                      edgecolor="white", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Relative L² error (log scale)", fontsize=11)
    ax.set_title("Accuracy Summary (lower = better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=len(epoch_budgets))
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# =============================================================================
#  Tabular outputs
# =============================================================================

def save_csv(
    results: List[BenchmarkResult],
    out_dir: str,
    filename: str = "benchmark_results.csv",
) -> str:
    path = os.path.join(out_dir, filename)
    if not results:
        return path
    fieldnames = list(results[0].as_row().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_row())
    print(f"Saved: {path}")
    return path


def save_markdown(
    results: List[BenchmarkResult],
    out_dir: str,
    filename: str = "benchmark_summary.md",
) -> str:
    """One row per problem at the *largest* epoch budget."""
    # Best (largest epochs) per problem
    best: Dict[str, BenchmarkResult] = {}
    for r in results:
        if r.problem not in best or r.epochs > best[r.problem].epochs:
            best[r.problem] = r

    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write("# underPINN Benchmark Results\n\n")
        f.write("| Problem | Epochs | Rel-L2 | Max AE | "
                "Loss (final) | Time (s) | ms/epoch |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for prob in sorted(best.keys()):
            r = best[prob]
            _f = lambda v: f"{v:.3e}" if not math.isnan(v) else "—"
            f.write(
                f"| {r.label} "
                f"| {r.epochs:,} "
                f"| {_f(r.rel_l2)} "
                f"| {_f(r.max_ae)} "
                f"| {_f(r.loss_final)} "
                f"| {r.wall_time_s:.1f} "
                f"| {r.ms_per_epoch:.2f} |\n"
            )
        f.write(f"\n*Generated by underPINN benchmark suite.*\n")
    print(f"Saved: {path}")
    return path


# =============================================================================
#  All-in-one
# =============================================================================

def generate_report(
    results: List[BenchmarkResult],
    runner=None,
    out_dir: str = "outputs/bench",
) -> None:
    """Write all plots + tables to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Generating benchmark report → {out_dir}/")
    print(f"{'='*55}")

    plot_accuracy_vs_epochs(results, out_dir)
    plot_summary_bar(results, out_dir)
    plot_wall_time_vs_epochs(results, out_dir)
    plot_ms_per_epoch(results, out_dir)
    if runner is not None:
        plot_loss_grid(runner, out_dir)

    save_csv(results, out_dir)
    save_markdown(results, out_dir)

    print(f"\nAll outputs in: {out_dir}/\n")
