"""Regenerate ms_per_epoch.pdf and accuracy_summary_bar.pdf from real
multi-seed data (mean +/- std error bars), replacing the stale, single-run,
early-draft PDFs in underPINN_paper/pdf_images/.

Combines:
  benchmarks/suite/results/multiseed_cheap_throughput_accuracy.json
      (Burgers, Wave, Helmholtz, Steady Heat, ODE Harmonic; 5,000 epochs,
      3 seeds each)
  benchmarks/suite/results/multiseed_heavy_throughput_accuracy.json
      (Ramp 20k / Toro3 60k / Pipe Flow 60k / Ramp NS 30k epochs, 3 seeds
      each)

into one 9-problem, mean+/-std dataset, then plots both bar charts using
the same visual style as underPINN.benchmark_utils.report (colour-blind
palette, log-scale accuracy axis, white bar edges, light grid) with error
bars added -- a feature the original plot_ms_per_epoch/plot_summary_bar
never had, since they were written for single-run BenchmarkResult objects.

Text rendering: the paper sets Times via LaTeX's `times` package, but this
machine has no external `latex`/`pdflatex` binary (only `tectonic`, which
matplotlib's `text.usetex`/pgf backends cannot drive -- they shell out to a
specific `latex`-family executable and expect DVI/its own PDF pipeline, not
tectonic's). True `usetex=True` rendering is therefore not available here.
Instead we configure matplotlib's own (no-external-call) text/math
renderer to match as closely as this machine's fonts allow: "Nimbus Roman"
for regular text (a metric-compatible Times substitute, bundled with
Ghostscript and already present in this environment's font cache) and the
"stix" mathtext fontset for in-figure LaTeX math ($...$ strings below),
since STIX was designed to pair with Times rather than Computer Modern.

Usage:
    python benchmarks/suite/regenerate_summary_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

from underPINN.benchmark_utils.report import _PALETTE

CHEAP_JSON = "benchmarks/suite/results/multiseed_cheap_throughput_accuracy.json"
HEAVY_JSON = "benchmarks/suite/results/multiseed_heavy_throughput_accuracy.json"
OUT_DIR = "underPINN_paper/pdf_images"

# Canonical order, matching Table~\ref{tab:benchmark-suite}.
ORDER = ["burgers", "wave", "helmholtz", "heat_steady", "ode_harmonic",
         "ramp", "toro3", "pipe_flow", "ramp_ns"]
DISPLAY = {
    "burgers": "1-D\nBurgers",
    "wave": "1-D\nWave",
    "helmholtz": "2-D\nHelmholtz",
    "heat_steady": "2-D\nSteady Heat",
    "ode_harmonic": "ODE\nHarmonic",
    "ramp": "2-D\nRamp",
    "toro3": "1-D\nToro Test 3",
    "pipe_flow": "3-D\nPipe Flow",
    "ramp_ns": "2-D\nRamp NS",
}
EPOCHS = {
    "burgers": 5000, "wave": 5000, "helmholtz": 5000, "heat_steady": 5000,
    "ode_harmonic": 5000, "ramp": 20000, "toro3": 60000, "pipe_flow": 60000,
    "ramp_ns": 30000,
}


def load_all() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for path in (CHEAP_JSON, HEAVY_JSON):
        with open(path) as f:
            d = json.load(f)
        for prob, rows in d.items():
            data[prob] = rows
    missing = [p for p in ORDER if p not in data]
    if missing:
        raise RuntimeError(f"Missing multi-seed data for: {missing}")
    return data


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def plot_ms_per_epoch(data: dict[str, list[dict]]) -> str:
    labels = [DISPLAY[p] for p in ORDER]
    means = [mean_std(data[p], "ms_per_epoch")[0] for p in ORDER]
    stds = [mean_std(data[p], "ms_per_epoch")[1] for p in ORDER]

    fig, ax = plt.subplots(figsize=(max(6, len(ORDER) * 1.5), 4.5))
    bars = ax.bar(range(len(ORDER)), means, yerr=stds, capsize=6,
                  color=[_PALETTE[i % len(_PALETTE)] for i in range(len(ORDER))],
                  edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.6, "ecolor": "black"})
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel("ms / epoch", fontsize=22)
    ax.set_title("Training Throughput (lower = faster), mean $\\pm$ std over 3 seeds",
                fontsize=22, fontweight="bold")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(means) * 0.02, f"{m:.2f}",
                ha="center", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = f"{OUT_DIR}/ms_per_epoch.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_accuracy_summary(data: dict[str, list[dict]]) -> str:
    labels = [DISPLAY[p] for p in ORDER]
    means = [mean_std(data[p], "rel_l2")[0] for p in ORDER]
    stds = [mean_std(data[p], "rel_l2")[1] for p in ORDER]

    fig, ax = plt.subplots(figsize=(max(9, len(ORDER) * 1.7), 6))
    bars = ax.bar(range(len(ORDER)), means, yerr=stds, capsize=6,
                  color=[_PALETTE[i % len(_PALETTE)] for i in range(len(ORDER))],
                  edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.6, "ecolor": "black"})
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel("Relative $L^2$ error (log scale)", fontsize=22)
    ax.set_title("Accuracy at Largest Budget Tested (lower = better), "
                "mean $\\pm$ std over 3 seeds", fontsize=22, fontweight="bold")
    ymin, ymax = min(m - s for m, s in zip(means, stds)), max(m + s for m, s in zip(means, stds))
    ax.set_ylim(ymin * 0.3, ymax * 4.0)
    for i, p in enumerate(ORDER):
        ax.text(i, (means[i] + stds[i]) * 1.5, f"{EPOCHS[p]//1000}k ep" if EPOCHS[p] >= 1000
                else f"{EPOCHS[p]} ep", ha="center", fontsize=13, color="dimgray")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()

    path = f"{OUT_DIR}/accuracy_summary_bar.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main() -> None:
    data = load_all()
    print("Per-problem mean +/- std:")
    for p in ORDER:
        ms_m, ms_s = mean_std(data[p], "ms_per_epoch")
        l2_m, l2_s = mean_std(data[p], "rel_l2")
        print(f"  {p:12s} n={len(data[p])}  ms/ep={ms_m:7.3f}+/-{ms_s:5.3f}  "
              f"rel_l2={l2_m:.4e}+/-{l2_s:.1e}  (epochs={EPOCHS[p]})")
    plot_ms_per_epoch(data)
    plot_accuracy_summary(data)


if __name__ == "__main__":
    main()
