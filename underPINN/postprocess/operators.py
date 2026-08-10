"""Shared plotting helpers for the neural-operator (PINO/DeepONet) examples.

Every operator example needs the same two plots — a training-loss curve with
2-3 named components, and a prediction-vs-exact-vs-error panel (1-D line plot
or 2-D field plot) — so they're written once here instead of once per example.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from underPINN.postprocess.plotting import CMAP, field, cbar, save_fig


def plot_operator_loss(hist: dict, out_path: str, title: str = "Training loss"):
    """``hist``: e.g. ``{"loss": [...], "data": [...], "pde": [...]}`` —
    any number of named histories, all plotted on one semilog-y axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in hist.items():
        if values:
            ax.semilogy(values, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, out_path)


def plot_prediction_1d(x: np.ndarray, u_pred: np.ndarray, u_exact: np.ndarray,
                       out_path: str, title: str = "Prediction vs. exact"):
    """Line-plot comparison for a 1-D field: prediction, exact, |error|."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.plot(x, u_exact, label="exact", lw=2)
    ax0.plot(x, u_pred, "--", label="predicted", lw=2)
    ax0.set_xlabel("x")
    ax0.set_ylabel("u")
    ax0.set_title(title)
    ax0.legend()

    err = np.abs(u_pred - u_exact)
    ax1.plot(x, err, color="crimson")
    ax1.set_xlabel("x")
    ax1.set_ylabel("|error|")
    ax1.set_title("Absolute error")
    return save_fig(fig, out_path)


def plot_prediction_2d(X: np.ndarray, Y: np.ndarray, u_pred: np.ndarray,
                       u_exact: np.ndarray, out_path: str,
                       title: str = "Prediction vs. exact"):
    """Three-panel comparison for a 2-D field: prediction, exact, |error|."""
    err = np.abs(u_pred - u_exact)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, C, name in zip(axes, (u_pred, u_exact, err),
                           ("predicted", "exact", "|error|")):
        cmap = CMAP if name != "|error|" else "inferno"
        cf = field(ax, X, Y, C, cmap=cmap)
        cbar(fig, cf, ax, name)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    return save_fig(fig, out_path)
