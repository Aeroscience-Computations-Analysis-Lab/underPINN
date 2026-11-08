"""Frequency content of an error or residual signal on a uniform 1-D grid.

    amplitude_spectrum     |s^(omega)| with a Hann window. Applied to the error e = u_theta - u* at
                           each checkpoint it shows spectral bias: which frequencies of the error
                           are removed first and which linger. Squared, on the residual, it is the
                           residual power spectrum.
    energy_fraction_below  share of the spectral energy below omega_c. For a residual this says how
                           much of it sits where the operator passes it into the error: for the
                           damped oscillator, residual near or below resonance is amplified and
                           high-frequency residual is suppressed, so equal losses can give very
                           different errors.

Frequencies are angular, omega = 2 pi f, in radians per unit of the grid variable.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _window(n: int, window: Optional[str]) -> np.ndarray:
    if window in (None, "none", "boxcar"):
        return np.ones(n)
    if window == "hann":
        return np.hanning(n)
    raise ValueError(f"unknown window {window!r} (use 'hann' or None)")


def grid_step(x) -> float:
    """Spacing of a uniform 1-D grid (raises if the grid is not uniform)."""
    x = np.asarray(x, np.float64).reshape(-1)
    d = np.diff(x)
    if d.size == 0 or not np.allclose(d, d[0], rtol=1e-6, atol=0.0):
        raise ValueError("needs a uniform, increasing 1-D grid")
    return float(d[0])


def amplitude_spectrum(signal, dt: float, window: Optional[str] = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """(omega, |s^(omega)|), one-sided, along axis 0 of ``signal`` (N,) or (N, k).

    Normalised so that A sin(omega_0 t) with omega_0 on a frequency bin peaks at about A
    (window coherent gain removed); the DC and Nyquist bins are not doubled."""
    s = np.asarray(signal, np.float64)
    n = s.shape[0]
    w = _window(n, window).reshape((n,) + (1,) * (s.ndim - 1))
    amp = np.abs(np.fft.rfft(w * s, axis=0)) * 2.0 / float(np.sum(w))
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    return 2.0 * np.pi * np.fft.rfftfreq(n, dt), amp


def energy_fraction_below(signal, dt: float, omega_c: float, window: Optional[str] = "hann") -> float:
    """sum_{omega < omega_c} |s^|^2 / sum_omega |s^|^2 (one-sided spectrum)."""
    om, amp = amplitude_spectrum(signal, dt, window)
    p = np.sum(np.reshape(amp, (amp.shape[0], -1)) ** 2, axis=1)
    tot = float(p.sum())
    return float(p[om < omega_c].sum() / tot) if tot > 0 else float("nan")


def radial_spectrum(field, spacings, n_shells: Optional[int] = None, window: Optional[str] = "hann"):
    """Radially averaged amplitude spectrum of a field on a uniform N-D grid.

    ``field``: array of shape (n_1, ..., n_d); ``spacings``: grid step per axis. A separable
    window is applied per axis. Returns (|k| shell centres in rad/unit, mean |F| per shell).
    Works for any d (a 1-D input reproduces ``amplitude_spectrum`` up to binning).
    """
    f = np.asarray(field, np.float64)
    sp = np.broadcast_to(np.asarray(spacings, np.float64), (f.ndim,))
    w = np.ones_like(f)
    for ax, n in enumerate(f.shape):
        shape = [1] * f.ndim
        shape[ax] = n
        w = w * _window(n, window).reshape(shape)
    F = np.abs(np.fft.fftn((f - f.mean()) * w)) / f.size
    ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(f.shape, sp)], indexing="ij")
    kmag = np.sqrt(sum(k ** 2 for k in ks))
    kmax = min(np.pi / d for d in sp)                      # Nyquist of the coarsest axis
    n_shells = n_shells or max(4, min(f.shape) // 2)
    edges = np.linspace(0.0, kmax, n_shells + 1)
    idx = np.clip(np.searchsorted(edges, kmag.ravel(), side="right") - 1, 0, n_shells - 1)
    keep = kmag.ravel() <= kmax
    cnt = np.bincount(idx[keep], minlength=n_shells)
    amp = np.bincount(idx[keep], weights=F.ravel()[keep], minlength=n_shells) / np.maximum(cnt, 1)
    return 0.5 * (edges[:-1] + edges[1:]), amp


def spectral_error(u_pred, u_ref, spacings, n_shells: Optional[int] = None, tol: float = 0.1,
                   significant: float = 1e-2, window: Optional[str] = "hann"):
    """Per-wavenumber relative error |E(k)| / |U(k)| on an N-D grid (spectral bias, made a number).

    Returns shell centres, the error and solution spectra, their ratio, and
    ``resolved_k``: the largest wavenumber below which every *significant* shell (amplitude
    ≥ ``significant`` × the peak) has ratio < ``tol`` -- the bandwidth the network actually
    reproduces. A Hann window suppresses leakage from non-periodic grids, which would
    otherwise smear a dominant mode into shells the solution does not really occupy.
    """
    k, E = radial_spectrum(np.asarray(u_pred) - np.asarray(u_ref), spacings, n_shells, window=window)
    _, U = radial_spectrum(u_ref, spacings, n_shells, window=window)
    ratio = E / np.maximum(U, 1e-300)
    sig = U >= significant * U.max()
    resolved = 0.0
    for kk, r, s in zip(k, ratio, sig):
        if s and r >= tol:
            break
        resolved = kk
    return {"k": k, "error_amp": E, "solution_amp": U, "ratio": np.where(sig, ratio, np.nan),
            "resolved_k": float(resolved), "tol": tol}
