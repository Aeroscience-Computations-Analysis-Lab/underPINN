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
