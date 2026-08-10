"""Training timing utilities for underPINN solvers."""
from __future__ import annotations


def fmt_train_time(elapsed: float, t_first: float | None, n_ep: int) -> str:
    """Return a concise timing string for the training-complete banner.

    Detects JAX JIT compilation overhead by comparing the first gradient
    step against the average of subsequent steps.  When the first step is
    ≥ 3 s *and* at least 4× slower than the per-epoch average, the output
    separates JIT time from actual training time so users understand why
    the first epoch appears slow.

    Parameters
    ----------
    elapsed : float
        Total wall-clock seconds for this training run.
    t_first : float or None
        Wall-clock seconds for the very first gradient step (or first
        outer scan step).  Pass ``None`` when not measured.
    n_ep : int
        Number of epochs/steps actually trained in this run (excluding
        history restored from a restart checkpoint).

    Returns
    -------
    str
        E.g. ``"12.4s  [JIT≈8s + 4.4ms/ep]"``  or  ``"3.1s  [1.6ms/ep]"``

    Examples
    --------
    >>> fmt_train_time(12.4, 8.2, 1000)
    '12.4s  [JIT≈8s + 4.2ms/ep]'
    >>> fmt_train_time(3.1, None, 2000)
    '3.1s  [1.5ms/ep]'
    """
    if n_ep <= 0 or elapsed <= 0:
        return f"{elapsed:.1f}s"

    # ── JIT detection ──────────────────────────────────────────────────────
    if t_first is not None and n_ep > 1:
        avg_rest = (elapsed - t_first) / (n_ep - 1)
        if avg_rest > 0 and t_first >= 3.0 and t_first > 4 * avg_rest:
            return (
                f"{elapsed:.1f}s  "
                f"[JIT≈{t_first:.0f}s + {avg_rest * 1000:.1f}ms/ep]"
            )

    # ── Plain per-epoch average ────────────────────────────────────────────
    ms_ep = elapsed / n_ep * 1000
    return f"{elapsed:.1f}s  [{ms_ep:.1f}ms/ep]"
