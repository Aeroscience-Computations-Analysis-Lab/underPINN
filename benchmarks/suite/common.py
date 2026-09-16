"""Shared helpers for the reviewer-response benchmark suite.

Everything here exists to make the numbers in ``benchmarks/suite`` auditable:
device provenance is recorded in every result file, compile time is always
reported separately from steady-state time, and a run that silently fell back
to CPU can never be mistaken for a GPU measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


# ── device provenance ─────────────────────────────────────────────────────────

def jax_device_info(require_gpu: bool) -> dict:
    """Return JAX backend info, raising unless *require_gpu* is satisfied."""
    import jax
    dev = jax.devices()[0]
    info = {"framework": "jax", "platform": dev.platform,
            "device_name": getattr(dev, "device_kind", str(dev))}
    if require_gpu and dev.platform != "gpu":
        raise RuntimeError(
            f"JAX default device platform is '{dev.platform}', not 'gpu'. "
            "Timing numbers from a CPU fallback would look like a GPU result "
            "without being one. Install a CUDA-enabled jaxlib, or pass "
            "--allow-cpu to run this as a correctness smoke test only.")
    return info


def torch_device_info(require_gpu: bool) -> tuple[Any, dict]:
    """Return ``(device, info)`` for PyTorch, raising unless GPU is satisfied."""
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        platform_str = "cuda"
    elif require_gpu:
        raise RuntimeError(
            "CUDA GPU not available. PyTorch and JAX must run on the same "
            "device for an identical-hardware comparison. Pass --allow-cpu to "
            "run this as a correctness smoke test only.")
    else:
        dev = torch.device("cpu")
        name = platform.processor() or "cpu"
        platform_str = "cpu"
    return dev, {"framework": "torch", "torch_version": torch.__version__,
                 "platform": platform_str, "device_name": name}


def torch_sync(device) -> None:
    """Block until all queued work on *device* has finished."""
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()


def ensure_host_cc() -> None:
    """Repair a broken ``CC``/``CXX`` before any ``torch.compile`` call.

    On HPC systems that load a compiler toolchain via an environment-modules
    system (e.g. NVIDIA HPC SDK's ``nvhpc`` module), ``CC``/``CXX`` are often
    exported globally to that toolchain's ``nvc``/``nvc++``. If the module was
    unloaded, the filesystem it points into isn't mounted here, or the module
    load was otherwise stale, Triton (which torch.compile/Inductor uses to
    generate and link host-side wrapper code) fails opaquely deep inside
    Inductor -- e.g. ``FileNotFoundError: .../nvc`` -- with no indication the
    root cause is an environment variable rather than a torch/Triton bug. A
    JAX-only run never touches this path, so this only matters for the
    PyTorch baselines.

    We only intervene when the *current* CC/CXX point at a file that does not
    exist -- a working, deliberately-chosen toolchain (nvc++ included, if it
    is actually present) is left alone.
    """
    import os
    import shutil

    for var, candidates in (("CC", ("cc", "gcc", "clang")),
                            ("CXX", ("c++", "g++", "clang++"))):
        current = os.environ.get(var)
        if current and os.path.isfile(current):
            continue          # already valid (or a deliberately-set, real path)
        for name in candidates:
            found = shutil.which(name)
            if found:
                if current:
                    print(f"NOTE: ${var}={current!r} does not exist on this "
                          f"host; falling back to {found} so torch.compile's "
                          f"Triton backend has a working host compiler.")
                os.environ[var] = found
                break


# ── timing ────────────────────────────────────────────────────────────────────

def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run *fn*, returning ``(result, wall_seconds)``."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def measure_compiled(run: Callable[[], Any], label: str = "") -> dict:
    """Time *run* twice: first call includes trace+compile, second is steady state.

    Reporting both is the point -- a headline number that hides one-time
    compilation inside it is exactly the kind of measurement the reviewers
    (correctly) pushed back on.
    """
    _, t_first = timed(run)
    _, t_second = timed(run)
    return {
        "wall_s_first_call_incl_compile": t_first,
        "wall_s_second_call_compiled_only": t_second,
        "compile_overhead_s_estimate": max(t_first - t_second, 0.0),
        "label": label,
    }


# ── result IO ─────────────────────────────────────────────────────────────────

def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def save_result(name: str, payload: dict) -> str:
    """Write *payload* to ``results/<name>.json`` with provenance attached."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("git_commit", git_commit())
    payload.setdefault("python", platform.python_version())
    payload.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResult saved -> {path}")
    return path


def load_results(prefix: str = "") -> dict[str, dict]:
    """Load every ``results/*.json`` whose stem starts with *prefix*."""
    if not os.path.isdir(RESULTS_DIR):
        return {}
    out = {}
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith(".json") and fn.startswith(prefix):
            with open(os.path.join(RESULTS_DIR, fn)) as fh:
                out[fn[:-5]] = json.load(fh)
    return out


def save_raw_arrays(name: str, rows: dict) -> str:
    """Write every ``_``-prefixed array field of a ``{arm: {key: value}}``
    results dict to ``results/<name>_raw.npz``, so a plotting function that
    needs them (e.g. ``plot_migration``/``plot_solutions`` in the
    ``ablate_qr_deim*.py`` scripts) can be rerun later from the saved file
    instead of retraining -- ``save_result`` alone strips these (they can be
    tens of MB across a handful of arms) before the human-readable JSON.

    Flattened as ``"<arm>::<key>"`` -> array, one npz. Companion to
    :func:`load_raw_arrays`.
    """
    import numpy as np
    os.makedirs(RESULTS_DIR, exist_ok=True)
    flat = {}
    for arm, r in rows.items():
        if "error" in r:
            continue
        for k, v in r.items():
            if k.startswith("_"):
                flat[f"{arm}::{k}"] = np.asarray(v)
    path = os.path.join(RESULTS_DIR, f"{name}_raw.npz")
    np.savez_compressed(path, **flat)
    print(f"Raw plotting arrays saved -> {path} "
         f"({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def load_raw_arrays(name: str) -> dict:
    """Inverse of :func:`save_raw_arrays`: reconstruct the
    ``{arm: {key: array}}`` nesting from ``results/<name>_raw.npz``."""
    import numpy as np
    path = os.path.join(RESULTS_DIR, f"{name}_raw.npz")
    rows: dict = {}
    with np.load(path) as data:
        for flat_key in data.files:
            arm, key = flat_key.split("::", 1)
            rows.setdefault(arm, {})[key] = data[flat_key]
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=5000,
                   help="training epochs per timed run (default: 5000)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit a CPU run; results are marked as such and must "
                        "NOT be reported as GPU numbers")
    p.add_argument("--seed", type=int, default=0)
    return p


def warn_if_cpu(info: dict) -> None:
    if info.get("platform") not in ("gpu", "cuda"):
        print("\n" + "!" * 72)
        print("!! CPU RUN -- correctness smoke test only.")
        print("!! These timings are NOT valid for the paper's hardware claims.")
        print("!" * 72 + "\n")
