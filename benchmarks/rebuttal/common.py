"""Shared helpers for the reviewer-response benchmark suite.

Everything here exists to make the numbers in ``benchmarks/rebuttal`` auditable:
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
