"""Print a consolidated table of every reviewer-response benchmark result.

Reads whatever ``results/*.json`` files exist and renders one section per
concern raised in review. Missing results are reported as missing rather than
silently skipped, so a partial run is never mistaken for a complete one.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_results  # noqa: E402

RULE = "=" * 78


def _device_line(payload: dict) -> str:
    dev = payload.get("device") or payload.get("devices", {})
    if isinstance(dev, dict) and "platform" in dev:
        return f"{dev.get('platform')} ({dev.get('device_name')})"
    if isinstance(dev, dict):
        return "  ".join(
            f"{k}={v.get('platform')}({v.get('device_name')})"
            for k, v in dev.items() if isinstance(v, dict))
    return "unknown"


def section_parity(res: dict) -> None:
    p = res.get("parity_pipeflow_dispatch")
    print(f"\n{RULE}\n 1. Throughput discrepancy (Table 1 vs Figure 2)\n{RULE}")
    if not p:
        print("  MISSING -- run parity/dispatch_parity.py")
        return
    print(f"  device: {_device_line(p)}   epochs: {p['epochs']}")
    variants = p["variants"]
    base = variants["A_evaluator_style"]["ms_per_epoch"]
    print(f"\n  {'variant':22s} {'ms/epoch':>10s} {'vs A':>8s}   what changed")
    print("  " + "-" * 74)
    for k, v in variants.items():
        print(f"  {k:22s} {v['ms_per_epoch']:10.3f} "
              f"{base / v['ms_per_epoch']:7.2f}x   {v['description']}")
    best = min(variants.values(), key=lambda v: v["ms_per_epoch"])
    print(f"\n  => host-side scaffolding alone accounts for a "
          f"{base / best['ms_per_epoch']:.1f}x spread on identical math.")


def section_baselines(res: dict) -> None:
    b = res.get("baselines_burgers")
    print(f"\n{RULE}\n 2. Strong baselines (not just eager PyTorch)\n{RULE}")
    if not b:
        print("  MISSING -- run baselines/burgers_baselines.py")
        return
    print(f"  device: {_device_line(b)}   epochs: {b['epochs']}")
    variants = b["variants"]

    def ms(v):
        if "error" in v or v.get("unavailable"):
            return None
        return v.get("ms_per_epoch",
                     v.get("ms_per_epoch_second_call_compiled_only"))

    base = ms(variants.get("torch_eager", {}))
    print(f"\n  {'variant':22s} {'ms/epoch':>10s} {'vs eager':>9s}")
    print("  " + "-" * 74)
    for k in ["torch_eager", "torch_script", "torch_func_eager",
              "torch_func_compile", "jax_jit", "jax_scan"]:
        v = variants.get(k)
        if not v:
            continue
        m = ms(v)
        if m is None:
            print(f"  {k:22s} {'n/a':>10s}")
            continue
        rel = f"{base / m:.2f}x" if base else "-"
        print(f"  {k:22s} {m:10.3f} {rel:>9s}")
    unavail = variants.get("torch_compile_autograd")
    if unavail and unavail.get("unavailable"):
        print(f"\n  note: torch.compile on the classic PINN formulation is "
              f"unavailable --\n        {unavail['reason']}")


def section_ablations(res: dict) -> None:
    print(f"\n{RULE}\n 3. Feature ablations (are the advertised features "
          f"earning their place?)\n{RULE}")
    f = res.get("ablation_features_burgers")
    if not f:
        print("  MISSING -- run ablations/ablate_features.py")
    else:
        print(f"  device: {_device_line(f)}   epochs: {f['epochs']}   "
              f"metric: {f['metric']}")
        arms = {k: v for k, v in f["arms"].items() if "error" not in v}
        base = arms.get("mlp", {}).get("rel_l2")
        print(f"\n  {'arm':14s} {'params':>8s} {'rel L2':>11s} {'vs mlp':>8s}")
        print("  " + "-" * 74)
        for k, v in arms.items():
            rel = f"{base / v['rel_l2']:.2f}x" if base else "-"
            print(f"  {k:14s} {v['n_params']:8d} {v['rel_l2']:11.4e} {rel:>8s}")

    a = res.get("ablation_artificial_viscosity_toro3")
    print()
    if not a:
        print("  MISSING -- run ablations/ablate_artificial_viscosity.py")
        return
    arms = {k: v for k, v in a["arms"].items() if "error" not in v}
    base = arms.get("fixed", {}).get("rel_l2")
    print(f"  artificial viscosity ({a['metric']}), epochs: {a['epochs']}")
    print(f"\n  {'arm':12s} {'rel L2':>11s} {'vs fixed':>9s} {'eps used':>13s}")
    print("  " + "-" * 74)
    for k, v in arms.items():
        rel = f"{base / v['rel_l2']:.2f}x" if base else "-"
        print(f"  {k:12s} {v['rel_l2']:11.4e} {rel:>9s} {v['eps_final']:13.6g}")


def main() -> int:
    res = load_results()
    if not res:
        print("No results found. Run: bash benchmarks/rebuttal/run_all.sh")
        return 1
    section_parity(res)
    section_baselines(res)
    section_ablations(res)
    print(f"\n{RULE}")
    cpu = [k for k, v in res.items()
           if "cpu" in str(v.get("device", v.get("devices", ""))).lower()]
    if cpu:
        print(" WARNING: these results contain CPU runs and are NOT valid for")
        print(f" the paper's hardware claims: {', '.join(cpu)}")
        print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
