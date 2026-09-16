"""Print mean+-std summaries from multiseed_heavy_throughput_accuracy.json
and baselines_{burgers,pipe_flow,ramp_ns}_seed*.json, for pasting into the
paper once the background runs finish.
"""
import glob
import json

import numpy as np

RESULTS = "benchmarks/suite/results"


def summarize_heavy():
    path = f"{RESULTS}/multiseed_heavy_throughput_accuracy.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print("(heavy multiseed results not yet available)")
        return
    print("\n=== Heavy-problem multi-seed throughput + accuracy ===")
    for prob, rows in d.items():
        ms = [r["ms_per_epoch"] for r in rows]
        l2 = [r["rel_l2"] for r in rows if "rel_l2" in r]
        print(f"{prob:12s} n={len(rows)}  ms/ep = {np.mean(ms):.3f} +/- {np.std(ms):.3f}"
              + (f"   rel_l2 = {np.mean(l2):.4e} +/- {np.std(l2):.1e}" if l2 else ""))


def summarize_baselines():
    print("\n=== Eager PyTorch vs JAX jit/scan, multi-seed ===")
    for prob in ["burgers", "pipe_flow", "ramp_ns"]:
        files = sorted(glob.glob(f"{RESULTS}/baselines_{prob}_seed*.json"))
        if not files:
            print(f"{prob}: (no results yet)")
            continue
        torch_ms, jit_ms, scan_ss_ms, scan_incl_ms = [], [], [], []
        for f in files:
            d = json.load(open(f))
            v = d["variants"]
            if "torch_eager" in v and "ms_per_epoch" in v["torch_eager"]:
                torch_ms.append(v["torch_eager"]["ms_per_epoch"])
            if "jax_jit" in v:
                jit_ms.append(v["jax_jit"]["ms_per_epoch"])
            if "jax_scan" in v:
                scan_ss_ms.append(v["jax_scan"]["ms_per_epoch_second_call_compiled_only"])
                scan_incl_ms.append(v["jax_scan"]["ms_per_epoch_first_call_incl_compile"])
        print(f"\n{prob}  (n={len(files)} seeds)")
        def fmt(a):
            return f"{np.mean(a):.3f} +/- {np.std(a):.3f}" if a else "N/A"
        print(f"  torch_eager          {fmt(torch_ms)} ms/ep")
        print(f"  jax_jit              {fmt(jit_ms)} ms/ep")
        print(f"  jax_scan steady      {fmt(scan_ss_ms)} ms/ep")
        print(f"  jax_scan incl.compile{fmt(scan_incl_ms)} ms/ep")
        if torch_ms and jit_ms:
            print(f"  speedup jit vs torch: {np.mean(torch_ms)/np.mean(jit_ms):.2f}x")
        if torch_ms and scan_ss_ms:
            print(f"  speedup scan(ss) vs torch: {np.mean(torch_ms)/np.mean(scan_ss_ms):.2f}x")
        if torch_ms and scan_incl_ms:
            print(f"  speedup scan(incl) vs torch: {np.mean(torch_ms)/np.mean(scan_incl_ms):.2f}x")


if __name__ == "__main__":
    summarize_heavy()
    summarize_baselines()
