"""underPINN command-line interface.

Usage
-----
::

    # Run a single experiment
    python -m underPINN run examples/burgers/config.yaml

    # Hyperparameter sweep (Cartesian product of sweep values)
    python -m underPINN sweep examples/burgers/burgers_nu_sweep.yaml

    # List registered problem runners
    python -m underPINN list

    # Print the resolved config without running
    python -m underPINN show examples/burgers/config.yaml

    # Accuracy-vs-epoch benchmark suite
    python -m underPINN bench
    python -m underPINN bench --problems burgers wave --epochs 500 2000 5000
    python -m underPINN bench --all --output outputs/bench_full

    # Continue a completed run from its last checkpoint
    #   Step 1 — raise epochs in the YAML (e.g. 5000 → 10000)
    #   Step 2 — unlock the snapshot so the new config hash is accepted
    python -m underPINN resume examples/wave/config.yaml
    #   Step 3 — run normally; training picks up from the saved epoch
    python -m underPINN run    examples/wave/config.yaml

    # Inspect the current restart snapshot state
    python -m underPINN status examples/wave/config.yaml
"""

import os
import sys

# ── JAX memory allocation ─────────────────────────────────────────────────────
# By default JAX pre-allocates ~90 % of GPU VRAM on the first import, which
# makes `nvidia-smi` show 70–75 GB even for a 3-layer MLP.
# Setting PREALLOCATE=false makes JAX grow memory on demand (like PyTorch).
# Users can override by setting XLA_PYTHON_CLIENT_PREALLOCATE=true in their
# shell, or cap usage with XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 (= 40 % of VRAM).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse


def _cmd_run(args):
    from underPINN.config import load_config
    from underPINN.runner import get_runner

    cfg    = load_config(args.config)
    runner = get_runner(cfg.problem)
    runner(cfg)


def _cmd_sweep(args):
    import copy
    from underPINN.config import generate_sweep_configs, cfg_get
    from underPINN.runner import get_runner

    configs = generate_sweep_configs(args.config)
    n = len(configs)
    print(f"Sweep: {n} configuration(s) to run.")

    for i, cfg in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"  Run {i + 1} / {n}")
        print(f"{'=' * 60}")

        # Give each run a unique sub-directory so outputs don't collide
        base_dir = cfg_get(cfg, "output", "dir", default="outputs/sweep")
        # Mutate the output.dir in the namespace (SimpleNamespace is mutable)
        try:
            cfg.output.dir = os.path.join(base_dir, f"run_{i:03d}")
        except AttributeError:
            import types
            cfg.output = types.SimpleNamespace(
                dir=os.path.join(base_dir, f"run_{i:03d}"))

        runner = get_runner(cfg.problem)
        runner(cfg)

    print(f"\nSweep complete — {n} runs finished.")


def _cmd_list(_args):
    from underPINN.runner import list_problems
    print("Registered problem runners:")
    for p in list_problems():
        print(f"  {p}")


def _cmd_show(args):
    import yaml
    from underPINN.config.loader import load_config, _ns_to_dict
    cfg = load_config(args.config)
    print(yaml.dump(_ns_to_dict(cfg), default_flow_style=False, sort_keys=False))


def _cmd_bench(args):
    """Delegate to benchmarks/run_benchmarks.py main() with the parsed args."""
    from underPINN.benchmark_utils import (
        BenchmarkRunner, generate_report)
    from underPINN.benchmark_utils.benchmark_suite import BenchmarkRunner as BR

    if args.list_problems:
        from underPINN.benchmark_utils.evaluators import (
            EVALUATOR_REGISTRY, SLOW_PROBLEMS)
        print("Registered evaluators:")
        for k, cls in sorted(EVALUATOR_REGISTRY.items()):
            speed = " [slow]" if k in SLOW_PROBLEMS else ""
            print(f"  {k:<20s}  {cls.__name__}{speed}")
        return

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    if args.from_json:
        print(f"Loading results from {args.from_json} …")
        results = BR.load_json(args.from_json)
        generate_report(results, runner=None, out_dir=out_dir)
        return

    epoch_budgets = args.epochs or [500, 1000, 2000, 5000]
    runner = BenchmarkRunner(
        problems=args.problems or None,
        epoch_budgets=epoch_budgets,
        seed=args.seed,
        fast_only=not args.all,
        verbose=not args.quiet,
    )

    print("=" * 60)
    print("  underPINN Benchmark Suite")
    print(f"  Problems : {runner._problems}")
    print(f"  Epochs   : {epoch_budgets}")
    print(f"  Output   : {out_dir}/")
    print("=" * 60)

    results = runner.run(out_dir=out_dir)
    runner.save_json(os.path.join(out_dir, "results.json"))
    runner.save_loss_npz(os.path.join(out_dir, "loss_hists.npz"))
    generate_report(results, runner=runner, out_dir=out_dir)


def _cmd_resume(args):
    """Unlock a completed restart snapshot so the next ``run`` continues from it.

    Workflow
    --------
    A completed run marks its snapshot as ``done`` so that a plain re-run always
    starts fresh.  When you intentionally want to extend training (e.g. ran 5000
    epochs and now want 10 000), use this command::

        # 1. Edit the YAML — raise training.epochs to the new target
        # 2. Unlock the snapshot (updates the config hash + clears done flag)
        python -m underPINN resume examples/wave/config.yaml
        # 3. Run normally — picks up from the last saved epoch
        python -m underPINN run    examples/wave/config.yaml
    """
    import hashlib, json, pathlib, types
    from underPINN.config.loader import load_config, cfg_get

    cfg = load_config(args.config)

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out = cfg_get(cfg, "output", default=None)
        out_dir = cfg_get(out, "dir", default=None) if out else None
        if not out_dir:
            print(
                "\nError: could not find output.dir in the config.\n"
                "Pass it explicitly with:  --out-dir outputs/wave\n"
            )
            sys.exit(1)

    restart_dir = pathlib.Path(out_dir) / "restart"
    meta_path   = restart_dir / "meta.json"
    params_path = restart_dir / "params.msgpack"

    # ── Validate snapshot ─────────────────────────────────────────────────────
    if not meta_path.exists():
        print(f"\nNo restart snapshot found in  {restart_dir}/")
        print(
            "Run the experiment at least once with  save_restart_every > 0  "
            "in training: to create one.\n"
        )
        sys.exit(1)

    if not params_path.exists():
        print(f"\nmeta.json found but params.msgpack is missing in  {restart_dir}/")
        print(
            "The snapshot may be corrupt.  Delete the restart/ folder and "
            "re-run from scratch.\n"
        )
        sys.exit(1)

    # ── Read current metadata ─────────────────────────────────────────────────
    meta        = json.loads(meta_path.read_text())
    saved_epoch = meta.get("epoch", -1)
    was_done    = meta.get("done", False)
    old_hash    = meta.get("cfg_hash", "")

    # ── Recompute hash for the (possibly modified) config ─────────────────────
    try:
        from underPINN.config.loader import _ns_to_dict
        d = _ns_to_dict(cfg)
    except Exception:
        d = vars(cfg) if isinstance(cfg, types.SimpleNamespace) else {}

    new_hash = hashlib.md5(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()

    # ── Update snapshot ───────────────────────────────────────────────────────
    meta["done"]     = False
    meta["cfg_hash"] = new_hash
    meta_path.write_text(json.dumps(meta, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    try:
        new_epochs = cfg.training.epochs
    except AttributeError:
        new_epochs = "?"

    hash_changed = (old_hash != new_hash)
    print()
    print("  ┌─ Restart snapshot unlocked ─────────────────────────────┐")
    print(f"  │  Snapshot dir     : {restart_dir}/")
    print(f"  │  Last saved epoch : {saved_epoch}")
    print(f"  │  Was marked done  : {was_done}")
    if hash_changed:
        print(f"  │  Config hash      : {old_hash[:8]}… → {new_hash[:8]}…  (config changed)")
    else:
        print(f"  │  Config hash      : {new_hash[:8]}…  (unchanged)")
    print(f"  │  Will resume from : epoch {saved_epoch + 1}")
    print(f"  │  New epoch target : {new_epochs}")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"  Next step:  python -m underPINN run {args.config}")
    print()


def _cmd_status(args):
    """Show the current state of the restart snapshot for a config."""
    import json, pathlib
    from underPINN.config.loader import load_config, cfg_get

    cfg = load_config(args.config)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out = cfg_get(cfg, "output", default=None)
        out_dir = cfg_get(out, "dir", default=None) if out else None
        if not out_dir:
            print("\nError: no output.dir in config.  Pass --out-dir.\n")
            sys.exit(1)

    restart_dir = pathlib.Path(out_dir) / "restart"
    meta_path   = restart_dir / "meta.json"

    if not meta_path.exists():
        print(f"\n  No snapshot in  {restart_dir}/  — training has never saved a checkpoint.\n")
        return

    meta        = json.loads(meta_path.read_text())
    saved_epoch = meta.get("epoch", -1)
    done        = meta.get("done", False)
    cfg_hash    = meta.get("cfg_hash", "n/a")

    files = {
        "params.msgpack":   (restart_dir / "params.msgpack").exists(),
        "opt_state.msgpack":(restart_dir / "opt_state.msgpack").exists(),
        "hists.npz":        (restart_dir / "hists.npz").exists(),
    }

    try:
        new_epochs = cfg.training.epochs
    except AttributeError:
        new_epochs = "?"

    print()
    print(f"  ┌─ Restart snapshot status ───────────────────────────────┐")
    print(f"  │  Directory  : {restart_dir}/")
    print(f"  │  Saved at   : epoch {saved_epoch}  (resume would start at {saved_epoch + 1})")
    print(f"  │  Done flag  : {done}  {'← completed run; use resume to extend' if done else '← interrupted run; re-run to continue'}")
    print(f"  │  Config hash: {cfg_hash[:8]}…")
    print(f"  │  Epoch target (current config): {new_epochs}")
    print(f"  │  Files      : {', '.join(k for k,v in files.items() if v)}")
    missing = [k for k,v in files.items() if not v]
    if missing:
        print(f"  │  Missing    : {', '.join(missing)}")
    print(f"  └─────────────────────────────────────────────────────────┘")

    if done:
        print()
        print("  To extend training:")
        print(f"    1. Edit {args.config}  →  increase training.epochs")
        print(f"    2. python -m underPINN resume {args.config}")
        print(f"    3. python -m underPINN run    {args.config}")
    else:
        print()
        print(f"  To continue interrupted training (same config):")
        print(f"    python -m underPINN run {args.config}")
    print()


def _cmd_version(_args):
    from underPINN._version import __version__, version_tag
    print(f"underPINN {version_tag}  (build {__version__})")


def main():
    from underPINN._version import version_tag

    parser = argparse.ArgumentParser(
        prog="python -m underPINN",
        description=f"underPINN {version_tag} — YAML-driven PINN experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m underPINN version
  python -m underPINN run    examples/burgers/config.yaml
  python -m underPINN sweep  examples/burgers/burgers_nu_sweep.yaml
  python -m underPINN list
  python -m underPINN show   examples/pipe_flow/pipe_flow.yaml
  python -m underPINN status examples/wave/config.yaml
  python -m underPINN resume examples/wave/config.yaml  # then run again
""",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Run a single experiment from a YAML config")
    p.add_argument("config", help="Path to config .yaml file")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("sweep",
                       help="Run a hyperparameter sweep from a sweep YAML")
    p.add_argument("config", help="Path to sweep .yaml file")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("version", help="Print underPINN version and exit")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser(
        "resume",
        help="Unlock a completed snapshot so the next run extends training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Unlock a completed restart snapshot for continued training.\n\n"
            "Workflow:\n"
            "  1. Edit the YAML — raise training.epochs to the new target\n"
            "  2. python -m underPINN resume <config>   ← this command\n"
            "  3. python -m underPINN run    <config>   ← resumes from snapshot"
        ),
    )
    p.add_argument("config", help="Path to the YAML config file")
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Override output.dir from config (useful if dir was changed)",
    )
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser(
        "status",
        help="Show the restart snapshot state for a config",
    )
    p.add_argument("config", help="Path to the YAML config file")
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Override output.dir from config",
    )
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("list", help="List registered problem runners")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="Print resolved config without running")
    p.add_argument("config", help="Path to config .yaml file")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser(
        "bench",
        help="Run accuracy-vs-epoch benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Systematic accuracy vs. epoch budget comparisons.",
        epilog="""
examples:
  python -m underPINN bench
  python -m underPINN bench --problems burgers wave ode_exp --epochs 500 2000 5000
  python -m underPINN bench --all --output outputs/bench_full
  python -m underPINN bench --from-json outputs/bench/results.json
""",
    )
    p.add_argument("--problems", nargs="+", default=None, metavar="PROB",
                   help="Evaluator keys (default: all fast problems).")
    p.add_argument("--epochs", nargs="+", type=int, default=None, metavar="N",
                   help="Epoch budgets (default: 500 1000 2000 5000).")
    p.add_argument("--all", action="store_true",
                   help="Include slow evaluators (e.g. 3-D pipe flow).")
    p.add_argument("--seed", type=int, default=0, metavar="S",
                   help="Base PRNG seed.")
    p.add_argument("--output", "-o", default="outputs/bench", metavar="DIR",
                   help="Output directory (default: outputs/bench).")
    p.add_argument("--from-json", default=None, metavar="FILE",
                   help="Reload from a previous JSON run (no training).")
    p.add_argument("--list-problems", action="store_true",
                   help="List registered evaluators and exit.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress per-epoch training output.")
    p.set_defaults(func=_cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
