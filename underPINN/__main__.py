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
