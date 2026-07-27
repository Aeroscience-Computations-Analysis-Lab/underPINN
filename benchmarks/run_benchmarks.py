"""Standalone benchmark runner for underPINN.

Usage
-----
::

    # Run all fast problems — smooth PDEs get the default epoch budget;
    # problems marked complex=True (ramp, toro3) get a larger one automatically
    python benchmarks/run_benchmarks.py

    # Select problems and a custom budget for the simple ones
    python benchmarks/run_benchmarks.py \\
        --problems burgers wave ramp toro3 helmholtz heat_steady ode_harmonic \\
        --epochs 500 1000 2000 5000 \\
        --output outputs/bench

    # Include slow problems (3-D pipe flow, viscous ramp NS — also complex=True)
    python benchmarks/run_benchmarks.py --all

    # Override the complex-problem budget directly (shocks / viscous SBLI / 3-D
    # N-S converge far slower than smooth PDEs)
    python benchmarks/run_benchmarks.py --all --complex-epochs 2000 5000 15000

    # Load + replot from a previous JSON run (no training)
    python benchmarks/run_benchmarks.py --from-json outputs/bench/results.json

Or via the underPINN CLI:

    python -m underPINN bench
    python -m underPINN bench --problems burgers wave --epochs 1000 5000
    python -m underPINN bench --all --output outputs/bench_full
    python -m underPINN bench --all --complex-epochs 5000 20000 60000
"""

import argparse
import os
import sys

# Allow running as a script from the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="underPINN benchmark — accuracy vs. epoch budget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python benchmarks/run_benchmarks.py
  python benchmarks/run_benchmarks.py --problems burgers wave --epochs 1000 5000
  python benchmarks/run_benchmarks.py --all
  python benchmarks/run_benchmarks.py --all --complex-epochs 2000 5000 15000
  python benchmarks/run_benchmarks.py --from-json outputs/bench/results.json
""",
    )
    parser.add_argument(
        "--problems", nargs="+", default=None, metavar="PROB",
        help="Evaluator keys to run (default: all fast problems). "
             "See --list-problems for available names.",
    )
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=None, metavar="N",
        help="Epoch budgets for simple (smooth-PDE) problems "
             "(default: 500 1000 2000 5000).",
    )
    parser.add_argument(
        "--complex-epochs", nargs="+", type=int, default=None, metavar="N",
        help="Epoch budgets for problems marked complex=True (shocks / "
             "viscous SBLI / 3-D N-S: ramp, toro3, pipe_flow, ramp_ns) — "
             "these converge much slower than smooth PDEs, so they get "
             "their own, larger budget (default: 2000 5000 15000 40000).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include slow evaluators (e.g. 3-D pipe flow, viscous ramp NS).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S",
        help="Base PRNG seed (default: 0).",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/bench", metavar="DIR",
        help="Output directory for plots and tables (default: outputs/bench).",
    )
    parser.add_argument(
        "--from-json", default=None, metavar="FILE",
        help="Load results from a previous JSON run and regenerate report only.",
    )
    parser.add_argument(
        "--list-problems", action="store_true",
        help="List registered evaluator keys and exit.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-epoch training output.",
    )

    args = parser.parse_args(argv)

    from underPINN.benchmark_utils import (
        BenchmarkRunner, EVALUATOR_REGISTRY, generate_report, generate_report_pgf)
    from underPINN.benchmark_utils.benchmark_suite import BenchmarkRunner as BR

    # ── list mode ──────────────────────────────────────────────────────────────
    if args.list_problems:
        from underPINN.benchmark_utils.evaluators import SLOW_PROBLEMS
        print("Registered evaluators:")
        for k, cls in sorted(EVALUATOR_REGISTRY.items()):
            tags = ""
            if k in SLOW_PROBLEMS:
                tags += " [slow]"
            if getattr(cls, "complex", False):
                tags += " [complex]"
            print(f"  {k:<20s} {cls().__class__.__name__}{tags}")
        return

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # ── replay mode ────────────────────────────────────────────────────────────
    if args.from_json:
        print(f"Loading results from {args.from_json} …")
        results = BR.load_json(args.from_json)
        print(f"  {len(results)} result(s) loaded.")
        generate_report_pgf(results, runner=None, out_dir=os.path.join(out_dir, "pgf"))
        generate_report(results, runner=None, out_dir=out_dir)
        return

    # ── run mode ────────────────────────────────────────────────────────────────
    epoch_budgets = args.epochs or [500, 1000, 2000, 5000]
    complex_epoch_budgets = args.complex_epochs or [2000, 5000, 15000, 40000]
    problems      = args.problems  # None → all

    runner = BenchmarkRunner(
        problems=problems,
        epoch_budgets=epoch_budgets,
        complex_epoch_budgets=complex_epoch_budgets,
        seed=args.seed,
        fast_only=not args.all,
        verbose=not args.quiet,
    )

    print("=" * 60)
    print("  underPINN Benchmark Suite")
    print(f"  Problems         : {runner._problems}")
    print(f"  Epochs (simple)  : {epoch_budgets}")
    print(f"  Epochs (complex) : {complex_epoch_budgets}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output           : {out_dir}/")
    print("=" * 60)

    results = runner.run(out_dir=out_dir)

    # Save raw data
    runner.save_json(os.path.join(out_dir, "results.json"))
    runner.save_loss_npz(os.path.join(out_dir, "loss_hists.npz"))

    # PGFPlots data export
    generate_report_pgf(results, runner=runner, out_dir=os.path.join(out_dir, "pgf"))

    # Generate plots + tables
    generate_report(results, runner=runner, out_dir=out_dir)


if __name__ == "__main__":
    main()
