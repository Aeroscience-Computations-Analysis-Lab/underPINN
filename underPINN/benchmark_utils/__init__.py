"""underPINN.benchmark_utils — systematic accuracy-vs-epoch benchmarking.

Public API
----------
:class:`BenchmarkResult`
    Dataclass holding a single (problem, epochs) measurement.
:class:`BenchmarkRunner`
    Orchestrates multi-problem, multi-epoch runs.
:data:`EVALUATOR_REGISTRY`
    Dict mapping problem name → evaluator class.
:func:`generate_report`
    Convenience wrapper: run + generate all plots and tables.
"""

from underPINN.benchmark_utils.benchmark_suite import (
    BenchmarkResult,
    BenchmarkRunner,
)
from underPINN.benchmark_utils.evaluators import EVALUATOR_REGISTRY
from underPINN.benchmark_utils.report import generate_report

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "EVALUATOR_REGISTRY",
    "generate_report",
]
