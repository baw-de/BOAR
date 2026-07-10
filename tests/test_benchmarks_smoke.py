"""
Smoke tests for the benchmark scripts.

Fast tests import each benchmark module and check it exposes a callable
``main()``.  A single end-to-end run is provided but marked ``slow`` so it
only executes when explicitly requested (``pytest -m slow``).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import matplotlib
import pytest

# Benchmarks render plots; force a headless backend so CI has no display.
matplotlib.use("Agg")

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
# Benchmarks do sys.path manipulation and import each other by bare name.
sys.path.insert(0, str(BENCH_DIR))


def _benchmark_modules() -> list[str]:
    """Names of all benchmark modules except the runner itself."""
    return [m.name for m in pkgutil.iter_modules([str(BENCH_DIR)]) if m.name != "run_all_benchmarks"]


@pytest.mark.parametrize("mod_name", _benchmark_modules())
def test_benchmark_imports(mod_name: str) -> None:
    """Each benchmark imports cleanly and exposes a callable main()."""
    module = importlib.import_module(mod_name)
    assert callable(getattr(module, "main", None)), f"benchmark '{mod_name}' has no callable main()"


@pytest.mark.slow
def test_ackley_benchmark_runs() -> None:
    """End-to-end: the Ackley benchmark completes without raising.

    Marked ``slow`` — run with ``pytest -m slow``.
    """
    module = importlib.import_module("ackley_function_constrained")
    module.main()
