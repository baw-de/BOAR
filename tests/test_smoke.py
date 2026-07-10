"""Smoke tests: the package and core modules import cleanly."""

from __future__ import annotations

import importlib

import pytest

CORE_MODULES = [
    "boar",
    "src.utils",
    "src.functions_io",
    "src.functions_sampler",
    "src.bo_optimizer",
    "src.basement_tools",
    "src.cell_centroid",
    "src.friction_calibration",
]


@pytest.mark.parametrize("mod_name", CORE_MODULES)
def test_module_imports(mod_name: str) -> None:
    """Each core module imports without raising."""
    importlib.import_module(mod_name)
