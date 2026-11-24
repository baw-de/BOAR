# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BOAR"
copyright = "de Oliveira et al. (2026)"
author = "LED. de Oliveira, MJ. Franca, NP. Huber, D. Vanzo"
release = "1.1.0"

# -- Creates the auto-documentation -------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../user_defined_configs"))
# Benchmark modules, imported by name in tests (as pytest's pythonpath = [".", "benchmarks"])
sys.path.insert(0, os.path.abspath("../benchmarks"))

# Test import
try:
    import src.basement_tools

    print("SUCCESS: src.basement_tools imported")
except ImportError as e:
    print(f"ERROR importing src.basement_tools: {e}")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": True,
    "inherited-members": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # table cells wrap, wide equations scroll

# Mock heavy C-extension / optional dependencies that are not available
# in the documentation build environment (sphinx-build runs without GPU libs).
autodoc_mock_imports = [
    "h5py",
    "torch",
    "scipy",
    "sklearn",
    "scikit_learn",
    "scikit_optimize",
    "skopt",
    "optuna",
    "pandas",
    "openpyxl",
    "SciencePlots",
]

html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
