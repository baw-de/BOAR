# BOAR: Bayesian Optimization for Automated Roughness calibration in two-dimensional hydrodynamic models

**Documentation** — build locally with Sphinx (see [Building the docs](#building-the-docs) below) · Quickstart · Installation · Configuration · API Reference

Also available at: https://baw-de.github.io/BOAR/

BOAR is a tool for automating the calibration of roughness parameters in two-dimensional hydrodynamic models using Bayesian Optimization. The project is designed to reduce manual trial-and-error during model calibration and support more efficient, reproducible parameter estimation workflows.

## Overview

Calibrating hydraulic roughness parameters is a critical step in two-dimensional hydrodynamic modeling. Traditional calibration approaches often require repeated simulations, manual parameter tuning, and expert judgment. BOAR addresses this challenge by coupling hydrodynamic model evaluation with Bayesian Optimization to search for roughness values that improve agreement between simulations and reference observations.

The repository is intended for users working with computational hydraulics, model calibration, uncertainty-aware parameter tuning, and automated simulation workflows.

## Features

- Automated roughness calibration workflow based on Bayesian Optimization-based parameter search
- Reproducible calibration experiments
- Extensible structure for custom objective functions and model setups

## Use Cases

BOAR can be used for:

- Calibrating spatially distributed or grouped roughness parameters
- Improving hydraulic model fit to measured water levels, velocities, or other flow variables
- Running sensitivity-informed calibration studies

## Installation

BOAR requires **Python ≥ 3.11** and a working installation of **BASEMENT v4.3** or **v4.2** (set `basement_version` in `basement_options` to match; default `"4.3"`). The dependencies are listed in `pyproject.toml`. BOAR runs from the project folder in a virtual environment `.venv`, set up either with [uv](https://docs.astral.sh/uv/getting-started/installation/) (default; uses the pinned versions in `uv.lock` and downloads a suitable Python if needed) or with pip.

```bash
# 1. Clone the repository
git clone https://github.com/baw-de/boar.git
cd boar
```

**2a. With uv (default)**
```bash
uv sync --no-dev                 # creates .venv with the locked dependencies
```

BOAR calls BASEMENT's `BMv4_setup`, `BMv4_simulation` and `BMv4_results` from the install folder of `basement_version` in `basement_options` (default `"4.3"`): `C:\Program Files\BASEMENT <version>.0\bin` on Windows (e.g. `C:\Program Files\BASEMENT 4.3.0\bin`) and `/usr/bin` on Linux, where the BASEMENT Ubuntu package (`sudo apt install ./<downloaded BASEMENT package>.deb`) installs them. If BASEMENT is installed somewhere else, set `BASEMENT_PATH` in [`user_defined_configs/constants.py`](user_defined_configs/constants.py) to that folder.

For development (linting, tests, docs), leave out `--no-dev` to install the `dev` dependency group as well: `uv sync`.

**2b. With pip**, from the same `pyproject.toml` (versions not pinned by `uv.lock`):
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # runtime dependencies
pip install --group dev          # development tools (optional, pip >= 25.1)
```

> For a step-by-step walkthrough including BASEMENT setup, build the docs locally and open `docs/_build/html/quickstart.html` and `docs/_build/html/installation.html` — see [Building the docs](#building-the-docs) below.
> Alternatively check the online documentation pages at: https://baw-de.github.io/BOAR/quickstart.html and https://baw-de.github.io/BOAR/installation.html

## Configuration

All user-facing settings live in [`user_defined_configs/user_options.yaml`](user_defined_configs/user_options.yaml).
Edit this file before running BOAR. The key sections are:

| Section | What to configure |
|---------|------------------|
| `general_options` | Logging verbosity, fresh-start flag |
| `basement_options` | BASEMENT backend (`seq`, `omp`, `cuda`), number of CPU cores, BASEMENT version (`4.2` or `4.3`) |
| `simulation_options` | Path to discharge input files |
| `optimization_variable_options` | Roughness regions, parameter bounds, constraint expressions |
| `sampling_options` | Number of Latin Hypercube samples, random seed |
| `surrogate_model_options` | GPR iterations, stopping tolerances, max evaluations |

> See the inline comments in `user_options.yaml` for a description of every option.
> For a full parameter reference, build the docs and open `optimization_configuration.html` and `loss_function.html`.
> Or: https://baw-de.github.io/BOAR/optimization_configuration.html and https://baw-de.github.io/BOAR/loss_function.html

## Running BOAR

Ready-to-use launchers are provided: [`run_boar.bat`](run_boar.bat) (Windows) and [`run_boar.sh`](run_boar.sh) (Linux / macOS). They run `boar.py` from the project root with the Python of `.venv` (from `uv sync` or pip; otherwise `python` / `python3` on the PATH; set `PYTHON_EXE` to use another one) and pass their options on. There is nothing to adjust in them: BOAR itself generates `setup.h5` from `model.json` and runs BASEMENT of the version set in `basement_options`.

```bash
# Windows
run_boar.bat
run_boar.bat --log-file my_run.log         # custom log file (default: boar.log)

# Linux / macOS
chmod +x run_boar.sh                       # once, if the file is not executable
./run_boar.sh
./run_boar.sh --log-file my_run.log
```

BOAR can also be started directly with `uv run boar.py [--log-file my_run.log]` (or `python boar.py` in an activated pip environment). To run several instances side by side, copy the project folder; each copy runs from its own root.

### Output

BOAR never modifies the original BASEMENT configuration in `simulation_configs/` (`model.json`, `simulation.json`, `results.json`). Each run copies it to `output/<simulation_name>/`, where `simulation_name` is the `SETUP.simulation_name` entry of `model.json`. All files the run produces go there: `setup.h5`, `results.h5`, `optimization_memory.csv` and `tried_vectors/`. With `overwrite: True` in `general_options`, that folder is deleted before the run. With `False`, the run resumes from its `optimization_memory.csv`. A log file (`boar.log` by default) is created in the project root and records the full calibration history.

## Building the docs

The documentation is built locally with **Sphinx**. Make sure the dev dependencies are installed (`uv sync`, or `pip install --group dev`) and `.venv` is activated (`source .venv/bin/activate`, Windows: `.venv\Scripts\activate`), then:

```bash
# Optional: regenerate API stubs and YAML config pages first
python docs/generate_docs.py

# Build HTML docs
cd docs
make html

# Open in browser
open _build/html/index.html        # macOS
start _build/html/index.html       # Windows
xdg-open _build/html/index.html   # Linux
```

The generated HTML is in `docs/_build/html/`. Key pages:

| Page | File |
|------|------|
| Quickstart | `_build/html/quickstart.html` |
| Installation | `_build/html/installation.html` |
| Optimization configuration | `_build/html/optimization_configuration.html` |
| Loss function | `_build/html/loss_function.html` |
| API Reference | `_build/html/source/modules.html` |

---

## How to Cite

If you use BOAR in your research, please cite:

> de Oliveira, L.E.D., Franca, M.J., Huber, N.P., Vanzo, D., 2026. BOAR: Bayesian optimization for automated roughness calibration in two-dimensional hydrodynamic models. SoftwareX 35, 102899. https://doi.org/10.1016/j.softx.2026.102899

On GitHub, click **Cite this repository** in the right-hand sidebar to copy this reference in APA or BibTeX format (generated from [CITATION.cff](CITATION.cff)).

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide (Gitflow, coding standards, tests). Suggested contributions include:

- documentation improvements
- bug fixes
- performance enhancements
- support for additional calibration strategies
- integration with new hydrodynamic modeling workflows

## Supported hydrodynamic softwares

Currently BOAR only supports:

- BASEMENT v4.2 [[1]](#1)[[2]](#2)

## References

<a id="1">[1]</a>
Vanzo, D., Peter, S., Vonwiller, L., Bürgler, M., Weberndorfer, M., Siviglia, A., Conde, D., Vetsch, D.F., 2021. basement v3: A modular freeware for river process modelling over multiple computational backends. Environmental Modelling & Software 143, 105102. https://doi.org/10.1016/j.envsoft.2021.105102

<a id="2">[2]</a>
Vetsch, D.F., Frei, S., Halso, M.C., Schierjott, J.C., Bürgler, M., Vanzo, D., 2024. Basement V4—A Multipurpose Modelling Environment for Simulation of Flood Hazards and River Morphodynamics Across Scales, in: Gourbesville, P., Caignaert, G. (Eds.), Advances in Hydroinformatics—SimHydro 2023 Volume 1. Springer Nature, Singapore, pp. 125–138. https://doi.org/10.1007/978-981-97-4072-7_8
