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

BOAR requires **Python ≥ 3.10** and a working installation of **BASEMENT v4.2** (or compatible version).

```bash
# 1. Clone the repository
git clone https://github.com/baw-de/boar.git
cd boar

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install BOAR and its dependencies
pip install -e .
```

For development (linting, tests, docs):
```bash
pip install -e ".[dev]"
```

> For a step-by-step walkthrough including BASEMENT setup, build the docs locally and open `docs/_build/html/quickstart.html` and `docs/_build/html/installation.html` — see [Building the docs](#building-the-docs) below.
> Alternatively check the online documentation pages at: https://baw-de.github.io/BOAR/quickstart.html and https://baw-de.github.io/BOAR/installation.html

## Configuration

All user-facing settings live in [`user_defined_configs/user_options.yaml`](user_defined_configs/user_options.yaml).
Edit this file before running BOAR. The key sections are:

| Section | What to configure |
|---------|------------------|
| `general_options` | Logging verbosity, fresh-start flag |
| `basement_options` | BASEMENT backend (`seq`, `omp`, `cuda`), number of CPU cores |
| `simulation_options` | Path to discharge input files |
| `optimization_variable_options` | Roughness regions, parameter bounds, constraint expressions |
| `sampling_options` | Number of Latin Hypercube samples, random seed |
| `surrogate_model_options` | GPR iterations, stopping tolerances, max evaluations |

> See the inline comments in `user_options.yaml` for a description of every option.
> For a full parameter reference, build the docs and open `optimization_configuration.html` and `loss_function.html`.
> Or: https://baw-de.github.io/BOAR/optimization_configuration.html and https://baw-de.github.io/BOAR/loss_function.html

## Running BOAR

### Linux / macOS

```bash
# Run with default log file (boar.log)
python boar.py

# Run with a custom log file
python boar.py --log-file my_run.log
```

### Windows

A ready-to-use batch script is provided. Before the first run, open [`run_boar.bat`](run_boar.bat) and adjust:

- `PYTHON_EXE` — path to your Python executable inside the virtual environment
- `SETUP_EXE` — path to the BASEMENT `BMv4_BASEHPC_setup.exe`
- `SIM_DIR` — path to the directory containing your `model.json` and simulation files

Then launch via:
```bat
run_boar.bat
```

The batch script will regenerate `setup.h5` from `model.json` and start the calibration automatically.

### Output

BOAR writes results to the output directory defined in `user_options.yaml`. A log file (`boar.log` by default) is created in the project root and records the full calibration history.

## Building the docs

The documentation is built locally with **Sphinx**. Make sure the dev dependencies are installed (`pip install -e ".[dev]"`), then:

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

BibTeX:

```bibtex
@article{DEOLIVEIRA2026102899,
title = {BOAR: Bayesian optimization for automated roughness calibration in two-dimensional hydrodynamic models},
journal = {SoftwareX},
volume = {35},
pages = {102899},
year = {2026},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2026.102899},
url = {https://www.sciencedirect.com/science/article/pii/S2352711026003900},
author = {Luiz E.D. {de Oliveira} and Mário J. Franca and Nils P. Huber and Davide Vanzo},
keywords = {Bayesian optimization, Hydraulic roughness calibration, Shallow water solvers, BASEMENT, Surrogate model, Gaussian process, Model calibration},
abstract = {BOAR is an open-source Python framework for automated hydraulic roughness calibration in two-dimensional shallow-water numerical solvers. BOAR combines Bayesian optimization with conditional sampling to integrate expert knowledge, including feasible parameter ranges and inter-parameter constraints, thereby reducing the number of model evaluations. It standardizes calibration workflows, improves reproducibility, and supports flexible, user-defined loss functions. Applied to 25 laboratory vegetated compound-channel flow cases, it converged within a median of 9 model runs for a depth-error of 0.4%–1.5% tolerance. The current implementation interfaces with BASEMENT and can be extended to other river-modeling tools, providing a flexible and extensible solution for hydraulic modeling.}
}
```

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
