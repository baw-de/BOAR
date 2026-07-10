# Contributing to BOAR

Thank you for your interest in contributing to **BOAR** (Bayesian Optimization for Automated Roughness calibration)!
Contributions of any kind are welcome — bug reports, feature requests, documentation improvements, and code changes.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching & Workflow](#branching--workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

Please be respectful and constructive in all interactions.
We expect contributors to adhere to basic open-source community norms: be inclusive, patient, and collaborative.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/boar.git
   cd boar
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/<org>/boar.git
   ```

---

## Development Setup

BOAR requires **Python ≥ 3.10**. We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs all runtime dependencies plus the development extras defined in `pyproject.toml`:
`ruff`, `mypy`, `pytest`, `pytest-cov`, `bandit`, `pip-audit`, and `Sphinx`.

---

## Branching & Workflow (Gitflow)

BOAR follows the **[Gitflow](https://nvie.com/posts/a-successful-git-branching-model/)** branching model.

```
main ─────────────────────────────────────────────── (tagged releases)
  └─ develop ──────────────────────────────────────── (integration)
        ├─ feature/<name>   (new features)
        ├─ fix/<name>       (bug fixes)
        ├─ docs/<name>      (documentation only)
        └─ release/<x.y.z>  (release preparation → merges into main + develop)
```

### Branch Rules

| Branch | Branches off | Merges into | Purpose |
|--------|-------------|-------------|---------|
| `main` | — | — | Stable, tagged releases only |
| `develop` | `main` | `main` via release | Ongoing integration |
| `feature/<name>` | `develop` | `develop` | New features or enhancements |
| `fix/<name>` | `develop` | `develop` | Bug fixes |
| `hotfix/<name>` | `main` | `main` + `develop` | Critical production fixes |
| `release/<x.y.z>` | `develop` | `main` + `develop` | Release preparation (changelog, version bump) |
| `docs/<name>` | `develop` | `develop` | Documentation-only changes |

### Commit Message Convention

Use the **[Conventional Commits](https://www.conventionalcommits.org/)** format:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer: Closes #123]
```

| Type | When to use |
|------|------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or correcting tests |
| `ci` | CI/CD pipeline changes |
| `chore` | Maintenance (deps, build, tooling) |
| `perf` | Performance improvement |

**Examples:**
```
feat(optimizer): add TPE sampler support for optuna backend
fix(io): handle missing HDF5 dataset gracefully
docs(quickstart): add BASEMENT v4.2 setup example
ci: add pip-audit step to security job
```

- Subject line: imperative mood, ≤ 72 characters, no trailing period.
- Reference issues with `Closes #<n>` or `Refs #<n>` in the footer.

---

## Coding Standards

All code must pass the following checks before a pull request is merged:

### Linting & Formatting — `ruff`

```bash
ruff check .          # lint
ruff format .         # auto-format
```

Configuration lives in `pyproject.toml` under `[tool.ruff]`.
Line length is **120 characters**.

### Type Checking — `mypy`

```bash
mypy src/ boar.py
```

Start lenient; we aim to tighten toward `--strict` before v2.0.

### Security — `bandit` & `pip-audit`

```bash
bandit -r src/ boar.py
pip-audit
```

These checks are also enforced in CI (see `.gitlab-ci.yml`).

---

## Testing

BOAR uses **pytest** with coverage reporting:

```bash
pytest
```

- Tests live in the `tests/` directory.
- New features **must** include corresponding tests.
- Bug fixes **should** include a regression test.
- Coverage must not decrease; aim to improve it where possible.

---

## Documentation

Documentation is built with **Sphinx** and lives in the `docs/` directory.

```bash
cd docs
make html
# Open docs/_build/html/index.html in a browser
```

- Follow existing `.rst` formatting conventions.
- Update the relevant docstrings when changing public API.
- For significant new features, add or extend a documentation page.

---

## Submitting Changes

1. Make sure all checks pass locally (`ruff`, `mypy`, `pytest`, `bandit`).
2. Push your branch to your fork.
3. Open a **Pull Request** against the `develop` branch.
4. Fill in the PR template (if available) and describe:
   - *What* the change does.
   - *Why* it is needed.
   - Any relevant issue numbers (`Closes #123`).
5. A maintainer will review your PR. Please respond to review feedback promptly.

---

## Reporting Bugs

Please open a **GitHub Issue** and include:

- A short, descriptive title.
- Steps to reproduce the problem.
- Expected vs. actual behaviour.
- BOAR version (`pyproject.toml`), Python version, and OS.
- Any relevant log output or error tracebacks.

---

## Suggesting Features

Open a **GitHub Issue** with the label `enhancement` and describe:

- The problem you are trying to solve.
- Your proposed solution or API sketch.
- Any alternatives you considered.

We particularly welcome contributions related to:

- Support for additional hydrodynamic modelling software (beyond BASEMENT).
- New calibration strategies or objective functions.
- Performance improvements.
- Extended documentation and usage examples.

---

*Thank you for helping improve BOAR!*
