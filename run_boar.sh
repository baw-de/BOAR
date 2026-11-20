#!/usr/bin/env bash
# Runs BOAR from the project root on Linux / macOS (Windows: run_boar.bat).
# Copy the project folder to run several calibrator instances side by side; each copy runs from its own root.
# BOAR itself generates setup.h5 from model.json and runs BASEMENT, using 'basement_version' in
# user_defined_configs/user_options.yaml (default 4.3) or BASEMENT_PATH in user_defined_configs/constants.py.
#
# Usage: ./run_boar.sh [boar.py options, e.g. --log-file my_run.log]

set -euo pipefail

# BOAR reads user_defined_configs/ relative to the project root, so run from there.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Python of the virtual environment. Set PYTHON_EXE beforehand to use another one.
PYTHON_EXE="${PYTHON_EXE:-$PROJECT_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_EXE" ]; then
    # Fallback: python3 on the PATH.
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"
    else
        echo "ERROR: Python was not found:" >&2
        echo "  \"$PYTHON_EXE\"" >&2
        echo "Create .venv (python3 -m venv .venv && .venv/bin/pip install -e .) or set PYTHON_EXE." >&2
        exit 1
    fi
fi

exec "$PYTHON_EXE" "$PROJECT_ROOT/boar.py" "$@"
