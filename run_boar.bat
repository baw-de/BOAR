@echo off
:: Runs BOAR from the project root on Windows (Linux / macOS: run_boar.sh).
:: Copy the project folder to run several calibrator instances side by side; each copy runs from its own root.
:: BOAR itself generates setup.h5 from model.json and runs BASEMENT, using 'basement_version' in
:: user_defined_configs/user_options.yaml (default 4.3) or BASEMENT_PATH in user_defined_configs/constants.py.
::
:: Usage: run_boar.bat [boar.py options, e.g. --log-file my_run.log]

setlocal

:: Resolve the project root to an absolute path without a trailing backslash.
set "PROJECT_ROOT=%~dp0"
for %%I in ("%PROJECT_ROOT%.") do set "PROJECT_ROOT=%%~fI"

:: BOAR reads user_defined_configs/ relative to the project root, so run from there.
cd /d "%PROJECT_ROOT%"

:: Python of the virtual environment. Set PYTHON_EXE beforehand to use another one.
if not defined PYTHON_EXE set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if exist "%PYTHON_EXE%" goto :run

:: Fallback: python on the PATH.
where python >nul 2>nul
if %errorlevel% neq 0 (
	echo ERROR: Python was not found:
	echo   "%PYTHON_EXE%"
	echo Create .venv in the project folder ^(uv sync, or python -m venv .venv and pip install -e .^), or set PYTHON_EXE.
	exit /b 1
)
set "PYTHON_EXE=python"

:run
"%PYTHON_EXE%" "%PROJECT_ROOT%\boar.py" %*
exit /b %errorlevel%
