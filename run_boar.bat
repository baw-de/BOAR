:: This batch file is designed to set up the environment and run the main Python script for a project that uses BASEMENT 4.2.0 for simulation.
:: It allows the user to copy and paste instances of the calibrator code and seamlessly run the simulation without needing to worry about the underlying setup.
:: Make sure to adjust the paths for the Python executable and the BASEHPC setup executable as needed before running this batch file, and replicating it for multiple instances if necessary.

setlocal

set "PROJECT_ROOT=%~dp0"
for %%I in ("%PROJECT_ROOT%.") do set "PROJECT_ROOT=%%~fI"
set "SIM_DIR=%PROJECT_ROOT%\simulation"  :: This assumes that the simulation directory is located at the root of the project. Adjust if necessary.
set "PYTHON_EXE=D:\2024_KIT_BAW_Unsteady_Compound_Channels\10_Repositories\basement-tools\.venv\Scripts\python.exe" :: Example: C:\Users\YourUsername\Envs\YourEnv\Scripts\python.exe
set "SETUP_EXE=C:\Program Files\BASEMENT 4.2.0\bin\BMv4_BASEHPC_setup.exe" :: Adjust the path to the BASEHPC setup executable if it's different on your system.

if exist "%SIM_DIR%\setup.h5" del "%SIM_DIR%\setup.h5"
"%SETUP_EXE%" --file "%SIM_DIR%\model.json" --output "%SIM_DIR%\setup.h5" :: Optional - This command generates the setup.h5 file from the model.json file.
"%PYTHON_EXE%" "%PROJECT_ROOT%\boar.py"

endlocal
