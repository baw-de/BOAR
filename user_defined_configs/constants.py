"""
Defines project-wide path constants used by the configuration modules.

Attributes
----------
PROJECT_ROOT : pathlib.Path
    Root directory of the repository (one level above this file's parent directory).
BASEMENT_PATH : str | None
    Installation path to the BASEMENT executable binaries. ``None`` (default) uses the folder of
    ``basement_version`` in user_options.yaml: ``C:\\Program Files\\BASEMENT <version>.0\\bin`` on
    Windows, ``/usr/bin`` on Linux (where the .deb package installs).
SIMULATION_CONFIG_FOLDER : pathlib.Path
    Directory holding the original BASEMENT configuration (model.json, simulation.json, results.json) of the
    case to calibrate. BOAR only reads these files; it never modifies them.
OUTPUT_FOLDER : pathlib.Path
    Directory where each run works: the configuration is copied to ``OUTPUT_FOLDER / <simulation_name>``,
    where setup.h5, results.h5 and optimization_memory.csv are written.
SIMULATION_NAME_KEY : str
    Key in the SETUP section of model.json (BASEMENT's own ``simulation_name``) that names the output folder.
MESH_FOLDER : pathlib.Path
    Directory containing mesh input files.
SUPPORT_DATA_FOLDER : pathlib.Path
    Directory containing supplementary support data files.
GROUND_TRUTH_FOLDER : pathlib.Path
    Directory containing reference/validation ground-truth data.
"""
# Imports
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Constants for the project
# The paths can also be full paths if needed, using pathlib allows for relative paths that are more flexible across different environments.
# BASEMENT_PATH = None: use the install folder of 'basement_version' in user_options.yaml (default 4.3):
#   Windows: C:\Program Files\BASEMENT <version>.0\bin    Linux: /usr/bin (BASEMENT's .deb package)
# Set a folder to override it, e.g. r"D:\Tools\BASEMENT 4.3.0\bin". It must hold the version set in user_options.yaml.
BASEMENT_PATH: str | None = None
SIMULATION_CONFIG_FOLDER = PROJECT_ROOT / "simulation_configs"
OUTPUT_FOLDER = PROJECT_ROOT / "output"
# Read from model.json: {"SETUP": {"simulation_name": "my_case", ...}} -> OUTPUT_FOLDER / "my_case"
SIMULATION_NAME_KEY = "simulation_name"
MESH_FOLDER = PROJECT_ROOT / "meshes"
SUPPORT_DATA_FOLDER = PROJECT_ROOT / "support_files"
GROUND_TRUTH_FOLDER = PROJECT_ROOT / "ground_truth"
