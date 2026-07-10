"""
Defines project-wide path constants used by the configuration modules.

Attributes
----------
PROJECT_ROOT : pathlib.Path
    Root directory of the repository (one level above this file's parent directory).
BASEMENT_PATH : str
    Installation path to the BASEMENT executable binaries.
SIMULATION_FOLDER : pathlib.Path
    Directory containing simulation setup and run files.
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
BASEMENT_PATH = r"C:\Program Files\BASEMENT 4.2.0\bin"
SIMULATION_FOLDER = PROJECT_ROOT / "_mock_simulation_folder"
MESH_FOLDER = PROJECT_ROOT / "_mock_meshes"
SUPPORT_DATA_FOLDER = PROJECT_ROOT / "_mock_support_files"
GROUND_TRUTH_FOLDER = PROJECT_ROOT / "ground_truth"
