#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Import user defined functions
from user_defined_configs import constants as c

# Create a logger for file_import.py
logger = logging.getLogger("file_import")

MeshData = dict[str, np.ndarray | dict[str, np.ndarray] | None]

# BASEMENT configuration files copied from SIMULATION_CONFIG_FOLDER into the run folder
SIMULATION_CONFIG_FILES = ("model.json", "simulation.json", "results.json")


def read_mesh_file(filepath: str) -> MeshData:
    """
    Imports the mesh from a given filepath.

    Args:
        filepath (str): filepath to the mesh file

    Returns:
        MeshData: dictionary containing the mesh data, including nodes, cells, and boundary conditions
    """
    data: MeshData = {
        "nodes": None,  # Structured array for nodes
        "cells": None,  # Structured array for elements
        "boundary_conditions": {},  # Dictionary for node sets
    }

    nodes_list = []
    elements_list = []
    node_sets = {}

    with open(filepath) as file:
        for line in file:
            line = line.strip()
            if line.startswith("ND"):
                # Parse ND: ID X Y Z
                _, node_id, x, y, z = line.split()
                nodes_list.append((int(node_id), float(x), float(y), float(z)))
            elif line.startswith("E3T"):
                # Parse E3T: ID ND1 ND2 ND3 REGION Z
                _, elem_id, nd1, nd2, nd3, region, z = line.split()
                elements_list.append((int(elem_id), int(nd1), int(nd2), int(nd3), int(region), float(z)))
            elif line.startswith("NS"):
                # Parse NS: ND1 ND2 ... -ND_n NAME
                parts = line.split()
                set_name = parts[-1]
                node_ids = [int(x) for x in parts[1:-1] if not x.startswith("-")]
                node_sets[set_name] = np.array(node_ids, dtype=int)
            elif line.startswith("MESH2D") or line.startswith("MESHNAME") or line.startswith("NUM_MATERIALS_PER_ELEM"):
                # Skip header lines
                continue
            else:
                print(f"Unrecognized line format: {line}")

    # Convert lists to structured NumPy arrays
    if nodes_list:
        data["nodes"] = np.array(nodes_list, dtype=[("id", "i4"), ("x", "f8"), ("y", "f8"), ("z", "f8")])
    if elements_list:
        data["cells"] = np.array(
            elements_list,
            dtype=[("id", "i4"), ("nd1", "i4"), ("nd2", "i4"), ("nd3", "i4"), ("region", "i4"), ("z", "f8")],
        )
    if node_sets:
        data["boundary_conditions"] = {k: v for k, v in node_sets.items()}

    return data


def find_available_meshes(paths: dict) -> dict:
    """
    Creates the high level organition to read the meshes.
    This function and the code can be optimized by creating a lazy read of this data.
    It is possible because if the centroid was previously calculated, there would be no need
    to import it again.

    Args:
        paths (dict): dictionary containing all relevant paths

    Returns:
        dict: dictionary containing the loaded meshes
    """
    # Creates a list with the mesh files to be imported (skips .gitkeep, mesh_centroids/ and other files)
    to_import = [file for file in os.listdir(paths["folder_mesh"]) if file.endswith(".2dm")]

    # Checks if there are files to be imported
    if len(to_import) == 0:
        logger.error(
            "No .2dm mesh files found in the mesh folder. Please check the path and the files in the mesh folder."
        )
        raise FileNotFoundError(
            "No .2dm mesh files found in the mesh folder. Please check the path and the files in the mesh folder."
        )

    # Reads the mesh
    dict_meshes = dict()

    for file in to_import:
        filepath = os.path.join(paths["folder_mesh"], file)
        dict_meshes[file] = filepath

    return dict_meshes


def read_csv_files_in_folder(path: str) -> dict:
    """
    Reads all CSV files from a given directory and stores them in a dictionary.

    Args:
        path (str): path to the directory containing the CSV files

    Returns:
        dict: dictionary containing the loaded CSV data
    """
    # Creates a list of files to be imported
    to_import = os.listdir(path)

    # Remove git associated files
    to_import = [file for file in to_import if not file.startswith(".")]

    # Remove subdirectories
    to_import = [file for file in to_import if os.path.isfile(os.path.join(path, file))]

    # Reads the CSV data
    output = dict()
    for filename in to_import:
        # Reads the CSV data
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath)

        # Stores into the output dictionary
        output[filename] = df.copy()

    return output


def _absolute_path(value: str, search_dirs: list[Path], label: str, logger: logging.Logger) -> str:
    """
    Return ``value`` as an absolute path, resolving a relative one against ``search_dirs`` in order.

    A path found in none of them is returned unchanged, with a warning.
    """
    path = Path(value)
    if path.is_absolute():
        return value
    for folder in search_dirs:
        candidate = (folder / path).resolve()
        if candidate.exists():
            return str(candidate)
    logger.warning(f"{label} '{value}' not found in {[str(d) for d in search_dirs]}; left unchanged.")
    return value


def prepare_simulation_folder(
    config_folder: str | Path, output_root: str | Path, overwrite: bool, logger: logging.Logger
) -> Path:
    """
    Copy the original BASEMENT configuration into ``output_root / <simulation_name>`` and return that folder.

    The files in ``config_folder`` are only read. The name comes from ``SETUP.simulation_name`` in model.json. The JSON files are always copied fresh, so every run starts from the original friction and
    discharge. With ``overwrite`` the case folder is deleted first; without it, earlier results such as
    optimization_memory.csv stay and the run resumes from them.

    In the copied model.json relative ``mesh_file`` and ``discharge_file`` paths are made absolute, resolved
    against ``config_folder`` and then the project root.

    Args:
        config_folder: Folder holding model.json, simulation.json and results.json.
        output_root: Folder in which the case folder is created.
        overwrite: Delete an existing case folder before copying.
        logger: Logger for progress and error messages.

    Returns:
        Path: The case folder, in which BOAR runs the calibration.

    Raises:
        FileNotFoundError: If a configuration file is missing.
        KeyError: If model.json has no simulation name.
        ValueError: If the simulation name is not a plain folder name.
    """
    config_folder = Path(config_folder)
    output_root = Path(output_root)

    missing = [name for name in SIMULATION_CONFIG_FILES if not (config_folder / name).is_file()]
    if missing:
        message = f"Missing BASEMENT configuration files in {config_folder}: {missing}"
        logger.error(message)
        raise FileNotFoundError(message)

    with open(config_folder / "model.json") as f:
        model_config = json.load(f)

    name = model_config.get("SETUP", {}).get(c.SIMULATION_NAME_KEY)
    if not isinstance(name, str) or not name.strip():
        message = (
            f"{config_folder / 'model.json'} has no 'SETUP.{c.SIMULATION_NAME_KEY}'. Set it to name the case, "
            f'e.g. {{"SETUP": {{"{c.SIMULATION_NAME_KEY}": "my_case", ...}}}}.'
        )
        logger.error(message)
        raise KeyError(message)
    name = name.strip()
    if name in (".", "..") or any(sep in name for sep in ("/", "\\", ":")):
        message = f"'SETUP.{c.SIMULATION_NAME_KEY}' must be a plain folder name, got {name!r}."
        logger.error(message)
        raise ValueError(message)

    case_folder = output_root / name
    if overwrite and case_folder.exists():
        logger.info(f"overwrite is True: deleting {case_folder}")
        shutil.rmtree(case_folder)
    case_folder.mkdir(parents=True, exist_ok=True)

    for filename in SIMULATION_CONFIG_FILES:
        shutil.copyfile(config_folder / filename, case_folder / filename)

    # Adapt the copied model.json to its new location
    search_dirs = [config_folder.resolve(), Path(c.PROJECT_ROOT).resolve()]
    plane = model_config.get("SETUP", {}).get("BASEHPC", {}).get("BASEPLANE_2D", {})
    geometry = plane.get("GEOMETRY", {})
    if isinstance(geometry.get("mesh_file"), str):
        geometry["mesh_file"] = _absolute_path(geometry["mesh_file"], search_dirs, "mesh_file", logger)
    for bc in plane.get("HYDRAULICS", {}).get("BOUNDARY", {}).get("STANDARD", []):
        if isinstance(bc, dict) and isinstance(bc.get("discharge_file"), str):
            bc["discharge_file"] = _absolute_path(bc["discharge_file"], search_dirs, "discharge_file", logger)
    with open(case_folder / "model.json", "w") as f:
        json.dump(model_config, f, indent=4)

    logger.info(f"Simulation '{name}': originals in {config_folder}, run folder {case_folder}")
    return case_folder


def setup_directories(project_path, user_options, logger):
    folder_mesh = c.MESH_FOLDER
    folder_centroids = os.path.join(folder_mesh, "mesh_centroids")
    folder_ground_truth = c.GROUND_TRUTH_FOLDER
    folder_output = c.OUTPUT_FOLDER
    folder_support = c.SUPPORT_DATA_FOLDER

    if not os.path.exists(folder_ground_truth):
        os.makedirs(folder_ground_truth)

        logger.error("ground_truth folder not found. Please populate the newly created directories accourdingly.")
        raise FileNotFoundError(
            "ground_truth folder not found. Please populate the newly created directories accourdingly."
        )

    general_options = user_options.get("general_options", {})
    if "overwrite" in general_options:
        overwrite = bool(general_options["overwrite"])
    elif "clear_start" in general_options:
        logger.warning("general_options 'clear_start' is deprecated; rename it to 'overwrite'.")
        overwrite = bool(general_options["clear_start"])
    else:
        overwrite = False

    folder_simulation = prepare_simulation_folder(c.SIMULATION_CONFIG_FOLDER, folder_output, overwrite, logger)

    if folder_support is None or not os.path.exists(folder_support):
        logger.error(
            "Support data folder not specified. The inclusion of a lookup table containing (discharge_code, discharge_value) is required."
        )
        raise ValueError(
            "Support data folder not specified. The inclusion of a lookup table containing (discharge_code, discharge_value) is required."
        )

    return {
        "path_project": project_path,
        "folder_ground_truth": folder_ground_truth,
        "folder_simulation": str(folder_simulation),
        "folder_simulation_configs": c.SIMULATION_CONFIG_FOLDER,
        "folder_mesh": folder_mesh,
        "folder_centroids": folder_centroids,
        "folder_output": folder_output,
        "folder_support": folder_support,
    }


def main(paths: dict) -> dict:
    """
    Data import

    Args:
        paths (dict): dictionary containing all relevant paths

    Returns:
        dict: dictionary containing the data
    """
    logger.debug("Importing data")

    # Creates the data holding structure
    output = dict()

    # Reads the meshes
    output["mesh"] = find_available_meshes(paths)

    # Reads ground truth data
    output["ground_truth_data"] = read_csv_files_in_folder(paths["folder_ground_truth"])

    # Reads the discharge data if existing
    if os.path.exists(c.SUPPORT_DATA_FOLDER):
        output["support_data"] = read_csv_files_in_folder(paths["folder_support"])

    logger.info("Data imported")

    return output
