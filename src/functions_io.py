#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import logging
import os
import shutil

import numpy as np
import pandas as pd

# Import user defined functions
from user_defined_configs import constants as c

# Create a logger for file_import.py
logger = logging.getLogger("file_import")

MeshData = dict[str, np.ndarray | dict[str, np.ndarray] | None]


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
    # Creates a list with the files to be imported
    to_import = os.listdir(paths["folder_mesh"])

    # Validate that the files to be imported are mesh files
    for file in to_import:
        if not file.endswith(".2dm"):
            to_import.remove(file)

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


def setup_directories(project_path, user_options, logger):
    folder_simulation = c.SIMULATION_FOLDER
    folder_mesh = c.MESH_FOLDER
    folder_centroids = os.path.join(folder_mesh, "mesh_centroids")
    folder_ground_truth = c.GROUND_TRUTH_FOLDER
    folder_output = os.path.join(project_path, "output")
    folder_support = c.SUPPORT_DATA_FOLDER

    if not os.path.exists(folder_ground_truth):
        os.makedirs(folder_ground_truth)

        logger.error("ground_truth folder not found. Please populate the newly created directories accourdingly.")
        raise FileNotFoundError(
            "ground_truth folder not found. Please populate the newly created directories accourdingly."
        )

    if user_options["general_options"]["clear_start"] and os.path.exists(folder_output):
        shutil.rmtree(folder_output)
        os.makedirs(folder_output)

    if folder_simulation is None or not os.path.exists(folder_simulation):
        logger.error(
            "Basement simulation folder not specified. Please provide the path to the simulation folder in: user_defined_configs/constants.py"
        )
        raise ValueError(
            "Basement simulation folder not specified. Please provide the path to the simulation folder in: user_defined_configs/constants.py"
        )

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
        "folder_simulation": folder_simulation,
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
