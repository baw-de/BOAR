#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import logging
import os

# Third-party imports
import numpy as np
import pandas as pd

# Import Basement-Calibrator modules
from src import functions_io

# Set up logging
logger = logging.getLogger("cell_centroid")


def calculate_centroids(
    mesh: dict, mesh_name: str, output_dir: str = "config", main_args: dict | None = None
) -> np.ndarray:
    """
    Calculate the centroid of each cell in a loaded 2D mesh.

    Parameters:
        mesh (dict): Dictionary containing 'nodes' and 'cells'.
        mesh_name (str): Name of the mesh file.
        output_dir (str): Directory to save or load the centroid file.
        main_args (dict): Additional arguments for file saving.

    Returns:
        np.ndarray: Structured array containing element IDs and their centroids.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate centroids
    logger.info("Calculating centroids...")
    centroids_list = []
    nodes = {node["id"]: (node["x"], node["y"], node["z"]) for node in mesh["nodes"]}

    for elem in mesh["cells"]:
        nd1, nd2, nd3 = elem["nd1"], elem["nd2"], elem["nd3"]
        coords = np.array([nodes[nd1], nodes[nd2], nodes[nd3]])
        centroid = coords.mean(axis=0)
        centroids_list.append((elem["id"], *centroid))

    # Convert to structured array
    centroids = pd.DataFrame(centroids_list, columns=["id", "cx", "cy", "cz"])

    # Save centroids
    filepath = os.path.join(output_dir, f"{mesh_name}_cell_centroids.csv")
    logger.info(f"Saving centroids to {filepath}...")
    centroids.to_csv(filepath, index=False)

    return centroids


def main(data: dict, paths: str, user_opts: dict) -> dict:
    """
    Calculate the centroids of the cells in the mesh.

    Parameters:
        data (dict): Dictionary containing all the imported data, including mesh data.
        paths (dict): dictionary containing all relevant paths
        user_opts (dict): Main arguments from main.py.

    Returns:
        dict: Updated data dictionary with centroids.
    """
    logger.info("Starting centroid calculation.")

    # Extract mesh data
    mesh_paths = data.get("mesh", None)
    if mesh_paths is None:
        logger.error("No mesh data found in the input data.")
        raise ValueError("Mesh data is required for centroid calculation.")

    # Creates the centroids directory
    folder_centroids = paths["folder_centroids"]
    if not os.path.exists(folder_centroids):
        logger.info(f"Creating centroids directory at {folder_centroids}...")
        os.makedirs(folder_centroids, exist_ok=True)

    # Creates the output data structure for the centroids
    data["centroids"] = {}

    for mesh_name, filepath in mesh_paths.items():
        mesh_name = mesh_name.split(".")[0]
        centroid_file = os.path.join(folder_centroids, f"{mesh_name}_cell_centroids.csv")

        if os.path.exists(centroid_file):
            logger.info(f"Loading centroids from {centroid_file}...")
            centroids = pd.read_csv(centroid_file)
        else:
            logger.info(f"No centroids found for {mesh_name}. Calculating...")
            mesh = functions_io.read_mesh_file(filepath)
            centroids = calculate_centroids(mesh, mesh_name, output_dir=folder_centroids, main_args=user_opts)

        # Update mesh data structure with centroids
        data["centroids"][mesh_name] = centroids

    logger.info("Centroid calculation completed.")

    return data
