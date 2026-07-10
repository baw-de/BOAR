#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASEMENT interface module for simulation management.

This module provides the SimulationManager class which handles:
    - Simulation setup and execution via BASEMENT software
    - Results processing and data extraction
    - Friction and discharge parameter management

Example:
    >>> from src.basement_tools import SimulationManager
    >>> manager = SimulationManager(sim_folder='simulation/', run_parameters={})
    >>> manager.run_simulation()
"""

if __name__ == "__main__":
    raise Exception("This file must be run as a module")

import json
import logging
import os
import subprocess  # nosec B404 – required for BASEMENT CLI invocation
from typing import Any

import h5py
import numpy as np

# Import Basement-Calibrator modules
from src import functions_io, utils
from user_defined_configs import constants as c


class SimulationManager:
    """
    A class to manage simulation setup, execution, data extraction, and friction definition.

    Attributes:
        sim_folder (str): Path to the simulation folder.
        run_parameters (dict): Simulation run parameters.
        logger (logging.Logger): Logger instance for tracking operations.
        basement_path (str): Path to BASEMENT installation.
    """

    def __init__(
        self,
        sim_folder: str,
        run_parameters: dict,
        logger: logging.Logger | None = None,
        silent: bool = False,
        log_dev: bool = False,
    ):
        """
        Initialize the SimulationManager.

        Args:
            sim_folder: Path to the simulation folder.
            run_parameters: Dictionary containing the simulation parameters.
            logger: Logger instance for logging messages (default: None).
            silent: If True, suppresses all logging output.
            log_dev: If True, enables development logging.
        """
        self.sim_folder = sim_folder
        self.silent = silent
        self.log_dev = log_dev
        self.run_parameters = run_parameters
        self.mesh_file = self.get_computational_mesh_name()
        self.save_freq = self._get_save_freq()

        if logger is None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.basement_path = c.BASEMENT_PATH

    def get_computational_mesh_name(self) -> str:
        """
        Get the name of the computational mesh file.

        Reads the model configuration and extracts the mesh filename.

        Returns:
            str: The basename of the mesh file, or empty string if not found.
        """
        with open(os.path.join(self.sim_folder, "model.json")) as f:
            model_config = json.load(f)
            mesh_file = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["GEOMETRY"]["mesh_file"]

        mesh_file = os.path.basename(mesh_file)
        return mesh_file

    def _get_save_freq(self) -> int:
        """
        Get the saving frequency from the simulation configuration.

        Returns:
            int: The saving frequency in seconds.
        """
        with open(os.path.join(self.sim_folder, "simulation.json")) as f:
            simulation_config = json.load(f)
            save_freq = simulation_config["SIMULATION"]["TIME"]["out"]
        return save_freq

    def _load_model_config(self) -> dict[str, Any]:
        """
        Load the model configuration from the JSON file.

        Returns:
            dict: The model configuration dictionary.
        """
        model_config_path = os.path.join(self.sim_folder, "model.json")
        with open(model_config_path) as f:
            model_config = json.load(f)
        return model_config

    def _validate_paths(self, paths: list[str], error_message: str | None = None, raise_error: bool = True) -> bool:
        """
        Validate the existence of required file paths.

        Args:
            paths: List of file paths to validate.
            error_message: Custom error message if validation fails.
            raise_error: Whether to raise an error if validation fails.

        Returns:
            bool: True if all paths are valid, False otherwise.

        Raises:
            FileNotFoundError: If a path does not exist and raise_error is True.
        """
        invalid_paths = [path for path in paths if not os.path.exists(path)]
        if invalid_paths:
            error_message = error_message or "One or more required paths are invalid."
            utils.write_log(self.logger, f"{error_message} Invalid paths: {invalid_paths}", level="error")
            if raise_error:
                raise FileNotFoundError(f"{error_message} {invalid_paths}")
            return False
        return True

    def load_simulation_setup(self) -> dict[str, Any]:
        """
        Load the simulation setup from the simulation folder.

        Reads the setup.h5 file and extracts mesh name and friction data.

        Returns:
            dict: Dictionary containing mesh_name and friction array.
        """
        sim_setup_file = os.path.join(self.sim_folder, "setup.h5")

        with h5py.File(sim_setup_file, "r") as f:
            friction = f["CellsAll"]["Friction"][()]

        sim_setup = {"mesh_name": self.mesh_file, "friction": friction}
        return sim_setup

    def run_setup(
        self,
        run_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Process the simulation setup using BASEMENT software.

        Args:
            run_parameters: Dictionary of simulation parameters (optional).

        Raises:
            FileNotFoundError: If required files are missing.
            Exception: If setup processing fails.
        """
        # Define paths
        exe_results_path = os.path.join(self.basement_path, "BMv4_setup.exe")
        model_file = os.path.join(self.sim_folder, "model.json")
        setup_file = os.path.join(self.sim_folder, "setup.h5")

        # Validate paths
        utils.write_log(self.logger, "Validating required files.", silent=not self.log_dev)
        required_files = [exe_results_path, setup_file, model_file]
        self._validate_paths(required_files, error_message="Required simulation files are missing.")

        # Process results if required
        try:
            results_command = [exe_results_path, model_file, "--output", setup_file]
            utils.write_log(self.logger, f"Setup command: {results_command}", level="debug", silent=not self.log_dev)

            utils.write_log(self.logger, "Starting setup...", silent=not self.log_dev)
            proc_results = subprocess.run(results_command, shell=False, capture_output=True, text=True)  # nosec B603 – args are validated file paths from os.path.join/config

            if proc_results.stdout:
                utils.write_log(self.logger, proc_results.stdout.strip(), level="info", silent=not self.log_dev)
            if proc_results.stderr:
                utils.write_log(self.logger, proc_results.stderr.strip(), level="error", silent=not self.log_dev)

            if proc_results.returncode != 0:
                utils.write_log(
                    self.logger,
                    f"Setup failed with return code {proc_results.returncode}. Command: {results_command}",
                    level="error",
                )
                raise Exception("Setup failed. Check the log for details.")

            utils.write_log(self.logger, "Setup completed successfully.", silent=not self.log_dev)

        except Exception:
            raise

    def run_simulation(
        self,
        run_parameters: dict[str, Any] | None = None,
    ) -> int | None:
        """
        Execute a simulation using the BASEMENT software.

        Args:
            run_parameters: Dictionary of simulation parameters (optional).

        Raises:
            FileNotFoundError: If required files are missing.
            Exception: If simulation fails.
        """
        # Extract parameters
        backend = self.run_parameters.get("backend", "omp")
        nthreads = self.run_parameters.get("nthreads", -1)

        if nthreads == -1:
            nthreads = os.cpu_count()

        if backend in ["cuda", "cudaC"]:
            nthreads = 1

        # Start the simulation
        utils.write_log(
            self.logger, f"Running simulation with backend: {backend} and nthreads: {nthreads}", silent=not self.log_dev
        )

        # Define paths
        exe_sim_path = os.path.join(self.basement_path, "BMv4_simulation.exe")
        run_config_file = os.path.join(self.sim_folder, "simulation.json")
        setup_file = os.path.join(self.sim_folder, "setup.h5")
        output_file = os.path.join(self.sim_folder, "results.h5")

        # Validate paths
        utils.write_log(self.logger, "Validating required files.", silent=not self.log_dev)
        required_files = [exe_sim_path, run_config_file, setup_file]
        self._validate_paths(required_files, error_message="Required simulation files are missing.")

        # Simulation command
        sim_command = [
            exe_sim_path,
            run_config_file,
            setup_file,
            "--output",
            output_file,
            "--backend",
            backend,
            "--nthreads",
            str(nthreads),
        ]
        utils.write_log(self.logger, f"Simulation command: {sim_command}", level="debug", silent=not self.log_dev)

        # Run the simulation process
        try:
            utils.write_log(self.logger, "Starting simulation...", silent=not self.log_dev)
            with subprocess.Popen(sim_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:  # nosec B603 – args are validated file paths from os.path.join/config
                stdout, stderr = proc.communicate()

                if stdout:
                    utils.write_log(self.logger, stdout.decode().strip(), level="info", silent=not self.log_dev)
                if stderr:
                    utils.write_log(self.logger, stderr.decode().strip(), level="error")

                if proc.returncode != 0:
                    if "MASS not conserved" in stderr.decode():
                        utils.write_log(
                            self.logger,
                            "Simulation failed due to mass conservation error. Check the log for details.",
                            level="error",
                        )
                        return 1

                    utils.write_log(
                        self.logger,
                        f"Simulation failed with return code {proc.returncode}. Command: {sim_command}",
                        level="error",
                    )
                    raise Exception("Simulation failed. Check the log for details.")

            utils.write_log(self.logger, "Simulation completed successfully.", silent=not self.log_dev)

        except Exception as e:
            utils.write_log(self.logger, f"Simulation error: {e}", level="exception")
            raise

    def run_results_processing(
        self,
        run_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Process simulation results using the BASEMENT software.

        Args:
            run_parameters: Dictionary of simulation parameters (optional).

        Raises:
            FileNotFoundError: If required files are missing.
            Exception: If results processing fails.
        """
        # Define paths
        exe_results_path = os.path.join(self.basement_path, "BMv4_results.exe")
        setup_file = os.path.join(self.sim_folder, "setup.h5")
        output_file = os.path.join(self.sim_folder, "results.h5")
        results_config_file = os.path.join(self.sim_folder, "results.json")
        results_output_file = os.path.join(self.sim_folder, "results.xdmf")

        # Validate paths
        utils.write_log(self.logger, "Validating required files.", silent=not self.log_dev)
        required_files = [exe_results_path, setup_file, output_file, results_config_file]
        self._validate_paths(required_files, error_message="Required simulation files are missing.")

        # Process results if required
        try:
            results_command = [exe_results_path, results_config_file, output_file, "--output", results_output_file]
            utils.write_log(self.logger, f"Results command: {results_command}", level="debug", silent=not self.log_dev)

            utils.write_log(self.logger, "Starting results processing...", silent=not self.log_dev)
            proc_results = subprocess.run(results_command, shell=False, capture_output=True, text=True)  # nosec B603 – args are validated file paths from os.path.join/config

            if proc_results.stdout:
                utils.write_log(self.logger, proc_results.stdout.strip(), level="info", silent=not self.log_dev)
            if proc_results.stderr:
                # Look for common error
                error_ignore = " -> Reading results from file failed: Accessing the HDF5 dataset 'Parameters/MeanGrainSize' failed."
                error_lines = proc_results.stderr.split("\n")
                if any(line != error_ignore for line in error_lines if line != ""):
                    utils.write_log(self.logger, proc_results.stderr.strip(), level="error", silent=not self.log_dev)

            if proc_results.returncode != 0:
                utils.write_log(
                    self.logger,
                    f"Results processing failed with return code {proc_results.returncode}. Command: {results_command}",
                    level="error",
                )
                raise Exception("Results processing failed. Check the log for details.")

            utils.write_log(self.logger, "Results processing completed successfully.", silent=not self.log_dev)

        except Exception as e:
            utils.write_log(
                self.logger, f"An error occurred during simulation or results processing: {e}", level="error"
            )
            raise

    def extract_sim_data(self, centroids_dict: dict, sim_time: int = -1) -> dict[str, Any]:
        """
        Extract calibration data from a simulation folder.

        The results are stored in an HDF5 file under the following structure::

            results.h5
            └── RESULTS
                └── CellsAll
                    └── Saving times
                        └── Results array (wse, ux, uy)

        Water surface elevation (wse) is stored in the first column. To calculate
        water depth, subtract bed elevation from wse. Bed elevation is located in::

            results.h5
            └── CellsAll
                └── BottomEl

        Args:
            centroids_dict: Dictionary containing mesh centroids indexed by mesh name.
            sim_time: Simulation time to extract data from. Default is -1 (last timestep).

        Returns:
            dict: Dictionary containing extracted simulation data with keys:
                - time: Simulation time in seconds
                - centroids: Cell centroids array
                - hyd: Hydrodynamic data (wse, qx, qy, h, ux, uy)
                - rey: Reynolds state data
                - trb: Turbulence state data
                - turb_k: Turbulent kinetic energy
                - turb_rey: Turbulent Reynolds stress

        Raises:
            FileNotFoundError: If results file is not found.
            ValueError: If specified simulation time not found in data.
        """
        results_file = os.path.join(self.sim_folder, "results.h5")
        results_aux_file = os.path.join(self.sim_folder, "results_aux.h5")
        if not os.path.exists(results_file):
            utils.write_log(self.logger, f"Results file not found: {results_file}", level="error")
            raise FileNotFoundError(f"Results file not found: {results_file}")

        try:
            with h5py.File(results_file, "r") as h5file:
                cells_all = h5file["RESULTS"]["CellsAll"]

                # Extract bottom elevation
                bottom_el = np.array(h5file["CellsAll"]["BottomEl"]).flatten()
                utils.write_log(self.logger, f"Bottom elevation shape: {bottom_el.shape}", silent=not self.log_dev)

                # Process saving times and extract relevant data
                saving_times_ids = list(cells_all["HydState"].keys())
                saving_times = [int(t) * self.save_freq for t in saving_times_ids]

                if sim_time == -1:
                    time = saving_times_ids[-1]
                else:
                    if sim_time not in saving_times:
                        utils.write_log(
                            self.logger, f"Simulation time {sim_time} not found in saving times.", level="error"
                        )
                        raise ValueError(f"Simulation time {sim_time} not found in saving times.")
                    time = saving_times_ids[saving_times.index(sim_time)]

                results_hyd = np.array(cells_all["HydState"][time])  # Hydrodynamic state
                results_rey = np.array(cells_all["ReyState"][time])  # Reynolds state
                results_trb = np.array(cells_all["TrbState"][time])  # Turbulence state

            with h5py.File(results_aux_file, "r") as h5file:
                # Process saving times and extract relevant data
                saving_times_ids = list(h5file["flow_velocity"].keys())
                saving_times = [int(t) * self.save_freq for t in saving_times_ids]
                if sim_time == -1:
                    time = saving_times_ids[-1]
                else:
                    if sim_time not in saving_times:
                        utils.write_log(
                            self.logger, f"Simulation time {sim_time} not found in saving times.", level="error"
                        )
                        raise ValueError(f"Simulation time {sim_time} not found in saving times.")
                    time = saving_times_ids[saving_times.index(sim_time)]
                results_velocity = np.array(h5file["flow_velocity"][time])  # Hydrodynamic state
                results_turb_k = np.array(h5file["turb_k"][time])  # Reynolds state
                results_turb_rey = np.array(h5file["turb_reynolds"][time])  # Turbulence state

        except Exception as e:
            utils.write_log(self.logger, f"Error accessing or processing results file: {e}", level="error")
            raise

        # Calculate water depth and velocity
        try:
            # Calculate water depth, ux, and uy
            water_depth = results_hyd[:, 0] - bottom_el  # Assuming WSE is column 0
            ux = results_velocity[:, 0]  # Assuming Qx is column 1
            uy = results_velocity[:, 1]  # Assuming Qy is column 2

            # Define the structured array dtype
            hyd_dtype = np.dtype(
                [
                    ("wse", "f8"),
                    ("qx", "f8"),
                    ("qy", "f8"),  # Original columns
                    ("h", "f8"),
                    ("ux", "f8"),
                    ("uy", "f8"),  # New calculated columns
                ]
            )

            # Construct the structured array
            results_hyd_structured = np.zeros(len(results_hyd), dtype=hyd_dtype)
            results_hyd_structured["wse"] = results_hyd[:, 0]
            results_hyd_structured["qx"] = results_hyd[:, 1]
            results_hyd_structured["qy"] = results_hyd[:, 2]
            results_hyd_structured["h"] = water_depth
            results_hyd_structured["ux"] = ux
            results_hyd_structured["uy"] = uy

        except Exception as e:
            utils.write_log(self.logger, f"Error calculating water depth: {e}", level="error")
            raise

        # Extract mesh centroids
        cell_centroids = centroids_dict.get(self.mesh_file.replace(".2dm", ""), None)

        # Construct the final data dictionary
        data = {
            "time": int(time) * self.save_freq,
            "centroids": cell_centroids,
            "hyd": results_hyd_structured,
            "rey": results_rey,
            "trb": results_trb,
            "turb_k": results_turb_k,
            "turb_rey": results_turb_rey,
        }

        utils.write_log(self.logger, "Data extraction completed.", silent=not self.log_dev)

        return data

    def define_friction(
        self,
        mesh_filepath: dict[str, str],
        friction: np.ndarray = np.empty(0),
        target_region: list | None = None,
        read_model_friction: bool = False,
        define_in_h5: bool = False,
    ) -> np.ndarray:
        """
        Define or update friction values for the simulation based on mesh regions.

        Args:
            mesh_filepath: Dictionary mapping mesh file names to their file paths.
            friction: Existing friction array if updating (default: None).
            target_region: List of region names to update (default: None).
            read_model_friction: If True, return model friction without applying (default: False).
            define_in_h5: If True, update friction values in the HDF5 file (default: False).

        Returns:
            np.ndarray: Updated or initialized friction array.

        Raises:
            Exception: For errors during file reading, processing, or saving data.
        """

        # Define auxiliary functions
        def load_model_friction(model_config: dict, target_region: list | None = None) -> np.ndarray:
            """
            Initialize the friction array based on the model configuration.

            Args:
                model_config: The model configuration dictionary.
                target_region: List of region names to update (default: None).

            Returns:
                np.ndarray: The initialized friction array.
            """
            regions = np.array(
                [
                    (d["index"][0], d["name"])
                    for d in model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["GEOMETRY"]["REGIONDEF"]
                ],
                dtype=[("index", "i4"), ("name", "U64")],
            )
            default_friction = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["FRICTION"]["regions"]

            # Filter only the target regions
            if target_region is not None:
                regions = regions[np.isin(regions["name"], target_region)]
                default_friction = [d for d in default_friction if d["region_name"] in regions["name"]]

            # Use float64 to avoid introducing float32 artifacts before serialization.
            friction_array = np.empty(len(regions), dtype=[("region", "i2"), ("friction", "f8")])
            for i, region in enumerate(regions):
                reg_id = region["index"]
                reg_name = region["name"]
                friction_array[i]["region"] = reg_id
                friction_array[i]["friction"] = next(
                    (d["friction"] for d in default_friction if d["region_name"] == reg_name), 0.0
                )

            return friction_array

        def update_friction_in_h5(self, new_friction_values: np.ndarray) -> None:
            """
            Update the friction values in the HDF5 file.

            Args:
                new_friction_values: The new friction values to update.
            """
            sim_setup_file = os.path.join(self.sim_folder, "setup.h5")

            try:
                with h5py.File(sim_setup_file, "r+") as f:
                    if "CellsAll" in f and "Friction" in f["CellsAll"]:
                        dataset = f["CellsAll"]["Friction"]
                        reshaped_values = new_friction_values.reshape(-1, 1)
                        if dataset.shape == reshaped_values.shape:
                            dataset[...] = reshaped_values
                            if self.log_dev:
                                utils.write_log(self.logger, "Updated friction values in HDF5 successfully.")
                        else:
                            utils.write_log(
                                self.logger,
                                f"Shape mismatch: Dataset {dataset.shape}, new data {reshaped_values.shape}",
                                level="error",
                            )
                            raise ValueError(
                                f"Shape mismatch: Dataset {dataset.shape}, new data {reshaped_values.shape}"
                            )
                    else:
                        utils.write_log(
                            self.logger, "Dataset 'CellsAll/Friction' not found in HDF5 file.", level="error"
                        )
                        raise KeyError("Dataset 'CellsAll/Friction' not found in HDF5 file.")
            except Exception as e:
                utils.write_log(
                    self.logger, f"Unidenfied error while saving friction values to HDF5: {e}", level="error"
                )
                raise

        def update_model_config(self, model_config: dict, updated_friction: np.ndarray) -> None:
            """
            Update the model configuration with new friction values.

            Args:
                model_config: The model configuration dictionary.
                updated_friction: The updated friction values.
            """

            def _clean_binary_float(value: float, max_decimals: int = 10) -> float:
                val = float(value)
                # Snap tiny binary artifacts (e.g. 43.29999923706055) to the nearest short decimal representation.
                for decimals in range(max_decimals + 1):
                    rounded = round(val, decimals)
                    tolerance = max(1e-12, 10 ** (-(decimals + 6)))
                    if abs(val - rounded) <= tolerance:
                        return float(rounded)
                return val

            regions = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["GEOMETRY"]["REGIONDEF"]
            region_mapping = {r["index"][0]: r["name"] for r in regions}
            target_shape = [
                {"friction": _clean_binary_float(row["friction"]), "region_name": region_mapping[row["region"]]}
                for row in updated_friction
            ]
            model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["FRICTION"]["regions"] = target_shape
            model_config_path = os.path.join(self.sim_folder, "model.json")
            try:
                with open(model_config_path, "w") as f:
                    json.dump(model_config, f, indent=4)
                if self.log_dev:
                    utils.write_log(self.logger, "Saved updated model configuration to JSON successfully.")
            except Exception as e:
                utils.write_log(self.logger, f"Error saving model configuration: {e}", level="error")
                raise

        # Load the model configuration
        model_config = self._load_model_config()
        model_friction_setup = load_model_friction(model_config)

        if read_model_friction:
            return load_model_friction(model_config, target_region)

        # Update the friction array
        mesh = functions_io.read_mesh_file(mesh_filepath[self.mesh_file])
        new_friction = np.zeros(mesh["cells"].size, dtype="f")

        for line in model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["GEOMETRY"]["REGIONDEF"]:
            region_id = line["index"][0]
            region_cells = np.where(mesh["cells"]["region"] == region_id)[0]
            if target_region is not None and line["name"] not in target_region:
                friction_idx = np.where(model_friction_setup["region"] == region_id)[0]
                new_friction[region_cells] = model_friction_setup["friction"][friction_idx]
            else:
                friction_idx = np.where(friction["region"] == region_id)[0]
                new_friction[region_cells] = friction["friction"][friction_idx]

        # Updates the friction in the file
        if target_region is not None:
            for i, region in enumerate(model_friction_setup):
                if region["region"] in friction["region"]:
                    # find the index of the region in the friction array
                    friction_idx = np.where(friction["region"] == region["region"])[0]

                    if len(friction_idx) != 1:
                        raise ValueError(f"Expected 1 match for region {region['region']}, got {friction_idx}")

                    model_friction_setup[i]["friction"] = float(friction["friction"][friction_idx[0]])

        else:
            model_friction_setup = friction

        # Save updates
        if define_in_h5:
            update_friction_in_h5(self, new_friction)
        update_model_config(self, model_config, model_friction_setup)

    def define_discharge(self, new_discharge: float | str, file: bool = False) -> None:
        """
        Define or update the discharge boundary condition.

        Args:
            new_discharge: The new discharge value or file path.
            file: If True, new_discharge is treated as a file path (default: False).
        """

        model_config = self._load_model_config()
        model_config_path = os.path.join(self.sim_folder, "model.json")

        boundaries = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]

        possible_bcs = ["froude_in", "uniform_in"]

        for ii, bc_dict in enumerate(boundaries):
            # Check boundary type
            if bc_dict["type"] in possible_bcs:
                if file:
                    bc_dict["discharge_file"] = new_discharge
                    try:
                        del bc_dict["discharge"]
                    except KeyError:
                        pass

                else:
                    bc_dict["discharge"] = new_discharge
                    try:
                        del bc_dict["discharge_file"]
                    except KeyError:
                        pass

        try:
            with open(model_config_path, "w") as f:
                json.dump(model_config, f, indent=4)
            if self.log_dev:
                utils.write_log(self.logger, "Saved updated model configuration to JSON successfully.")
        except Exception as e:
            utils.write_log(self.logger, f"Error saving model configuration: {e}", level="error")
            raise
