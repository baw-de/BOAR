#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import json
import logging
import os
import shutil
from dataclasses import dataclass
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
# Inflow boundaries whose discharge BOAR sets (template replication in batch mode and every evaluation)
INFLOW_BOUNDARY_TYPES = ("froude_in", "uniform_in")
# Lookup table (Code, discharge value) in DISCHARGE_FILES_FOLDER, needed for inflows with a constant discharge
DISCHARGE_DICTIONARY_FILE = "discharge_dictionary.csv"


@dataclass(frozen=True)
class Case:
    """
    One calibration of the queue.

    Attributes:
        name: Ground-truth subfolder name ``SCENARIO_DISCHARGE``; None in single mode.
        scenario: Scenario part of the name (None in single mode: read from the ground-truth file names).
        discharge: Discharge code part of the name, e.g. ``Q001`` (None in single mode).
        config_folder: Folder holding the BASEMENT configuration used for this case.
        ground_truth_folder: Folder holding the case's ground-truth CSV files.
        shared_config: True when the case replicates the flat template ``simulation_configs/model.json``.
        error: Why the case cannot run (it is reported as failed), or None.
    """

    name: str | None
    scenario: str | None
    discharge: str | None
    config_folder: Path
    ground_truth_folder: Path
    shared_config: bool = False
    error: str | None = None

    @property
    def batch(self) -> bool:
        """True for a case of batch mode (a ground-truth subfolder)."""
        return self.name is not None


def default_folders() -> dict[str, Path]:
    """
    The project folders, read from ``user_defined_configs/constants.py`` at call time.

    The dictionary is passed to batch workers, which therefore do not depend on module state of the parent
    process (Windows starts worker processes fresh).
    """
    return {
        "SIMULATION_CONFIG_FOLDER": Path(c.SIMULATION_CONFIG_FOLDER),
        "GROUND_TRUTH_FOLDER": Path(c.GROUND_TRUTH_FOLDER),
        "OUTPUT_FOLDER": Path(c.OUTPUT_FOLDER),
        "MESH_FOLDER": Path(c.MESH_FOLDER),
        "DISCHARGE_FILES_FOLDER": Path(c.DISCHARGE_FILES_FOLDER),
    }


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
                # A minus sign marks the last node of the string; keep that node (as its absolute id)
                node_ids = [abs(int(x)) for x in parts[1:-1]]
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
    config_folder: str | Path,
    output_root: str | Path,
    overwrite: bool,
    logger: logging.Logger,
    inflow: dict | None = None,
) -> Path:
    """
    Copy the original BASEMENT configuration into ``output_root / <simulation_name>`` and return that folder.

    The files in ``config_folder`` are only read. The name comes from ``SETUP.simulation_name`` in model.json. The JSON files are always copied fresh, so every run starts from the original friction and
    discharge. With ``overwrite`` the case folder is deleted first; without it, earlier results such as
    optimization_memory.csv stay and the run resumes from them.

    In the copied model.json relative ``mesh_file`` and ``discharge_file`` paths are made absolute, resolved
    against ``config_folder`` and then the project root. ``inflow`` (from :func:`resolve_inflow`) replaces the
    discharge of the inflow boundaries in the copy, as batch mode does for each case.

    Args:
        config_folder: Folder holding model.json, simulation.json and results.json.
        output_root: Folder in which the case folder is created.
        overwrite: Delete an existing case folder before copying.
        logger: Logger for progress and error messages.
        inflow: ``{"discharge_file": path}`` or ``{"discharge": value}`` for the inflow boundaries, or None.

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
    if inflow:
        if set_inflow_discharge(model_config, **inflow):
            logger.info(f"Inflow discharge of {case_folder / 'model.json'} set to {inflow}")
        else:
            logger.warning(f"{config_folder / 'model.json'} has no inflow boundary {INFLOW_BOUNDARY_TYPES}.")
    with open(case_folder / "model.json", "w") as f:
        json.dump(model_config, f, indent=4)

    logger.info(f"Simulation '{name}': originals in {config_folder}, run folder {case_folder}")
    return case_folder


def setup_directories(
    project_path,
    user_options: dict,
    logger: logging.Logger,
    folders: dict[str, Path] | None = None,
    case: "Case | None" = None,
    inflow: dict | None = None,
) -> dict:
    """
    Check the project folders and prepare the run folder of one case.

    Args:
        project_path: Root folder of the project.
        user_options: The user options (user_options.yaml).
        logger: Logger for progress and error messages.
        folders: Project folders (see :func:`default_folders`); default: read from constants.py.
        case: The case to prepare; None is the single-case layout (flat simulation_configs/ and ground_truth/).
        inflow: Inflow discharge for the copied model.json (see :func:`resolve_inflow`), or None.

    Returns:
        dict: The paths used by the calibration.
    """
    folders = folders or default_folders()
    folder_mesh = folders["MESH_FOLDER"]
    folder_centroids = os.path.join(folder_mesh, "mesh_centroids")
    folder_output = folders["OUTPUT_FOLDER"]
    folder_ground_truth = case.ground_truth_folder if case is not None else folders["GROUND_TRUTH_FOLDER"]
    folder_configs = case.config_folder if case is not None else folders["SIMULATION_CONFIG_FOLDER"]

    if not os.path.exists(folder_ground_truth):
        os.makedirs(folder_ground_truth)

        logger.error("ground_truth folder not found. Please populate the newly created directories accourdingly.")
        raise FileNotFoundError(
            "ground_truth folder not found. Please populate the newly created directories accourdingly."
        )

    overwrite = overwrite_option(user_options, logger)

    # Batch cases run in output/<SCENARIO_DISCHARGE>/<simulation_name>/
    output_root = Path(folder_output) / case.name if case is not None and case.name else Path(folder_output)
    folder_simulation = prepare_simulation_folder(folder_configs, output_root, overwrite, logger, inflow=inflow)

    folder_discharge = discharge_files_folder(user_options.get("simulation_options") or {}, folders, logger)

    return {
        "path_project": project_path,
        "folder_ground_truth": folder_ground_truth,
        "folder_simulation": str(folder_simulation),
        "folder_simulation_configs": folder_configs,
        "folder_mesh": folder_mesh,
        "folder_centroids": folder_centroids,
        "folder_output": folder_output,
        "folder_discharge": folder_discharge,
    }


def overwrite_option(user_options: dict, logger: logging.Logger) -> bool:
    """``general_options.overwrite`` (or the deprecated ``clear_start``); default False."""
    general_options = user_options.get("general_options", {})
    if "overwrite" in general_options:
        return bool(general_options["overwrite"])
    if "clear_start" in general_options:
        logger.warning("general_options 'clear_start' is deprecated; rename it to 'overwrite'.")
        return bool(general_options["clear_start"])
    return False


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
    output: dict = {}

    # Reads the meshes
    output["mesh"] = find_available_meshes(paths)

    # Reads ground truth data
    output["ground_truth_data"] = read_csv_files_in_folder(paths["folder_ground_truth"])

    # Reads the discharge dictionary if existing (needed only for inflows with a constant discharge)
    output["discharge_table"] = read_discharge_table(paths["folder_discharge"])

    logger.info("Data imported")

    return output


# ── Batch mode ───────────────────────────────────────────────────────────────────────────────────────
def set_inflow_discharge(model_config: dict, discharge_file: str | None = None, discharge: float | None = None) -> bool:
    """
    Set the discharge of the inflow boundaries (``froude_in``, ``uniform_in``) of a model.json dictionary.

    Exactly one of ``discharge_file`` and ``discharge`` is given; the other key is removed from the boundary.

    Args:
        model_config: The parsed model.json, changed in place.
        discharge_file: Path of the discharge time series.
        discharge: Constant discharge value.

    Returns:
        bool: True if at least one inflow boundary was found.
    """
    if (discharge_file is None) == (discharge is None):
        raise ValueError("Give either discharge_file or discharge.")
    boundaries = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
    found = False
    for bc in boundaries:
        if bc.get("type") in INFLOW_BOUNDARY_TYPES:
            found = True
            if discharge_file is not None:
                bc["discharge_file"] = discharge_file
                bc.pop("discharge", None)
            else:
                bc["discharge"] = discharge
                bc.pop("discharge_file", None)
    return found


def read_discharge_table(folder_discharge: str | Path) -> pd.DataFrame | None:
    """The discharge dictionary (Code, value) of the discharge folder, or None if it does not exist."""
    path = Path(folder_discharge) / DISCHARGE_DICTIONARY_FILE
    return pd.read_csv(path) if path.is_file() else None


_warned_legacy_directory = False
_warned_legacy_list = False


def discharge_files_folder(
    simulation_options: dict, folders: dict | None = None, logger: logging.Logger = logger
) -> Path:
    """
    Folder of the discharges: ``DISCHARGE_FILES_FOLDER`` (discharge_files/, with ``<CODE>.txt`` and the dictionary).

    The deprecated ``simulation_options.discharge_file_directory`` still overrides it, with a warning; a
    relative path is read from the project root.

    Args:
        simulation_options: ``simulation_options`` of the user options.
        folders: Project folders (see :func:`default_folders`); default: read from constants.py.
        logger: Logger for the deprecation warning.

    Returns:
        Path: The absolute folder.
    """
    global _warned_legacy_directory, _warned_legacy_list
    unused = [key for key in ("discharge_file", "discharge_file_list") if key in simulation_options]
    if unused and not _warned_legacy_list:
        logger.warning(
            f"simulation_options {unused} are no longer used and can be removed: the discharge codes come from "
            "the ground-truth file names (CASE_<CODE>_*.csv), and the inflow boundary of model.json tells whether "
            "the discharge is a time series (discharge_file) or a constant (discharge)."
        )
        _warned_legacy_list = True
    legacy = simulation_options.get("discharge_file_directory")
    if legacy not in (None, "", "None"):
        directory = Path(str(legacy))
        directory = directory if directory.is_absolute() else Path(c.PROJECT_ROOT) / directory
        if not _warned_legacy_directory:
            logger.warning(
                "simulation_options 'discharge_file_directory' is deprecated: put the discharge files in "
                f"discharge_files/ (DISCHARGE_FILES_FOLDER in constants.py). Using {directory}."
            )
            _warned_legacy_directory = True
        return directory.resolve()

    folders = folders or default_folders()
    return Path(folders.get("DISCHARGE_FILES_FOLDER") or c.DISCHARGE_FILES_FOLDER).resolve()


def inflow_discharge_kind(model_config: dict) -> str | None:
    """
    How model.json gives the discharge of its inflow boundaries (``froude_in``, ``uniform_in``).

    Returns:
        ``"file"`` when they hold a ``discharge_file`` (time series), ``"value"`` when they hold a constant
        ``discharge``, or None when model.json has no such inflow boundary (BOAR then leaves the discharge as it is).
    """
    try:
        boundaries = model_config["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
    except (KeyError, TypeError):
        return None
    inflows = [bc for bc in boundaries if isinstance(bc, dict) and bc.get("type") in INFLOW_BOUNDARY_TYPES]
    if any("discharge_file" in bc for bc in inflows):
        return "file"
    if any("discharge" in bc for bc in inflows):
        return "value"
    return None


def read_inflow_discharge_kind(config_folder: str | Path) -> str | None:
    """:func:`inflow_discharge_kind` of ``config_folder/model.json`` (None if it cannot be read)."""
    try:
        with open(Path(config_folder) / "model.json") as f:
            return inflow_discharge_kind(json.load(f))
    except (OSError, ValueError):
        return None


def resolve_inflow(
    discharge_code: str, kind: str | None, discharge_table: pd.DataFrame | None, discharge_folder: Path
) -> dict | None:
    """
    The inflow discharge of a discharge code, as BOAR's calibration sets it.

    ``kind`` comes from model.json (see :func:`inflow_discharge_kind`): ``"file"`` uses the time series
    ``<discharge_folder>/<code>.txt`` (see :func:`discharge_files_folder`), ``"value"`` the value of the code in
    the discharge dictionary, and None leaves the discharge of model.json unchanged (returns None).

    Returns:
        dict | None: ``{"discharge_file": absolute path}``, ``{"discharge": value}`` or None.

    Raises:
        FileNotFoundError: The discharge file does not exist.
        KeyError: The code is not in the discharge dictionary.
    """
    if kind is None:
        return None
    if kind == "file":
        path = (Path(discharge_folder) / f"{discharge_code}.txt").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Discharge file {path} not found.")
        return {"discharge_file": str(path)}

    if discharge_table is None:
        raise KeyError(
            f"model.json gives a constant inflow discharge, so {Path(discharge_folder) / DISCHARGE_DICTIONARY_FILE} "
            f"(columns Code and the discharge value) is needed to look up {discharge_code}."
        )
    match = discharge_table[discharge_table["Code"].astype(str) == str(discharge_code)]
    if match.empty:
        raise KeyError(f"Discharge code {discharge_code} not found in {DISCHARGE_DICTIONARY_FILE}.")
    return {"discharge": float(match.iloc[0, 1])}


def simulation_name(config_folder: str | Path) -> str | None:
    """``SETUP.simulation_name`` of a configuration's model.json (the run folder name), or None."""
    try:
        with open(Path(config_folder) / "model.json") as f:
            name = json.load(f).get("SETUP", {}).get(c.SIMULATION_NAME_KEY)
    except (OSError, ValueError):
        return None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _subfolders(folder: Path) -> list[Path]:
    """Visible subfolders, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(d for d in folder.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))


def discover_cases(config_root: str | Path, ground_truth_root: str | Path, logger: logging.Logger) -> list[Case]:
    """
    Find the calibrations to run.

    Single mode: ``ground_truth/`` has no subfolders; one case with the flat layout (as before batch mode).
    Batch mode: every subfolder ``ground_truth/SCENARIO_DISCHARGE/`` is a case. Its configuration is
    ``simulation_configs/SCENARIO_DISCHARGE/`` when that folder has a model.json, otherwise the flat template
    ``simulation_configs/model.json``, replicated with the case's discharge.

    Args:
        config_root: The simulation_configs folder.
        ground_truth_root: The ground_truth folder.
        logger: Logger for warnings and errors.

    Returns:
        list[Case]: The cases sorted by name; cases that cannot run carry an ``error``.
    """
    config_root = Path(config_root)
    ground_truth_root = Path(ground_truth_root)
    case_folders = _subfolders(ground_truth_root)

    if not case_folders:
        return [Case(None, None, None, config_root, ground_truth_root)]

    template = (config_root / "model.json").is_file()
    cases = []
    for folder in case_folders:
        name = folder.name
        scenario, _, discharge = name.rpartition("_")
        own = config_root / name
        error = None
        if not scenario or not discharge:
            error = f"folder name '{name}' is not SCENARIO_DISCHARGE (e.g. S1_Q001)"
        elif not any(f.suffix.lower() == ".csv" for f in folder.iterdir() if f.is_file()):
            error = f"no ground-truth CSV files in {folder}"

        if (own / "model.json").is_file():
            case = Case(name, scenario or None, discharge or None, own, folder, False, error)
        elif template:
            case = Case(name, scenario or None, discharge or None, config_root, folder, True, error)
        else:
            error = error or f"no {own / 'model.json'} and no template {config_root / 'model.json'}"
            case = Case(name, scenario or None, discharge or None, own, folder, False, error)
        cases.append(case)

        if case.error:
            logger.error(f"Case {name}: {case.error}")
        else:
            codes = {f.stem.split("_")[1] for f in folder.glob("*.csv") if len(f.stem.split("_")) > 1}
            if codes and codes != {case.discharge}:
                logger.warning(
                    f"Case {name}: the ground-truth file names carry discharge codes {sorted(codes)}, "
                    f"the folder name {case.discharge}. The calibration runs the codes of the file names."
                )

    names = {f.name for f in case_folders}
    for extra in _subfolders(config_root):
        if (extra / "model.json").is_file() and extra.name not in names:
            logger.warning(f"simulation_configs/{extra.name}/ has no ground_truth/{extra.name}/ and is skipped.")
    return cases


def discharge_codes(keys) -> list[str]:
    """Discharge codes of ground-truth file names ``CASE_<CODE>_*.csv``, in order of appearance."""
    codes: list[str] = []
    for key in keys:
        parts = Path(key).stem.split("_")
        if len(parts) > 1 and parts[1] not in codes:
            codes.append(parts[1])
    return codes


def scenario_discharge_from_ground_truth(keys) -> tuple[str, str]:
    """
    Scenario and discharge of the single-case layout, from file names ``CASE_Q001_*.csv``.

    Several values are joined with ``+`` (e.g. ``Q001+Q002``), in order of appearance.
    """
    scenarios: list[str] = []
    codes: list[str] = []
    for key in keys:
        parts = Path(key).stem.split("_")
        if parts[0] not in scenarios:
            scenarios.append(parts[0])
        if len(parts) > 1 and parts[1] not in codes:
            codes.append(parts[1])
    return "+".join(scenarios), "+".join(codes)


SUMMARY_BASE_COLUMNS = ("scenario", "discharge")
SUMMARY_TAIL_COLUMNS = ("best_error", "n_iter_best", "device", "status")


def write_summary_row(summary_path: str | Path, row: dict) -> pd.DataFrame:
    """
    Add or replace the row of one case in the calibration summary CSV.

    Args:
        summary_path: Path of the summary CSV (created if missing).
        row: ``scenario``, ``discharge``, ``best_vector``, ``region_names``, ``best_error``, ``n_iter_best``,
            ``device`` and ``status``. The best vector is written as one column per region,
            ``friction_<region>``. A row with the same scenario and discharge is replaced.

    Returns:
        pd.DataFrame: The summary table as written.
    """
    summary_path = Path(summary_path)
    record: dict = {"scenario": row.get("scenario"), "discharge": row.get("discharge")}
    vector = row.get("best_vector")
    if vector is not None:
        names = [str(n) for n in (row.get("region_names") or [])]
        if len(names) != len(vector):
            names = [str(i) for i in range(len(vector))]
        for name, value in zip(names, vector, strict=True):
            record[f"friction_{name}"] = float(value)
    for key in SUMMARY_TAIL_COLUMNS:
        record[key] = row.get(key)

    new = pd.DataFrame([record])
    if summary_path.is_file():
        table = pd.read_csv(summary_path, dtype={"scenario": str, "discharge": str})
        same = (table["scenario"].astype(str) == str(record["scenario"])) & (
            table["discharge"].astype(str) == str(record["discharge"])
        )
        table = table[~same]
        table = new if table.empty else pd.concat([table, new], ignore_index=True)
    else:
        table = new

    friction = [col for col in table.columns if col.startswith("friction_")]
    others = [col for col in table.columns if col not in (*SUMMARY_BASE_COLUMNS, *friction, *SUMMARY_TAIL_COLUMNS)]
    table = table[[*SUMMARY_BASE_COLUMNS, *friction, *SUMMARY_TAIL_COLUMNS, *others]]
    table = table.sort_values(["scenario", "discharge"], kind="stable").reset_index(drop=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_path, index=False)
    return table
