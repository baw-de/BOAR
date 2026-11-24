#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for BOAR (Bayesian Optimization for Automated Roughness calibration in two-dimensional hydrodynamic models).

This module orchestrates the calibration workflow by:
    - Loading user configuration from YAML files
    - Setting up logging for execution tracking
    - Finding the cases to calibrate: one (flat ground_truth/) or several (ground_truth/SCENARIO_DISCHARGE/)
    - For each case: importing the data, calculating mesh cell centroids and running the friction calibration
    - Writing the best friction vector of every case to output/calibration_summary.csv

Example:
    Run the calibration with default settings:

    >>> python boar.py

    Run with a custom log file:

    >>> python boar.py --log-file custom_log.log
"""

__version__ = "BOAR v.1.1.0"

# Standard library imports
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Import Basement-Calibrator modules
module_path = Path(__file__).parent
sys.path.insert(0, str(module_path))

from src import basement_tools, batch, functions_io


def setup_logger(filename) -> logging.Logger:
    """
    Set up and configure a logger for the application.

    Args:
        filename (str): Name of the log file to create.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        filename=filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following:
            - log_file (str): Name of the log file (default: 'boar.log').
    """
    parser = argparse.ArgumentParser(description="Run Basement-Calibrator.")
    parser.add_argument("--log-file", default="boar.log", help="Name of the log file to use.")
    return parser.parse_args()


def main(logger_filename: str = "boar.log") -> None:
    """
    Run the BOAR calibration workflow.

    This is the main script for the Basement-Calibrator, which orchestrates
    the entire calibration process. It performs the following steps:

        1. Loads user-defined options from a YAML configuration file.
        2. Sets up a logger to track the execution of the script.
        3. Logs the user options for reference.
        4. Finds the cases to calibrate (single mode or batch mode, see src/batch.py).
        5. For each case: prepares its run folder, imports the data, calculates cell centroids and
           optimizes the friction regions using Bayesian optimization.
        6. Writes the best vector of every case to output/calibration_summary.csv.

    Args:
        logger_filename (str): Name of the log file. Default is 'boar.log'.
    """
    # Load configuration file
    with open(r"user_defined_configs/user_options.yaml") as file:
        user_opt = yaml.full_load(file)

    # Setup logger
    logger = setup_logger(logger_filename)
    basement_version = user_opt.get("basement_options", {}).get(
        "basement_version", basement_tools.DEFAULT_BASEMENT_VERSION
    )
    logger.info(f"{__version__} | BASEMENT v.{basement_version}")
    logger.info("Starting main function")

    # Log user options
    logger.info("User arguments:")
    for key, value in user_opt.items():
        logger.info(f"\t{key}: {value}")

    # Find the cases: single mode (flat ground_truth/) or batch mode (one subfolder per scenario/discharge)
    folders = functions_io.default_folders()
    folders["PROJECT_ROOT"] = module_path
    cases = functions_io.discover_cases(folders["SIMULATION_CONFIG_FOLDER"], folders["GROUND_TRUTH_FOLDER"], logger)

    # Calibrate them (serially or in parallel, see batch_options) and write the summary CSV
    batch.run(cases, user_opt, folders)

    logger.info("Main function finished")


if __name__ == "__main__":
    args = parse_args()
    main(args.log_file)
