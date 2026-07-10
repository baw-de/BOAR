#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for BOAR (Bayesian Optimization for Automated Roughness calibration in two-dimensional hydrodynamic models).

This module orchestrates the calibration workflow by:
    - Loading user configuration from YAML files
    - Setting up logging for execution tracking
    - Importing simulation and observation data
    - Calculating mesh cell centroids
    - Running the friction calibration process

Example:
    Run the calibration with default settings:

    >>> python boar.py

    Run with a custom log file:

    >>> python boar.py --log-file custom_log.log
"""

__version__ = "BOAR v.1.0.0"

# Standard library imports
import argparse
import logging
import sys
from pathlib import Path

# Third-party imports
import yaml

# Import Basement-Calibrator modules
module_path = Path(__file__).parent
sys.path.insert(0, str(module_path))

from src import cell_centroid, friction_calibration, functions_io


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
        3. Creates necessary directories for input and output data.
        4. Logs the user options for reference.
        5. Imports simulation and observation data from CSV files.
        6. Calculates cell centroids from simulation data.
        7. Optimizes the friction region using Bayesian optimization.

    Args:
        logger_filename (str): Name of the log file. Default is 'boar.log'.
    """
    # Load configuration file
    with open(r"user_defined_configs/user_options.yaml") as file:
        user_opt = yaml.safe_load(file)

    # Setup logger
    logger = setup_logger(logger_filename)
    logger.info("Starting main function")

    dict_paths = functions_io.setup_directories(module_path, user_opt, logger)

    # Log user options
    logger.info("User arguments:")
    for key, value in user_opt.items():
        logger.info(f"\t{key}: {value}")

    # Import CSV
    data = functions_io.main(dict_paths)

    # Calculate cell centroids
    data = cell_centroid.main(data, dict_paths, user_opt)

    # Optimize the friction region
    data = friction_calibration.main(data, dict_paths, user_opt)

    logger.info("Main function finished")


if __name__ == "__main__":
    args = parse_args()
    main(args.log_file)
