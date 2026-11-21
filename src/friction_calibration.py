#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import ast
import csv
import logging
import os
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src import basement_tools as bmt
from src import bo_optimizer as bo
from src import functions_io, functions_sampler, utils

# Import user defined functions
from user_defined_configs import function_loss as lf


class FrictionCalibration:
    """
    Class to handle the calibration process.
    """

    def __init__(
        self, calibration_data: dict, simulation_folder: str, user_options: dict, logger: "logging.Logger | None" = None
    ):
        # General options
        self.logger = logger or logging.getLogger(__name__)
        self.calibration_data = calibration_data
        self.sim_folder = simulation_folder

        # Logging options
        self.silent = user_options["general_options"].get("silent", True)
        self.log_dev = user_options["general_options"].get("log_dev", False)

        # Creates as simulation manager instance
        self.sim_manager = bmt.SimulationManager(
            simulation_folder,
            user_options["basement_options"],
            logger=self.logger,
            silent=self.silent,
            log_dev=self.log_dev,
        )
        self.mesh_file = self.sim_manager.get_computational_mesh_name()
        self.save_freq = self.sim_manager._get_save_freq()

        # Extract optimization variable arguments
        self.opt_args = self._optimization_var_args(user_options)

    def _quantize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Snap a vector to the exact Decimal grid defined by bounds and precision."""
        quantized = np.asarray(vector, dtype=float).copy()
        bounds = self.opt_args.get("bounds", None)
        precision = self.opt_args.get("precision", None)

        if bounds is None or precision is None or precision <= 0:
            return quantized

        step_dec = Decimal(str(precision))
        for i, (lb, ub) in enumerate(bounds):
            lb_dec = Decimal(str(lb))
            ub_dec = Decimal(str(ub))

            raw = float(np.clip(quantized[i], lb, ub))
            raw_dec = Decimal(str(raw))

            k = ((raw_dec - lb_dec) / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            snapped = lb_dec + k * step_dec

            if snapped < lb_dec:
                snapped = lb_dec
            elif snapped > ub_dec:
                snapped = ub_dec

            quantized[i] = float(snapped)

        return quantized

    def _optimization_var_args(
        self,
        main_args: dict,
    ) -> dict:
        """
        Extracts and processes the optimization variable user options

        Parameters:
            main_args (dict): Dictionary containing the main arguments.

        Returns:
            dict: Dictionary containing the optimization arguments.
        """
        # Extracts the user arguments
        args_optimization_var = main_args["optimization_variable_options"]

        # Extract optimization variable arguments
        initial_vector_type = args_optimization_var.get("initial_vector", "file")
        target_regions = args_optimization_var.get("regions", None)

        if initial_vector_type == "file":
            vector_0 = self.sim_manager.define_friction(
                self.calibration_data["mesh"], read_model_friction=True, target_region=target_regions
            )
            initial_vector = vector_0["friction"]

        elif type(initial_vector_type) in [float, int]:
            initial_vector = np.array([initial_vector_type])
            _ = self.sim_manager.define_friction(
                self.calibration_data["mesh"], friction=initial_vector, target_region=target_regions
            )  # Updates the friction in the simulation with the initial vector

        elif isinstance(initial_vector_type, list):
            initial_vector = np.array([value for _, value in initial_vector_type])
            _ = self.sim_manager.define_friction(
                self.calibration_data["mesh"], friction=initial_vector, target_region=target_regions
            )  # Updates the friction in the simulation with the initial vector

        ## Constraints
        constraints = args_optimization_var.get("constraints", None)
        constraint_variables = None
        if constraints is not None and constraints != "None":
            expression = constraints.get("expression", None)
            variables = constraints.get("variables", None)
            if expression is None or variables is None:
                utils.write_log(self.logger, "Constraints must include 'expression' and 'variables' keys.", "error")
                raise ValueError("Constraints must include 'expression' and 'variables' keys.")

            constraints = utils.create_constraint_function(expression, variables)
            constraint_variables = variables

        ## Bounds
        bounds = args_optimization_var.get("bounds", None)
        if isinstance(bounds, list):
            if len(bounds) == 1:
                bounds = [bounds[0]] * len(initial_vector)
            else:
                # Check if the bounds match the number of regions
                if len(bounds) != len(initial_vector):
                    utils.write_log(self.logger, "Number of bounds must match the number of friction regions.", "error")
                    raise ValueError("Number of bounds must match the number of friction regions.")

        else:
            bounds = None

        save_tried_vectors = args_optimization_var.get("save_tried_vectors", False)

        ## Optimization precision
        precision = args_optimization_var.get("precision", 1)
        if isinstance(precision, str):
            precision = None if precision.strip().lower() in ("none", "null", "") else float(precision)

        ## Basement parameters
        args_basement = main_args["basement_options"]

        ## Surrogate model options
        args_surrogate = main_args["surrogate_model_options"]

        tolerance = args_surrogate.get("tolerance", 1e-5)
        if isinstance(tolerance, str):
            tolerance = float(tolerance)

        # Test population: None (default) derives it from bounds, precision and constraints
        test_population = args_surrogate.get("test_population", None)
        if isinstance(test_population, str):
            test_population = None if test_population.strip().lower() in ("none", "null", "") else test_population
        if test_population is not None:
            test_population = int(float(test_population))

        max_test_population = int(float(args_surrogate.get("max_test_population", bo.DEFAULT_MAX_TEST_POPULATION)))
        if max_test_population <= 0:
            utils.write_log(self.logger, "'max_test_population' must be a positive integer.", "error")
            raise ValueError("'max_test_population' must be a positive integer.")

        # Export the optimization variable arguments
        output = {
            # General arguments
            "vector_0": vector_0,
            "friction_values_0": initial_vector,
            "bounds": bounds,
            "constraints": constraints,
            "constraint_variables": constraint_variables,
            "regions": target_regions,
            "precision": precision,
            "save_tried_vectors": save_tried_vectors,
            ## Basement arguments
            "basement_path": self.sim_manager.basement_path,
            "nthreads": args_basement.get("nthreads", -1),
            "backend": args_basement.get("backend", "omp"),
            "basement_version": args_basement.get("basement_version", bmt.DEFAULT_BASEMENT_VERSION),
            "save_errors": args_optimization_var.get("save_errors", False),
            ## Surrogate model options
            "GPR_iterations": args_surrogate.get("GPR_iterations", 500),
            "GPR_alpha": float(args_surrogate.get("GPR_alpha", 1e-6)),
            "EI_exploration-exploitation": args_surrogate.get("EI_exploration-exploitation", 0.01),
            "tolerance": tolerance,
            "opt_mem_override": args_surrogate.get("opt_mem_override", False),
            "n_initial": args_surrogate.get("n_initial", 5),
            "max_no_improvement": args_surrogate.get("max_no_improvement", 100),
            "test_population": test_population,
            "max_test_population": max_test_population,
            "max_tested_vectors": args_surrogate.get("max_tested_vectors", 100),
            ## Simulation arguments
            "simulation_arguments": main_args.get("simulation_options") or {},
        }

        return output

    def opt_memory(
        self,
        vector: np.ndarray | None,
        error: float = 0.0,
        search: bool = False,
        initial: bool = False,
    ) -> tuple | int | None:
        """
        Saves the optimization memory to a file. This includes the
        friction vector and the corresponding calibration error.

        This function will either create or append the memory file.

        The function is called before the optimization process starts
        and loads the results from the memory file if it exists. Additionally,
        the memory file is updated after each optimization iteration.

        Args:
            vector (np.array): Array containing the friction values.
            error (float): The calibration error.
            search (bool): If True, the function will search for the memory file.
            initial (bool): If True, the function will search the memory file and return as a solution if
                            the file is longer than the number of samples.
            override (bool): Override the first memory check. This allows the creation of new LHS samples.

        Returns:
            tuple: If initial is True and the memory file exists, returns a tuple containing the friction vectors and their corresponding calibration errors.
            int: If search is True and the memory file does not exist, returns 0.
            None: If search is False and the memory file does not exist, returns None.
        """
        # Define the path to the memory file and check if it exists
        memory_file_path = os.path.join(self.sim_folder, "optimization_memory.csv")
        file_exists = os.path.isfile(memory_file_path)
        if vector is not None:
            vector = self._quantize_vector(vector)

        # Load the memory data if the file exists
        if initial and file_exists:
            with open(memory_file_path) as file:
                reader = csv.reader(file)
                memory_data = list(reader)[1:]  # Skip the header row

            # Check if memory file is larger than the required number of samples
            if not self.opt_args["opt_mem_override"]:
                # Initialize friction and error arrays
                vector = np.empty((len(memory_data), len(ast.literal_eval(memory_data[0][0]))))
                error = np.empty(len(memory_data))

                # Populate the arrays
                for i, row in enumerate(memory_data):
                    vector[i] = self._quantize_vector(np.array(ast.literal_eval(row[0])))
                    error[i] = float(row[1])

                return vector, error

        elif search and not file_exists:
            with open(memory_file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Friction Vector", "Calibration Error"])
            return 0

        elif search and file_exists:
            with open(memory_file_path) as file:
                reader = csv.reader(file)
                memory_data = list(reader)

            # Search for the friction vector in the memory data
            for i, row in enumerate(memory_data):
                if i == 0:
                    continue
                # Converts the string to a list
                mem_vector = self._quantize_vector(np.array(ast.literal_eval(row[0])))
                if np.isclose(vector, mem_vector).all():
                    error = float(row[1])

                    return error

            return 0

        else:
            memory_data = []

        # Update the memory file with the new data
        with open(memory_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Friction Vector", "Calibration Error"])
            local_friction = self._quantize_vector(vector).tolist()
            local_error = np.round(error, 16)
            writer.writerow([local_friction, local_error])

    def objective_function(self, vector: np.ndarray, verbose: bool = True, rerun: bool = False) -> float:
        """
        Optimization function to minimize the calibration error.

        Args:
            vector (np.ndarray): Array containing the friction values.
            verbose (bool): If True, the function will print the current friction vector and the corresponding calibration error.
            rerun (bool): Simulate the vector even if the optimization memory holds it, without adding it to the
                memory again (used to leave the results of the best vector in the run folder).

        Returns:
            float: The calibration error.
        """
        # Unpack the optimization arguments
        friction_0 = self.opt_args["vector_0"]
        regions = self.opt_args["regions"]

        # Snap friction values to the exact optimization grid
        vector = self._quantize_vector(vector)

        # Log the friction values
        if verbose:
            utils.write_log(self.logger, f"Current vector (rounded): {vector}", "info")

        # Search the optimization memory for the current friction values (a rerun simulates anyway)
        val_error = 0 if rerun else self.opt_memory(vector, search=True)

        # If the error is not found in the memory, run the simulation
        if val_error != 0:
            utils.write_log(self.logger, f"Calibration error: {val_error}", "info")
            return val_error

        # Rebuild the structured friction array
        updated_vector = np.zeros_like(friction_0)
        updated_vector["region"] = friction_0["region"]
        updated_vector["friction"] = vector

        # Update the friction values in the simulation
        self.sim_manager.define_friction(
            self.calibration_data["mesh"], updated_vector, target_region=regions, define_in_h5=True
        )

        # Run the simulation for all discharges of the ground truth
        total_validation_error = []

        # Inflow of each discharge code, resolved before the first run (see main): a time series, a constant, or
        # None when model.json has no uniform_in/froude_in inflow (its discharge is then left as it is)
        for discharge_code, inflow in self.opt_args["inflows"].items():
            if inflow is not None and "discharge_file" in inflow:
                self.sim_manager.define_discharge(inflow["discharge_file"], file=True)
            elif inflow is not None:
                self.sim_manager.define_discharge(inflow["discharge"])

            self.sim_manager.run_setup(self.opt_args)
            run_status = self.sim_manager.run_simulation(self.opt_args)
            res_status = self.sim_manager.run_results_processing(self.opt_args)

            if run_status == 1:
                utils.write_log(self.logger, "Mass conservation error detected. Returning high error value.", "error")
                val_error = 1e6  # Return a high error value for mass conservation errors
                if not rerun:
                    self.opt_memory(vector, val_error)
                return val_error

            # Extract simulation data
            sim_data = self.sim_manager.extract_sim_data(self.calibration_data["centroids"], sim_time=-1)

            # Calculate the calibration error
            ground_truth_data = []
            for key, val in self.calibration_data["ground_truth_data"].items():
                if discharge_code in key:
                    ground_truth_data.append(val)

            obj_error, errors = lf.loss_function(sim_data, ground_truth_data)
            total_validation_error.append(obj_error)

            # Creates a copy of the results.h5 file for each tried vector
            if res_status is None and self.opt_args["save_tried_vectors"]:
                # Creates a folder to store the tried vectors
                os.makedirs(os.path.join(self.sim_folder, "tried_vectors"), exist_ok=True)
                shutil.copyfile(
                    os.path.join(self.sim_folder, "results.h5"),
                    os.path.join(self.sim_folder, "tried_vectors", f"{vector}_{discharge_code}_results.h5"),
                )
                shutil.copyfile(
                    os.path.join(self.sim_folder, "results_aux.h5"),
                    os.path.join(self.sim_folder, "tried_vectors", f"{vector}_{discharge_code}_results_aux.h5"),
                )

            # Saves the error comparison points for all tried vectors
            if self.opt_args["save_errors"]:
                error_df = pd.DataFrame(errors, columns=["Error"])

                os.makedirs(os.path.join(self.sim_folder, "tried_vectors"), exist_ok=True)
                error_df.to_csv(
                    os.path.join(self.sim_folder, "tried_vectors", f"{vector}_{discharge_code}_pts_error.csv"),
                    index=False,
                )

        # Condenses all errors
        val_error = sum(total_validation_error)

        # Log the calibration error
        if verbose:
            utils.write_log(self.logger, f"Friction vector: {vector}; Loss: {val_error}", "info")

        # Write the trial and error to a csv file (a rerun is already in it)
        if not rerun:
            self.opt_memory(vector, val_error)

        return val_error

    def initialize_process(self, sampler):
        """
        Initializes the optimization process.
        """
        # Log the start of the optimization process
        utils.write_log(self.logger, "Starting optimization process...", "info")

        try:
            lhs_samples, lhs_values = self.opt_memory(
                None, initial=True
            )  # Fails if file does not exist, however it creates an empty one
        except Exception:
            lhs_samples = np.empty((0, len(self.opt_args["friction_values_0"])))
            lhs_values = np.empty(0)

        # Test the user provided initial vector
        init_vector = self.opt_args["friction_values_0"]
        for i, value in enumerate(init_vector):
            if value < self.opt_args["bounds"][i][0] or value > self.opt_args["bounds"][i][1]:
                utils.write_log(
                    self.logger, "Initial vector is outside the bounds. Using the bounds instead.", "warning"
                )
                init_vector = np.clip(init_vector, self.opt_args["bounds"][i][0], self.opt_args["bounds"][i][1])

        # Evaluate the initial vector unless the memory (of an earlier run) already holds it
        quantized_init = self._quantize_vector(init_vector)
        if any(np.isclose(quantized_init, value).all() for value in lhs_samples):
            utils.write_log(
                self.logger, "Initial vector is already in the memory. Using the stored value instead.", "info"
            )
        else:
            init_error = self.objective_function(init_vector)
            lhs_samples = np.vstack([lhs_samples, quantized_init])
            lhs_values = np.append(lhs_values, init_error)

        if len(lhs_samples) < self.opt_args["n_initial"]:
            utils.write_log(self.logger, "Stored samples are lower than the minimum. Generating new LHS samples...")
            remaining_samples = self.opt_args["n_initial"] - len(lhs_samples)

            new_samples = sampler.extend_samples(lhs_samples, remaining_samples, only_new=True)
            new_samples_values = np.array([self.objective_function(x) for x in new_samples])

            lhs_samples = np.vstack([lhs_samples, new_samples])
            lhs_values = np.append(lhs_values, new_samples_values)

        return lhs_samples, lhs_values


def best_from_memory(sim_folder: str | os.PathLike) -> tuple[list[float], float, int]:
    """
    Best friction vector of a calibration, read from its optimization_memory.csv.

    Every evaluation of both engines (boar, optuna) is appended to the memory in the order it was run, so the
    row number of the best error is the number of evaluations until the best error was found.

    Args:
        sim_folder: Run folder holding optimization_memory.csv.

    Returns:
        tuple: (best vector, best error, n_iter_best), where n_iter_best is the 1-based row of the first
        minimum error (a resumed run also counts the evaluations of the earlier runs).

    Raises:
        FileNotFoundError: No memory file.
        ValueError: The memory file holds no evaluation.
    """
    memory_file_path = os.path.join(sim_folder, "optimization_memory.csv")
    with open(memory_file_path) as file:
        rows = list(csv.reader(file))[1:]
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError(f"{memory_file_path} holds no evaluation.")
    errors = np.array([float(row[1]) for row in rows])
    best = int(np.argmin(errors))
    vector = [float(v) for v in ast.literal_eval(rows[best][0])]
    return vector, float(errors[best]), best + 1


def resume_state(errors, n_initial: int, opt_args: dict) -> tuple[int, int, bool]:
    """
    Where an earlier run of the "boar" engine stopped, from the errors in its optimization memory.

    The first ``n_initial`` evaluations are the initial samples; every later one is one iteration of the
    Bayesian optimization. The stopping rules of ``BayesianOptimizer.optimize`` are replayed on them.

    Args:
        errors: Calibration errors of the memory, in evaluation order.
        n_initial: Number of initial samples.
        opt_args: Optimization options (max_tested_vectors, max_no_improvement, tolerance).

    Returns:
        tuple: (iterations done, iterations without improvement at the end, finished).
    """
    errors = [float(e) for e in errors]
    if len(errors) <= n_initial:
        return 0, 0, False
    best = min(errors[:n_initial])
    no_improvement = 0
    iterations = errors[n_initial:]
    for done, error in enumerate(iterations, start=1):
        if error <= opt_args["tolerance"]:
            return done, no_improvement, True
        if error < best:
            best, no_improvement = error, 0
        else:
            no_improvement += 1
        if no_improvement >= opt_args["max_no_improvement"]:
            return done, no_improvement, True
    done = len(iterations)
    return done, no_improvement, done >= opt_args["max_tested_vectors"]


def _region_names(calibrator: "FrictionCalibration", n_values: int) -> list[str]:
    """Names of the calibrated regions: the ``regions`` option, else the region ids of the model."""
    regions = calibrator.opt_args.get("regions")
    if regions and len(regions) == n_values:
        return [str(r) for r in regions]
    vector_0 = calibrator.opt_args.get("vector_0")
    if vector_0 is not None and len(vector_0) == n_values:
        return [str(r) for r in vector_0["region"]]
    return [str(i) for i in range(n_values)]


def main(data: dict, paths: dict, user_opts: dict) -> dict:
    """
    This function orchestrates the calibration process. It initializes the calibrator class, sets up the sampler and the Bayesian optimizer, and runs the calibration.

    Parameters:
        data (dict): Dictionary containing the simulation data.
        paths (dict): dictionary containing all relevant paths
        user_opts (dict): The user options.

    Returns:
        dict: ``best_vector``, ``best_error``, ``n_iter_best`` (see :func:`best_from_memory`) and
        ``region_names``.
    """
    # Create a logger
    logger = logging.getLogger(__name__)

    # Initialization
    folder_simulation = paths["folder_simulation"]

    # general_options.overwrite decides what is kept: True deletes the run folder (prepare_simulation_folder),
    # False keeps it to resume; each simulation writes a new results.h5 anyway.
    if "cleanup" in user_opts.get("basement_options", {}):
        utils.write_log(
            logger,
            "basement_options 'cleanup' is no longer used and can be removed: general_options.overwrite decides "
            "whether earlier results are deleted.",
            "warning",
        )

    # Creates the calibrator instance
    calibrator = FrictionCalibration(data, folder_simulation, user_opts, logger)
    # Inflow of every discharge code of the ground truth, of the kind model.json gives (time series or constant).
    # Resolved here, so a missing <CODE>.txt or dictionary entry stops the run before any simulation.
    discharge_folder = paths.get("folder_discharge") or functions_io.discharge_files_folder(
        user_opts.get("simulation_options") or {}, logger=logger
    )
    kind = functions_io.inflow_discharge_kind(calibrator.sim_manager._load_model_config())
    table = data.get("discharge_table") if kind == "value" else None
    calibrator.opt_args["inflows"] = {
        code: functions_io.resolve_inflow(code, kind, table, discharge_folder)
        for code in functions_io.discharge_codes(data["ground_truth_data"].keys())
    }
    descriptions = {
        "file": f"time series <CODE>.txt from {discharge_folder}",
        "value": f"constant value of <CODE> from {Path(discharge_folder) / functions_io.DISCHARGE_DICTIONARY_FILE}",
        None: "model.json has no uniform_in/froude_in inflow, so its discharge is not changed",
    }
    utils.write_log(logger, f"Inflow discharge: {descriptions[kind]}; codes {list(calibrator.opt_args['inflows'])}")

    # Generate setup.h5 from model.json (with the initial friction written above) before the first iteration,
    # which writes friction into setup.h5. Uses the BASEMENT version set in basement_options.
    utils.write_log(logger, "Generating setup.h5 from model.json.")
    calibrator.sim_manager.run_setup()

    if user_opts["optimization_variable_options"].get("opt_engine", "boar") == "boar":
        sampler = functions_sampler.LatinHypercube(
            bounds=calibrator.opt_args["bounds"],
            precision=calibrator.opt_args["precision"],
            seed=user_opts["sampling_options"].get("seed", None),
            constraint_fns=[calibrator.opt_args["constraints"]],
            logger=logger,
            silent=calibrator.silent,
            log_dev=calibrator.log_dev,
        )

        initialization = calibrator.initialize_process(sampler)

        bo_method = bo.BayesianOptimizer(
            initial_samples=initialization,
            obj_func=calibrator.objective_function,
            sampler=sampler,
            opt_args=calibrator.opt_args,
            logger=logger,
        )

        # An interrupted calibration continues its budget (the memory holds the evaluations in order)
        start_iteration, no_improvement, finished = resume_state(
            initialization[1], calibrator.opt_args["n_initial"], calibrator.opt_args
        )
        best_sample, _ = bo_method.optimize(
            start_iteration=start_iteration, no_improvement=no_improvement, finished=finished
        )

    elif user_opts["optimization_variable_options"].get("opt_engine", "boar") == "optuna":
        bounds = calibrator.opt_args.get("bounds", None)
        if bounds is None:
            raise ValueError("Bounds must be provided when using Optuna optimization.")

        precision = calibrator.opt_args["precision"]
        seed = user_opts["sampling_options"].get("seed", None)
        n_trials = calibrator.opt_args.get("max_tested_vectors", 200)
        n_initial = calibrator.opt_args.get("n_initial", 5)
        user_constraint = calibrator.opt_args.get("constraints", None)

        def objective(trial: optuna.Trial) -> float:
            vector = np.array(
                [trial.suggest_float(f"x{i}", lb, ub, step=precision) for i, (lb, ub) in enumerate(bounds)],
                dtype=float,
            )

            return calibrator.objective_function(vector, verbose=False)

        def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
            vector = np.array([trial.params[f"x{i}"] for i in range(len(bounds))], dtype=float)
            return (0.0 if user_constraint(vector) else 1.0,)

        if user_constraint is not None:
            sampler = optuna.samplers.GPSampler(
                seed=seed, n_startup_trials=n_initial, constraints_func=constraints_func
            )
        else:
            sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_initial)

        # Route Optuna's log output through the application logger instead of the console
        optuna.logging.disable_default_handler()
        optuna_logger = optuna.logging.get_logger("optuna")
        _log = logger
        while _log:
            for handler in _log.handlers:
                optuna_logger.addHandler(handler)
            if not _log.propagate:
                break
            _log = _log.parent

        study = optuna.create_study(direction="minimize", sampler=sampler)

        initial_vector = np.asarray(calibrator.opt_args["friction_values_0"], dtype=float)

        # Load optimization memory and warm-start the study with all prior trials
        memory_file_path = os.path.join(calibrator.sim_folder, "optimization_memory.csv")
        initial_in_memory = False

        if os.path.isfile(memory_file_path):
            with open(memory_file_path) as _mem_f:
                memory_rows = list(csv.reader(_mem_f))[1:]  # skip header

            if memory_rows:
                distributions: dict[str, optuna.distributions.BaseDistribution] = {
                    f"x{i}": optuna.distributions.FloatDistribution(lb, ub, step=precision)
                    for i, (lb, ub) in enumerate(bounds)
                }

                for row in memory_rows:
                    mem_vector = calibrator._quantize_vector(np.array(ast.literal_eval(row[0]), dtype=float))
                    mem_error = float(row[1])
                    mem_params = {f"x{i}": float(mem_vector[i]) for i in range(len(bounds))}

                    mem_user_attrs: dict = {}
                    mem_system_attrs: dict = {}
                    if user_constraint is not None:
                        is_feasible = bool(user_constraint(mem_vector))
                        mem_constraints = (0.0 if is_feasible else 1.0,)
                        mem_user_attrs["constraints"] = mem_constraints
                        mem_system_attrs["constraints"] = mem_constraints

                    prior_trial = optuna.trial.create_trial(
                        params=mem_params,
                        distributions=distributions,
                        value=mem_error,
                        user_attrs=mem_user_attrs,
                        system_attrs=mem_system_attrs,
                    )
                    study.add_trial(prior_trial)

                    if np.isclose(mem_vector, initial_vector).all():
                        initial_in_memory = True

        if not initial_in_memory:
            enqueued_params = {}
            for i, (lb, ub) in enumerate(bounds):
                raw_value = float(np.clip(initial_vector[i], lb, ub))

                if precision is not None and precision > 0:
                    lb_dec = Decimal(str(lb))
                    ub_dec = Decimal(str(ub))
                    step_dec = Decimal(str(precision))
                    val_dec = Decimal(str(raw_value))

                    # Snap to the exact grid defined by low + k * step to match Optuna's distribution.
                    k = ((val_dec - lb_dec) / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    val_dec = lb_dec + k * step_dec

                    if val_dec < lb_dec:
                        val_dec = lb_dec
                    elif val_dec > ub_dec:
                        val_dec = ub_dec

                    enqueued_params[f"x{i}"] = float(val_dec)
                else:
                    enqueued_params[f"x{i}"] = raw_value

            study.enqueue_trial(enqueued_params)

        remaining_trials = n_trials - len(study.trials)
        study.optimize(objective, n_trials=remaining_trials)

        best_trial = study.best_trial

        best_sample = np.array([best_trial.params[f"x{i}"] for i in range(len(bounds))], dtype=float)

        utils.write_log(logger, f"Optuna best sample: {best_sample}", "info")
        utils.write_log(logger, f"Optuna best value: {best_trial.value}", "info")

    else:
        raise ValueError(
            f"Invalid optimization engine specified: {user_opts['optimization_variable_options'].get('opt_engine', 'boar')}. Supported engines are 'boar' and 'optuna'."
        )

    best_vector, best_error, n_iter_best = best_from_memory(folder_simulation)
    utils.write_log(logger, f"Best vector {best_vector}, error {best_error}, found at evaluation {n_iter_best}.")

    # The run folder holds the results of the last simulated vector: simulate the best one once more, so that
    # results.h5 (and results_aux.h5, results.xdmf) belong to it. The memory is not changed.
    utils.write_log(logger, f"Running the best vector {best_vector} once more for the results in the run folder.")
    rerun_error = calibrator.objective_function(np.asarray(best_vector, dtype=float), rerun=True)
    if not np.isclose(rerun_error, best_error, rtol=1e-6, atol=1e-12):
        utils.write_log(
            logger,
            f"The rerun of the best vector gave error {rerun_error}, the memory {best_error}: the simulation is not "
            "reproducible (the summary keeps the memory value).",
            "warning",
        )
    return {
        "best_vector": best_vector,
        "best_error": best_error,
        "n_iter_best": n_iter_best,
        "region_names": _region_names(calibrator, len(best_vector)),
    }
