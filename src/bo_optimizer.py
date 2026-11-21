#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian Optimization module using Gaussian Processes and Latin Hypercube Sampling.

This module provides the BayesianOptimizer class for performing Bayesian
optimization to find optimal friction parameters for hydraulic simulations.

Example:
    >>> import numpy as np
    >>> from src.bo_optimizer import BayesianOptimizer
    >>> optimizer = BayesianOptimizer(initial_samples, obj_func, sampler, opt_args)
    >>> best_point, best_value = optimizer.optimize()
"""

if __name__ == "__main__":
    raise Exception("This file must be run as a module")

# Standard library imports
import logging
from collections.abc import Callable
from typing import cast

# Third-party imports
import numpy as np
import scipy.optimize as opt
import sklearn.gaussian_process.kernels as gpr_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from skopt.acquisition import gaussian_ei

# Import Basement-Calibrator modules
from src import utils

# Default cap on the auto-sized test population (surrogate_model_options -> max_test_population)
DEFAULT_MAX_TEST_POPULATION = 100_000

# Largest raw grid (before constraints) that is enumerated point by point when test_population is None.
# Larger grids are sampled instead. Raised to max_test_population if the user sets a larger cap.
GRID_ENUMERATION_LIMIT = 1_000_000


def build_candidate_population(
    sampler,
    test_population: int | str | None = None,
    max_test_population: int = DEFAULT_MAX_TEST_POPULATION,
    seed: int | None = None,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Build the fixed candidate population on which the acquisition function is evaluated.

    With an explicit ``test_population`` the sampler draws that many samples. When it is None, the size is
    derived from the bounds, precision and constraints: the full feasible precision grid is used, or a random
    subset of ``max_test_population`` points of it when it is larger. Grids too large to enumerate, and
    continuous spaces (no precision), are sampled with ``max_test_population`` points.

    Args:
        sampler: Sampler with ``generate_samples`` and, for the automatic size, ``grid_size`` and
            ``enumerate_grid`` (see ``functions_sampler.LatinHypercube``).
        test_population: Number of candidates, or None (or "None") for the automatic size.
        max_test_population: Cap on the automatic size.
        seed: Seed for the random subset of an oversized feasible grid.
        logger: Logger instance (optional).

    Returns:
        np.ndarray: Candidate points of shape (n_candidates, n_dim).

    Raises:
        ValueError: If max_test_population is not positive, or no grid point satisfies the constraints.
    """
    logger = logger or logging.getLogger(__name__)

    if isinstance(test_population, str) and test_population.strip().lower() in ("none", "null", ""):
        test_population = None

    if test_population is not None:
        test_population = int(test_population)
        utils.write_log(logger, f"Test population set by the user: {test_population} candidates.", "info")
        return cast(np.ndarray, sampler.generate_samples(test_population))

    max_test_population = int(max_test_population)
    if max_test_population <= 0:
        raise ValueError("max_test_population must be a positive integer.")

    grid_size = sampler.grid_size() if hasattr(sampler, "grid_size") else None
    enumeration_limit = max(GRID_ENUMERATION_LIMIT, max_test_population)

    if grid_size is not None and grid_size <= enumeration_limit:
        candidates = sampler.enumerate_grid()
        n_feasible = len(candidates)
        if n_feasible == 0:
            utils.write_log(logger, "No grid point satisfies the constraints within the bounds.", "error")
            raise ValueError("No grid point satisfies the constraints within the bounds.")

        if n_feasible > max_test_population:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_feasible, size=max_test_population, replace=False)
            candidates = candidates[np.sort(idx)]
            utils.write_log(
                logger,
                f"Test population: random subset of {max_test_population} of the {n_feasible} feasible grid points "
                f"(capped by max_test_population).",
                "info",
            )
        else:
            utils.write_log(logger, f"Test population: all {n_feasible} feasible grid points.", "info")
        return cast(np.ndarray, candidates)

    reason = "no precision is given" if grid_size is None else f"the grid has {grid_size} points"
    utils.write_log(logger, f"Test population: {max_test_population} sampled candidates ({reason}).", "info")
    return cast(np.ndarray, sampler.generate_samples(max_test_population))


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Processes with Latin Hypercube Sampling.

    This class implements Bayesian optimization for finding optimal parameters
    using Gaussian Process Regression as a surrogate model.

    Attributes:
        obj_func (Callable): Objective function to minimize.
        sampler (Callable): Sampler for generating candidate points.
        opt_args (dict): Optimization configuration arguments.
        sample (np.ndarray): Current sample points.
        value (np.ndarray): Objective function values at sample points.
        scaler (StandardScaler): Scaler for normalizing samples.
        gp (GaussianProcessRegressor): Gaussian Process regressor model.
        logger (logging.Logger): Logger instance for tracking progress.

    Example:
        >>> initial_samples = (np.array([[1.0], [2.0]]), np.array([1.5, 2.5]))
        >>> optimizer = BayesianOptimizer(initial_samples, objective_func, sampler, opts)
        >>> best_point, best_value = optimizer.optimize()
    """

    def __init__(
        self,
        initial_samples: np.ndarray,
        obj_func: Callable,
        sampler: Callable,
        opt_args: dict,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the Bayesian Optimizer.

        Args:
            initial_samples: Tuple of (sample_points, objective_values).
            obj_func: Objective function to minimize.
            sampler: Sampler for generating candidate points.
            opt_args: Dictionary of optimization arguments.
            logger: Logger instance for tracking progress (optional).
        """
        self.obj_func = obj_func
        self.sampler = sampler
        self.opt_args = opt_args
        self.logger = logger or logging.getLogger(__name__)

        # Initialize the kernel
        self.sample, self.value = initial_samples

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.sample_scaled = self.x_scaler.fit_transform(self.sample)
        self.value_scaled = self.y_scaler.fit_transform(self.value.reshape(-1, 1)).ravel()

        kernel = gpr_kernels.Matern(length_scale=1.0, nu=2.5)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.opt_args["GPR_iterations"],
            alpha=self.opt_args["GPR_alpha"],
            normalize_y=False,
            optimizer=self._optimizer(),
        )
        self.gp.fit(self.sample_scaled, self.value_scaled)

        utils.write_log(self.logger, "BayesianOptimizer initialized.", "info")

    def _optimizer(self) -> Callable:
        """
        Create a custom optimizer for GP hyperparameter tuning.

        Returns:
            Callable: Custom optimizer function using L-BFGS-B.
        """

        def l_bfgs_b_modified(obj_func, initial_theta, bounds):
            """
            Custom optimizer using L-BFGS-B for GP hyperparameter optimization.

            Args:
                obj_func: The objective function to be minimized.
                initial_theta: Initial guess for the optimization parameters.
                bounds: The bounds on the optimization parameters.

            Returns:
                Tuple: (theta_opt, func_min) - Optimized parameters and minimum value.
            """
            # Perform optimization with dual_annealing
            result = opt.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 1e4, "ftol": 1e-8, "gtol": 1e-6},
            )

            # Return optimized parameters and the minimum function value
            theta_opt = result.x
            func_min = result.fun
            return theta_opt, func_min

        optimizer_type = self.opt_args.get("optimizer", None)

        if optimizer_type is None or optimizer_type == "l_bfgs_b":
            optimizer_callable = l_bfgs_b_modified
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer_callable

    def optimize(self, return_attempted_points=False, start_iteration=0, no_improvement=0, finished=False):
        """
        Perform Bayesian Optimization to find optimal parameters.

        This method iteratively:
            1. Predicts candidate points using the Gaussian Process surrogate.
            2. Evaluates the objective function at candidate points.
            3. Updates the GP model with new observations.
            4. Stops when tolerance is reached or max iterations exceeded.

        A calibration resumed from its optimization memory continues its budget: ``start_iteration`` iterations of
        ``max_tested_vectors`` are already done and ``no_improvement`` of them since the last improvement (see
        ``friction_calibration.resume_state``); ``finished`` skips the search.

        Args:
            return_attempted_points: If True, return all attempted points.
            start_iteration: Iterations already done in an earlier run of this calibration.
            no_improvement: Iterations without improvement at the end of the earlier run.
            finished: The earlier run already stopped (budget, tolerance or no improvement).

        Returns:
            Tuple: (best_point, best_value) or (best_point, best_value, attempted_points)
                - best_point (np.ndarray): Optimal parameter values found.
                - best_value (float): Objective function value at best_point.
                - attempted_points (list): All points tried during optimization (optional).

        Raises:
            ValueError: If max_test_population is not positive, or no grid point satisfies the constraints.

        Example:
            >>> best_params, best_score = optimizer.optimize()
            >>> best_params, best_score, all_points = optimizer.optimize(return_attempted_points=True)
        """
        utils.write_log(self.logger, "Starting optimization.", "info")

        # Extract best point and value
        idx_best = np.argmin(self.value)
        best_point = self.sample[idx_best]
        best_value = self.value[idx_best]

        utils.write_log(self.logger, f"Initial best point: {best_point}, value: {best_value}", "info")

        # Start counters (continued from an earlier run when resuming)
        attempted_points = []
        no_improvement_iterations = no_improvement

        # Generate fixed candidate population (auto-sized from bounds, precision and constraints when None)
        X_candidates = build_candidate_population(
            self.sampler,
            test_population=self.opt_args.get("test_population", None),
            max_test_population=self.opt_args.get("max_test_population", DEFAULT_MAX_TEST_POPULATION),
            seed=getattr(self.sampler, "seed", None),
            logger=self.logger,
        )

        # Extract the Exploration-Exploitation parameter for Expected Improvement
        xi = self.opt_args.get("EI_exploration-exploitation", 0.01)

        # Resume: the seed advances twice per iteration, as in the uninterrupted run
        if start_iteration or no_improvement or finished:
            if self.sampler.seed is not None:
                self.sampler.seed += 2 * start_iteration
            state = "already finished" if finished else f"{no_improvement} without improvement"
            utils.write_log(
                self.logger,
                f"Resuming the optimization: {start_iteration} of {self.opt_args['max_tested_vectors']} iterations "
                f"done in an earlier run ({state}).",
                "info",
            )

        i = start_iteration - 1
        iterations = range(0) if finished else range(start_iteration, self.opt_args["max_tested_vectors"])
        for i in iterations:
            utils.write_log(self.logger, f"Iteration {i + 1} of {self.opt_args['max_tested_vectors']}.", "info")

            if self.sampler.seed is not None:
                self.sampler.seed += 1

            # Refit scalers on the CURRENT training data
            self.sample_scaled = self.x_scaler.fit_transform(self.sample)
            self.value_scaled = self.y_scaler.fit_transform(self.value.reshape(-1, 1)).ravel()

            X_candidates_scaled = self.x_scaler.transform(X_candidates)

            # Fit GP on current data
            self.gp.fit(self.sample_scaled, self.value_scaled)

            # Compute EI using the best scaled value observed so far
            y_opt = np.min(self.value_scaled)

            ei = gaussian_ei(X_candidates_scaled, self.gp, y_opt=y_opt, xi=xi)

            # Prevent re-evaluating already tested points
            for x in self.sample:
                duplicate_mask = np.all(np.isclose(X_candidates, x), axis=1)
                ei[duplicate_mask] = -np.inf

            # If all candidates have already been tested, stop
            if np.all(~np.isfinite(ei)):
                utils.write_log(
                    self.logger, "All candidate points have already been evaluated. Stopping optimization.", "info"
                )
                break

            # Select candidate with largest EI
            candidate_idx = np.argmax(ei)
            candidate_sample = X_candidates[candidate_idx]

            # Evaluate objective function at the candidate point
            candidate_value = self.obj_func(candidate_sample)
            attempted_points.append((candidate_sample, candidate_value))

            # Add new sample to the training data
            self.sample = np.vstack((self.sample, candidate_sample))
            self.value = np.append(self.value, candidate_value)

            # Check tolerance and improvement
            if candidate_value <= self.opt_args["tolerance"]:
                best_point = candidate_sample
                best_value = candidate_value

                utils.write_log(
                    self.logger,
                    f"Optimization reached the target value. ({candidate_value} <= {self.opt_args['tolerance']})",
                    "info",
                )
                utils.write_log(
                    self.logger,
                    f"Number of iterations: {i + 1}",
                    "info",
                )
                break

            elif candidate_value < best_value:
                best_point = candidate_sample
                best_value = candidate_value
                no_improvement_iterations = 0

                utils.write_log(
                    self.logger,
                    f"New best sample: {best_point}, Value: {best_value}",
                    "info",
                )

            else:
                no_improvement_iterations += 1
                utils.write_log(
                    self.logger,
                    f"No significant improvement. ({no_improvement_iterations}/{self.opt_args['max_no_improvement']})",
                    "info",
                )

            # Stop if no improvement for too long
            if no_improvement_iterations >= self.opt_args["max_no_improvement"]:
                utils.write_log(
                    self.logger,
                    "Stopping optimization due to no significant improvement.",
                    "info",
                )
                utils.write_log(
                    self.logger,
                    f"Number of iterations: {i + 1}",
                    "info",
                )
                break

            if self.sampler.seed is not None:
                self.sampler.seed += 1

        # Extract true best point and value from all evaluated samples
        idx_best = np.argmin(self.value)
        best_point = self.sample[idx_best]
        best_value = self.value[idx_best]

        # Updates the simulation with the best point and value
        utils.write_log(
            self.logger,
            "Updating simulation results with best value",
            "info",
        )
        self.obj_func(best_point)

        utils.write_log(
            self.logger,
            f"Optimization completed with {i + 1} iterations.\nBest sample: {best_point} \nBest value: {best_value}",
            "info",
        )

        if return_attempted_points:
            return best_point, best_value, attempted_points
        return best_point, best_value
