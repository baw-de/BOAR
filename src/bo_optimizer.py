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

# Third-party imports
import numpy as np
import scipy.optimize as opt
import sklearn.gaussian_process.kernels as gpr_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from skopt.acquisition import gaussian_ei

# Import Basement-Calibrator modules
from src import utils

# Import user defined functions


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

        self.scaler = StandardScaler()
        self.sample_scaled = self.scaler.fit_transform(self.sample)
        self.value_scaled = self.scaler.fit_transform(self.value.reshape(-1, 1)).ravel()

        kernel = gpr_kernels.Matern(length_scale=1.0, nu=2.5)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.opt_args["GPR_iterations"],
            alpha=1e-6,
            normalize_y=True,
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

        if optimizer_type is None:
            optimizer_callable = l_bfgs_b_modified

        return optimizer_callable

    def optimize(self, return_attempted_points=False):
        """
        Perform Bayesian Optimization to find optimal parameters.

        This method iteratively:
            1. Predicts candidate points using the Gaussian Process surrogate.
            2. Evaluates the objective function at candidate points.
            3. Updates the GP model with new observations.
            4. Stops when tolerance is reached or max iterations exceeded.

        Args:
            return_attempted_points: If True, return all attempted points.

        Returns:
            Tuple: (best_point, best_value) or (best_point, best_value, attempted_points)
                - best_point (np.ndarray): Optimal parameter values found.
                - best_value (float): Objective function value at best_point.
                - attempted_points (list): All points tried during optimization (optional).

        Raises:
            ValueError: If test population and constraints are both unspecified.

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
        test_population = self.opt_args.get("test_population", None)
        constraints_bool = self.opt_args.get("constraints", None) is not None

        # Start counters
        attempted_points = []
        no_improvement_iterations = 0

        # Generate test samples
        if test_population is None and not constraints_bool:
            bnds = self.sampler.bounds
            pres = self.sampler.precision
            n_max_samples = int((np.max(bnds) - np.min(bnds)) / pres) + 1
            X_candidates = self.sampler.generate_samples(n_max_samples)
        elif test_population is not None:
            X_candidates = self.sampler.generate_samples(self.opt_args["test_population"])
        else:
            utils.write_log(
                self.logger,
                "The user must either specify a test population or remove the sampling constraints.",
                "error",
            )
            raise ValueError("The user must either specify a test population or remove the sampling constraints.")
        X_candidates_scaled = self.scaler.fit_transform(X_candidates)

        for i in range(self.opt_args["max_tested_vectors"]):
            utils.write_log(self.logger, f"Iteration {i + 1} of {self.opt_args['max_tested_vectors']}.", "info")

            if self.sampler.seed is not None:
                self.sampler.seed += 1

            mu, sigma = self.gp.predict(X_candidates_scaled, return_std=True)
            ei = gaussian_ei(X_candidates_scaled, self.gp)
            candidate_sample = X_candidates[np.argmax(ei)]
            candidate_value = self.obj_func(candidate_sample)
            attempted_points.append((candidate_sample, candidate_value))

            # Check for improvement
            if candidate_value <= self.opt_args["tolerance"]:
                best_sample, best_value = candidate_sample, candidate_value
                utils.write_log(
                    self.logger,
                    f"Optimization reached the target value. ({candidate_value} <= {self.opt_args['tolerance']})",
                    "info",
                )
                utils.write_log(self.logger, f"Number of iterations: {i + 1}", "info")
                break

            elif candidate_value < best_value:
                best_sample, best_value = candidate_sample, candidate_value
                utils.write_log(self.logger, f"New best sample: {best_sample}, Value: {best_value}", "info")
                no_improvement_iterations = 0  # Reset the counter

            else:
                no_improvement_iterations += 1
                utils.write_log(
                    self.logger,
                    f"No significant improvement. ({no_improvement_iterations}/{self.opt_args['max_no_improvement']})",
                    "info",
                )

            # Terminate if no improvement for max_no_improvement iterations
            if no_improvement_iterations >= self.opt_args["max_no_improvement"]:
                utils.write_log(self.logger, "Stopping optimization due to no significant improvement.", "info")
                utils.write_log(self.logger, f"Number of iterations: {i + 1}", "info")
                break

            # Add the new sample to the training set
            self.sample = np.vstack((self.sample, candidate_sample))
            self.value = np.append(self.value, candidate_value)
            self.sample_scaled = self.scaler.fit_transform(self.sample)
            self.value_scaled = self.scaler.fit_transform(self.value.reshape(-1, 1)).ravel()

            # Update the GP
            self.gp.fit(self.sample_scaled, self.value_scaled)

        # Extract the best point and value
        idx_best = np.argmin(self.value)
        best_point = self.sample[idx_best]
        best_value = self.value[idx_best]

        # Updates the simulation with the best point and value
        utils.write_log(self.logger, "Updating simulation results with best value", "info")
        self.obj_func(best_point)

        utils.write_log(
            self.logger,
            f"Optimization completed with {i + 1} iterations.\nBest sample: {best_point} \nBest value: {best_value}",
            "info",
        )

        if return_attempted_points:
            return best_point, best_value, attempted_points
        return best_point, best_value
