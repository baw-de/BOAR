#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    raise Exception("This file must be run as module")

# Standard library imports
import logging
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import cast

import numpy as np
from scipy.stats import qmc

from src import utils


class LatinHypercube:
    """
    Optimized Latin Hypercube Sampling with constraints and precision handling.

    Attributes:
        bounds (List[Tuple[float, float]]): List of (min, max) tuples for each parameter.
        precision (float): Decimal precision (e.g., 0.1 for 1-decimal-place precision).
        seed (Optional[int]): Random seed for reproducibility.
        constraint_fns (Optional[List[Callable[[np.ndarray], bool]]]): List of constraint functions.
            Each function takes a sample as input and returns True if the sample satisfies the constraint.
        sampler (qmc.LatinHypercube): Instance of the Latin Hypercube sampler from scipy.stats.qmc.

    Methods:
        generate_samples(n_samples: int) -> np.ndarray:
            Generate Latin Hypercube Samples within specified bounds, ensuring the samples
            have the desired precision and satisfy constraints.

        extend_samples(existing_samples: np.ndarray, n_samples: int, only_new: bool = False) -> np.ndarray:
            Extend the existing samples with new, unique samples.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        precision: float | None = None,
        seed: int | None = None,
        constraint_fns: list[Callable[[np.ndarray], bool]] | None = None,
        logger: logging.Logger | None = None,
        silent: bool = True,
        log_dev: bool = False,
    ):
        """
        Creates a LatinHypercube sampler with specified bounds, precision, and constraints.

        Attributes:
            bounds (List[Tuple[float, float]]): List of (min, max) tuples for each parameter.
            precision (float): Decimal precision (e.g., 0.1 for 1-decimal-place precision).
            seed (Optional[int]): Random seed for reproducibility.
            constraint_fns (Optional[List[Callable[[np.ndarray], bool]]]): List of constraint functions.
                Each function takes a sample as input and returns True if the sample satisfies the constraint.
            logger (Optional[logging.Logger]): Logger for logging messages.
            silent (bool): If True, suppress all logging messages.
            log_dev (bool): If True, log detailed development messages.

        Methods:
            generate_samples(n_samples: int) -> np.ndarray:
                Generate Latin Hypercube Samples within specified bounds, ensuring the samples
                have the desired precision and satisfy constraints.
        """
        self.bounds = bounds
        self.precision = precision
        self.seed = seed
        self.constraint_fns = constraint_fns or []

        self.sampler = qmc.LatinHypercube(d=len(self.bounds), seed=self.seed)
        self.silent = silent
        self.log_dev = log_dev

        self.logger = logger or logging.getLogger(__name__)
        if not silent:
            self.logger.debug("LatinHypercubeSampler initialized.")

    def _scale_and_round(self, raw_samples: np.ndarray) -> np.ndarray:
        bounds_scaling = np.array(self.bounds)

        scaled = qmc.scale(raw_samples, bounds_scaling[:, 0], bounds_scaling[:, 1])
        if self.precision is not None:
            quantized = np.empty_like(scaled, dtype=float)
            step_dec = Decimal(str(self.precision))

            for col, (lb, ub) in enumerate(self.bounds):
                lb_dec = Decimal(str(lb))
                ub_dec = Decimal(str(ub))

                for row in range(scaled.shape[0]):
                    val_dec = Decimal(str(float(scaled[row, col])))
                    k = ((val_dec - lb_dec) / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    snapped = lb_dec + k * step_dec

                    if snapped < lb_dec:
                        snapped = lb_dec
                    elif snapped > ub_dec:
                        snapped = ub_dec

                    quantized[row, col] = float(snapped)

            scaled = quantized
        return cast(np.ndarray, scaled)

    def _apply_constraints(self, samples: np.ndarray) -> np.ndarray:
        if self.constraint_fns == [None]:
            return samples
        mask = np.all([np.array([fn(sample) for fn in self.constraint_fns]) for sample in samples], axis=1)
        return cast(np.ndarray, np.asarray(samples[mask], dtype=float))

    def extend_samples(self, existing_samples: np.ndarray, n_samples: int, only_new: bool = False) -> np.ndarray:
        """
        Extend the existing samples with new, unique samples.

        Parameters:
            existing_samples (np.ndarray): Existing samples to extend.
            n_samples (int): Number of new samples to generate.
            only_new (bool): If True, return only the new samples.

        Returns:
            np.ndarray: Array of unique, constrained, and precision-rounded samples.
        """
        total_length = len(existing_samples) + n_samples
        extended_samples = self.generate_samples(total_length, existing_samples)

        if only_new:
            new_samples = np.array(
                [sample for sample in extended_samples if tuple(sample) not in set(map(tuple, existing_samples))]
            )
            return new_samples

        return extended_samples

    def generate_samples(
        self, n_samples: int, samples: np.ndarray | None = None, stop_on_fail: bool = False
    ) -> np.ndarray:
        """
        Generate constrained, unique samples using Latin Hypercube Sampling.

        Parameters:
            n_samples (int): Number of samples to generate.
            samples (np.array): Existing samples to extend.
            stop_on_fail (bool): If True, raise an error if the desired number of samples cannot be generated.

        Returns:
            np.ndarray: Array of unique, constrained, and precision-rounded samples.
        """
        utils.write_log(self.logger, f"Generating {n_samples} samples.", "info", silent=not self.log_dev)

        # Initialize values
        samples_tuple = [tuple(sample) for sample in (samples if samples is not None else [])]
        generated_samples = set(samples_tuple)

        attempts = 0
        max_attempts = 10000 * n_samples
        batch_size = max(10, n_samples)
        seed_flag = False

        while len(generated_samples) < n_samples and attempts < max_attempts:
            raw_samples = self.sampler.random(batch_size)
            rounded_samples = self._scale_and_round(raw_samples)
            valid_samples = self._apply_constraints(rounded_samples)

            for sample in valid_samples:
                if len(generated_samples) >= n_samples:
                    break
                generated_samples.add(tuple(sample))

            attempts += 1

            if attempts / max_attempts > 0.3:
                self.seed = np.random.randint(1, int(1e6))
                self.sampler = qmc.LatinHypercube(d=len(self.bounds), seed=self.seed)
                seed_flag = True

        if seed_flag:
            utils.write_log(
                self.logger,
                "Seed randomly adjusted to generate the desired number of samples.",
                "info",
                silent=not self.log_dev,
            )

        if len(generated_samples) < n_samples:
            utils.write_log(self.logger, f"Failed to generate {n_samples} samples.", "error")
            if stop_on_fail:
                raise ValueError(
                    "Could not generate enough samples.\nPlease check the constraints and bounds.\n This error typically occurs when one of the constraints has an equality and a high or no precision is given."
                )
            else:
                utils.write_log(self.logger, f"Returning {len(generated_samples)} samples.", "info")
                return cast(np.ndarray, np.asarray(list(generated_samples), dtype=float))

        utils.write_log(
            self.logger, f"Successfully generated {len(generated_samples)} samples.", "info", silent=not self.log_dev
        )
        return cast(np.ndarray, np.asarray(list(generated_samples), dtype=float))


class Uniform_sampler:
    """
    Uniform Sampler for generating random samples within specified bounds.

    Attributes:
        n_dim (int): Number of dimensions for the samples.
        bounds (List[Tuple[float, float]]): List of (min, max) tuples for each parameter.
        precision (float): Decimal precision (e.g., 0.1 for 1-decimal-place precision).
        seed (Optional[int]): Random seed for reproducibility.

    Methods:
        generate_samples(n_samples: int) -> np.ndarray:
            Generate uniformly distributed samples within specified bounds and precision.
    """

    def __init__(
        self,
        n_dim: int,
        bounds: list[tuple[float, float]],
        precision: float | None = None,
        seed: int | None = None,
    ):
        self.n_dim = n_dim
        self.bounds = bounds
        self.precision = precision
        self.seed = seed

    def generate_samples(self, n_samples: int) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        samples = np.random.uniform(
            low=[b[0] for b in self.bounds], high=[b[1] for b in self.bounds], size=(n_samples, self.n_dim)
        )

        if self.precision is not None:
            quantized = np.empty_like(samples, dtype=float)
            step_dec = Decimal(str(self.precision))

            for col, (lb, ub) in enumerate(self.bounds):
                lb_dec = Decimal(str(lb))
                ub_dec = Decimal(str(ub))

                for row in range(samples.shape[0]):
                    val_dec = Decimal(str(float(samples[row, col])))
                    k = ((val_dec - lb_dec) / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    snapped = lb_dec + k * step_dec

                    if snapped < lb_dec:
                        snapped = lb_dec
                    elif snapped > ub_dec:
                        snapped = ub_dec

                    quantized[row, col] = float(snapped)

            samples = quantized

        return cast(np.ndarray, samples)


class SOBOL_sampler:
    """
    Sobol Sampler for generating low-discrepancy samples within specified bounds.

    Attributes:
        n_dim (int): Number of dimensions for the samples.
        bounds (List[Tuple[float, float]]): List of (min, max) tuples for each parameter.
        precision (float): Decimal precision (e.g., 0.1 for 1-decimal-place precision).
        seed (Optional[int]): Random seed for reproducibility.

    Methods:
        generate_samples(n_samples: int) -> np.ndarray:
            Generate low-discrepancy samples within specified bounds and precision.
    """

    def __init__(
        self,
        n_dim: int,
        bounds: list[tuple[float, float]],
        precision: float | None = None,
        seed: int | None = None,
    ):
        self.n_dim = n_dim
        self.bounds = bounds
        self.precision = precision
        self.seed = seed
        self.sampler = qmc.Sobol(d=self.n_dim, scramble=True, seed=self.seed)

    def generate_samples(self, n_samples: int) -> np.ndarray:
        raw_samples = self.sampler.random(n_samples)
        scaled_samples = qmc.scale(raw_samples, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

        if self.precision is not None:
            quantized = np.empty_like(scaled_samples, dtype=float)
            step_dec = Decimal(str(self.precision))

            for col, (lb, ub) in enumerate(self.bounds):
                lb_dec = Decimal(str(lb))
                ub_dec = Decimal(str(ub))

                for row in range(scaled_samples.shape[0]):
                    val_dec = Decimal(str(float(scaled_samples[row, col])))
                    k = ((val_dec - lb_dec) / step_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    snapped = lb_dec + k * step_dec

                    if snapped < lb_dec:
                        snapped = lb_dec
                    elif snapped > ub_dec:
                        snapped = ub_dec

                    quantized[row, col] = float(snapped)

            scaled_samples = quantized

        return cast(np.ndarray, scaled_samples)
