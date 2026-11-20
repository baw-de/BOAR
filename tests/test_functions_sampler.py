"""Tests for the precision grid helpers and the auto-sized test population."""

import numpy as np
import pytest

from src import bo_optimizer, functions_sampler, utils

BOUNDS_2D = [(10, 60), (10, 60)]


def _x_ge_y():
    return utils.create_constraint_function("x > y or np.isclose(x, y)", ["x", "y"])


def test_grid_size_full_grid():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, seed=0)
    assert sampler.grid_size() == 51 * 51


def test_axis_includes_upper_bound_when_step_does_not_divide():
    sampler = functions_sampler.LatinHypercube(bounds=[(0, 1.05)], precision=0.1, seed=0)
    values = sampler._axis_values(0, 1.05)
    assert values[0] == 0.0
    assert values[-2] == 1.0
    assert values[-1] == 1.05
    assert sampler.grid_size() == 12


def test_grid_size_none_without_precision():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=None, seed=0)
    assert sampler.grid_size() is None


def test_enumerate_grid_with_constraint():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, constraint_fns=[_x_ge_y()], seed=0)
    grid = sampler.enumerate_grid()
    assert grid.shape == (51 * 52 // 2, 2)
    assert np.all(grid[:, 0] >= grid[:, 1])
    assert np.allclose(grid, np.round(grid))
    assert len({tuple(p) for p in grid}) == len(grid)


def test_enumerate_grid_matches_sampler_values():
    sampler = functions_sampler.LatinHypercube(bounds=[(10, 60), (0.5, 2)], precision=0.1, seed=1)
    grid = {tuple(p) for p in sampler.enumerate_grid()}
    samples = sampler.generate_samples(200)
    assert all(tuple(s) in grid for s in samples)


def test_build_candidates_uses_full_feasible_grid():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, constraint_fns=[_x_ge_y()], seed=0)
    candidates = bo_optimizer.build_candidate_population(sampler, test_population="None")
    assert len(candidates) == 1326


def test_build_candidates_respects_cap():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, constraint_fns=[_x_ge_y()], seed=0)
    full = {tuple(p) for p in sampler.enumerate_grid()}
    candidates = bo_optimizer.build_candidate_population(sampler, None, max_test_population=500, seed=0)
    assert len(candidates) == 500
    assert len({tuple(p) for p in candidates}) == 500
    assert all(tuple(p) in full for p in candidates)


def test_build_candidates_explicit_population():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, seed=0)
    candidates = bo_optimizer.build_candidate_population(sampler, test_population=50)
    assert len(candidates) == 50


def test_build_candidates_no_feasible_point():
    never = utils.create_constraint_function("x > y and y > x", ["x", "y"])
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, constraint_fns=[never], seed=0)
    with pytest.raises(ValueError):
        bo_optimizer.build_candidate_population(sampler, None)


def test_build_candidates_rejects_non_positive_cap():
    sampler = functions_sampler.LatinHypercube(bounds=BOUNDS_2D, precision=1, seed=0)
    with pytest.raises(ValueError):
        bo_optimizer.build_candidate_population(sampler, None, max_test_population=0)
