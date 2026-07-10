"""
Smoke tests for the BOAR utility module.

These are minimal tests that verify the public API works without requiring
external simulation files (BASEMENT executables, HDF5 results, etc.).
"""

import numpy as np
import pytest

from src import utils


class TestWriteLog:
    """Tests for utils.write_log."""

    def test_info_message_is_logged(self, caplog):
        import logging

        logger = logging.getLogger("test")
        with caplog.at_level(logging.INFO, logger="test"):
            utils.write_log(logger, "hello", level="info", silent=False)
        assert "hello" in caplog.text

    def test_silent_suppresses_info(self, caplog):
        import logging

        logger = logging.getLogger("test_silent")
        with caplog.at_level(logging.INFO, logger="test_silent"):
            utils.write_log(logger, "suppressed", level="info", silent=True)
        assert "suppressed" not in caplog.text

    def test_error_is_never_suppressed(self, caplog):
        import logging

        logger = logging.getLogger("test_error")
        with caplog.at_level(logging.ERROR, logger="test_error"):
            utils.write_log(logger, "critical", level="error", silent=True)
        assert "critical" in caplog.text


class TestFindClosestCell:
    """Tests for utils.find_closest_cell."""

    def _make_centroids(self, coords):
        dtype = np.dtype([("cx", "f8"), ("cy", "f8")])
        arr = np.empty(len(coords), dtype=dtype)
        for i, (x, y) in enumerate(coords):
            arr[i]["cx"] = x
            arr[i]["cy"] = y
        return arr

    def test_finds_exact_match(self):
        centroids = self._make_centroids([(0, 0), (1, 1), (2, 2)])
        idx = utils.find_closest_cell(centroids, (1.0, 1.0), npoints=1)
        assert idx[0] == 1

    def test_finds_nearest_neighbour(self):
        centroids = self._make_centroids([(0, 0), (10, 10), (20, 20)])
        idx = utils.find_closest_cell(centroids, (9.9, 9.9), npoints=1)
        assert idx[0] == 1

    def test_returns_distance_when_requested(self):
        centroids = self._make_centroids([(0, 0), (3, 4)])
        idx, dists = utils.find_closest_cell(centroids, (0, 0), npoints=2, distance=True)
        assert dists[0] == pytest.approx(0.0)
        assert dists[1] == pytest.approx(25.0)  # squared distance


class TestCreateConstraintFunction:
    """Tests for utils.create_constraint_function."""

    def test_simple_greater_than(self):
        fn = utils.create_constraint_function("x > y", ["x", "y"])
        assert fn([2, 1]) is True
        assert fn([1, 2]) is False

    def test_arithmetic_expression(self):
        fn = utils.create_constraint_function("x + y > 5", ["x", "y"])
        assert fn([3, 3]) is True
        assert fn([2, 2]) is False

    def test_disallowed_node_raises(self):
        # Direct built-in calls like __import__ are rejected
        with pytest.raises(ValueError, match="not permitted"):
            utils.create_constraint_function("__import__('os')", [])

    def test_disallowed_attribute_call_raises(self):
        # Only np.* attribute calls are allowed — other module calls are rejected
        with pytest.raises(ValueError, match="only numpy"):
            utils.create_constraint_function("os.system('ls')", ["os"])

    def test_numpy_call_is_allowed(self):
        # np.* calls must work (used in real constraints like np.clip)
        fn = utils.create_constraint_function("np.clip(x, 0, 1) > 0.5", ["x"])
        assert fn([0.8]) == True  # noqa: E712 — numpy returns np.bool_, not Python bool
        assert fn([0.3]) == False  # noqa: E712

    def test_wrong_number_of_values_raises(self):
        fn = utils.create_constraint_function("x > 0", ["x"])
        with pytest.raises(ValueError):
            fn([1, 2])  # too many values


class TestModifyStructuredArray:
    """Tests for utils.modify_structured_array."""

    def _make_array(self):
        dtype = np.dtype([("a", "f8"), ("b", "i4")])
        arr = np.array([(1.0, 2), (3.0, 4)], dtype=dtype)
        return arr

    def test_rename_column(self):
        arr = self._make_array()
        result = utils.modify_structured_array(arr, rename_dict={"a": "alpha"})
        assert "alpha" in result.dtype.names
        assert "a" not in result.dtype.names

    def test_delete_column(self):
        arr = self._make_array()
        result = utils.modify_structured_array(arr, delete_cols=["b"])
        assert "b" not in result.dtype.names
        assert "a" in result.dtype.names

    def test_non_structured_array_raises(self):
        plain = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            utils.modify_structured_array(plain)
