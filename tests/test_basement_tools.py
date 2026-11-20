"""
Tests for reading BASEMENT results of different versions in src.basement_tools.

BASEMENT 4.2 stores the bed elevation once in ``CellsAll/BottomEl``; BASEMENT 4.3
stores it with every output in ``RESULTS/CellsAll/BottomEl/<time key>``. The
tests build small synthetic results files of both layouts, without BASEMENT.
"""

import logging

import h5py
import numpy as np
import pytest

from src import basement_tools as bmt

N_CELLS = 4
N_OUTPUTS = 12  # > 10, so lexicographic and numeric key order differ for the aux keys
SAVE_FREQ = 10


def _bed(step: int) -> np.ndarray:
    """Bed elevation of output ``step`` (changes per step, to check the right one is read)."""
    return np.full((N_CELLS, 1), 1.0 + 0.1 * step)


def _write_results(folder, version: str) -> None:
    """Write a minimal results.h5 / results_aux.h5 pair in the layout of ``version``."""
    with h5py.File(folder / "results.h5", "w") as f:
        if version == "4.2":
            f["CellsAll/BottomEl"] = _bed(0)
        for step in range(N_OUTPUTS):
            key = f"{step:07d}"
            wse = _bed(step)[:, 0] + 0.5  # water depth 0.5 everywhere
            f[f"RESULTS/CellsAll/HydState/{key}"] = np.column_stack([wse, np.full(N_CELLS, step), np.zeros(N_CELLS)])
            f[f"RESULTS/CellsAll/ReyState/{key}"] = np.zeros((N_CELLS, 8))
            f[f"RESULTS/CellsAll/TrbState/{key}"] = np.zeros((N_CELLS, 6))
            if version == "4.3":
                f[f"RESULTS/CellsAll/BottomEl/{key}"] = _bed(step)

    with h5py.File(folder / "results_aux.h5", "w") as f:
        for step in range(N_OUTPUTS):
            key = str(step)  # aux keys are not zero-padded
            f[f"flow_velocity/{key}"] = np.full((N_CELLS, 2), float(step))
            f[f"turb_k/{key}"] = np.zeros((N_CELLS, 1))
            f[f"turb_reynolds/{key}"] = np.zeros((N_CELLS, 2))


def _manager(folder, version: str) -> bmt.SimulationManager:
    """SimulationManager without reading mesh or setup files."""
    manager = bmt.SimulationManager.__new__(bmt.SimulationManager)
    manager.sim_folder = str(folder)
    manager.logger = logging.getLogger("test_basement_tools")
    manager.log_dev = False
    manager.silent = True
    manager.save_freq = SAVE_FREQ
    manager.mesh_file = "mesh.2dm"
    manager.basement_version = bmt.SimulationManager._parse_basement_version(version)
    return manager


class TestParseBasementVersion:
    @pytest.mark.parametrize(
        "value, expected",
        [("4.3", (4, 3)), ("4.2", (4, 2)), ("4.2.0", (4, 2)), ("v4.3", (4, 3)), (4.3, (4, 3)), (" 4.3.1 ", (4, 3))],
    )
    def test_accepts_supported_versions(self, value, expected):
        assert bmt.SimulationManager._parse_basement_version(value) == expected

    @pytest.mark.parametrize("value", ["4", "four", "", None, "4.1", "5.0", "3.9"])
    def test_rejects_invalid_or_unsupported(self, value):
        with pytest.raises(ValueError):
            bmt.SimulationManager._parse_basement_version(value)

    def test_default_is_4_3(self):
        assert bmt.SimulationManager._parse_basement_version(bmt.DEFAULT_BASEMENT_VERSION) == (4, 3)


class TestReadBottomElevation:
    @pytest.mark.parametrize("version", ["4.2", "4.3"])
    def test_reads_layout_of_version(self, tmp_path, version):
        _write_results(tmp_path, version)
        manager = _manager(tmp_path, version)
        with h5py.File(tmp_path / "results.h5", "r") as f:
            bed = manager._read_bottom_elevation(f, "0000005")
        expected_step = 5 if version == "4.3" else 0
        np.testing.assert_allclose(bed, _bed(expected_step).ravel())

    @pytest.mark.parametrize("configured, written", [("4.3", "4.2"), ("4.2", "4.3")])
    def test_wrong_version_gives_hint(self, tmp_path, configured, written):
        _write_results(tmp_path, written)
        manager = _manager(tmp_path, configured)
        with h5py.File(tmp_path / "results.h5", "r") as f, pytest.raises(KeyError, match=f'"{written}"'):
            manager._read_bottom_elevation(f, "0000000")


class TestExtractSimData:
    @pytest.mark.parametrize("version", ["4.2", "4.3"])
    def test_last_output_and_water_depth(self, tmp_path, version):
        _write_results(tmp_path, version)
        data = _manager(tmp_path, version).extract_sim_data({}, sim_time=-1)

        last = N_OUTPUTS - 1
        assert data["time"] == last * SAVE_FREQ
        np.testing.assert_allclose(data["hyd"]["qx"], last)  # HydState of the last output
        np.testing.assert_allclose(data["hyd"]["ux"], last)  # aux keys sorted numerically ('11', not '9')
        if version == "4.3":
            np.testing.assert_allclose(data["hyd"]["h"], 0.5)  # bed of the same output step

    def test_specific_time_v43(self, tmp_path):
        _write_results(tmp_path, "4.3")
        data = _manager(tmp_path, "4.3").extract_sim_data({}, sim_time=3 * SAVE_FREQ)
        assert data["time"] == 3 * SAVE_FREQ
        np.testing.assert_allclose(data["hyd"]["h"], 0.5)
