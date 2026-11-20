"""Tests for prepare_simulation_folder: originals stay read-only, runs work on a copy in output/<name>/."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src import basement_tools as bmt
from src import functions_io as fio

LOGGER = logging.getLogger("test_functions_io")

MESH = """MESH2D
ND 1 0.0 0.0 0.0
ND 2 1.0 0.0 0.0
ND 3 0.0 1.0 0.0
ND 4 1.0 1.0 0.0
E3T 1 1 2 3 1 0.0
E3T 2 2 4 3 2 0.0
"""


def _model(name: str | None = "case_a") -> dict:
    model = {
        "SETUP": {
            "BASEHPC": {
                "BASEPLANE_2D": {
                    "GEOMETRY": {
                        "mesh_file": "mesh.2dm",
                        "REGIONDEF": [{"index": [1], "name": "main_channel"}, {"index": [2], "name": "floodplain"}],
                    },
                    "HYDRAULICS": {
                        "FRICTION": {
                            "regions": [
                                {"friction": 30.0, "region_name": "main_channel"},
                                {"friction": 20.0, "region_name": "floodplain"},
                            ]
                        },
                        "BOUNDARY": {"STANDARD": [{"type": "uniform_in", "discharge_file": "Q001.txt"}]},
                    },
                }
            }
        }
    }
    if name is not None:
        model["SETUP"]["simulation_name"] = name
    return model


@pytest.fixture
def config_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "simulation_configs"
    folder.mkdir()
    (folder / "model.json").write_text(json.dumps(_model()))
    (folder / "simulation.json").write_text(json.dumps({"SIMULATION": {"TIME": {"out": 10}}}))
    (folder / "results.json").write_text(json.dumps({"RESULTS": {}}))
    (folder / "mesh.2dm").write_text(MESH)
    (folder / "Q001.txt").write_text("0 1\n")
    return folder


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_copies_configs_into_named_folder(config_folder, tmp_path):
    case = fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)

    assert case == tmp_path / "output" / "case_a"
    for name in fio.SIMULATION_CONFIG_FILES:
        assert (case / name).is_file()

    model = json.loads((case / "model.json").read_text())
    assert model["SETUP"]["simulation_name"] == "case_a"
    plane = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]
    assert Path(plane["GEOMETRY"]["mesh_file"]) == (config_folder / "mesh.2dm").resolve()
    bc = plane["HYDRAULICS"]["BOUNDARY"]["STANDARD"][0]
    assert Path(bc["discharge_file"]) == (config_folder / "Q001.txt").resolve()


def test_originals_unchanged_after_friction_and_discharge_updates(config_folder, tmp_path):
    before = _snapshot(config_folder)
    case = fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)

    manager = bmt.SimulationManager(str(case), {}, silent=True)
    friction = np.array([(1, 45.0), (2, 12.5)], dtype=[("region", "i2"), ("friction", "f8")])
    manager.define_friction({"mesh.2dm": str(config_folder / "mesh.2dm")}, friction)
    manager.define_discharge(123.0)

    assert _snapshot(config_folder) == before
    copied = json.loads((case / "model.json").read_text())
    regions = copied["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["FRICTION"]["regions"]
    assert [r["friction"] for r in regions] == [45.0, 12.5]


def test_overwrite_true_removes_previous_run(config_folder, tmp_path):
    case = fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)
    (case / "optimization_memory.csv").write_text("Friction Vector,Calibration Error\n")

    fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)
    assert not (case / "optimization_memory.csv").exists()


def test_overwrite_false_keeps_memory_but_refreshes_json(config_folder, tmp_path):
    case = fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)
    (case / "optimization_memory.csv").write_text("Friction Vector,Calibration Error\n")
    (case / "model.json").write_text("{}")  # as if a previous run had modified it

    fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=False, logger=LOGGER)
    assert (case / "optimization_memory.csv").exists()
    assert "SETUP" in json.loads((case / "model.json").read_text())


def test_missing_name_raises(config_folder, tmp_path):
    (config_folder / "model.json").write_text(json.dumps(_model(name=None)))
    with pytest.raises(KeyError, match="simulation_name"):
        fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)


@pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", "..", ""])
def test_invalid_name_raises(config_folder, tmp_path, name):
    (config_folder / "model.json").write_text(json.dumps(_model(name=name)))
    with pytest.raises((ValueError, KeyError)):
        fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)
    assert not (tmp_path / "escape").exists()


def test_missing_config_file_raises(config_folder, tmp_path):
    (config_folder / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match="results.json"):
        fio.prepare_simulation_folder(config_folder, tmp_path / "output", overwrite=True, logger=LOGGER)
