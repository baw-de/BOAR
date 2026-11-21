"""Tests for batch mode: case discovery, template replication, summary CSV, workers and the queue."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from src import batch, cpu_topology
from src import friction_calibration as fc
from src import functions_io as fio
from src.cpu_topology import GpuInfo, NumaNode, Placement
from tests.test_cpu_topology import core_record, numa_record
from tests.test_functions_io import MESH, _model

LOGGER = logging.getLogger("test_batch")


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────────────────
def _write_configs(folder: Path, name: str = "case_a", discharge_file: str = "Q001.txt") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model = _model(name)
    model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"][0]["discharge_file"] = (
        discharge_file
    )
    (folder / "model.json").write_text(json.dumps(model))
    (folder / "simulation.json").write_text(json.dumps({"SIMULATION": {"TIME": {"out": 10}}}))
    (folder / "results.json").write_text(json.dumps({"RESULTS": {}}))
    (folder / "mesh.2dm").write_text(MESH)


def _ground_truth(folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    scenario, _, code = name.rpartition("_")
    pd.DataFrame({"X": [0.5], "Y": [0.5], "H": [1.0]}).to_csv(folder / f"{scenario}_{code}_1.csv", index=False)


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Path]:
    """A batch project: template config, discharge files and a discharge dictionary."""
    folders = {
        "SIMULATION_CONFIG_FOLDER": tmp_path / "simulation_configs",
        "GROUND_TRUTH_FOLDER": tmp_path / "ground_truth",
        "OUTPUT_FOLDER": tmp_path / "output",
        "MESH_FOLDER": tmp_path / "meshes",
        "DISCHARGE_FILES_FOLDER": tmp_path / "discharge_files",
    }
    for folder in folders.values():
        folder.mkdir(exist_ok=True)
    _write_configs(folders["SIMULATION_CONFIG_FOLDER"], "template_case")
    for code in ("Q001", "Q002"):
        (folders["DISCHARGE_FILES_FOLDER"] / f"{code}.txt").write_text("0 1\n")
    pd.DataFrame({"Code": ["Q001", "Q002"], "Q": [10.0, 20.0]}).to_csv(
        folders["DISCHARGE_FILES_FOLDER"] / "discharge_dictionary.csv", index=False
    )
    return folders


# ── Case discovery ───────────────────────────────────────────────────────────────────────────────────
def test_single_mode_without_subfolders(project):
    _ground_truth(project["GROUND_TRUTH_FOLDER"], "CASE_Q001")  # flat CSV, no subfolder
    cases = fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], project["GROUND_TRUTH_FOLDER"], LOGGER)
    assert len(cases) == 1
    assert not cases[0].batch
    assert cases[0].config_folder == project["SIMULATION_CONFIG_FOLDER"]


def test_batch_mode_template_own_and_missing(project, tmp_path):
    gt = project["GROUND_TRUTH_FOLDER"]
    for name in ("A_Q001", "B_Q002", "my_scen_Q001", "nounderscore"):
        _ground_truth(gt / name, name if "_" in name else "X_Q001")
    _write_configs(project["SIMULATION_CONFIG_FOLDER"] / "B_Q002", "b_case")
    _write_configs(project["SIMULATION_CONFIG_FOLDER"] / "ORPHAN_Q001", "orphan")

    cases = {c.name: c for c in fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], gt, LOGGER)}
    assert list(cases) == ["A_Q001", "B_Q002", "my_scen_Q001", "nounderscore"]
    assert cases["A_Q001"].shared_config and cases["A_Q001"].config_folder == project["SIMULATION_CONFIG_FOLDER"]
    assert not cases["B_Q002"].shared_config
    assert cases["B_Q002"].config_folder == project["SIMULATION_CONFIG_FOLDER"] / "B_Q002"
    assert (cases["my_scen_Q001"].scenario, cases["my_scen_Q001"].discharge) == ("my_scen", "Q001")
    assert cases["nounderscore"].error and "SCENARIO_DISCHARGE" in cases["nounderscore"].error
    assert all(c.error is None for n, c in cases.items() if n != "nounderscore")


def test_batch_case_without_template_or_own_config_fails(project):
    for name in ("model.json", "simulation.json", "results.json"):
        (project["SIMULATION_CONFIG_FOLDER"] / name).unlink()
    _ground_truth(project["GROUND_TRUTH_FOLDER"] / "A_Q001", "A_Q001")
    (case,) = fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], project["GROUND_TRUTH_FOLDER"], LOGGER)
    assert case.error and "template" in case.error


# ── Inflow discharge ─────────────────────────────────────────────────────────────────────────────────
def test_set_inflow_discharge_changes_only_inflow_boundaries():
    model = _model()
    bcs = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
    bcs.append({"type": "zhydrograph", "discharge_file": "keep.txt"})
    assert fio.set_inflow_discharge(model, discharge=12.5)
    assert bcs[0] == {"type": "uniform_in", "discharge": 12.5}
    assert bcs[1]["discharge_file"] == "keep.txt"
    assert fio.set_inflow_discharge(model, discharge_file="/abs/Q002.txt")
    assert bcs[0] == {"type": "uniform_in", "discharge_file": "/abs/Q002.txt"}
    with pytest.raises(ValueError):
        fio.set_inflow_discharge(model)


def test_inflow_discharge_kind_from_model_json():
    model = _model()  # uniform_in with a discharge_file
    assert fio.inflow_discharge_kind(model) == "file"
    fio.set_inflow_discharge(model, discharge=12.5)
    assert fio.inflow_discharge_kind(model) == "value"
    model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"] = [
        {"type": "zqhydrograph", "wse_file": "w.txt", "spec_discharge_file": "q.txt"}
    ]
    assert fio.inflow_discharge_kind(model) is None
    assert fio.inflow_discharge_kind({}) is None


def test_resolve_inflow_file_and_table(project):
    table = fio.read_discharge_table(project["DISCHARGE_FILES_FOLDER"])
    folder = fio.discharge_files_folder({}, project)
    assert folder == project["DISCHARGE_FILES_FOLDER"].resolve()
    assert Path(fio.resolve_inflow("Q002", "file", table, folder)["discharge_file"]).name == "Q002.txt"
    with pytest.raises(FileNotFoundError):
        fio.resolve_inflow("Q009", "file", table, folder)
    assert fio.resolve_inflow("Q002", "value", table, folder) == {"discharge": 20.0}
    with pytest.raises(KeyError):
        fio.resolve_inflow("Q009", "value", table, folder)
    # The dictionary is needed only for constant discharges
    (project["DISCHARGE_FILES_FOLDER"] / "discharge_dictionary.csv").unlink()
    assert fio.read_discharge_table(folder) is None
    assert Path(fio.resolve_inflow("Q001", "file", None, folder)["discharge_file"]).name == "Q001.txt"
    with pytest.raises(KeyError, match="constant inflow discharge"):
        fio.resolve_inflow("Q001", "value", None, folder)
    assert fio.resolve_inflow("Q009", None, table, folder) is None  # no inflow to change


def test_discharge_files_folder_default_and_deprecated(project, tmp_path, caplog):
    assert fio.discharge_files_folder({}, project) == project["DISCHARGE_FILES_FOLDER"].resolve()
    # Folders without the key: DISCHARGE_FILES_FOLDER of constants.py
    without = {k: v for k, v in project.items() if k != "DISCHARGE_FILES_FOLDER"}
    assert fio.discharge_files_folder({}, without) == Path(fio.c.DISCHARGE_FILES_FOLDER).resolve()
    legacy = {"discharge_file": True, "discharge_file_directory": str(tmp_path / "old"), "discharge_file_list": []}
    with caplog.at_level(logging.WARNING):
        assert fio.discharge_files_folder(legacy, project) == (tmp_path / "old").resolve()


def test_template_replicated_with_case_discharge(project):
    template = project["SIMULATION_CONFIG_FOLDER"]
    original = (template / "model.json").read_bytes()
    _ground_truth(project["GROUND_TRUTH_FOLDER"] / "A_Q002", "A_Q002")
    (case,) = fio.discover_cases(template, project["GROUND_TRUTH_FOLDER"], LOGGER)
    inflow = fio.resolve_inflow("Q002", "file", None, fio.discharge_files_folder({}, project))

    paths = fio.setup_directories(project["OUTPUT_FOLDER"].parent, {}, LOGGER, project, case, inflow)

    run_folder = Path(paths["folder_simulation"])
    assert run_folder == project["OUTPUT_FOLDER"] / "A_Q002" / "template_case"
    copied = json.loads((run_folder / "model.json").read_text())
    bc = copied["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"][0]
    assert Path(bc["discharge_file"]) == (project["DISCHARGE_FILES_FOLDER"] / "Q002.txt").resolve()
    assert (template / "model.json").read_bytes() == original
    assert paths["folder_ground_truth"] == project["GROUND_TRUTH_FOLDER"] / "A_Q002"


# ── Summary ──────────────────────────────────────────────────────────────────────────────────────────
def test_discharge_codes():
    keys = ["CASE_Q002_1.csv", "CASE_Q001_1.csv", "CASE_Q002_2.csv", "nocode.csv"]
    assert fio.discharge_codes(keys) == ["Q002", "Q001"]


def test_scenario_discharge_from_ground_truth():
    keys = ["CASE_Q001_1.csv", "CASE_Q001_2.csv", "CASE_Q002_1.csv"]
    assert fio.scenario_discharge_from_ground_truth(keys) == ("CASE", "Q001+Q002")


def test_write_summary_row_adds_and_replaces(tmp_path):
    path = tmp_path / "summary.csv"
    row = {
        "scenario": "A",
        "discharge": "Q001",
        "best_vector": [30.0, 20.5],
        "region_names": ["main_channel", "floodplain"],
        "best_error": 0.5,
        "n_iter_best": 7,
        "device": "cpu",
        "status": "ok",
    }
    fio.write_summary_row(path, row)
    fio.write_summary_row(path, {**row, "scenario": "B"})
    table = fio.write_summary_row(path, {**row, "best_error": 0.25, "n_iter_best": 9})
    assert list(table.columns) == [
        "scenario",
        "discharge",
        "friction_main_channel",
        "friction_floodplain",
        "best_error",
        "n_iter_best",
        "device",
        "status",
    ]
    assert table["scenario"].tolist() == ["A", "B"]
    assert table.loc[0, "best_error"] == 0.25 and table.loc[0, "n_iter_best"] == 9
    failed = fio.write_summary_row(path, {"scenario": "C", "discharge": "Q001", "status": "failed"})
    assert failed.loc[2, "status"] == "failed" and pd.isna(failed.loc[2, "best_error"])


def test_best_from_memory(tmp_path):
    rows = ["Friction Vector,Calibration Error", '"[30.0, 20.0]",0.5', '"[31.0, 21.0]",0.1', '"[32.0, 22.0]",0.1']
    (tmp_path / "optimization_memory.csv").write_text("\n".join(rows) + "\n")
    assert fc.best_from_memory(tmp_path) == ([31.0, 21.0], 0.1, 2)


# ── Workers ──────────────────────────────────────────────────────────────────────────────────────────
def two_sockets() -> list[NumaNode]:
    """2 x 8 cores without hyper-threading, both sockets in one processor group."""
    numa = numa_record(0, 0, (1 << 8) - 1) + numa_record(1, 0, ((1 << 8) - 1) << 8)
    cores = b"".join(core_record(0, 1 << i) for i in range(16))
    return cpu_topology.parse_topology(numa, cores)


def one_socket_smt() -> list[NumaNode]:
    """1 x 8 cores with hyper-threading (like this project's development laptop)."""
    return cpu_topology.parse_topology(
        numa_record(0, 0, (1 << 16) - 1), b"".join(core_record(0, 0b11 << (2 * i)) for i in range(8))
    )


@pytest.fixture
def machine(monkeypatch):
    """Fake the hardware seen by the workers: ``machine(nodes, gpus)``."""

    def set_machine(nodes, gpus=None):
        monkeypatch.setattr(cpu_topology, "detect_topology", lambda: nodes)
        monkeypatch.setattr(cpu_topology, "detect_gpus", lambda override=None: gpus)

    return set_machine


def _options(n_cpu, n_gpu, **extra):
    return {
        "basement_options": {"backend": "omp"},
        "batch_options": {"n_cpu_workers": n_cpu, "n_gpu_workers": n_gpu, **extra},
    }


def test_workers_single_socket(machine):
    machine(one_socket_smt())
    workers = batch.Workers(_options(2, 0))
    assert workers.n_threads == 4  # 8 physical cores (not 16 logical CPUs) shared by 2 workers
    assert "single socket" in workers.describe()
    slots = [workers.acquire(), workers.acquire()]
    assert [s.label for s, _ in slots] == ["cpu", "cpu"]
    assert workers.acquire() is None
    workers.release(*slots[0])
    assert workers.acquire() is not None


def test_workers_pin_cpu_runs_to_sockets(machine):
    machine(two_sockets())
    workers = batch.Workers(_options(4, 0))
    assert workers.n_threads == 4 and "on (2 sockets)" in workers.pinning
    slots = [workers.acquire()[0] for _ in range(4)]
    assert [s.label for s in slots] == ["cpu:socket0", "cpu:socket1", "cpu:socket0", "cpu:socket1"]
    assert slots[1].placement.cpus == (8, 9, 10, 11)
    assert workers.acquire() is None


def test_workers_gpu_and_cpu(machine):
    machine(two_sockets(), [GpuInfo(0, 1)])
    workers = batch.Workers(_options(1, 1))
    assert workers.n_threads == 7  # 8 cores of a socket, less one host core of the GPU run
    gpu, gpu_slot = workers.acquire()  # GPU first by default
    cpu, _ = workers.acquire()
    assert (gpu.label, gpu.backend, gpu.nthreads) == ("gpu:0", "cudaC", 1)
    assert gpu.placement.node_id == 1 and len(gpu.placement.cpus) == 1  # host core on the card's socket
    assert cpu.label.startswith("cpu:socket") and cpu.backend == "omp"
    assert workers.acquire() is None
    workers.release(gpu, gpu_slot)
    assert workers.acquire()[0].kind == "gpu"


def test_workers_preferred_cpu_and_explicit_cores(machine):
    machine(one_socket_smt(), [GpuInfo(0, None)])
    workers = batch.Workers(_options(1, 1, preferred_device="cpu", cpu_cores=6))
    assert workers.n_threads == 6
    assert workers.acquire()[0].kind == "cpu"
    assert workers.acquire()[0].kind == "gpu"


def test_workers_unpinned_when_disabled(machine):
    machine(two_sockets())
    workers = batch.Workers(_options(2, 0, pin_to_socket=False))
    assert workers.n_threads == 8 and "pin_to_socket: false" in workers.pinning
    assert workers.acquire()[0].placement is None


@pytest.mark.parametrize(
    "options", [_options(0, 0), _options(1, 1, gpu_backend="omp"), _options(-1, 2), _options(1, 0, cpu_cores=0)]
)
def test_workers_reject_invalid(machine, options):
    machine(one_socket_smt())
    with pytest.raises(ValueError):
        batch.Workers(options)


def test_case_options_carry_the_placement():
    slot = batch.Slot("cpu", "omp", 4, Placement(node_id=1, cpus=(8, 9, 10, 11), node_mask=0xF))
    options = batch._case_options({"basement_options": {"backend": "cudaC", "nthreads": -1}}, slot)
    assert options["basement_options"] == {"backend": "omp", "nthreads": 4, "_placement": slot.placement}


# ── Queue ────────────────────────────────────────────────────────────────────────────────────────────
def fake_runner(case, user_opt, slot, folders, log_file, inflow, loss_function):
    """Stands in for a calibration in a worker process (module level, so it can be pickled)."""
    if case.scenario == "BAD":
        raise RuntimeError("simulated BASEMENT failure")
    return {
        "scenario": case.scenario,
        "discharge": case.discharge,
        "best_vector": [0.03],
        "region_names": ["channel"],
        "best_error": 0.01,
        "n_iter_best": 3,
        "device": slot.label,
        "status": "ok",
        "elapsed": 0.0,
    }


def test_parallel_queue_cpu_and_gpu(project, machine, capsys):
    machine(one_socket_smt(), [GpuInfo(0, None)])
    gt = project["GROUND_TRUTH_FOLDER"]
    for name in ("A_Q001", "B_Q002", "BAD_Q001", "C_Q001", "MISSING_Q009"):
        _ground_truth(gt / name, name)
    options = _options(1, 1)
    cases = fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], gt, LOGGER)

    rows = batch.run(cases, options, project, runner=fake_runner)

    summary = pd.read_csv(project["OUTPUT_FOLDER"] / batch.SUMMARY_FILE_NAME)
    status = dict(zip(summary["scenario"], summary["status"], strict=True))
    assert status == {"A": "ok", "B": "ok", "BAD": "failed", "C": "ok", "MISSING": "failed"}
    assert {row["device"] for row in rows if row["status"] == "ok"} <= {"gpu:0", "cpu"}
    assert len(rows) == 5

    out = capsys.readouterr().out
    assert out.index("BOAR batch: 5 case(s), parallel") < out.index("NOTE: 4 case(s)")
    assert "MISSING_Q009  FAILED" in out


# ── Restart ──────────────────────────────────────────────────────────────────────────────────────────
OPT = {"tolerance": 1e-4, "max_no_improvement": 3, "max_tested_vectors": 10}


@pytest.mark.parametrize(
    "errors, expected",
    [
        ([0.5, 0.4, 0.6], (0, 0, False)),  # still in the initial samples
        ([0.5, 0.4, 0.6, 0.3, 0.35], (2, 1, False)),  # 2 iterations, the last without improvement
        ([0.5, 0.4, 0.6, 0.45, 0.41, 0.42], (3, 3, True)),  # stopped: 3 without improvement
        ([0.5, 0.4, 0.6, 0.3, 0.00001], (2, 0, True)),  # stopped: tolerance reached
        ([0.5, 0.4, 0.6] + [0.3 - 0.01 * i for i in range(10)], (10, 0, True)),  # budget used up
    ],
)
def test_resume_state(errors, expected):
    assert fc.resume_state(errors, 3, OPT) == expected


def test_restart_skips_finished_cases(project, machine, capsys):
    machine(one_socket_smt())
    gt = project["GROUND_TRUTH_FOLDER"]
    for name in ("A_Q001", "B_Q001"):
        _ground_truth(gt / name, name)
    # A_Q001 finished in an earlier run: its run folder holds the marker
    run_folder = project["OUTPUT_FOLDER"] / "A_Q001" / "template_case"
    run_folder.mkdir(parents=True)
    earlier = {
        **fake_runner(fio.Case("A_Q001", "A", "Q001", Path(), Path()), {}, batch.Slot("cpu", "omp", 1), {}, *[None] * 3)
    }
    earlier["device"] = "earlier-run"
    (run_folder / batch.FINISHED_FILE_NAME).write_text(json.dumps(earlier))

    options = _options(2, 0)
    cases = fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], gt, LOGGER)

    options["general_options"] = {"overwrite": False}
    batch.run(cases, options, project, runner=fake_runner)
    summary = pd.read_csv(project["OUTPUT_FOLDER"] / batch.SUMMARY_FILE_NAME).set_index("scenario")
    assert summary.loc["A", "device"] == "earlier-run"  # not run again, row kept
    assert summary.loc["B", "device"] == "cpu"
    assert "A_Q001  finished earlier, skipped" in capsys.readouterr().out

    options["general_options"] = {"overwrite": True}  # overwrite runs every case again
    batch.run(cases, options, project, runner=fake_runner)
    summary = pd.read_csv(project["OUTPUT_FOLDER"] / batch.SUMMARY_FILE_NAME).set_index("scenario")
    assert summary.loc["A", "device"] == "cpu"


def test_restart_skips_finished_single_case(project, capsys):
    _ground_truth(project["GROUND_TRUTH_FOLDER"], "CASE_Q001")  # flat layout: single mode
    (case,) = fio.discover_cases(project["SIMULATION_CONFIG_FOLDER"], project["GROUND_TRUTH_FOLDER"], LOGGER)
    run_folder = project["OUTPUT_FOLDER"] / "template_case"
    run_folder.mkdir(parents=True)
    earlier = {"scenario": "CASE", "discharge": "Q001", "best_vector": [0.03], "region_names": ["c"],
               "best_error": 0.01, "n_iter_best": 3, "device": "earlier-run", "status": "ok", "folder": str(run_folder)}  # fmt: skip
    (run_folder / batch.FINISHED_FILE_NAME).write_text(json.dumps(earlier))

    rows = batch.run([case], {"general_options": {"overwrite": False}}, project)
    assert rows == [earlier]
    assert "finished in an earlier run and is not run again" in capsys.readouterr().out
    summary = pd.read_csv(project["OUTPUT_FOLDER"] / batch.SUMMARY_FILE_NAME)
    assert summary.loc[0, "device"] == "earlier-run"
