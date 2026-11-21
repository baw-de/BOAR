#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration queue: one case (single mode) or several scenario/discharge cases (batch mode).

Batch mode runs every ``ground_truth/SCENARIO_DISCHARGE/`` folder as its own calibration. With one worker the
cases run one after the other in this process; with more workers (CPU and/or GPU, ``batch_options``) they run
at the same time in worker processes. Cores and GPU cards are placed with :mod:`src.cpu_topology` (from capiQ):
on a machine with several CPU sockets each CPU run stays inside one socket on its own physical cores, and each
GPU run gets its own card plus one core on the card's socket. Every finished case adds a row to
``output/calibration_summary.csv``; a failed case is logged, marked ``failed`` and the queue goes on.

The parallel queue, ``choose_device`` and ``kill_process_trees`` are adapted from capiQ 0.1.0
(https://github.com/KIT-IWU-HydroEng/capiQ, src/capiq/calculate_sim_queue.py and src/capiq/utils.py),
Copyright (c) 2026 Karlsruhe Institute of Technology (KIT), MIT License: see THIRD_PARTY_NOTICES.md.
"""

if __name__ == "__main__":
    raise Exception("This file must be run as module")

import concurrent.futures
import copy
import json
import logging
import multiprocessing
import os
import time
import types
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
from tqdm import tqdm

from src import cell_centroid, cpu_topology, friction_calibration, functions_io
from src.cpu_topology import Placement
from src.functions_io import Case

logger = logging.getLogger("batch")

GPU_BACKENDS = ("cuda", "cudaC", "cudaO")  # BASEMENT's CUDA solvers (BMv4_BASEHPC_<backend>)
SUMMARY_FILE_NAME = "calibration_summary.csv"
# Written into a case's run folder when its calibration has finished; with overwrite False the case is skipped
FINISHED_FILE_NAME = "calibration_finished.json"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass(frozen=True)
class Slot:
    """
    What one case runs on.

    Attributes:
        kind: ``"cpu"`` or ``"gpu"``.
        backend: BASEMENT backend (omp/seq for CPU runs, cuda/cudaC for GPU runs).
        nthreads: ``--nthreads`` of BASEMENT (-1 keeps basement_options' meaning in single mode).
        placement: CPU socket, cores and GPU card (see :class:`cpu_topology.Placement`); None runs unpinned.
    """

    kind: str
    backend: str
    nthreads: int
    placement: Placement | None = None

    @property
    def label(self) -> str:
        """Short device name for logs and the summary, e.g. ``cpu``, ``cpu:socket1``, ``gpu:0``."""
        placement = self.placement or Placement()
        if self.kind == "gpu":
            return f"gpu:{placement.cuda_device}" if placement.cuda_device is not None else "gpu"
        return f"cpu:socket{placement.node_id}" if placement.pinned else "cpu"


def single_slot(user_opt: dict) -> Slot:
    """What single mode runs on: basement_options exactly as configured, no pinning."""
    options = user_opt.get("basement_options", {})
    backend = options.get("backend", "omp")
    return Slot("gpu" if backend in GPU_BACKENDS else "cpu", backend, int(options.get("nthreads", -1)))


# ── Workers ──────────────────────────────────────────────────────────────────────────────────────────
def choose_device(preferred: str, gpu_free: bool, cpu_free: bool) -> str | None:
    """The worker for the next case: ``preferred`` when both are free, else the free one, else None (capiQ)."""
    if gpu_free and cpu_free:
        return preferred
    if gpu_free:
        return "gpu"
    if cpu_free:
        return "cpu"
    return None


class Workers:
    """
    The CPU and GPU workers of batch mode (``batch_options``) and the cores and cards they run on.

    Raises:
        ValueError: Invalid worker counts, backends or core counts.
    """

    def __init__(self, user_opt: dict):
        options = user_opt.get("batch_options") or {}
        self.n_cpu = int(options.get("n_cpu_workers", 1))
        self.n_gpu = int(options.get("n_gpu_workers", 0))
        if self.n_cpu < 0 or self.n_gpu < 0 or self.n_cpu + self.n_gpu < 1:
            raise ValueError("batch_options: n_cpu_workers + n_gpu_workers must be at least 1 (and neither negative).")

        self.gpu_backend = options.get("gpu_backend", "cudaC")
        if self.n_gpu and self.gpu_backend not in GPU_BACKENDS:
            raise ValueError(f"batch_options: gpu_backend must be one of {GPU_BACKENDS}, got {self.gpu_backend!r}.")

        self.cpu_backend = user_opt.get("basement_options", {}).get("backend", "omp")
        if self.cpu_backend in GPU_BACKENDS:
            logger.warning(f"basement_options backend {self.cpu_backend!r} is a GPU backend; CPU workers use 'omp'.")
            self.cpu_backend = "omp"

        self.preferred = str(options.get("preferred_device", "gpu")).lower()
        if self.preferred not in ("gpu", "cpu"):
            logger.warning(f"batch_options: invalid preferred_device {self.preferred!r}; using 'gpu'.")
            self.preferred = "gpu"

        pin = bool(options.get("pin_to_socket", True))
        nodes = cpu_topology.detect_topology()
        self.n_threads = self._threads(options.get("cpu_cores", -1), nodes, pin)

        # capiQ's placement: socket pinning only with several sockets, GPU cards from nvidia-smi
        topology_options = {"pin_to_socket": pin, "physical_cores_only": True}
        if options.get("gpu_numa_nodes"):
            topology_options["gpu_numa_nodes"] = options["gpu_numa_nodes"]
        self.resources = cpu_topology.setup_resources(topology_options, self.n_threads, self.n_gpu)
        self.cpu_running = 0

        if nodes is None:
            self.pinning = "CPU pinning: off (topology unknown)"
        elif len(nodes) < 2:
            self.pinning = "CPU pinning: off (single socket)"
        elif not pin:
            self.pinning = "CPU pinning: off (pin_to_socket: false)"
        elif self.resources.cpu_cores is None and self.n_cpu:
            self.pinning = f"CPU pinning: off ({self.n_threads} cores do not fit in one socket)"
        else:
            self.pinning = f"CPU pinning: on ({len(nodes)} sockets)"

    def _threads(self, cpu_cores, nodes: list[cpu_topology.NumaNode] | None, pin: bool) -> int:
        """
        Physical cores of each CPU run.

        -1: the physical cores are shared evenly between the CPU workers, inside each socket when runs are
        pinned (less one core per GPU worker on the socket, for the host thread of the GPU runs).
        """
        if self.n_cpu == 0:
            return 1
        total = sum(len(node.cores) for node in nodes) if nodes else (psutil.cpu_count(logical=False) or 1)

        if cpu_cores in (None, "None", "none") or int(cpu_cores) == -1:
            if nodes and pin and len(nodes) > 1:
                per_socket = -(-self.n_cpu // len(nodes))  # ceil
                gpu_per_socket = -(-self.n_gpu // len(nodes))
                smallest = min(len(node.cores) for node in nodes)
                return max(1, (smallest - gpu_per_socket) // per_socket)
            return max(1, (total - self.n_gpu) // self.n_cpu) if total > self.n_gpu else 1

        threads = int(cpu_cores)
        if threads < 1:
            raise ValueError("batch_options: 'cpu_cores' must be -1 or a positive integer.")
        if threads * self.n_cpu > total:
            logger.warning(
                f"{self.n_cpu} CPU worker(s) x {threads} cores = {threads * self.n_cpu} cores requested, but the "
                f"machine has {total} physical cores: runs will share cores."
            )
        return threads

    @property
    def total(self) -> int:
        return self.n_cpu + self.n_gpu

    def describe(self) -> str:
        parts = []
        if self.n_gpu:
            parts.append(f"{self.n_gpu} GPU ({self.gpu_backend})")
        if self.n_cpu:
            parts.append(f"{self.n_cpu} CPU ({self.n_threads} physical cores each)")
        return f"workers: {' + '.join(parts)} | {self.pinning}"

    def acquire(self) -> tuple[Slot, int | None] | None:
        """Take a free worker (and its cores or card) for the next case; None when none is free."""
        cpu_cores, host_cores, gpus = self.resources.cpu_cores, self.resources.host_cores, self.resources.gpus

        # A pinned CPU run also needs enough free cores inside one socket
        cpu_free = self.cpu_running < self.n_cpu
        placement = None
        if cpu_free and cpu_cores is not None:
            placement = cpu_cores.acquire(self.n_threads)
            cpu_free = placement is not None

        device = choose_device(self.preferred, gpu_free=gpus is not None and gpus.free(), cpu_free=cpu_free)
        if device != "cpu" and cpu_cores is not None:
            cpu_cores.release(placement)  # cores reserved above are not used

        if device == "gpu":
            gpu_slot, gpu_placement = gpus.acquire(host_cores)  # type: ignore[union-attr]
            return Slot("gpu", self.gpu_backend, 1, gpu_placement), gpu_slot
        if device == "cpu":
            self.cpu_running += 1
            return Slot("cpu", self.cpu_backend, self.n_threads, placement), None
        return None

    def release(self, slot: Slot, gpu_slot: int | None) -> None:
        """Give back the worker, cores and card of a finished case."""
        if slot.kind == "cpu":
            self.cpu_running -= 1
        if self.resources.host_cores is not None:
            self.resources.host_cores.release(slot.placement)
        if self.resources.gpus is not None and gpu_slot is not None:
            self.resources.gpus.release(gpu_slot)


# ── One case ─────────────────────────────────────────────────────────────────────────────────────────
def _case_options(user_opt: dict, slot: Slot) -> dict:
    """User options of one case: backend, threads and placement of its worker."""
    options = copy.deepcopy(user_opt)
    basement = options.setdefault("basement_options", {})
    basement["backend"] = slot.backend
    basement["nthreads"] = slot.nthreads
    if slot.placement is not None:
        basement["_placement"] = slot.placement
    else:
        basement.pop("_placement", None)
    return options


def run_case(
    case: Case,
    user_opt: dict,
    slot: Slot,
    folders: dict,
    log_file: str | None = None,
    inflow: dict | None = None,
    loss_function: Callable | None = None,
    in_worker: bool = False,
) -> dict:
    """
    Calibrate one case: prepare its run folder, read the data, compute centroids and optimize.

    It is a module-level function so that worker processes can run it (Windows starts them with "spawn",
    so everything the case needs is passed in, not taken from module state of the parent process).

    Args:
        case: The case.
        user_opt: The user options.
        slot: What it runs on (backend, threads, socket, cores, GPU card).
        folders: Project folders (see :func:`functions_io.default_folders`).
        log_file: Extra log file of the case (``output/<SCENARIO_DISCHARGE>/boar.log``), or None.
        inflow: Inflow discharge for the copied model.json (see :func:`functions_io.resolve_inflow`).
        loss_function: Replaces ``user_defined_configs/function_loss.loss_function`` (must be picklable for
            parallel runs), or None.
        in_worker: True in a worker process (it then also limits torch's threads to the case's cores).

    Returns:
        dict: The summary row of the case (see :func:`functions_io.write_summary_row`) plus ``elapsed`` [s]
        and ``folder`` (the run folder).
    """
    start = time.perf_counter()
    root_logger = logging.getLogger()
    handler = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="a")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(handler)
        if in_worker:
            root_logger.setLevel(logging.INFO)

    saved_loss = friction_calibration.lf
    try:
        if loss_function is not None:
            friction_calibration.lf = types.SimpleNamespace(loss_function=loss_function)  # type: ignore[assignment]
        if in_worker and slot.nthreads > 0:
            try:
                import torch

                torch.set_num_threads(slot.nthreads)
            except ImportError:
                pass

        case_logger = logging.getLogger("case")
        where = slot.placement.describe() if slot.placement is not None else "unpinned"
        case_logger.info(
            f"Case {case.name or 'single'} on {slot.label} ({where}; backend {slot.backend}, {slot.nthreads} threads)"
        )
        options = _case_options(user_opt, slot)
        project_path = Path(folders.get("PROJECT_ROOT", Path.cwd()))
        paths = functions_io.setup_directories(
            project_path, options, case_logger, folders=folders, case=case, inflow=inflow
        )
        data = functions_io.main(paths)
        data = cell_centroid.main(data, paths, options)
        result = friction_calibration.main(data, paths, options)

        if case.batch:
            scenario, discharge = case.scenario, case.discharge
        else:
            scenario, discharge = functions_io.scenario_discharge_from_ground_truth(data["ground_truth_data"].keys())

        row = {
            "scenario": scenario,
            "discharge": discharge,
            **result,
            "device": slot.label,
            "status": "ok",
            "elapsed": time.perf_counter() - start,
            "folder": paths["folder_simulation"],
        }
        # Marks the case as finished: a restart with overwrite False skips it
        with open(Path(paths["folder_simulation"]) / FINISHED_FILE_NAME, "w") as f:
            json.dump(row, f, indent=4)
        return row
    finally:
        friction_calibration.lf = saved_loss
        if handler is not None:
            root_logger.removeHandler(handler)
            handler.close()


def _run_case_in_worker(
    case: Case,
    user_opt: dict,
    slot: Slot,
    folders: dict,
    log_file: str | None = None,
    inflow: dict | None = None,
    loss_function: Callable | None = None,
) -> dict:
    """Entry point of worker processes."""
    return run_case(case, user_opt, slot, folders, log_file, inflow, loss_function, in_worker=True)


def _failed_row(case: Case, slot: Slot | None) -> dict:
    return {
        "scenario": case.scenario or case.name,
        "discharge": case.discharge,
        "device": slot.label if slot else None,
        "status": "failed",
    }


def kill_process_trees(pids: list[int]) -> None:
    """
    Kill processes and all the processes they started (capiQ).

    BMv4_simulation starts the solver (BMv4_BASEHPC_*) as a child, and pinned runs add a launcher in front
    (cmd.exe, numactl or taskset), so stopping only the direct child would leave the solver running. The whole
    tree is read first and then killed: once a process is gone, its children can no longer be found.
    """
    processes: list[psutil.Process] = []
    for pid in pids:
        try:
            root = psutil.Process(pid)
            processes += [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
    for process in processes:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(processes, timeout=5)


# ── Queue ────────────────────────────────────────────────────────────────────────────────────────────
def finished_row(case: Case, folders: dict) -> dict | None:
    """The summary row of a case that finished in an earlier run (its run folder holds the finished marker)."""
    name = functions_io.simulation_name(case.config_folder)
    if name is None:
        return None
    # Batch mode: output/<SCENARIO_DISCHARGE>/<simulation_name>/; single mode: output/<simulation_name>/
    case_root = Path(folders["OUTPUT_FOLDER"]) / case.name if case.name else Path(folders["OUTPUT_FOLDER"])
    marker = case_root / name / FINISHED_FILE_NAME
    try:
        with open(marker) as f:
            row: dict = json.load(f)
            return row
    except (OSError, ValueError):
        return None


def _preflight(cases: list[Case], user_opt: dict, folders: dict) -> tuple[list[tuple[Case, dict | None]], dict]:
    """Check every batch case before the queue starts: the inflow discharge must resolve."""
    simulation_options = user_opt.get("simulation_options", {})
    discharge_folder = functions_io.discharge_files_folder(simulation_options, folders, logger)
    table = functions_io.read_discharge_table(discharge_folder)
    ready: list[tuple[Case, dict | None]] = []
    failed: dict = {}
    for case in cases:
        if case.error:
            failed[case.name] = case.error
            continue
        try:
            # Time series or constant, as the inflow boundary of the model.json of the case gives it
            kind = functions_io.read_inflow_discharge_kind(case.config_folder)
            inflow = functions_io.resolve_inflow(str(case.discharge), kind, table, discharge_folder)
        except (FileNotFoundError, KeyError) as exc:
            failed[case.name] = str(exc).strip("'\"")
            continue
        ready.append((case, inflow))
    return ready, failed


def _notice(cases, ready, failed, done, workers: Workers, folders, summary_path) -> list[str]:
    """Console notice of a batch run, printed before the progress bar."""
    mode = "parallel" if workers.total > 1 else "serial"
    lines = [f"BOAR batch: {len(cases)} case(s), {mode} | {workers.describe()}"]
    if done:
        lines.append(
            f"  RESUME: {len(done)} case(s) finished in an earlier run and are skipped (overwrite: False); "
            "interrupted cases continue from their optimization memory."
        )

    config_root = Path(folders["SIMULATION_CONFIG_FOLDER"])
    shared = [case for case, _ in ready if case.shared_config]
    if shared:
        lines.append(
            f"  NOTE: {len(shared)} case(s) have no folder in {config_root.name}/ and replicate the template "
            f"{config_root.name}/model.json, with the inflow discharge replaced per case."
        )
    width = max(len(str(case.name)) for case in cases)
    for case in cases:
        if case.name in failed:
            lines.append(f"    {case.name:<{width}}  FAILED: {failed[case.name]}")
            continue
        if case.name in done:
            lines.append(f"    {case.name:<{width}}  finished earlier, skipped")
            continue
        inflow = next(i for c, i in ready if c is case)
        if inflow is None:
            discharge = "as in model.json (no uniform_in/froude_in inflow)"
        elif "discharge_file" in inflow:
            discharge = Path(inflow["discharge_file"]).name
        else:
            discharge = f"{inflow['discharge']} m3/s"
        source = "template  " if case.shared_config else f"own config ({config_root.name}/{case.name}/)"
        lines.append(f"    {case.name:<{width}}  {source} -> discharge {discharge}")
    lines.append(f"  Summary: {summary_path}")
    return lines


def _progress(total: int) -> tqdm:
    return tqdm(
        total=total,
        desc="Cases",
        unit="case",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def _report(case: Case, row: dict, summary_path: Path, pbar: tqdm | None) -> None:
    """Write the summary row and tell the user (through tqdm, so the bar keeps its line)."""
    functions_io.write_summary_row(summary_path, row)
    if row["status"] == "ok":
        message = (
            f"  {case.name}: ok on {row['device']} in {row.get('elapsed', 0):.0f} s, best error "
            f"{row['best_error']:.6g} at evaluation {row['n_iter_best']}"
        )
        logger.info(message.strip())
    else:
        message = f"  {case.name}: FAILED on {row.get('device')} (see the log)"
        logger.error(message.strip())
    if pbar is not None:
        tqdm.write(message)
        pbar.update(1)


def _centroid_prepass(folders: dict, user_opt: dict) -> None:
    """Compute the mesh centroids once, so that parallel workers only read the cached files."""
    paths = {
        "folder_mesh": folders["MESH_FOLDER"],
        "folder_centroids": os.path.join(folders["MESH_FOLDER"], "mesh_centroids"),
    }
    try:
        cell_centroid.main({"mesh": functions_io.find_available_meshes(paths)}, paths, user_opt)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Mesh centroids could not be computed before the queue: {exc}")


def run(
    cases: list[Case],
    user_opt: dict,
    folders: dict | None = None,
    loss_function: Callable | None = None,
    runner: Callable | None = None,
) -> list[dict]:
    """
    Run the calibration queue and write the summary CSV at the output root.

    Args:
        cases: From :func:`functions_io.discover_cases` (one case without name: single mode).
        user_opt: The user options.
        folders: Project folders; default from constants.py.
        loss_function: Optional replacement of the user's loss function (picklable for parallel runs).
        runner: Replaces the worker entry point (tests); forces worker processes.

    Returns:
        list[dict]: The summary rows, in the order the cases finished.
    """
    folders = folders or functions_io.default_folders()
    summary_path = Path(folders["OUTPUT_FOLDER"]) / SUMMARY_FILE_NAME

    # Single mode: as before batch mode, failures raise
    if len(cases) == 1 and not cases[0].batch:
        # Restart (overwrite False) of a case that finished earlier: not calibrated again
        finished = None if functions_io.overwrite_option(user_opt, logger) else finished_row(cases[0], folders)
        if finished is not None:
            message = (
                f"BOAR: the calibration in {finished.get('folder')} finished in an earlier run and is not run again "
                "(overwrite: False). Set general_options.overwrite: True to calibrate it again."
            )
            logger.info(message)
            print(message, flush=True)
            functions_io.write_summary_row(summary_path, finished)
            return [finished]
        row = run_case(cases[0], user_opt, single_slot(user_opt), folders, loss_function=loss_function)
        functions_io.write_summary_row(summary_path, row)
        return [row]

    workers = Workers(user_opt)
    ready, failed = _preflight(cases, user_opt, folders)

    # Restart (overwrite False): cases that finished in an earlier run are not calibrated again
    done: dict = {}
    if not functions_io.overwrite_option(user_opt, logger):
        for case, _ in ready:
            finished = finished_row(case, folders)
            if finished is not None:
                done[case.name] = finished
        ready = [(case, inflow) for case, inflow in ready if case.name not in done]

    lines = _notice(cases, ready, failed, done, workers, folders, summary_path)
    for line in lines:
        logger.info(line.strip())
    print("\n".join(lines), flush=True)  # before the progress bar, which then keeps its own line

    rows = []
    for case in cases:
        if case.name in failed:
            row = _failed_row(case, None)
        elif case.name in done:
            row = done[case.name]
        else:
            continue
        functions_io.write_summary_row(summary_path, row)  # keeps the summary complete after a restart
        rows.append(row)

    if not ready:
        return rows

    _centroid_prepass(folders, user_opt)

    with _progress(len(ready)) as pbar:
        if workers.total == 1 and runner is None:
            rows.extend(_run_serial(ready, user_opt, workers, folders, loss_function, summary_path, pbar))
        else:
            rows.extend(_run_parallel(ready, user_opt, workers, folders, loss_function, runner, summary_path, pbar))
    return rows


def _log_file(folders: dict, case: Case) -> str:
    return str(Path(folders["OUTPUT_FOLDER"]) / str(case.name) / "boar.log")


def _run_serial(ready, user_opt, workers: Workers, folders, loss_function, summary_path, pbar) -> list[dict]:
    """One case after the other in this process, all on the same worker (and cores or card)."""
    acquired = workers.acquire()
    assert acquired is not None  # one free worker
    slot, gpu_slot = acquired
    if slot.placement is not None and (slot.placement.pinned or slot.placement.cuda_device is not None):
        logger.info(f"Cases run on {slot.placement.describe()}")

    rows = []
    try:
        for case, inflow in ready:
            logger.info(f"Case {case.name} started on {slot.label}")
            try:
                row = run_case(case, user_opt, slot, folders, _log_file(folders, case), inflow, loss_function)
            except Exception:
                logger.exception(f"Case {case.name} failed")
                row = _failed_row(case, slot)
            _report(case, row, summary_path, pbar)
            rows.append(row)
    except KeyboardInterrupt:
        logger.warning("Interrupted: stopping the running BASEMENT run")
        pbar.close()
        kill_process_trees([child.pid for child in psutil.Process().children()])
        raise
    finally:
        workers.release(slot, gpu_slot)
    return rows


def _run_parallel(ready, user_opt, workers: Workers, folders, loss_function, runner, summary_path, pbar) -> list[dict]:
    """capiQ's queue: give the next case to a free worker (GPU first by default) until all cases are done."""
    runner = runner or _run_case_in_worker
    pending = deque(ready)
    running: dict = {}  # future -> (case, slot, gpu slot)
    rows = []

    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers.total, mp_context=context)
    try:
        while pending or running:
            while pending:
                acquired = workers.acquire()
                if acquired is None:
                    break
                slot, gpu_slot = acquired
                case, inflow = pending.popleft()
                future = executor.submit(
                    runner, case, user_opt, slot, folders, _log_file(folders, case), inflow, loss_function
                )
                running[future] = (case, slot, gpu_slot)
                where = slot.placement.describe() if slot.placement is not None else "unpinned"
                logger.info(f"Case {case.name} started on {slot.label} ({where})")

            if not running:  # nothing fits (should not happen): avoid waiting forever
                raise RuntimeError("No worker can take the next case; check batch_options.")
            done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                case, slot, gpu_slot = running.pop(future)
                workers.release(slot, gpu_slot)
                try:
                    row = future.result()
                except Exception:
                    logger.exception(f"Case {case.name} failed")
                    row = _failed_row(case, slot)
                _report(case, row, summary_path, pbar)
                rows.append(row)
    except KeyboardInterrupt:
        logger.warning("Interrupted: stopping the running cases and their BASEMENT runs")
        pbar.close()
        # All worker processes at once, with the BASEMENT runs they started
        # (killing one worker makes the pool stop the others, which would orphan their runs)
        kill_process_trees(list(getattr(executor, "_processes", None) or {}))
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return rows
