"""
Socket detection, core and GPU allocation and the pinned launch of src/cpu_topology.py.

Taken from capiQ 0.2.0 (https://github.com/KIT-IWU-HydroEng/capiQ, tests/test_cpu_topology.py),
Copyright (c) 2026 Karlsruhe Institute of Technology (KIT), MIT License: see THIRD_PARTY_NOTICES.md. The tests
of capiQ's own queue and logs are left out. No BASEMENT or multi-socket machine is needed.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from src import cpu_topology
from src.cpu_topology import CoreAllocator, GpuInfo, GpuSlots, NumaNode, Placement


def numa_record(node: int, group: int, mask: int) -> bytes:
    body = struct.pack("<I18xH", node, 1) + struct.pack("<QH6x", mask, group)
    return struct.pack("<II", 1, 8 + len(body)) + body


def core_record(group: int, mask: int) -> bytes:
    smt = 1 if bin(mask).count("1") > 1 else 0
    body = struct.pack("<BB20xH", smt, 0, 1) + struct.pack("<QH6x", mask, group)
    return struct.pack("<II", 0, 8 + len(body)) + body


def two_sockets_smt() -> list[NumaNode]:
    """2 x 8 cores with hyper-threading, one processor group per socket."""
    numa = numa_record(0, 0, (1 << 16) - 1) + numa_record(1, 1, (1 << 16) - 1)
    cores = b"".join(core_record(group, 0b11 << (2 * i)) for group in (0, 1) for i in range(8))
    return cpu_topology.parse_topology(numa, cores)


def two_sockets() -> list[NumaNode]:
    """2 x 8 cores without hyper-threading, both sockets in one processor group."""
    numa = numa_record(0, 0, (1 << 8) - 1) + numa_record(1, 0, ((1 << 8) - 1) << 8)
    cores = b"".join(core_record(0, 1 << i) for i in range(16))
    return cpu_topology.parse_topology(numa, cores)


def test_parse_topology_with_smt():
    nodes = two_sockets_smt()
    assert [(n.node_id, n.group, len(n.cores)) for n in nodes] == [(0, 0, 8), (1, 1, 8)]
    assert nodes[1].cores[0] == (0, 1) and nodes[1].base == 0


def test_parse_topology_one_group():
    nodes = two_sockets()
    assert [(n.node_id, n.group, len(n.cores)) for n in nodes] == [(0, 0, 8), (1, 0, 8)]
    assert nodes[1].cores[0] == (8,) and nodes[1].base == 8


def test_detect_topology_of_this_machine():
    nodes = cpu_topology.detect_topology()
    if nodes is None:
        pytest.skip(f"topology not readable on {sys.platform}")
    assert all(node.cores for node in nodes)


def test_allocator_spreads_and_never_overlaps():
    cores = CoreAllocator(two_sockets())
    placements = [cores.acquire(4) for _ in range(4)]

    assert [p.node_id for p in placements] == [0, 1, 0, 1]
    assert placements[1].cpus == tuple(range(8, 12))
    # Masks for start /NODE are relative to the socket's first CPU
    assert [p.node_mask for p in placements] == [0xF, 0xF, 0xF0, 0xF0]
    assert cores.acquire(4) is None

    cores.release(placements[1])
    assert cores.acquire(4) == placements[1]


def test_allocator_physical_cores_only():
    placement = CoreAllocator(two_sockets_smt()).acquire(4)
    assert placement.cpus == (0, 2, 4, 6) and placement.node_mask == 0x55

    placement = CoreAllocator(two_sockets_smt(), physical_only=False).acquire(4)
    assert placement.cpus == (0, 1, 2, 3) and placement.node_mask == 0xF


def test_allocator_fits_one_socket():
    assert CoreAllocator(two_sockets()).fits(8)
    assert not CoreAllocator(two_sockets()).fits(9)
    assert not CoreAllocator(two_sockets_smt()).fits(9)
    assert CoreAllocator(two_sockets_smt(), physical_only=False).fits(16)


def test_gpu_on_its_socket():
    cores = CoreAllocator(two_sockets())
    gpus = GpuSlots([GpuInfo(0, 0), GpuInfo(1, 1)], 2)

    slot0, first = gpus.acquire(cores)
    slot1, second = gpus.acquire(cores)
    assert (first.cuda_device, first.node_id, first.cpus) == (0, 0, (0,))
    assert (second.cuda_device, second.node_id, second.cpus) == (1, 1, (8,))
    assert not gpus.free()

    # The host cores are taken from the CPU runs: 7 free cores left on each socket
    assert cores.free_cores(0) == cores.free_cores(1) == 7

    gpus.release(slot0)
    cores.release(first)
    assert cores.free_cores(0) == 8


def test_gpu_prefers_socket_with_free_core():
    cores = CoreAllocator(two_sockets())
    cores.acquire(8, node_id=0)
    _, placement = GpuSlots([GpuInfo(0, 0), GpuInfo(1, 1)], 2).acquire(cores)
    assert (placement.cuda_device, placement.node_id) == (1, 1)


def test_gpu_least_busy_card_and_unknown_socket():
    gpus = GpuSlots([GpuInfo(0, None), GpuInfo(1, None)], 3)
    devices = [gpus.acquire(None)[1].cuda_device for _ in range(2)]
    assert devices == [0, 1]

    _, placement = GpuSlots([GpuInfo(0, None)], 1).acquire(CoreAllocator(two_sockets()))
    assert placement == Placement(cuda_device=0)

    # Without nvidia-smi the card is left to BASEMENT
    assert GpuSlots(None, 2).acquire(None)[1].cuda_device is None


def test_parse_pnp_numa():
    # An adapter without NUMA node is skipped; a single adapter comes as a dict
    data = [{"bus": 193, "address": 0, "numa": 1}, {"bus": None, "address": None, "numa": None}]
    assert cpu_topology.parse_pnp_numa(data) == {(193, 0, 0): 1}
    assert cpu_topology.parse_pnp_numa({"bus": 5, "address": (3 << 16) | 1, "numa": 0}) == {(5, 3, 1): 0}


def test_parse_nvidia_smi():
    output = "0, 00000000:41:00.0\n1, 00000000:C1:00.0\n"
    assert cpu_topology.parse_nvidia_smi(output) == [(0, (0, 0x41, 0, 0)), (1, (0, 0xC1, 0, 0))]


def test_pinned_command_and_env():
    placement = Placement(node_id=1, cpus=tuple(range(8, 12)), node_mask=0xF, cuda_device=1)
    command = cpu_topology.pinned_command([r"C:\BASEMENT 4\BMv4_simulation.exe", "a b.json"], placement, "win32")
    assert command == (
        'cmd /d /c "start "" /B /WAIT /NODE 1 /AFFINITY 0xf "C:\\BASEMENT 4\\BMv4_simulation.exe" "a b.json" & exit"'
    )
    env = cpu_topology.launch_env(placement)
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert cpu_topology.launch_env(Placement(cuda_device=None)) is None


def fake_sys(root: Path, sockets: int, cores_per_socket: int, smt: bool, with_nodes: bool = True) -> Path:
    """A /sys/devices/system tree; with SMT, core c of socket s has CPUs n and n + total cores (Linux numbering)."""
    total = sockets * cores_per_socket
    for s in range(sockets):
        cpus = [s * cores_per_socket + c for c in range(cores_per_socket)]
        node_cpus = cpus + [cpu + total for cpu in cpus] if smt else cpus
        if with_nodes:
            node = root / "node" / f"node{s}"
            node.mkdir(parents=True)
            (node / "cpulist").write_text(",".join(map(str, node_cpus)) + "\n")
        for cpu in cpus:
            siblings = f"{cpu},{cpu + total}" if smt else f"{cpu}"
            for logical in [cpu, cpu + total] if smt else [cpu]:
                topology = root / "cpu" / f"cpu{logical}" / "topology"
                topology.mkdir(parents=True)
                (topology / "thread_siblings_list").write_text(siblings + "\n")
    return root


def test_parse_cpulist():
    assert cpu_topology.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert cpu_topology.parse_cpulist("5") == [5]


def test_linux_topology(tmp_path):
    root = fake_sys(tmp_path, sockets=2, cores_per_socket=4, smt=True)
    nodes = cpu_topology.linux_topology(root, allowed=set(range(16)))
    assert [(n.node_id, len(n.cores)) for n in nodes] == [(0, 4), (1, 4)]
    assert nodes[1].cores[0] == (4, 12)  # hyper-thread siblings form one physical core

    # Same allocator as on Windows: one thread per physical core, inside one socket
    placement = CoreAllocator(nodes).acquire(4)
    assert placement.node_id == 0 and placement.cpus == (0, 1, 2, 3)


def test_linux_topology_allowed_cpus_and_no_numa(tmp_path):
    root = fake_sys(tmp_path / "a", sockets=2, cores_per_socket=4, smt=False)
    # A Slurm job or container may use only some CPUs: the others are never pinned
    nodes = cpu_topology.linux_topology(root, allowed={0, 1, 4})
    assert [n.cores for n in nodes] == [((0,), (1,)), ((4,),)]

    # Without NUMA information all CPUs form one socket
    root = fake_sys(tmp_path / "b", sockets=1, cores_per_socket=4, smt=False, with_nodes=False)
    nodes = cpu_topology.linux_topology(root, allowed={0, 1, 2, 3})
    assert len(nodes) == 1 and len(nodes[0].cores) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="PCI folder names contain ':', not allowed in Windows file names")
def test_linux_gpu_numa_node(tmp_path):
    for name, value in [("0000:c1:00.0", "1"), ("0000:41:00.0", "-1")]:
        device = tmp_path / "bus" / "pci" / "devices" / name
        device.mkdir(parents=True)
        (device / "numa_node").write_text(value + "\n")
    assert cpu_topology.linux_gpu_numa_node((0, 0xC1, 0, 0), tmp_path) == 1
    assert cpu_topology.linux_gpu_numa_node((0, 0x41, 0, 0), tmp_path) is None  # -1: unknown
    assert cpu_topology.linux_gpu_numa_node((0, 0x05, 0, 0), tmp_path) is None  # missing


def test_linux_pinned_command(monkeypatch):
    placement = Placement(node_id=1, cpus=(8, 9, 10, 11))
    monkeypatch.setattr(cpu_topology.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    assert cpu_topology.pinned_command(["BMv4_simulation", "a.json"], placement, "linux") == [
        "numactl", "--physcpubind=8-11", "--localalloc", "BMv4_simulation", "a.json"
    ]  # fmt: skip
    monkeypatch.setattr(cpu_topology.shutil, "which", lambda tool: "/usr/bin/taskset" if tool == "taskset" else None)
    assert cpu_topology.pinned_command(["BMv4_simulation"], placement, "linux") == [
        "taskset", "-c", "8-11", "BMv4_simulation"
    ]  # fmt: skip


def test_launch_env_openmp_binding(monkeypatch):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.setenv("OMP_PLACES", "threads")  # set by the user: kept
    env = cpu_topology.launch_env(Placement(node_id=0, cpus=(0,)))
    assert env["OMP_PROC_BIND"] == "close" and env["OMP_PLACES"] == "threads"
    assert "CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS cannot pin processes to cores")
@pytest.mark.parametrize("tool", ["numactl", "taskset"])
def test_pinned_launch_real(tool, monkeypatch):
    """A program started through pinned_command runs on the requested CPU only, and its exit code comes back."""
    if sys.platform == "win32":
        if tool == "taskset":
            pytest.skip("one launcher on Windows")
    elif shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed")
    else:
        which = shutil.which  # the original: cpu_topology.shutil is this same module
        monkeypatch.setattr(cpu_topology.shutil, "which", lambda name: which(name) if name == tool else None)

    placement = Placement(node_id=0, cpus=(0,), node_mask=0b1)
    program = [sys.executable, "-c", "import sys, psutil; print(psutil.Process().cpu_affinity()); sys.exit(3)"]
    command = cpu_topology.pinned_command(program, placement)
    completed = subprocess.run(
        command, capture_output=True, text=True, env=cpu_topology.launch_env(placement), timeout=60
    )
    assert completed.returncode == 3, completed.stderr
    assert completed.stdout.strip() == "[0]"
