#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BASEMENT benchmark: steady flows in MacDonald channels with analytical solutions (SWASHES, section 3.5).

Light, self-contained BASEMENT 4.x (BASEplane 2D) cases with analytical reference solutions. They check that
BASEMENT and BOAR work together on the user's machine. Three cases cover the three flow regimes:

    ``subcritical``   section 3.5.1, subcritical flow in a short domain (default)
    ``supercritical`` section 3.5.2, supercritical flow in a short domain
    ``jump``          section 3.5.6, smooth transition followed by a hydraulic jump in a long domain

For each case:

1. The analytical solution, the structured mesh (.2dm) and the BASEMENT configuration are generated under
   ``output/benchmarks/<variant>/``.
2. The ``analytical_manning`` run: BASEMENT with the analytical Manning coefficient (n = 0.03), compared with the
   analytical water depth (information only).
3. The ``BOAR calibrated`` run: a BOAR calibration of the Manning coefficients of the friction regions (thirds of
   the short channels, before and after the jump), then BASEMENT again with the calibrated coefficients; its final
   water depth is compared with the analytical one (pass criterion).

The calibrated Manning coefficient is an *effective* value: it absorbs numerical effects (discretisation, mesh
resolution, numerical diffusion) and model differences, so it is not expected to equal the analytical n = 0.03.
The main model difference of the rectangular cases is the side-wall friction: the pseudo-2D solutions include it
through the hydraulic radius A/P, while the BASEMENT mesh ends at the walls, which are free-slip boundaries
(u.n = 0, du/dn = 0) without friction. In the trapezoidal case (3.5.6) the banks are part of the bathymetry, so
BASEMENT applies friction on them.

Cases (SWASHES section 3.5, Table 2), all with Q = 20 m3/s and n = 0.03 s/m^(1/3):
    - 3.5.1: L = 200 m, rectangular, B1(x) = 10 - 5 exp(-10 (x/200 - 1/2)^2),
      h(x) = 0.9 + 0.3 exp(-20 (x/200 - 1/2)^2); Q upstream, h_out downstream.
    - 3.5.2: same channel, h(x) = 0.5 + 0.5 exp(-20 (x/200 - 1/2)^2); Q and h_in upstream, free outflow.
    - 3.5.6: L = 400 m, isosceles trapezoid with side slope Z = 2 and bottom width
      B2(x) = 10 - 5 exp(-50 (x/400 - 1/3)^2) - 5 exp(-50 (x/400 - 2/3)^2);
      h(x) = 0.9 + 0.25 (exp(-x/40) - 1) + 0.25 exp(15 (x/400 - 3/10)) for x <= 120 m and
      h(x) = exp(-p (x - x*)) sum_i k_i ((x - x*)/(x** - x*))^i + phi(x) beyond, with x* = 120 m, x** = 400 m,
      p = 0.09, k = (-0.183691, 1.519577, -18.234429), phi(x) = 1.5 exp(0.16 (x/400 - 1)) - 0.3 exp(2 (x/400 - 1));
      Q upstream, h_out = 1.2 m downstream; hydraulic jump at x = 120 m.
    - Bed slope S0(x) from the closed-form expression of section 3.5, topography z(x) = int_x^L S0 dX.

Notes on the paper: for 3.5.1, Table 2 gives h_out = 0.902921 m while h(200) = 0.902021 m; the benchmark imposes
h(L) so that the boundary condition and the analytical profile agree. For 3.5.6, the paper prints the range of
the second branch as "120 m <= x <= 200 m"; with x** = 400 m and h(400) = 1.2 m = h_out it is 120-400 m.

Reference:
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.N.T., James, F., Cordier, S.
    (2013). SWASHES: a compilation of Shallow Water Analytic Solutions for Hydraulic and Environmental
    Studies. International Journal for Numerical Methods in Fluids, 72(3), 269-300.
    DOI: https://doi.org/10.1002/fld.3741. HAL: http://hal.archives-ouvertes.fr/hal-00628246

    The optional cross-check in ``tests/test_basement_macdonald.py`` uses pyswashes
    (https://github.com/lrntct/pyswashes, GPL-3.0), a Python wrapper of the SWASHES program.

Run:
    >>> uv run python benchmarks/basement_macdonald.py                         # subcritical case
    >>> uv run python benchmarks/basement_macdonald.py --case supercritical    # or --case jump
    >>> uv run python benchmarks/basement_macdonald.py --analytical-manning    # analytical_manning run only
    >>> uv run python benchmarks/basement_macdonald.py --turbulence k_epsilon  # k-epsilon (default of the jump)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import basement_tools as bmt
from src import batch, cell_centroid, friction_calibration, functions_io, utils
from user_defined_configs import constants as c

Array = np.ndarray

# ── Physical parameters (SWASHES section 3.5) ────────────────────────────────────────────────────────
GRAVITY = 9.81  # [m/s2]
DISCHARGE = 20.0  # [m3/s]
MANNING_N = 0.03  # [s/m^(1/3)] analytical Manning coefficient

# ── Benchmark set-up ────────────────────────────────────────────────────────────────────────────────
CASE_NAME = "swashes_macdonald"
# Friction regions along the channel, as (name, x_start, x_end); the short channels use thirds, the jump case
# one region before and one after the jump. BOAR calibrates one Manning coefficient per region.
THIRDS = (("upstream", 0.0, 200.0 / 3.0), ("contraction", 200.0 / 3.0, 400.0 / 3.0), ("downstream", 400.0 / 3.0, 200.0))
JUMP_REGIONS = (("before_jump", 0.0, 120.0), ("after_jump", 120.0, 400.0))
DISCHARGE_CODE = "Q001"
BENCHMARK_FOLDER = ROOT / "output" / "benchmarks" / CASE_NAME
DX = 2.0  # [m] mesh cell length along the channel
NY = 8  # mesh cells across the flat bottom
BANK_CELLS = 4  # mesh cells across each sloping bank (trapezoidal case)
BANK_HEIGHT = 2.0  # [m] bank crest above the bed, above the maximum water depth (1.27 m)
TURBULENCE_MODELS = ("none", "k_epsilon")  # no turbulence model, or BASEMENT's k-epsilon (default per case)
TURBULENCE_START = 300.0  # [s] k-epsilon starts once the channel has filled from the dry start
RAMP_TIME = 10.0  # [s] the inflow discharge rises from 0 to Q over this time (dry start, no mass imbalance)
CALIBRATION_START_N = 0.045  # initial vector of the calibration (away from the analytical value)
CALIBRATION_BOUNDS = (0.015, 0.060)
CALIBRATION_PRECISION = 0.001
DEPTH_RMSE_TOL = 0.05  # [m] pass criterion: RMSE of the final water depth against the analytical depth


# ── Analytical solutions ────────────────────────────────────────────────────────────────────────────
def _gauss(x: Array, length: float, centre: float, rate: float) -> Array:
    """exp(-rate (x/length - centre)^2) and its derivative."""
    s = x / length - centre
    e = np.exp(-rate * s**2)
    return e, -2.0 * rate * s / length * e


def _b1(x: Array) -> tuple[Array, Array]:
    e, de = _gauss(x, 200.0, 0.5, 10.0)
    return 10.0 - 5.0 * e, -5.0 * de


def _b2(x: Array) -> tuple[Array, Array]:
    e1, de1 = _gauss(x, 400.0, 1.0 / 3.0, 50.0)
    e2, de2 = _gauss(x, 400.0, 2.0 / 3.0, 50.0)
    return 10.0 - 5.0 * e1 - 5.0 * e2, -5.0 * de1 - 5.0 * de2


def _h_subcritical(x: Array) -> tuple[Array, Array]:
    e, de = _gauss(x, 200.0, 0.5, 20.0)
    return 0.9 + 0.3 * e, 0.3 * de


def _h_supercritical(x: Array) -> tuple[Array, Array]:
    e, de = _gauss(x, 200.0, 0.5, 20.0)
    return 0.5 + 0.5 * e, 0.5 * de


JUMP_X, JUMP_X2, JUMP_P, JUMP_K = 120.0, 400.0, 0.09, (-0.183691, 1.519577, -18.234429)


def _h_jump(x: Array) -> tuple[Array, Array]:
    # Upstream branch (x <= 120 m): subcritical, smooth transition to supercritical
    e1, e2 = np.exp(-x / 40.0), np.exp(15.0 * (x / 400.0 - 0.3))
    h_up = 0.9 + 0.25 * (e1 - 1.0) + 0.25 * e2
    dh_up = -0.25 / 40.0 * e1 + 0.25 * 15.0 / 400.0 * e2
    # Downstream branch (x > 120 m): subcritical after the jump
    s = (x - JUMP_X) / (JUMP_X2 - JUMP_X)
    decay = np.exp(-JUMP_P * (x - JUMP_X))
    poly = JUMP_K[0] + JUMP_K[1] * s + JUMP_K[2] * s**2
    dpoly = (JUMP_K[1] + 2.0 * JUMP_K[2] * s) / (JUMP_X2 - JUMP_X)
    f1, f2 = np.exp(0.16 * (x / 400.0 - 1.0)), np.exp(2.0 * (x / 400.0 - 1.0))
    h_down = decay * poly + 1.5 * f1 - 0.3 * f2
    dh_down = decay * (dpoly - JUMP_P * poly) + 1.5 * 0.16 / 400.0 * f1 - 0.3 * 2.0 / 400.0 * f2
    upstream = x <= JUMP_X
    return np.where(upstream, h_up, h_down), np.where(upstream, dh_up, dh_down)


@dataclass(frozen=True)
class MacDonaldCase:
    """
    One MacDonald pseudo-2D case of SWASHES (section 3.5).

    ``bottom`` and ``depth`` return a value and its x-derivative: the bottom width B(x) and the mean water depth
    h(x). The cross section is an isosceles trapezoid with side slope ``bank_slope`` Z (Z = 0: rectangle).
    """

    key: str
    section: str
    title: str
    regime: str  # "subcritical" (Q upstream, h_out downstream) or "supercritical" (Q and h_in upstream)
    length: float
    bank_slope: float
    bottom: Callable[[Array], tuple[Array, Array]]
    depth: Callable[[Array], tuple[Array, Array]]
    end_time: float  # [s] simulated time, enough to reach the steady state from a dry start
    gt_spacing: float  # [m] spacing of the comparison points along the centre line
    jump_at: float | None = None
    inflow: str = "uniform_in"  # subcritical inflow boundary: "uniform_in" or "froude_in"
    turbulence: str = "none"  # default turbulence model of the case: "none" or "k_epsilon"
    robust_error_scale: bool = False  # error map scale at the 90th percentile (a few large errors saturate)
    regions: tuple[tuple[str, float, float], ...] = (("channel", 0.0, float("inf")),)  # friction regions

    def region_index(self, x) -> Array:
        """1-based friction region of the positions x (the last region also takes x = L)."""
        starts = np.array([start for _, start, _ in self.regions])
        return np.searchsorted(starts, np.asarray(x, dtype=float), side="right")

    @property
    def region_names(self) -> list[str]:
        return [name for name, _, _ in self.regions]

    def width(self, x) -> Array:
        """Bottom width B(x) [m]."""
        return self.bottom(np.asarray(x, dtype=float))[0]

    def dwidth(self, x) -> Array:
        """dB/dx [-]."""
        return self.bottom(np.asarray(x, dtype=float))[1]

    def h(self, x) -> Array:
        """Analytical mean water depth h(x) [m]."""
        return self.depth(np.asarray(x, dtype=float))[0]

    def dh(self, x) -> Array:
        """dh/dx [-]."""
        return self.depth(np.asarray(x, dtype=float))[1]

    def area(self, x) -> Array:
        """Wetted cross-section area (B + Z h) h [m2]."""
        h = self.h(x)
        return (self.width(x) + self.bank_slope * h) * h

    def mean_velocity(self, x) -> Array:
        """Cross-section mean velocity Q / A [m/s]."""
        return DISCHARGE / self.area(x)

    def froude(self, x) -> Array:
        """Froude number U / sqrt(g A / T), with the top width T = B + 2 Z h."""
        top = self.width(x) + 2.0 * self.bank_slope * self.h(x)
        return self.mean_velocity(x) / np.sqrt(GRAVITY * self.area(x) / top)

    def bed_slope(self, x) -> Array:
        """
        Bed slope S0(x) of the pseudo-2D MacDonald channels (SWASHES section 3.5):

        S0 = (1 - Q^2 (B + 2Zh) / (g h^3 (B + Zh)^3)) h'
             + Q^2 n^2 (B + 2h sqrt(1 + Z^2))^(4/3) / (h^(10/3) (B + Zh)^(10/3))
             - Q^2 B' / (g h^2 (B + Zh)^3)
        """
        b, db = self.bottom(np.asarray(x, dtype=float))
        h, dh = self.depth(np.asarray(x, dtype=float))
        z = self.bank_slope
        q2 = DISCHARGE**2
        froude_term = (1.0 - q2 * (b + 2.0 * z * h) / (GRAVITY * h**3 * (b + z * h) ** 3)) * dh
        perimeter = b + 2.0 * h * np.sqrt(1.0 + z**2)
        friction_term = q2 * MANNING_N**2 * perimeter ** (4.0 / 3.0) / (h ** (10.0 / 3.0) * (b + z * h) ** (10.0 / 3.0))
        width_term = q2 * db / (GRAVITY * h**2 * (b + z * h) ** 3)
        return froude_term + friction_term - width_term

    def bed_elevation(self, x, n_quad: int = 40001) -> Array:
        """
        Bed elevation z(x) = int_x^L S0(X) dX [m], so that z(L) = 0.

        Trapezoidal rule on ``n_quad`` points (with one at the jump, where S0 changes branch), interpolated to x.
        """
        xs = np.linspace(0.0, self.length, n_quad)
        if self.jump_at is not None:
            xs = np.unique(np.append(xs, self.jump_at))
        s0 = self.bed_slope(xs)
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (s0[1:] + s0[:-1]) * np.diff(xs))))
        return np.interp(np.asarray(x, dtype=float), xs, cumulative[-1] - cumulative)

    def h_in(self) -> float:
        """Analytical water depth at x = 0 [m] (imposed upstream in the supercritical case)."""
        return float(self.h(0.0))

    def h_out(self) -> float:
        """Analytical water depth at x = L [m] (imposed downstream in the subcritical cases)."""
        return float(self.h(self.length))


CASES = {
    "subcritical": MacDonaldCase(
        "subcritical", "3.5.1", "Subcritical flow in a short domain", "subcritical",
        200.0, 0.0, _b1, _h_subcritical, end_time=900.0, gt_spacing=10.0, regions=THIRDS,
    ),
    "supercritical": MacDonaldCase(
        "supercritical", "3.5.2", "Supercritical flow in a short domain", "supercritical",
        200.0, 0.0, _b1, _h_supercritical, end_time=300.0, gt_spacing=10.0, regions=THIRDS,
        robust_error_scale=True,  # the imposed inflow water level gives a few large errors at the inlet
    ),
    "jump": MacDonaldCase(
        "jump", "3.5.6", "Smooth transition followed by a hydraulic jump in a long domain", "subcritical",
        400.0, 2.0, _b2, _h_jump, end_time=1200.0, gt_spacing=20.0, jump_at=JUMP_X, inflow="froude_in",
        regions=JUMP_REGIONS,
        turbulence="k_epsilon",  # k-epsilon improves the depth after the jump (RMSE 0.0076 vs 0.0091 m)
        robust_error_scale=True,  # the smeared jump gives a few large errors
    ),
}  # fmt: skip
DEFAULT_CASE = "subcritical"

# Module-level shortcuts of the default (subcritical) case
LENGTH = CASES[DEFAULT_CASE].length
width = CASES[DEFAULT_CASE].width
dwidth = CASES[DEFAULT_CASE].dwidth
h_exact = CASES[DEFAULT_CASE].h
dh_exact = CASES[DEFAULT_CASE].dh
bed_slope = CASES[DEFAULT_CASE].bed_slope
bed_elevation = CASES[DEFAULT_CASE].bed_elevation
h_out = CASES[DEFAULT_CASE].h_out


def get_case(case: str | MacDonaldCase) -> MacDonaldCase:
    """The case object for a key of ``CASES`` (or the object itself)."""
    if isinstance(case, MacDonaldCase):
        return case
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; use one of {tuple(CASES)}.")
    return CASES[case]


# ── Case generation ─────────────────────────────────────────────────────────────────────────────────
def mesh_columns(case: MacDonaldCase, ny: int = NY, bank_cells: int = BANK_CELLS) -> list[tuple[float, float, float]]:
    """
    Node columns of a cross section as (factor of B/2, lateral offset [m], elevation above the bed [m]).

    ``ny + 1`` columns over the flat bottom; for a trapezoid (Z > 0) ``bank_cells`` more on each side, each rising
    BANK_HEIGHT / bank_cells over Z times that, up to the bank crest.
    """
    columns = [(2.0 * j / ny - 1.0, 0.0, 0.0) for j in range(ny + 1)]
    if case.bank_slope > 0:
        steps = [BANK_HEIGHT * k / bank_cells for k in range(1, bank_cells + 1)]
        left = [(-1.0, -case.bank_slope * dz, dz) for dz in reversed(steps)]
        right = [(1.0, case.bank_slope * dz, dz) for dz in steps]
        columns = [*left, *columns, *right]
    return columns


def string_columns(columns: list[tuple[float, float, float]], depth: float) -> tuple[int, int]:
    """First and last node column of a boundary string: the edges whose mean elevation lies below ``depth``."""
    wet = [j for j in range(len(columns) - 1) if (columns[j][2] + columns[j + 1][2]) / 2 < depth]
    return wet[0], wet[-1] + 1


def inflow_froude(case: MacDonaldCase, ny: int = NY) -> float:
    """
    Froude number of a ``froude_in`` inflow that gives the analytical inflow depth h(0).

    BASEMENT sets h = ((Q / b)^2 / (g Fr^2))^(1/3) with b the plan length of the boundary string, so
    Fr = Q / (b h(0) sqrt(g h(0))).
    """
    columns = mesh_columns(case, ny)
    first, last = string_columns(columns, case.h_in())
    b0 = float(case.width(0.0))
    plan = [factor * b0 / 2 + offset for factor, offset, _ in columns[first : last + 1]]
    h0 = case.h_in()
    return DISCHARGE / ((plan[-1] - plan[0]) * h0 * np.sqrt(GRAVITY * h0))


def write_2dm(path: Path, case: str | MacDonaldCase = DEFAULT_CASE, dx: float = DX, ny: int = NY) -> Path:
    """
    Write a structured .2dm mesh of the channel.

    A regular grid of cross sections every ``dx`` over [0, L]. On each cross section, ``ny + 1`` nodes are spread
    over the local bottom width B(x), so the channel follows its shape instead of a staircase; for a trapezoid the
    sloping banks are part of the bathymetry (``mesh_columns``) and end in a dry crest. Every rectangle is split
    into two right triangles with the same diagonal (counter-clockwise), written as consecutive elements, all in
    region 1. The node elevation is the analytical bed z(x) (plus the bank height on the banks).

    Rectangular case: the mesh ends at y = +-B(x)/2, where the edges are default BASEMENT wall boundaries (free
    slip, no friction). Trapezoidal case: the outer edge lies on the dry bank crest.

    Nodestrings ``inflow`` (x = 0) and ``outflow`` (x = L) run in +y, so that the upstream side lies on their left.
    They span the bottom and the bank edges whose mean elevation lies below the boundary water surface: an edge
    entirely above it would get a negative depth from a prescribed water surface (BASEMENT stops with a mass
    error). The element lines carry the element elevation (``NUM_MATERIALS_PER_ELEM 2``), as BASEmesh writes them
    and as ``functions_io.read_mesh_file`` expects.
    """
    case = get_case(case)
    nx = round(case.length / dx)
    x = np.linspace(0.0, case.length, nx + 1)
    b = case.width(x)
    z = case.bed_elevation(x)
    columns = mesh_columns(case, ny)
    ncol = len(columns)

    def node_id(i: int, j: int) -> int:
        return i * ncol + j + 1

    lines = ["MESH2D", f'MESHNAME "{CASE_NAME}"', "NUM_MATERIALS_PER_ELEM 2"]
    node_z = {}
    for i in range(nx + 1):
        for j, (factor, offset, dz) in enumerate(columns):
            nid = node_id(i, j)
            node_z[nid] = z[i] + dz
            lines.append(f"ND {nid} {x[i]:.10f} {factor * b[i] / 2 + offset:.10f} {node_z[nid]:.10f}")

    elem = 0
    for i in range(nx):
        region = int(case.region_index((x[i] + x[i + 1]) / 2))  # both triangles of a rectangle share it
        for j in range(ncol - 1):
            a, b_, c_, d = node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)
            for tri in ((a, b_, c_), (a, c_, d)):
                elem += 1
                z_elem = np.mean([node_z[n] for n in tri])
                lines.append(f"E3T {elem} {tri[0]} {tri[1]} {tri[2]} {region} {z_elem:.10f}")

    for name, i, depth in (("inflow", 0, case.h_in()), ("outflow", nx, case.h_out())):
        first, last = string_columns(columns, depth)
        ids = [node_id(i, j) for j in range(first, last + 1)]
        ids[-1] = -ids[-1]  # a negative node number ends the node string
        lines.append("NS " + " ".join(str(n) for n in ids) + f" {name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    return path


def resolve_turbulence(case: str | MacDonaldCase, turbulence: str | None = None) -> str:
    """The turbulence model of a run: ``turbulence`` if given, otherwise the default of the case."""
    model = get_case(case).turbulence if turbulence is None else turbulence
    if model not in TURBULENCE_MODELS:
        raise ValueError(f"Unknown turbulence model {model!r}; use one of {TURBULENCE_MODELS}.")
    return model


def variant_name(case: str | MacDonaldCase = DEFAULT_CASE, turbulence: str | None = None) -> str:
    """
    Name (simulation_name and output folder) of a benchmark variant.

    ``swashes_macdonald`` for the default case with its default turbulence model; a ``_<case>`` suffix for the
    other cases and a ``_k_epsilon`` or ``_no_turbulence`` suffix when the turbulence model differs from the case
    default
    otherwise (e.g. ``swashes_macdonald_jump_k_epsilon``).
    """
    case = get_case(case)
    turbulence = resolve_turbulence(case, turbulence)
    name = CASE_NAME if case.key == DEFAULT_CASE else f"{CASE_NAME}_{case.key}"
    if turbulence == case.turbulence:
        return name
    return f"{name}_no_turbulence" if turbulence == "none" else f"{name}_{turbulence}"


def benchmark_folder(case: str | MacDonaldCase = DEFAULT_CASE, turbulence: str | None = None) -> Path:
    """Output folder of a benchmark variant, under ``output/benchmarks/``."""
    return BENCHMARK_FOLDER.parent / variant_name(case, turbulence)


def turbulence_config(turbulence: str = "none", end_time: float = float("inf")) -> dict:
    """HYDRAULICS/TURBULENCE block of model.json (empty without a turbulence model)."""
    if turbulence == "none":
        return {}
    return {
        "TURBULENCE": {
            "INITIAL": {"type": "zero"},
            # k-epsilon starts once the channel has filled (at most a third into the run, for the short supercritical one)
            "PARAMETER": {"turbulence_start": min(TURBULENCE_START, end_time / 3.0)},
            # Default k-epsilon constants of BASEMENT 4.3 (c_mu 0.09, c_eps1 1.44, c_eps2 1.92, ...)
            "REYNOLDS_STRESS": {"closure_model": turbulence},
        }
    }


def boundary_config(case: MacDonaldCase, discharge_folder: Path) -> list[dict]:
    """
    HYDRAULICS/BOUNDARY/STANDARD of model.json.

    Subcritical inflow: ``uniform_in`` (or ``froude_in``, see below) with the discharge hydrograph (BOAR rewrites
    its path for each run) and ``zhydrograph`` outflow at the analytical water surface z(L) + h_out. The jump case
    uses ``froude_in`` with the Froude number that gives h(0) (``inflow_froude``): after the dry start the water
    runs down its steep upstream reach as a supercritical flow, and ``uniform_in`` keeps that shallow inflow (a
    second steady state with a jump near the inlet) instead of the analytical subcritical one. Supercritical inflow: ``zqhydrograph``
    with the analytical water surface z(0) + h_in and the specific discharge Q / B(0) (hydrograph files with
    absolute paths), free outflow (``zero_gradient_out``).
    """
    if case.regime == "subcritical":
        inflow = {"name": "inflow", "string_name": "inflow", "type": case.inflow}
        inflow["discharge_file"] = f"../discharge_files/{DISCHARGE_CODE}.txt"
        if case.inflow == "froude_in":
            inflow["froude_number"] = round(inflow_froude(case), 6)
        else:
            inflow["slope"] = float(case.bed_slope(0.0))
        return [
            inflow,
            {
                "name": "outflow",
                "string_name": "outflow",
                "type": "zhydrograph",
                "wse": float(case.bed_elevation(case.length)) + case.h_out(),
                "inflowPossible": "true",  # required by the 4.3 schema (a string)
            },
        ]
    return [
        {
            "name": "inflow",
            "string_name": "inflow",
            "type": "zqhydrograph",
            "wse_file": str((discharge_folder / "inflow_wse.txt").resolve()),
            "spec_discharge_file": str((discharge_folder / "inflow_spec_discharge.txt").resolve()),
        },
        {"name": "outflow", "string_name": "outflow", "type": "zero_gradient_out"},
    ]


def model_config(
    case: str | MacDonaldCase = DEFAULT_CASE, turbulence: str | None = None, discharge_folder: Path | None = None
) -> dict:
    """BASEMENT 4.3 model.json of the benchmark (layout of the GUI schema / BASEHPC tutorial)."""
    case = get_case(case)
    turbulence = resolve_turbulence(case, turbulence)
    discharge_folder = Path("discharge_files") if discharge_folder is None else discharge_folder
    return {
        "SETUP": {
            "simulation_name": variant_name(case, turbulence),
            "BASEHPC": {
                "PHYSICAL_PROPERTIES": {"gravity": GRAVITY},
                "BASEPLANE_2D": {
                    "GEOMETRY": {
                        "mesh_file": f"../meshes/{CASE_NAME}.2dm",
                        "INTERPOLATION": {"method": "weighted"},
                        "STRINGDEF": [
                            {"name": "inflow", "upstream_direction": "left"},
                            {"name": "outflow", "upstream_direction": "left"},
                        ],
                        "REGIONDEF": [{"name": name, "index": [k]} for k, name in enumerate(case.region_names, 1)],
                    },
                    "HYDRAULICS": {
                        "PARAMETER": {
                            # CFL 0.5 keeps the wet/dry front of the dry start mass conservative
                            "CFL": 0.5,
                            "minimum_water_depth": 0.001,
                            "fluid_density": 1000.0,
                            "max_time_step": 1.0,
                            "safe_mode": "on",
                        },
                        # One Manning coefficient per region along the channel, all calibrated by BOAR.
                        "FRICTION": {
                            "type": "manning",
                            "default_friction": MANNING_N,
                            "regions": [{"region_name": name, "friction": MANNING_N} for name in case.region_names],
                        },
                        # Default: no TURBULENCE block, as the shallow-water solutions have no turbulent viscosity
                        # either; the k_epsilon variant adds BASEMENT's k-epsilon model.
                        **turbulence_config(turbulence, case.end_time),
                        "BOUNDARY": {"STANDARD": boundary_config(case, discharge_folder)},
                        # SWASHES starts dry (with a puddle at the outlet in the subcritical cases); the steady state
                        # does not depend on it, and a dry start with a rising inflow avoids a mass imbalance at t = 0.
                        "INITIAL": {"type": "dry"},
                    },
                },
            },
        }
    }


def simulation_config(case: str | MacDonaldCase = DEFAULT_CASE, turbulence: str | None = None) -> dict:
    """BASEMENT simulation.json: one output at the end of the run."""
    case = get_case(case)
    turbulence = resolve_turbulence(case, turbulence)
    turbulence_outputs = [] if turbulence == "none" else ["turb_k", "turb_reynolds"]
    return {
        "SIMULATION": {
            "TIME": {"start": 0.0, "end": case.end_time, "out": case.end_time},
            # flow_velocity (and turb_k, turb_reynolds) go to results_aux.h5, which BOAR reads.
            "OUTPUT": [
                "water_surface",
                "water_depth",
                "flow_velocity",
                "spec_discharge",
                "bottom_elevation",
                "ns_hyd_discharge",
                *turbulence_outputs,
            ],
        }
    }


def ground_truth(case: str | MacDonaldCase = DEFAULT_CASE) -> pd.DataFrame:
    """
    Analytical comparison points on the channel centre line over [0, L], every ``gt_spacing``.

    The point on a hydraulic jump is left out: the analytical depth is discontinuous there.
    """
    case = get_case(case)
    x = np.arange(0.0, case.length + case.gt_spacing / 2, case.gt_spacing)
    if case.jump_at is not None:
        x = x[~np.isclose(x, case.jump_at)]
    return pd.DataFrame(
        {
            "X [m]": x,
            "Y [m]": np.zeros_like(x),
            "H [m]": case.h(x),
            "Ux [m/s]": case.mean_velocity(x),
            "Uy [m/s]": np.zeros_like(x),
        }
    )


def _ramp(value: float, end_time: float) -> str:
    """Time series (time [s], value): 0 at t = 0, ``value`` from RAMP_TIME to the end."""
    rows = [(0.0, 0.0), (RAMP_TIME, value), (max(end_time, RAMP_TIME) + 1.0, value)]
    return "".join(f"{t}, {v}\n" for t, v in rows)


def discharge_hydrograph(end_time: float) -> str:
    """Inflow discharge hydrograph [m3/s]: 0 at t = 0, Q from RAMP_TIME to the end."""
    return _ramp(DISCHARGE, end_time)


def write_case(
    root: Path,
    case: str | MacDonaldCase = DEFAULT_CASE,
    turbulence: str | None = None,
    dx: float = DX,
    ny: int = NY,
) -> dict[str, Path]:
    """
    Generate the benchmark inputs in BOAR's folder layout.

    Args:
        root: Folder receiving ``simulation_configs/``, ``meshes/``, ``ground_truth/`` and ``discharge_files/``
            (the inflow time series and the discharge dictionary).
        case: Case key of ``CASES`` (or the case object).
        turbulence: Turbulence model of the variant, ``"none"`` or ``"k_epsilon"``.
        dx: Mesh cell length along the channel [m].
        ny: Number of mesh cells across the flat bottom.

    Returns:
        dict: The input folders, keyed like the constants they replace.
    """
    case = get_case(case)
    folders = {
        "SIMULATION_CONFIG_FOLDER": root / "simulation_configs",
        "MESH_FOLDER": root / "meshes",
        "GROUND_TRUTH_FOLDER": root / "ground_truth",
        "DISCHARGE_FILES_FOLDER": root / "discharge_files",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # BOAR caches the cell centroids per mesh name; drop the cache so a regenerated mesh is not mixed with it.
    shutil.rmtree(folders["MESH_FOLDER"] / "mesh_centroids", ignore_errors=True)
    write_2dm(folders["MESH_FOLDER"] / f"{CASE_NAME}.2dm", case, dx=dx, ny=ny)

    discharge_folder = folders["DISCHARGE_FILES_FOLDER"]
    (discharge_folder / f"{DISCHARGE_CODE}.txt").write_text(discharge_hydrograph(case.end_time))
    if case.regime == "supercritical":
        wse_in = float(case.bed_elevation(0.0)) + case.h_in()
        rows = [(0.0, wse_in), (case.end_time + 1.0, wse_in)]
        (discharge_folder / "inflow_wse.txt").write_text("".join(f"{t}, {v}\n" for t, v in rows))
        spec_discharge = DISCHARGE / float(case.width(0.0))
        (discharge_folder / "inflow_spec_discharge.txt").write_text(_ramp(spec_discharge, case.end_time))

    configs = {
        "model.json": model_config(case, turbulence, discharge_folder),
        "simulation.json": simulation_config(case, turbulence),
        "results.json": {"RESULTS": {"EXPORT": [{"format": "xdmf"}]}},
    }
    for name, content in configs.items():
        (folders["SIMULATION_CONFIG_FOLDER"] / name).write_text(json.dumps(content, indent=4))

    ground_truth(case).to_csv(folders["GROUND_TRUTH_FOLDER"] / f"MACDONALD_{DISCHARGE_CODE}_1.csv", index=False)
    pd.DataFrame({"Code": [DISCHARGE_CODE], "Discharge [m3/s]": [DISCHARGE]}).to_csv(
        folders["DISCHARGE_FILES_FOLDER"] / "discharge_dictionary.csv", index=False
    )
    return folders


# ── Comparison ──────────────────────────────────────────────────────────────────────────────────────
def triangle_areas(mesh_path: Path) -> np.ndarray:
    """Plan area of every mesh element [m2], in element order."""
    mesh = functions_io.read_mesh_file(str(mesh_path))
    nodes, cells = mesh["nodes"], mesh["cells"]
    index = {nid: i for i, nid in enumerate(nodes["id"])}
    corners = [np.array([index[n] for n in cells[k]]) for k in ("nd1", "nd2", "nd3")]
    (x1, y1), (x2, y2), (x3, y3) = ((nodes["x"][k], nodes["y"][k]) for k in corners)
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def block_means(sim_data: dict, areas: np.ndarray) -> pd.DataFrame:
    """
    Area-weighted means over the two triangles of every mesh rectangle.

    ``write_2dm`` writes the two triangles of a rectangle as consecutive elements, so element pairs
    (1, 2), (3, 4), ... form the blocks. The block centroid is the rectangle centroid.
    """
    pairs = areas.reshape(-1, 2)
    weights = pairs / pairs.sum(axis=1, keepdims=True)

    def mean(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float).reshape(-1, 2) * weights).sum(axis=1)

    centroids = sim_data["centroids"]
    return pd.DataFrame(
        {
            "cx": mean(centroids["cx"].to_numpy()),
            "cy": mean(centroids["cy"].to_numpy()),
            "h": mean(sim_data["hyd"]["h"]),
            "wse": mean(sim_data["hyd"]["wse"]),
            "speed": mean(np.hypot(sim_data["hyd"]["ux"], sim_data["hyd"]["uy"])),
        }
    )


def sample_depth(blocks: pd.DataFrame, points: pd.DataFrame) -> np.ndarray:
    """Simulated water depth at the comparison points: IDW of the 6 nearest blocks (as BOAR's loss on cells)."""
    values = []
    for x, y in points[["X [m]", "Y [m]"]].to_numpy():
        ids, dist = utils.find_closest_cell(blocks, (x, y), npoints=6, distance=True)
        values.append(np.average(blocks["h"].to_numpy()[ids], weights=1.0 / np.maximum(dist, 1e-9) ** 2))
    return np.asarray(values)


def depth_errors(sim_data: dict, areas: np.ndarray, points: pd.DataFrame) -> dict[str, float]:
    """RMSE and maximum absolute error of the block-averaged water depth against the analytical depth [m]."""
    diff = sample_depth(block_means(sim_data, areas), points) - points["H [m]"].to_numpy()
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "max_abs": float(np.max(np.abs(diff)))}


class DepthLoss:
    """
    Benchmark loss for BOAR: mean RMSE of the block-averaged water depth over the ground-truth files.

    A module-level class (not a closure), so that BOAR's parallel batch workers can receive it.
    """

    def __init__(self, areas: np.ndarray):
        self.areas = np.asarray(areas)

    def __call__(self, simulation_data: dict, ground_truth_data: list) -> tuple[float, None]:
        return float(np.mean([depth_errors(simulation_data, self.areas, gt)["rmse"] for gt in ground_truth_data])), None


def make_depth_loss(areas: np.ndarray) -> DepthLoss:
    """Benchmark loss for BOAR (see :class:`DepthLoss`)."""
    return DepthLoss(areas)


def _analytical_line(case: MacDonaldCase, n: int = 801) -> np.ndarray:
    """x positions for the analytical curves, with a NaN gap at the jump (so no line is drawn across it)."""
    x = np.linspace(0.0, case.length, n)
    if case.jump_at is not None:
        cut = np.searchsorted(x, case.jump_at, side="right")  # x = jump_at belongs to the upstream branch
        x = np.insert(x, cut, np.nan)
    return x


# Text sizes [pt] for the docs: the profile is shown at about its own size, the 11-inch field map at about
# 0.75 times, so these stay readable (at least about 11 px on screen).
PROFILE_STYLE = {"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "legend.fontsize": 11,
                 "xtick.labelsize": 12, "ytick.labelsize": 12}  # fmt: skip
FIELDS_STYLE = {"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "figure.titlesize": 16,
                "xtick.labelsize": 12, "ytick.labelsize": 12}  # fmt: skip


def _styled(style: dict):
    """Draw a figure function under a matplotlib style (text sizes), leaving the global settings unchanged."""

    def decorate(function):
        def wrapper(*args, **kwargs):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            with plt.rc_context(style):
                return function(*args, **kwargs)

        wrapper.__doc__, wrapper.__name__ = function.__doc__, function.__name__
        return wrapper

    return decorate


@_styled(PROFILE_STYLE)
def plot_profile(sim_data: dict, areas: np.ndarray, case: MacDonaldCase, path: Path, title: str) -> None:
    """Save the analytical and simulated water-surface and bed profiles along the channel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _analytical_line(case)
    x_bed = np.linspace(0.0, case.length, 801)
    blocks = block_means(sim_data, areas)
    wet = (blocks["h"] > 0.01) & (np.abs(blocks["cy"]) <= case.width(blocks["cx"]) / 2)  # flat bottom only
    points = ground_truth(case)

    fig, (ax_wse, ax_h) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    ax_wse.plot(x_bed, case.bed_elevation(x_bed), color="0.4", lw=1.2, label="Bed elevation")
    ax_wse.plot(x, case.bed_elevation(x) + case.h(x), color="k", lw=1.5, label="Analytical (SWASHES)")
    ax_wse.plot(
        blocks["cx"][wet], blocks["wse"][wet], ".", ms=2, color="tab:blue", alpha=0.4, label="Water surface elevation"
    )
    ax_wse.set_ylabel("Elevation [m]")
    ax_wse.legend(loc="upper right", frameon=False)
    ax_wse.set_title(title)

    ax_h.plot(x, case.h(x), color="k", lw=1.5, label="Analytical (SWASHES)")
    ax_h.plot(
        points["X [m]"], sample_depth(blocks, points), "o", ms=4, color="tab:blue", label="BASEMENT (centre line)"
    )
    ax_h.set_xlabel("x [m]")
    ax_h.set_ylabel("Water depth [m]")
    ax_h.legend(loc="best", frameon=False)
    ax_h.set_xlim(0.0, case.length)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def error_scale(errors: np.ndarray, percentile: float = 90.0) -> float:
    """
    Symmetric colour limit for a relative error map [%].

    The ``percentile`` of |error|, rounded up to 1, 2, 2.5 or 5 times a power of ten, so a few large values (at a
    hydraulic jump or the waterline) do not stretch the scale and wash out the errors elsewhere; values beyond the
    limit saturate at the end colours.
    """
    errors = np.abs(np.asarray(errors, dtype=float))
    if errors.size == 0 or not np.any(errors > 0):
        return 1.0
    value = float(np.percentile(errors, percentile))
    value = value if value > 0 else float(errors.max())
    magnitude = 10.0 ** np.floor(np.log10(value))
    step = next(s for s in (1.0, 2.0, 2.5, 5.0, 10.0) if s * magnitude >= value * (1 - 1e-12))
    return float(step * magnitude)


def _annotate_peak_error(ax, case: MacDonaldCase, per_cell: dict, dh: np.ndarray, judged: np.ndarray) -> None:
    """
    Short note "XX % at jump" on the error map of the jump case: the largest error within 10 m of the jump, which
    lies beyond the colour scale.
    """
    x = per_cell["cx"]
    candidates = judged & (np.abs(x - case.jump_at) <= 10.0)
    if not np.any(candidates):
        return
    k = np.flatnonzero(candidates)[np.argmax(np.abs(dh[candidates]))]
    text = f"{dh[k]:+.0f} % at jump"
    from matplotlib import patheffects

    # Text inside the panel near its top (x in data, y in axes fraction), arrow down to the block; a white outline
    # keeps the text readable where it crosses the water.
    ax.annotate(
        text,
        xy=(x[k], per_cell["cy"][k]),
        xytext=(x[k] + 0.05 * case.length, 0.96),
        textcoords=("data", "axes fraction"),
        ha="left",
        va="top",
        fontsize=11,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 14,
            "color": "0.15",
            "lw": 1.8,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )


@_styled(FIELDS_STYLE)
def plot_fields(
    sim_data: dict, mesh_path: Path, areas: np.ndarray, case: MacDonaldCase, path: Path, title: str
) -> None:
    """
    Save one figure with six 2D maps of the domain.

    Panels: computational mesh, bed elevation, friction zones, water depth, velocity magnitude and the relative
    water-depth error 100 (h - h_exact) / h_exact [%]. Every map except the mesh shows block values (the area-weighted
    mean of the two triangles of a mesh rectangle). h_exact = z(x) + h(x) - z_block is the local analytical depth: the
    analytical water surface minus the block bed. The error is shown where h_exact exceeds 15 cm, on a symmetric scale
    over its full range; in the cases with ``robust_error_scale`` (supercritical, jump) the scale ends at the 90th
    percentile of |error| (``error_scale``), so the few large errors at the inlet or the jump saturate instead of
    washing out the rest; in the jump case a note gives the error at the jump. Result maps leave
    dry blocks (h < 1 cm) blank. No velocity error is shown: the pseudo-2D solution gives only the cross-section mean
    velocity, not a local one. The axes are stretched across the channel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.tri import Triangulation

    mesh = functions_io.read_mesh_file(str(mesh_path))
    nodes, cells = mesh["nodes"], mesh["cells"]
    index = {nid: i for i, nid in enumerate(nodes["id"])}
    triangles = np.array([[index[n] for n in cells[k]] for k in ("nd1", "nd2", "nd3")]).T

    blocks = block_means(sim_data, areas)
    pair_weights = areas.reshape(-1, 2) / areas.reshape(-1, 2).sum(axis=1, keepdims=True)
    blocks["z"] = (cells["z"].reshape(-1, 2) * pair_weights).sum(axis=1)
    blocks["region"] = cells["region"].reshape(-1, 2)[:, 0]
    per_cell = {key: np.repeat(blocks[key].to_numpy(), 2) for key in blocks.columns}

    wet = per_cell["h"] >= 0.01
    h_ref = case.bed_elevation(per_cell["cx"]) + case.h(per_cell["cx"]) - per_cell["z"]
    judged = wet & (h_ref > 0.15)  # relative errors are meaningless at the waterline on the banks
    dh = 100.0 * (per_cell["h"] - h_ref) / np.where(judged, h_ref, 1.0)
    # Robust scale (90th percentile) where a few large errors would wash out the rest (inlet of the supercritical
    # case, the jump); the full range otherwise. Only the jump is annotated.
    dh_lim = error_scale(dh[judged], percentile=90.0 if case.robust_error_scale else 100.0)
    dh_beyond = bool(np.any(np.abs(dh[judged]) > dh_lim))

    regions = np.unique(cells["region"])
    region_cmap = ListedColormap(plt.get_cmap("tab10").colors[: len(regions)])
    region_norm = BoundaryNorm(np.append(regions - 0.5, regions[-1] + 0.5), len(regions))
    everywhere = np.ones(len(cells), dtype=bool)
    # Variable names go next to the colour bars; the error definition is in the docstring above.
    panels = [
        ("Bed elevation\n[m]", per_cell["z"], everywhere, {"cmap": "terrain"}),
        ("Friction zone", per_cell["region"], everywhere, {"cmap": region_cmap, "norm": region_norm}),
        ("Water depth\n[m]", per_cell["h"], wet, {"cmap": "Blues"}),
        ("Velocity\n[m/s]", per_cell["speed"], wet, {"cmap": "viridis"}),
        (
            "Depth error\nvs h_exact [%]",
            dh,
            judged,
            {"cmap": "RdBu_r", "norm": TwoSlopeNorm(0.0, -dh_lim, dh_lim)},
        ),
    ]

    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    axes[0].triplot(Triangulation(nodes["x"], nodes["y"], triangles), color="0.3", lw=0.25, rasterized=True)
    axes[0].set_title(f"Computational mesh ({len(cells)} triangles, {len(blocks)} blocks)", loc="left")
    fig.colorbar(plt.cm.ScalarMappable(cmap="Greys"), ax=axes[0], pad=0.01, aspect=10).ax.set_visible(False)

    for ax, (label, values, shown, style) in zip(axes[1:], panels, strict=True):
        tri = Triangulation(nodes["x"], nodes["y"], triangles, mask=~np.asarray(shown, dtype=bool))
        values = np.nan_to_num(np.asarray(values, dtype=float))
        # Rasterise the cells (thousands of triangles); axes, labels and colour bars stay vector in the SVG.
        art = ax.tripcolor(tri, facecolors=values, rasterized=True, **style)
        # With a robust scale, arrows on the colour bar mark the blocks beyond it.
        extend = "both" if label.startswith("Depth error") and dh_beyond else "neither"
        bar = fig.colorbar(art, ax=ax, pad=0.01, aspect=10, extend=extend)
        bar.set_label(label)
        if label == "Friction zone":
            bar.set_ticks(regions, labels=case.region_names)
        if label.startswith("Depth error") and dh_beyond and case.jump_at is not None:
            _annotate_peak_error(ax, case, per_cell, dh, judged)

    for ax in axes:
        ax.set_ylabel("y [m]")
        if case.jump_at is not None:
            ax.axvline(case.jump_at, color="0.5", lw=0.8, ls="--")
        ax.set_xlim(0.0, case.length)
    axes[-1].set_xlabel("x [m]")
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── Running ─────────────────────────────────────────────────────────────────────────────────────────
def basement_executable(version: str = bmt.DEFAULT_BASEMENT_VERSION) -> Path:
    """Path of BMv4_simulation for the configured BASEMENT folder and version."""
    folder = c.BASEMENT_PATH or bmt.default_basement_path(bmt.SimulationManager._parse_basement_version(version))
    return Path(folder) / f"BMv4_simulation{bmt.EXE_SUFFIX}"


def benchmark_options(
    basement_version: str = bmt.DEFAULT_BASEMENT_VERSION,
    case: str | MacDonaldCase = DEFAULT_CASE,
) -> dict:
    """BOAR user options of the benchmark (replaces user_defined_configs/user_options.yaml)."""
    regions = get_case(case).region_names
    return {
        "general_options": {"overwrite": True, "silent": True, "log_dev": False},
        "basement_options": {"backend": "omp", "nthreads": -1, "basement_version": basement_version},
        "optimization_variable_options": {
            "initial_vector": "file",
            "regions": regions,
            "constraints": None,
            "bounds": [CALIBRATION_BOUNDS],
            "precision": CALIBRATION_PRECISION,
            "save_errors": False,
            "save_tried_vectors": False,
            "opt_engine": "boar",
        },
        "sampling_options": {"seed": 42},
        "surrogate_model_options": {
            "tolerance": 1e-3,
            "test_population": None,
            "max_test_population": 100000,
            "opt_mem_override": False,
            "n_initial": 5 * len(regions),
            "max_tested_vectors": 15 * len(regions),
            "max_no_improvement": 5,
            "GPR_iterations": 50,
            "GPR_alpha": 1e-6,
            "EI_exploration-exploitation": 0.01,
        },
    }


@contextlib.contextmanager
def _benchmark_environment(folders: dict[str, Path], output_folder: Path, loss) -> Iterator[None]:
    """Point BOAR's folder constants and loss function at the benchmark, restoring them afterwards."""
    overrides = {**folders, "OUTPUT_FOLDER": output_folder}
    saved = {name: getattr(c, name) for name in overrides}
    saved_loss = friction_calibration.lf
    try:
        for name, value in overrides.items():
            setattr(c, name, value)
        friction_calibration.lf = types.SimpleNamespace(loss_function=loss)
        yield
    finally:
        for name, value in saved.items():
            setattr(c, name, value)
        friction_calibration.lf = saved_loss


def _set_friction(manager: bmt.SimulationManager, mesh: dict, values) -> None:
    """Write the Manning coefficients of the regions (one value for all, or one per region) into model.json."""
    values = np.atleast_1d(np.asarray(values, dtype=float))
    regions = int(
        manager._load_model_config()["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["GEOMETRY"]["REGIONDEF"][-1]["index"][0]
    )
    values = np.resize(values, regions)
    friction = np.array(list(zip(range(1, regions + 1), values)), dtype=[("region", "i2"), ("friction", "f8")])
    manager.define_friction(mesh, friction)


def _run_basement(manager: bmt.SimulationManager, data: dict) -> dict:
    """Run BASEMENT (setup, simulation, results) and return the last output."""
    manager.run_setup()
    if manager.run_simulation() == 1:
        raise RuntimeError("BASEMENT reported a mass conservation error.")
    manager.run_results_processing()
    return manager.extract_sim_data(data["centroids"], sim_time=-1)


def _best_from_memory(case_folder: Path) -> tuple[list[float], float]:
    """Best (friction vector, loss) pair stored in optimization_memory.csv."""
    vector, loss, _ = friction_calibration.best_from_memory(case_folder)
    return vector, loss


def _report(label: str, errors: dict[str, float], judged: bool = True) -> bool:
    """Print the depth errors; ``judged`` runs are compared with the tolerance."""
    passed = errors["rmse"] <= DEPTH_RMSE_TOL
    verdict = f"(tolerance {DEPTH_RMSE_TOL} m) -> {'PASS' if passed else 'FAIL'}" if judged else "(information only)"
    print(f"  {label}: depth RMSE = {errors['rmse']:.4f} m, max |dh| = {errors['max_abs']:.4f} m {verdict}")
    return passed


def run(
    case: str | MacDonaldCase = DEFAULT_CASE,
    calibrate: bool = True,
    basement_version: str = bmt.DEFAULT_BASEMENT_VERSION,
    turbulence: str | None = None,
) -> dict:
    """
    Generate a case, run BASEMENT with the analytical friction and, optionally, a BOAR calibration.

    Args:
        case: Case key of ``CASES`` ("subcritical", "supercritical" or "jump") or the case object.
        calibrate: Also run the BOAR calibrated run (calibration, then BASEMENT with the calibrated friction).
        basement_version: BASEMENT version to run ("4.2" or "4.3").
        turbulence: Turbulence model, ``"none"`` or ``"k_epsilon"``; ``None`` (default) uses the case default
            (k-epsilon for the jump case, none otherwise). Each variant writes to its own
            folder, ``output/benchmarks/<variant name>/``.

    Returns:
        dict: ``analytical_manning`` and (if calibrated) ``boar_calibrated`` depth errors, ``effective_n`` (per region),
        ``passed`` and the ``case_folder``.
    """
    case = get_case(case)
    turbulence = resolve_turbulence(case, turbulence)
    folder = benchmark_folder(case, turbulence)
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_macdonald")
    for old in list(logger.handlers):  # one log file per variant
        logger.removeHandler(old)
        old.close()
    handler = logging.FileHandler(folder / "benchmark.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    folders = write_case(folder / "inputs", case, turbulence)
    mesh_path = folders["MESH_FOLDER"] / f"{CASE_NAME}.2dm"
    areas = triangle_areas(mesh_path)
    points = ground_truth(case)
    user_opt = benchmark_options(basement_version, case=case)
    result: dict = {}

    with _benchmark_environment(folders, folder / "runs", make_depth_loss(areas)):
        paths = functions_io.setup_directories(ROOT, user_opt, logger)
        case_folder = Path(paths["folder_simulation"])
        data = cell_centroid.main(functions_io.main(paths), paths, user_opt)
        manager = bmt.SimulationManager(str(case_folder), user_opt["basement_options"], logger=logger, silent=True)

        options = user_opt["basement_options"]
        threads = os.cpu_count() if options["nthreads"] == -1 else options["nthreads"]
        print(
            f"SWASHES MacDonald benchmark, section {case.section} ({case.title}), turbulence model: {turbulence} "
            f"(BASEMENT {basement_version}, {options['backend']} backend, {threads} threads)"
        )
        print(f"  Case folder: {case_folder}")
        start = time.perf_counter()
        sim = _run_basement(manager, data)
        result["analytical_manning"] = depth_errors(sim, areas, points)
        label = f"analytical_manning run, n = {MANNING_N}"
        plot_profile(sim, areas, case, folder / "profile_analytical_manning.svg", label)
        plot_fields(sim, mesh_path, areas, case, folder / "fields_analytical_manning.svg", label)
        # The analytical_manning run checks the BASEMENT set-up; its depth is not judged.
        _report(
            f"analytical_manning run (n = {MANNING_N}, {time.perf_counter() - start:.0f} s)",
            result["analytical_manning"],
            judged=False,
        )
        passed = True

        if calibrate:
            start = time.perf_counter()
            _set_friction(manager, data["mesh"], CALIBRATION_START_N)
            friction_calibration.main(data, paths, user_opt)
            best_n, _ = _best_from_memory(case_folder)

            _set_friction(manager, data["mesh"], best_n)
            sim = _run_basement(manager, data)
            result["boar_calibrated"] = depth_errors(sim, areas, points)
            result["effective_n"] = dict(zip(case.region_names, best_n, strict=True))
            label = "BOAR calibrated run, n = " + ", ".join(f"{v:g}" for v in best_n)
            plot_profile(sim, areas, case, folder / "profile_boar_calibrated.svg", label)
            plot_fields(sim, mesh_path, areas, case, folder / "fields_boar_calibrated.svg", label)
            passed = _report(f"BOAR calibrated run ({time.perf_counter() - start:.0f} s)", result["boar_calibrated"])
            effective = ", ".join(f"{name} {v:g}" for name, v in result["effective_n"].items())
            print(f"  Effective Manning n: {effective} (include numerical effects; not expected to equal {MANNING_N})")

    result["passed"] = passed
    result["case_folder"] = case_folder
    print(f"  Profiles and log: {folder}")
    return result


# ── Batch mode ───────────────────────────────────────────────────────────────────────────────────────
BATCH_FOLDER = ROOT / "output" / "benchmarks" / "swashes_macdonald_batch"
# Two subcritical cases (the quickest benchmark): one replicates the flat template simulation_configs/model.json,
# the other has its own simulation_configs/<case>/ folder. Both run the analytical discharge Q001 = 20 m3/s.
BATCH_CASES = {"TEMPLATE_Q001": "template", "OWN_Q001": "own config"}
# Quick calibration settings of the batch benchmark (it checks the queue, not the optimizer)
BATCH_SURROGATE = {"n_initial": 4, "max_tested_vectors": 10, "max_no_improvement": 3, "GPR_iterations": 20}


def _start_friction(model: dict, value: float = CALIBRATION_START_N) -> dict:
    """Set every region of a benchmark model.json to the calibration start value (away from n = 0.03)."""
    friction = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["FRICTION"]
    for region in friction["regions"]:
        region["friction"] = value
    return model


def write_batch_case(root: Path) -> dict[str, Path]:
    """
    Generate the subcritical benchmark in BOAR's batch layout.

    ``simulation_configs/model.json`` is the shared template (used by TEMPLATE_Q001, which has no config folder);
    ``simulation_configs/OWN_Q001/`` holds a copy with absolute paths and its own simulation name. Every case
    has its ground truth in ``ground_truth/<case>/``.

    Returns:
        dict: The project folders (see ``functions_io.default_folders``) plus ``DISCHARGE``.
    """
    case = get_case("subcritical")
    folders = write_case(root, case, "none")
    configs = folders["SIMULATION_CONFIG_FOLDER"]
    discharge_folder = folders["DISCHARGE_FILES_FOLDER"]

    template = _start_friction(json.loads((configs / "model.json").read_text()))
    (configs / "model.json").write_text(json.dumps(template, indent=4))

    own = configs / "OWN_Q001"
    own.mkdir(exist_ok=True)
    model = json.loads(json.dumps(template))
    model["SETUP"]["simulation_name"] = "swashes_macdonald_own_config"
    plane = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]
    plane["GEOMETRY"]["mesh_file"] = str((folders["MESH_FOLDER"] / f"{CASE_NAME}.2dm").resolve())
    functions_io.set_inflow_discharge(model, discharge_file=str((discharge_folder / f"{DISCHARGE_CODE}.txt").resolve()))
    (own / "model.json").write_text(json.dumps(model, indent=4))
    for name in ("simulation.json", "results.json"):
        shutil.copyfile(configs / name, own / name)

    ground_truth_root = folders["GROUND_TRUTH_FOLDER"]
    flat = ground_truth_root / f"MACDONALD_{DISCHARGE_CODE}_1.csv"
    for name in BATCH_CASES:
        target = ground_truth_root / name
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        shutil.copyfile(flat, target / f"{name}_1.csv")
    flat.unlink()
    return folders


def run_batch(basement_version: str = bmt.DEFAULT_BASEMENT_VERSION, gpu: bool = False) -> dict:
    """
    Benchmark of BOAR's batch mode on the subcritical MacDonald channel.

    Runs the same two cases (template replication and own config) with several worker layouts and checks for
    each: both summary rows ``ok``, best depth RMSE within ``DEPTH_RMSE_TOL``, the replicated template points to
    the case's discharge file, and the same best vectors as the serial run (same seed).

    Args:
        basement_version: BASEMENT version to run.
        gpu: Also run 1 GPU + 1 CPU worker (needs CUDA).

    Returns:
        dict: ``runs`` (label -> summary table and wall time) and ``passed``.
    """
    BATCH_FOLDER.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=BATCH_FOLDER / "benchmark.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    folders = write_batch_case(BATCH_FOLDER / "inputs")
    loss = DepthLoss(triangle_areas(folders["MESH_FOLDER"] / f"{CASE_NAME}.2dm"))

    layouts = {"serial (1 CPU)": (1, 0), "parallel (2 CPU)": (2, 0)}
    if gpu:
        layouts["parallel (1 GPU + 1 CPU)"] = (1, 1)

    runs: dict = {}
    passed = True
    for label, (n_cpu, n_gpu) in layouts.items():
        print(f"\n== SWASHES MacDonald batch benchmark: {label} ==", flush=True)
        output = BATCH_FOLDER / "runs" / label.split(" ")[0] / f"{n_cpu}cpu_{n_gpu}gpu"
        shutil.rmtree(output, ignore_errors=True)
        run_folders = {**folders, "OUTPUT_FOLDER": output}

        user_opt = benchmark_options(basement_version, case="subcritical")
        user_opt["surrogate_model_options"].update(BATCH_SURROGATE)
        user_opt["batch_options"] = {"n_cpu_workers": n_cpu, "n_gpu_workers": n_gpu, "cpu_cores": -1}

        cases = functions_io.discover_cases(
            run_folders["SIMULATION_CONFIG_FOLDER"], run_folders["GROUND_TRUTH_FOLDER"], logging.getLogger("benchmark")
        )
        start = time.perf_counter()
        batch.run(cases, user_opt, run_folders, loss_function=loss)
        wall = time.perf_counter() - start

        summary = pd.read_csv(output / batch.SUMMARY_FILE_NAME)
        runs[label] = {"summary": summary, "wall": wall}
        print(summary.to_string(index=False))
        print(f"  Wall time: {wall:.0f} s")

        checks = {
            "all cases ok": len(summary) == len(BATCH_CASES) and (summary["status"] == "ok").all(),
            f"best depth RMSE <= {DEPTH_RMSE_TOL} m": bool((summary["best_error"] <= DEPTH_RMSE_TOL).all()),
        }
        replicated = json.loads(next((output / "TEMPLATE_Q001").glob("*/model.json")).read_text())
        inflow = [
            bc
            for bc in replicated["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
            if bc["type"] in functions_io.INFLOW_BOUNDARY_TYPES
        ]
        checks["template replicated with Q001.txt"] = bool(inflow) and all(
            Path(bc.get("discharge_file", "")).name == f"{DISCHARGE_CODE}.txt" for bc in inflow
        )
        # The run folder ends with the best vector (simulated once more after the optimization)
        friction_columns = [col for col in summary.columns if col.startswith("friction_")]
        holds_best = []
        for _, row in summary.iterrows():
            model = json.loads(
                next((output / f"{row['scenario']}_{row['discharge']}").glob("*/model.json")).read_text()
            )
            regions = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["FRICTION"]["regions"]
            in_folder = {f"friction_{r['region_name']}": float(r["friction"]) for r in regions}
            holds_best.append(all(np.isclose(in_folder[col], row[col]) for col in friction_columns))
        checks["run folders hold the best vector"] = all(holds_best)
        if label != "serial (1 CPU)" and "serial (1 CPU)" in runs and n_gpu == 0:
            reference = runs["serial (1 CPU)"]["summary"]
            columns = [col for col in summary.columns if col.startswith("friction_")] + ["best_error", "n_iter_best"]
            checks["same results as serial"] = bool(
                np.allclose(summary[columns].to_numpy(float), reference[columns].to_numpy(float), rtol=1e-6)
            )
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        passed &= all(checks.values())

    if "parallel (2 CPU)" in runs:
        speedup = runs["serial (1 CPU)"]["wall"] / runs["parallel (2 CPU)"]["wall"]
        print(f"\nParallel speed-up with 2 CPU workers: {speedup:.2f}x")
    print(f"Batch benchmark {'PASSED' if passed else 'FAILED'}; outputs and log in {BATCH_FOLDER}")
    return {"runs": runs, "passed": passed}


# ── Interrupt and restart ────────────────────────────────────────────────────────────────────────────
RESTART_FOLDER = BATCH_FOLDER / "restart"


def _batch_options(basement_version: str, n_cpu: int, overwrite: bool) -> dict:
    """Quick batch options of the subcritical benchmark."""
    user_opt = benchmark_options(basement_version, case="subcritical")
    user_opt["general_options"]["overwrite"] = overwrite
    user_opt["surrogate_model_options"].update(BATCH_SURROGATE)
    user_opt["batch_options"] = {"n_cpu_workers": n_cpu, "n_gpu_workers": 0, "cpu_cores": -1}
    return user_opt


def _run_batch_once(output: Path, basement_version: str, n_cpu: int, overwrite: bool) -> None:
    """One batch run of the benchmark cases into ``output`` (the inputs must exist)."""
    folders = {k: v for k, v in write_batch_case_folders(BATCH_FOLDER / "inputs").items()}
    folders["OUTPUT_FOLDER"] = output
    loss = DepthLoss(triangle_areas(folders["MESH_FOLDER"] / f"{CASE_NAME}.2dm"))
    cases = functions_io.discover_cases(
        folders["SIMULATION_CONFIG_FOLDER"], folders["GROUND_TRUTH_FOLDER"], logging.getLogger("benchmark")
    )
    batch.run(cases, _batch_options(basement_version, n_cpu, overwrite), folders, loss_function=loss)


def write_batch_case_folders(root: Path) -> dict[str, Path]:
    """The project folders of an existing batch benchmark layout (see :func:`write_batch_case`)."""
    return {
        "SIMULATION_CONFIG_FOLDER": root / "simulation_configs",
        "MESH_FOLDER": root / "meshes",
        "GROUND_TRUTH_FOLDER": root / "ground_truth",
        "DISCHARGE_FILES_FOLDER": root / "discharge_files",
    }


def _memory(output: Path) -> dict[str, pd.DataFrame]:
    """optimization_memory.csv of every case under ``output`` (raises if one cannot be read)."""
    memories = {}
    for path in sorted(output.glob("*/*/optimization_memory.csv")):
        memories[path.parent.parent.name] = pd.read_csv(path)
    return memories


def _basement_processes() -> list:
    """Running BASEMENT programs (BMv4_*)."""
    import psutil

    found = []
    for process in psutil.process_iter(["name"]):
        if (process.info.get("name") or "").startswith("BMv4_"):
            found.append(process)
    return found


def _batch_child(output: str, basement_version: str, n_cpu: int) -> None:
    """Child process of the restart check: a batch run that a console break interrupts like Ctrl+C."""
    import signal

    if sys.platform == "win32":

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGBREAK, interrupt)
    logging.basicConfig(filename=Path(output) / "interrupted.log", level=logging.INFO, force=True)
    _run_batch_once(Path(output), basement_version, n_cpu, overwrite=True)


def run_restart_check(basement_version: str = bmt.DEFAULT_BASEMENT_VERSION, n_cpu: int = 1) -> dict:
    """
    Interrupt a batch run like Ctrl+C and restart it with overwrite False.

    1. Reference: the two benchmark cases, uninterrupted.
    2. Interrupted: the same run in a child process, interrupted (console break, handled as Ctrl+C) when the
       first case has finished and the second is in the optimization (serial), or when both are in the
       optimization (parallel).
    3. Checks: the child stops, no BASEMENT program keeps running, the memory files can be read.
    4. Restart with overwrite False: finished cases are skipped, interrupted ones continue their budget; every
       case must end with the same evaluations and best vector as the reference.

    Args:
        basement_version: BASEMENT version to run.
        n_cpu: CPU workers (1: serial, 2: parallel).

    Returns:
        dict: ``checks`` (name -> passed) and ``passed``.
    """
    import signal
    import subprocess

    label = "serial" if n_cpu == 1 else f"parallel ({n_cpu} CPU)"
    print(f"\n== Interrupt and restart check: {label} ==", flush=True)
    RESTART_FOLDER.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=RESTART_FOLDER / "restart_check.log", filemode="w", level=logging.INFO, force=True)
    write_batch_case(BATCH_FOLDER / "inputs")
    n_initial = BATCH_SURROGATE["n_initial"]
    names = sorted(BATCH_CASES)

    reference = RESTART_FOLDER / f"{n_cpu}cpu" / "reference"
    shutil.rmtree(reference, ignore_errors=True)
    print("1. Reference run (uninterrupted)", flush=True)
    _run_batch_once(reference, basement_version, n_cpu, overwrite=True)
    expected = _memory(reference)

    interrupted = RESTART_FOLDER / f"{n_cpu}cpu" / "interrupted"
    shutil.rmtree(interrupted, ignore_errors=True)
    interrupted.mkdir(parents=True)
    print("2. Interrupted run", flush=True)
    flags = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--batch-child",
            str(interrupted),
            basement_version,
            str(n_cpu),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **flags,
    )

    def ready() -> bool:
        rows = {name: len(memory) for name, memory in _memory(interrupted).items()}
        finished = [n for n in names if list(interrupted.glob(f"{n}/*/{batch.FINISHED_FILE_NAME}"))]
        if n_cpu == 1:  # first case done, second some iterations into the optimization
            return len(finished) == 1 and rows.get(names[1], 0) >= n_initial + 2
        return not finished and all(rows.get(n, 0) >= n_initial + 2 for n in names)

    start = time.perf_counter()
    while not ready():
        if child.poll() is not None:
            raise RuntimeError("The batch run ended before the interruption point.")
        if time.perf_counter() - start > 900:
            child.kill()
            raise RuntimeError("The interruption point was not reached within 15 minutes.")
        time.sleep(0.2)
    at_interrupt = {name: len(memory) for name, memory in _memory(interrupted).items()}
    finished_before = [n for n in names if list(interrupted.glob(f"{n}/*/{batch.FINISHED_FILE_NAME}"))]
    print(f"   Interrupting with evaluations {at_interrupt}, finished {finished_before or 'none'}", flush=True)
    if sys.platform == "win32":
        child.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(child.pid, signal.SIGINT)

    checks: dict[str, bool] = {}
    try:
        child.wait(timeout=60)
        checks["run stops within 60 s"] = True
    except subprocess.TimeoutExpired:
        child.kill()
        checks["run stops within 60 s"] = False
    time.sleep(2)
    left = _basement_processes()
    checks["no BASEMENT process left running"] = not left
    for process in left:
        process.kill()
    try:
        memories = _memory(interrupted)
        checks["memory files readable"] = all(memory["Calibration Error"].notna().all() for memory in memories.values())
    except Exception:
        checks["memory files readable"] = False
    after = {name: len(memory) for name, memory in _memory(interrupted).items()}
    print(f"   Stopped (exit code {child.returncode}); evaluations kept {after}", flush=True)

    print("3. Restart with overwrite: False", flush=True)
    _run_batch_once(interrupted, basement_version, n_cpu, overwrite=False)
    final = _memory(interrupted)
    for name in finished_before:
        checks[f"{name} finished earlier: not run again"] = len(final[name]) == after[name]
    for name in names:
        same_count = len(final[name]) == len(expected[name])
        best = final[name].loc[final[name]["Calibration Error"].idxmin()]
        best_ref = expected[name].loc[expected[name]["Calibration Error"].idxmin()]
        checks[f"{name}: {len(final[name])} evaluations (reference {len(expected[name])})"] = same_count
        checks[f"{name}: same best vector as the reference"] = best["Friction Vector"] == best_ref["Friction Vector"]
    summary = pd.read_csv(interrupted / batch.SUMMARY_FILE_NAME)
    checks["summary complete after restart"] = sorted(summary["status"]) == ["ok"] * len(names)

    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    passed = all(checks.values())
    print(f"Interrupt and restart check {'PASSED' if passed else 'FAILED'}; outputs in {RESTART_FOLDER}")
    return {"checks": checks, "passed": passed}


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark; skip with a message where BASEMENT is not installed."""
    parser = argparse.ArgumentParser(description="SWASHES MacDonald channel benchmarks for BOAR + BASEMENT.")
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default=DEFAULT_CASE,
        help="subcritical (section 3.5.1, default), supercritical (3.5.2) or jump (3.5.6).",
    )
    parser.add_argument(
        "--analytical-manning",
        action="store_true",
        help="Run only the analytical_manning run (BASEMENT with n = 0.03, no calibration).",
    )
    parser.add_argument(
        "--turbulence",
        choices=TURBULENCE_MODELS,
        default=None,
        help="Turbulence model: none or k_epsilon (default: k_epsilon for --case jump, none otherwise); "
        "each variant has its own output folder.",
    )
    parser.add_argument("--basement-version", default=bmt.DEFAULT_BASEMENT_VERSION, help='"4.2" or "4.3"')
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch-mode benchmark: two subcritical cases (shared template and own config), calibrated serially, "
        "with 2 CPU workers and, with --gpu, with 1 GPU + 1 CPU worker.",
    )
    parser.add_argument("--gpu", action="store_true", help="--batch: also run with 1 GPU + 1 CPU worker (CUDA).")
    parser.add_argument(
        "--batch-restart",
        action="store_true",
        help="Interrupt a batch run like Ctrl+C (serial, then 2 CPU workers) and check that a restart with "
        "overwrite False skips finished cases and continues the others to the uninterrupted result.",
    )
    parser.add_argument("--batch-child", nargs=3, metavar=("OUTPUT", "VERSION", "N_CPU"), help=argparse.SUPPRESS)
    args = parser.parse_args([] if argv is None else argv)

    exe = basement_executable(args.basement_version)
    if not exe.exists():
        print(f"BASEMENT benchmark skipped: {exe} not found (set BASEMENT_PATH in user_defined_configs/constants.py).")
        return

    if args.batch_child:
        _batch_child(args.batch_child[0], args.batch_child[1], int(args.batch_child[2]))
        return

    if args.batch_restart:
        results = [run_restart_check(args.basement_version, n_cpu) for n_cpu in (1, 2)]
        if not all(r["passed"] for r in results):
            raise SystemExit(1)
        return

    if args.batch:
        if not run_batch(args.basement_version, gpu=args.gpu)["passed"]:
            raise SystemExit(1)
        return

    result = run(
        args.case,
        calibrate=not args.analytical_manning,
        basement_version=args.basement_version,
        turbulence=args.turbulence,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    os.chdir(ROOT)
    main(sys.argv[1:])
