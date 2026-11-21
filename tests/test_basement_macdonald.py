"""
Tests for the SWASHES MacDonald channel benchmarks (benchmarks/basement_macdonald.py).

The analytical solutions, mesh generation, case files and block comparison are tested without BASEMENT.
The cross-check against pyswashes runs only where the SWASHES program is installed, and the end-to-end
benchmarks only where BASEMENT is installed (``pytest -m "slow and requires_basement"``).
"""

from __future__ import annotations

import json
import logging

import basement_macdonald as bm
import numpy as np
import pandas as pd
import pytest

from src import basement_tools as bmt
from src import functions_io as fio
from user_defined_configs import function_loss

LOGGER = logging.getLogger("test_basement_macdonald")
SUB, SUPER, JUMP = bm.CASES["subcritical"], bm.CASES["supercritical"], bm.CASES["jump"]


class TestAnalyticalSolutions:
    def test_subcritical_matches_published_values(self):
        # Values of the pyswashes documentation example for this case (PseudoTwoDimensional(1, 1, 1, 5))
        np.testing.assert_allclose(bm.h_exact([20.0, 60.0, 100.0]), [0.912229, 1.0348, 1.2], atol=1e-4)

    def test_boundary_depths_match_table_2(self):
        assert SUPER.h_in() == pytest.approx(0.503369, abs=1e-6)
        assert JUMP.h_out() == pytest.approx(1.2, abs=1e-9)

    @pytest.mark.parametrize("case", bm.CASES.values(), ids=list(bm.CASES))
    def test_bed_falls_to_zero_at_outlet(self, case):
        x = np.linspace(0.0, case.length, 401)
        z = case.bed_elevation(x)
        assert z[-1] == pytest.approx(0.0, abs=1e-12)
        assert np.all(np.diff(z) < 0.0)  # S0 > 0 everywhere in these cases

    @pytest.mark.parametrize("case", bm.CASES.values(), ids=list(bm.CASES))
    def test_derivatives_match_finite_differences(self, case):
        x, eps = np.linspace(1.0, case.length - 1.0, 57), 1e-5
        if case.jump_at is not None:
            x = x[np.abs(x - case.jump_at) > 1.0]
        np.testing.assert_allclose(case.dwidth(x), (case.width(x + eps) - case.width(x - eps)) / (2 * eps), atol=1e-8)
        np.testing.assert_allclose(case.dh(x), (case.h(x + eps) - case.h(x - eps)) / (2 * eps), atol=1e-7)

    def test_flow_regimes(self):
        x = np.linspace(0.0, 200.0, 2001)
        assert np.all(SUB.froude(x) < 1.0)
        assert np.all(SUPER.froude(x) > 1.0)
        before, after = JUMP.jump_at - 1e-9, JUMP.jump_at + 1e-9
        assert JUMP.froude(0.0) < 1.0 and JUMP.froude(before) > 1.0 and JUMP.froude(after) < 1.0
        assert JUMP.froude(JUMP.length) < 1.0

    def test_jump_conserves_momentum(self):
        """The depths on both sides of the jump are conjugate: equal momentum function of the trapezoid."""
        b, z = float(JUMP.width(JUMP.jump_at)), JUMP.bank_slope

        def momentum(h):
            return bm.DISCHARGE**2 / (bm.GRAVITY * (b + z * h) * h) + b * h**2 / 2 + z * h**3 / 3

        h_up, h_down = float(JUMP.h(JUMP.jump_at)), float(JUMP.h(JUMP.jump_at + 1e-9))
        assert h_up == pytest.approx(0.9124, abs=1e-4) and h_down == pytest.approx(1.0834, abs=1e-4)
        assert momentum(h_up) == pytest.approx(momentum(h_down), rel=1e-3)

    def test_trapezoid_bed_slope_reduces_to_rectangle(self):
        rectangle = bm.MacDonaldCase(**{**JUMP.__dict__, "bank_slope": 0.0})
        x = np.linspace(130.0, 390.0, 7)
        b, db = JUMP.bottom(x)
        h, dh = JUMP.depth(x)
        q2, n2, g = bm.DISCHARGE**2, bm.MANNING_N**2, bm.GRAVITY
        expected = (1 - q2 / (g * h**3 * b**2)) * dh + q2 * n2 * (b + 2 * h) ** (4 / 3) / (h * b) ** (10 / 3)
        expected -= q2 * db / (g * h**2 * b**3)
        np.testing.assert_allclose(rectangle.bed_slope(x), expected, rtol=1e-12)

    def test_matches_pyswashes(self):
        """Cross-check with SWASHES; needs the swashes program (conda), so it is skipped elsewhere."""
        pyswashes = pytest.importorskip("pyswashes")
        try:
            solution = pyswashes.PseudoTwoDimensional(1, 1, 1, 200)
        except RuntimeError as exc:
            pytest.skip(f"SWASHES program not available: {exc}")
        df = solution.dataframe()
        x = df.index.to_numpy(dtype=float)
        np.testing.assert_allclose(df["depth"].to_numpy(), bm.h_exact(x), atol=1e-5)
        np.testing.assert_allclose(df["gd_elev"].to_numpy(), _swashes_topography(x), atol=1e-5)

    def test_swashes_topography_rule_matches_its_documented_output(self):
        """pyswashes documents the SWASHES output of this case on 5 cells; our S0 reproduces it with SWASHES' rule."""
        x = np.array([20.0, 60.0, 100.0, 140.0, 180.0])
        documented = [1.644770, 0.844012, 0.314957, 0.281316, 0.093772]  # pyswashes PseudoTwoDimensional(1, 1, 1, 5)
        np.testing.assert_allclose(_swashes_topography(x), documented, atol=1e-5)  # printed with 6 digits
        np.testing.assert_allclose(bm.h_exact(x), [0.912229, 1.0348, 1.2, 1.0348, 0.912229], atol=1e-5)


def _swashes_topography(x: np.ndarray) -> np.ndarray:
    """
    Topography as SWASHES writes it on its output grid (cell centres x, spacing dx): z = S0 dx / 2 in the last
    cell and z_i = z_(i+1) + S0_(i+1) dx upstream, a first-order sum that tends to int_x^L S0 dX as dx -> 0.
    """
    dx = x[1] - x[0]
    s0 = bm.bed_slope(x)
    z = np.empty_like(x)
    z[-1] = s0[-1] * dx / 2
    for i in range(len(x) - 2, -1, -1):
        z[i] = z[i + 1] + s0[i + 1] * dx
    return z


class TestMesh:
    def test_rectangular_round_trip(self, tmp_path):
        mesh = fio.read_mesh_file(str(bm.write_2dm(tmp_path / "mesh.2dm", SUB, dx=4.0, ny=4)))
        nx, ncol = 50, 5
        assert len(mesh["nodes"]) == (nx + 1) * ncol
        assert len(mesh["cells"]) == 2 * nx * (ncol - 1)
        assert set(mesh["cells"]["region"]) == {1, 2, 3}
        assert {k: len(v) for k, v in mesh["boundary_conditions"].items()} == {"inflow": ncol, "outflow": ncol}
        nodes = mesh["nodes"]
        np.testing.assert_allclose(nodes["z"], SUB.bed_elevation(nodes["x"]), atol=1e-8)
        np.testing.assert_allclose(
            np.abs(nodes["y"]).reshape(nx + 1, ncol).max(axis=1), SUB.width(nodes["x"][::ncol]) / 2
        )

    def test_trapezoid_banks(self, tmp_path):
        mesh = fio.read_mesh_file(str(bm.write_2dm(tmp_path / "mesh.2dm", JUMP, dx=8.0, ny=4)))
        nx, ncol = 50, 4 + 1 + 2 * bm.BANK_CELLS
        nodes = mesh["nodes"]
        assert len(nodes) == (nx + 1) * ncol and len(mesh["cells"]) == 2 * nx * (ncol - 1)

        x = nodes["x"]
        offset = np.maximum(np.abs(nodes["y"]) - JUMP.width(x) / 2, 0.0)  # horizontal distance up the bank
        np.testing.assert_allclose(nodes["z"], JUMP.bed_elevation(x) + offset / JUMP.bank_slope, atol=1e-8)
        crest = JUMP.bed_elevation(x) + bm.BANK_HEIGHT
        assert np.all(crest >= JUMP.bed_elevation(x) + JUMP.h(x) + 0.5)  # the crest stays dry

        # Boundary strings: the bottom plus the bank edges below the boundary water surface
        assert {k: len(v) for k, v in mesh["boundary_conditions"].items()} == {"inflow": 9, "outflow": 9}

    def test_triangles_are_counter_clockwise_pairs(self, tmp_path):
        path = bm.write_2dm(tmp_path / "mesh.2dm", JUMP, dx=10.0, ny=2)
        mesh = fio.read_mesh_file(str(path))
        nodes = {n["id"]: (n["x"], n["y"]) for n in mesh["nodes"]}
        for cell in mesh["cells"]:
            (x1, y1), (x2, y2), (x3, y3) = (nodes[cell[k]] for k in ("nd1", "nd2", "nd3"))
            assert (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) > 0.0
        x = np.linspace(0.0, JUMP.length, 40001)
        plan = (JUMP.width(x) + 2 * JUMP.bank_slope * bm.BANK_HEIGHT).mean() * JUMP.length
        assert bm.triangle_areas(path).sum() == pytest.approx(plan, rel=1e-3)

    def test_inflow_froude_gives_the_analytical_depth(self):
        columns = bm.mesh_columns(JUMP)
        first, last = bm.string_columns(columns, JUMP.h_in())
        b = (columns[last][1] - columns[first][1]) + JUMP.width(0.0)
        fr = bm.inflow_froude(JUMP)
        assert ((bm.DISCHARGE / b) ** 2 / (bm.GRAVITY * fr**2)) ** (1 / 3) == pytest.approx(JUMP.h_in())


class TestCaseFiles:
    def _model(self, folders):
        return json.loads((folders["SIMULATION_CONFIG_FOLDER"] / "model.json").read_text())

    def test_subcritical_layout(self, tmp_path):
        folders = bm.write_case(tmp_path / "inputs")
        for name in fio.SIMULATION_CONFIG_FILES:
            assert (folders["SIMULATION_CONFIG_FOLDER"] / name).is_file()
        model = self._model(folders)
        assert model["SETUP"]["simulation_name"] == "swashes_macdonald"
        hyd = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]
        assert "TURBULENCE" not in hyd
        assert hyd["INITIAL"] == {"type": "dry"} and hyd["PARAMETER"]["CFL"] == 0.5
        inflow, outflow = hyd["BOUNDARY"]["STANDARD"]
        assert inflow["type"] == "uniform_in" and outflow["type"] == "zhydrograph"
        assert outflow["wse"] == pytest.approx(bm.h_out())  # z(L) = 0

        hydrograph = np.loadtxt(tmp_path / "inputs" / "discharge" / f"{bm.DISCHARGE_CODE}.txt", delimiter=",")
        assert hydrograph[0].tolist() == [0.0, 0.0] and hydrograph[1, 1] == bm.DISCHARGE

        gt = pd.read_csv(folders["GROUND_TRUTH_FOLDER"] / f"MACDONALD_{bm.DISCHARGE_CODE}_1.csv")
        assert list(gt.columns) == ["X [m]", "Y [m]", "H [m]", "Ux [m/s]", "Uy [m/s]"]
        assert len(gt) == 21 and gt["X [m]"].max() == bm.LENGTH

    def test_supercritical_boundaries(self, tmp_path):
        folders = bm.write_case(tmp_path / "inputs", "supercritical")
        model = self._model(folders)
        assert model["SETUP"]["simulation_name"] == "swashes_macdonald_supercritical"
        inflow, outflow = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
        assert inflow["type"] == "zqhydrograph" and outflow["type"] == "zero_gradient_out"
        wse = np.loadtxt(inflow["wse_file"], delimiter=",")
        q = np.loadtxt(inflow["spec_discharge_file"], delimiter=",")
        assert np.allclose(wse[:, 1], SUPER.bed_elevation(0.0) + SUPER.h_in())
        assert q[0, 1] == 0.0 and q[1, 1] == pytest.approx(bm.DISCHARGE / SUPER.width(0.0))

    def test_jump_case(self, tmp_path):
        folders = bm.write_case(tmp_path / "inputs", "jump")
        model = self._model(folders)
        assert model["SETUP"]["simulation_name"] == "swashes_macdonald_jump"
        inflow, outflow = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["BOUNDARY"]["STANDARD"]
        assert inflow["type"] == "froude_in" and inflow["froude_number"] == pytest.approx(bm.inflow_froude(JUMP))
        assert outflow["wse"] == pytest.approx(1.2)
        gt = pd.read_csv(folders["GROUND_TRUTH_FOLDER"] / f"MACDONALD_{bm.DISCHARGE_CODE}_1.csv")
        assert len(gt) == 20 and JUMP.jump_at not in gt["X [m]"].to_numpy()  # point on the jump left out

    def test_boar_discharge_update_keeps_the_supercritical_inflow(self, tmp_path):
        """define_discharge rewrites uniform_in/froude_in only, so the zqhydrograph inflow is left as written."""
        folders = bm.write_case(tmp_path / "inputs", "supercritical")
        case = fio.prepare_simulation_folder(
            folders["SIMULATION_CONFIG_FOLDER"], tmp_path / "runs", overwrite=True, logger=LOGGER
        )
        before = json.loads((case / "model.json").read_text())
        manager = bmt.SimulationManager.__new__(bmt.SimulationManager)
        manager.sim_folder, manager.logger, manager.log_dev = str(case), LOGGER, False
        manager.define_discharge("Q001.txt", file=True)
        assert json.loads((case / "model.json").read_text()) == before

    def test_k_epsilon_is_the_jump_default(self, tmp_path):
        """The jump case runs k-epsilon by default; the short channels run without a turbulence model."""
        folders = bm.write_case(tmp_path / "inputs", "jump")
        model = self._model(folders)
        assert model["SETUP"]["simulation_name"] == "swashes_macdonald_jump"
        turbulence = model["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]["TURBULENCE"]
        assert turbulence["REYNOLDS_STRESS"]["closure_model"] == "k_epsilon"
        assert turbulence["PARAMETER"]["turbulence_start"] == bm.TURBULENCE_START < JUMP.end_time
        outputs = json.loads((folders["SIMULATION_CONFIG_FOLDER"] / "simulation.json").read_text())
        assert {"turb_k", "turb_reynolds"} <= set(outputs["SIMULATION"]["OUTPUT"])
        for key in ("subcritical", "supercritical"):
            assert "TURBULENCE" not in bm.model_config(key)["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]

    @pytest.mark.parametrize(
        "key, turbulence, name",
        [
            ("subcritical", None, "swashes_macdonald"),
            ("subcritical", "k_epsilon", "swashes_macdonald_k_epsilon"),
            ("supercritical", "none", "swashes_macdonald_supercritical"),
            ("jump", None, "swashes_macdonald_jump"),
            ("jump", "k_epsilon", "swashes_macdonald_jump"),
            ("jump", "none", "swashes_macdonald_jump_no_turbulence"),
        ],
    )
    def test_variant_names_follow_the_case_default(self, key, turbulence, name):
        assert bm.variant_name(key, turbulence) == name
        assert bm.benchmark_folder(key, turbulence) == bm.BENCHMARK_FOLDER.parent / name

    @pytest.mark.parametrize("key", list(bm.CASES))
    def test_k_epsilon_starts_within_the_run(self, key):
        hyd = bm.model_config(key, "k_epsilon")["SETUP"]["BASEHPC"]["BASEPLANE_2D"]["HYDRAULICS"]
        assert 0 < hyd["TURBULENCE"]["PARAMETER"]["turbulence_start"] < bm.get_case(key).end_time

    @pytest.mark.parametrize(
        "key, names",
        [
            ("subcritical", ["upstream", "contraction", "downstream"]),
            ("supercritical", ["upstream", "contraction", "downstream"]),
            ("jump", ["before_jump", "after_jump"]),
        ],
    )
    def test_friction_regions(self, tmp_path, key, names):
        """One Manning region per reach: thirds of the short channels, before and after the jump."""
        case = bm.get_case(key)
        folders = bm.write_case(tmp_path / "inputs", key)
        geometry = self._model(folders)["SETUP"]["BASEHPC"]["BASEPLANE_2D"]
        assert [r["name"] for r in geometry["GEOMETRY"]["REGIONDEF"]] == names
        assert [r["index"] for r in geometry["GEOMETRY"]["REGIONDEF"]] == [[k] for k in range(1, len(names) + 1)]
        assert [r["region_name"] for r in geometry["HYDRAULICS"]["FRICTION"]["regions"]] == names

        mesh = fio.read_mesh_file(str(folders["MESH_FOLDER"] / f"{bm.CASE_NAME}.2dm"))
        cells, nodes = mesh["cells"], mesh["nodes"]
        assert np.all(cells["region"][0::2] == cells["region"][1::2])  # both triangles of a rectangle
        x = dict(zip(nodes["id"], nodes["x"], strict=True))
        cx = np.array([np.mean([x[c[k]] for k in ("nd1", "nd2", "nd3")]) for c in cells[0::2]])
        cx = (cx // bm.DX + 0.5) * bm.DX  # centre x of each rectangle
        np.testing.assert_array_equal(cells["region"][0::2], case.region_index(cx))
        if case.jump_at is not None:
            assert set(cells["region"][0::2][cx < case.jump_at]) == {1}
            assert set(cells["region"][0::2][cx > case.jump_at]) == {2}

        options = bm.benchmark_options(case=key)["optimization_variable_options"]
        assert options["regions"] == names

    def test_unknown_names_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown turbulence model"):
            bm.variant_name("subcritical", "smagorinsky")
        with pytest.raises(ValueError, match="Unknown case"):
            bm.get_case("dam_break")

    @pytest.mark.parametrize("key", list(bm.CASES))
    def test_case_is_accepted_by_boar(self, tmp_path, key):
        folders = bm.write_case(tmp_path / "inputs", key)
        case = fio.prepare_simulation_folder(
            folders["SIMULATION_CONFIG_FOLDER"], tmp_path / "runs", overwrite=True, logger=LOGGER
        )
        plane = json.loads((case / "model.json").read_text())["SETUP"]["BASEHPC"]["BASEPLANE_2D"]
        assert (folders["MESH_FOLDER"] / f"{bm.CASE_NAME}.2dm").resolve().samefile(plane["GEOMETRY"]["mesh_file"])


def _synthetic_sim(path, h_of_x) -> tuple[dict, np.ndarray]:
    """Simulation data on the benchmark mesh with a prescribed depth field."""
    mesh = fio.read_mesh_file(str(path))
    nodes = {n["id"]: (n["x"], n["y"]) for n in mesh["nodes"]}
    cx = np.array([np.mean([nodes[c[k]][0] for k in ("nd1", "nd2", "nd3")]) for c in mesh["cells"]])
    cy = np.array([np.mean([nodes[c[k]][1] for k in ("nd1", "nd2", "nd3")]) for c in mesh["cells"]])
    hyd = np.zeros(len(cx), dtype=[(k, "f8") for k in ("wse", "qx", "qy", "h", "ux", "uy")])
    hyd["h"] = h_of_x(cx)
    hyd["ux"] = 1.0
    sim = {"centroids": pd.DataFrame({"cx": cx, "cy": cy}), "hyd": hyd}
    return sim, bm.triangle_areas(path)


class TestComparison:
    def test_exact_depth_gives_small_error(self, tmp_path):
        sim, areas = _synthetic_sim(bm.write_2dm(tmp_path / "m.2dm"), bm.h_exact)
        assert bm.depth_errors(sim, areas, bm.ground_truth())["rmse"] < 2e-3  # only the IDW smoothing

    def test_offset_depth_is_measured(self, tmp_path):
        sim, areas = _synthetic_sim(bm.write_2dm(tmp_path / "m.2dm"), lambda x: bm.h_exact(x) + 0.1)
        assert bm.depth_errors(sim, areas, bm.ground_truth())["rmse"] == pytest.approx(0.1, abs=2e-3)
        loss, _ = bm.make_depth_loss(areas)(sim, [bm.ground_truth()])
        assert loss == pytest.approx(0.1, abs=2e-3)

    def test_block_means_pair_consecutive_triangles(self, tmp_path):
        sim, areas = _synthetic_sim(bm.write_2dm(tmp_path / "m.2dm"), bm.h_exact)
        sim["hyd"]["h"] = np.tile([1.0, 3.0], len(areas) // 2)
        blocks = bm.block_means(sim, areas)
        assert len(blocks) == len(areas) // 2
        np.testing.assert_allclose(blocks["h"], 2.0, atol=0.05)  # the two triangles have nearly equal areas


def test_default_loss_accepts_velocity_columns():
    """function_loss iterates over the point coordinates only, so ground truth may carry Ux/Uy columns."""
    centroids = pd.DataFrame({"cx": [0.0, 1.0, 0.0, 1.0, 0.5, 0.5], "cy": [0.0, 0.0, 1.0, 1.0, 0.5, 0.2]})
    hyd = np.zeros(6, dtype=[(k, "f8") for k in ("h", "ux", "uy")])
    hyd["h"], hyd["ux"] = 1.0, 2.0
    gt = pd.DataFrame({"X [m]": [0.4], "Y [m]": [0.4], "H [m]": [1.0], "Ux [m/s]": [2.0], "Uy [m/s]": [0.0]})
    loss, errors = function_loss.loss_function({"centroids": centroids, "hyd": hyd}, [gt])
    assert loss == pytest.approx(0.0)
    assert len(errors) == 1


@pytest.mark.slow
@pytest.mark.requires_basement
@pytest.mark.parametrize(
    "case, turbulence",
    [("subcritical", "none"), ("subcritical", "k_epsilon"), ("supercritical", "none"), ("jump", None)],
)
def test_benchmark_end_to_end(case, turbulence):
    """analytical_manning and BOAR calibrated runs; the final water depth must match the analytical depth."""
    if not bm.basement_executable().exists():
        pytest.skip(f"BASEMENT not installed: {bm.basement_executable()}")
    result = bm.run(case, calibrate=True, basement_version=bmt.DEFAULT_BASEMENT_VERSION, turbulence=turbulence)
    assert result["passed"]
    assert result["boar_calibrated"]["rmse"] <= bm.DEPTH_RMSE_TOL
