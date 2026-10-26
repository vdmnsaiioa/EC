"""E3-QP8's configuration without a build: configs/water_qaqua_pol8.yaml is E3-QP7's with its rungs replaced -- M_16C2pm,
M_16C2pm_w and the field rungs with their channels apart (M_16C2pm_ws: the self terms at their own reach 4-5 A;
M_16C2pm_us: those self terms beside degree-2 cross terms by species) -- the one change, and it resolves and validates."""
import copy
import importlib
from pathlib import Path

import pytest

from sseft.units import ang_to_bohr

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent
RUNGS = ["M_16C2pm", "M_16C2pm_w", "M_16C2pm_ws", "M_16C2pm_us"]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("experiment_config"), importlib.import_module("e3_water")


def parse_script(water, config, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["e3_water.py", *argv])
    out = {}

    class Parsed(Exception):
        pass

    def capture(parser, kind, **kwargs):
        out["args"] = config.parse_selection(parser, kind, **kwargs)
        raise Parsed
    monkeypatch.setattr(water, "parse_selection", capture)
    with pytest.raises(Parsed):
        water.main()
    return out["args"]


def fake_build(tmp_path):
    root = tmp_path / "build_pol"
    (root / "q-AQUA").mkdir(parents=True)
    (root / "libqaqua_pol.so").write_bytes(b"")
    for name in ("2b_data.xyz", "3b_data.xyz", "4b_data.xyz"):
        (root / "q-AQUA" / name).write_text("")
    return str(root)


def test_shipped_config_resolves(modules, monkeypatch, tmp_path):
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol8.yaml")
    assert config.config_parts(config.load_config(path)) == ["clusters"]
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    assert a.part == "clusters" and a.rungs.split(",") == RUNGS
    assert (a.F, a.rcut, a.l1, a.response_edge, a.heldout_pin_charges, a.w_contact_energy) == (32, 6.0, 0.75, 0.75, "rigid", 0.125)
    assert (a.w_energy, a.w_force, a.steps, a.lbfgs, a.batch, a.lbfgs_chunk) == (100.0, 1.0, 4000, 1000, 32, 64)
    assert (a.readout_stage, a.readout_orders, a.readout_solve_iterations, a.readout_ridge_selection) == ("far_first", "all", 4, "groups")
    water.validate_water_args(a)
    for name in RUNGS:
        rung = water.configure(water.RUNGS[name], a)
        assert rung.e0_field == (name != "M_16C2pm") and rung.msites
        if rung.e0_field:
            assert rung.e0_field_whole and rung.e0_field_off == rung.e0["r_cut"]
            assert rung.e0_field_form == "mlp" and not rung.e0_field_cross
            assert (rung.e0_field_self_off > 0) == (name != "M_16C2pm_w")
            assert rung.e0_field_cross_species == (name == "M_16C2pm_us")
            if rung.e0_field_self_off > 0:
                assert abs(rung.e0_field_self_off - (rung.e0["r_cut"] - ang_to_bohr(1.0))) < 1e-12
                assert abs(rung.e0_field_self_on - (rung.e0["r_cut"] - ang_to_bohr(2.0))) < 1e-12


def test_one_change_against_e3qp7(modules):
    """E3-QP8 is E3-QP7 with its rungs replaced: every other value is E3-QP7's (and so E3-QP6's, E3-QP5's and E3-QP4 job
    B's)."""
    config, _ = modules
    load = lambda name: config.load_config(str(REPO / "configs" / name))
    seven, eight = load("water_qaqua_pol7.yaml"), load("water_qaqua_pol8.yaml")
    eight_as_seven = copy.deepcopy(eight)
    assert eight_as_seven["parts"]["clusters"].pop("rungs") == RUNGS
    seven["parts"]["clusters"].pop("rungs")
    assert eight_as_seven == seven


def test_field_rungs_refused_without_the_site_model(modules, monkeypatch, tmp_path):
    """the field term reads TTM3-F's site charges: q-AQUA-pol only, like the other site-model rungs (on q-AQUA v1 every
    rung of the run is refused, the split rungs among them)."""
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol8.yaml")
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    a.truth = "qaqua"
    with pytest.raises(ValueError, match="M_16C2pm, M_16C2pm_w, M_16C2pm_ws, M_16C2pm_us"):   # (for TTM3-F's alpha first)
        water.validate_water_args(a)
