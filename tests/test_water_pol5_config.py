"""E3-QP5's configuration without a build: configs/water_qaqua_pol5.yaml is E3-QP4's job B with the field rungs
(Rung.e0_field) beside their bases -- the one change -- and it resolves and validates."""
import copy
import importlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent
RUNGS = ["M_16pm", "M_16pm_f", "M_16C2pm", "M_16C2pm_f", "M_inf_f"]


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
    path = str(REPO / "configs" / "water_qaqua_pol5.yaml")
    assert config.config_parts(config.load_config(path)) == ["clusters"]
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    assert a.part == "clusters" and a.rungs.split(",") == RUNGS
    assert (a.F, a.rcut, a.l1, a.response_edge, a.heldout_pin_charges, a.w_contact_energy) == (32, 6.0, 0.75, 0.75, "rigid", 0.125)
    assert (a.w_energy, a.w_force, a.steps, a.lbfgs, a.batch, a.lbfgs_chunk) == (100.0, 1.0, 4000, 1000, 32, 64)
    assert (a.readout_stage, a.readout_orders, a.readout_solve_iterations, a.readout_ridge_selection) == ("far_first", "all", 4, "groups")
    water.validate_water_args(a)
    for name in RUNGS:
        rung = water.configure(water.RUNGS[name], a)
        assert rung.e0_field == name.endswith("_f") and rung.msites
        if rung.e0_field:
            assert rung.e0_field_off < rung.e0["r_cut"]


def test_one_change_against_job_b(modules):
    """E3-QP5 is E3-QP4's job B with the field rungs beside their bases: every other value is job B's."""
    config, _ = modules
    load = lambda name: config.load_config(str(REPO / "configs" / name))
    b, five = load("water_qaqua_pol4b.yaml"), load("water_qaqua_pol5.yaml")
    five_as_b = copy.deepcopy(five)
    assert five_as_b["parts"]["clusters"].pop("rungs") == RUNGS
    b["parts"]["clusters"].pop("rungs")
    assert five_as_b == b


def test_field_rungs_refused_without_the_site_model(modules, monkeypatch, tmp_path):
    """the field term reads TTM3-F's site charges: q-AQUA-pol only, like the other site-model rungs."""
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol5.yaml")
    a = parse_script(water, config, ["--config", path, "--qaqua_root", fake_build(tmp_path)], monkeypatch)
    a.truth, a.rungs = "qaqua", "M_inf_f"                    # (the M_16 rungs are refused first, for TTM3-F's alpha)
    with pytest.raises(ValueError, match="site charges"):
        water.validate_water_args(a)
