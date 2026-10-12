"""The water model mapping: architecture settings from YAML to the configured rung and into the results json."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sseft.units import bohr_to_ang, ang_to_bohr


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("experiment_config"), importlib.import_module("e3_water")


def write_config(tmp_path, model, experiment=None):
    path = tmp_path / "water.yaml"
    path.write_text(yaml.safe_dump({"experiment": experiment or {"type": "water", "part": "ion", "rungs": ["M_16S0p"]},
                                    "model": model}))
    return str(path)


def parse_with_script(water, config, argv, monkeypatch):
    """parse through e3_water.py's own parser (the training flags included), without running anything."""
    monkeypatch.setattr("sys.argv", ["e3_water.py", *argv])
    parsed = {}

    class Parsed(Exception):
        pass

    def capture(parser, kind, **kwargs):
        parsed["args"] = config.parse_selection(parser, kind, **kwargs)
        raise Parsed
    monkeypatch.setattr(water, "parse_selection", capture)
    with pytest.raises(Parsed):
        water.main()
    return parsed["args"]


MODEL = {"charges": None, "charge_neutrality": "fragment", "band_edge_angstrom": 0.75, "rs_over_l1": 0.0,
         "r_cut_angstrom": 6.0, "features": 16, "analytic_envelope_angstrom": 4.0}


def test_model_fields_reach_the_script_arguments(tmp_path, modules, monkeypatch):
    config, water = modules
    args = parse_with_script(water, config, ["--config", write_config(tmp_path, MODEL)], monkeypatch)
    assert (args.charges, args.charge_neutrality, args.l1, args.rs_over_l1, args.rcut, args.F, args.lA) == \
        (None, "fragment", 0.75, 0.0, 6.0, 16, 4.0)


def test_null_band_edge_means_the_rung_default_and_cli_overrides_yaml(tmp_path, modules, monkeypatch):
    config, water = modules
    path = write_config(tmp_path, {**MODEL, "band_edge_angstrom": None, "charge_neutrality": "structure"})
    args = parse_with_script(water, config, ["--config", path], monkeypatch)
    assert args.l1 == 0.0
    for name in ("M_1", "M_16S0p"):
        rung = water.configure(water.RUNGS[name], args)
        assert abs(bohr_to_ang(rung.l1) - 1.5) < 1e-12 and rung.frag_neutral is False
    args = parse_with_script(water, config, ["--config", path, "--l1", "1.0", "--rs_over_l1", "0.5",
                                             "--charge_neutrality", "fragment", "--F", "8"], monkeypatch)
    assert (args.l1, args.rs_over_l1, args.charge_neutrality, args.F) == (1.0, 0.5, "fragment", 8)
    rung = water.configure(water.RUNGS["M_16S0p"], args)
    assert abs(bohr_to_ang(rung.l1) - 1.0) < 1e-12 and abs(bohr_to_ang(rung.bf_rs) - 0.5) < 1e-12
    assert rung.frag_neutral and rung.e0["F"] == 8


def test_near_source_switch_scales_with_the_configured_band_edge(modules):
    _, water = modules
    args = SimpleNamespace(charges=None, charge_neutrality="fragment", lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=1.0)
    rung = water.configure(water.RUNGS["M_16S0p"], args)
    assert abs(bohr_to_ang(rung.bf_rs) - 0.75) < 1e-12      # r_s = rs_over_l1 x the band edge actually used
    args.rs_over_l1 = 0.0
    assert water.configure(water.RUNGS["M_16S0p"], args).bf_rs == 0.0


@pytest.mark.parametrize("model", [
    {"band_edge_angstrom": 0}, {"band_edge_angstrom": -0.5}, {"band_edge_angstrom": "0.75"},
    {"rs_over_l1": -1}, {"r_cut_angstrom": 0}, {"features": 0}, {"features": 16.0}, {"features": True},
    {"analytic_envelope_angstrom": float("inf")}, {"charge_neutrality": "atom"}, {"charge_neutrality": None},
    {"l1": 0.75}, {"rcut": 6.0}, {"pin_q": True},
])
def test_invalid_model_fields_are_rejected(tmp_path, modules, model):
    config, _ = modules
    with pytest.raises(ValueError):
        config.load_config(write_config(tmp_path, model))


def test_invalid_cli_architecture_is_rejected(modules, monkeypatch):
    config, water = modules
    for flags in (["--l1", "-1"], ["--rcut", "0"], ["--F", "0"], ["--rs_over_l1", "-0.1"]):
        with pytest.raises(SystemExit):
            parse_with_script(water, config, flags, monkeypatch)


def test_results_record_the_architecture_as_run(tmp_path, modules):
    _, water = modules
    args = SimpleNamespace(out=str(tmp_path / "result.json"), rungs="M_16S0p,M_1mu,M_inf", charges=None,
                           charge_neutrality="fragment", lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0)
    water._save(args, {name: {"rmse": [0.0]} for name in args.rungs.split(",")})
    saved = json.loads(Path(args.out).read_text())
    assert saved["M_16S0p"]["model"] == {
        "charges": "pinned", "charge_neutrality": "fragment", "dipoles": "off", "polarisabilities": "pinned",
        "band_fields": 0, "band_edge_angstrom": 0.75, "near_source_switch_angstrom": 0.0,
        "r_cut_angstrom": 6.0, "features": 16, "analytic_envelope_angstrom": 4.0, "periodic": False}
    assert saved["M_1mu"]["model"]["charges"] == "learned" and saved["M_1mu"]["model"]["dipoles"] == "learned"
    assert saved["M_1mu"]["model"]["band_fields"] is None and saved["M_1mu"]["model"]["near_source_switch_angstrom"] is None
    assert saved["M_inf"]["model"]["charges"] == "off" and saved["M_inf"]["model"]["polarisabilities"] == "off"
    args.charge_neutrality = "structure"; args.l1 = 0.0
    water._save(args, {"M_1mu": {"rmse": [0.0]}})
    saved = json.loads(Path(args.out).read_text())
    assert saved["M_1mu"]["model"]["charge_neutrality"] == "structure"
    assert abs(saved["M_1mu"]["model"]["band_edge_angstrom"] - 1.5) < 1e-12


def test_shipped_water_config_is_the_e3r_cluster_protocol(modules, monkeypatch):
    """configs/water.yaml reproduces the command of the E3-R cluster run (notes/e3r-results.md section 1, final
    schedule): the three permanent-source daggers at l_1 = 0.75 A without the radial switch, K = 2, Adam 1000 with
    the read-out frozen, its order-0 linear solve, L-BFGS 300, batch 12, chunk 6, the pre-registered cluster data."""
    config, water = modules
    args = parse_with_script(water, config, ["--config", str(REPO / "configs" / "water.yaml")], monkeypatch)
    assert (args.part, args.rungs) == ("clusters", "M_16p,M_16S0p,M_16S2p")
    assert config.training_seeds(args) == [0, 1]
    kwargs = config.training_kwargs(args)
    assert (kwargs["steps"], kwargs["lbfgs_steps"], kwargs["batch_size"], kwargs["lbfgs_chunk"]) == (1000, 300, 12, 6)
    assert kwargs["readout_adam"] is False and kwargs["readout_linear_solve"] is True
    assert (kwargs["lr"], kwargs["w_energy"], kwargs["w_force"]) == (3e-3, 1.0, 1.0)
    assert water.cluster_data_settings(args) == dict(config.CLUSTER_DEFAULTS, train_sizes=[3, 4, 5], test_sizes=[6, 2])
    for name in args.rungs.split(","):
        rung = water.configure(water.RUNGS[name], args)
        assert rung.pin_q and rung.pin_alpha and rung.dispersion and rung.frag_neutral
        assert abs(bohr_to_ang(rung.l1) - 0.75) < 1e-12 and abs(bohr_to_ang(rung.e0["r_cut"]) - 6.0) < 1e-12
        if rung.band_fields >= 0:
            assert rung.bf_rs == 0.0 and rung.bf_fragments
