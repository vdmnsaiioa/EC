"""E3-QP2's configuration without a build: the training options of v0.0.11 (YAML and CLI), the far-field matching's
requirements, the site-model rungs' refusal on other truths, the readings' pins, the probe helpers for cores of any size,
and the shipped configs/water_qaqua_pol2.yaml."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent


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
    path = str(REPO / "configs" / "water_qaqua_pol2.yaml")
    assert config.config_parts(config.load_config(path)) == ["dimer", "clusters"]
    root = fake_build(tmp_path)
    a = parse_script(water, config, ["--config", path, "--part", "clusters", "--qaqua_root", root], monkeypatch)
    assert (a.readout_stage, a.readout_orders, a.readout_solve_iterations, a.readout_ridge_selection) == ("far_first", "all", 4, "groups")
    assert (a.lbfgs_freeze_readout, a.far_labels, a.w_far, a.lbfgs) == (True, "interaction", 1.0, 500)
    assert (a.probe_train, a.probe_train_cores, a.probe_train_core_sizes, a.probe_test, a.probe_test_cores) == (2, 16, "3,4,5", 10, 2)
    assert a.rungs.split(",") == ["M_inf", "M_16p", "M_16pm", "M_16C0p", "M_16S0pm", "M_16C0pm", "M_16S2pm", "M_16C2pm"]
    settings = water.cluster_data_settings(a)
    assert settings["probe_train_core_sizes"] == [3, 4, 5]
    kw = config.training_kwargs(a)
    assert (kw["readout_orders"], kw["readout_solve_iterations"], kw["lbfgs_freeze_readout"]) == ("all", 4, True)
    d = parse_script(water, config, ["--config", path, "--part", "dimer", "--qaqua_root", root], monkeypatch)
    assert d.rungs == "M_1pm,M_16pm" and d.readout_stage == "after_adam" and d.far_labels == "total"


@pytest.mark.parametrize("training,message", [
    ({"readout": {"stage": "first"}}, "readout.stage"),
    ({"readout": {"orders": "two"}}, "readout.orders"),
    ({"readout": {"solve_iterations": 0}}, "solve_iterations"),
    ({"readout": {"ridge_selection": "cv"}}, "ridge_selection"),
    ({"readout": {"far_weight": 0}}, "far_weight"),
    ({"lbfgs": {"freeze_readout": "yes"}}, "freeze_readout"),
    ({"loss": {"far_labels": "difference"}}, "far_labels"),
])
def test_invalid_v0011_training_fields(modules, training, message):
    config, _ = modules
    with pytest.raises(ValueError, match=message):
        config.validate_training(training)


@pytest.mark.parametrize("argv,message", [
    (["--rungs", "M_16S0pm,M_16S0p", "--readout_stage", "far_first", "--far_labels", "interaction",
      "--lbfgs_freeze_readout"], "species"),
    (["--rungs", "M_16S0pm", "--readout_stage", "far_first", "--lbfgs_freeze_readout"], "far_labels"),
    (["--rungs", "M_16S0pm", "--readout_stage", "far_first", "--far_labels", "interaction"], "freeze_readout"),
])
def test_far_field_matching_requirements(modules, monkeypatch, capsys, tmp_path, argv, message):
    config, water = modules
    root = fake_build(tmp_path)
    with pytest.raises(SystemExit):
        parse_script(water, config, ["--truth", "qaqua_pol", "--part", "clusters", "--qaqua_root", root, "--probe_train", "1",
                                     *argv], monkeypatch)
    assert message in capsys.readouterr().err


def test_site_model_rungs_need_q_aqua_pol(modules, monkeypatch, capsys):
    config, water = modules
    with pytest.raises(SystemExit):
        parse_script(water, config, ["--part", "clusters", "--rungs", "M_16pm"], monkeypatch)
    assert "q-AQUA-pol" in capsys.readouterr().err


def test_response_edge_reaches_the_rungs(modules):
    _, water = modules
    from sseft.units import ang_to_bohr
    a = SimpleNamespace(lA=4.0, rcut=6.0, F=16, l1=0.75, rs_over_l1=0.0, response_edge=1.2)
    r = water.configure(water.RUNGS["M_16C0pm"], a)
    assert abs(r.bf_l - ang_to_bohr(1.2)) < 1e-12 and abs(r.l1 - ang_to_bohr(0.75)) < 1e-12
    assert water.configure(water.RUNGS["M_16pm"], a).bf_l == 0.0                  # no read-out, no response edge


def test_probe_helpers_for_cores_of_any_size(modules):
    _, water = modules
    from sseft import water as W
    rng = np.random.default_rng(0)
    structs = []
    for n_core, shift in ((3, 9.0), (4, 3.0)):
        core = W.random_cluster(n_core, rng, mc_steps=0)
        probe = W.monomer_positions() + core[0::3].mean(0) + np.array([shift + 6.0, 0.0, 0.0])
        structs.append(W.water_structure(np.concatenate([core, probe]), energy=0.0))
    cores, pairs, owner = water.probe_parts(structs)
    assert [len(c) for c in cores] == [9, 12] and list(owner) == [0, 0, 0, 1, 1, 1, 1] and len(pairs) == 7
    gaps = [np.linalg.norm(c[:, None] - water.bohr_to_ang(s.positions)[-3:][None], axis=-1).min() for c, s in zip(cores, structs)]
    np.testing.assert_array_equal(water.far_field_blind(structs, 6.0), [g > 6.0 for g in gaps])


def test_readings_apply_the_rungs_pins(modules, monkeypatch):
    """model_nbody pins what the rung reads (E3-QP's M_16 2-body readings lacked the dispersion alpha)."""
    _, water = modules
    from sseft import water as W
    seen = []
    monkeypatch.setattr(water.T, "predict_energies", lambda params, rung, structs: seen.extend(structs) or np.zeros(len(structs)))
    frames = np.array([np.concatenate([W.monomer_positions(), W.monomer_positions() + [3.0, 0.0, 0.0]])])
    water.model_nbody(None, water.RUNGS["M_16p"], frames, np.tile(W.monomer_charges(), (1, 2)), 2, truth=W.PW_TRUTH)
    assert seen and all("alpha" in s.pinned for s in seen)
