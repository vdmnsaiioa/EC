"""Multi-part water configurations (experiment.part as a list, per-part overrides under parts), the dimer design and the
truth mapping in YAML, the dispatcher's validate-then-run sequence, and the q-AQUA refusals -- without running anything."""
import argparse
import importlib
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("experiment_config"), importlib.import_module("e3_water")


def write(tmp_path, content, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content))
    return str(path)


MULTI = {
    "experiment": {"type": "water", "part": ["dimer", "clusters"]},
    "model": {"rs_over_l1": 0.0, "features": 16},
    "training": {"adam": {"learning_rate": 0.003}, "lbfgs": {"memory_size": 10}},
    "parts": {
        "dimer": {"rungs": ["M_1", "M_1p"], "model": {"r_cut_angstrom": 5.0},
                  "data": {"window_nodes": 30, "re_angstrom": 2.9}, "training": {"seeds": [0, 1, 2], "adam": {"steps": 50}}},
        "clusters": {"rungs": ["M_S0p"], "model": {"band_edge_angstrom": 0.75},
                     "data": {"n_train": 2}, "training": {"seeds": [5], "lbfgs": {"steps": 7}}},
    },
}


def parse_script(water, config, argv, monkeypatch):
    """parse through e3_water.py's own parser, stopping before any computation."""
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


def test_parts_merge_over_the_shared_sections(tmp_path, modules, monkeypatch):
    config, water = modules
    path = write(tmp_path, MULTI)
    loaded = config.load_config(path)
    dimer = config.part_config(loaded, "dimer")
    assert dimer["experiment"] == {"type": "water", "part": "dimer", "rungs": ["M_1", "M_1p"]}
    assert dimer["model"] == {"rs_over_l1": 0.0, "features": 16, "r_cut_angstrom": 5.0}
    assert dimer["training"] == {"adam": {"learning_rate": 0.003, "steps": 50}, "lbfgs": {"memory_size": 10}, "seeds": [0, 1, 2]}
    a = parse_script(water, config, ["--config", path, "--part", "dimer"], monkeypatch)
    assert (a.part, a.rungs, a.rcut, a.rs_over_l1, a.window_nodes, a.re_angstrom, a.steps) == ("dimer", "M_1,M_1p", 5.0, 0.0, 30, 2.9, 50)
    assert config.training_seeds(a) == [0, 1, 2] and a.learning_rate == 0.003
    a = parse_script(water, config, ["--config", path, "--part", "clusters"], monkeypatch)
    assert (a.part, a.rungs, a.l1, a.n_train, a.lbfgs, a.rcut) == ("clusters", "M_S0p", 0.75, 2, 7, 6.0)
    assert config.training_seeds(a) == [5]
    a = parse_script(water, config, ["--config", path, "--part", "clusters", "--n_train", "3", "--l1", "1.5"], monkeypatch)
    assert (a.n_train, a.l1) == (3, 1.5)                                    # explicit flags still win


def test_several_parts_need_a_selection(tmp_path, modules, monkeypatch, capsys):
    config, water = modules
    with pytest.raises(SystemExit):
        parse_script(water, config, ["--config", write(tmp_path, MULTI)], monkeypatch)
    assert "select one with --part" in capsys.readouterr().err


@pytest.mark.parametrize("change", [
    {"parts": {"dimmer": {}}},                                               # unknown part
    {"parts": {"dimer": {"truth": {}}}},                                     # a section a part cannot override
    {"parts": {"dimer": []}},
    {"data": {"n_train": 2}},                                                # shared data with parts
    {"parts": {"dimer": {"data": {"n_train": 2}}}},                          # cluster data under the dimer part
    {"parts": {"dimer": {"data": {"window_ratio": 1.0}}}},
    {"parts": {"dimer": {"data": {"window_nodes": 1}}}},
    {"parts": {"far": {"data": {"sigma": 0.0}}}},                            # a part without a data schema
    {"parts": {"clusters": {"rungs": []}}},
    {"parts": {"clusters": {"training": {"adam": {"steps": -1}}}}},
    {"experiment": {"type": "water", "part": ["dimer", "dimer"]}},
    {"experiment": {"type": "water", "part": []}},
    {"truth": {"name": "ccsdt"}},
    {"truth": {"heldout": {"sets": ["5b"]}}},
    {"truth": {"heldout": {"rigid_tolerance": {"oh_angstrom": 0}}}},
    {"truth": {"heldout": {"max_frames": 0}}},
    {"truth": {"source": "x"}},
])
def test_invalid_multipart_configs_are_rejected(tmp_path, modules, change):
    config, _ = modules
    content = {**MULTI, **change}
    if "parts" in change:                                                    # replace or add single parts
        content["parts"] = {**MULTI["parts"], **change["parts"]}
    with pytest.raises(ValueError):
        config.load_config(write(tmp_path, content))


def test_truth_and_parts_are_water_only(tmp_path, modules):
    config, _ = modules
    for content in ({"experiment": {"type": "x2b_toy"}, "truth": {"name": "pw"}},
                    {"experiment": {"type": "dimer_ladder"}, "parts": {"dimer": {}}}):
        with pytest.raises(ValueError):
            config.load_config(write(tmp_path, content))


def test_single_part_dimer_data(tmp_path, modules, monkeypatch, capsys):
    config, water = modules
    path = write(tmp_path, {"experiment": {"type": "water", "part": "dimer"}, "data": {"window_nodes": 24, "phi_degrees": 10.0}})
    a = parse_script(water, config, ["--config", path], monkeypatch)
    assert (a.window_nodes, a.phi_degrees, a.window_ratio) == (24, 10.0, 1.8)
    with pytest.raises(SystemExit):                                          # dimer data with the clusters part
        parse_script(water, config, ["--config", path, "--part", "clusters"], monkeypatch)
    assert "unsupported" in capsys.readouterr().err


def test_truth_mapping_reaches_the_arguments(tmp_path, modules, monkeypatch):
    config, water = modules
    content = {"experiment": {"type": "water", "part": "clusters", "rungs": ["M_1p"]},
               "truth": {"name": "pw", "root": "/nowhere", "heldout": None}}
    a = parse_script(water, config, ["--config", write(tmp_path, content)], monkeypatch)
    assert (a.truth, a.qaqua_root, a.heldout) == ("pw", "/nowhere", "none")
    content["truth"] = {"name": "pw", "heldout": {"sets": ["3b"], "rigid_tolerance": {"oh_angstrom": 0.05},
                                                  "strict_tolerance": {"angle_degrees": 2.0}, "max_frames": 9}}
    with pytest.raises(SystemExit):                                          # held-out sets need the q-AQUA truth
        parse_script(water, config, ["--config", write(tmp_path, content)], monkeypatch)


@pytest.mark.parametrize("argv,message", [
    (["--truth", "qaqua", "--part", "far"], "dimer and clusters parts"),
    (["--truth", "qaqua", "--part", "clusters", "--rungs", "M_1p,M_16p"], "M_16p"),
    (["--truth", "qaqua", "--part", "clusters", "--rungs", "M_1pemb"], "M_1pemb"),
    (["--truth", "qaqua", "--part", "dimer", "--rungs", "M_1p", "--qaqua_root", "/nowhere"], "build.sh"),
])
def test_qaqua_refusals(modules, monkeypatch, capsys, argv, message):
    config, water = modules
    with pytest.raises(SystemExit):
        parse_script(water, config, argv, monkeypatch)
    assert message in capsys.readouterr().err


def test_validate_only_exits_before_running(tmp_path, modules, monkeypatch, capsys):
    config, water = modules
    monkeypatch.setattr("sys.argv", ["e3_water.py", "--config", write(tmp_path, MULTI), "--part", "dimer", "--validate_only"])
    for name in ("run_dimer", "run_clusters", "run_far", "run_ion"):
        monkeypatch.setattr(water, name, lambda *args, **kw: pytest.fail("ran"))
    with pytest.raises(SystemExit) as exc:
        water.main()
    assert exc.value.code == 0 and "configuration valid: water, part dimer" in capsys.readouterr().out


def test_dispatcher_validates_every_part_then_runs_them(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    run_experiment = importlib.import_module("run_experiment")
    path = write(tmp_path, MULTI, "multi.yaml")
    calls = []
    monkeypatch.setattr(run_experiment.runpy, "run_path", lambda target, **kw: calls.append(sys.argv[:]))
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", path, "--out_dir", str(tmp_path / "out"), "--steps", "2"])
    run_experiment.main()
    assert [c[3:5] for c in calls] == [["--part", "dimer"], ["--part", "clusters"]] * 2
    assert all("--validate_only" in c for c in calls[:2]) and not any("--validate_only" in c for c in calls[2:])
    for call, part in zip(calls[2:], ("dimer", "clusters")):
        assert call[call.index("--out") + 1] == str(tmp_path / "out" / f"{part}.json")
        assert call[call.index("--cache") + 1] == str(tmp_path / "out" / f"{part}_cache.pkl")
        assert "--steps" in call
    # one part on the command line runs alone, without the validation pass
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", path, "--part", "clusters"])
    run_experiment.main()
    assert len(calls) == 1 and calls[0][1:] == ["--config", path, "--part", "clusters"]
    # several parts cannot share one --out
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", path, "--out", "x.json"])
    with pytest.raises(SystemExit):
        run_experiment.main()


def test_dispatcher_stops_before_running_when_a_part_is_invalid(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    run_experiment = importlib.import_module("run_experiment")
    calls = []

    def fake(target, **kw):
        calls.append(sys.argv[:])
        if "--validate_only" in sys.argv and "clusters" in sys.argv:
            raise SystemExit(2)
    monkeypatch.setattr(run_experiment.runpy, "run_path", fake)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", write(tmp_path, MULTI), "--out_dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        run_experiment.main()
    assert "clusters" in str(exc.value.code) and all("--validate_only" in c for c in calls)


def test_shipped_qaqua_config_is_the_registered_protocol(modules):
    """configs/water_qaqua.yaml as pre-registered in notes/e3q-preregistration.md (checked without the library)."""
    config, _ = modules
    loaded = config.load_config(REPO / "configs" / "water_qaqua.yaml")
    assert config.config_parts(loaded) == ["dimer", "clusters"]
    assert loaded["truth"]["name"] == "qaqua" and loaded["truth"]["heldout"]["sets"] == ["2b", "3b", "4b"]
    dimer, clusters = (config.part_config(loaded, p) for p in ("dimer", "clusters"))
    assert dimer["experiment"]["rungs"] == ["M_inf", "M_G", "M_A", "M_1", "M_1mu", "M_1p"]
    assert dimer["model"]["r_cut_angstrom"] == 5.0 and dimer["model"]["band_edge_angstrom"] is None
    assert dimer["training"]["seeds"] == list(range(8)) and dimer["training"]["adam"]["steps"] == 1500
    assert dimer["training"]["adam"]["batch_size"] is None and dimer["training"]["lbfgs"] == {
        "learning_rate": None, "memory_size": 10, "scale_init_precond": True, "max_linesearch_steps": 20,
        "readout_only": False, "steps": 300, "chunk_size": 48}
    assert clusters["experiment"]["rungs"] == ["M_inf", "M_1p", "M_S0p", "M_S2p", "M_1mu"]
    assert clusters["model"]["band_edge_angstrom"] == 0.75 and clusters["model"]["r_cut_angstrom"] == 6.0
    assert clusters["training"]["seeds"] == [0, 1, 2, 3] and clusters["training"]["adam"]["steps"] == 1000
    assert clusters["data"] == {**{k: v for k, v in config.CLUSTER_DEFAULTS.items() if k not in ("train_sizes", "test_sizes")},
                                "train_sizes": [3, 4, 5], "test_sizes": [6, 2]}
