"""E3-QP's configuration without a build: the qaqua_pol truth's refusals, the probe data of the clusters part (validation,
dataset settings that leave runs without probes as they were), the shipped configs/water_qaqua_pol.yaml, the pinned
dispersion band with a truth's own alpha and C6, and the PW probe generator as before (values recorded at 4a81a84)."""
import hashlib
import importlib
import math
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
    """a directory that passes the build checks: the library file and the raw sets (empty)."""
    root = tmp_path / "build_pol"
    (root / "q-AQUA").mkdir(parents=True)
    (root / "libqaqua_pol.so").write_bytes(b"")
    for name in ("2b_data.xyz", "3b_data.xyz", "4b_data.xyz"):
        (root / "q-AQUA" / name).write_text("")
    return str(root)


@pytest.mark.parametrize("argv,message", [
    (["--truth", "qaqua_pol", "--part", "far"], "clusters part"),
    (["--truth", "qaqua_pol", "--part", "ion"], "dimer and clusters parts"),
    (["--truth", "qaqua_pol", "--part", "clusters", "--rungs", "M_1pemb"], "embedded"),
    (["--truth", "qaqua_pol", "--part", "dimer", "--rungs", "M_1p,M_16p", "--qaqua_root", "/nowhere"], "build_pol.sh"),
])
def test_qaqua_pol_refusals(modules, monkeypatch, capsys, argv, message):
    config, water = modules
    with pytest.raises(SystemExit):
        parse_script(water, config, argv, monkeypatch)
    assert message in capsys.readouterr().err


def test_permanent_source_daggers_are_admitted_on_pol(modules, monkeypatch, tmp_path):
    config, water = modules
    root = fake_build(tmp_path)
    a = parse_script(water, config, ["--truth", "qaqua_pol", "--part", "clusters", "--rungs", "M_16p,M_16S0p,M_S0p",
                                     "--qaqua_root", root, "--heldout", "3b"], monkeypatch)
    assert a.truth == "qaqua_pol" and water.truth_root(a) == root
    a.qaqua_root = None
    assert water.truth_root(a) == "external/qaqua/build_pol"


@pytest.mark.parametrize("data,message", [
    ({"probe_train": -1}, "probe_train"),
    ({"probe_test_cores": 0}, "probe_test_cores"),
    ({"probe_distances_angstrom": []}, "probe_distances_angstrom"),
    ({"probe_distances_angstrom": [8.0, 8.0]}, "probe_distances_angstrom"),
    ({"probe_seed": 1.5}, "probe_seed"),
])
def test_probe_data_validation(modules, data, message):
    config, _ = modules
    with pytest.raises(ValueError, match=message):
        config.validate_cluster_data(data)


def test_probe_settings_only_when_requested(modules):
    config, water = modules
    base = {k: v for k, v in config.CLUSTER_DEFAULTS.items()}
    a = SimpleNamespace(**base, **config.CLUSTER_PROBE_DEFAULTS)
    settings = water.cluster_data_settings(a)
    assert set(settings) == set(config.CLUSTER_DEFAULTS)             # the dataset settings of runs before probes existed
    a.probe_test = 3
    settings = water.cluster_data_settings(a)
    assert settings["probe_distances_angstrom"] == [8.0, 10.0, 12.0, 15.0] and settings["probe_test"] == 3
    assert set(settings) == set(config.CLUSTER_DEFAULTS) | set(config.CLUSTER_PROBE_DEFAULTS)


def test_shipped_config_resolves(modules, monkeypatch, tmp_path):
    config, water = modules
    path = str(REPO / "configs" / "water_qaqua_pol.yaml")
    loaded = config.load_config(path)
    assert config.config_parts(loaded) == ["dimer", "clusters"]
    root = fake_build(tmp_path)
    a = parse_script(water, config, ["--config", path, "--part", "clusters", "--qaqua_root", root], monkeypatch)
    assert (a.truth, a.n_train, a.n_test, a.probe_train, a.probe_test, a.probe_train_cores, a.probe_test_cores) == \
        ("qaqua_pol", 100, 24, 5, 10, 8, 2)
    assert a.probe_distances_angstrom == "8.0,10.0,12.0,15.0" and a.rungs.split(",")[-2:] == ["M_16p", "M_16S0p"]
    assert (a.steps, a.batch, a.lbfgs, a.lbfgs_chunk, a.w_far) == (2000, 32, 500, 64, 1.0)
    d = parse_script(water, config, ["--config", path, "--part", "dimer", "--qaqua_root", root], monkeypatch)
    assert (d.lbfgs, d.lbfgs_chunk, d.batch, d.rcut) == (1000, 48, None, 5.0) and "M_16p" in d.rungs


def test_pinned_oscillator_takes_the_truths_alpha_and_c6(modules):
    _, water = modules
    from sseft import kernels as kn, water as W
    omega, w = kn.casimir_polder_grid(8)
    pw = water.pinned_alpha_oscillator([8, 1])
    w0 = 4 * water.C6_OO_TRUTH / (3 * W.ALPHA_O ** 2)
    np.testing.assert_array_equal(pw[0], W.ALPHA_O / (1 + (np.array(omega) / w0) ** 2))       # PW's, as before
    assert np.all(pw[1] == 0)
    own = water.pinned_alpha_oscillator([8], alpha=9.7446, c6=52.469)[0]
    assert abs(own[0] - 9.7446 / (1 + (float(omega[0]) / (4 * 52.469 / (3 * 9.7446 ** 2))) ** 2)) < 1e-12
    assert abs(float(kn.c6_from_alpha(own, own, w)) / 52.469 - 1) < 1e-4                      # the quadrature's C6
    truth = SimpleNamespace(alpha=9.7446, c6=52.469)
    assert water.truth_dispersion(truth) == (9.7446, 52.469) and water.truth_dispersion(None) == (None, None)


def test_pw_probe_generator_reproduces_the_recorded_data():
    from sseft import water as W
    st, Ec = W.probe_configurations(3, (6.0, 8.0), 1, seed=7, mc_steps=15, n_cores=1)
    np.testing.assert_allclose([s.energy for s in st], [0.002544890471715524, 0.002740737246340629], rtol=1e-12, atol=0)
    np.testing.assert_allclose(Ec, [0.002504389109810147] * 2, rtol=1e-12, atol=0)
    digest = hashlib.sha256(np.ascontiguousarray(st[0].positions, dtype=np.float64).tobytes()).hexdigest()[:16]
    if digest != "1671624490eea5c9":                                 # bit-for-bit on the recording platform
        again, _ = W.probe_configurations(3, (6.0, 8.0), 1, seed=7, mc_steps=15, n_cores=1, truth=W.PW_TRUTH)
        np.testing.assert_array_equal(st[0].positions, again[0].positions)
    np.testing.assert_array_equal(st[0].pinned["q"], np.tile(W.monomer_charges(), 4))
