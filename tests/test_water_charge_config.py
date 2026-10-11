"""Water charge-source selection from YAML through the Coulomb and field paths."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from sseft import bands as B, bandfields as BF, e0 as E0, heads as H
from sseft import model as M, structure as S, water as W


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("experiment_config"), importlib.import_module("e3_water")


def write_config(tmp_path, **sections):
    path = tmp_path / "water.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"type": "water"}, **sections}))
    return str(path)


def parse(config, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="dimer")
    parser.add_argument("--rungs", default="M_1p")
    parser.add_argument("--charges", choices=["pinned", "learned"], default=None)
    return config.parse_selection(parser, "water", argv)


def architecture_args(charges=None, **extra):
    return SimpleNamespace(charges=charges, lA=4.0, rcut=6.0, F=2,
                           l1=0.0, rs_over_l1=1.0, **extra)


@pytest.mark.parametrize("mode", ["pinned", "learned", None])
def test_yaml_charge_mode_becomes_water_argument(tmp_path, modules, mode):
    config, _ = modules
    path = write_config(tmp_path, model={"charges": mode})
    assert parse(config, ["--config", path]).charges == mode


@pytest.mark.parametrize("sections", [{}, {"model": {}}, {"model": {"charges": None}}])
def test_unspecified_charge_mode_preserves_rung_defaults(tmp_path, modules, sections):
    config, water = modules
    args = parse(config, ["--config", write_config(tmp_path, **sections)])
    assert args.charges is None
    for name in ("M_1", "M_1p", "M_S0", "M_S0p"):
        preset = water.RUNGS[name]
        assert water.configure(preset, architecture_args(args.charges)).pin_q == preset.pin_q


@pytest.mark.parametrize("yaml_mode,cli_mode", [("pinned", "learned"), ("learned", "pinned")])
def test_cli_charge_mode_overrides_yaml(tmp_path, modules, yaml_mode, cli_mode):
    config, _ = modules
    path = write_config(tmp_path, model={"charges": yaml_mode})
    assert parse(config, ["--config", path, "--charges", cli_mode]).charges == cli_mode


@pytest.mark.parametrize("model", [
    None, [], "pinned", {"pin_q": True}, {"charges": False},
    {"charges": 1}, {"charges": "off"}, {"charges": []},
])
def test_invalid_model_settings_are_rejected(tmp_path, modules, model):
    config, _ = modules
    with pytest.raises(ValueError):
        config.load_config(write_config(tmp_path, model=model))


@pytest.mark.parametrize("kind", ["dimer_ladder", "x2b_toy"])
def test_water_charge_model_settings_are_rejected_for_other_experiments(tmp_path, modules, kind):
    config, _ = modules
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"type": kind}, "model": {}}))
    with pytest.raises(ValueError):
        config.load_config(path)


@pytest.mark.parametrize("mode", ["pinned", "learned"])
def test_water_entry_point_accepts_charge_cli_before_computation(modules, monkeypatch, mode):
    config, water = modules
    monkeypatch.setattr("sys.argv", ["e3_water.py", "--charges", mode])

    class Parsed(Exception):
        pass

    def capture(parser, kind, **kwargs):
        assert config.parse_selection(parser, kind, **kwargs).charges == mode
        raise Parsed

    monkeypatch.setattr(water, "parse_selection", capture)
    with pytest.raises(Parsed):
        water.main()


@pytest.mark.parametrize("name,mode", [("M_S0", "pinned"), ("M_S0p", "learned"),
                                      ("M_S2mu", "pinned"), ("M_16pemb", "learned")])
def test_charge_override_only_changes_charge_pinning(modules, name, mode):
    _, water = modules
    preset = water.RUNGS[name]
    configured = water.configure(preset, architecture_args(mode))
    assert configured.pin_q is (mode == "pinned")
    for attribute in ("charges", "dipoles", "pin_mu", "dispersion", "pin_alpha",
                      "dispersion8", "pin_alpha2", "band_fields"):
        assert getattr(configured, attribute) == getattr(preset, attribute)


@pytest.mark.parametrize("mode", ["pinned", "learned"])
@pytest.mark.parametrize("name", ["M_inf", "M_G", "M_A"])
def test_charge_override_keeps_charge_free_models_charge_free(modules, mode, name):
    _, water = modules
    preset = water.RUNGS[name]
    configured = water.configure(preset, architecture_args(mode))
    assert configured.charges is False
    assert configured.pin_q == preset.pin_q


def test_learning_charges_preserves_required_dispersion_pins(modules):
    _, water = modules
    structure = W.water_structure(W.dimer_positions(4.0))
    water.pin_sources(architecture_args("learned", rungs="M_16S0p"), [structure])
    np.testing.assert_array_equal(structure.pinned["q"], np.tile(W.monomer_charges(), 2))
    assert "alpha" in structure.pinned
    assert np.any(structure.pinned["alpha"] > 0)


def test_charge_override_controls_actual_coulomb_and_field_sources(modules):
    _, water = modules
    structure = W.water_structure(W.dimer_positions(4.0))
    batch = {key: value[0] for key, value in S.pad_batch([structure]).items()}
    base = water.configure(water.RUNGS["M_S0p"], architecture_args("learned"))
    base = M.with_rung(base, e0={**base.e0, "T": 1, "n_rbf": 2, "n_rbf_readout": 2},
                       site_energies=False, fragments=False)
    params = M.init_params(jax.random.PRNGKey(12), base)
    # A known learned head gives sources distinct from the synthetic pinned truth.
    params["heads"]["charge"] = jax.tree.map(jnp.zeros_like, params["heads"]["charge"])
    params["heads"]["q_ref"] = params["heads"]["q_ref"].at[8].set(0.2).at[1].set(-0.1)
    s, _, _, (distances, radii, pair_mask) = E0.apply_e0(
        params["e0"], base.e0, batch["positions"], batch["numbers"],
        batch["mask"], batch["cell"], batch["pbc"])
    learned_q = H.charges(params["heads"], s, batch["numbers"],
                          batch["mask"], batch["total_charge"])
    same_fragment = (batch["frag_mask"].T.astype(float) @ batch["frag_mask"].astype(float)) > 0.5
    fields = []
    for pinned in (False, True):
        rung = M.with_rung(base, pin_q=pinned)
        energy, aux = M.energy_single(params, rung, batch)
        expected_q = batch["pin_q"] if pinned else learned_q
        np.testing.assert_allclose(aux["q"], expected_q, atol=1e-14)
        expected_field = BF.band_field_tensors(
            batch["positions"], expected_q, pair_mask, rung.l1,
            rung.band_fields, rung.bf_rs, src_mask=~same_fragment)[0]
        np.testing.assert_allclose(aux["Efield"], expected_field, atol=1e-14)
        expected_coulomb = B.coulomb_band(expected_q, None, distances, radii, pair_mask, rung.l1)
        np.testing.assert_allclose(aux["E_coul"], expected_coulomb, atol=1e-14)
        np.testing.assert_allclose(energy, expected_coulomb, atol=1e-14)
        fields.append(np.asarray(aux["Efield"]))
    assert not np.allclose(learned_q, batch["pin_q"])
    assert not np.allclose(fields[0], fields[1])


@pytest.mark.parametrize("name,requested,effective", [
    ("M_S0p", "learned", "learned"), ("M_S0", "pinned", "pinned"),
    ("M_1p", None, "pinned"), ("M_1", None, "learned"),
    ("M_inf", "pinned", "off"),
])
def test_saved_results_record_effective_charge_mode(tmp_path, modules, name, requested, effective):
    _, water = modules
    args = SimpleNamespace(out=str(tmp_path / "result.json"), rungs=name, charges=requested)
    water._save(args, {name: {"rmse": [0.0]}})
    saved = json.loads(Path(args.out).read_text())
    assert saved[name]["model"]["charges"] == effective
