"""Training YAML, command-line precedence, and fit argument forwarding."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("training_experiment_config", SCRIPTS / "experiment_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

TRAINING = {
    "seeds": [7, 19], "energy_scale": 0.02, "verbose": False,
    "adam": {
        "steps": 23, "learning_rate": 0.004, "final_learning_rate_ratio": 0.15,
        "beta1": 0.8, "beta2": 0.95, "epsilon": 2e-7, "epsilon_root": 1e-9,
        "gradient_clip_norm": 1.5, "batch_size": 5,
    },
    "readout": {"freeze_during_adam": True, "linear_solve": True, "ridge": 3e-6},
    "lbfgs": {
        "steps": 11, "chunk_size": 3, "learning_rate": 0.7, "memory_size": 4,
        "scale_init_precond": False, "max_linesearch_steps": 9, "readout_only": True,
    },
    "loss": {"energy_weight": 2.0, "force_weight": 0.5, "far_weight": 4.0},
}


def write_config(tmp_path, training, kind="water", part="clusters", rungs=None):
    experiment = {"type": kind}
    if kind == "water":
        experiment.update(part=part, rungs=rungs or ["M_S0p"])
    elif kind == "dimer_ladder":
        experiment.update(systems=["Ar2"], truth="tt", rungs=rungs or ["M_S0"])
    else:
        experiment["orders"] = [0, 2]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"experiment": experiment, "training": training}))
    return str(path)


def parser(kind="water"):
    result = argparse.ArgumentParser()
    if kind == "water":
        result.add_argument("--part", choices=["dimer", "clusters", "far", "ion"], default="dimer")
        result.add_argument("--rungs", default="M_inf")
    elif kind == "dimer_ladder":
        result.add_argument("--system", default="Ar2")
        result.add_argument("--truth", choices=["tt", "published"], default="tt")
        result.add_argument("--rungs", default="M_inf")
    else:
        result.add_argument("--orders", default="0,2")
    config.add_training_arguments(result, kind)
    return result


def parse(kind="water", argv=()):
    rungs = {"M_inf": SimpleNamespace(band_fields=-1),
             "M_S0p": SimpleNamespace(band_fields=0), "M_S0": SimpleNamespace(band_fields=0)}
    return config.parse_selection(parser(kind), kind, list(argv), rungs=None if kind == "x2b_toy" else rungs)


def test_all_training_values_are_forwarded_to_fit(tmp_path):
    path = write_config(tmp_path, TRAINING)
    args = parse(argv=["--config", path])
    assert config.training_seeds(args) == [7, 19]
    assert args.w_far == 4.0
    assert config.training_kwargs(args) == {
        "steps": 23, "lr": 0.004, "lr_final_ratio": 0.15,
        "adam_b1": 0.8, "adam_b2": 0.95, "adam_eps": 2e-7, "adam_eps_root": 1e-9,
        "gradient_clip_norm": 1.5, "batch_size": 5,
        "readout_adam": False, "readout_linear_solve": True, "readout_ridge": 3e-6,
        "lbfgs_steps": 11, "lbfgs_chunk": 3, "lbfgs_lr": 0.7, "lbfgs_memory": 4,
        "lbfgs_scale_init_precond": False, "lbfgs_max_linesearch_steps": 9,
        "lbfgs_readout_only": True, "w_energy": 2.0, "w_force": 0.5,
        "scale": 0.02, "verbose": False,
    }


def test_cli_overrides_training_yaml_without_discarding_other_fields(tmp_path):
    path = write_config(tmp_path, TRAINING)
    args = parse(argv=[
        "--config", path, "--steps", "9", "--lbfgs", "8", "--learning_rate", "0.01",
        "--model_seeds", "2,4", "--no-freeze_readout", "--no-readout_linear_solve",
        "--no-lbfgs_readout_only", "--batch", "2",
    ])
    kwargs = config.training_kwargs(args)
    assert config.training_seeds(args) == [2, 4]
    assert kwargs["steps"] == 9
    assert kwargs["lbfgs_steps"] == 8
    assert kwargs["lr"] == 0.01
    assert kwargs["batch_size"] == 2
    assert kwargs["readout_adam"] is True
    assert kwargs["readout_linear_solve"] is False
    assert kwargs["lbfgs_readout_only"] is False
    assert kwargs["lbfgs_memory"] == 4
    assert kwargs["w_force"] == 0.5


@pytest.mark.parametrize("kind,legacy,expected", [
    ("water", ["--seeds", "3"], [0, 1, 2]),
    ("dimer_ladder", ["--seeds", "2"], [0, 1]),
    ("x2b_toy", ["--seed", "5"], [5]),
])
def test_explicit_legacy_seed_option_overrides_yaml_seed_list(tmp_path, kind, legacy, expected):
    path = write_config(tmp_path, {"seeds": [7, 19]}, kind=kind, rungs=["M_inf"])
    assert config.training_seeds(parse(kind, ["--config", path, *legacy])) == expected


@pytest.mark.parametrize("kind,flags", [
    ("water", ["--seeds", "2", "--model_seeds", "3,7"]),
    ("x2b_toy", ["--seed", "2", "--model_seeds", "3,7"]),
])
def test_conflicting_explicit_seed_options_are_rejected(kind, flags):
    with pytest.raises(SystemExit):
        parse(kind, flags)


@pytest.mark.parametrize("kind,part,batch,chunk,freeze,verbose", [
    ("water", "dimer", None, 48, True, False),
    ("water", "clusters", 12, 6, True, False),
    ("water", "far", 12, 6, True, False),
    ("water", "ion", 12, 6, True, False),
    ("dimer_ladder", None, None, 48, True, False),
    ("x2b_toy", None, 48, 48, False, True),
])
def test_existing_workflow_defaults_are_preserved(kind, part, batch, chunk, freeze, verbose):
    args = parse(kind, ["--part", part] if part else [])
    kwargs = config.training_kwargs(args)
    assert kwargs["batch_size"] == batch
    assert kwargs["lbfgs_chunk"] == chunk
    assert kwargs["readout_adam"] is (not freeze)
    assert kwargs["verbose"] is verbose
    assert kwargs["lr"] == 0.003
    assert kwargs["lr_final_ratio"] == 0.01


def test_null_batch_and_scale_select_automatic_values(tmp_path):
    path = write_config(tmp_path, {"energy_scale": None, "adam": {"batch_size": None}})
    kwargs = config.training_kwargs(parse(argv=["--config", path]))
    assert kwargs["batch_size"] is None
    assert kwargs["scale"] is None


@pytest.mark.parametrize("training", [
    None, [], "bad", {"unknown": 1}, {"adam": None}, {"lbfgs": []},
    {"adam": {"unknown": 1}}, {"readout": {"unknown": True}},
    {"seeds": []}, {"seeds": [1, 1]}, {"seeds": [-1]}, {"seeds": [True]},
    {"seeds": [0.5]}, {"seeds": "0,1"},
    {"energy_scale": 0}, {"energy_scale": float("nan")},
    {"verbose": 1}, {"adam": {"steps": -1}}, {"adam": {"steps": True}},
    {"adam": {"learning_rate": 0}}, {"adam": {"learning_rate": float("inf")}},
    {"adam": {"final_learning_rate_ratio": 1.1}}, {"adam": {"beta1": 1.0}},
    {"adam": {"beta2": -0.1}}, {"adam": {"epsilon": -1e-8}},
    {"adam": {"gradient_clip_norm": 0}}, {"adam": {"batch_size": 0}},
    {"readout": {"freeze_during_adam": "yes"}}, {"readout": {"linear_solve": 1}},
    {"readout": {"ridge": -1}}, {"lbfgs": {"steps": -1}},
    {"lbfgs": {"chunk_size": 0}}, {"lbfgs": {"memory_size": 0}},
    {"lbfgs": {"learning_rate": 0}}, {"lbfgs": {"max_linesearch_steps": 0}},
    {"lbfgs": {"readout_only": 1}}, {"loss": {"energy_weight": -1}},
    {"loss": {"force_weight": True}}, {"loss": {"far_weight": -1}},
])
def test_invalid_training_yaml_is_rejected(tmp_path, training):
    with pytest.raises(ValueError):
        config.load_config(write_config(tmp_path, training))


@pytest.mark.parametrize("training", [
    {"adam": {"steps": 0}, "lbfgs": {"steps": 0}},
    {"loss": {"energy_weight": 0, "force_weight": 0}},
    {"adam": {"epsilon": 0, "epsilon_root": 0}},
])
def test_invalid_combined_training_settings_are_rejected(tmp_path, training):
    path = write_config(tmp_path, training)
    with pytest.raises((ValueError, SystemExit)):
        parse(argv=["--config", path])


def test_zero_adam_budget_is_valid_with_subsequent_steps(tmp_path):
    path = write_config(tmp_path, {"adam": {"steps": 0}, "lbfgs": {"steps": 10}})
    kwargs = config.training_kwargs(parse(argv=["--config", path]))
    assert kwargs["steps"] == 0
    assert kwargs["lbfgs_steps"] == 10


def test_response_only_polish_requires_response_model(tmp_path):
    path = write_config(tmp_path, {"lbfgs": {"readout_only": True}}, rungs=["M_inf"])
    with pytest.raises(SystemExit):
        parse(argv=["--config", path])


@pytest.mark.parametrize("training,rungs", [
    ({"lbfgs": {"readout_only": True}}, ["M_inf", "M_S0p"]),
    ({"readout": {"freeze_during_adam": False},
      "lbfgs": {"readout_only": True, "steps": 0}}, ["M_S0p"]),
    ({"readout": {"freeze_during_adam": True, "linear_solve": True},
      "lbfgs": {"steps": 0}}, ["M_S0p"]),
])
def test_incompatible_response_stage_settings_are_rejected(tmp_path, training, rungs):
    path = write_config(tmp_path, training, rungs=rungs)
    with pytest.raises(SystemExit):
        parse(argv=["--config", path])


def test_zero_adam_epsilon_is_valid_with_positive_epsilon_root(tmp_path):
    path = write_config(tmp_path, {"adam": {"epsilon": 0.0, "epsilon_root": 1e-8}})
    kwargs = config.training_kwargs(parse(argv=["--config", path]))
    assert kwargs["adam_eps"] == 0.0
    assert kwargs["adam_eps_root"] == 1e-8


@pytest.mark.parametrize("flags", [
    ["--steps", "-1"], ["--learning_rate", "0"], ["--model_seeds", "1,1"],
    ["--seeds", "0"], ["--steps", "0", "--lbfgs", "0"],
    ["--w_energy", "0", "--w_force", "0"],
])
def test_invalid_cli_training_overrides_are_rejected(flags):
    with pytest.raises(SystemExit):
        parse(argv=flags)
