"""Water cluster configuration, sampling, and size-independent evaluation."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("cluster_experiment_config", SCRIPTS / "experiment_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

DATA = {
    "train_sizes": [3, 4, 5], "test_sizes": [6, 2],
    "n_train": 12, "n_test": 8, "train_seed": 0, "test_seed": 100,
    "mc_steps": 300, "temperature_kelvin": 300.0,
    "radius_angstrom": None, "min_oo_angstrom": 2.7,
    "min_atom_distance_angstrom": 1.7, "sigma": 1e-9,
    "monomer_references": 4,
}


def write_config(tmp_path, data=None, experiment=None):
    path = tmp_path / "water.yaml"
    content = {"experiment": experiment or {"type": "water", "part": "clusters", "rungs": ["M_inf"]}}
    if data is not None:
        content["data"] = data
    path.write_text(yaml.safe_dump(content))
    return str(path)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--part", choices=["dimer", "clusters", "far", "ion"], default="dimer")
    result.add_argument("--rungs", default="M_inf")
    for key, value in DATA.items():
        if isinstance(value, list):
            result.add_argument("--" + key, default=",".join(map(str, value)))
        else:
            result.add_argument("--" + key, type=float if value is None else type(value), default=value)
    return result


@pytest.fixture
def water_script(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("water_clusters_script", SCRIPTS / "e3_water.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_data_config_and_selection_compatibility(tmp_path):
    path = write_config(tmp_path, DATA)
    assert config.load_config(path)["data"] == DATA
    assert config.load_selection(path) == {"type": "water", "part": "clusters", "rungs": ["M_inf"]}
    args = config.parse_selection(parser(), "water", ["--config", path], rungs=["M_inf"])
    assert args.part == "clusters"
    for key, expected in DATA.items():
        if isinstance(expected, list):
            expected = ",".join(map(str, expected))
        assert getattr(args, key) == expected


def test_data_cli_override_and_partial_defaults(tmp_path):
    path = write_config(tmp_path, {"train_sizes": [4, 5], "n_train": 20, "mc_steps": 500})
    args = config.parse_selection(parser(), "water", [
        "--config", path, "--train_sizes", "2,3", "--n_train", "2",
    ], rungs=["M_inf"])
    assert args.train_sizes == "2,3"
    assert args.n_train == 2
    assert args.mc_steps == 500
    assert args.test_sizes == "6,2"
    assert args.n_test == 8
    assert args.radius_angstrom is None


@pytest.mark.parametrize("key,value", [
    ("train_sizes", []), ("train_sizes", [1]), ("train_sizes", [3, 3]),
    ("test_sizes", "2,6"), ("test_sizes", [True, 6]), ("test_sizes", [2.5]),
    ("n_train", 0), ("n_test", -1), ("n_train", True),
    ("train_seed", -1), ("test_seed", 1.5),
    ("mc_steps", -1), ("monomer_references", -1),
    ("temperature_kelvin", 0), ("temperature_kelvin", float("inf")),
    ("radius_angstrom", -1), ("min_oo_angstrom", 0),
    ("min_atom_distance_angstrom", float("nan")), ("sigma", -1e-9),
    ("unknown_setting", 1),
])
def test_invalid_cluster_data_rejected(tmp_path, key, value):
    with pytest.raises(ValueError):
        config.load_config(write_config(tmp_path, {key: value}))


@pytest.mark.parametrize("data", [None, [], "bad"])
def test_data_requires_mapping(tmp_path, data):
    path = tmp_path / "water.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"type": "water", "part": "clusters"}, "data": data}))
    with pytest.raises(ValueError):
        config.load_config(path)


def test_data_cannot_be_silently_ignored_for_other_part(tmp_path):
    path = write_config(tmp_path, {"n_train": 2})
    with pytest.raises(SystemExit):
        config.parse_selection(parser(), "water", ["--config", path, "--part", "dimer"], rungs=["M_inf"])


def test_cli_can_select_clusters_for_data_config(tmp_path):
    path = write_config(tmp_path, {"n_train": 2}, {"type": "water"})
    args = config.parse_selection(parser(), "water", ["--config", path, "--part", "clusters"], rungs=["M_inf"])
    assert args.part == "clusters"
    assert args.n_train == 2


def test_water_entry_point_registers_data_cli(water_script, tmp_path, monkeypatch):
    path = write_config(tmp_path, {**DATA, "train_sizes": [2, 4], "test_sizes": [3], "n_train": 3})
    monkeypatch.setattr(water_script.sys, "argv", [
        "e3_water.py", "--config", path, "--n_train", "2", "--test_seed", "8",
    ])
    captured = []
    monkeypatch.setattr(water_script, "run_clusters", lambda args: captured.append(args) or {})
    monkeypatch.setattr(water_script, "_save", lambda *args: None)
    water_script.main()
    args = captured[0]
    assert args.train_sizes == "2,4"
    assert args.test_sizes == "3"
    assert args.n_train == 2
    assert args.test_seed == 8
    assert args.monomer_references == 4


def _mock_cluster_runtime(module, monkeypatch):
    """Keep real size dispatch/pair loops while replacing expensive truth and fitting."""
    calls, fits, pair_batches = [], [], []

    def structure(n, k=0, energy=None):
        positions_ang = np.array([[5.0 * m, atom * 0.5, float(k)] for m in range(n) for atom in range(3)])
        return SimpleNamespace(
            positions=module.ang_to_bohr(positions_ang), n_atoms=3 * n,
            energy=float(n * n if energy is None else energy), info={"n": n}, pinned={},
        )

    def dataset(sizes, n_per_size, **kwargs):
        calls.append((tuple(sizes), n_per_size, kwargs))
        return [structure(n, k) for n in sizes for k in range(n_per_size)]

    def fit(rung, structs, **kwargs):
        fits.append(list(structs))
        return {}, {"time": 0.0}

    def pair_structure(positions, **kwargs):
        result = structure(2)
        result.positions = module.ang_to_bohr(positions)
        result.info["pair"] = True
        return result

    def predict(params, rung, structs):
        if structs and all(s.info.get("pair") for s in structs):
            pair_batches.append(list(structs))
        return np.array([s.energy for s in structs]), None

    monkeypatch.setattr(module.W, "cluster_dataset", dataset)
    monkeypatch.setattr(module.W, "monomer_structure", lambda truth=None: structure(1, energy=0))
    monkeypatch.setattr(module.W, "water_structure", pair_structure)
    monkeypatch.setattr(module.W, "pair_additive_energy", lambda positions, n, truth=None: 4.0 * n * (n - 1) / 2)
    monkeypatch.setattr(module.T, "fit", fit)
    monkeypatch.setattr(module.T, "predict", predict)
    monkeypatch.setattr(module, "configure", lambda rung, args: SimpleNamespace(pin_mu=False, pin_alpha=False))
    monkeypatch.setattr(module, "_save", lambda *args: None)
    return calls, fits, pair_batches


def runtime_args(tmp_path, **changes):
    values = {**DATA, "train_sizes": "3,4", "test_sizes": "4,2", "n_train": 2, "n_test": 2,
              "train_seed": 7, "test_seed": 40, "mc_steps": 5, "monomer_references": 1,
              "temperature_kelvin": 325.0, "radius_angstrom": 6.0, "min_oo_angstrom": 2.9,
              "min_atom_distance_angstrom": 1.8, "cache": "", "rungs": "M_inf", "seeds": 1,
              "steps": 1, "lbfgs": 0, "batch": 2, "lbfgs_chunk": 2, "out": str(tmp_path / "out.json")}
    values.update(changes)
    return SimpleNamespace(**values)


def test_arbitrary_cluster_sizes_reach_sampling_and_evaluation(water_script, tmp_path, monkeypatch):
    calls, fits, pair_batches = _mock_cluster_runtime(water_script, monkeypatch)
    args = runtime_args(tmp_path)
    result = water_script.run_clusters(args)
    assert [(sizes, count, kw["seed"]) for sizes, count, kw in calls] == [
        ((3, 4), 2, 7), ((4,), 2, 40), ((2,), 2, 140),
    ]
    for _, _, kwargs in calls:
        assert kwargs["mc_steps"] == 5
        assert kwargs["T_K"] == 325.0
        assert kwargs["radius_ang"] == 6.0
        assert kwargs["min_OO"] == 2.9
        assert kwargs["min_any"] == 1.8
    assert calls[0][2]["sigma"] == DATA["sigma"]
    assert all(call[2].get("sigma", 0) == 0 for call in calls[1:])
    assert [s.n_atoms for s in fits[0]] == [9, 9, 12, 12, 3]
    assert result["dataset"]["train_count"] == 4
    assert result["dataset"]["test_counts"] == {"4": 2, "2": 2}
    metrics = result["M_inf"]["clusters_by_size"]
    assert set(metrics) == {"4", "2"}
    for values in metrics.values():
        assert values["energy_rmse"] == [0.0]
        assert values["nonadditive_rmse"] == [0.0]
    assert result["reference_nonadditive"]["4"] == [-8.0, -8.0]
    assert sorted(len(batch) for batch in pair_batches) == [1, 1, 6, 6]
    expected_pairs = {(float(5 * i), float(5 * j)) for i in range(4) for j in range(i + 1, 4)}
    for batch in pair_batches:
        if len(batch) == 6:
            assert {tuple(np.round(water_script.bohr_to_ang(s.positions)[[0, 3], 0], 12)) for s in batch} == expected_pairs
    assert "hexamer" not in result["M_inf"]
    assert "nadd6_truth" not in result


def test_default_cluster_sizes_keep_legacy_result_keys(water_script, tmp_path, monkeypatch):
    calls, _, _ = _mock_cluster_runtime(water_script, monkeypatch)
    result = water_script.run_clusters(runtime_args(
        tmp_path, train_sizes="3,4,5", test_sizes="6,2", train_seed=0, test_seed=100,
    ))
    assert [kwargs["seed"] for _, _, kwargs in calls] == [0, 100, 200]
    assert result["nadd6_truth"] == result["reference_nonadditive"]["6"]
    rung = result["M_inf"]
    assert rung["hexamer"] == rung["clusters_by_size"]["6"]["energy_rmse"]
    assert rung["dimer"] == rung["clusters_by_size"]["2"]["energy_rmse"]
    assert rung["nonadditive"] == rung["clusters_by_size"]["6"]["nonadditive_rmse"]


def test_matching_cache_reused_and_changed_settings_rejected(water_script, tmp_path, monkeypatch):
    calls, fits, _ = _mock_cluster_runtime(water_script, monkeypatch)
    args = runtime_args(tmp_path, cache=str(tmp_path / "clusters.pkl"))
    first = water_script.run_clusters(args)
    second = water_script.run_clusters(args)
    assert len(calls) == 3  # The second run reads the datasets rather than regenerating them.
    assert len(fits) == 2
    assert first == second
    args.n_train += 1
    with pytest.raises(ValueError, match="cache"):
        water_script.run_clusters(args)
    assert len(calls) == 3
    assert len(fits) == 2


def test_legacy_cache_rejected_before_training(water_script, tmp_path, monkeypatch):
    calls, fits, _ = _mock_cluster_runtime(water_script, monkeypatch)
    cache = tmp_path / "legacy.pkl"
    with cache.open("wb") as stream:
        water_script.pickle.dump(([], [], []), stream)
    with pytest.raises(ValueError, match="new --cache"):
        water_script.run_clusters(runtime_args(tmp_path, cache=str(cache)))
    assert not calls
    assert not fits


def test_generator_produces_reproducible_clusters_with_requested_geometry(water_script):
    kwargs = dict(seed=31, sigma=0.0, mc_steps=0, T_K=325.0,
                  radius_ang=5.0, min_OO=2.9, min_any=1.8)
    first = water_script.W.cluster_dataset([2, 3], 1, **kwargs)
    second = water_script.W.cluster_dataset([2, 3], 1, **kwargs)
    assert [st.n_atoms for st in first] == [6, 9]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.positions, right.positions)
        assert left.energy == right.energy
        assert np.isfinite(left.energy)
        assert left.forces.shape == left.positions.shape
        assert np.all(np.isfinite(left.forces))
        pos = water_script.bohr_to_ang(left.positions)
        oxygen = pos[::3]
        assert np.all(np.linalg.norm(oxygen, axis=1) <= 5.0)
        distances = np.linalg.norm(oxygen[:, None] - oxygen[None, :], axis=-1)
        assert np.all(distances[~np.eye(len(oxygen), dtype=bool)] >= 2.9)
        molecules = np.arange(left.n_atoms) // 3
        distances = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        assert np.all(distances[molecules[:, None] != molecules[None, :]] >= 1.8)


@pytest.mark.parametrize("data", [
    {"radius_angstrom": 1.0},
    {"radius_angstrom": 1.35},
    {"radius_angstrom": 2.0, "min_oo_angstrom": 4.0},
    {"radius_angstrom": 2.0, "min_oo_angstrom": 4.1},
])
def test_impossible_radius_rejected_with_resolved_minimum(tmp_path, data):
    with pytest.raises(ValueError, match="radius_angstrom.*min_oo_angstrom"):
        config.load_config(write_config(tmp_path, data))


def test_radius_uses_explicit_minimum_instead_of_default(tmp_path):
    data = {"radius_angstrom": 1.2, "min_oo_angstrom": 2.0}
    assert config.load_config(write_config(tmp_path, data))["data"] == data


@pytest.mark.parametrize("radius,minimum", [(1.0, 2.7), (1.35, 2.7), (2.0, 4.0)])
def test_random_cluster_impossible_radius_fails_before_sampling(water_script, radius, minimum):
    # This object cannot sample; the configuration guard must run before using it.
    with pytest.raises(ValueError, match="radius.*minimum oxygen separation"):
        water_script.W.random_cluster(2, object(), radius_ang=radius, min_OO=minimum)


@pytest.mark.parametrize("old", [
    {"dataset": {"n_train": 3}, "M_inf": {"hexamer": [0.1]}},
    {"M_inf": {"hexamer": [0.1]}},
])
def test_save_rejects_different_or_unknown_dataset_and_preserves_file(water_script, tmp_path, old):
    out = tmp_path / "existing.json"
    original = water_script.json.dumps(old, indent=2).encode()
    out.write_bytes(original)
    with pytest.raises(ValueError, match="new --out"):
        water_script._save(SimpleNamespace(out=str(out)), {"dataset": {"n_train": 2}, "M_1p": {}})
    assert out.read_bytes() == original


def test_save_merges_results_for_identical_dataset(water_script, tmp_path):
    out = tmp_path / "existing.json"
    dataset = {"train_sizes": [3, 4], "n_train": 2, "train_seed": 0}
    old = {"dataset": dataset, "M_inf": {"hexamer": [0.1]}}
    out.write_text(water_script.json.dumps(old))
    water_script._save(SimpleNamespace(out=str(out)), {"dataset": dataset, "M_1p": {"hexamer": [0.2]}})
    assert water_script.json.loads(out.read_text()) == {**old, "M_1p": {"hexamer": [0.2]}}
